<!-- The full project guide as of 2026-11-20 (AGENTS.md before it was cut down to the current research focus). Every architecture, dataset, tool and measurement is described here; AGENTS.md keeps a summary and points here. -->

# Project Guide

Spherical language models on Intel XPU. Nothing needs to be exported — no
`KERAS_BACKEND`, no `LD_LIBRARY_PATH`. XPU is the default device.

Run everything as `uv run model ...` (it uses the project's `.venv`, built by
`uv sync` from `uv.lock`). Don't run `python model ...` from another
activated venv: it won't have the XPU torch wheel or `keras_hub`.

## Create a run

```bash
model create --run-name baseline
```

Writes `runs/baseline/config.json` with the architecture's defaults and the
dataset's vocabulary size. Use `--arch` / `--dataset` to pick something other
than the defaults (`dual-sphere` / `tinystories-10k`).

## Edit config

```bash
model config --run-name baseline                                   # print it
model config --run-name baseline set training.learning_rate 0.0005
model config --run-name baseline set data.batch_size 256
model config --run-name baseline set arch_params.hidden_dim 1024
```

Dotted key, JSON-parsed value (`true`, `0.001`, `null` all work). Unknown keys
are rejected — there's no other way to change a hyperparameter.

`training.jit_compile` (default `false`) wraps the train/test step in
`torch.compile` for fused kernels. Only enable it on CUDA
(`model config --run-name my-cuda-run set training.jit_compile true`) — the
compiler backend doesn't work on this project's Intel Iris Xe XPU.

## Train / resume

```bash
model train --run-name baseline
```

Same command both times. If `runs/baseline/model.keras` + `metrics.csv`
already exist, it resumes automatically (weights + optimizer state) from the
last completed epoch. Override the epoch count for one run without touching
the config: `--epochs 5`.

`training.checkpoint_every N` (default 1): a numbered checkpoint
(`checkpoints/checkpoint_NNNNN.keras`) every N epochs, 0 for none; `model.keras` (latest, the resume point) and
`best.keras` are always written. Checkpoints hold the optimizer state too (gated-deltanet: 645 MB each): with short
epochs, per-epoch checkpoints filled the pod's 19 GB `/workspace`. Runs created before the key existed need it added
to their `config.json` (`model config set` only edits existing keys).

## Evaluate

```bash
model evaluate --run-name baseline
model evaluate --run-name baseline --checkpoint best
model evaluate --run-name baseline --checkpoint 12   # a specific epoch
```

Scores the full validation set: loss, perplexity, bpc, top-1 accuracy,
parameter count. Writes `runs/baseline/eval_<checkpoint>.json`.

## Generate

```bash
model generate --run-name baseline --prompt "Once upon a time"
```

Greedy by default (`--temperature 0.0`). For sampling:

```bash
model generate --run-name baseline --prompt "Once upon a time" \
  --temperature 0.8 --top-k 50 --top-p 0.9 --max-new-tokens 128
```

## Longer context: extrapolate or fine-tune

RoPE archs (`rope-logit-sum`, `rope-log-gaussian-sum`, `rope-kernel-attention`)
can run at a window they were not trained on; other archs refuse.

```bash
model generate --run-name X --prompt "Once upon a time" --context-length 256
model evaluate --run-name X --context-length 256          # writes eval_latest_ctx256.json
model create   --run-name X-ft256 --init-from X --context-length 256
model train    --run-name X-ft256 --epochs 3
```

Same weights; per-distance weights (`distance_weights`) are zero beyond the
trained window, and `basis_window` pins the kernel basis to the training
window. `--init-from` copies X's config, sets `init_from`, and starts from
X's weights with a fresh optimizer (`--init-checkpoint best` to pick another
checkpoint).

To start a fresh run from another run's embedding tables only (any architecture config, same shapes):

```bash
model config --run-name X set init_weights_from '{"run": "Y", "checkpoint": "latest", "weights": ["token_embedding", "readout"]}'
```

Copied when the run is first built (not on resume); every other weight keeps the architecture's own init.
`weights` entries are matched against the weight names (`l0_in_proj`, `final_norm`, `token_embedding`, ...) and take
shell wildcards (`"l0_*"`). Growing a `min-gru` run: a new run with more `num_layers`, `arch_params.silent_from` = the
old layer count (those layers start with `write_scale` 0: they write nothing, so step 0 gives exactly the trained
model's outputs) and `init_weights_from` the old run with `["token_embedding", "readout", "final_norm", "l0_*", ...]`
(e.g. `mingru-8L-v2` from `mingru-4L-v2`; fresh optimizer).

## Infinite context (`wave-arrow`)

`wave-arrow` is a pure recurrence with no window. It trains on rows of
`window_size` (default 1024) tokens packed from consecutive documents, a
target at every position, the state restarting at each `<bos>`
(`PackedDocumentDataset`, batch 64 rows by default). `generate` carries the
state token by token, so the context is the whole text. `--context-length N`
on `generate` / `evaluate` reads only the last N tokens (finite context).

`arch_params.num_layers` (default 1) stacks layers: layer n + 1 writes from space 1 of layer n, every layer's MLP
output (`dim`, 128) adds to the same vector before the readout. Only the 128-wide outputs leave a layer, and each
MLP recomputes its 2048-wide hidden layer in the backward pass, so memory stays near the single-layer model:
about 0.6 GiB per layer at batch 64 x 1024 (scales with the batch). `arch_params.mlp_gelu` is `tanh` (default) or `exact`: the tanh GELU is
within ~1e-4 of erf and 3x cheaper on the XPU. `arch_params.mlp_tf32` (default true) runs the MLP
matmuls in TF32 on CUDA only (the scan and readout stay fp32); set false for full fp32.

## RoPE state + linear-attention memory (`rope-state-memory`)

Infinite context like `wave-arrow` (packed rows, state restarts at `<bos>`, stateful `generate`). Per layer: a state
`s_t = lambda R_1 s_{t-1} + sqrt(1 - lambda^2) u_t` over the normalized input `u = RMSNorm(x)` (RoPE rotation, one
learned decay, init half-life 2); value `W_v u_t` (linear, no RoPE). Memory (`arch_params.memory`): `"banks"` (default) has `num_banks` (128) banks, a learned decay each
(half-lives log-spaced 1..1024, a power-law mixture); the addresses are the bank logits directly: write
`softmax(W_k s_{t-1})`, read `softmax(W_q s_t)` (W_q, W_k: dim -> num_banks, init 0: uniform), the read normalized
by the written mass, so it is a weighted average of past values. A write gate attenuates each token's write scores:
`softmax(W_k s_{t-1}) * sigmoid(w_w . s_{t-1} + b_w)`, its own projection of the state before the update
(`write_gate_bias`, default -6: starts at 0.0025 like the read gate, `w_w` 0; only the ratios between tokens matter,
since the read divides by the written mass; `null` = no gate, which is how runs saved before it load;
`write_gate_input "logits"` is the first version, gated from the write logits, which saturated open in `rsm-1L-wgate`).
`"linear"` is decayed linear attention with L2-normalized addresses. Each mix has two independent sigmoid gates
(an LSTM-like keep / add pair, each with its own gradient): the read `h = a s + g o` (a keeps the state, g takes the
memory read, both from the state after the update `s_t`, as the write gate reads `s_{t-1}`: a bounded input, unlike
the read logits (`read_gate_input "logits"`, how earlier runs load); a -> 0, g -> 1 replaces the state) and the output `m = MLP(RMSNorm(h))` (4x),
`y = a2 x + g2 m`, x the layer input (a2 keeps the input, g2 takes the MLP output, both from `RMSNorm(m)`). Keep gates
start at 0.9975 (`keep_gate_bias` 6), the memory read, MLP and write gates at 0.0025 (biases -6), all projections 0,
so a fresh model is about the bigram `W_out RMSNorm(embedding)` and learns how much memory and MLP to add or to put in
place. Other options: `read_mix "add"` (`h = s + g o`), `"tanh"` (one coupled weight: in `rsm-1L-mix` its gate never
opened, val 2.744 vs 2.419); `mlp_residual "input"` (`y = x + m`), `"state"`, `"mix"` (coupled). Runs saved before
these options load as `"add"` / their saved `mlp_residual`. `tie_embeddings` (default false): true makes the readout the token embedding table transposed,
`logits = RMSNorm(y) E^T`. Pre-norm (`pre_norm`, default true): RMSNorm on each layer input and before the
readout. Embedding std 0.02. `num_layers` stacks layers. Runs saved before these options load as `"add"` / their
saved `mlp_residual`.
`uv run python ad_hoc_scripts/rsm_gate_retention.py --run-name X` reports the learned decays, bank use and the
read gate; `rsm_name_recall.py` tests recall of rare multi-token names.
`model scripts rsm-bank-stepper --run-name X [--text "..."] [--name " Roxy"]` writes an interactive page
(`reports/rsm_bank_stepper.html`, open at a step with `#doc=0&t=79&bank=71`): the read as exact attention over
past tokens (a weighted average of values, so token j's weight is `sum_b r_tb (1-mu_b) mu_b^(t-j) w_jb / sum_b r_tb z_b`),
each bank's write / read at the token, a banks x tokens heatmap, what the selected bank stores, and the state's token
shares.

On the XPU it trains through custom SYCL kernels (`model/kernels/rsm_scan.sycl`, built by torch into
`.kernel_build/` on first use with oneAPI 2025.x `icpx`): sequential state / memory scans (and their backward
scans), fused RMSNorm / L2-normalize / bias+GELU passes around oneDNN matmuls. Elsewhere, or with dims the kernels
do not cover (dim and key_dim in 16..128, multiples of 64 for the fused norms), it falls back to the chunked torch
path, which is what the kernels are tested against (`tests/test_arch_rope_state_memory.py`).
`uv run python ad_hoc_scripts/bench_rope_state_memory.py --layers 1 2 4 [--profile]` times a training step;
`bash ad_hoc_scripts/disasm_rsm_kernels.sh` AOT-compiles the kernels for `tgllp` and disassembles each with
`iga64 -p=12p1` (`$IGA64`, else PATH, else `~/projects/tiny-lm/.toolchain/igc-tools/usr/bin/iga64`), plus SIMD width,
GRF and spill sizes from the zebin.

Wide models and long rows (run `rsm-8L-768-16k`: dim 768, 8 layers, 32 banks, 16384-token rows, 44.5M params). The
kernels take any even dim: the memory keeps its rows (banks / key_dim: 16, 32, 64, 128) on the register-tiled axis,
and for values wider than 144 its dq / dk run as column read-outs (`outer_scan_cols`); row norms past 256 use a
two-sweep kernel. At one or two rows a scan is latency-bound (a few hundred threads walking 16k dependent steps), so
`rsm_scan._segmented` cuts the row into time segments scanned as separate rows from zero, then adds each segment's
carried-in state with one batched matmul (exact; `RSM_SEGMENT_ROWS`, default 8 rows). The last layer's MLP and output
mix run only at positions with a target (`dense_hidden_valid`: that output feeds only the readout). The bank read
is joined to the state by one fused kernel (`join_read`). `arch_params.mlp_recompute false` keeps the MLP
pre-activations (rows x mlp_hidden floats per layer) instead of recomputing them in the backward pass.
Throughput at 1 x 16384: 3.75 s per row, ~4,400 tokens/s (~2,100 targets/s on agent-mix-nemotron), peak 4.8 GiB;
matmuls are ~68% of it (fp32 at 1.74 TFLOPS). `uv run python ad_hoc_scripts/bench_rsm_run.py --run-name X
[--batch N] [--window T] [--profile]` times full steps of a run's config on real batches (writes nothing).
Iris Xe torch quirks: the embedding backward fails for 1025..2048 ids and the backward of boolean-mask indexing at
some shapes (work-groups over 512): the arch gathers with `index_select` instead.

Large batches of long rows: `data.batch_size` rows per optimizer update, `data.micro_batch_size` rows per forward
pass; the dense train step sums the gradients of the micro-batches' summed losses and divides by all their targets
(exactly the 128-row mean gradient; a micro-batch without targets counts but adds nothing). `steps_per_epoch` and
`warmup_steps` count updates. `training.dense_validation true` (packed archs): validation on packed rows scoring every
target, the same rows every epoch (`PackedDocumentDataset(fixed=True)`), instead of a whole window per sampled target.

Truncated BPTT (`training.stateful true`, with `data.stream_chunks`, default 16): row b of each batch continues row b of
the previous one (`PackedDocumentDataset(stream_chunks=...)`, a pure function of the global batch number; every
`stream_chunks` batches the rows jump to new random document starts), and the model carries each layer's state and
memory between updates, detached: the forward pass sees the whole document, the gradient reaches back one row
(`window_size`). The kernels scan from zeros and the carry's decayed contribution is added in torch (`_CarryState`
for the state, `q . mu^(t+1) M_0` for the memory), so the decay gradients stay exact (`test_carried_state_training_gradients`).
No micro-batches with it. `rsm-8L-768-16k` now: window 4096, 8 streams (32k tokens per update), 200 updates per epoch,
`mlp_residual "state"` (`y = h + MLP(RMSNorm(h))`, h = state + memory read), ~10 s per update, peak ~10.7 GiB.
Validation rows still start at a document start and are `window_size` long (no carry).

fp16 matmuls (`arch_params.mlp_fp16 true`): the fused MLP's six matmuls run in an assembly kernel
(`model/kernels/hgemm/gen_hgemm.py`, ported from tiny-lm's `kernels/asm/matmul_f16.py`). oneDNN's fp16 gemm on this GPU
converts to f32 and does f32 mads (its ISA: `mov :f <- :hf`, `mad (16) :f`), so it is no faster than fp32 (1.7
TFLOPS); a SIMD16 `mad :hf` is 2x a `mad :f` (measured 3.7 vs 1.9 TFLOPS). The kernel: one hardware thread per
16 x 32 tile, A broadcast through a register region, accumulators updated in place on the opposite register parity
from B (tiny-lm: ~5%), a scalar `jmpi` loop (compiled SYCL keeps a fused-EU NoMask workaround, ~10 instructions
per loop); products summed in fp16 over 16 k steps, then into f32 accumulators (fp16-only sums were 10-25x less
accurate). Gradient operands are scaled by a power of two chosen on the device from their max (they sit below fp16's
normal range); the kernel divides it out. ~6e-4 relative error; 2.0-3.2 TFLOPS at the MLP's shapes vs ~1.7 for
torch's fp32 `mm`; step 8.6 s vs 10.1 s (~3,750 tokens/s) on `rsm-8L-768-16k`. Build on first use (`hgemm.build`,
cached in `.kernel_build/hgemm/`): iga64 assembles the generator's output, ocloc compiles the OpenCL stub
(`hgemm_stub.cl`: arguments, binding table; payload offset X is register r(2 + X/32)), the stub's `.text` is swapped
for the assembly, and the zebin is loaded through Level Zero into torch's SYCL queue (`hgemm.sycl`), checking that the
driver runs our code. Needs `iga64` ($IGA64, PATH or ~/projects/tiny-lm/.toolchain) and the driver's `ocloc`.
Operands are packed to fp16 panels by one pass each (`pack`, `pack_rows`: bandwidth-bound, ~1.6 s of the step).
`ad_hoc_scripts/hgemm/`: `build.sh` + `hgemm.cpp` (the SYCL prototype, ISA stats), `zeload.cpp` (loading check),
`hgemm_bench.cpp` (the assembly kernel standalone). `RSM_HGEMM=0` disables it (the arch then raises if `mlp_fp16`).

## minGRU stack (`min-gru`)

```bash
model create --run-name X --arch min-gru --dataset agent-mix-bpe1024
model train --run-name X
uv run python ad_hoc_scripts/mingru_inspect.py --run-name X [--text "..."]   # what the states hold
```

minGRU layers (Feng et al. 2024) writing into a residual stream through tanh, plain PyTorch (`model/archs/min_gru.py`,
no custom kernels: runs on CUDA as is). Per layer: `layer_input = rms_norm(stream)`, `gate_logit, raw = layer_input @
in_proj`, `update_gate = sigmoid(gate_logit)`, `state = (1 - update_gate) * state_before + update_gate * tanh(raw)` (0
keep at `<bos>`: the state restarts), `stream += write_scale * tanh(rms_norm(state) @ out_proj)`. No MLP. The state
stays in -1..1; each layer adds at most `write_scale` (per channel, init 0.1).
Channels start with half-lives log-spaced from `half_life_min` to `half_life_max` (1..1024 tokens, from the gate bias).
The scan (`linear_scan`): chunks of ~sqrt(T) (a divisor of T, so no padding: padding copied keep / write and cost
+30-37% at T = 1000 / 4000; T with no divisor near sqrt(T) runs as a prefix + the rest), a loop inside all chunks at
once, a loop over chunk ends, one fix-up pass; linear space (keep factors in 0..1, nothing to overflow), exact to float rounding; its backward is the same scan
in reverse and stores only keep factors and states. Compute and memory are linear in the row length (measured 60-65k
tokens/s at rows of 512 to 16384 with the same tokens per step and the same memory; `recompute true`: 2.6 -> 1.0 GiB
at 4 x 16384, 1.3x the time). Defaults (dim 128, hidden 256, 4 layers, 0.66M params): ~85k tokens/s at 16 x 1024 on
the Iris Xe (191-194 ms per step), fp32, ~44 GFLOP per step (~0.23 TFLOPS: bandwidth-bound elementwise passes, the
matmuls are ~18% of the time). RMSNorm has a hand-written backward (`RMSNorm`, ~30% faster than autograd's).
`uv run python ad_hoc_scripts/profile_min_gru.py [--step-only] [--set key=value ...]` times the real train step on
agent-mix batches and each part alone. (The paper's log-space scan with `torch.logcumsumexp` was 15x slower here: the
Iris Xe's `logcumsumexp` is ~100x a `cumsum`; removed.)
Agent data: the loss covers only `loss_spans` (training and validation: prompts, tool lists and results are read,
never scored); the last layer's write and the readout run only at positions with a target (`dense_hidden_valid`).
Defaults for new runs: rows of 32 x 512 with `data.random_windows true` (below), `corrupt_prob` 0.15. On
`agent-mix-v2-bpe1024`: 3,030 steps per epoch (one pass, 49.7M tokens), ~195-207 ms per step, ~10 min per pass.
Supports `training.stateful` (carried states) and stateful generation. Tests: `tests/test_arch_min_gru.py`.
`arch_params.gate_heads G`: one sigmoid update gate per head of hidden_size / G channels instead of one per channel
(the scan takes the per-head keep factors and broadcasts them). Faster (4 layers, 32 x 512: 188 -> 152 / 156 / 175 ms
per step at G = 4 / 16 / 64) but worse per step and per wall clock early on: after one quarter-pass epoch from scratch
val_loss 1.278 per channel vs 1.385 (G = 4) and 1.349 (G = 16). Default: per channel.

`arch_params.mlp_mult k` (default null: the tanh projection): each layer writes through a GELU MLP instead,
`write_scale * (gelu(rms_norm(state) @ mlp_in + mlp_bias) @ mlp_out)`, hidden k x dim (tanh-approximated GELU,
`mlp_out` init std 0.02; weights `mlp_in`, `mlp_bias`, `mlp_out` replace `out_proj`). `mingru-16L-v2-mem-mlp-cuda`
(k = 4, 6.4M params, embeddings and readout from `mingru-16L-v2-mem-cuda`): ~350 ms per step at 96 x 512 on the A5000
(tanh: 283), 18.4 GB.

`arch_params.static_memory true`: the contents of the `<system>` / `<tools>` spans never enter the layers (25% of
agent-mix-v2's tokens). The rows hold the rest (the span tags stay), so a pass is 37.3M tokens: 2,274 steps of 32 x 512.
A memory encoder (one minGRU layer, `mem_enc_*`) runs over each document's span tags and contents; each content token
j stores, per layer, key `elu(rms_norm(encoder state before j) @ mem_k) + 1` (induction-style) and value
`rms_norm(embedding_j) @ mem_v`, summed into one constant-size matrix per document, memory and head:
`C = sum_j k_j [v_j, 1]`. Only the reading positions (`<assistant>` and the tokens inside it) query, with the layer's
state after the update: `read = S^T q / z^T q` (a weighted average of the stored values), gated by a sigmoid per memory
and head, through `mem_out` (init 0) into the stream next to the state's write. Every system / tool span precedes its
document's first `<assistant>` (checked), so a reading position reads its document's whole memory.
Batches are dicts (`model/memory_rows.py`): the data loader (`PackedDocumentDataset(static_memory=...)`, from the
model's `static_memory_spec`) cuts the rows from the corpus without span contents, so a row cut inside a document
still gets that document's whole memory. Raw token rows (evaluate, generate) are converted. Reads go in chunks of 32
positions of one document: `q @ C_doc`, one batched product, with no positions x span-tokens scores. Every index list is
distinct: on the Iris Xe autograd's scatter-add backward of a padded gather (thousands of atomics on one row) was the
bottleneck (`GatherRows`, `RepeatSorted`, `place_rows`). 32 x 512: 398 ms per step (185 without memory), ~15 min per
pass. Generation carries the open document's memories and encoder state. `inspect_memory` (web chat "Inspect state")
inspects the sequence the layers see (without the span contents) plus, per layer, each reading position's read weights
over the system / tool tokens (`q . k_j`, normalized per memory), its gates and the size of its addition to the stream;
checked: they rebuild every read exactly.
A document without one of the spans has an empty memory (z = 0): its read divides by 1, not by a 1e-6 clamp. The clamp
gave 1e7-sized gradient rows that, summed through `RepeatSorted`'s cumsum, wrecked every other memory gradient on the
`min-gru` path (the local `mingru-4L-v2-mem` trained with it; the CUDA fused path gathers with a plain backward and
was exact). `test_memory_gradients_match_plain_autograd` checks it.

`min-gru-cuda` (`model/archs/min_gru_cuda.py`): the same model and weights for CUDA GPUs; `min-gru` stays for local
training. The scan is one parallel associative scan (torch's `associative_scan` compiled to a Triton kernel: 0.14 ms
forward vs 1.7 ms for the chunked scan at 32 x 512 x 256 on an RTX A5000), its backward the same scan from the end
(`ParallelScan`); shapes are padded to buckets (one compile each). On the A5000 with the static memory: 47 ms per step
(`min-gru`: 54), ~2 min per pass; the GPU is busy ~25 ms of it (the rest is launch / Python overhead).
Tests: `tests/test_arch_min_gru_cuda.py` (CUDA only). `ad_hoc_scripts/bench_scan.py` compares the scans;
`profile_min_gru.py --device cuda --arch min-gru-cuda` times the real step.

## Gated DeltaNet + MoE (`gated-deltanet`)

`model/archs/gated_deltanet.py`, `model/archs/delta_rule.py`. 16 layers of a Gated DeltaNet token mixer (Yang, Kautz,
Hatamizadeh 2024, arXiv 2412.06464) and a mixture-of-experts FFN, pre-norm residual stream (dim 256):
`q, k, v = silu(conv4(rms_norm(x) Wqkv))` (causal depthwise conv, never across a `<bos>`), q / k L2-normalized,
per head (4 x 64) `S_t = alpha_t S_{t-1} + k_t (beta_t (v_t - alpha_t S_{t-1}^T k_t))^T`, `o_t = S_t^T q_t`, alpha =
exp(-exp(A_log) softplus(x Wa + dt_bias)) (Mamba2 init), beta = sigmoid(x Wb); output `rms_norm(o) * silu(x Wgate) Wo`.
MoE: 6 GELU experts (256 -> 1024 -> 256), top-1 routing, load balanced without an auxiliary loss: a per-expert bias
on the routing logits (the choice only) moves by `router_bias_rate` towards the mean load, queued and applied at the
next forward pass (so a recomputed layer routes as in its forward). 56M params, ~9M active per token. No static memory.

The delta rule three ways (tests check them against each other, values and gradients, resets and carried state): a
token loop (reference), chunks of 64 in torch (WY form: one unit-triangular solve per chunk; Iris Xe / CPU,
generation with a carried state), and on CUDA flash-linear-attention's fused Triton kernel `chunk_gated_delta_rule`
(rows flattened into one sequence cut at row and document starts, `cu_seqlens`). The pod needs
`pip install --no-deps flash-linear-attention fla-core einops` (`pod.sh _setup` does it; no deps: keeps the image's
torch / triton). CUDA training path (`GDN_FUSED=0` disables it): matmuls in bf16 on the tensor cores (TF32 for the
fp32 ones left: gates, router, the readout), elementwise chains fused by `torch.compile` (`_mixer_proj`, `_mixer_post`,
`_router`), and custom Triton kernels (`model/archs/gdn_kernels.py`): the document-masked causal conv + SiLU
(forward, d_pre + weight gradient with per-tile atomics, input gradient; `GDN_TRITON_CONV=0`: the compiled version)
and the MoE as one autograd function (`MoEExperts`): a grouped GEMM over the experts for the up-projection with
bias + GELU in its epilogue, its backward GEMM with the GELU derivative, the recomputed GELU and the bias gradient
in the epilogue (autotune resets the atomically accumulated bias gradient between its benchmark runs), cuBLAS per
expert for the rest, weight gradients written in place. A5000, 96 x 512, steps: unfused 628 ms (83% of the GPU time
elementwise, 9% tensor-core matmuls, 7.5 TFLOPS of the 111 bf16 peak) -> compiled 396 -> + memory work 389 -> + TF32
and one MoE function 342 -> + Triton conv 305 -> + grouped GELU GEMMs 274-279 ms: ~178k tokens/s, ~17 TFLOPS, the
GPU busy 91% of the step (a plain bf16 matmul reaches ~89 TFLOPS here; at dim 256 memory-bound passes still take
most of the time; batch 192: 192k tokens/s, 18.2 GiB). Memory: the backward pass kept 29 KB per token and layer (20
GiB at 96 x 512); now ~11 KB (9.5 GiB): `activation_memory_budget` 0.5 (a global AOTAutograd setting: the compiled
graphs recompute their elementwise parts), the experts save only input and pre-activation in bf16, routed rows and
the gate in bf16. `arch_params.recompute true`: 2.5 GiB at +21% time.
`ad_hoc_scripts/bench_gdn.py [--batch N] [--set dim=512 ...] [--profile]` times steps, TFLOPS against the peak and
the GPU time by kind (tensor-core matmuls, fp32 SIMT matmuls, fla kernels, elementwise).
`ad_hoc_scripts/gdn_moe_health.py --run-name X [--dataset D] [--docs N] [--per-layer]`: expert load with / without the
router bias, how many choices the bias overrides, gates, expert output similarity, routing ablations (in
`gdn-16L-256-moe6-*` the routers of layers 0-10 send 75-100% of tokens to one expert and the bias does the routing;
layers 0-3's MoE is worth ~0.05 nats). `ad_hoc_scripts/gdn_subword_trace.py --run-name X [--dataset D]`: how words are
assembled from BPE pieces (logit lens by depth inside a word, MoE / mixer / conv ablations, first-piece probes,
routing by piece position). Found: layer 0's heads have half-lives of 0.1-1.9 tokens (a window of the last pieces;
its mixer is worth ~2.6 nats), layers 1-3 are nearly idle (~0.11 nats together), layers 4-11 complete words gradually,
12-15 choose the word and the next one. `ad_hoc_scripts/gdn_logit_attribution.py --run-name X --text-file F (--pos N |
--at TOKEN) [--a A --b B]`: exact direct logit attribution of logit(A) - logit(B) at one position (the final rms and
each head's read rms fixed): the embedding, each layer's mixer and MoE, each head, and each past position through the
unrolled read (the votes sum to the logit difference, printed as a check). Tests:
`tests/test_arch_gated_deltanet.py` (CUDA parts skip elsewhere).

Large corpora without the static memory: above 1B tokens `PackedDocumentDataset` computes the loss mask per batch
from each row's documents (`_row_loss_mask`) instead of a per-token mask (int64 position arrays over the corpus).

## LRU + banks (`lru-banks`)

`model/archs/lru_banks.py`: the bank layer of the synthetic tests (`bank_memory.py`) as a language model. Factorized
tied embedding: ids -> E (32,768 x 128) -> W_in -> a 512 stream; logits = (rms_norm(h) W_out) E^T (`_TiedReadout`).
4 layers, each: an LRU (256 complex channels, state s = 512 reals, half-lives 1..16 at init; layers >= `gate_from` (1)
with the input gate g_t = sigmoid(x w + b): lambda_t = lambda^g_t, gamma_t = sqrt(1 - |lambda_t|^2), g = 0 skips the
token; bias 3 at init), then banks: 16 heads x 32 (keys, values), 128 slots per head, top-2 routing renormalized,
"mass" normalization, fixed slot decays (half-lives 1..1024, log-spaced), key from the LRU state before the token,
query after; hop 1: h += W_o read(q), h += FFN; hop 2: q2 = normalize(W_q2 rms(h)), h += W_o2 read(q2) from the same
banks, h += FFN (the same GELU FFN, 4 x 512, one per layer). States restart at `<bos>` (right after each `<eos>`).
20.3M params (4.2M embedding). Packed rows, dense supervision, stateful generation (`advance`, the torch path).

CUDA training path (`model/archs/bank_kernels.py`, `LRU_BANKS_FUSED=0` disables it; sizes in multiples of 16):
- `lru_scan`: a Triton kernel per (row, 32 channels): gate, gamma and lambda computed inside (no per-token lambda
  tensor), one read of the projection and one write of the states; the backward walks the row in reverse from the
  saved states (d nu, d theta per channel and d g per position by atomics).
- the written mass z (the read divides by it): an exact fp32 Triton scan.
- the bank read on fla 0.6's gated-slot-attention kernels (same two passes): scores over the slots with
  `chunk_gsa_fwd_k` (keys k, slot weights r, per-slot log decays), then softmax(tau ok / z) and w = p / z in one
  compiled pass, then the values with `chunk_gla_fwd`; backward `chunk_gla_bwd`, the normalization's backward
  (compiled), `chunk_gsa_bwd_k`. The decays are the same in every layer: their chunk-local cumulative sums once per
  forward (`BankContext`); rows flattened with `cu_seqlens` cut at row and document starts.
- everything between the kernels compiled with explicit bf16 matmuls (`_f_pre`, `_f_addr`, `_f_post`, `_f_q2`):
  norms, casts, the residual adds and the GELU in the matmul epilogues; routing as top-2 of the logits (a softmax
  over the two largest equals the renormalized top-2 of the full softmax: no softmax over 128 slots, no topk).
- on CUDA the torch path (generation) runs in fp32: its chunked forms multiply factors up to e^30 and their
  inverses, exact in fp32, not in bf16. `gated_deltanet.py` turns TF32 on globally when imported.
Speed on the A5000, batch 64 x 512 (`ad_hoc_scripts/bench_lru_banks.py [--batch N] [--profile]`): first version
752 ms per step, 43.6k tokens/s, 16.6 GiB (44% of the GPU time unfused elementwise: fp32 adds, bf16 casts, the
routing softmax + topk); compiled pieces + top-2 shortcut: 582 ms, 56.3k tokens/s, 11.3 GiB; keeping w for the
backward: 566 ms, 57.9k tokens/s, 12.1 GiB (batch 96: 58.2k, 18.1 GiB; 128 does not fit). Now ~48% of the GPU time is
fla's bank kernels (they scale with the 128 slots), ~17% tensor-core matmuls, ~17% compiled elementwise.
Possible next step: share the chunk states of the two hops (they depend on k, r, v only).
Tests: `tests/test_arch_lru_banks.py` (rows == generation with hops, gates and resets; on CUDA: the LRU kernel and the
bank read vs the float64 torch reference, values and gradients (fp32 operands tight, bf16 within 8%), the fused model
vs the torch path); `tests/test_bank_memory.py` for the torch reference.

Growing deeper (no restart): `arch_params.hops` takes one value or one per layer (`[2, 2, 2, 2, 1, 1]`: the hop-2
weights exist only in layers with 2), and `arch_params.silent_from N` starts layers >= N with `w_o`, `w_o2`, `w2`
(and `b2_bias`) at 0, so both reads and both FFN passes add exactly 0: a model grown from a trained one starts with
its outputs (checked on `lru-banks-4L-512`: 4 layers -> 6 with two one-read layers, 20.3M -> 27.3M params, every
position's loss identical; `TestGrowth`). The zeros get gradients at once (the reads and FFN hidden units are not 0);
the rest of a new layer follows when they move. Recipe: a new run with `num_layers` 6, `hops` [2, 2, 2, 2, 1, 1],
`silent_from` 4 and `init_weights_from` `{"run": "lru-banks-4L-512", "checkpoint": "best", "weights":
["token_embedding", "w_in", "w_out", "final_norm", "l0_*", "l1_*", "l2_*", "l3_*"]}` (fresh optimizer, short warmup).

Learning-rate schedule (any run): `training.lr_decay_start`, `training.lr_decay_steps` (0: none),
`training.lr_min_ratio` (0.1): warmup-stable-decay, linear from `lr_decay_start` to `lr_min_ratio` x the rate over
`lr_decay_steps` updates (`LinearWarmup`, saved in checkpoints; runs without the keys: unchanged).
`data.corrupt_prob` defaults to 0.15 and packed rows apply it (input tokens replaced at random, training only): set it
to 0 for pretraining on clean text.

Run `lru-banks-4L-512` (pod, from scratch): `pretrain-mix-bpe32768`, batch 64 x 512, epochs of 0.02 pass (1,199
steps, ~11.5 min at 566 ms per step), 100 epochs = 2 passes (3.93B tokens, ~19 h), lr 2e-3 (warmup 500, decay to 0.1x
over the last 24,000 of ~119,900 updates), weight decay 0.1, clipnorm 1.0, `corrupt_prob` 0, `checkpoint_every` 10.
Target: ~2.2 nats per token = 0.72 bits per character at 4.38 characters per token.

## RunPod (`pod.sh`)

```bash
./pod.sh up                          # create the pod (RTX A5000: community, else secure), id in .pod
./pod.sh push                        # code + agent-mix-v2-bpe1024 (+ its tokenizer) by rsync, venv on first push
./pod.sh sh 'nvidia-smi; ls runs'    # any command in /root/rnn-research with the venv active (no args: a shell)
./pod.sh train <run>                 # model train in the background, log runs/<run>/train.log; ./pod.sh log <run>
./pod.sh pull <run> [--all]          # runs/<run> back here: config, metrics, latest + best checkpoints (--all: every one);
                                     # its dataset's metadata + validation split too if missing here (web chat, evaluate)
./pod.sh status | stop | start | rm
```

Scheduled pulls: `ad_hoc_scripts/pull_checkpoints.sh <run>` pulls, checks the copies' zips (a pull that met the pod
mid-save gets a truncated file: it pulls again after 30 s) and logs one line to `logs/pull_<run>.log`. A user crontab
entry runs it every 5 min for `lru-banks-4L-512` (before: `gdn-16L-256-moe6-mix`) (with `PATH` including `~/.local/bin` for runpodctl, `flock`
against overlaps); remove it (`crontab -e`) when the pod is stopped.

Code, data and venv live on the container disk (`/root/rnn-research`: the `/workspace` network volume is slow for
memory-mapped reads), `runs/` links to `/workspace/runs` (kept when the pod stops; the container disk is not: `push`
again after a start). Runs on the pod set `device` to `"cuda"` and use `min-gru-cuda`.

`data.random_windows true` (packed archs, `PackedDocumentDataset`): one epoch is one pass over the corpus. Each epoch the
token stream is cut into consecutive windows of the model's window from a random offset, served in a shuffled order:
every token is trained once per epoch, rows start at document starts, middles and ends, always from a zero state (the
state still restarts at each `<bos>`). `steps_per_epoch` is derived (windows / batch_size); the configured value is
ignored. `data.epoch_passes` (default 1): the fraction of a pass per epoch (0.25: four epochs per pass, the pass
continuing across them). Not with `training.stateful`.

## Agent dataset (`agent-shell-bytes`)

Synthetic tool-use conversations, byte-level: 12 special tags (`<bos> <eos> <tools> </tools> <user> </user>
<assistant> </assistant> <call> </call> <result> </result>`, ids 0-11), every other token one UTF-8 byte (id 12 +
byte), vocabulary 268. No system prompt: each document lists its tools (a random subset of `run`, `read_file`,
`write_file`, `calculator`, `python`, plus tools it never needs), then user / assistant turns with calls such as
`run(command="grep -n mango fruits.txt")` and their results. Every result is executed: a simulated bash over a
virtual filesystem (checked identical to real bash: `--check-shell`), real Python for code, exact arithmetic. Task
families: files, inspection, lookup (grep, reading, follow-ups answered from what the context already shows,
lists pasted in the message), math, code (write + test, fix a bug, run a script), refusals when no listed tool fits.

```bash
.venv/bin/python datasets/build_agent_shell.py                 # 50k train / 1k validation docs (~26M tokens), ~25 s
.venv/bin/python datasets/build_agent_shell.py --samples 5     # print documents
.venv/bin/python datasets/build_agent_shell.py --check-shell   # simulator vs real bash
```

Loss: only what the assistant writes counts. `loss_spans` in the metadata (`[[<assistant>, </assistant>]]`, the
close tag included, ~29% of tokens) makes the training rows ignore every other target, and validation / `model
evaluate` draw their targets only from those spans (`model.data.loss_target_mask`, cached next to the token file);
any dataset can set `loss_spans`. Documents: median 480 tokens, p99 ~1250, max ~1950 (a 1024 window holds ~97% whole). `parse_call` in the builder
parses a model's call for an agent-loop evaluation.

BPE version (`agent-shell-bpe1024`): the same documents re-encoded with a byte-level BPE learned on the train split,
1024 ids in total: the 12 tags (0-11) and the 256 bytes (12 + byte) as in the byte dataset, then 756 merges (GPT-2
pre-tokenization, so merges stay inside words, spaces and punctuation; numbers split into 1-2 digit pieces). 2.65
bytes/token overall, 2.31 inside the assistant spans (`avg_chars_per_token`, so `bpc` compares with the byte runs);
documents median 178 tokens, max 843. Same `loss_spans`.

```bash
.venv/bin/python datasets/build_agent_shell_bpe.py [--vocab-size 1024]   # needs agent-shell-bytes, ~10 s
```

Mixed version (`agent-mix-bpe1024`): `agent-shell-bytes` (50k docs) + Atum09/agent-training-dataset (ChatML, rev
8b96a95, raw files in `datasets/raw/atum09-agent/`; 59k rows, 27k after removing exact duplicates; web search,
email, finance, code, memory..., 20% non-English) in one format, with its system prompts and thinking: 16 tags
(the 12 above plus `<system> </system> <think> </think>`, ids 12-15), bytes at 16 + byte, 752 merges learned on
the mix. ChatML system -> `<system>`, tool schemas -> `<tools>` lines `name(param: type, ...): description`, `<think>`
-> the think tags, `<tool_call>{json}</tool_call>` -> `<call>name(arg=<JSON value>)</call>`, tool responses ->
`<result>` (their `content`). The loss covers `<assistant>` spans, thinking included. 2.39 bytes/token, documents
median 275 tokens, p99 1018, max 2187 (rows start at a document start, so tokens past the window are never
trained: use a 1024 window). Atum09 is templated (1.1k distinct first user messages): validation conversations
are new, their prompts mostly are not.

```bash
.venv/bin/python datasets/build_agent_mix_bpe.py [--samples 3]   # ~50 s; the curl commands are in its docstring
```

Mixed version 2 (`agent-mix-v2-bpe1024`, `datasets/build_agent_mix_v2.py`): the two sources above plus
MuskumPillerum/General-Knowledge (Q/A, 1-3 per conversation), Sp1786/multiclass-sentiment-analysis-dataset (varied
"is this positive, negative or neutral" questions; half with a system prompt asking for one word, answered with the
word), 10k English two-person multi-turn chats from mookiezi/Discord-Dialogues (at least 2 messages each), and 40k
synthetic memory tasks (save a fact with a memory tool call, recall it after unrelated turns from the context or with a
recall call in a new conversation, several facts at once, exact copies of texts / codes / list items; the memory tools
rotate between 4 naming conventions). Every category mixes no / system prompt / tools / both (tools always when a call
is made), with system prompts written for that kind of task and unneeded tools added. BPE retrained on the mix: 1024
ids in total (16 tags + 256 bytes + 752 merges), 2.38 bytes/token. Train 49.7M tokens, 178,671 documents:

| category | docs | tokens | assistant tokens | share |
|---|---|---|---|---|
| agent-shell | 50,000 | 12.1M | 4.1M | 24.4% |
| atum09 | 27,105 | 13.3M | 7.3M | 26.9% |
| general-qa | 20,334 | 6.7M | 4.9M | 13.5% |
| sentiment | 31,232 | 3.5M | 0.16M | 7.0% |
| chat | 10,000 | 1.1M | 0.38M | 2.3% |
| memory | 40,000 | 12.9M | 5.2M | 25.9% |

`{split}.categories.npy`: one row per document (start token, length, category id; ids in `metadata.json`
`category_ids`), to rebalance the categories without rebuilding.

```bash
uv run --with pyarrow --with pandas python datasets/build_agent_mix_v2.py [--samples 2]   # ~45 s; curls in its docstring
```

Cosmopedia (`cosmopedia-small-bpe1024`, `datasets/build_cosmopedia.py`): HuggingFaceTB/cosmopedia (rev 0ae6ec6,
Mixtral-written textbooks / tutorials from a prompt) as `<bos><user>prompt</user><assistant>text</assistant><eos>`,
tokenized with the agent-mix-v2 BPE (same ids: agent-mix-v2 runs train on it as is); the loss covers the assistant
text; uint16 ids (`dtype` in the metadata), parquet streamed file by file and deleted, ~3.5M tokens/s at 5 tokenizer
threads. Built on the pod and stopped after 4.45B tokens (auto_math_text, khanacademy, part of openstax; documents
median ~2,100 tokens), then combined in place by `datasets/build_cosmo_agent_mix.py` (fixes the interrupted files: they
end after their last complete document) into `cosmo-agent-mix-bpe1024`: cosmopedia + agent-mix-v2 train x30 (25% of
5.94B tokens), validation 16.2M tokens (cosmopedia's 2,000 per subset + agent-mix-v2's once; not comparable with
agent-mix-v2's val_loss). Categories: agent-mix-v2's and `cosmopedia`. The other subsets (`--subsets stories
web_samples_v1 web_samples_v2`, ~80 GB of parquet) need a bigger disk.

Large corpora with the static memory: `PackedDocumentDataset(static_memory=...)` keeps nothing per token. The rows'
stream (the corpus without span contents) is a list of runs of consecutive corpus positions (`kept_runs`, built in
50M-token chunks cut at document starts, cached at `<path>.kept_<spans>.npz`), and the read / loss masks and each
read document's span tokens are computed per batch from the documents a row touches (44 ms for 96 rows). Batches
are identical to the per-token version (checked on agent-mix-v2's validation tokens). Without the static memory
`loss_target_mask` still builds per-token arrays (int64: too big for billions of tokens).
Run `mingru-16L-cosmo-agent-mlp-cuda`: all weights from `mingru-16L-v2-mem-mlp-cuda` (epoch 70, val 0.457), fresh
optimizer (warmup 200), epochs of 0.02 pass (2,265 steps, ~11 min; a pass ~9 h), ~290 ms per step, 17 GB.

```bash
uv run --with pyarrow python datasets/build_cosmopedia.py [--subsets khanacademy ...] [--threads 5] [--samples 2]
```

Pretraining mix `cosmo-code-math-pretrain-bpe1024` (all `<bos>document<eos>`, every token scored, no tags, the
agent-mix-v2 BPE): 70% `cosmopedia-pretrain-bpe1024` (2.93B, `datasets/build_cosmopedia_pretrain.py`: the 8 subsets
balanced by water-filling), 25% `stack-edu-bpe1024` (1.05B, `datasets/build_stack_edu.py`), 5% `finemath-bpe1024`
(0.21B, `datasets/build_finemath.py`), concatenated by `datasets/build_pretrain_mix.py` (categories of every part:
the 8 subsets, the 15 languages, `finemath`). Stack-edu (HuggingFaceTB/stack-edu rev eeec5ca, The Stack v2 files
kept by per-language educational classifiers) ships only blob ids: the contents are read anonymously from
`https://softwareheritage.s3.amazonaws.com/content/<blob_id>` (gzip; no AWS credentials needed), 128 downloads at a
time (~220 files/s on the pod, ~80 min). Per language (C, C#, C++, Go, Java, JavaScript, Markdown, PHP, Python, Ruby,
Rust, SQL, Shell, Swift, TypeScript) an equal share, from one random metadata shard: files of 200 B..50 KB, the
highest classifier-score threshold (4.5, 4, 3.5, ...) that holds 3x the share. Code costs ~1.4-1.6 bytes per token with
this BPE (prose ~2.2). Finemath: HuggingFaceTB/finemath rev e92b25a, `finemath-4plus` (math pages from Common Crawl,
classifier score >= 4), random shards, 1.91 chars per token. On the pod the parts sit on the container disk and the
mix on `/workspace/datasets` (linked into `datasets/`).
Run `gdn-16L-256-moe6-mix`: `gdn-16L-256-moe6-pretrain` at epoch 118 (val 1.2524 on cosmopedia-pretrain; its
checkpoints are kept locally only) continued on the mix with a fresh optimizer (warmup 200, lr 1e-3), epochs of 0.02
pass (1,704 steps, ~8.4 min at ~295 ms per step), 100 epochs = 2 passes (~14 h), `checkpoint_every` 50. Stopped
after epoch 52 (val 1.2931) to train `lru-banks` on the pod; latest + best pulled here. It trained with the default
`data.corrupt_prob` 0.15 (15% of the input tokens replaced at random; see `lru-banks` below).

Back to text (to retrain the tokenizer without downloading again; byte-level BPE is lossless, checked by
re-encoding): `datasets/decode_to_text.py --dataset D [--cpu-fraction 0.8]` writes `datasets/raw/<D>-text/`: per split
shards of UTF-8 text + int64 byte offsets (documents + 1) + int16 categories, and `metadata.json`. Workers: the
fraction of the container's CPU quota (cgroup `cpu.max`; the pod has 7.65 CPUs although `nproc` says 96). ~8 small
shards per worker: on the pod, 6 growing 1.5 GB files made the filesystem reserve several GB extra until closed (the
first run nearly filled the disk). The mix: 8.68 GB of text, 2,715,102 documents, 48 + 29 shards, 135 s on the pod
(container disk: lost if the pod stops; rebuilt in ~2 min).

`datasets/train_bpe_from_text.py --text DIR --vocab-size N [--sample-bytes 1e9]`: a byte-level BPE on a random sample
of that text with the agent tokenizers' fixed ids (16 tags at 0-15, byte b at 16 + b, then merges; GPT-2
pre-tokenization), checked lossless on every validation document and compared per category with the old tokenizer
(`tokenizers/<name>/report.json`); caps its own memory (RLIMIT_AS, default 20 GB) so an OOM cannot hit the training
run. `pretrain-mix-bpe32768` (32,768 ids, 1 GB sample, 87 s on the pod, ~3 GB RSS): 4.12 characters per token on the
mix validation vs 1.98 for agent-mix-v2-bpe1024 (0.48x the tokens): prose 4.3-5.5 (0.42-0.50x), code 2.7-4.3
(0.45-0.57x), finemath 3.48 (0.56x).

`datasets/encode_text.py --text DIR --tokenizer T --name N [--out DIR] [--cpu-fraction 0.8]` (the inverse of
decode_to_text): each document as `<bos>` + text + `<eos>`, output exactly like `build_pretrain_mix.py` (uint16 ids,
`{split}.categories.npy`, metadata with per-category documents / tokens, `avg_chars_per_token` = characters / tokens);
one job per text shard, results appended in order (no temporary files: the pod's container disk is small); checks
document boundaries and that validation documents decode back byte for byte. `pretrain-mix-bpe32768` (the mix with
the 32k BPE, on the pod in `/workspace/datasets`, 576 s at 6 workers): train 1.965B tokens (0.47x), 2,715,102
documents, 4.38 characters per token (validation 11.7M tokens, 4.11); documents p50 653 / p99 3,742 tokens.

```bash
uv run --with pyarrow --with requests python datasets/build_stack_edu.py [--budget N] [--languages Go ...] [--samples 2]
uv run --with pyarrow python datasets/build_finemath.py [--budget N] [--samples 2]
uv run python datasets/build_pretrain_mix.py --parts cosmopedia-pretrain-bpe1024 stack-edu-bpe1024 finemath-bpe1024 \
    --name cosmo-code-math-pretrain-bpe1024 [--out DIR]
```

Terminal trajectories (nvidia/Nemotron-Terminal-Corpus, `skill_based_easy`, CC-BY-4.0, rev 67a3e03; parquet in
`datasets/raw/nemotron-terminal/easy/`): Terminus-2 agents (DeepSeek-V3.2) solving Linux tasks, converted to the
mixed format by `datasets/convert_nemotron_terminal.py` (tools `terminal(keystrokes, duration)` and
`task_complete()`, the screen after each batch as `<result>`; the JSON-format instructions dropped, their first
paragraph kept as `<system>`). Kept 19,950 of 44,809 trajectories (those where every assistant message is valid
JSON and the harness gave no warnings), 769 MB, documents median 34 KB (p99 92 KB): ~70x our synthetic documents.
Output `datasets/raw/nemotron-terminal/easy_converted.jsonl` (`text`, `skill`, `task`); not a dataset yet.

```bash
uv run --with pyarrow python datasets/convert_nemotron_terminal.py [--samples 1]   # ~10 s
```

## Bank memory with an LRU (`model/archs/bank_memory.py`, plain torch, not an arch yet)

KEY FINDINGS, TO USE IN THE REAL MODEL (details below):
- ONE bank layer does induction (~99% of the repeated copy by step 200-600; one LRU or one GDN layer: chance).
- ONE bank layer with TWO HOPS (read, FFN, query from the stream, second read of the same banks) does two-hop recall:
  91.6% at 1,000 steps where two stacked bank layers reached 13-15%. The circuit study shows real composition at
  inference (redirecting a -> b_j gives c_j 91.9%). It stays fully parallel: reads never change the banks, so hops
  share one write pass.
- Cost: a hop is +56% of a layer's train step (+13% params); a second layer is +147%. Hops search the same key space
  ("the context was X"), which is why they succeed where stacking stalls (layer 2's keys live in another space).
- Banks learn standard recall tasks faster than GDN (MQAR in 200 steps vs 600-800).
- Settled design: "mass" normalization (averages), top-2 renormalized routing, 2 heads with keys >= 32 dims, LRU
  keys (state before the token) and queries (state after); the LRU state encodes order (last ~3-4 tokens).
- The fast-feedback loop (CPU, batch 128, lr 6e-3, 60-90 s runs, `bank_recall_demo.py` / `twohop_circuits.py`)
  is the way to test ideas before scaling.

A slot memory read by content: an LRU (complex diagonal, `h_t = lambda h_{t-1} + gamma B u_t`) gives the key (its
state before the token: the context it followed) and the query (after the token); the value is the token's projection;
each key is filed in the top-K of M slots by a router on the key (`topK(softmax(k R))`, renormalized to sum 1), into a
bank of keys and a bank of values, with a fixed decay per slot (log-spaced half-lives); the read is
`softmax(tau (Kb q) / z)` over the non-empty slots against each slot's average key, then the average values.
`norm`: "mass" (divide by the written mass z: averages), "ema" (writes x (1 - a)), "lru" (x sqrt(1 - a^2)); a per-slot
write scale cancels under "mass". The module doc proves the chunked form equals the token loop: every coefficient
(decays, routing, resets) is known before the recurrence and the read never feeds back, so the banks are a linear
recurrence (chunked matmuls, one state per chunk boundary). `tests/test_bank_memory.py`: loop vs chunks (values and
gradients, float64, 1e-10, resets, every chunk size and norm), carried state, the read not feeding back, routing, the
layer row-wise vs position by position. The torch chunks materialize (B, H, M, C, C) decay masks: a reference for
fla's `chunk_gsa`-style kernel, not for 512 slots at scale.
`ad_hoc_scripts/bank_recall_demo.py [--arch bank|lru|gdn|all]`: MQAR (key -> value pairs, then keys asked again), one
layer, CPU. 16 pairs, 64 slots: bank 98.9% recall at step 200, 100% at 800; LRU alone 4.1% (chance 1.6%); one
`gated-deltanet` layer (2 heads x 32) 2.6% / 66% / 94.5% / 97.4% at steps 200 / 400 / 600 / 800. 48 pairs in 64 slots:
bank 99.8% at step 600. `--task induction` (the textbook test: 24 distinct random tokens of 128, then the same again;
copy 1 is unpredictable, copy 2 needs induction): bank copy 2 96.3% at step 200, 98.8% at 600, copy 1 at chance
(0.8%: no leak); LRU alone and one `gated-deltanet` layer stay at chance on copy 2 through 600 steps (the GDN layer can
express induction with its short conv, as in H3 / Based; it did not learn it in that budget).
Valuable finding (induction, the standard test): ONE bank layer does induction, ~99% of the repeated copy predicted
by step 200-600, while a single LRU cannot (chance) and a single GDN layer had not learned it after 600 steps.
Write normalization under superposition (MQAR, 32 pairs written top-2 into 16 slots: ~4 items per slot; recall at
steps 200 / 400): "mass" (averages) 46.7% / 58.6%, "none" (raw sums) 26.2% / 28.3%, "lru" (sqrt(1 - a^2)) 7.2% / 8.8%,
"ema" (1 - a) 2.0% / 4.7%. Without the division, slow slots get tiny writes (1 - a ~ 7e-4 at a 1024-token half-life),
their scores stay near 0 and only the fast slots, which forget, are read; raw sums let heavily written slots dominate.
The average keeps every slot at full scale whatever its decay or write count: it is what protects superposed items
here (short rows of 81 tokens and 400 steps; with ema / lru the per-head temperature would also have to grow a lot).
Decision: the bank layer uses "mass" normalization (averages) with top-K writes renormalized to sum 1 (top-2).
`--task copy` (the copying benchmark of "Repeat After Me", Jelassi et al. 2024: <bos> s <sep> s, random strings over
32 letters with repeats; one bank layer, 64 slots x 2 heads, top-2, trained 60 s = 452 steps on lengths 8-64), token
accuracy / exact string per length: L 8 1.000 / 1.000, 16 0.997 / 0.953, 32 0.929 / 0.113, 48 0.768 / 0, 64 0.588 / 0,
96 0.315 / 0, 128 0.181 / 0 (96 and 128 untrained). Copies short strings perfectly; from ~32 letters (64 writes per
head into 64 slots) the items overlap in the slots and accuracy falls with length: the fixed-state limit that paper
measures for SSMs (a transformer copies at every length). One minute of training: not yet separated from capacity.
After the speedup below, 90 s of training (`--seconds 90`, weights in `runs/_synthetic/copy/`), token accuracy /
exact / bits recovered:
- 1 layer x 64 slots (1,606 steps): L 32 0.993 / 0.809 / 157, 48 0.975 / 0.312 / 226, 64 0.941 / 0.039 / 281,
  96 0.796 / 0 / 313 (peak, untrained length), 128 0.557 / 0 / 233.
- 2 layers x 32 slots each, one shared FFN (`--layers 2`, 1,163 steps): L 32 0.984 / 0.582 / 154, 48 0.941 / 0.082 /
  211, 64 0.881 / 0 / 249 (peak), 96 0.649 / 0 / 224, 128 0.442 / 0 / 160.
Copying is one lookup: splitting the slots over two layers halves each layer's capacity and the second layer adds
none (and it trained fewer steps in the same time). With more training the single layer recovers ~313 bits (~63
letters, about one item per slot), twice the one-minute 147: the one-minute limit was mostly training time.

- 1 layer, 4 heads x 16 (keys and values; 64 slots per head, about the same bank state: `--heads 4 --head-dim 16`,
  1,259 steps at 117 ms vs 78 ms for 2 x 32): L 32 0.991 / 0.738 / 156, 48 0.935 / 0.027 / 208, 64 0.794 / 0 / 208
  (peak), 96 0.481 / 0 / 137. More, smaller heads did not reduce the noise: 16-dim unit keys overlap more than 32-dim
  ones and the heads hold the same items (the peak falls from ~313 to ~208 bits; also fewer steps in the time).
- 1 layer, 1 head x 64 (`--heads 1 --head-dim 64`, 1,957 steps at 73 ms): L 8 0.989 / 0.914, 16 0.974 / 0.695,
  32 0.923 / 0.098 / 135, 48 0.854 / 0.004 / 177, 64 0.779 / 0 / 201 (peak), 96 0.581 / 0 / 187.
Heads at the same total key / value width (64), 64 slots per head, 90 s: 1 x 64 ~201 bits, 2 x 32 ~313, 4 x 16 ~208:
a sweet spot at 2 heads. One head has a single router and one set of 64 slots (errors even on short strings); two
heads can file different items (twice the slots) with keys still wide enough; at 16 dims the keys get too noisy.
Rule of thumb for larger models: as many heads as keep the key size >= ~32-64 (one seed, small model: verify at
scale).
Two-hop recall on ONE bank layer (`twohop_circuits.py --variants bank`, 1,500 steps, 44 s): fails as expected:
accuracy 6.7% (chance 2.1%), and 6.7% is what a shortcut gives: the answer c only ever appears as the second token of
a pair, so "pick a token seen in second position" (16 candidates: the b's and c's) scores ~1/16 = 6.25%. It answers
the one-hop b 5.0% of the time. Its slots partly collapsed (usage entropy 0.60 / 0.66 of max, 56% / 62% dead): a
layer that cannot solve the task stops spreading its writes.
Two-hop on TWO bank layers (`--variants bank+bank`, 32 slots each, 1,500 steps, 78 s): 13.2% (2.3 / 2.9 / 8.8 /
12.1 / 14.8 / 14.5% at steps 250-1500, still rising): above the 6.25% shortcut, not solved yet. Both layers are needed:
without layer 0's read 2.3% (chance), without layer 1's read 6.6% (the shortcut). Layer 1's read is still diffuse
(max probability 0.12-0.13 at the queries, it lands on a slot holding c only 11-13% of the time); slots 31-53% dead.
The second hop is learned slowly: layer 1's query must come from what layer 0 retrieved (b as a read-out value),
while its keys come from b as a token in the pairs, two different representations it must align.
Hops (`LRUBankLayer(hops=n)`, demo `--hops n`): the layer is `write` (LRU, keys, routing, the chunk prep: once) then
`read(mem, q, hop)` any number of times; hop n+1's query is `normalize(W_q^(n+1) stream)` from the stream after hop n
and the shared FFN, its own temperature and output projection. Still parallel: reads never change the banks
(`TestHops`: chunks == loop == generation, values and gradients). Copy, 60 s, batch 128, lr 6e-3, 64 slots x 2 heads
(peak bits / L 32 exact): 2 hops 427 steps (248 ms/step) 197 / 0.52; 1 hop 588 steps (184 ms) 237 / 0.68; 1 hop at
the same 427 steps 188 / 0.42; 2 hops at the same 588 steps (86 s) 263 / 0.77 (L 48 0.963 / 0.176, L 64 0.908,
L 96 0.691 / 248 bits). Per step hop 2 helps (+26 bits at 588 steps, one seed); per second it loses. Copy is a one-hop
lookup, so this only shows the cost; the test for it is two-hop recall.
2 layers x 2 hops (32 slots per layer, shared FFN; 588 steps, 136 s, ~231 ms/step; weights
`runs/_synthetic/copy/bank_2layer_2hop_64slots_588steps_b128.pt`): L 16 0.994 / 0.918, L 32 0.954 / 0.262 / 144
bits, L 48 0.886 / 0 / 188, L 64 0.790 / 0 / 206 (peak), L 96 0.517 / 154, L 128 0.298 / 83. Less storage than 1
layer x 64 slots (1 hop 237-259 bits, 2 hops 263) at the same steps: splitting the slots halves each layer's bank
and copying gains nothing from depth (as at batch 64: 2 layers x 1 hop 249 vs 313 bits).
Query-routed erase (`LRUBankLayer(erase=True)`, demo `--erase`). Before the write at t, the slots where the query
of t - 1 would be filed (`erase_address`: the top-K of softmax(normalize(W_q s_{t-1}) R) divided by the largest) shrink
by `e_t = erase_max sigmoid(w s_{t-1} + b)` per head (max 0.95, bias -4: starts near 0). It is a data-dependent
decay, `log a_tj + log(1 - e_t rho_tj)`. Every factor comes from the LRU state, never a read, so it stays a linear
recurrence. The chunks use the cumulative log decay, exp(L_t - L_i), with the chunk cut so exp(-L) < e^30
(`TestErase`: chunks == loop == generation, values and gradients). Under "mass" an erase does not change a slot's
average; it lowers the old items' share against the next write, so it overwrites. Cost: +28% per step (161 -> 207 ms,
copy, batch 128 x 129).
- Copy, 588 steps: erase 266 bits peak (L 32 0.990 / 0.750, L 64 0.915, L 96 0.721), none 259 (0.992 / 0.785,
  0.902, 0.693). Even. Copy has no overwrites, and run-to-run noise is ~+-20 bits: "none" here vs 237 for the
  same 588 steps earlier, after a float-rounding change in the chunk code.
- MQAR with overwrites (`--overwrites 4`: 16 pairs, then 4 keys get new values, 8 queries = those 4 + 4 unchanged),
  latest-value accuracy on overwritten keys / old value returned, steps 200 / 400 / 800: erase 0.975 / 0.023, 0.991 /
  0.009, 0.997 / 0.003; none 0.873 / 0.122, 0.980 / 0.020, 0.981 / 0.018 (overall 0.997 vs 0.968 at 800, and the
  no-erase run had a loss spike at 800). Erase learns overwrites faster and ends cleaner. Without it the recency of
  the slot decays still gets most of the way, because the overwrite comes only ~10 tokens before its query here.
- With a longer gap (`--gap 24`: 24 filler pairs with fresh keys, 48 tokens, between the overwrites and the queries),
  400 steps, latest / old value on overwritten keys, unchanged keys, overall: erase step 200 0.879 / 0.088, 0.945,
  0.912; step 400 0.981 / 0.017, 0.995, 0.988. None step 200 0.746 / 0.234, 0.972, 0.859; step 400 0.945 / 0.046,
  0.896, 0.921. The gap makes it harder without erase (the old value wins 23% at step 200, and the unchanged keys
  suffer too); erase stays near the short-gap results (one seed each).
- Read-driven erase (`erase="read"`, demo `--erase read`): two banks by role, a key bank (keys + mass, never erased)
  that addresses, and a value bank (values + mass) erased before the write at t + 1 where the key bank's read at t
  landed (p / max p), by a gate on [s_t, matched average key, best match]. The value bank depends on the key bank's
  reads, never the reverse: two linear recurrences in sequence (`TestReadErase`: chunks == loop == generation,
  values and gradients; gate shut == the plain layer). Slower than expected: 325 ms per step (query erase 218, plain
  165), from two write passes, the extra key aggregation and layout changes. Same gap test, 400 steps: step 200
  0.859 / 0.102, unchanged 0.933, overall 0.896; step 400 0.950 / 0.041, 0.987, 0.969. Between the query erase
  (0.981) and none (0.945) on overwritten keys, at 1.5x its cost. Not kept: the simple bank stays the default.
Two-hop recall SOLVED by ONE bank layer with two hops (`twohop_circuits.py --variants bank2`: read, FFN, a query
from the stream, a second read of the same banks, the shared FFN again; 32 slots, 1,000 steps, 71 s, batch 128, lr
6e-3; weights `runs/_synthetic/twohop/bank2_s0.pt`). Accuracy 2.3% / 6.8% / 83.4% / 92.1% at steps 250 / 500 /
750 / 1000 (final eval 91.6%; chance 2.1%), against 13-15% for two stacked bank layers at 1,500-2,000 steps. The
circuit is the intended one:
- the first read's top slot holds b (written by pair a b) 99.8% / 99.2% of the time per head (max read probability
  0.98);
- the second read's top slot holds c (pair b c) 94.9% / 94.5% (max probability 0.89);
- without the second read: 2.3% (chance), and the stream then answers b 15% of the time; the final answer is b only
  0.2%.
Circuit study (`--study runs/_synthetic/twohop/bank2_s0.pt`, 256 rows x 4 queries), all as expected:
- logit lens inside the layer, top-1 = b / = c: embedding 1.8 / 1.7%, + read 1 39.0 / 1.1% (b appears), + FFN 32.2 /
  1.6%, + read 2 0.6 / 91.1% (b replaced by c, p(c) 0.86), final 0.6 / 91.4%;
- read 1 puts 97-99% of its mass on the slots holding b, read 2 91% on the slots holding c (~9% crossover: slots
  shared under top-2 routing);
- query 1 ranks the key stored with b (filed under "context a") 1st of 44 in 98.2% of cases; query 2 ranks the key
  stored with c ("context b") 1st in 94.3% (mean rank 1.1-1.2). Query 2 has cosine 0.76 with the LRU's own query
  right after b in the pair (b c) (0.003 for another chain's b): the FFN rebuilds the query "context b" from the
  retrieved value;
- counterfactuals: (b c) -> (b c') with c' fresh: answers c' 91.7%, c 0.0%; (a b) -> (a b') fresh: c 7.0% (the
  shortcut level); REDIRECT (a b) -> (a b_j), b_j the middle of another chain: answers c_j 91.9% (never paired with
  a), c 0.5%, b_j 0.4%; without read 2: 2.4%.
So it composes two lookups at inference time, not a stored a -> c association.
Three-hop chains (`--chain-hops 3`, a b / b c / c d shuffled, "? a" -> d; `bankN` = N hops): stopped early, no
result yet. Three runs in parallel with 2 threads each were 3-4x slower per step. Accuracy so far: 1 layer x 3 hops
4.4% at step 1,250, 2 layers x 2 hops 2.6% at 1,000, 1 layer x 2 hops (control) 4.3% at 1,750 (chance 2.1%).
Next try: one run at a time at full threads, and mixed 1/2/3-hop questions so the reads get a signal early.
2 layers x 2 hops alone (`bank2+bank2`, 2,000 steps, 366 s, ~183 ms/step; `runs/_synthetic/khop3/bank2+bank2_s0.pt`):
2.0% until step 750, then 7.8 / 10.8 / 13.1 / 14.2 / 14.2% at steps 1,000-2,000: a plateau at the SHORTCUT level,
not the chain. The task's shortcuts: the answer d is the only kind of token that never appears first in a pair (8
chain ends: 12.5%), and earlier questions' answers can be ruled out (mean over 4 questions ~15.5%). Signs it is the
shortcut: layer 0's reads never land on a slot written by any link of the asked chain (0.0% for all 3 links) and its
second read is worthless (without it 14.6%); layer 1's reads are diffuse (max probability 0.22-0.44, top slot on a
link 13-44%). Resumed (fresh optimizer, warmup 100) and stopped at step 4,000: 16.5 / 16.1 / 16.4% at steps
3,500-4,000, just above the ~15.5% shortcut ceiling and barely moving (not saved; the 2,000-step weights remain).
To test real chaining the task needs shortcut-proofing: an extra never-asked pair "d e" per chain (ends
also appear first), mixed question depths, one question per row.
Why it works where two layers stalled: both hops query the same key space (the key is "the context was X"), so the
FFN only has to map the retrieved value b to the query for "the context was b". In two layers, layer 1's keys came
from its own LRU over a different stream. Phase change between steps 500 and 750, as in induction-head formation.
Needle in a haystack (`ad_hoc_scripts/niah_bank.py`: filler tokens from their own vocabulary, needles "KEY k v" at
random depths, questions "ASK k"; `--task twohop`: "KEY k1 k2" + "KEY k2 v", ASK k1 -> v, 4 chains per row (3
distractors: picking a chain = 25%), plus a never-asked "KEY v x" per chain so the end is also a key; trained on
haystacks of 16-96; weights `runs/_synthetic/niah/twohop_bank2_s0.pt`). One bank layer x 2 hops, 64 slots, 1,000
steps (~3.5 min): NOT learned. Accuracy at haystack 96: 0.4 / 9.0 / 14.1 / 11.3 / 9.4% at steps 200-1,000, below
the 25% floor of picking a chain. Final accuracy by length 64-2,048: 5-12%. The needles nearest the question score
best (last depth fifth 33-50% at 64-256), the deepest 2-8%: it gets only recent needles, and the filler writes
(every token writes into the banks) seem to bury the rest. Untested: one-hop (`--task single`), more steps or
questions per row (only 2 targets per row here, 256 per step).
LRU input gate (`LRUBankLayer(input_gate=True)`: g_t = sigmoid(w x_t + b) from the layer's input, lambda_t =
lambda^g_t, gamma_t = sqrt(1 - |lambda_t|^2), so g = 0 skips the token; still linear (cumulative log decays in the
chunks); `TestInputGate`). Two-hop needle task, 2 layers x 2 hops, gate on layer 1 only (`--layers 2 --gate-second`,
1,000 steps, ~150k params; `runs/_synthetic/niah/twohop_2x2_gate_s0.pt`): accuracy at haystack 96 5.1 / 8.6 /
10.9 / 12.1 / 11.7% at steps 200-1,000; by length 8-19% (small samples past 256). Still below the 25% floor, a bit
above the 1-layer run (9.4%). The gate did not learn to skip anything: filler 0.894, KEY 0.905, needle entities
0.893, ASK 0.881 (from 0.95 at init). Two layers also train ~2x slower per step.
Learned LRU decays per layer (half-lives in tokens: median / usage-weighted median; init 0.5-4 log-spaced, median
1.37; rotations init uniform):
- solved 2-hop (`bank2`, 1 layer): 0.39 / 0.38. It shrank below the init: the key is just the previous token.
- 3-hop stuck at the shortcut (`bank2+bank2`): layer 0 3.52 / 3.85, layer 1 3.28 / 3.44. Both lengthened, with
  slow non-rotating channels (half-life 24, 34, 42; period 66-195 tokens) that average many past tokens (a bag-like
  memory of what was seen: what the "never first in a pair" shortcut needs).
- stalled 2-hop (`bank+bank`): 2.74 then 1.83.
- copy: 1 layer 2.81 / 2.97; 2 x 2 hops 1.79, then 0.75 in layer 1; 2 x 1 hop 1.87, then 3.32 (no consistent role
  for layer 1).
Long channels with rotation ~pi (period 2 tokens: a parity clock, first vs second token of a pair) appear in every
pair-task model (half-life 10-60). In copy the long channels rotate with a period of ~14 tokens instead (position
clocks).
The models that solved the pair tasks used short keys. The ones stuck on the shortcut grew long, slowly-rotating
memory that mixes earlier pairs into the key. That points to the previous-token key (`StreamBankLayer`, or a short
LRU init) for pair-structured recall.
Order, not a bag of words: the LRU state is `h_t = sum_j lambda^j b_{t-j}`; swapping the tokens at offsets i, j changes
it by exactly `(lambda^i - lambda^j)(b_x - b_y)` (a bag is lambda = 1). `tests/test_bank_memory.py` checks the
identity and that all 120 orders of 5 tokens give distinct states (one state at lambda = 1).
`ad_hoc_scripts/lru_order_probe.py [--ckpt X]` measures how much order survives. On the trained copy layer (half-lives
0.7-19, median 2.4), a linear probe on the state gives the token at offset 0 / 1 / 2 / 3 / 4 / 6 / 9 with 1.00 /
1.00 / 0.94 / 0.74 / 0.54 / 0.28 / 0.14 accuracy mid-sequence (chance 0.03). After 8 distinct tokens, a bag scores
at most 0.125 at any offset; the state scores 1.00 / 1.00 / 0.97 / 0.82 / 0.65 / 0.49 / 0.36 / 0.30, and the key
nearly as much. The two-hop layer 0 and a fresh layer are similar but fade faster. So the key is an ordered context
of about the last 3-4 tokens, fading after that. It holds no position and no long-range order: the order of the
pairs 20 tokens back is gone. Weak spot: key cosines between orders of the same 4-6 tokens are 0.82-0.93 (trained).

PERFORMANCE RULE for chunked recurrences with per-channel / per-slot decays (apply at large scale): factorize the decay
`a^(t - i) = a^t * a^(-i)` so every in-chunk term is a plain matmul ((C x C) scores times (C x M) decayed writes),
batched over all chunks at once, and carry only the chunk-start states in a short loop (one multiply-add per chunk).
Never materialize a (slots x C x C) decay tensor: per chunk it made thousands of tiny ops (the per-op overhead
dominated; a 2nd layer cost +50%), batched over chunks it flooded memory (slower still). Cap the chunk so a^(-i) stays
below e^30 for the fastest decay (`_safe_chunk`); GLA-style kernels do the same with sub-chunks. Measured (CPU,
batch 64 x 129 tokens, 2 heads, 10 steps): 1 layer x 64 slots 177 -> 89 ms per step, 2 layers x 32 slots 269 -> 106
ms (+19% for the 2nd layer instead of +52%); exactness tests unchanged.
Valuable finding: on these standard synthetic recall tasks the banks learn much faster than the alternatives: MQAR
solved within 200 steps where one GDN layer needs ~600-800, and induction within 200 steps where one GDN layer did not
get there in 600. The design builds the lookup in: the key is the context before the token, the slots are addressed
by content, so the layer does not have to discover the circuit by gradient descent.

## Dense supervision

```bash
model config --run-name X set training.dense_supervision true
```

RoPE archs only. Each training sample is `2T-1` tokens and predicts the next
token at all `T` last positions (each with a full `T` window) in one pass,
with the fused readout loss (`model/fused_linear_cross_entropy.py`, constant
logits memory). About 1.2 s/step at batch 512 x 128 for ~65k targets, i.e.
~6x more targets per second than single-target training. Validation stays
single-target, so `val_loss` is comparable with every other run.

## Web chat

```bash
uv run python webchat/server.py [--port 8000] [--device xpu|cpu]    # open http://localhost:8000
```

A local page (stdlib server, `webchat/server.py` + `webchat/index.html`) to chat with any run / checkpoint: tokens
stream to the page (server-sent events). The sidebar builds the document as in training: a system prompt (the
training prompts of `build_agent_mix_v2.SYSTEM`, editable), tool lists (the training tool lines, presets, unneeded
tools), example requests (written ones plus first user messages of the run's validation data, by category).
The model's `<call>`s run for real and their `<result>` is appended, then the model continues (up to "tool rounds"):
`run` / `read_file` / `write_file` on the synthetic dataset's simulated shell and a random training-style workspace
(files listed in the sidebar), `calculator`, `python` (a separate `python -I` process, temp dir, 10 s timeout: it runs
the model's code on this machine), every memory tool naming (persisted in `.webchat/memory.json`). Other listed tools
answer `{"error": "tool not available"}`. "Raw document" shows the exact text fed to the model.
Thinking: only the Atum09 documents think (27k of 179k), and nearly all list Atum09's own tools, so the models think
when those tools are listed (P(`<think>`) after `<assistant>` ~1.0 on `mingru-16L-v2-mem-cuda`, any request) and
almost never otherwise (~0.0 with the synthetic tools or none; "What is Big O notation?", 409 Atum09 copies, also
needs them). Setup has them under "Atum09 tools" and as "atum09 (thinks): ..." tool sets, with its five system prompts;
`execute_bash`, `list_directory`, `execute_python`, `read_file`, `write_file`, `memory_search` run, the rest answer
not available.

The page: a top bar with a searchable model picker (run name, arch or dataset; the last choice is remembered), the
checkpoint, Chat / Inspect, New chat and Setup (a right drawer, for the chat only: system prompt, tools, example
requests, generation settings, the workspace and saved memories, the raw document). Under the last reply: Regenerate
(drops everything after your last message, tool calls included, and samples again) next to a temperature field (the
same as Setup's; at 0 generation is greedy, so the reply comes back the same). Edit on a user message (shown on hover) rewrites it
and answers again; everything after it is dropped (Enter sends, Escape cancels). Light, sober design (IBM Plex Sans /
Mono; one teal accent).

Complete (plain text completion, any run; the only mode for runs whose dataset has no `loss_spans`, e.g. the
pretraining run `gdn-16L-256-moe6-pretrain`: Chat and Setup are hidden): a prompt box, the continuation streamed after
the prompt (`/api/generate` with `"mode": "completion"`: `<bos>` + the text, stops at `<eos>` only), Generate (Ctrl+Enter),
Stop, Continue (more after the text so far), Regenerate (a new sample of the same prompt), Edit as prompt; its own
temperature (0.8), top-k (50), repetition penalty (1.1) and max tokens; starters: openings of random validation documents
per category (`/api/starters`). `/api/runs` gives each run's `mode` (chat / completion) and `inspect` (min-gru runs).
`gated-deltanet` generates on the Iris Xe with the torch delta rule (~10 tokens/s for the 56M-param model).

Inspect (`min-gru` / `min-gru-cuda` runs, `/api/inspect`, up to 1024 tokens): any text, a validation document or the
chat, shown whole; with a static memory the system / tool contents stay in place, dimmed (the layers never see them).
Three views: Loss (each token the training scores, inside the dataset's `loss_spans`, by the final readout's prediction
at the previous position: green where it was the top-1 choice, stronger with p; red where it was not, stronger with
its loss, capped at 5 nats: the reds are the mistakes; the top-1 share, the mean loss and entropy), Stored (each past token's share of the state at
the selected token: its exact weight in each channel, `gate_j * prod_{s>j} (1 - gate_s)`, times |candidate|, averaged
over the channels and the chosen layer or all; the last N tokens ignored, they always dominate) and Recall (static
memory: the assistant's tokens coloured by the size of the memories' addition, the system / tool tokens by the selected
token's read weights, `q . k_j` normalized per memory; the gates). The side panel: the selected token's probability,
loss and entropy, what the model expected there, the view's own details (most stored tokens; the most read memory
tokens with each memory's gate) and the next-token prediction layer by layer (logit lens after the embedding and each
layer). Arrow keys step through the tokens. A run trained on a pod (`device` "cuda") loads on the XPU here.

Inspect on `gated-deltanet` runs: Loss (as above) and Attention. Attention unrolls each head's delta-rule read into an
exact attention over the document's past tokens: `o_t = sum_i c_i v_i`, `c_i = beta_i k_i^T A_{i+1} ... A_t q_t`,
`A_s = alpha_s (I - beta_s k_s k_s^T)` (`delta_rule_read_weights` in `model/archs/delta_rule.py`: one sweep back from
t, O(t K) per head; tested to rebuild the loop's read to 1e-10 with document resets; the page shows the error against
the model's own read, ~1e-7 in fp32). `/api/inspect` keeps every layer's q, k, v (after the conv), g, beta and read
for the last text; `/api/attend` returns, for one position, c and each token's share of the read
`c_i (v_i . o_t) / |o_t|^2` (sums to 1; negative where a later write to a similar key took more than the token wrote)
for all layers and heads (~10 ms). The tokens are coloured by the selected layer / head (or the heads' mean): teal
positive, orange negative; "share of the read" or "weight on the value". The side panel: a layers x heads grid with
each head's most read past token, its share and the split this token / the conv's 3 before / older (click a cell to
select that head; hover: half-life, write strength), then the selected head's most read tokens and the logit lens.
A value already mixes its token with the 3 before it (the short conv), so "this token" means that window.

Inspect on `lru-banks` runs (`inspect_banks`: the torch path with each layer's routing weights r, values v, read
probabilities p over the slots and reads o recorded; `/api/banks` unrolls one position, ~0.1-0.2 s): Loss, Stored,
Recall, Gate, with Layer / Head / Read (hop) selectors. Slot m holds the average of its writes, position j weighted
`r_jm a_m^(t-j) / z_tm`, and a read takes `sum_m p_tm` of those averages, so each read is exactly
`sum_j c_j v_j` with `c_j = sum_m p_tm r_jm a_m^(t-j) / z_tm` (>= 0, sums to 1; the page shows the rebuild error,
~1e-7). Recall colours the positions by c; Stored by each position's share of the banks (the mean over the non-empty
slots of its share of each slot's average, the last N positions ignored). In layer 1 v_j is the token's own embedding
(W_v rms(embedding W_in)), so the colours attribute to input tokens; above, v_j is position j's stream (the token plus
what earlier layers read and the FFNs added), so the colours show where a read lands, not what it carries. For that
the side panel has "What each read retrieved": per layer and read, the positions taken most and the value lens, the
tokens the read's addition to the stream (o W_o, per head or all) raises most through the final norm's gain and the
readout (direct logit effect). Then every head of the selected layer (its most read position per hop) and, for one
head, the 3 slots it reads most (half-life, read share, the positions they hold). Gate: tokens coloured by 1 - g of
the LRU input gate (layers >= `gate_from`), and every layer's gate at the selected token in the side panel.

## Scripts

```bash
model scripts <name> --run-name baseline [args...]
```

One-off analysis scripts, each writing an HTML report to
`runs/<run-name>/reports/`. See `model scripts <name> --help` for arguments.

- `pga-dual-embeddings` — Principal Geodesic Analysis of the big embedding table vs. the small one.
- `spherical-k-means` — clusters both embedding tables and compares agreement.
- `table-neighbor-agreement` — checks whether the small table's neighbor structure matches the big table's.
- `query-recall-trace` — traces Block 1 attention and retrieval for hand-crafted name-recall prompts.
- `retrieval-behavior-trace` — same trace, sampled from real training data (hits vs. top-10 misses).
- `retrieval-ablation-loss` — measures the validation-loss cost of corrupting the retrieval step.
- `retrieval-ablation-loss-tokenid` — same measurement for `alibi-token-retriever-sphere`'s token-anchored retrieval.
- `token-retrieval-trace` — traces Block 1 attention, in-context query similarity, and the token-id scatter for `alibi-token-retriever-sphere`'s recall prompts (`--prompt-set names|nouns|long-context`).
- `order-sensitivity-loss` — measures the validation-loss cost of shuffling context token order.
- `name-swap-trace` — swaps the in-context name and checks whether the prediction tracks it.
- `token-frequency` — counts training-corpus occurrences for a list of words.
- `log-gaussian-kernel-trace` — for `rope-log-gaussian-sum`: which context distances the adaptive kernel weights per input, and what the adaptivity is worth in loss.
- `log-gaussian-kernel-stepper` — interactive page: step through a story (or `--text`, or `--generate N` to sample from `<bos>`) word by word and see the kernel the model generates at each step; for `rope-kernel-attention` also each head's attention.
- `rsm-bank-stepper` — for `rope-state-memory` (banks): step through a story and see each bank's writes, what it stores, and the read as attention over past tokens.
- `compare-runs` — no `--run-name`; interactive validation curves + ranking of all runs at any epoch N. Writes `runs/compare_runs.html`.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

Current best architecture: `alibi-retriever-sphere` (run `transformer-alibi-retriver`).
At 10k steps: baseline loss 3.3622, best arch loss 3.19.
