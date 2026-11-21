# lru-banks cross-entropy training on CUDA: review, optimizations, dead ends

Written 2026-11-20. Scope: the next-token (cross-entropy) training pipeline of `lru-banks` (`model/archs/lru_banks.py`,
`model/archs/bank_kernels.py`, `model/fused_linear_cross_entropy.py`, `model/archs/_dense.py`, `model/train.py`,
`model/data.py`), read against the literature. Nothing was run on a GPU for this review. The measured numbers come
from AGENTS.md (the 4-layer `lru-banks-4L-512` profile on the RTX A5000) and from your observation of the 6-layer run
(~2 updates per second at 64 x 512). Everything else is arithmetic from the code and is labelled as an estimate.

Requests covered:
- the most efficient pipeline for this architecture, down to the lowest level NVIDIA exposes;
- recompute instead of storing, as in FlashAttention;
- larger batches and longer rows (the batch-64 run was too noisy mid-training);
- a from-scratch C/C++ trainer specialized to this model;
- DeepSeek's training tricks (FP8 and others) and Meta's Byte Latent Transformer;
- your architecture idea: binary read / write gates replacing the LRU input gate, no `<bos>` null slot, and skipping
  the computation of gated-off reads.

## 1. Verdict

You are right that the current pipeline is far from the hardware.

- **Utilization:** the 6-layer step at 64 x 512 is ~8.2 TFLOP of training math. At ~500 ms per step that is ~16
  TFLOPS, about 15% of the A5000's 111 bf16 TFLOPS. A compute floor for the same math is ~120 ms per step: the dense
  matmuls at the 89 TFLOPS this GPU measured, the bank reads at 30 TFLOPS, the loss at 80 TFLOPS.
- **Where the time goes:** the gap is not instruction scheduling, so assembly will not close it. It is memory traffic
  and the kernel structure of the bank reads. In the 4-layer profile, fla's gated-slot-attention kernels take ~48% of
  the GPU time. They do ~1/7 of the math at ~3% of peak. They push slot-space tensors (16 heads x 128 slots = 2,048
  values per token, 4x the stream width) through GPU memory many times per read. They also compute a gradient for the
  slot decays, which are fixed, and then discard it.
- **Memory:** the same slot-space tensors are most of the activation memory (~90 KB per token per layer, estimated
  below). That is what caps the batch at 64 x 512 and makes long rows expensive.

The order of value, highest first:

| Level | What | Expected effect (estimate) |
|---|---|---|
| Algorithm / memory traffic | One fused bank kernel (state pass shared by both hops; read with the slot softmax in registers; decays from constants; compact top-2 routing; recompute in the backward); fused routing; fused loss | 2 to 2.8x faster step; 2 to 3x less activation memory per token |
| Removing overhead | No host syncs, static shapes, one CUDA graph per step, a static memory plan; a C++ trainer (section 6) | 10 to 25% on top, once the kernels are fast |
| Precision | FP8 GEMMs with DeepSeek-V3's fine-grained scaling, on Ada / Hopper only (no FP8 on the A5000) | up to ~2x on the FFN matmuls, which are 2/3 of the math; worth it only with quantization fused into the producer kernels |
| PTX | `mma.sync`, `ldmatrix`, `cp.async` (Ampere / Ada); `wgmma` / TMA (Hopper only) through inline PTX or CuTe | needed inside the custom kernels, not as a goal in itself |
| SASS | hand-edited machine code (CuAssembler) | dead end here: unofficial toolchain, +5 to 15% only on compute-bound GEMMs; our bottleneck is bytes |

The noise problem is solved today without new code. Gradient accumulation (`data.micro_batch_size`) already gives
any batch at the same tokens per second (section 8). The kernel work then makes each token cheaper and allows larger
micro-batches or longer rows.

## 2. What one training step does today

Per step at 64 x 512 = 32,768 tokens, 6 layers (hops `[2, 2, 2, 2, 1, 1]`), dim 512, 16 heads x 32, 128 slots,
vocabulary 32,768 with a 128-wide tied embedding.

- **Data:** `PackedDocumentDataset.__getitem__` (`model/data.py:434`) gathers 64 random windows from the memory-mapped
  uint16 corpus in the main Python thread. It returns int32 `x, y` (`y = -1` at document ends). That is cheap.
- **Forward:** `LRUBanksLM._run` (`model/archs/lru_banks.py:326`) builds `BankContext` (`bank_kernels.py:264`). That
  means cu_seqlens via `torch.nonzero`, fla's chunk indices, and a materialized `(B*T, 16, 128)` fp32 tensor of
  cumulative slot log decays. Then, for each layer (`_layer_fused`, `lru_banks.py:268`):
  - compiled RMSNorm + bf16 matmuls (`_f_pre_gated`);
  - the Triton LRU scan (one program per row and 32 channels, walking all T steps);
  - `_f_addr`: keys, queries, values, and router logits as a dense `(B, T, 16, 128)` fp32 tensor turned into a dense
    bf16 top-2 routing tensor `r`;
  - `MassScan` (the written mass `z`, dense `(B, T, 16, 128)` fp32);
  - per hop: `BankRead` = fla `chunk_gsa_fwd_k` (scores `ok`), a compiled softmax pass (`w`), fla `chunk_gla_fwd`
    (values), then `_f_post` (W_o, residual, FFN).
- **Loss:** `fused_linear_cross_entropy` (`model/fused_linear_cross_entropy.py:40`) on `readout_input(h)`, which is fp32
  (`lru_banks.py:373-375`), against the fp32 tied table. The matmuls run in TF32 (`gated_deltanet.py:69` turns TF32 on
  globally and the registry imports it). Chunks of 2,048 rows each write a 2,048 x 32,768 fp32 logits block (268 MB),
  run softmax as a separate kernel, and read the probabilities twice more for both gradients.
- **Backward:** autograd through the compiled pieces and the custom functions, including fla's `chunk_gla_bwd` and
  `chunk_gsa_bwd_k`.
- **Optimizer:** Keras AdamW on torch `_foreach` ops (`keras/src/backend/torch/optimizers/torch_adam.py`), with global
  norm clipping.
- **Host syncs per step** (the CPU waits for the GPU, then the GPU idles while the CPU queues the next kernels):
  - `int(valid.sum())` (`model/archs/_dense.py:146`);
  - `torch.nonzero` for cu_seqlens (`bank_kernels.py:273`);
  - fla's `prepare_chunk_indices` (a `.tolist()`);
  - `valid.nonzero()` (`lru_banks.py:353`);
  - the boolean indexing for the accuracy metric (`_dense.py:196`).

  All of them are computable on the CPU from the numpy batch before it is copied.

### FLOP budget (computed from the shapes)

| Part | Forward MACs per token | Training TFLOP per step |
|---|---|---|
| Dense matmuls (LRU in, W_k, W_q, W_v, W_o, hop-2 projections, FFNs) | 31.3M, of which the FFNs 21.0M | 6.16 |
| Bank reads (10 reads, two chunked passes each) | 5.9M | 1.16 |
| Loss (128 x 32,768 readout) | 4.2M | 0.82 |
| Total | | 8.17 |

The FFN (applied twice per two-hop layer, same weights) is 2/3 of the math. This matters for the gating ideas in
section 10.

### Where the GPU time goes (measured, 4-layer model, AGENTS.md)

566 ms per step, 57.9k tokens per second, 12.1 GiB at 64 x 512. GPU time:

- ~48% fla bank kernels;
- ~17% tensor-core matmuls;
- ~17% compiled elementwise;
- the rest: scans, the loss's softmax, the optimizer, copies.

The bank part does ~0.93 TFLOP per step in ~270 ms, about 3.4 TFLOPS: ~3% of the bf16 peak.

### Memory per token (estimate, checked against the measured 12.1 GiB)

12.1 GiB / 32,768 tokens / 4 layers = ~90 KB per token per layer. The estimate from the saved tensors agrees:

- **Slot-space tensors** (2,048 values per token): `z` 8 KB fp32, `r` 4 KB, per read `ok` 4 KB + `w` 4 KB + fla's
  intra-chunk matrix `Av` ~4 KB + chunk states `hk` / `hv` 2 to 4 KB each, and possibly the router logits (8 KB fp32)
  kept by the compiled routing backward. That is ~45 to 55 KB per two-hop layer.
- **The stream and the FFN:** ~35 to 40 KB.

So slot-space tensors are ~55 to 60% of the activation memory.

### Findings in the code (file:line)

| # | Finding | Cost |
|---|---|---|
| F1 | `BankContext` materializes the cumulative slot log decays as a `(B*T, H, M)` fp32 tensor (`bank_kernels.py:276-277`): 268 MB, read by every fla kernel of every layer and hop. The decays are fixed constants (`loga`, not trainable, identical in all layers), so `G = (position in chunk + 1) * loga` has a closed form. | Memory traffic in every bank kernel |
| F2 | fla's backward computes the decay gradient unconditionally: an fp32 `(NK, B*T, H, M)` buffer plus a reverse cumsum (`chunk_gsa_bwd_k`, checked in fla's source). `BankRead.backward` discards it (`bank_kernels.py:299-304`). | Pure waste, 2 of them per read |
| F3 | Two fla passes per read plus a separate softmax pass: `ok` and `w` go through GPU memory (`bank_kernels.py:286-289`). The softmax over 128 slots fits in registers. This is exactly what FlashAttention removed for attention. | ~3 slot-space tensors written and read per read |
| F4 | Hop 2 recomputes the chunk states that hop 1 already built; both depend only on k, r, v (also noted as a next step in AGENTS.md). | One extra state pass per two-hop layer |
| F5 | Routing is top-2 of 128 slots, but `r` is stored dense (`_f_addr`, `lru_banks.py:119-120`): 256 bytes per head where 4 would do (2 indices + 1 weight). The router logits go through memory as a dense fp32 tensor and back. | 64x more routing storage than needed |
| F6 | The written mass `z` is its own scan and an 8 KB per token fp32 tensor (`MassScan`, `bank_kernels.py:205`). It is the same decayed sum as the key state, with a constant key, so it can be one extra row of the state in the fused kernel. | 268 MB per layer, read by every read and its backward |
| F7 | The loss runs in TF32 on fp32 inputs and writes ~21 GB of logits / probabilities per step (16 chunks x ~5 passes over 268 MB): ~28 ms of memory traffic at 768 GB/s plus ~0.82 TFLOP at TF32 rate, an estimated ~45 ms. | ~9% of the step |
| F8 | `dr1.float() + dr2.float()` then a cast (`bank_kernels.py:304`), and `.float()` copies around the kernels. | Small, slot-space sized |
| F9 | The LRU and mass scans run one program per (row, 32 or 128 channels) walking all T steps. At 64 x 512 that is 512 to 1,024 programs (fine). At long rows with few of them (2 x 16,384) it is 16 to 32 programs on 64 SMs, each walking 16k dependent steps: latency-bound. `rsm_scan._segmented` solved exactly this for `rope-state-memory`. | Blocks long rows |
| F10 | Five or more host syncs per step (listed above), dynamic shapes (cu_seqlens vary per batch). | Blocks CUDA graphs; bubbles |
| F11 | `lru-banks` refuses `training.stateful` (`lru_banks.py:369-370`): the kernels take no initial state, although fla's do (`initial_state`, `h0`). | No truncated BPTT for very long contexts |
| F12 | At a `<bos>` the previous state is zeroed (`lru_banks.py:115`), so the key is `normalize(k_bias)`, the same constant for every document, and the value is the `<bos>` embedding's projection. Every document writes the same entry into the same 2 slots per head. Under mass normalization that slot's average stays exactly that value whatever its decay. A head that reads "nothing" reads it: a learned null slot, the attention-sink pattern. | Capacity, see section 10 |

## 3. Principle: recompute instead of storing

A kernel is memory-bound when it does fewer FLOPs per byte moved than the GPU's ridge point:

| GPU | bf16 dense TFLOPS (fp32 accumulate) | Bandwidth | Ridge (FLOP/byte) | L2 |
|---|---|---|---|---|
| RTX A5000 | 111 | 768 GB/s | 145 | 6 MB |
| RTX 4090 | 165 | 1,008 GB/s | 164 | 72 MB |
| A100 SXM 80GB | 312 | ~2.0 TB/s | ~153 | 40 MB |
| H100 SXM | 989 | 3.35 TB/s | ~295 | 50 MB |

(RTX 4090 figures from NVIDIA's Ada whitepaper, Appendix A. The others are the usual published specs.)

- **The rule:** recomputing a value is cheaper than storing it and reading it back whenever the recompute costs fewer
  FLOPs than ridge x the bytes saved.
- **Example:** a bf16 score `ok` costs 4 bytes (written once, read once) and ~64 to 200 tensor-core FLOPs to rebuild
  from `q` and the chunk-start state, below the 145 x 4 = 580 FLOP break-even. So the scores, the slot softmax and the
  normalized weights should never leave the chip: compute them in the forward, recompute them in the backward.
- **Literature:** this is the argument of FlashAttention (tiling plus recompute in the backward pass, arXiv
  2205.14135; FlashAttention-2, 2307.08691) and of Mamba's selective scan (fuse, keep the state in on-chip memory,
  recompute in the backward, 2312.00752). Cut Cross-Entropy applies it to the loss (2411.09009).
- **What should be stored:** only what is expensive to rebuild relative to its size. That is the chunk-start states,
  and even those can be recomputed by a cheap state pass when memory is tight.

## 4. Optimizations, ranked

### 4.1 Measure first (one cheap pod session)

Before writing kernels, profile the 6-layer configuration on the GPU you will use:

- `bench_lru_banks.py --profile`, which needs a small change: its `--set` cannot pass the `hops` list.
- An Nsight Systems timeline (bubbles and syncs).
- If permitted, Nsight Compute on the top 5 kernels: achieved bandwidth and FLOP/s, stall reasons. RunPod containers
  may not expose the GPU performance counters (the error is `ERR_NVGPUCTRPERM`); check that in the first minutes,
  before relying on it.

That confirms F1 to F10 with numbers and sets the baseline every later change is measured against.

### 4.2 One fused bank kernel ("FlashBank"): the main win

Per layer, two kernels (forward) and their mirrors (backward), written for this architecture only:

1. **State pass**, once per layer and shared by both hops (fixes F4).
   - Inputs: keys `k` and values `v` (bf16, 16 x 32 per token), compact routing (2 slot indices as uint8 + one weight
     per head; fixes F5), the reset bits, and the constant decays `loga`.
   - Per (row, head) it walks chunks of 64 tokens, carrying three states: the key state `S_K` (32 x 128), the value
     state `S_V` (128 x 32), and the mass `S_Z` (128). `S_Z` is the key state with a constant key, so `z` is no longer
     a separate scan or tensor (fixes F6).
   - Chunk-start decays come from a per-slot table in shared memory: `a_m^n` for n in 1..64. There is no `g` tensor
     (fixes F1) and no decay gradient (fixes F2).
   - Output: the chunk-start states, stored in bf16 or recomputed in the backward (memory knob).
2. **Read**, once per hop, one program per (chunk, head).
   - Load `q` (64 x 32) and the chunk-start states.
   - Expand the compact routing into a dense 64 x 128 tile in shared memory, so tensor cores still do the dense math.
   - Scores = `(q S_K) * a^(pos+1)` + the intra-chunk term (`q k^T` masked causal and at resets, times the
     decay-scaled routing). The mass `z` is computed the same way.
   - The softmax over 128 slots in registers (64 x 128 fp32 = 32 KB per program): `w = p / z`.
   - Value read: `(w * a^(pos+1)) S_V` + intra-chunk `((w a^t)(R a^-i)^T) V`.
   - Only `o` (64 x 32) is written (fixes F3).
3. **Backward:** recompute scores, mass, `p` and `w` per chunk from `q` and the states. Do the softmax backward in
   registers. Gradients of `q`, `k`, `v` and of the routing weight at the 2 chosen slots only. A reverse state pass
   carries `dS_K`, `dS_V`, `dS_Z`.

Numerics:
- With fixed decays, the factorized form `a^t * a^-i` is exact up to rounding as long as `a^-i` stays in range. The
  project's rule (`_safe_chunk`: below e^30 for the fastest decay) still applies. With the 1-token half-life slot,
  `a^-63` ~ e^44 is still inside fp32's range, but sub-chunks of 32 for the fastest slots keep the margin.
- The float64 reference (`bank_memory.py`) and its tests are the oracle.

Expected effect (estimate):
- **Forward memory traffic per read:** from ~2 GB (slot-space tensors through memory) to ~0.3 GB (q, k, v, routing,
  states, o).
- **Math:** ~0.4 MFLOP per token per read on tensor cores, plus the recompute in the backward.
- **Time:** the bank part from ~270 ms per step (4-layer measurement) to an estimated 40 to 70 ms.
- **Memory:** per-layer activations from ~90 KB to ~35 to 45 KB per token. Add recomputing the FFN hidden layer (as
  `gated-deltanet` does with its memory budget) and it reaches ~25 KB.

Literature:
- Gated Slot Attention computes the same read as two GLA passes joined by a softmax; that is exactly the fla
  implementation used now (arXiv 2409.07146).
- Fusing those passes is the FlashAttention step.
- Tiled Flash Linear Attention (2503.14376) shows the next lever: a second level of tiling inside the chunk lets the
  chunk grow, which divides the number of stored states (the state-storage knob above).
- ThunderKittens (2410.20399) reports 14x over the strongest baseline on linear attention with tile primitives: the
  hand-written fla-style kernels are often far from peak.

### 4.3 Fused addressing and routing

One kernel per layer from the LRU state and `x`:
- the key from the state before the token, the query, the value;
- the 128 router logits per head in registers (32 x 128 per head), the top 2, and `sigmoid(l1 - l2)`;
- the compact routing written out.

Backward: the logit gradient is non-zero at 2 slots, so `dk = d(l1 - l2) (router[:, i1] - router[:, i2])`, a
gather. This removes the dense fp32 logits, the dense routing tensor and their gradients (F5).

### 4.4 Fused loss (Cut Cross-Entropy style)

Two passes over vocabulary tiles of 128 x 128, the logits only in registers:
1. Log-sum-exp per row.
2. Recompute the logits, take the softmax, accumulate `dh = p E` and `dE += p^T h`, subtract the target column.

In bf16 with fp32 accumulation, ~1.1 TFLOP and no logits in memory: an estimated 10 to 15 ms instead of ~45 ms.
Cut Cross-Entropy (arXiv 2411.09009) adds gradient filtering: skip tiles whose probabilities are all below bf16
precision, which removes most of the backward work at a 32k vocabulary. Liger's fused linear cross-entropy
(2410.10989) is the scheme in use now (3 matmuls, chunked logits in memory).

### 4.5 Scans that stay parallel at long rows

- Rewrite the LRU scan as a two-level chunked scan (F9): each chunk's local scan in parallel, then a scan over the
  chunk carries, then a fix-up pass. Parallelism then grows with B x T instead of B.
- `rsm_scan._segmented` in this repo, Martin & Cundy (arXiv 1709.04057) and the accelerated-scan library (warp
  shuffles plus shared memory per block) are the references.
- The mass scan disappears into the state pass of 4.2.
- Accept initial states in all kernels: that enables truncated BPTT (F11) through the existing stream data path
  (`stream_chunks`).

### 4.6 Remove host syncs, make shapes static, capture the step

- **CPU-side metadata:** compute `n_targets`, the reset bits, valid rows and document starts in the data loader from
  the numpy batch. Copy them with the batch, from pinned memory, without blocking.
- **Static shapes:** with resets handled inside the kernels (as the LRU scan already does) there are no cu_seqlens and
  every shape is fixed by (B, T).
- **One CUDA graph per step:** forward, loss, backward, clipping and AdamW in a single launch. DeepSeek-V4 states the
  motivation: as kernels get faster, CPU orchestration caps throughput. They cut per-call host overhead from tens to
  hundreds of µs to under 1 µs with generated host launchers (DeepSeek-V4 report, section 3.2, arXiv 2606.19348).

### 4.7 Optimizer

- **Fused AdamW:** one kernel over a flat parameter buffer, with the clip factor read from device memory. Small gain.
- **Muon for the weight matrices:** a larger, cheap win per token rather than per second.
  - DeepSeek-V4 switched most modules to Muon "due to its faster convergence and improved training stability". It
    keeps AdamW for the embedding, the output head, norms, biases and gates (section 2.4).
  - Their recipe: Nesterov momentum, 10 Newton-Schulz iterations in two coefficient stages, the update RMS rescaled so
    the AdamW hyper-parameters carry over, and weight decay.
  - Moonshot reports roughly 2x compute efficiency over AdamW ("Muon is Scalable for LLM Training", arXiv 2502.16982),
    and Muon is used in the 124M modded-nanogpt speedruns.
  - For this model: W_lru, W_k, W_q, W_v, W_o, W_q2, W_o2, W1, W2 (512 x 512 to 512 x 2048), a few small matmuls per
    step. Keep AdamW for the embedding, norms, biases, `nu` / `theta`, gates, `log_tau` and probably the router.

### 4.8 FP8 (Ada / Hopper only)

DeepSeek-V3's recipe (arXiv 2412.19437, section 3.3):
- E4M3 everywhere, with online scaling: one scale per 1 x 128 activation tile and per 128 x 128 weight block.
- Partial sums promoted to fp32 every 128 elements: H800 tensor cores keep only ~14 bits in FP8 accumulation, up to
  ~2% error at K = 4096.
- Embedding, output head, gating, normalization and attention kept in bf16 / fp32. Master weights and gradients in
  fp32, AdamW moments in bf16.
- Linear-layer inputs cached in FP8 for the weight-gradient pass.
- Result: under 0.25% relative loss error against bf16, validated at large scale.

For this model:
- **Where:** the FFN, W_lru and the projections; not the recurrences, the bank read, the routing, the norms or the
  loss.
- **Hardware:** FP8 tensor cores exist on Ada (sm_89: RTX 4090 330 TFLOPS dense FP8 with fp32 accumulate, L40S, RTX
  6000 Ada) and Hopper, not on the A5000 / A6000 / A100.
- **Shape caveat:** at dim 512 to 2048 the quantization passes cost as much as the matmul saves. torchao's float8
  documentation says small shapes run slower than bf16. FP8 pays here only when the producer kernel (RMSNorm, GELU
  epilogue) writes FP8 plus its scales directly: possible in the custom pipeline, not through torchao.
- **Storage trick for any GPU:** caching linear-layer inputs in FP8 bytes for the weight-gradient pass halves that
  memory even on GPUs without FP8 tensor cores; a few shift instructions convert it back to bf16 before the bf16
  matmul.

## 5. The lowest level NVIDIA gives you

- **PTX** is the documented, stable virtual ISA. You can write it inline in CUDA C++ (`asm volatile`) and in Triton
  (`tl.inline_asm_elementwise`). Everything a fast kernel needs on Ampere and Ada is reachable in PTX:
  - `mma.sync` (bf16; e4m3 on sm_89);
  - `ldmatrix`, `cp.async`;
  - warp shuffles, `ex2.approx`, `rcp.approx`.

  Hopper's `wgmma`, TMA and `setmaxnreg` exist only as PTX, through CuTe/CUTLASS or a DSL.
- **SASS** is the real machine code. NVIDIA documents its opcode list (CUDA Binary Utilities, for disassembly) and
  ships no assembler. CuAssembler is a third-party assembler that lists sm_60 to sm_86. Its README calls the project
  early-stage. sm_89 (Ada) is not listed.
- **Measured gains from SASS-level work** come from compute-bound GEMMs:
  - maxas (Maxwell SGEMM);
  - TuringAs HGEMM (Yan et al., IPDPS 2020);
  - DeepSeek's DeepGEMM, which flips yield / reuse bits in FFMA instructions of its FP8 GEMM on Hopper. Their README
    and discussions report 10%+ in some cases; the semantics are a community guess, undocumented by NVIDIA.
  - The microbenchmark papers (Jia et al. 2018 on Volta, arXiv 1804.06826; Abdelkhalik et al. 2022 on Ampere,
    2208.11174; Luo et al. 2024 on Hopper, 2402.13499) are the map of instruction latencies if a kernel ever gets
    there.
- **Your Intel experience does not transfer here.** On the Iris Xe the assembly fp16 kernel paid off because oneDNN's
  fp16 GEMM converts to fp32 and runs at the fp32 rate (AGENTS.md: 3.7 vs 1.9 TFLOPS). On NVIDIA, cuBLAS already ran
  at ~89 of 111 TFLOPS on the A5000. There is no library deficiency to beat; the losses are in the bank kernels' memory
  traffic, which no instruction encoding fixes.

Recommendation: write the custom kernels in CUDA C++ with inline PTX where needed. Look at SASS only through
`cuobjdump -sass` and Nsight Compute's source view, to check register spills and the instruction mix.

## 6. A from-scratch C/C++ trainer specialized to this model

C++ over C: templates for tile sizes and dtypes, RAII for device buffers, the same speed. CUDA C++ is the natural
language for the kernels.

### What it removes, and what that is worth

Framework overhead in the current path:
- Keras and Python per step;
- torch dispatch;
- fla's Python wrappers and autotune lookups;
- autograd bookkeeping;
- the host syncs.

With the GPU ~500 ms busy per step this is likely 10 to 25% today. The share grows as the kernels get faster.

Reference point: llm.c (Karpathy, GPT-2 in C/CUDA) reports being ~7% faster than PyTorch nightly with torch.compile
on GPT-2 124M. It got there with fused kernels, a single parameter buffer, one AdamW kernel and cuBLASLt, against a
framework that already had FlashAttention. Our current framework path has much more overhead than that baseline, so
the margin should be larger.

The decisive gains (sections 4.2 to 4.5) come from the kernels. The same kernels can also run inside torch, and staging
the work that way removes most of the risk.

### What to hand-write and what to keep

| Component | From scratch | Note |
|---|---|---|
| Data stream | yes | memory-mapped uint16 corpus; random windows or streams; loss masks from document boundaries and `loss_spans`; reset bits, valid rows, counts; a CPU thread fills pinned double buffers; async copies on a copy stream |
| Parameters, gradients | yes | one flat fp32 master buffer, bf16 working copies written by the optimizer kernel, one flat gradient buffer |
| Memory | yes | static shapes, so one arena with precomputed offsets; a per-tensor recompute-or-store table (DeepSeek-V4 section 3.4.4 did this at tensor granularity on top of autograd) |
| Embedding gather, scatter-add backward | yes | combined with the readout's dE (tied table) |
| RMSNorm forward / backward | yes | writes bf16 (later FP8 plus scales) for the next GEMM |
| Dense GEMMs | **keep cuBLASLt** | it is NVIDIA's GEMM, not framework overhead; its epilogues fuse bias + GELU (forward, keeping the pre-activation) and dGELU + bias gradient (backward), and the residual add by accumulating into C; FP8 with scales on sm_89 / sm_90; check that its GELU is the tanh form the model uses |
| LRU scan (chunked) forward / backward | yes | port of the Triton kernel plus 4.5 |
| Addressing + routing forward / backward | yes | 4.3 |
| FlashBank state pass + read, forward / backward | yes | 4.2 |
| Loss forward + backward | yes | 4.4 |
| Global norm + clip + AdamW / Muon | yes | Muon's Newton-Schulz with cuBLASLt matmuls |
| Step | yes | the whole step captured once as a CUDA graph |
| Checkpoints | yes | a flat binary file plus a converter to `.keras`, so evaluation, `calc_x_eval.py` and the web chat keep working |

Writing your own GEMM to replace cuBLASLt costs weeks for at most a few percent at these shapes. Simon Boehm's CUDA
matmul write-up shows how long it takes just to reach cuBLAS. DeepSeek-V4 writes most of its kernels in a DSL
(TileLang) and still relies on libraries where they are fast.

### Correctness

Every hand-derived backward needs a reference. The repo has it:
- the float64 torch reference (`bank_memory.py`, `lru_banks.py`'s torch path);
- the tests in `tests/test_arch_lru_banks.py` and `tests/test_bank_memory.py`.

Expose each C++ kernel through a thin C ABI (ctypes, or a torch extension for tests only) and check values and
gradients against the reference on small shapes with resets, plus finite-difference checks on the end-to-end loss.
Without this, a from-scratch trainer will have silent gradient bugs.

### Cost and the main risk

- **Cost:** about 4 to 8 weeks of focused work for a correct, fast trainer (estimate).
- **Ongoing cost:** every architecture change (your gates, section 10) means rewriting forward and backward by hand.
  DeepSeek-V4 names this trade-off explicitly. Manual layer implementations give fine-grained control but "loses the
  convenience of the automatic differentiation framework, substantially increasing development complexity". They kept
  autograd with tensor-level checkpoint annotations instead (section 3.4.4).
- **For one small fixed architecture the trade is different:** llm.c shows it works. The condition is that the
  architecture is settled first.

Recommended staging:

| Stage | Work | Where |
|---|---|---|
| 0 | Settle the architecture changes (gates, section 10) in torch on the fast synthetic loop and short local runs | local, no GPU spend |
| 1 | Measure (4.1) | 1 pod session |
| 2 | Write the CUDA C++ kernels (FlashBank, routing, LRU, loss) as standalone `.cu` files with a C ABI; test against the reference; plug them into the current trainer | pod; most of the speedup, low risk |
| 3 | The C++ trainer: data stream, arena, optimizer, CUDA graph; reuses stage 2 unchanged | pod; the last 10 to 25%, and the specialization you want |
| 4 | FP8 FFN on sm_89, fused quantization | only if training moves to Ada / Hopper |

## 7. GPU for this work

- **Ampere and Ada share a programming model.** Kernels written for Ampere (`mma.sync`, `cp.async`, `ldmatrix`) run on
  Ada unchanged.
- **Ada (sm_89) adds:**
  - FP8 tensor cores;
  - a much larger L2 on the RTX 4090 (72 MB vs 6 MB on the A5000 / A6000), so a layer's 33 MB bf16 stream can stay in
    L2 between consecutive kernels;
  - 1.5x the bf16 rate of the A5000.
- **Hopper** reaches its peak only with `wgmma` / TMA kernels, a separate code path. For a 27M-parameter model its
  price is not justified.

Choice, from the availability seen this week:
- **RTX 4090** (24 GB, $0.34 per hour community when in stock) for development and training.
- **L40 / L40S / RTX 6000 Ada** (48 GB, $0.69 to $0.79 per hour) when bigger batches are needed before the memory
  savings land.

## 8. Batch size, noise, long rows

### Noise

- **Gradient accumulation works today.** Set `data.micro_batch_size 64` with `data.batch_size 256` (or 512). The train
  step already sums the micro-batches' gradients and divides by all their targets (`_dense.py:129-188`).
  Updates become less noisy at the same tokens per second (one update per ~2 s at 256).
- **Literature:**
  - Smith et al. (arXiv 1711.00489): raising the batch has the effect of decaying the learning rate on the gradient
    noise.
  - McCandlish et al. (1812.06162): the gradient noise scale grows as the loss falls, so a batch ramp mid-training is
    principled.
  - Zhang et al. (2410.21676): the critical batch size grows with the amount of data trained on, not the model size,
    so a 27M model on ~4B tokens likely has a critical batch well above 32k tokens late in training.
- **Measure instead of guessing:** McCandlish's estimator needs only the squared norms of the micro-batch gradients and
  of the accumulated gradient, both available inside the accumulation loop at almost no cost. Log it per epoch and
  raise the batch when it grows.
- **The decay phase also cuts noise:** a warmup-stable-decay schedule's final decay (`training.lr_decay_*`, already
  supported) does that.

### Long rows

- Per-token cost is independent of the row length here (a recurrence, no quadratic attention). 16 x 2048 and
  2 x 16,384 cost the same as 64 x 512 in math.
- **What limits long rows today:**
  - activation memory per token (fixed by 4.2 and recompute);
  - the scans' parallelism at small B (4.5);
  - no carried state for contexts beyond one row (F11, 4.5).
- Sequence-length warmup (Li et al., arXiv 2108.06084) is an optional schedule once long rows are cheap.

## 9. DeepSeek: what to adapt

| Paper | Trick | For this model |
|---|---|---|
| V3 (2412.19437) | FP8 GEMMs, fine-grained scaling, fp32 promotion every 128, E4M3 everywhere | Yes on Ada / Hopper, for the FFN and projections, with quantization fused into producers (4.8) |
| V3 | FP8 caching of linear inputs for the weight gradient | Yes: halves that memory even on GPUs without FP8 tensor cores (shift-only conversion) |
| V3 | Recompute all RMSNorms in the backward | Yes, part of the recompute table |
| V3 | BF16 AdamW moments, parameter EMA on the CPU | Moments: irrelevant at 27M (216 MB of state). EMA: cheap, may help the noisy mid-training evaluations |
| V3 | Auxiliary-loss-free load balancing (bias on the routing logits, used only for the choice) | Yes, for the bank router: the synthetic runs showed dead slots and collapsing slot use (AGENTS.md); `gated-deltanet` already uses the trick for its experts |
| V3 | Multi-token prediction | Probably not at 27M: Gloeckle et al. (2404.19737) find the gain grows with model size |
| V3.2 (2512.02556) | Sparse attention with a lightning indexer: ReLU scores (chosen for throughput), FP8, top-k; a dense warm-up that trains only the indexer, then sparse training | The two-stage schedule and the ReLU scores carry over to the skip gates (section 10) |
| NSA (2502.11089) | Sparsity aligned to hardware blocks, gated branches | The rule for any skipping: per 16-row tile or compacted, not per token |
| Engram (2601.07372) | Hashed N-gram embedding lookup, gated into the stream; O(1) memory as a second sparsity axis; U-shaped split between computation and memory | Fits your observation that many tokens need ≤ 8 tokens of context: a lookup table can carry local patterns cheaply and leave the banks / FFN for long-range work. An experiment, not a pipeline fix |
| mHC (2512.24880) / V4 | Wider residual stream with doubly-stochastic mixing | Low priority at 27M; extra memory traffic |
| V4 (2606.19348) | Muon; a DSL (TileLang) with generated host launchers; tensor-level recompute; deterministic kernels | Muon: yes (4.7). The DSL route is the alternative to hand-written C++. Deterministic kernels: nice to have |

## 10. Your architecture proposal: read / write gates, no null slot, skip zero reads

### Analysis

1. **The `<bos>` null slot is real and predictable from the code (F12).** Every document's `<bos>` writes the same key
   (`normalize(k_bias)`) and value into the same slots, and mass normalization keeps that average fixed.
   - A head that needs "no memory" for a token parks its read there. It wastes slots, the read's probability mass, and
     the head's output (W_o must learn to cancel that constant).
   - It is the attention-sink pattern. Gated Attention (Qiu et al., NeurIPS 2025, arXiv 2505.06708) shows that a
     per-head, query-dependent gate on the attention output removes the sink (83% of attention on the first token
     down to 4% in their example layer), improves training stability and quality, and is now used in Qwen3-Next.
   - Related: StreamingLLM's sinks (2309.17453) and vision transformers' register tokens (2309.16588).
   - Measure the effect cheaply before changing anything: the share of read weight landing on `<bos>` positions.
     `inspect_banks` / the web chat's lru-banks Recall view already computes the exact read weight `c_j` per past
     position.
2. **Read gate:** `h += g_t * W_o read(q_t)`, one gate per head computed from the query's input: the LRU state after
   the update for hop 1, the stream for hop 2. With it, the `<bos>` write can be dropped (reset positions write
   nothing), freeing those slots.
3. **Write gate:** under mass normalization a soft write gate matters only relative to other tokens (AGENTS.md). A
   hard 0 removes a token from the banks.
   - That is what the needle tests needed: "the filler writes ... seem to bury the rest".
   - With top-2 routing it is `r_t <- w_t r_t`; slots no token wrote stay masked out of the read (`z = 0`), as today.
4. **Binary gates need an estimator and a sparsity pressure.** This project's own runs show that sigmoid gates left
   alone stay open:
   - the LRU input gate in the needle test: 0.894 on filler vs 0.905 on keys;
   - the `rope-state-memory` write gate "saturated open".

   Options:
   - **Straight-through estimator:** hard 0/1 forward, sigmoid derivative backward (Bengio et al., arXiv 1308.3432),
     plus a budget loss on the share of open gates. Skip RNN uses this for skipping state updates (Campos et al.,
     1708.06834).
   - **Hard-concrete gates with an L0 penalty** (Louizos et al., 1712.01312): exact zeros, a smooth training signal.
   - **ReLU gate plus an L1 target:** `g = relu(w . s + b)` gives exact zeros and an ordinary gradient where open (the
     "ReLU" analogy you made). Precedents: rectified linear attention (Zhang et al., 2104.07012), activation sparsity
     in "ReLU Strikes Back" (2310.04564), and V3.2's ReLU indexer scores. Risk: gates that die and never reopen;
     start with a positive bias.
   - **Capacity top-k routing (Mixture-of-Depths**, Raposo et al., 2404.02258): a fixed number of tokens per row take
     the block, so shapes are static. Their best variant matched an isoFLOP baseline with up to 50% fewer FLOPs per
     forward pass. Caveat: top-k over a row looks at the future; at inference they use a causal predictor.
   - **Training schedule (V3.2):** first dense with soft gates, then harden, then run sparse.
5. **Where skipping saves time.**
   - The reads are ~14% of the step's math (1.16 of 8.17 TFLOP); the FFNs are 2/3.
   - Skipping only the read saves at most ~14%. A gate that skips the whole hop (read + W_o + FFN) saves much more:
     hop 2 is ~43% of a two-hop layer's dense math.
   - Gate per (layer, hop), computed from the query side, as you suggest: "the query could be the gate".
   - On a GPU, skipping saves time only if the skipped tokens are removed from the dense tiles. Per chunk, a prefix
     sum on the device gives the open tokens' positions; the kernels process `ceil(open / 16)` tiles of 16 rows
     (tensor-core granularity), with a capacity bound so the launch grid is static. This is the NSA and MoD lesson;
     scattered per-token zeros inside full tiles save nothing.
   - The LRU still updates its state at every token (it is cheap). Keys and writes are unaffected by read skipping.
6. **Removing the LRU input gate.** That gate is the LRU's selective decay (g = 0 freezes the state, like Mamba's
   step size), not a memory gate. Removing it makes the LRU time-invariant between resets.
   - Its value is measurable with `ad hoc scripts/lru_banks_layer_use.py` (it reports the gate mean per layer). Add
     the loss with the gate fixed at its mean.
   - Test the removal rather than assume it. The read / write gates do a different job (what enters and leaves the
     banks), so they do not replace the LRU gate by construction.
7. **The "tokens predictable from ≤ 8 tokens" hypothesis** sets an upper bound on skippable reads. Measure it first:
   per token, the loss change when the reads are silenced (extend `lru_banks_layer_use.py` from whole layers to
   per-token read ablation). If most tokens lose < ε, hard read gates will have room. An Engram-style N-gram table
   (section 9) is the complementary idea for those tokens.

### Experiment order

Run all of this in torch, on the fast synthetic loop first (`bank_recall_demo.py`, `niah_bank.py`, `twohop_circuits.py`:
60 to 90 s per run on the CPU), then in a short real run. Port to the CUDA kernels only once the design is settled
(stage 0 of section 6).

1. **Measure:** the `<bos>` read share, the per-token read ablation, the LRU gate stats.
2. **Soft per-head read gate** (Qiu et al.) + no `<bos>` write. This should fix the sink without skipping.
3. **Write gate** (hard-concrete or ReLU + target sparsity), checked on the needle task where writes were burying
   needles.
4. **Hard hop gate** (MoD capacity or straight-through estimator + budget), then the compacted kernels.

## 11. Byte Latent Transformer and H-Net

- **BLT** (Pagnoni et al., arXiv 2412.09871, ACL 2025): bytes grouped into patches where a small byte model's
  next-byte entropy is high. The large latent transformer runs once per patch; small local models run per byte. In a
  FLOP-controlled study up to 8B parameters and 4T bytes, it scales better than tokenized models at fixed inference
  cost, by growing patch and model size together.
- **H-Net** (Hwang, Wang, Gu, arXiv 2507.07955): learns the chunking end-to-end with SSM encoders / decoders. It is
  closer to this recurrent model. A byte-level H-Net with one stage matches a BPE transformer at matched compute above
  1B parameters, with ~4.5 to 5 bytes per learned chunk.
- **For a 27M model:**
  - The 32k readout would disappear (it is ~10% of the math now).
  - Positions would multiply by ~4.4, since tokens average 4.38 characters.
  - The claimed savings are shown at ≥ 1B.

  This is an architecture research direction, not a pipeline optimization. The part that transfers now is the idea
  under both: spend computation where the next symbol is uncertain. That is the hop / read gate of section 10, with
  the gate driven by how predictable the token is.

## 12. Dead ends (do not spend money here)

| Idea | Why not | Reference |
|---|---|---|
| Hand-editing SASS | No official assembler (CuAssembler is third-party, lists up to sm_86, not Ada); gains of 5 to 15% shown only on compute-bound GEMMs; our bottleneck is memory traffic | CuAssembler README; DeepGEMM README; Jia et al. 1804.06826; Abdelkhalik et al. 2208.11174 |
| Own GEMMs instead of cuBLASLt | cuBLAS already at ~80% of peak on the A5000 (AGENTS.md); weeks for ≤ a few % | llm.c (keeps cuBLASLt) |
| FP8 through a library at dim 512 | Quantization overhead exceeds the gain on small shapes; no FP8 on Ampere | torchao float8 docs; DeepSeek-V3 3.3 (fused quantization) |
| FP8 for the recurrences, bank read, routing, norms, loss | Precision-sensitive; DeepSeek keeps the equivalents in bf16 / fp32 | DeepSeek-V3 3.3 |
| Using the top-2 sparsity for fewer FLOPs (scatter, atomics) | Dense 16 x 16 tensor-core tiles beat scattered updates at 128 slots; use the sparsity for storage only | NSA 2502.11089 (block alignment) |
| Per-token skipping without compaction or block alignment | Zeros inside full tiles save no time | MoD 2404.02258; NSA |
| Sigmoid gates without sparsity pressure | They stay open (this project: LRU input gate 0.89 on filler; rope-state-memory write gate saturated) | AGENTS.md; Skip RNN 1708.06834 (budget loss) |
| Token-by-token (recurrent) bank kernel for training | Parallelism is only B x heads: too few programs at long rows; fine for generation | (design analysis) |
| Log-space scans | `logcumsumexp` was 100x a `cumsum` on the XPU; linear-space chunked scans are the standard | AGENTS.md (min-gru); Mamba 2312.00752 |
| Keeping per-token decay tensors (`g`) for fixed decays | 268 MB read by every bank kernel; the decays have a closed form | F1 |
| Full associative scan over T for the bank state | 8,320 state values per head per token: memory explodes; the chunked form exists for this | Mamba-2 2405.21060; GLA 2312.06635 |
| FlashRNN-style register-resident recurrent matrices | The LRU is diagonal: there is no recurrent matmul to keep in registers | FlashRNN 2412.07752 |
| Multi-token prediction at 27M | Gains reported to grow with model size | Gloeckle et al. 2404.19737 |
| H100 for this model | Peak needs `wgmma` / TMA kernels (a second code path); price not justified at 27M | FlashAttention-3 2407.08608 (Hopper-specific design) |
| Larger micro-batches without recompute | Memory-bound now (128 x 512 did not fit for the 4-layer model, AGENTS.md); use accumulation until 4.2 lands | AGENTS.md |

## 13. References

Kernels and IO-aware algorithms
- Dao et al. 2022, FlashAttention, arXiv 2205.14135. Dao 2023, FlashAttention-2, 2307.08691. Shah et al. 2024,
  FlashAttention-3, 2407.08608.
- Gu & Dao 2023, Mamba, 2312.00752. Dao & Gu 2024, Mamba-2 / SSD, 2405.21060.
- Yang et al. 2023, Gated Linear Attention (FlashLinearAttention), 2312.06635.
- Zhang et al. 2024, Gated Slot Attention, 2409.07146 (the kernels used now: fla `chunk_gsa_*`, `chunk_gla_*`).
- Beck et al. 2025, Tiled Flash Linear Attention, 2503.14376.
- Pöppel et al. 2024, FlashRNN, 2412.07752.
- Spector et al. 2024, ThunderKittens, 2410.20399.
- Martin & Cundy 2018, Parallelizing linear recurrent networks over sequence length, 1709.04057.
- Kyrylov, accelerated-scan (github.com/proger/accelerated-scan).
- Orvieto et al. 2023, LRU, 2303.06349.

Loss
- Wijmans et al. 2024, Cut Your Losses in Large-Vocabulary Language Models (Cut Cross-Entropy), 2411.09009.
- Hsu et al. 2024, Liger Kernel, 2410.10989.

Memory
- Chen et al. 2016, Training Deep Nets with Sublinear Memory Cost, 1604.06174.
- Korthikanti et al. 2022, Reducing Activation Recomputation in Large Transformer Models, 2205.05198.

Hardware and low level
- NVIDIA Ada GPU Architecture whitepaper, Appendix A (RTX 4090: 165.2 bf16 TFLOPS with fp32 accumulate, 330.3 FP8,
  72 MB L2, 1,008 GB/s).
- Jia et al. 2018, Volta microbenchmarks, 1804.06826. Abdelkhalik et al. 2022, Ampere, 2208.11174. Luo et al. 2024,
  Hopper, 2402.13499.
- CuAssembler (github.com/cloudcores/CuAssembler). DeepGEMM (github.com/deepseek-ai/DeepGEMM).
- Karpathy, llm.c (github.com/karpathy/llm.c).

DeepSeek
- DeepSeek-V3 technical report, 2412.19437. DeepSeek-V3.2, 2512.02556. Native Sparse Attention, 2502.11089.
- mHC, 2512.24880. Engram (Conditional Memory via Scalable Lookup), 2601.07372. DeepSeek-V4, 2606.19348.

Optimizers, batch size
- Liu et al. 2025, Muon is Scalable for LLM Training, 2502.16982.
- Smith et al. 2017, Don't Decay the Learning Rate, Increase the Batch Size, 1711.00489.
- McCandlish et al. 2018, An Empirical Model of Large-Batch Training, 1812.06162.
- Zhang et al. 2024, How Does Critical Batch Size Scale in Pre-training?, 2410.21676.
- Li et al. 2021, sequence length warmup, 2108.06084.

Gates and conditional computation
- Qiu et al. 2025, Gated Attention for LLMs (attention-sink-free), 2505.06708.
- Xiao et al. 2023, attention sinks (StreamingLLM), 2309.17453. Darcet et al. 2023, Vision Transformers Need
  Registers, 2309.16588.
- Raposo et al. 2024, Mixture-of-Depths, 2404.02258.
- Campos et al. 2017, Skip RNN, 1708.06834.
- Bengio et al. 2013, straight-through estimator, 1308.3432. Louizos et al. 2017, L0 / hard-concrete, 1712.01312.
  Jang et al. 2016, Gumbel-softmax, 1611.01144.
- Zhang et al. 2021, rectified linear attention, 2104.07012. Mirzadeh et al. 2023, ReLU Strikes Back, 2310.04564.
- Gloeckle et al. 2024, multi-token prediction, 2404.19737.

Bytes
- Pagnoni et al. 2024, Byte Latent Transformer, 2412.09871.
- Hwang, Wang, Gu 2025, H-Net (dynamic chunking), 2507.07955.
