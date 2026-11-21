# Research log: an efficient training pipeline for `lru-banks`

This file keeps the research on track. Read it before proposing or starting anything; add an entry after every
experiment. Its purpose:
- not to redo what already failed;
- to know the current numbers to beat;
- to cut the search to the paths that are still open.

The analysis behind the queue is in `studies/cuda-training-pipeline-review.md` (where time and memory go, the plan,
literature, dead ends with references).

## Rules

- **One row per experiment** in the log below:
  - ID, date, commit, hardware;
  - **tried** (one line);
  - **result** (numbers, and the command that measured them);
  - **status**: `planned`, `running`, `implemented`, `abandoned`, `blocked` or `superseded by X`;
  - **because** (one line).
- **Commit first:** the commit is the one that holds the code that produced the result, so commit before measuring.
  - `cuda@<sha>` is a commit of the C++/CUDA pipeline repository, https://github.com/SamuelHaidu/lru-banks-cuda
    (public).
  - A plain sha is a commit of this repository.
  - Rows from before this log say `cc6e23b` (the initial import: the code exists there, the measurement predates
    the history) or `uncommitted`.
- **IDs** by area: `P` pipeline / performance, `A` architecture, `T` training recipe and results, `R` reviews. Never
  reuse or delete an ID; supersede it.
- **Speed numbers** name the script and arguments, the GPU, the shapes, the warm-up and the timed steps. Microbenchmark
  a component first; a change reaches the full step only after it beats the component baseline.
- **Abandoned ideas stay abandoned** unless the "because" no longer holds (new GPU, new kernel, new shapes). Say which
  in the new row.
- **When a result beats a line of "Current standing"**, update that line and point it at the row.

## Current standing (the numbers to beat)

| Metric | Value | Config | GPU | Row |
|---|---|---|---|---|
| **C++ pipeline, all gates open (B1)** | **95.7 ms per step, 342.5k tokens/s, 10.5 updates/s** (disk reads, CUDA graph), 5.43 GiB | 4 layers x 2 hops, one FFN per layer, binary per-head gates, 64 x 512, loss at every position | RTX 4090 | P45 |
| C++ pipeline, reads closed (B2) | 67.6 ms, 485k tokens/s | same, read gates -20 | RTX 4090 | P45 |
| C++ pipeline, gates trained toward 30% open | 77.9 ms, 420.6k tokens/s | budget 1, target 0.3, after 300 steps | RTX 4090 | P44 |
| C++ pipeline, long rows | 8 x 4096: 109.0 ms; 2 x 16384: 117.6 ms (LRU scan 11.8 ms there) | same, 32768 tokens per step | RTX 4090 | P43, P45 |
| C++ pipeline, GPU time (B1) | GEMMs 33.3 ms (bf16 GEMMs at 150-153 TFLOPS in isolation, 93% of peak), bank walks 42.3, loss softmax 5.7, norms and elementwise 5.1, addressing 7.5, LRU 1.5, optimizer 1.5 | 64 x 512 | RTX 4090 | P45 |
| C++ pipeline, first version | 121.5 ms, 269.6k tokens/s, 7.2 GiB | same | RTX 4090 | P31 |
| Step time | 566 ms | 4 layers, 64 x 512 | RTX A5000 | P03 |
| Throughput | 57.9k tokens/s (batch 96: 58.2k) | 4 layers, 64 x 512 | RTX A5000 | P03 |
| Peak memory | 12.1 GiB (batch 96: 18.1 GiB; 128 does not fit) | 4 layers | RTX A5000 | P03 |
| Step time, 6 layers | ~500 ms, observed during training (~2 updates/s), **not benchmarked** | 6 layers `[2,2,2,2,1,1]`, 64 x 512 | RTX A5000 | needs P20 |
| Math per step | 8.17 TFLOP (dense 6.16, bank reads 1.16, loss 0.82) | 6 layers, 64 x 512 | any | R01 |
| Utilization | ~15% of the 111 bf16 TFLOPS peak (estimate: 8.17 TFLOP at ~500 ms) | 6 layers | RTX A5000 | R01 |
| GPU-time split | fla bank kernels ~48%, tensor-core matmuls ~17%, compiled elementwise ~17%, rest ~18% | 4 layers | RTX A5000 | P03 |
| Plain bf16 GEMM | ~89 of 111 TFLOPS | GDN shapes | RTX A5000 | P05 |
| Activation memory | ~90 KB per token per layer (12.1 GiB / 32,768 tokens / 4 layers) | 4 layers | | P03, R01 |
| Targets (estimates, R01) | step ~180-250 ms on an A5000-class GPU (floor ~120 ms); ≤ 45 KB per token per layer | 6 layers | | |

## Queue (in order; microbenchmarks first)

Done on 2026-11-20 (rows in the log): P33 -> P37, P38; P24 -> P37, P38 (an exact walk instead of fla's chunked form);
P34 -> P39 (L2-sized chunks, not the fully fused kernel: P23 stays open); P35 -> P40; P36 -> P41; P22 -> P42
(abandoned at this width); long rows for the banks -> P43; P27 -> P45; skipping GEMM rows -> P44 (abandoned with
per-head gates). P21 is mostly moot (bf16 GEMMs at 93% of peak in isolation, P42).

| ID | Experiment | Measure | Go / no-go |
|---|---|---|---|
| P20 | Baseline of the 6-layer model on the target GPU (RTX 4090) | `bench_lru_banks.py` (after letting `--set` take the hops list) at 64 x 512, 128 x 512, 16 x 2048: step ms, tokens/s, GPU busy %, per-kernel split, peak memory, kernel launches per step; an Nsight Systems timeline; check whether Nsight Compute is allowed (`ERR_NVGPUCTRPERM`) | Required before everything below |
| P21 | GEMMs of the step: torch bf16 vs cuBLASLt with fused epilogues (bias + GELU forward, dGELU + bias gradient backward, residual by accumulating into C) | TFLOPS per shape: 32768 x 512 x 512, 32768 x 512 x 2048, 32768 x 2048 x 512, and the dX / dW shapes | Use the epilogues where they win |
| P23 | Loss: current chunked TF32 vs chunked bf16 vs fused Cut-Cross-Entropy-style (two passes, logits in registers), ± gradient filtering | ms and memory at N 32,768, D 128, V 32,768; loss, dh, dE vs fp32 | Adopt the fastest that matches |
| P25 | Fused addressing + routing vs `_f_addr` + `_top2` | ms, memory | |
| P26 | LRU scan: sequential (current) vs chunked two-level | 64 x 512, 16 x 2048, 2 x 16,384 | Needed for long rows: 11.8 ms at 2 x 16384 vs 1.5 ms at 64 x 512 (P45) |
| P28 | Optimizer step: Keras AdamW vs one fused AdamW kernel vs Muon (cost per step) | ms per step | Loss per token in T12 |
| P29 | Memory policy: per-tensor recompute (FFN hidden, RMSNorm, chunk states) | largest batch at 24 GB, ms per token | |
| P30 | The C++ trainer: what remains for real runs (warmup and a schedule, checkpoints, a validation loss, Muon) | loss per token, a full pass | Done in P46-P48 (Muon remains: T12) |
| A10 | Measure the read weight landing on `<bos>` positions, and the per-token loss change when the reads are silenced | `inspect_banks` on validation rows (local); extend `lru_banks_layer_use.py` to per-token read ablation | Bounds what read gates can save |
| A11 | Soft per-head read gate on the hop output (Gated Attention) + no `<bos>` write | synthetic loop (`bank_recall_demo.py`, `niah_bank.py`, `twohop_circuits.py`), then a short real run | |
| A12 | Hard write gate (hard-concrete + L0, or ReLU + L1 target) | needle task (writes were burying needles, A06) | |
| A13 | Hard hop gate skipping read + W_o + FFN (Mixture-of-Depths capacity, or straight-through + budget), then compacted kernels | loss at equal FLOPs, then step ms | |
| A14 | Remove the LRU input gate | its value first: gate mean per layer, loss with the gate fixed at its mean | Do not assume: it is the LRU's selective decay, not a memory gate |
| A15 | Auxiliary-loss-free balancing bias on the bank router (DeepSeek-V3) | slot usage entropy, dead slots, loss | |
| T10 | Gradient accumulation, batch 256 (micro-batch 64) vs 64 | loss per token, curve noise | Config only, no code |
| T11 | Log the gradient noise scale (McCandlish) from micro-batch and accumulated gradient norms | per epoch | Chooses the batch size |
| T12 | Muon (matrices) + AdamW (embedding, norms, biases, decays, gates) vs AdamW | loss per token, short run | |
| T13 | KodCode fine-tune from `lru-banks-6L-512-top2` (dataset built, T05) | agent-loop pass rate on the test split, in the sandbox | Paused thread |

## Log

### Pipeline and performance

| ID | Date | Commit | HW | Tried | Result | Status | Because |
|---|---|---|---|---|---|---|---|
| P01 | pre-log | cc6e23b | RTX A5000 | lru-banks CUDA v1: fla gated-slot-attention kernels + Triton LRU scan, everything else unfused | 752 ms per step at 64 x 512, 43.6k tokens/s, 16.6 GiB; 44% of GPU time in unfused elementwise (fp32 adds, bf16 casts, routing softmax + topk) | superseded by P02 | |
| P02 | pre-log | cc6e23b | RTX A5000 | `torch.compile` for everything between kernels, explicit bf16 matmuls; top-2 routing as a softmax over the 2 largest logits | 582 ms, 56.3k tokens/s, 11.3 GiB | implemented | -23% step time |
| P03 | pre-log | cc6e23b | RTX A5000 | Keep `w` (the normalized read weights) for the backward instead of recomputing | 566 ms, 57.9k tokens/s, 12.1 GiB; batch 96: 58.2k tokens/s, 18.1 GiB; 128 out of memory | implemented, **current baseline** | -3% for +0.8 GiB |
| P04 | pre-log | cc6e23b | CUDA | The torch chunked path in bf16 | wrong results: the chunked forms multiply factors up to e^30 and their inverses | abandoned (rule) | generation runs that path in fp32 |
| P05 | pre-log | cc6e23b | RTX A5000 | `gated-deltanet` pipeline (the model for this work): compiled elementwise, TF32 for the remaining fp32 matmuls, one MoE autograd function, Triton conv, grouped GEMMs with GELU in the epilogue | 628 -> 274-279 ms per step (2.3x), ~178k tokens/s, ~17 TFLOPS, GPU busy 91%; plain bf16 matmul ~89 of 111 TFLOPS | implemented (that arch) | custom fusion paid 2.3x on a similar stack |
| P06 | pre-log | cc6e23b | Iris Xe | fp16 assembly GEMM (`hgemm`, iga64) for the `rope-state-memory` MLP | 2.0-3.2 TFLOPS vs 1.7 (torch fp32); step 8.6 vs 10.1 s | implemented (XPU only) | oneDNN's fp16 GEMM converts to fp32 there. Not transferable to NVIDIA: cuBLAS already reaches ~80% of peak (P05) |
| P07 | pre-log | cc6e23b | Iris Xe | Log-space scan with `logcumsumexp` (minGRU paper) | 15x slower than the chunked linear-space scan | abandoned | `logcumsumexp` ~100x a `cumsum` on that GPU |
| P08 | pre-log | cc6e23b | CPU | Chunked bank recurrence: factorized decays `a^t a^-i`, all chunks batched, only chunk-start states carried | 1 layer x 64 slots 177 -> 89 ms, 2 layers 269 -> 106 ms | implemented (rule) | |
| P09 | pre-log | cc6e23b | CPU | Materialized (slots x C x C) decay tensors | per-op overhead per chunk, memory flood when batched | abandoned | P08 replaces it |
| P10 | pre-log | cc6e23b | RTX A5000 | First-order scan as torch `associative_scan` compiled to Triton (`min-gru-cuda`) | 0.14 ms vs 1.7 ms (chunked) at 32 x 512 x 256 | implemented (min-gru-cuda) | candidate for P26 |
| P11 | pre-log | cc6e23b | Iris Xe | Segmented scan for 1-2 long rows (`rsm_scan._segmented`): segments scanned as rows, the carried-in state added by one matmul | exact; fixes the latency bound at 16k-token rows | implemented (rope-state-memory) | template for P26 |
| P12 | pre-log | cc6e23b | CPU | Read-driven erase in the bank layer | 325 ms per step vs 218 (query erase) vs 165 (plain); overwrites 0.950 vs 0.981 (query erase) | abandoned | 1.5x the cost for a smaller gain |
| P13 | pre-log | cc6e23b | Iris Xe | `min-gru` per-head gates (`gate_heads`) | faster (188 -> 152 ms) but worse per step (val 1.385 vs 1.278) | not default | worse per wall-clock early on |
| P31 | 2026-11-20 | cuda@207355c | RTX 4090 | First custom C++/CUDA pipeline (`train_bench`): CUDA runtime + cuBLASLt for the dense GEMMs, every other kernel hand-written with its backward (the banks as an exact walk per (row, head), one thread per slot); fused AdamW; memory-mapped corpus read by a loader thread | 121.5 ms per step, 269.6k tokens/s at 64 x 512. GPU time: GEMMs 32 ms (~123 TFLOPS), bank kernels 44 ms (forward 13, re-walk 19, adjoint 12), loss softmax 16 ms, addressing 14 ms, norms and elementwise 11 ms, LRU 1.5 ms, optimizer 1.5 ms. Loader 0.96 ms per batch on its thread (its "wait" is backpressure: the CPU runs 3 steps ahead). Loss 10.41 -> ~8.1 in 35 steps from scratch. `train_bench --data train.npy --steps 30 --warmup 5` | implemented, **C++ baseline** | Not comparable 1:1 with P03 (other GPU; one FFN per layer instead of two; gates) |
| P32 | 2026-11-20 | cuda@207355c | RTX 4090 | Skipping the compute of closed reads (binary read gates, one per head) | read gate bias +20 / 0 / -20 (100% / ~38% / 0% open): 125.9 / 111.8 / 100.0 ms per step, bank kernels 50.2 / 37.5 / 24.2 ms | implemented | The closed-read floor (24 ms) is the walk itself: per-token decay of 128 slots x 65 values and one barrier per step (next: P33) |
| P37 | 2026-11-20 | cuda@0b21967 | RTX 4090 | Opt 1, bank walk floor: running slot averages (mass normalization cancels the decay except in the write update and the validity test), one packed 8-byte word per step, closed reads without a barrier, barrier-free adjoint with lazy per-thread decay and per-slot gradient buffers | grad check PASS (2 seeds, --bf16). B1 (all gates open) 125.9 -> 120.7 ms; B2 (reads closed) 100.0 -> 89.8 ms. Bank kernels B1 50.2 -> 44.8 ms (forward 15.3 -> 12.3, re-walk 21.4 -> 21.9, adjoint 13.5 -> 10.7), B2 24.2 -> 12.9 ms. `bash scripts/check_and_bench.sh train.npy` | implemented | The open-read re-walk did not improve: two barriers per open step and 805 MB of coefficients written per hop (next: P38) |
| P38 | 2026-11-20 | cuda@09ca447 | RTX 4090 | Opt 2, coefficient scratch: the re-walk's 3 x 128 per-slot coefficients per open read stored in bf16; the query-gradient sum deferred to the next open read's barrier (one barrier per open read instead of two) | grad check PASS. B1 120.7 -> 117.2 ms (re-walk 21.9 -> 19.7, adjoint 10.7 -> 9.5 ms); B2 89.8 ms (unchanged: closed reads store nothing); memory 7.37 -> 6.62 GiB | implemented | The open-read walk is now about 2.5 ms (re-walk) and 1.5 ms (forward) per read: latency of a sequential walk with one barrier per open step |
| P39 | 2026-11-20 | cuda@3b918e4 | RTX 4090 | Opt 3, loss: one-pass online softmax (max and sum together) and loss chunks sized to stay in the 4090's 72 MB L2 (256 rows: fp32 logits + bf16 gradients = 50 MB); not the fully fused Cut-Cross-Entropy kernel | grad check PASS. Chunk sweep (B1): 4096 / 1024 / 512 / 256 / 128 rows -> 112.9 / 112.9 / 112.0 / 106.9 / 114.1 ms (the one-pass softmax alone: 15.9 -> 11.6 ms at 4096 rows). B1 117.2 -> 107.0 ms (softmax 5.7 ms, GEMMs 32.7 ms); B2 89.8 -> 79.5 ms; memory 6.62 -> 5.91 GiB | implemented | A fused kernel (logits never in memory) would also remove the 3 loss GEMMs' traffic; the L2 already absorbs most of it |
| P40 | 2026-11-20 | cuda@2e8caf4 | RTX 4090 | Opt 4, addressing: block-level reductions instead of one global atomic per (row, head) into 2-16 addresses; router-gradient flush from 1024 blocks instead of 8192 (512 rows per block); bf16 GEMM outputs feeding it, bf16 per-slot gradients, no 69 MB clear per layer | grad check PASS (4 seeds after the reference also rounds those GEMM outputs to bf16; before that, seed 2 missed on the router gradient at 5.6%). Addressing 14.3 -> 7.5 ms (forward 2.8, backward 3.0, hop-2 queries 1.7); B1 107.0 -> 99.6 ms; B2 79.5 -> 72.1 ms; memory 5.91 -> 5.69 GiB. Nsight Compute is not allowed on RunPod (ERR_NVGPUCTRPERM): analysis by byte and instruction counts | implemented | The forward is compute-bound on the router's 32 x 128 product per (row, head) from shared memory: tensor cores (mma.sync over 16 rows) would cut it |
| P41 | 2026-11-20 | cuda@f7250a0 | RTX 4090 | Opt 5, norms and elementwise: FFN bias + GELU (keeping the pre-activation) and the second bias in cuBLASLt epilogues, the GELU backward + bias gradient in the backward GEMM's epilogue; every norm backward also writes the bf16 gradient copy (and the b2 column sums); the loss gradient GEMM writes bf16 | grad check PASS; tools/check_epilogue PASS (cuBLASLt GELU = tanh form: 5e-6 vs 4.7e-4 for erf; an fp32 bias with a bf16 output is refused on sm_89, so b1 goes in bf16). Norms and elementwise 10.8 -> 5.1 ms (GEMMs 32.7 -> 33.3 ms with the epilogues); B1 99.6 -> 94.4 ms; B2 72.1 -> 67.1 ms; memory 5.69 -> 5.43 GiB | implemented | Left: 13 RMSNorm forwards (fp32 stream read) and the optimizer's weight preparation |
| P42 | 2026-11-20 | cuda@2b8900c | RTX 4090 | Opt 6, FP8 FFN (P22): microbenchmark of cuBLASLt E4M3 GEMMs (per-tensor scales, the TN layout sm_89 requires) against bf16 at the FFN shapes, and of a separate activation quantization pass (amax + cast) (`bench_fp8`, cold L2) | bf16 150-153 TFLOPS (93% of peak). FP8: up 0.450 -> 0.324 ms (1.39x, 212 TFLOPS), down 0.452 -> 0.228 ms (1.98x, 301 TFLOPS). With the activation quantized per call: 1.08x and 0.65x (slower). Best case with every producer writing FP8 (delayed scaling) and the TN-only weight-gradient GEMMs needing transposed activation copies: under ~4 ms per step (4%) before the transposes | abandoned (at this width) | Quantization traffic costs as much as the GEMM saves at D 512 / F 2048. Revisit at larger widths, or on GPUs with block-scaled FP8 and no TN constraint |
| P43 | 2026-11-20 | cuda@1fbd269 | RTX 4090 | Opt 7, long rows: walks parallel over chunks of 512 tokens. Forward and re-walk from slot-state checkpoints built in two parallel passes (local chunk states, then a per-slot merge: mass-weighted averages compose exactly). The adjoint (linear) as a local pass per chunk + a per-slot combine across chunks (cut at resets) + the final pass | grad check PASS, also with chunks of 8 (identical losses across chunk sizes). At 32768 tokens per step, all gates open: 8 x 4096 154.5 -> 109.0 ms (bank kernels 99.6 -> 53.4), 2 x 16384 450.8 -> 117.9 ms (387.5 -> 54.4). 64 x 512: B1 94.4 -> 96.1 ms (+1.7 ms, the heavier chunk-aware walks), B2 67.1 -> 68.0 ms. Tried and dropped: step words batched 32 at a time per warp (no gain, more registers; the one-pass checkpoint was divergence-bound, not load-bound) | implemented | Long rows now cost 13-23% more than 64 x 512 instead of 1.6-4.8x |
| P44 | 2026-11-20 | cuda@9d92c00 | RTX 4090 | Opt 8, skipping GEMM rows for closed reads: measured first how often a token has all 16 heads' reads closed (only then is its W_o row zero), after 300 training steps with the budget loss (budget 1, target 0.3) (`train_bench --steps 300 --budget 1 --budget-target 0.3 --gate-report`) | Open reads 16.5-51.5% per layer and hop; tokens with every head closed 0-1%: row skipping would save at most 1% of W_o rows. Side findings: the budget works (open rates settle near the target) but the gates learned whole heads on / off (open rates are exact multiples of 1/16), not per-token decisions; loss spiked to 15.0 at step 51 (lr 2e-3, no warmup, budget) and recovered by step 126. At ~30% open reads: 77.9 ms per step, 420.6k tokens/s | abandoned (per-head gates) | Revisit with a hop-level gate (all heads together) or if gates become token-dependent; a closed-for-all-tokens head could instead be pruned (K slices of W_o) |
| P45 | 2026-11-20 | cuda@f2d764e | RTX 4090 | Opt 9, the step as one CUDA graph: loss normalization and Adam's step count moved to the device, the step captured after an eager first step and replayed (batch copies outside) | grad check PASS; loss trajectories identical to eager to the third decimal. B1 96.2 -> 95.7 ms, B2 68.1 -> 67.6 ms (0.5-0.7%) | implemented (default on) | The CPU already queued work ahead of the GPU (no host syncs in the step), so launch overhead was mostly hidden; the graph also frees the CPU |
| P46 | 2026-11-20 | cuda@a18862c..95f45bc | RTX 4090 | P30, the real trainer: `lbc create / config / train RUN_DIR` (config.json with arch_params / data / training / runtime, unknown keys rejected), epochs of epoch_passes over random windows (loader a pure function of the global batch: exact resume), Keras-style bar + epoch line with val_loss / bpc, metrics.csv (Python's 5 columns first), model.lbc + checkpoints/best.lbc + checkpoint_NNNNN.lbc (weights + Adam), warmup-stable-decay lr computed on the device (graph-safe), forward-only validation graph. Arch options: factorized / tied / per-layer hops (1-hop layers skip hop 2, b2 gradient from the next norm's column sums) | check_and_bench.sh: grad checks PASS for default, untied, unfactorized, hops [2,1], all options (2 seeds each; ~80 extra seed checks). B1 96.2 / B2 68.0 ms (was 95.7 / 67.6; +0.5 ms is the zpre store of P47). Trainer wall clock 96 ms/step = benchmark. Variants (train_bench 20 steps, gates w2/r20): untied 96.9 ms 5.52 GiB; unfactorized (V x 512) 123.0 ms 5.74 GiB; hops [2,2,1,1] 84.0 ms 5.23 GiB; 6L [2,2,2,2,1,1] 123.8 ms 6.75 GiB. Resume: resumed epoch-2 metrics within run-to-run noise (atomics); kill mid-epoch resumes at the epoch start | implemented | The pipeline can now run real training; first finding below (P47) |
| P47 | 2026-11-20 | cuda@a18862c | RTX 4090 | Why the first real training diverged: per-tensor gradient report (train_bench --grad-report) at steps 30/45/60/90 with trained gates (biases 2) | The write-gate logit gradient explodes (l0 W_s write-gate columns 0.004 -> 4.5 -> 1470 -> 6100; keys 0.1), clip 1.0 then starves every other gradient: loss 8.0 -> 9.1 by step 150 (all-open: 5.81 at 300). Cause: a closed write gate's straight-through gradient is the tangent at weight 0 of a mass-normalized average, (k - avg)/Z, unbounded as the slot mass Z decays. Fix: the secant to the open state, tangent x Z/(Z + r) (Z = slot mass before the write, stored by the hop-1 forward walk). 300 steps: loss 6.23, grad norm 0.49. With reads held open (bias 20) and writes trained: 5.808 = all-open 5.813, writes 81-92% open | implemented | Without it trained write gates cannot be used at all. Also found: the grad check's tolerance misses for R and bwg are a bf16-adjoint precision floor (~1e-6 absolute), present in every configuration: explicit tolerance + measured noise floor in check_grads.py |
| A18 | 2026-11-20 | cuda@a18862c | RTX 4090 | Trained binary read gates (bias 2), 300 steps, lr 2e-3 warmup 50: no budget vs gate_budget 1 / 10 (target 0.5) | No budget: read gates collapse (hop 1 0-1% open, hop 2 8-19%) because a closed read gets no gradient (absorbing state, the MoE dead-expert problem); loss 6.23. Budget 1: 6.29, budget 10: 6.18, all gates ~50%, 82 ms/step. Reads held open: 5.81 | open question | lbc create defaults to reads held open (gate_bias_read 20). Read gating needs a way back: the budget works; literature fixes are load-balancing losses or dense (computed-anyway) gradients for the skipped path. Compare at a full epoch before deciding |
| P48 | 2026-11-20 | cuda@20759a5, cuda@b9ecda0 | RTX 4090 | Per-layer bank size (arch_params.bank_slots, 2..128: routing restricted to the first n slots, half-lives spread over n) and head size 32 or 64 (bank and addressing kernels templated on DH: slot state DH wide per thread, per-lane vectors in chunks of 32; router in dynamic shared memory for the 64 backward). lbc create default: 8 heads x 64 | Grad checks PASS: slots [64,16], head_dim 64 (and the chunked walk at 64); protocol B1 94.7 / B2 68.1 ms (16 x 32, unchanged). Registers at 64: fwd 230, rewalk 255, adjoint 255, no spills (the long-row merge uses 1.6 KB stack). 8 x 64 model (4L, 2 hops, 512, factorized 128): 90.8 ms/step bench, 89 ms trainer, 367k tok/s, 5.0 GiB; 3 epochs of 239 steps val_loss 5.89 / 4.88 / 4.37 (8 x 32: 73 ms, 4.50 at the same steps; bank_slots [32,64,128,128]: 6.02 at step 300, 16 x 32 5.81) | implemented | The model the user specified (512 split into 8 x 64) runs at 0.95x the 16 x 32 step time; bank sizes below 128 change the model, not the step time (the walks still run 128 threads) |
| P49 | 2026-11-20 | cuda@50fe5e3, cuda@0eff58d | RTX 4090 | Plan check 4 completed on the current build: one trainer epoch (239 steps, lr 2e-3 warmup 100, 8 x 64 heads unless noted) per variant; and the head-64 grad-check miss (l1_brg) characterized | No NaN anywhere. ms/step, val_loss after 1 epoch, memory: untied 90 ms 5.95 5.08 GiB; unfactorized (V x 512) 117 ms 4.42 5.30 GiB; hops [2,2,1,1] 80 ms 5.84 4.80 GiB; 6L [2,2,2,2,1,1] 116 ms 5.83 6.28 GiB; bank_slots [32,64,128,128] 92 ms 5.93 5.01 GiB; 16 x 32 95 ms 5.86 5.44 GiB. Fixed: create validated num_layers before its hops. Grad check: per-head scalars (log_tau, brg, bwg) have an absolute rounding floor of 1.2e-6 to 5.6e-6 per check, head size 32 and 64 alike (30 checks); explicit 1e-5 floor; 30/30 seeds and 14/14 protocol checks PASS | done | The unfactorized embedding leads after one epoch (4.42 vs 5.8-5.95) at 1.3x the step time: worth a longer comparison |
| A19 | 2026-11-20 | cuda@5dd181c | RTX 4090 | Synthetic copy task (scripts/make_copy_task.py: <bos> x_1..x_L <sep> x_1..x_L <eos>, V 512, L uniform 8-64, scored after <sep>), 2 layers x 2 hops, dim 512, not factorized, 8 heads x 64, 128 slots, reads open; 7 epochs x 366 steps (~2 min, 38 ms/step); write gates trained (bias 2) vs held open (bias 20); sweep with lbc eval | Writes trained: epoch 1 train copy acc 0.61, then collapse (val 0.43 -> 0.39), grad norm up to 3.6, writes ~40% open; sweep L16 0.69, L32 0.49, L64 0.29, L128 0.15, L200 0.10. Writes open: val copy acc 0.9999 from epoch 2, grad norm 0.04; sweep L16 1.000, L32 1.000, L64 0.9993, L128 0.61, L200 0.33 (runs /workspace/runs/copy-2L-512-nf, copy-2L-512-nf-wopen) | open question | The model copies perfectly when every token is written; the trained write gate breaks copying. Suspect the P47 secant estimator: it scales a closed write's gradient by Z/(Z + r), ~0 for empty or light slots, exactly the writes a copy needs. Hold writes open (gate_bias_write 20) for memory tests until the estimator or a write budget is fixed |
| R01 | 2026-11-20 | uncommitted | none | Review of the CUDA pipeline from the code and the literature, no measurement | estimates: 8.17 TFLOP per step (6 layers), ~15% of A5000 peak, floor ~120 ms; findings F1-F12 (decays materialized, discarded decay gradient, two passes + softmax through memory, hop-2 state recompute, dense top-2 routing, mass scan, TF32 loss writing ~21 GB per step, scans at long rows, host syncs, no carried state, `<bos>` null slot) | done | queue P20-P30, A10-A15 |

### Architecture

| ID | Date | Commit | HW | Tried | Result | Status | Because |
|---|---|---|---|---|---|---|---|
| A01 | pre-log | cc6e23b | CPU | Sigmoid gates with no sparsity pressure: the LRU input gate (two-hop needle, 2 x 2 hops); the `rope-state-memory` write gate from the logits | input gate 0.894 on filler vs 0.905 on keys (learned nothing); the write gate saturated open | finding | any skip gate needs a budget, L0 / L1 or capacity (A12, A13) |
| A02 | pre-log | cc6e23b | CPU | Write normalization under superposition (MQAR, 32 pairs in 16 slots), recall at steps 200 / 400 | mass 46.7 / 58.6%, none 26.2 / 28.3%, lru 7.2 / 8.8%, ema 2.0 / 4.7% | implemented (mass) | averages keep slow slots at full scale |
| A03 | pre-log | cc6e23b | CPU | Two-hop recall: one layer with two hops vs two stacked layers | 91.6% at 1,000 steps vs 13-15% at 1,500-2,000 | implemented (hops) | both hops query the same key space |
| A04 | pre-log | cc6e23b | CPU | Heads at the same width (copy task, peak bits) | 2 x 32: ~313; 1 x 64: ~201; 4 x 16: ~208 | implemented (heads of 32) | keys below ~32 dims get noisy |
| A05 | pre-log | cc6e23b | CPU | Query-routed erase | overwrites learned faster (overall 0.997 vs 0.968 at 800 steps); +28% step time | not adopted in lru-banks | cost; reconsider with write gates |
| A06 | pre-log | cc6e23b | CPU | Two-hop needle in a haystack, 1 bank layer x 2 hops, 64 slots | not learned (below the 25% chain-picking floor); only recent needles found | open | filler writes appear to bury needles: the case for A12 |
| A16 | 2026-11-20 | cuda@207355c | RTX 4090 | Gradient check of the C++ model: loss and every gradient against an independent float64 PyTorch implementation (`tests/check_grads.py`), tiny config, gate biases 0 (half the gates closed) | loss within 4-7e-6; with the reference's forward rounded to bf16 where the C++ stores bf16, every tensor within 5% relative / cosine > 0.998 on 2 seeds (pure float64: only near-cancelling sums such as `brg`, `log_tau2` miss by bf16 noise) | implemented (the test) | |
| A17 | 2026-11-20 | cuda@207355c | RTX 4090 | Found by A16: with `b_k = 0`, the key of a token with no previous state (a row's first token, `<bos>`) is `normalize(b_k)` = normalize(0), whose gradient is divided by the 1e-12 epsilon (key-bias gradient norm ~1e9) | the C++ model now never writes such a token (as for `<bos>`): key-bias gradient 0.5 | implemented (C++); the Python `lru-banks` has the same flaw at every row start of the first step(s) (clip by global norm then shrinks every other gradient until `b_k` moves) | |
| A07 | pre-log | uncommitted | CPU | Growing 4 -> 6 layers with `silent_from` | every position's loss identical at step 0 | implemented | |

### Training recipe and results (6-layer model)

| ID | Date | Commit | HW | Tried | Result | Status | Because |
|---|---|---|---|---|---|---|---|
| T01 | ~2026-11-16 | uncommitted | RTX A5000 | Calc-X SFT from `top2` on `calc-x-mix-bpe32768`, 16 epochs | full test 15.1% (ape210k 17.8, aqua 15.0, asdiv_a 13.2, gsm8k 3.0, mawps 56.3, svamp 7.2) | done | |
| T02 | ~2026-11-16 | uncommitted | RTX A5000 | GRPO, binary reward, 1024 questions x 8 samples per update, lr 5e-5 (`model/grpo.py`) | test sample 19.2% -> 24.1% (update 30); full test at update 5: 17.2%; short problems up (asdiv_a 13 -> 28%); gsm8k stays ~3% (train set 3.5% too) | done, best kept | GSM8K looks like a capacity limit, not an RL problem |
| T03 | ~2026-11-16 | uncommitted | RTX A5000 | GSM8K-heavy cross-entropy fine-tune, 1,000 steps, after RL | no GSM8K gain; erased the RL gains | abandoned, checkpoint deleted | |
| T04 | ~2026-11-16 | uncommitted | RTX A5000 | GRPO with partial step credit (reward 0.5 x matched steps), `--no-std`, from T02's best | groups with reward variance 68% vs ~40%; best 25.3% (test sample) at update 5; stopped at update 12 by a pod restart | paused, **best math checkpoint** | +1.2 points over T02 on the test sample: a marginal gain so far |
| T05 | ~2026-11-17 | uncommitted | pod CPU | Code ability of `top2` vs RL; KodCode-V1 converted to tool-use coding (sandboxed pytest runs) | ~1.5 of 7 simple tasks; dataset 35,329 train docs (28.3M tokens), 597 validation, 1,144 test | dataset built | next: T13 |

## Dead ends (do not spend time here; references in the review, section 12)

- **Hand-edited SASS:** no official assembler (CuAssembler is third-party and lists sm_60 to sm_86); gains only on
  compute-bound GEMMs. The bottleneck here is memory traffic.
- **Own GEMMs instead of cuBLASLt:** cuBLAS already at ~80% of peak (P05).
- **FP8 through a library at width 512:** quantization overhead exceeds the gain; no FP8 on Ampere. Only P22's fused
  variant is open.
- **FP8 for the recurrences, the bank read, the routing, the norms or the loss.**
- **Using the top-2 sparsity for fewer FLOPs** (scatter, atomics): dense tensor-core tiles win; use the sparsity for
  storage.
- **Per-token skipping without compaction or block alignment:** no speedup.
- **Sigmoid gates without sparsity pressure** (A01).
- **A token-by-token bank kernel for training:** too few programs at long rows; fine for generation.
- **Log-space scans** (P07); materialized decay tensors (P09); per-token decay tensors for fixed decays (R01, F1).
- **A full associative scan over T for the bank state:** memory explodes; the chunked form exists for this.
- **FlashRNN-style register-resident recurrent matrices:** the LRU is diagonal.
- **Multi-token prediction at 27M:** the gains grow with model size.
- **H100 for this model:** its peak needs a second (wgmma / TMA) kernel path; the price is not justified at 27M.
- **GSM8K-heavy cross-entropy after RL** (T03).
