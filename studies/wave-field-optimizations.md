# wave-field: training and inference optimizations

The model is a diagonal linear state space: every channel is one complex number with a fixed rotation and decay,
`S <- lambda S + w` with `lambda = e^{-alpha + i omega}`. That is the class (S4D, LRU, RetNet, Mamba-2 without
input-dependent decay) whose promise is parallel training, linear cost in length and O(1) generation. This page
lists every optimization we have identified, what was measured, and what is still open.

Setup for all numbers: standard `wave-field` (4 state spaces x 64 channels, cascade read, readout MLP 1024),
batch 512, window T = 128, dense supervision (255 inputs, 128 targets per sample, 65,536 targets per step),
Intel Iris Xe (XPU), fp32, eager PyTorch through Keras.

## 1. Where the time goes (measured)

Profile of 5 dense train steps (`/tmp/prof/prof.py`, torch profiler, CPU + XPU):

- 3.63 s/step wall, 3.59 s/step of GPU kernels: the GPU is busy; launch / CPU overhead is not the problem.
- Peak memory 4.28 GiB allocated (5.16 reserved).
- Forward + fused CE + optimizer 2.05 s; autograd backward 1.55 s.

| Part | ms/step | Share |
|---|---|---|
| Keys transpose copy (2B,S,C) -> (C,S,2B) in `dense_hidden` / `_band_matmul` (profiler attribution, see 2.1) | 873 | 24% |
| Fused readout + cross-entropy (10k vocab) | ~555 | 15% |
| Readout MLP 768 -> 1024 -> 128 (fwd + bwd) | ~330 | 9% |
| Cascade read elementwise on per-state-space slices (fwd + bwd) | ~300 | 8% |
| Write-side backward: mix multiply, its sum, slice copies on (B,S,L,K) | ~230 | 6% |
| State transpose back to (B,Q,L,K), backward | ~103 | 3% |
| Band matmul itself (bmm) | 28 | <1% |
| Write table build + gather | 32 | 1% |

Lesson: by FLOPs the state part is ~9% of the step; in time it is much larger because the cost is memory
traffic (copies, transposes, elementwise passes on 100-270 MB tensors), not math. The per-op attribution of the
profiler on this XPU is not reliable for copies (see 2.1).

Run-to-run noise: the same code measured 2.78, 2.93 and 3.26 s/step in different rounds of one session (clock /
thermal drift). Differences under ~10% need interleaved timing over 20+ steps.

## 2. Tested

### 2.1 State computation (exact alternatives, `/tmp/prof/scan_test*.py`)

All variants match the current code: forward to 5e-7, gradients of every weight to 2e-6.

| Variant | Step | State part fwd+bwd | Peak memory | Verdict |
|---|---|---|---|---|
| Band matmul (current) | 2.78 s (2.93 / 3.26 on reruns) | 1.62-1.79 s | 4.35-4.55 GiB | baseline |
| Log-step parallel scan in the natural (B,S,C) layout + window subtraction `S(t) = (P(t) - lambda^T P(t-T)) / Z` | 3.43 s | 2.14 s | 6.45 GiB | slower, more memory |
| Band matmul, one call for both kernels (keys transposed once) | 2.82 s | 1.56 s | 4.55 GiB | no measurable gain |
| One call + space mix folded into the write table | 2.75 s | 1.48 s | 4.31 GiB | ~9% of the state part, ~1% of the step: within noise |

Why the scan lost: 8 Hillis-Steele passes over 267 MB complex tensors, each saved by autograd. Halving the
transposes also did not halve the time, so the profiler's 873 ms for the copy overstates it.

### 2.2 Mixed precision

Not tested here; fp16 / bf16 on this GPU is known to be unreliable. Excluded.

## 3. Optimization catalogue

Legend. Exact: same outputs as now. Model: changes what the model computes (needs a retrain).
Status: measured / estimated / idea.

### A. State computation (the state space part)

| # | Optimization | Exact? | Expected gain | Status |
|---|---|---|---|---|
| A1 | Fold the space mix into the (V, L, 2K) write table before the gather (`p` depends only on the token) | exact | ~0.1 s/step, less memory | measured, within noise |
| A2 | One `_band_matmul` call with real and imaginary kernels stacked (keys transposed once) | exact | small | measured, within noise |
| A3 | Smaller query blocks in `_band_matmul` (b = 16-32) to cut the 50% zero half of each (W, 2W) block | exact | the bmm is <1% of the step: negligible | estimated |
| A4 | Gauss 3-multiplication complex product instead of 4 real ones in the band matmul | exact | negligible (bmm <1%) | idea |
| A5 | Produce the keys directly in channel-major layout: gather from a transposed, column-reordered table `(L*K*2, V)[:, ids]` -> `(C, 2B, S)`, and run the band matmul with S as the inner axis | exact | removes the big transpose; the transpose back for the read remains | idea |
| A6 | Keep the state per state space contiguous, `(L, B, Q, K)`, so the cascade's elementwise ops run on contiguous tensors | exact | part of the ~300 ms cascade elementwise | idea |
| A7 | Chunked scan (Mamba-2 / RetNet chunkwise): band matmul inside chunks of c tokens, carry the state between chunks with a short loop (one step per chunk), add `lambda^{i+1} carry` in each chunk | exact (with window subtraction) | linear in length; the base for B1 | idea |
| A8 | Log-step parallel scan (S5 / LRU style) | exact | slower here | measured, rejected |
| A9 | FFT convolution (S4 convolutional mode) for the per-channel kernel | exact | unclear on XPU (complex FFT support and speed unknown) | idea |
| A10 | Fused custom kernel (Triton / SYCL) for write + scan + read of one state space, never materializing (B,S,C) in global memory (what Mamba does) | exact | largest possible win on the state part (memory traffic -> on-chip) | idea; needs a kernel toolchain for XPU |

### B. Training formulation (state space promise: linear in length, no window)

| # | Optimization | Exact? | Expected gain | Status |
|---|---|---|---|---|
| B1 | Drop the window: pure recurrence from the document start (`window_size` removed); normalization `1/(1-|lambda|)` or LRU's `sqrt(1-|lambda|^2)` | model | unlimited context; enables B2-B4 and C1 | idea |
| B2 | Train on whole documents, every position a target: each token is an input once instead of ~2x (255 inputs per 128 targets now) | model | ~half the write / state work per target | idea |
| B3 | Pack several documents per row with state resets at boundaries (lambda = 0 at a document start), no padding | model | no wasted padded positions | idea |
| B4 | Fixed tokens per batch (e.g. 64 rows x 1024) so memory stays as now whatever the document length | - | memory bounded | idea |
| B5 | Truncated backprop over segments for very long documents (carry the state, detach it) | model | constant memory in document length | idea |
| B6 | Decay init covering document-length half-lives (now 1 .. 64 tokens at init) | model | lets B1 use long context | idea |

### C. Inference

| # | Optimization | Exact? | Expected gain | Status |
|---|---|---|---|---|
| C1 | Recurrent generation step: keep the (L x K) complex state, `S <- lambda S + w_new`, read with the cascade; O(1) per token | exact for B1; with the window, add `- lambda^T w_{i-T}` from a ring buffer of the last T ids | generation no longer reruns the whole window per token | idea |
| C2 | Cache the (V, L, 2K) write table for evaluation and generation instead of rebuilding it every forward (655M MACs) | exact | small in training, large per generated token | idea |

### D. Read and readout (not specific to the state space, but most of the time)

| # | Optimization | Exact? | Expected gain | Status |
|---|---|---|---|---|
| D1 | Compute `query_norm[l](q)` once per stage (now twice, `wave_field.py` `_read`) | exact | small | idea |
| D2 | Avoid `hidden[valid]` (copy of (N, 768)) and `int(valid.sum())` (host sync) in the dense step: loss with ignored targets on all rows | exact | small, removes a sync | idea |
| D3 | `mlp_hidden` 1024 -> 512 | model | ~17% of step FLOPs | estimated |
| D4 | `reader_hidden` 256 -> 128 | model | ~5% of step FLOPs | estimated |
| D5 | Activation checkpointing of the cascade and readout MLP (recompute in backward) | exact | ~2 GB less memory, ~+25% compute on those parts; buys batch size, not speed | estimated |
| D6 | Fused cross-entropy chunk size tuning | exact | unknown | idea |
| D7 | Mixed precision for readout / MLP / CE | ~exact | ~2x on ~75% of FLOPs in theory | excluded (unreliable on this GPU) |
| D8 | `torch.compile` / `jit_compile` to fuse elementwise chains | exact | could cut the memory traffic that dominates | excluded for now: the backend does not work on this XPU (AGENTS.md) |

## 4. Reading the catalogue

- The state space part is cheap in math and expensive in memory traffic. The big wins on it are the ones that
  stop materializing (B, S, C) tensors and their transposes: a fused kernel (A10) or, short of that, a layout
  that never transposes (A5 + A6).
- The state space promise (linear training, unlimited context, O(1) generation) is not yet used: the model
  trains with a fixed 128 window and overlapping inputs. B1-B4 with a chunked computation (A7) and C1 deliver
  it; memory then scales with tokens per batch, not with document length.
- The readout side (cross-entropy + MLP) is about a quarter of the time and is not a state space question;
  without mixed precision it only shrinks by making the model smaller (D3, D4).
- Exact, safe, small: A1, A2, D1, D2, C2.
