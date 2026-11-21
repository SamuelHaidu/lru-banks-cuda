# lru-banks-cuda

A C++/CUDA training pipeline specialized to one model: `lru-banks`, a recurrent language model made of an LRU and
slot memories ("banks") read twice per layer. The goal is the fastest possible next-token training on one GPU,
reached one measured step at a time. Every component starts as a microbenchmark against the current implementation;
only what wins is kept.

The model's reference implementation (PyTorch, tested in float64) and the research log live in the companion
research repository. The log holds every experiment with its commit here, its result and the decision. This
repository holds the CUDA side.

## Build

Requirements: CUDA 12 (nvcc), CMake >= 3.26, Ninja, a C++20 compiler.

```bash
cmake --preset release            # build/release, for this machine's GPU (CMAKE_CUDA_ARCHITECTURES=native)
cmake --build --preset release
./build/release/device_info       # the GPU's properties
```

For another GPU: `cmake --preset release -DCMAKE_CUDA_ARCHITECTURES="86;89"`. A debug build uses
`--preset debug`.

## Layout

```
include/lbc/   shared headers: check.hpp (error checks), bench.hpp (timing, L2 flush, result lines)
tools/         lbc (the trainer), train_bench (benchmark, gradient dumps), device_info, check_epilogue, bench_fp8
bench/         microbenchmarks, one file per question (added as the search goes)
scripts/       pod_setup.sh
```

## Training

```bash
build/release/lbc create /workspace/runs/NAME --dataset /root/data/pretrain-mix-bpe32768 [section.key=value ...]
build/release/lbc config /workspace/runs/NAME [section.key value]     # print, or set one key (validated)
bash scripts/train.sh /workspace/runs/NAME [--epochs N]                # background; tail -f RUN_DIR/train.log
```

`config.json` holds every setting:
- `arch_params`: dim, embed_dim, factorized, tied_embeddings, num_layers, hops (per layer, 1 or 2), lru_dim, num_heads,
  head_dim (32 or 64), bank_slots (per layer, 2..128), ffn_mult, half-lives, tau, gate biases and budget, window_size;
- `data`: dataset directory, batch_size, epoch_passes (an epoch = that fraction of a pass in shuffled random windows),
  seed;
- `training`: learning_rate, warmup_steps, lr_decay_start / lr_decay_steps / lr_min_ratio, weight_decay, betas,
  epsilon, global_clipnorm, epochs, validation_steps, checkpoint_every;
- `runtime`: bank_chunk, ce_chunk, cuda_graph.

Every epoch writes `model.lbc` (latest, the resume point: weights and Adam state), `checkpoints/best.lbc` (lowest
val_loss), `checkpoints/checkpoint_NNNNN.lbc` (every `checkpoint_every` epochs) and a `metrics.csv` row (the research
repository's columns first). `lbc train` on an existing run resumes at the next epoch with the exact next batch.

The benchmark and gradient-check protocol is `scripts/check_and_bench.sh` (`tools/train_bench.cu`).

## Conventions

- **One microbenchmark per question.** It prints one result line per variant: median and min time, TFLOPS or GB/s.
  It says whether the L2 was warm or flushed.
- **A variant counts only after it matches a reference** (fp32 / fp64) within a stated tolerance.
- **Commit before measuring.** The research log records the commit that produced each number.
- **Plain CUDA C++ plus NVIDIA's libraries** (cuBLASLt for dense GEMMs); no framework.

## On a RunPod pod

The container disk is wiped when the pod stops; keep the clone in `/workspace`. The pod pushes with a deploy key
that has write access to this repository only. The key is stored in `/workspace/.ssh`. `/workspace` does not keep
file permissions, so ssh uses a `0600` copy in `~/.ssh` (made by `pod_setup.sh`).

```bash
# after every start (the clone already exists):
cd /workspace/lru-banks-cuda && bash scripts/pod_setup.sh   # Ninja, git over the deploy key, build, device_info

# first clone on a pod with the key in /workspace/.ssh:
install -m 700 -d ~/.ssh && install -m 600 /workspace/.ssh/github_lru_banks_cuda ~/.ssh/
GIT_SSH_COMMAND="ssh -i ~/.ssh/github_lru_banks_cuda -o IdentitiesOnly=yes \
  -o UserKnownHostsFile=/workspace/.ssh/known_hosts_github" \
  git clone git@github.com:SamuelHaidu/lru-banks-cuda.git /workspace/lru-banks-cuda
```

## License

MIT, see `LICENSE`.
