# Project Guide

Recurrent language models with slot memories. **Current focus: the most efficient CUDA training pipeline for the
`lru-banks` architecture** (next-token cross-entropy training: pretraining and fine-tuning), then architecture changes
that save computation (binary read / write gates, see the log). This file is the short guide. The full previous guide
(every architecture, dataset, tool and measurement) is `docs/agents-reference.md`: search it before re-deriving or
re-measuring anything.

## Research rules (read first)

- **`RESEARCH_LOG.md` is the tracking document.** Before proposing or starting an experiment, read it: the current
  numbers to beat, the queue, and every idea already tried with its result and why it was kept or abandoned. Do not
  retry an abandoned idea unless the condition its entry names has changed (another GPU, a new kernel, ...).
- **After every experiment, add its entry** (ID, date, commit, hardware, what was tried, result with the command that
  measured it, status, because) and update "Current standing" if it beats it. Never delete entries: supersede them.
- **Commit the code before recording a result**, so the entry's commit reproduces it.
- **Microbenchmark a component before changing the full step**, then confirm on the full step. Every speed number names
  the script and arguments, the GPU, the shapes, the warm-up and the number of timed steps.
- The analysis behind the queue (where time and memory go, the plan, the literature, dead ends with references):
  `studies/cuda-training-pipeline-review.md`.
- **The C++/CUDA pipeline lives in its own public repository**, https://github.com/SamuelHaidu/lru-banks-cuda
  (CMake + Ninja, C++20, CUDA 12).
  - Local clone: `~/projects/lru-banks-cuda`; on the pod: `/workspace/lru-banks-cuda`.
  - The pod pushes with a deploy key (write access to that repository only). It is stored in `/workspace/.ssh`;
    `scripts/pod_setup.sh` copies it to `~/.ssh` with mode 0600 because `/workspace` ignores permissions.
  - Commits use the GitHub no-reply address, never a private email.
  - Log rows cite its commits as `cuda@<sha>`.

## Environment and commands

Local machine: Intel Iris Xe (XPU) by default, no exports needed. Always `uv run model ...` (the project's `.venv`,
built by `uv sync` from `uv.lock`; another venv lacks the XPU torch wheel and `keras_hub`). Training runs on RunPod
CUDA pods (below); local runs are for tests, evaluation, the web chat and small synthetic experiments.

```bash
model create --run-name X --arch lru-banks --dataset pretrain-mix-bpe32768   # writes runs/X/config.json
model config --run-name X                                                     # print
model config --run-name X set data.batch_size 256                             # dotted key, JSON value; unknown keys rejected
model train --run-name X [--epochs N]            # resumes automatically from runs/X/model.keras + metrics.csv
model evaluate --run-name X [--checkpoint best|latest|N]                      # eval_<ckpt>.json
model generate --run-name X --prompt "..." [--temperature 0.8 --top-k 50]
model create --run-name Y --init-from X [--init-checkpoint best]              # same weights, fresh optimizer
.venv/bin/python -m unittest discover -s tests -t .                           # tests (CUDA parts skip locally)
```

- **Config keys that matter here:**
  - `data.batch_size` / `data.micro_batch_size`: gradient accumulation, the exact mean gradient over the whole batch.
  - `data.random_windows`, `data.epoch_passes`.
  - `training.dense_supervision` / `dense_validation`.
  - `training.lr_decay_start` / `lr_decay_steps` / `lr_min_ratio`: warmup-stable-decay.
  - `training.checkpoint_every`.
  - `init_weights_from` (copy named weights from another run, shell wildcards).
  - `training.freeze_except`.
- **Checkpoints:**
  - `model.keras` (latest, the resume point);
  - `checkpoints/best.keras`;
  - `checkpoints/checkpoint_NNNNN.keras` every `checkpoint_every` epochs. These hold the optimizer state too.

## `lru-banks` (the architecture under study)

`model/archs/lru_banks.py`. The torch reference is `model/archs/bank_memory.py`; the CUDA kernels are
`model/archs/bank_kernels.py`.

- **Embedding:** factorized and tied. ids -> E (32,768 x 128) -> W_in -> a 512-wide stream.
- **Readout:** `logits = (rms(h) W_out) E^T`, through the fused linear cross-entropy (`model/fused_linear_cross_entropy.py`).
- **Layer n:** `x = rms(h)`, then:
  - an **LRU**: 256 complex channels, state s = 512 reals, half-lives 1..16 at init. Layers >= `gate_from` have an
    input gate `g_t = sigmoid(x w + b)` that scales the decay (g = 0 skips the token).
  - **banks**: 16 heads x 32 (keys and values), 128 slots per head.
    - key = `normalize(W_k s_{t-1} + b_k)`, the state before the token; query = `normalize(W_q s_t)`; value `W_v x`.
    - Written top-2 into the slots a router picks on the key (renormalized).
    - Each slot is an average ("mass" normalization) with a fixed decay (half-lives 1..1024, identical in all layers).
    - Read: `softmax(tau Kb q / z)` over the slots.
  - **Hop 1:** `h += W_o read(q)`; `h += FFN(rms h)` (GELU, 4x).
  - **Hop 2** (layers with 2 hops): `q2 = normalize(W_q2 rms h)`, read the same banks, `h += W_o2 read(q2)`, the same
    FFN again.
- States restart at `<bos>` (id 0). At a `<bos>` the key is the constant `normalize(b_k)`, so every document writes
  the same entry: a learned "null" slot.
- **Main runs:** 6 layers, hops `[2, 2, 2, 2, 1, 1]`, 27.3M params, packed rows of 512, batch 64.
  - Grown from `lru-banks-4L-512` with `silent_from` 4: the new layers start with their output projections at 0, so
    every loss is unchanged at step 0.
- **CUDA path** (`LRU_BANKS_FUSED=0` disables it):
  - a Triton LRU scan (gate inside, reverse-scan backward);
  - a Triton mass scan;
  - the bank read on fla 0.6's gated-slot-attention kernels (`chunk_gsa_fwd_k` + a compiled softmax + `chunk_gla_fwd`);
  - compiled bf16 pieces between them (`_f_pre`, `_f_addr`, `_f_post`, `_f_q2`);
  - rows flattened with `cu_seqlens` cut at row and document starts.
- **Torch path:** generation and CPU / XPU. On CUDA it runs in fp32: its chunked forms multiply factors up to e^30 and
  their inverses, which is wrong in bf16.
- **Tests:** `tests/test_arch_lru_banks.py` (row == generation, kernels vs the float64 reference: values and gradients)
  and `tests/test_bank_memory.py`. They are the oracle for any new kernel.
- **Benchmark:** `uv run python ad_hoc_scripts/bench_lru_banks.py [--batch N] [--window T] [--set k=v] [--profile]`
  (step time, tokens/s, TFLOPS, GPU-time split, peak memory). Its `--set` takes scalars only: the per-layer `hops`
  list needs a small change first.
- **Inspection:** `ad_hoc_scripts/lru_banks_layer_use.py` (per layer: output size, gate mean, loss when silenced) and
  `lru_banks_weight_stats.py`. Web chat Inspect: Loss / Stored / Recall / Gate, the exact read weights over past
  positions.

Measured performance and its history are in `RESEARCH_LOG.md` (current standing table).

## Data (local copies; push to the pod as needed)

| Dataset | What | Notes |
|---|---|---|
| `pretrain-mix-bpe32768` | 70% cosmopedia, 25% stack-edu (15 languages), 5% finemath; 1.965B train tokens, 4.38 chars/token, uint16 | train split stored as `train.npy.zst` (1.9 GB): decompress on the pod after `push`; tokenizer `tokenizers/pretrain-mix-bpe32768` |
| `calc-x-bpe32768` | Calc-X (gsm8k, ape210k, math_qa, aqua_rat, mawps, asdiv_a, svamp) with `calculator` tool calls | `loss_spans` (assistant turns); test split for the evaluation |
| `calc-x-mix-bpe32768` | Calc-X weighted + 30% pretraining documents (`datasets/build_calc_x_mix.py`) | every token scored, `"chat": true` so the web chat offers Chat |
| `kodcode-agent-bpe32768` | KodCode-V1 (CC BY-NC 4.0) as tool-use coding (write / run pytest / edit), 35k train docs, 28.3M tokens; solve and fix | built with a sandboxed runner (`datasets/build_kodcode_agent.py`); not trained on yet |
| `toucan-bpe32768` | Toucan-1.5M (Apache 2.0, arXiv 2510.01179): tool-agent trajectories over 495 real MCP servers (Kimi-K2 / OSS / Qwen3 generations, SFT subset for unseen uuids), converted to our tags (`<tools>` signatures, `<call>name(k=<JSON>)</call>`, `<result>`, OSS reasoning as `<think>`); artifacts dropped (bad call JSON, no final answer, tag text in content), non-Latin script > 0.5% of letters dropped; 6.13B train tokens, 1.32M docs, validation / test ~1.2k docs | `datasets/build_toucan.py`; stored as `datasets/toucan-bpe32768.tar.zst` (train as `train.npy.zst`, multi-frame); for long-sequence fine-tuning; loss_spans = assistant turns |

Packed rows (`PackedDocumentDataset`, `model/data.py`): `<bos>doc<eos>` documents concatenated, random windows (one
pass per epoch, or `epoch_passes` of one), targets -1 at document ends and outside `loss_spans`. Older datasets
(agent-shell, agent-mix, cosmopedia parts, 1024-token BPEs) are in `docs/agents-reference.md`.

## Runs and checkpoints (local)

| Run | What | Result |
|---|---|---|
| `lru-banks-6L-512-top2` | pretrained base, before Calc-X | the start for the coding fine-tune |
| `lru-banks-6L-512-calcx` | Calc-X SFT (16 epochs) | Calc-X full test 15.1% (gsm8k 3.0) |
| `lru-banks-6L-512-grpo-b1024` | GRPO, binary reward, 1024 x 8 per update | best 24.1% (test sample) |
| `lru-banks-6L-512-grpo-pc` | GRPO with partial step credit (0.5, `--no-std`) | **best math checkpoint**, `checkpoints/best.keras`, 25.3% (test sample) |

- **Calc-X evaluation:** `uv run python ad_hoc_scripts/calc_x_eval.py --run-name X [--checkpoint best] [--limit N]
  [--device xpu|cuda]` (agent loop with the calculator; `model/calcx.py`).
- **RL:** `python -m model.grpo --run-name Y --init-run X [--partial-credit 0.5 --no-std ...]` (`model/grpo.py`).
- **Every training stage of the current 4-layer model** (pretraining, context 1024, Toucan anneals, Calc-X SFT, RL) with its exact config and results: `docs/training-recipes.md`.
- **RL in C++ (lru-banks-cuda, no Python):** `lbc grpo RUN_DIR` with `RUN_DIR/grpo.json` (init_from, Calc-X docs, tokenizer, mode `grpo` | `raft`, group, questions, ppo_epochs, tis_cap, save, ...); ~3 s per update of 512 x 8 rollouts on the 4090 (log P54, T08, T27-T30). Checks: `lbc test-calcx` (tokenizer / calculator / reward oracles), `lbc decode-check` (incremental decoder vs the training forward). Collapse guards (log T39-T43): `max_repeat`, `unique`, `replay_rows` / `replay_coef`, `novelty`; use them (RAFT without them collapses, plain GRPO converges on one solution).

## RunPod (`pod.sh`)

- **`remote-pod`** (`tools/remote-pod/`, Go): the replacement, built and tested locally, not yet proven on a live pod.
  `remote-pod --help` is its manual: `up` / `sh` / `cp` / `sync` / `logs` (an agent in our image, reached through
  RunPod's HTTPS proxy: no SSH, community pods work), `train` (datasets and checkpoints in R2; the pod stops itself),
  `build` / `image` (lbc compiled in a local CUDA container). Until its live test, `pod.sh` below is the working way.

```bash
./pod.sh gpus [--all]          # GPUs in stock now, cheapest first: id for `up`, VRAM, $/hr, stock, data centers
./pod.sh up [gpu-id]           # create (community with a public IP, else secure); id in .pod
./pod.sh push [dataset]        # code + one dataset + its tokenizer by rsync; venv on first push
./pod.sh sh 'cmd'              # run in /root/rnn-research with the venv
./pod.sh train <run>           # background training, log runs/<run>/train.log; ./pod.sh log <run>
./pod.sh pull <run> [--all]    # runs/<run> back (config, metrics, latest + best; --all: everything)
./pod.sh status | stop | start | rm
```

- **Disks:** code, data and the venv sit on the container disk (fast, wiped when the pod stops: `push` again). `runs/`
  links to `/workspace/runs` (kept while the pod exists). The pod's CPU quota is ~7.65 cores although `nproc` says
  more.
- **Rules:**
  - Upload only code and the needed datasets; never `.env`, `credentials.txt`, venvs or `runs/`.
  - Evaluations run on the pod, not locally; with `OMP_NUM_THREADS=3` and nice 19 while the GPU trains.
  - Remove any crontab pull entry when the pod stops.
  - `/workspace` has a 20 GB quota (`df` shows the shared filesystem, not the quota): a full volume kills a run at its
    next checkpoint write (log P55, P59). Keep only `model.lbc` + `best.lbc` on the pod; the cron sync
    (`sync_run.sh`, never deletes locally) keeps the rest; delete on the pod only after an md5 match.
  - Watch every long run (a 5-minute check: process alive, volume use, eval answered / accuracy).
- **SSH:** `push` and `pull` need full SSH (rsync), so a community pod must have a public IP (`--public-ip`, port
  22/tcp). RunPod's proxy SSH (`ssh.runpod.io`) is interactive only.
- **Known problem (2026-11-17 to 11-20):**
  - `rest.runpod.io` does not resolve (NXDOMAIN everywhere). runpodctl 2.9.0 uses it for `pod create / get / stop`.
  - `gpus` works because it goes through the GraphQL API (`https://api.runpod.io/graphql`).
  - Workaround: call the GraphQL mutations (`podStop`, `podFindAndDeployOnDemand`) with curl and the key from
    `~/.runpod/config.toml`. Never print the key.
- The last pod (`51d0e5uzmw5mf3`) no longer exists (found 2026-11-20); everything needed is local.
- **GPU choice for the pipeline work:** RTX 4090 (Ada sm_89: FP8, 72 MB L2, 165 bf16 TFLOPS, $0.34/hr community when
  in stock) or a 48 GB Ada card. The earlier runs used an RTX A5000 (Ampere, 111 bf16 TFLOPS, 6 MB L2, no FP8).

## Findings to keep in mind (details and numbers: `docs/agents-reference.md`)

- **Bank memory (synthetic tests):**
  - One bank layer does induction.
  - One layer with two hops over the same banks does two-hop recall (91.6%) where two stacked layers stall (13-15%).
  - Use "mass" normalization (averages), top-2 renormalized routing, heads with keys >= 32 dims (2 heads x 32 beat
    1 x 64 and 4 x 16).
  - LRU keys encode the last ~3-4 tokens in order.
  - Banks learn MQAR / induction in ~200 steps where one GDN layer needs 600-800.
- **Gates:** sigmoid gates left alone stay open (the LRU input gate in the needle test: 0.894 on filler vs 0.905 on
  keys; the `rope-state-memory` write gate saturated open). Any skip / sparse gate needs a sparsity pressure (budget,
  L0 / L1, or capacity).
- **Needle in a haystack:** every token writes into the banks, and the filler writes appear to bury the needles. This
  is the case for a write gate.
- **Performance rules:**
  - Factorize per-slot decays in chunked recurrences (`a^(t-i) = a^t a^-i`), batch every chunk at once, carry only
    chunk-start states. Never materialize slots x C x C decay tensors. Cap the chunk so `a^-i` < e^30.
  - At 1-2 long rows a sequential scan is latency-bound: segment it (`rsm_scan._segmented`).
  - Log-space scans (`logcumsumexp`) were 15-100x slower on the XPU.
- **Iris Xe quirks (local runs):**
  - the embedding backward fails for 1025..2048 ids, and boolean-mask indexing backward fails at some shapes: gather
    with `index_select`;
  - `torch.compile` does not work there;
  - oneDNN's fp16 GEMM runs at the fp32 rate (hence the assembly `hgemm`, `mlp_fp16`, for `rope-state-memory`).

## Other architectures and tools (reference only, see `docs/agents-reference.md`)

- **Archs** (registry `model/archs/__init__.py`):
  - the dual-sphere / ALiBi / RoPE transformers (`alibi-retriever-sphere` was the best of that family);
  - `wave-arrow`;
  - `rope-state-memory` (SYCL kernels, fp16 assembly GEMM on the XPU);
  - `min-gru` / `min-gru-cuda` (static memory);
  - `gated-deltanet` (+ MoE, its CUDA optimization history is the template for this work).
- **Web chat:** build the page once (`cd webchat/ui && npm install && npm run build`, needs Node), then
  `uv run python -m webchat.server [--port 8000] [--device xpu|cpu]`, at http://localhost:8000. Chat with tool
  execution, Complete for plain-text runs, Inspect (lru-banks: read weights over past positions). For the math models:
  clear the system prompt, tick only `calculator`. Two processes: a FastAPI server (`webchat/server.py`: page, tools,
  catalogs) and an inference worker it starts (`webchat/worker.py`), over a Unix socket with msgpack frames
  (`webchat/ipc.py`). The page is React + TypeScript (`webchat/ui/`), built into one self-contained HTML file.
- **Reports, per architecture** (`model/reports/`): `model report --run-name X` lists the reports for X's
  architecture; `model report <name> --run-name X` writes one as HTML under `runs/X/reports/`. A report declares
  `ARCHS` (architectures, a family such as `dual-tables`, `"any"` or `"no-run"`); add one by dropping a module there.
