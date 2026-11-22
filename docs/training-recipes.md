# Training recipes: the 4-layer `lru-banks` model, from pretraining to RL

Every stage of the current model, in order, with the exact settings that produced it. All stages run in the C++/CUDA
pipeline (https://github.com/SamuelHaidu/lru-banks-cuda): `lbc train RUN_DIR` reads `RUN_DIR/config.json`;
`lbc grpo RUN_DIR` reads `RUN_DIR/grpo.json`. Results and their measurements are rows of `RESEARCH_LOG.md` (IDs in
brackets). Local copies of every run are in `runs/<name>/`.

```
pretrain-code-mix (2.0B tokens)
  -> 1. pretraining, 512-token rows, 100 epochs                     lru-banks-4L-512-8x64-nf
  -> 2. context 1024, warm-restart cosine, 24 epochs                lru-banks-4L-512-8x64-nf-ctx1024-cyc4
  -> 3. anneals with tool-use data (Toucan) at 10% / 20% / 30%      anneal-t10-ctx1024, anneal-t20-ctx1024, anneal-t30-ctx2048
  -> 4. Calc-X SFT (70% Calc-X, 30% pretraining)                    calcx-sft-t30
  -> 5. RL on Calc-X: RAFT, then GRPO + guards + novelty            rl-raft-sft-t30, rl-ab-nov10
```

## Model

Shared by every stage (`arch_params`):

| Key | Value |
|---|---|
| vocabulary_size | 32,768 (tokenizer `pretrain-mix-bpe32768`) |
| dim / num_layers / hops | 512 / 4 / `[2, 2, 2, 2]` |
| embedding | not factorized (`factorized: false`, `embed_dim` unused), tied |
| lru_dim / lru_half_life | 256 / [1, 16] |
| banks | num_heads 8, head_dim 64, 128 slots per head, top-2 routing, slot_half_life [1, 1024], tau 8 |
| gates | gate_bias_write 2, gate_bias_read 20 (reads open), gate_budget 0 |
| ffn_mult | 4 |
| parameters | 32,838,304 (embeddings 16,777,216, mechanism 16,061,088) |

`window_size` changes per stage (512, 1024, 2048, 512).

## Data

| Dataset | What | Size |
|---|---|---|
| `pretrain-code-mix-bpe32768` | `pretrain-mix-bpe32768` (70% cosmopedia, 25% stack-edu, 5% finemath) + `kodcode-agent-bpe32768` (tool-use coding), concatenated (`merge_npy.py`) | 1,993,396,628 train tokens (1,965,111,433 + 28,285,195), 2.72M documents; validation 12.15M |
| `toucan-1b-bpe32768` | random whole documents of `toucan-bpe32768` train (p 0.1632, numpy seed 1234), `loss_spans [[6, 7]]` (assistant turns only) | 998,424,089 tokens, 215,204 documents; zstd -22 --ultra --long: 0.937 bpc on validation (easy to compress, so its losses overstate ability) |
| `calc-x-bpe32768` (on the pod: `/root/data/cx`) | Calc-X with `calculator` tool calls, `loss_spans [[6, 7]]`; `train.docs.jsonl` / `test.docs.jsonl` for RL | 46.9M train tokens, 317,844 documents; test 6,068 questions (6 sources) |

Mixing (stages 3 and 4): `data.mix_datasets` / `data.mix_fractions` give every batch a fixed number of rows from each
extra dataset (`round(fraction x batch_size)`), the rest from `data.dataset`; each source has its own random windows
and passes, so the pretraining distribution is never thinned out by sampling. Validation then reports `val_loss` on
the main dataset and `val_mix_loss` on the first mixed one. With `loss_spans`, a mixed dataset's share of the scored
targets is lower than its share of rows (stage 3, 30% of rows: 12.8% of targets).

## 1. Pretraining: `lru-banks-4L-512-8x64-nf`

| Key | Value |
|---|---|
| data | `pretrain-code-mix-bpe32768`, batch 256 x 512, `epoch_passes` 0.02 (304 steps per epoch), seed 42 |
| optimizer | AdamW lr 1e-3, betas (0.9, 0.999), eps 1e-7, weight decay 0.1, clip 1 |
| schedule | warmup 100, constant, then cosine from step 14,950 over 14,950 steps to 0.1 x (1e-4) |
| length | 100 epochs, ~3.96B tokens; RTX 4090 |
| result | val 2.1846 (512-token rows); ARC-Easy 0-shot acc 40.07 / acc_norm 37.12 [A29] |

## 2. Context extension to 1024: `lru-banks-4L-512-8x64-nf-ctx1024-cyc4`

| Key | Value |
|---|---|
| init | stage 1 final weights, fresh optimizer (`training.init_from`) |
| data | same dataset, `window_size` 1024, batch 256 rows as 2 micro-batches of 128 (`data.micro_batch_size`: exact gradient accumulation), 152 steps per epoch, seed 43 |
| schedule | `lr_cycles` 4: warm-restart cosine, each cycle 1,900 steps from 1e-3 to 1e-4 (100-step warmup back to the peak at each restart) |
| length | planned 50 epochs; stopped at epoch 24 (3,648 steps, 0.96B tokens); RTX 5090, ~984 ms per step |
| result | val 2.1620 (1024-token rows); ARC-Easy 40.11 / 37.16 (unchanged); loss far into documents much lower (896-1024: 2.435 -> 2.235) [T32, T33] |

## 3. Anneals with tool-use data

Each anneal: fresh optimizer from the previous one, warmup 100 to lr 1e-3, then **linear** decay to 1e-4 over the whole
run (`lr_decay_shape` linear, `lr_decay_start` 0), weight decay 0.1, clip 1. Every batch mixes the pretraining data
and Toucan at a fixed ratio.

| Run | window | batch | Toucan rows | steps | tokens | seed | init | val (pretraining) | val Toucan | ARC-Easy |
|---|---|---|---|---|---|---|---|---|---|---|
| `anneal-t10-ctx1024` | 1024 | 256 (2 x 128) | 10% (26) | 12 x 152 = 1,824 | 478M | 44 | stage 2, epoch 24 | 2.1666 | 2.4477 | 40.53 / 37.50 |
| `anneal-t20-ctx1024` | 1024 | 256 (2 x 128) | 20% (51) | 1,824 | 478M | 45 | t10 | 2.1642 | 2.2624 | 40.32 / 37.46 |
| `anneal-t30-ctx2048` | 2048 | 384 (4 x 96) | 30% (115) | 25 x 50 = 1,250 | 983M | 46 | t20 | 2.1851 (2048 rows) | 2.2374 | 39.60 / 37.33 |

RTX 5090: 1024 x 256 at ~966 ms per step (271k tokens/s); 2048 x 384 at ~2.80 s per step (280k tokens/s, 27.4 GiB)
[T35-T37].

## 4. Calc-X SFT: `calcx-sft-t30`

| Key | Value |
|---|---|
| init | `anneal-t30-ctx2048` final, fresh optimizer |
| data | `window_size` 512, batch 256 (2 x 128): 30% pretraining rows (77) + 70% Calc-X rows (179) every batch; scored targets ~50 / 50 |
| schedule | warmup 50 to **5e-4** (half the pretraining peak), linear decay to 5e-5 over 1,200 steps (24 epochs of 50, ~2 passes over Calc-X) |
| length | 1,200 steps, 157M tokens, 11 min on the RTX 5090 (488 ms per step) |
| result | Calc-X val loss 0.487, val accuracy 87.4%; pretraining val (512 rows) 2.2636; Calc-X greedy agent eval 14.3% (300 per source), **10.15% on the full test**; ARC-Easy 39.10 / 35.69 [T38] |

```json
"data": {"dataset": "/root/data/pretrain-code-mix-bpe32768", "batch_size": 256, "micro_batch_size": 128,
         "epoch_passes": 0.0033, "seed": 47, "mix_datasets": ["/root/data/cx"], "mix_fractions": [0.7]},
"training": {"learning_rate": 5e-4, "warmup_steps": 50, "lr_decay_start": 0, "lr_decay_steps": 1200,
             "lr_min_ratio": 0.1, "lr_decay_shape": "linear", "weight_decay": 0.1, "epochs": 24, ...}
```

## 5. RL on Calc-X

The agent loop: the model writes `<call>calculator(expression="...")</call></assistant>`, the host evaluates it and
feeds `<result>...</result><assistant>`, until a turn without a call; the answer is the last "The answer is X". Up to
200 generated tokens. Training questions: gsm8k, mawps, ape210k (numeric answers), 203,537 of them.

### 5a. RAFT, 1 hour: `rl-raft-sft-t30` (collapsed at update 1700; keep `best-u1600`)

`mode raft` (train only on correct rollouts, weight 1), 512 questions x 8, temperature 1, AdamW lr 5e-5 (warmup 3,
weight decay 0, beta2 0.99, clip 1), TIS cap 2, GRESO skipping on. Full test 10.15% -> **21.65% at update 1600**, then
collapse at 1700 (11.8%, answered 47%): correct rollouts that wrote dozens of newlines before the answer were trained
as positives, until greedy decoding looped on newlines; pretraining val drifted 2.22 -> 2.70 [T39].

### 5b. Final recipe: GRPO + collapse guards + novelty (`rl-ab-nov10`)

```json
{"init_from": ".../rl-raft-sft-t30/checkpoints/best-u1600.lbc",
 "train_docs": "/root/data/cx/train.docs.jsonl", "test_docs": "/root/data/cx/test.docs.jsonl",
 "tokenizer": "/root/data/tok/tokenizer.json", "pretrain_dataset": "/root/data/pretrain-code-mix-bpe32768",
 "mode": "grpo", "questions": 512, "chunk": 512, "group": 8,
 "max_repeat": 4, "unique": true, "replay_rows": 64, "replay_coef": 0.2, "novelty": 1.0,
 "updates": 3150, "eval_per_source": 2000, "eval_every": 50, "checkpoint_every": 50, "save": true}
```

Defaults that apply: temperature 1, `max_new` 200, partial credit 0.5 (a wrong answer earns 0.5 x the share of the
reference's intermediate values it reproduced), Dr. GRPO advantage r - mean (`no_std`), token-level mean loss, one
optimizer step per batch, TIS cap 2, AdamW lr 5e-5 (warmup 3, weight decay 0, beta2 0.99, clip 1), GRESO on,
`micro_rows` 64.

What each guard does and why it is there:

| Key | Effect | Without it |
|---|---|---|
| `max_repeat 4` | a rollout with a run of > 4 identical generated tokens gets reward 0 (never a positive) | RAFT trained correct-but-degenerate rollouts as positives and collapsed (5a) |
| `unique true` | correct rollouts with the same solution (calculator expressions without spaces + answer) count once per group | one solution is reinforced 5-8 times per group |
| `replay_rows 64`, `replay_coef 0.2` | every update adds 64 pretraining rows to the gradient, weighted to 0.2 of the RL loss | pretraining val drifted 2.22 -> 2.70 in 1,700 updates; with replay it falls back to 2.21-2.26 |
| `novelty 1.0` | a correct rollout's reward + 1 / (copies of its solution in the group) | GRPO converged on one solution per question: entropy 0.6 -> 0.056, ~460 duplicate solutions per update, test accuracy falling (21.75 -> 18.75% in 150 updates) |

Comparison from the same start (`best-u1600`, 21.75% full test), all with the first three guards [T40-T43]:

| Variant | Full test at 150 / 200 | Entropy | Duplicates per update | Notes |
|---|---|---|---|---|
| RAFT | 20.8 / 22.8% | 1.7 (rising) | ~25 | explores but drifts; longer rollouts |
| GRPO | 18.75 / - | 0.056 | ~480 | converges on one solution |
| GRPO + novelty 0.5 | 21.0 / - | 0.062 | ~458 | stops the accuracy loss, not the convergence |
| **GRPO + novelty 1.0** | 20.5 / 22.3%, best 23.70% at 950 | **0.25, stable** | **~5** | stable and exploring; ~2.1-2.4 s per update on the RTX 5090 |

Checkpoints: `model.lbc` (latest, the resume point: raise `updates` in `grpo.json` and run `lbc grpo` again; it
continues with the optimizer state), `checkpoints/best.lbc` (best full-test accuracy). Every eval writes
`eval_<update>.jsonl` (each test question's output, correctness, finished, calls, length) for inspection.

Watch while it runs: `grep eval RUN/train.log` (test accuracy, answered, pretraining val), entropy and `dupes` in the
update lines (entropy toward 0 or `dupes` in the hundreds: converging on single solutions; entropy rising with
`degen` and `len`: degenerating), answered below ~0.95 (collapse). A statistical trend check of the per-update metrics
(Newey-West slopes, block bootstrap, windows): `/workspace/anneal/solved_stats.py RUN BURN_IN`.

## Operations notes

- The RTX 5090 pod's volume (`/workspace`) has a **20 GB quota**; a full volume kills a run at its next checkpoint
  write. Keep intermediate checkpoints local (the cron sync never deletes local files) and delete them on the pod
  after an md5 check.
- Evaluations share the GPU with training only in small batches (about 4 GB free next to a 2048 x 96 run).
