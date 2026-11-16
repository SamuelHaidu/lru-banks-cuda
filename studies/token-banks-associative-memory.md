# Token banks + associative memories: a parallel, recurrent-style language model

Run of record: `rope-token-banks-4-mem4-norm` (arch `rope-token-banks`), TinyStories-10k.

## Summary

- Full validation set at epoch 240: **loss 2.0855, perplexity 8.05, top-1 accuracy 51.1%**.
  - The model has 4.71M parameters: 1.28M in embeddings, 3.43M in the mechanism.
  - The best per-epoch val loss was 2.020.
- **Training cost.** 240 epochs x 50 steps, batch 128 x 255 targets (dense supervision), about 2.6 s/step on the Iris Xe.
  - Checkpoint timestamps give **about 8.4 h wall clock**, validation included. The user measured about 7.7 h of training.
  - That is 12k gradient steps and about 392M supervised targets.
- **Comparison with the transformer baseline:**
  - The local `simple-gpt-baseline` (4 layers, 4 heads, dim 128, MLP 512, single-target training) reached a best val of **2.324** after 33 epochs x 1000 steps x 512.
  - The user reports simple-gpt runs reaching **about 2.2 only after about 18 h**.
  - This model beats both, at less than half the wall clock. On loss per wall-clock hour it is the best model trained in this project so far.
- **Caveat on the comparison.** This run's context is **255 tokens** (128-token banks + a 128-write memory window); the transformer and the earlier banks runs use 128.
  - Validation windows therefore differ.
  - A window-255 baseline without memories is still needed to separate the context gain from the memory gain (see "Open experiments").
- **Generation.** The model writes coherent stories end to end (greedy from `<bos>`):
  - characters stay consistent ("a teddy bear named Teddy")
  - the plot moves from a setup to a problem, a resolution and a moral, then `<eos>`

## Lineage and results

All runs: vocabulary 10k, dim 128, dense supervision.

| Run | What it added | Result |
|---|---|---|
| `abl-exp-osc8` | single shared 8-oscillator kernel over RoPE-rotated tokens | val 2.917 |
| `rope-token-kernel-osc8` | per-token prior and forgetting lambda (`w = s_tok * kernel(k^lambda_tok)`), BM25 saturation, normalization | val 2.959 (10 x 50 steps) |
| `rope-token-banks-4` | 4 banks, each with its own kernel and per-token (prior, lambda); softmax router 128 -> 4 on the general state | best val 2.838 (13 epochs) |
| `rope-token-banks-4-gen64` | 64 kernels in the general state (one per RoPE pair), general state added to the output | eval 2.6293, best val 2.590 |
| `rope-token-banks-4-gen64-mlp` | residual GELU block 128 -> 2048 -> 128 before the readout | eval 2.4254 (epoch 10), best val 2.412 |
| **`rope-token-banks-4-mem4-norm`** | **4 associative memories (binders / unbinders, free keys, write gate), separate RMSNorms, gate-normalized memory** | **eval 2.0855 (epoch 240)**, val 2.539 at epoch 10 |

## Final architecture (information flow for one prediction)

Code: `model/archs/rope_token_banks.py`. Config: `runs/rope-token-banks-4-mem4-norm/config.json`.

1. **General state** (last 128 tokens):
   - RoPE-rotated embeddings weighted by 64 distance-only oscillator kernels (one per RoPE pair)
   - normalized by `sum|w|`, times a gain
2. **Banks** (4 of them, last 128 tokens): token j at distance k weighs `softplus(prior_b[tok_j]) * w_b(k^{lambda_b[tok_j]})`.
   - `w_b` is the bank's own 5-oscillator kernel.
   - The sum is normalized, times a gain.
   - **Retention is decided by the stored token itself.**
3. **Three softmax routers** (128 -> 4) on the general state mix the banks:
   - *output mix* (plus the general state) = `h`
   - *binder mix*, at every write position
   - *unbinder mix*, at the read position
4. **Binders** (4 MLPs, 128 -> 256 -> 97, at every position). Each outputs:
   - an 8-oscillator **value kernel** over the last 16 tokens
   - a **free write key** (64 phases)
   - a **write gate**

   The value is the kernel-weighted, RoPE-ordered average of the projected tokens `P_n x`. The write is `g * (e^{i phi} ⊙ v)`.
5. **Memory.** Each memory is the gate-weighted average of the last 128 writes: `M = sum g (e^{i phi} ⊙ v) / sum g`.
6. **Unbinders** (4 MLPs) output a read key psi; the read is `r_n = conj(e^{i psi}) ⊙ M_n`.
7. **Final block:** `out = h + MLP_{640 -> 2048 -> 128}([RMSNorm(h) ; RMSNorm(r_1) .. RMSNorm(r_4)])`, then the readout (128 -> 10k).

Training is fully parallel: banks and the general state are banded matmuls, and the memory is a cumulative sum. Dense training and single-window inference agree exactly (`tests/test_dense_supervision.py::TestTokenBanksMemory`).

## What the model learned

### Banks specialize: writer-defined retention works as predicted

Learned (lambda, prior) per bank. Lambda near 0 means `k^0 = 1`: a flat, never-forgotten weight. Lambda near 1 means the plain kernel decay.

| Token | Bank 1 | Bank 2 | Bank 3 | Bank 4 |
|---|---|---|---|---|
| rhinoceros | 0.99 / 0.14 | 0.84 / 0.43 | 0.91 / 2.27 | **0.00 / 3.00** |
| forest | 0.80 / 0.77 | 0.73 / 0.54 | 0.38 / 0.61 | **0.04 / 1.54** |
| icy | 0.99 / 0.84 | 0.83 / 0.55 | 0.69 / 0.91 | **0.01 / 1.39** |
| hill | 0.92 / 0.60 | 0.67 / 0.57 | 0.43 / 0.51 | **0.03 / 1.96** |
| bird | 0.77 / 0.90 | 0.67 / 0.78 | 0.81 / 0.77 | **0.08 / 2.79** |
| `"` | 1.00 / 1.35 | 0.60 / **2.41** | 0.22 / **3.33** | 0.58 / 0.98 |
| `?"` | 0.98 / 1.85 | 0.74 / **3.47** | 0.30 / 2.46 | 0.44 / 0.06 |
| newline | 1.00 / 2.09 | 0.76 / **3.01** | 0.20 / 2.44 | 0.33 / 0.32 |
| named | 0.78 / 1.34 | 0.85 / **3.21** | 0.11 / 0.78 | 0.58 / 0.16 |
| asked | 0.70 / 1.33 | 0.62 / **3.13** | 0.26 / 2.64 | 0.34 / 0.62 |
| Billy | 1.00 / 0.57 | 0.73 / 1.12 | 0.37 / 1.64 | 0.39 / **0.03** |
| Lily | 1.00 / 0.92 | 0.72 / 1.35 | 0.26 / 2.01 | 0.33 / 0.19 |
| `.` | 0.99 / 2.04 | 0.78 / 2.98 | 0.36 / 2.08 | 0.12 / **0.06** |

- **Bank 4 holds the story's environment and objects.**
  - Concrete nouns and adjectives get lambda of about 0 (kept forever, flat over distance) and high priors.
  - Its lowest-lambda tokens are all objects and properties: orange, purple, ancient, pebble, chain, tiger, pillow, melon (lambda 0.00).
  - Names and punctuation are muted (priors 0.03-0.06).
  - In the stepper it lights up only "forest, rhinoceros, rocks, icy hill" across the whole window: a persistent scene description.
  - This is the "token acts like a constant weight" case from `studies/token_coefficient_kernel.html`, discovered by the model.
- **Bank 2 tracks the dialogue and structure regime.**
  - Quotes, `?"`, newlines, "named", "asked" and "saw" carry the highest priors, with moderate decay (lambda 0.6-0.85).
  - It keeps recent structural markers, i.e. whether a quotation or question is open.
  - This is the role an RNN's forget and input gates usually learn: tracking how long ago a regime started.
- **Bank 3 holds sentence structure and the cast.**
  - Openers (Its, Instead, Their, Sometimes), quotes and names (Lily 2.01, Billy 1.64) get high priors with long retention (lambda 0.1-0.4).
- **Bank 1 is a fast, recent context** (lambda about 1 for most tokens).
- **The router routes by context.**
  - Example: at a dialogue boundary (`?"\n\n`) the router gives bank 3 67.6% and bank 2 22.0%, and bank 4 only 0.3%.
  - The softmax head works as a context-feature filter that chooses which summary to pass on.
- Earlier, without memories (`rope-token-banks-4`), the same split appeared: one bank for dialogue and structure, one for verbs and adjectives, one for names.
- The gains, initialized at 30, shrank to 0.4-1.4 once the RMSNorms existed: scale is now set by the norms.

**k-means over (lambda, prior).** Script: `model scripts token-banks-clusters --run-name rope-token-banks-4-mem4-norm --k 8`. It writes `reports/token_banks_clusters.html`.

Setup:
- 9,201 tokens seen at least 100 times in training.
- 8 features per token: lambda and log prior in each bank, z-scored.
- k-means++ with 20 restarts, all in torch on the CPU.

The silhouette is low for every k (0.17 at k = 4, about 0.14 at k = 8-16): the parameters form a continuum, not separate islands. The clusters still line up with word types, as table rows of lambda / prior / retention. Retention is `|w(64^lambda)| / |w(1)|`: how much weight a token keeps 64 tokens later.

| Cluster | Share of text | Content | Bank 1 | Bank 3 | Bank 4 |
|---|---|---|---|---|---|
| 1 | 80.0% | function words, punctuation, pronouns, dialogue verbs (`. and \n , the to a was it her She said "`) | 0.81 / 1.48 | 0.31 / 1.22 / 0.05 | 0.35 / 0.59 / 0.01 |
| 2 | 8.6% | common adjectives, people, places (little, big, girl, friends, park, dog, mommy, dad, Max) | 0.91 / 1.03 | 0.37 / 0.67 | 0.11 / 0.72 / **0.39** |
| 3 | 4.7% | everyday setting nouns and verbs (boy, bird, tree, house, water, room, sky, kitchen) | 0.70 / 1.15 | 0.61 / 0.43 | 0.17 / 0.80 / 0.07 |
| 4 | 3.7% | **concrete objects and animals** (car, toy, ball, bear, fish, truck, rabbit, cake, boat, butterfly; typical: anchor, tornado, oyster, valley) | 0.96 / 0.51 | 0.69 / 0.60 | **0.03 / 1.86 / 0.89** |
| 5 | 1.4% | **names** (Daisy, Tommy, Billy, Amy, Jill, Mary, Emma, Fluffy, Mittens) | 0.97 / 0.41 | **0.55 / 1.19** | 0.33 / 0.23 / 0.01 |
| 6-8 | 1.5% | rare subword pieces, `<eos>`, uncommon verbs | priors 0.16-0.93 | priors 0.16-0.33 | priors 0.21-0.56 |

Reading:
- **Bank 4 filters topic words.** The more concrete the noun, the flatter and stronger its weight: objects have retention 0.89 (cluster 4), people and places 0.39 (cluster 2), setting words 0.07 (cluster 3).
  - Bank 4 mutes grammar (cluster 1, prior 0.59) and names (cluster 5, prior 0.23).
  - Through these tokens, bank 4 holds a persistent "what is this story about" vector over the whole window. The scene and its objects stay present long after they were mentioned.
- **Names are a separate cluster with a separate home.** Bank 3 gives them a high prior (1.19) with medium retention (lambda 0.55), while bank 4 mutes them. The cast is tracked apart from the scene, and memory 1 holding "Billy" adds a long-range copy.
- **Grammar goes to the fast banks.** Cluster 1 (80% of all text) has the highest priors in banks 1-3 with fast-to-moderate decay. It is the local syntactic context.
- **Rare pieces are actively muted.** Their priors started at 1 and fell to 0.16-0.4 in every bank. This isn't weight decay, which is off for priors.
- **On a story** (Roxy and Billy), the coloring separates the scene (forest, rhinoceros, rocks, icy hill: cluster 4), the cast (Roxy, Billy: cluster 5), the setting and actions (climb, trees, leaves, feet: clusters 2-3) and the grammar (cluster 1).

**Contrast with SSMs and gated linear attention.**
- In Mamba, GLA and Gated DeltaNet, the decay at step t is computed from the token *arriving* at t and applied to the *whole state*. How long an old item survives depends on everything that came after it.
- Here each stored token carries its own lambda: a token's retention depends only on its identity and its distance. The writer decides what to keep forever and what to keep for a few steps.
- Mathematically it is still a weighted sum (a convolution per token type). The difference is who controls retention.

### The final MLP holds the "if-else" priors

- After "a bird named", the gen64-mlp model predicts bird-like names (Tweet, Chirpy). In other contexts it predicts common boy or girl names.
- It is a context-conditioned prior over the vocabulary, ordered roughly by dataset frequency: "in this kind of context the next token is a name, of this kind".
- Analysis: `model/scripts/token_banks_stepper.py` (votes panel, copy table).

### Memories act as persistent slots

Stepper: `model scripts token-banks-stepper --run-name rope-token-banks-4-mem4-norm`. It writes `runs/rope-token-banks-4-mem4-norm/reports/token_banks_stepper.html` (add `#260` to the URL for step 260).

Story: Roxy the rhinoceros and Billy the bird, steps 250-270.
- **Memory 1 holds Billy.**
  - Its read comes from the write at "a little bird named Billy" (105-125 positions back, 12-14% share) on every step.
  - Its gate fires at the "Billy" mentions.
  - It is a long-range entity slot.
- **Memory 2 holds "climb"** (about 70 back), the story's main verb.
- **Memory 3 holds the earlier problem part** ("sad / about", 110-125 back).
- **Memory 4 tracks the recent clause** ("From / played / and", 1-12 back), with the largest shares (14-21%).
- **Slots are sticky.** The source write stays fixed and drifts one step further back each step. The memories behave like slots, not like a fresh search each step.
- **Effect measured by mean ablation** (a read replaced by its average over the story): all four reads together add **+1.07 nats per token** to the target's log probability, averaged over the story. At step 260 ("From that" -> "day") they add +4.8 nats.
- **The decoded content is a mixture.** The binder's value kernel stores a weighted mix of the 16 tokens before the write, so memory 1's read decodes to "falling / for" while its source is the Billy write.

### How it learns to write for later without BPTT

- The memory is not a recurrent state. Each read is an explicit sum over the writes in its window, computed in one parallel pass.
- The gradient from the loss at read t reaches each write a in one hop: `d r_t / d bound_a = e^{-i psi_t} / sum g`.
- In the backward pass the cumulative sum becomes a reverse cumulative sum, so each write gets the summed gradient of all future reads that include it. Then:
  - the gate rises when the write helped future reads
  - the write key moves toward the read keys that benefited (through `cos(phi - psi)`), and vice versa
  - the value kernel learns which tokens to store
- The signal exists from step 0 because the unbinder is initialized as a copy of the binder's key network, so early reads already retrieve "positions like me".
- This is the same mechanism by which transformers learn long-range use. Here it is compressed into 4 fixed-size vectors (linear attention with a complex phasor kernel).
- The limit is the window: a write older than 128 positions can be neither read nor trained.

## Engineering findings

- **Speed** (batch 128, context 255, forward + backward):
  - 3.42 s -> 2.15 s from two exact rewrites, outputs and gradients identical to float precision:
    - the memory's 16-lag value loop became an absolute-RoPE-frame block-banded matmul (601 -> 99 ms for that piece)
    - all band sums now use blocks of the window width (`_band_matmul`)
  - Earlier, for the banks:
    - reading the token weight table along its first axis avoids a 30x-slow backward of `index_select` along axis 1 on the XPU
    - a flip + pad-reshape skew replaced gathers
- **Stability:**
  - The fresh 640-wide final block exploded at lr 0.01 (output norm 0 -> 275 in two Adam steps).
  - Fixed with linear warmup (200 steps) and global gradient clipping at 1.0: `training.warmup_steps`, `training.global_clipnorm`.
  - `LinearWarmup` lives in `model/schedules.py` so checkpoints load outside training.
- **Scale:**
  - One RMSNorm over `[h ; reads]` would have hidden the reads: at init `h` had RMS 0.49 and the reads 0.008.
  - `h` and each read now have their own RMSNorm.
- **Memory normalization:** the memory is a gate-weighted average (`/ sum g`), bounded by the largest value however many gates are open.
- **XPU quirks:**
  - `take` backward and `topk` hit "work-items ... cannot exceed 512" at tiny batches or non-contiguous inputs.
  - Keras `RMSNormalization` backward fails above about 10k rows (hence `model/archs/_norm.py`).

## The delta rule: next step for write-once robustness

Today's memory is additive.
- Every gated write is added, and the read is a gate-weighted average.
- A name written once (Billy) keeps only a share of the read (12-14%) that shrinks as other gates open.
- It disappears after 128 positions.

The **delta rule** (Widrow-Hoff, used by DeltaNet and Gated DeltaNet) writes only the error:

```
v_hat = read(M, k)                 # what the memory already returns for this key
M    <- M + beta * write(k, v - v_hat)
```

With an outer-product matrix memory `M` (D x D) and unit keys, `beta = 1` makes `M k = v` exactly after the write:
- re-writing the same key **overwrites** instead of piling up
- writes with unrelated (near-orthogonal) keys barely interfere

**Why it helps write-once information:**
1. **No dilution.** A slot written once keeps its value until a write with a similar key replaces it. Later writes elsewhere don't shrink it, as the `/ sum g` average does now.
2. **No duplicate accumulation.** Repeating "Billy" doesn't inflate that slot, and a changed fact replaces the old one.
3. **No hard window.** Content persists until overwritten (or decays, with a gate as in Gated DeltaNet), so a name from 500 tokens ago is reachable.
4. **More capacity with a matrix state.** A D x D matrix stores up to about D near-orthogonal pairs per memory. The current 128-dim phasor vector holds about 8-16 well-separated pairs before cross-talk.

**Costs:**
- The update depends on the previous M, so it is truly recurrent.
- It can still be trained in parallel with the chunked WY / UT-transform algorithm (Yang et al., 2024, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length").
- That is more complex than a cumulative sum, and the state is D x D per memory (4 x 128 x 128 here).
- A phasor-vector version (`M += beta * e^{i phi} ⊙ (v - conj(e^{i phi}) ⊙ M)`) is cheap but only cancels the error along that key's own phases, so cross-talk remains.

**Test plan:**
- Add a delta-rule variant to `studies/assoc_memory_synthetic.py` and compare write-once recall with distractors against the current gated average.
- Measure recall as the number of writes between storage and query grows.

## Open experiments

1. **Window-255 baseline without memories** (`gen64-mlp` copy at window 255). It separates the context gain from the memory gain.
2. **The same model with `memory_heads` 0 at context 255.** It isolates the memories directly.
3. **Delta-rule memory** (above), synthetic first, then in the model.
4. **Exclude the current position's own write from the read,** to see whether the memories carry only the past.
5. **Shrink the final block's input** (sum or project the reads), worth about -250 ms/step.
6. **Longer training samples** to cut the 1.5x state overhead of dense supervision at context 255.

## Files

- Arch: `model/archs/rope_token_banks.py`. Memory: `_build_memory`, `_memory_forward`, `memory_trace`. Fast band sums: `_band_matmul`.
- Tests: `tests/test_dense_supervision.py` (`TestTokenBanksMemory`).
- Clusters: `model/scripts/token_banks_clusters.py` (k-means of per-token lambda / prior, torch).
- Report: `model/scripts/token_banks_stepper.py`, which shows banks, votes, the unbinding timeline, per-memory writes / gates / key match / stored kernels, and copy.
- Training: `model/train.py` (warmup, clipping), `model/schedules.py`.
- Related studies:
  - `studies/token_coefficient_kernel.html` (token prior and lambda design)
  - `studies/selective-context-token-kernel.md`
  - `studies/hyperdimensional_binding.html` (binding demo)
  - `studies/assoc_memory_synthetic.py` (free keys: recall 1.00 vs 0.05 local, 0.20 bank + MLP)

## Wave field: one mechanism instead of banks + RoPE (`wave-field`)

Idea (`studies/wave_field_tokens.html`, `studies/resonance_selection.html`): every token broadcasts damped
waves on K shared stations (frequencies), split over L decay classes by its own mix (its propagation
length); the field at a position is the windowed sum of everything broadcast before it; each token reads it
through its own receiver filter, giving a contextualized feature. No banks, no RoPE (distance is the wave
phase); with shared decay classes the field is a running sum (constant-size state at inference).

Arch: `model/archs/wave_field.py` (emission table V x 2K, class mix V x L, learnable decays and frequencies,
receiver g = 1 + cW, features [Re, Im, |.|] of g F -> RMSNorm -> linear base + residual GELU block ->
readout). Exact dense supervision via the block-banded matmul (tests: dense == single-target).

Run `wave-field-64x4` (K = 64 stations, L = 4 classes, MLP 1024; batch 512, 10 x 50 steps, lr 0.01, warmup
200; 3.63M parameters, ~2.0 s/step):

| run (10 epochs) | full eval loss | top-1 | params |
|---|---|---|---|
| `rope-token-banks-4-gen64-mlp` | 2.6044 | 43.0% | 3.17M |
| `rope-token-banks-4-gen64-mlp-proj` | 2.5390 | 43.9% | 3.25M |
| **`wave-field-64x4`** | **2.5096** | **44.9%** | 3.63M |
| `rope-token-banks-4-gen64-mlp-proj-catjoint` (warmup) | 2.4233 | 46.2% | 3.51M |

Val curve: 4.107 3.345 3.023 3.033 2.753 2.662 2.593 2.552 2.533 2.496. The single wave-field mechanism
beats the banks with projections and lands 0.086 behind the best banks variant at equal training, and it
generates coherent stories. Next: per-station class mixes, more stations/classes, the general_concat-style
joint input, and checking the constant-state generation path.

### Cascade read (`wave-field-cascade`)

Same training, plus three options (4.10M parameters):
- `class_projections`: each decay class writes its own 128x128 map of the signature (identity init);
- `clock_decay`: one decay rate per clock (4 x 64) instead of per class;
- `cascade_filter`: the read goes short range -> long range. A query starts as the reader's signature, sets the
  filter of class 1 (g = 1 + head(norm q)), what class 1 hears updates the query (q += MLP([norm q ; norm heard])),
  which sets the filter of class 2, and so on. All 4 heard classes go to the readout (768 features).

| run (10 epochs) | full eval loss | top-1 | params |
|---|---|---|---|
| `wave-field-64x4` | 2.5096 | 44.9% | 3.63M |
| `rope-token-banks-4-gen64-mlp-proj-catjoint` | 2.4233 | 46.2% | 3.51M |
| **`wave-field-cascade`** | **2.4094** | **46.8%** | 4.10M |

Val curve: 4.046 3.283 2.958 2.850 2.678 2.588 2.496 2.471 2.425 2.396. First wave-field run to beat the best
banks variant (with ~0.6M more parameters than it; an ablation of the three options is still to do). Learned
class half-lives (mean rate per class): 1.1 / 3.1 / 8.7 / 38.8 tokens, much shorter than the per-class init
(1 .. 64) for the slow class.

Stepper (`model scripts wave-field-stepper`, Cascade tab), story step "how -> she": stage 1 hears almost only the
reader (95%); stage 2 hears "and how"; stage 3's filter turns to the characters ("Billy", "told", "asked",
"Roxy?"); stage 4 hears the topic words of the whole story ("icy hill", "climb", "rocks", "cold"). The stages
separate by range and by role, as the cascade intends. After stage 1 the query is no longer close to any single
token signature (cosine ~0.3).

Lineage: the wave field is a diagonal complex linear recurrence (S4D / LRU), multi-scale decays as in RetNet,
rotation = RoPE/xPos, token-dependent write/read gating as in Mamba / gated linear attention. The cascade of
read filters conditioned on shorter ranges has no known exact precedent (not searched).
