# Learned per-distance weights decay as a power law, not exponentially

Run analyzed: `rope-logit-sum` (`tinystories-10k`, 3 epochs x 1,000 steps,
batch 512, `dim=512`, `window_size=128`, `corrupt_prob=0.15`).

## Model

A single linear layer with no nonlinearity before the softmax:

    logits = W_out^T  sum_k  a_k  R_k  E[t_{n-k}],   k = 1..128

`E` and `W_out` are shared tables. `R_k` is a RoPE rotation by distance `k`.
`a_k` is one learned scalar per distance, initialized to `1/k`.

## Uniform vs. weighted sum

| variant | val loss, epoch 2 | val loss, epoch 3 | val loss, epoch 4 |
|---|---|---|---|
| uniform sum (`a_k = 1`, fixed) | 4.46 | 4.27 | 4.17 |
| weighted sum (`a_k` learned, 128 params) | 3.38 | 3.23 | |
| `baseline` (`dual-sphere`) | 3.98 | 3.81 | 3.72 |

With the uniform sum, `R_k` preserves norm and the embedding table is shared
across distances, so every context token contributes with the same
magnitude. The model cannot make n-1 louder than n-100. Adding 128 scalars
made the model strictly more expressive (uniform is the special case
`a_k = 1`). That is added capacity and inductive bias, not regularization:
neither run overfits, because train loss stays above val loss.

## What the model learned

| k | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| learned `a_k` | 2.00 | 1.14 | 0.55 | 0.32 | 0.18 | ~0.12 | ~0.07 | ~0.05 |
| init `1/k` | 1.00 | 0.50 | 0.25 | 0.13 | 0.06 | 0.03 | 0.016 | 0.008 |

- All weights are positive and decrease with distance. `a_1 / a_2 = 1.75`.
  `a_1` is about 36x the mean of `a_64..a_128`.
- Half of the total `|a|` falls within k <= 18. Covering 80% takes k <= 69,
  and 90% takes k <= 99. Far tokens are individually weak, but they matter
  in total.
- Fitting `log a_k` against `log k` gives `a_k ~ k^-p`:

  | range | p |
  |---|---|
  | k = 1..32 | 0.84 |
  | k = 1..128 | 0.72 |
  | k = 32..128 | 0.53 (the tail flattens toward a floor of about 0.03 to 0.07) |

- A power law fits much better than an exponential: the log-space RMSE is
  0.188 for the power law and 0.420 for the exponential.
- The embedding and readout tables can absorb any global scale, so only the
  shape of `a_k` is meaningful, not its absolute values. AdamW weight decay
  (0.01) also acts on `a_k`, so the tail is probably biased slightly toward 0.

Plot: `runs/rope-logit-sum/reports/distance_weights.html`.

## Literature: the same shape in trained transformers

**Brändel, Köhler, Rheinbach (2026), "How Token Influence Decays with
Distance: A Green-Function View of Trained Language Models"**
([arXiv:2606.29139](https://arxiv.org/abs/2606.29139)).

The authors measure the gradient norm of a next-token logit with respect to
the input embedding at distance r. They take the median over samples, in
Pythia-14M/70M/160M/410M and Qwen2.5-0.5B. Their findings:

- The profile is "much better described by a power-law-type decay than by an
  exponential alternative".
- The fitted exponent is p of about 0.7 to 0.9, typically 0.8 to 0.9 (for
  example Pythia-14M 0.88, Pythia-410M 0.71, Qwen2.5-0.5B 0.88).
- Randomly initialized models give a flat profile, so the decay is learned.
- The power law survives shuffled text (p of about 0.84).

Our explicit `a_k` is the linear-model analogue of their Jacobian profile.
Our p of 0.84 (k <= 32) and 0.72 (full range) sit inside their range.

**Supporting work:**

- Lin and Tegmark, "Critical Behavior in Physics and Probabilistic Formal
  Languages" ([arXiv:1606.06737](https://arxiv.org/abs/1606.06737)):
  - Mutual information between symbols in text decays roughly as a power law
    with distance.
  - Regular grammars (Markov chains) can only produce exponential decay.
  - Context-free grammars can produce power-law decay.
  - This is disputed for some corpora: Sarkar and Howard,
    [arXiv:1912.07506](https://arxiv.org/abs/1912.07506), report exponential
    decay on Shakespeare and WSJ with their method.
- Khandelwal et al. (ACL 2018), "Sharp Nearby, Fuzzy Far Away: How Neural
  Language Models Use Context"
  ([P18-1027](https://aclanthology.org/P18-1027/)):
  - LSTM language models use about 200 tokens of context.
  - Word order matters only for the most recent ~20 tokens. Older context is
    kept as a coarse, bag-of-words summary.
  - This matches our head/tail split: steep decay up to k of about 20, then a
    flat, weak tail.
- Li et al. (ICLR 2023), "What Makes Convolutional Models Great on Long
  Sequence Modeling?" ([arXiv:2210.09298](https://arxiv.org/abs/2210.09298)):
  - A good global convolution kernel needs weights that decay with distance.
  - SGConv builds this in with multiscale sub-kernels.

## Implications

- A single exponential decay (`lambda^(k-1)`, the ALiBi/LTI-SSM shape)
  either kills the tail or flattens the head. A 2-parameter `c * k^-p` should
  match the 128 free scalars.
- The remaining gap to the attention archs has to come from
  content-dependent interactions. A fixed kernel, of any shape, cannot
  express "New" AND "city" -> "York".
