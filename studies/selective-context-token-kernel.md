# Selective context in the token kernel: what it does, where it loses

Run: `rope-token-kernel-osc8` (RoPE, 8-oscillator shared kernel, per-token prior `s`,
forgetting `lambda`, BM25 saturation `kappa`, normalized weighted average; 10 x 50 steps,
batch 512, window 128, dense). Tools: `model report token-kernel-stepper`,
`model report token-kernel-report`, `studies/selective_context_check.py`.

## Observations (stepper, story "Roxy the rhinoceros")

- **Selective context works, and normalization is what makes it work.** After
  `...she wanted to climb it.\n\nRoxy tried` almost all the energy sits on `tried`: the
  next token (`to`) only needs a Markov-1 context, and the model switches the rest off.
  One step later the energy is shared by `tried` + `to`: "here we need a Markov-2". The
  other kernels (free, shared, oscillators) cannot do this: without normalization and
  per-token numbers they have no way to turn anything off.
- **It also does full-context retrieval** when it can, e.g. `...was sad and asked, "Why
  are you sad,"` spreads energy over the whole sentence.
- **Names are kept longer than function words**: `Billy` lambda 0.56 vs ` to` 0.98, so it
  learned that names must last longer.
- **The failure: recent tokens take everything.** In `tried to ...` the recent pair holds the
  state, so the model cannot ask "what is the verb in the context?". The answer, `climb`,
  was 9 positions back; the model gave it 8.96%, too many bits. The loss always prefers
  recency, and one state has to serve both "what just happened" and "what matters in the
  story".

## Measured on 200 validation stories (48,687 steps)

`studies/selective_context_check.py`, against `abl-exp-osc8` (same shared kernel, no
per-token numbers, same budget):

| | token kernel | baseline |
|---|---|---|
| loss, all steps | 2.769 | 2.721 |
| loss when energy at k <= 2 > 50% (2% of steps) | **0.577** | 0.661 |
| loss, other 98% of steps | 2.811 | 2.761 |

- The state is local but not extreme: median 15.8 effective tokens, median top-1 share
  19%; strong locking on the last two tokens (> 50%) happens in 2% of steps, and there the
  selective model **beats** the baseline (0.58 vs 0.66). The `tried` case is real, but rare.
- **Copy targets** (the next token is a content word already in the window, k >= 3) are 13%
  of steps, and that is where the selective model loses most:

| latest occurrence at | n | energy on it | p(target) model | p(target) baseline | loss gap |
|---|---|---|---|---|---|
| k 3-8 | 498 | 4.80% | 0.142 | 0.149 | +0.151 |
| k 9-32 | 2654 | 2.19% | 0.128 | 0.133 | +0.044 |
| k 33-128 | 3193 | 0.81% | 0.091 | 0.108 | +0.119 |

- The more the state is concentrated on the last two tokens, the less energy reaches the
  earlier occurrence and the bigger the loss gap on copy targets:

| copy targets, by energy at k <= 2 (tertiles) | energy on target | loss gap vs baseline |
|---|---|---|
| low | 1.95% | +0.066 |
| mid | 1.72% | +0.087 |
| high | 1.43% | +0.118 |

So the hypothesis holds: recency crowds out the context a copy target needs. Copy targets
explain only part of the overall gap (13% of steps x ~0.09 nats); non-copy steps are also
+0.04 worse, which the two-state design would not address by itself.

## Proposed next step: two states and a selection MLP

- **State A, "focus":** what we have now: per-token `s_A`, `lambda_A`, normalized. Free to
  lock onto `tried to`.
- **State B, "memory":** its own `s_B`, `lambda_B` (initialized long-lived, e.g. lambda 0.3),
  normalized, so it stores the story's important tokens (names, objects, verbs) regardless
  of what A is doing. Both are parallel banded sums.
- **Selection from A:** an MLP reads A (`tried to`) and decides what to pull from B:
  - a per-feature gate, `h = A + sigmoid(MLP(A)) * B`: cheap, still parallel and linear-time;
  - or a query, `q = MLP(A)`, with retrieval weights `proportional to s_B,j * w(k^lambda_B,j) * exp(q . e_j)`:
    attention with the learned prior and forgetting as the score bias. Most expressive
    ("call the verb"), but costs a T x T score per sample.
- Separate parameters let B specialize: its contribution only pays off through the gate, so
  the loss's preference for recency no longer forces B to be recent.
