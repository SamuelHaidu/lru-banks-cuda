# Mid-gate and hypernetwork-token-transform: what's actually happening

Two new `dual-sphere` variants, both outperforming plain `dual-sphere`
(`baseline`, GELU, `hidden_dim=1024`) at comparable training progress:

| run | arch | epoch | train loss | val loss |
|---|---|---|---|---|
| `baseline` | `dual-sphere` | 31 | 3.4319 | 3.0981 |
| `mid-gate-baseline` | `dual-sphere-mid-gate` | 21 | 3.4451 | 3.1150 |
| `hyper-token-transform-residual-baseline` | `dual-sphere-hyper-token-transform` | 9 | 3.6235 | 3.2880 |

(Epochs are not matched across runs -- these are snapshots taken when each
run happened to be inspected, not a controlled comparison. Directionally,
both new architectures track close to or ahead of baseline's pace, worth
re-checking once all three sit at the same epoch count.)

## `dual-sphere-mid-gate`: what the gate does

Architecture (`model/archs/dual_sphere_mid_gate.py`): identical to
`dual-sphere` except a sigmoid gate is computed from `dense1`'s output in
parallel with `dense2`, then multiplies `dense2`'s output before `dense3`:

```
h1 = dense1(u)                    # hidden_dim, GELU
h2 = dense2(h1)                   # hidden_dim // 2, GELU -- reads h1 directly, unaffected by the gate
gate = sigmoid(gate_dense(h1))    # hidden_dim // 2, sigmoid, bias-free -- reads h1, parallel to dense2
h2_gated = h2 * gate
z = dense3(h2_gated)
```

Report: `runs/mid-gate-baseline/reports/dense_layers_token_activations.html`
(`dense1`, `gate_dense`, `dense2`, `dense3`, self-generated text).

**Observation (qualitative, not yet statistically confirmed):** the
context-length/exponential-decay-like activation pattern originally found in
plain `dual-sphere`'s `dense1` (see the earlier per-token activation
reports) is still visible in `mid-gate-baseline`'s `dense1`, but weaker/
noisier than in the un-gated baseline. The new `gate_dense` layer's own
activations look more like on/off selection than a smooth decay curve.

**Working hypothesis:** once a second learned pathway (`gate_dense`) exists
between `dense1` and the readout, it partially absorbs the role `dense1`
alone had to play in the plain baseline, redistributing "how much context
has accumulated" tracking away from a single smooth curve and toward a more
binary, content-conditional switch. This has NOT been verified with the
same rigor as the modular-arithmetic study (no curve fitting, no
permutation-null significance test, no check that `gate_dense`'s activations
are actually bimodal rather than just "gating" by eye). Treat as an
observation to follow up, not a finding.

## `dual-sphere-hyper-token-transform`: what the hypernetwork does

Architecture (`model/archs/dual_sphere_hyper_token_transform.py`): the
window's flattened, L2-normalized token embeddings (`flat`, 1024-dim at
defaults) generate the WEIGHTS of a tiny per-position FFN (a hypernetwork),
which is then applied identically to every position's own
`concat(prev_token_emb, this_token_emb, pos_emb)` input (24-dim), producing
an 8-dim output per position. Those 128 outputs are reassembled to 1024-dim,
added residually to `flat`, normalized, then fed into `dual-sphere`'s
unchanged `dense1/dense2/dense3`.

```
flat -> hyper_w1_proj -> reshape (24,24) -> W1   (called ONCE per forward pass, on the WHOLE window)
flat -> hyper_w2_proj -> reshape (24,24) -> W2
flat -> hyper_w3_proj -> reshape (24,8)  -> W3
per position t: GELU(input_t @ W1) -> GELU(· @ W2) -> · @ W3 = transformed[t]   (8-dim, SAME W1/W2/W3 for every t)
u = normalize(flat + reshape(transformed))
... dense1/dense2/dense3 unchanged from dual-sphere ...
```

**Two genuinely distinct things were being confused earlier in this
investigation, now disambiguated:**

1. **The generated WEIGHTS** (`hyper_w1_proj`/`hyper_w2_proj`/`hyper_w3_proj`'s
   own output, captured via forward hook in
   `dense_layers_token_activations.py --layer-names hyper_w1_proj hyper_w2_proj hyper_w3_proj ...`,
   report at `runs/hyper-token-transform-residual-baseline/reports/dense_layers_token_activations.html`)
   -- this is `context -> linear`, computed once per forward pass from the
   whole window. THIS is where the context-length/decay-like pattern showed
   up clearly across generation steps (more real content in the window as
   generation proceeds -> the generated weight values drift in a
   smooth, roughly monotonic way).
2. **The actual per-token FFN OUTPUT** (`transformed`, NOT a named layer,
   captured by manually replaying the forward pass in
   `hyper_token_transform_output_activations.py` and
   `hyper_token_transform_full_window_analysis.py`) -- this is what the
   generated weights actually COMPUTE when applied to each position's local
   input. This does NOT show the same clean decay pattern; it looks
   noisier/more idiosyncratic per position.

**Quantitative findings from `hyper_token_transform_full_window_analysis.py`**
(one fixed 128-token real training excerpt, `hyper-token-transform-residual-baseline`
@ latest):

- `transformed` is NOT close to identity relative to the raw token
  embedding: per-channel correlation with `token_emb` is weak (0.05-0.31
  across the 8 channels), and MSE-to-embedding-variance ratio is ~49 -- the
  hypernetwork's output is a substantial, non-trivial transform, not a
  small perturbation of the input.
- **Sensitivity test** (does position `t`'s output depend on the WHOLE
  window, or effectively only on `t`'s own local
  `(prev_token, this_token, pos)` input?): swapping one token 80 positions
  away from a probe position and re-running the forward pass shifts the
  probe's own output by a median-normalized ratio of **~0.10** (10% of the
  typical adjacent-position-to-position variation in this window; an
  earlier, buggy version of this measurement used one specific
  adjacent-position pair that happened to sit next to an output-magnitude
  outlier and originally reported ~0.035 -- the corrected, more robust
  median-based baseline gives ~0.10). Either way: real, non-zero, but
  small. The hypernetwork CAN propagate distant context into any given
  position's output (architecturally, since `W1/W2/W3` depend on the whole
  window), but in practice a position's output is dominated by its own
  local input, not by the rest of the window.
- **A handful of positions (roughly the last ~8 of the 128 in the
  inspected excerpt) show output L2 norms 5-20x the median** of the rest of
  the window (up to ~42 vs. a median of ~2). This is NOT explained by the
  positional embedding table (checked directly: `positional_embeddings`'
  per-row norms are unremarkable and roughly flat across all 128 positions,
  0.28-1.12, nothing special near the end). Root cause not yet identified --
  candidates not yet tested: an interaction between the hypernetwork's GELU
  nonlinearities and specific late-window content, or something structural
  in how `prev_token`/`this_token` combine near the window's causal edge.
  Worth checking whether this outlier band moves with window length (i.e.
  is it "near the true end of this document" or "near the true end of the
  128-token window regardless of content") -- not yet tested.

## Open questions / next steps

1. Confirm the mid-gate "gating vs. decay" observation with the same rigor
   used in `modular-arithmetic-findings.md`: curve-fit `gate_dense`'s units
   against exponential/logistic/constant families, and separately check
   whether their distribution is bimodal (consistent with real gating)
   rather than continuously graded.
2. Identify the cause of the late-window output-norm outlier in the
   hypernetwork's `transformed` tensor -- does it track absolute window
   position (positions ~120-127 specifically) or something about this
   particular excerpt's content near its end? Test with a different
   document and see if the same position range spikes.
3. Re-run the epoch-matched loss comparison once all three runs
   (`baseline`, `mid-gate-baseline`, `hyper-token-transform-residual-baseline`)
   have trained the same number of epochs, since the table above is not
   currently a fair comparison.
4. The hypernetwork's per-token output is far from identity (finding above)
   yet only weakly context-sensitive (~10% sensitivity ratio) -- worth
   checking whether that 10% is nonetheless where most of the loss
   improvement over baseline comes from (e.g. an ablation: freeze
   `hyper_w1_proj`/`hyper_w2_proj`/`hyper_w3_proj` to their random-init
   values, so the per-token FFN stops being context-conditional, and see
   how much loss degrades).
