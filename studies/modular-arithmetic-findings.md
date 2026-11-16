# Modular arithmetic (`modular-arithmetic-97`) findings

Scripts: `ad hoc scripts/modarith_operation_accuracy.py`,
`ad hoc scripts/modarith_addition_mechanism.py`,
`ad hoc scripts/modarith_addition_mechanism_over_time.py`,
`ad hoc scripts/modarith_multiplication_mechanism.py`. Runs:
`modarith-baseline` (lr=0.001, batch=512, trained slowly),
`modarith-baseline-v2` (same config except lr=0.1 fixed for the entire run,
batch=16384 -- see below).

## Confirmed findings (measured directly)

1. **Addition/subtraction mod 97 is learned via a Fourier-circle
   mechanism**, in the same spirit as Nanda et al.'s modular-addition
   grokking analysis, despite operands being fed in as separate digit
   tokens rather than a single embedded residue.
   - The query vector `q` (final normalized FFN output, dot-producted
     against the target table for logits) collapses onto `(a+b) mod 97`
     alone: variance within an anti-diagonal (`a+b` fixed) is ~40% of the
     variance across the full `(a,b)` grid.
   - 80-100% of hidden units (dense1, dense2, and query) show >50% of their
     residue-wise variance concentrated at a single FFT frequency, mostly
     k=1 and k=2 -- the same small-frequency-basis signature the paper
     describes.
   - Measured on `modarith-baseline` @ epoch 25. See
     `runs/modarith-baseline/reports/addition_mechanism.html`.

2. **The mechanism appears as a sharp phase transition, not gradual
   learning** (`modarith-baseline`, lr=0.001): chance-level accuracy
   (~10% r0, collapse ratio ~0.99) through epoch 7, then in a single epoch
   (7->8) the collapse ratio drops to 0.80, single-frequency unit fraction
   jumps from ~20% to ~99% simultaneously across all three layers, and r0
   accuracy jumps from 10.9% to 28.9%. Epochs 8-12 rapidly consolidate to
   >93% accuracy; the remainder of training refines precision rather than
   discovering new structure. See
   `runs/modarith-baseline/reports/addition_mechanism_over_time.html`.

3. **`+`/`-` grok far faster under a 100x learning rate**
   (`modarith-baseline-v2`, lr=0.1 vs. 0.001, otherwise identical config):
   `+` reaches 82% exact-match and `-` reaches 80% within epoch 1 (1000
   steps), ~99% by epoch 2 -- the whole plateau-then-transition pattern
   from finding #2 compresses into a fraction of one epoch instead of
   spanning ~7 epochs.

4. **Multiplication mod 97 does not show the same rapid transition.**
   Under lr=0.001 it plateaus near chance (~23% exact-match after 25
   epochs, vs. 99.8%+ for `+`/`-`). Under lr=0.1 (`modarith-baseline-v2`)
   it climbs slowly and monotonically the whole time it's been trained so
   far: 1.1% (ep1) -> 15.1% (ep3) -> 52-64% (ep5-6) -> 71% (ep7) -> 75%
   (ep9) `equation_exact_match_*`, with per-token r0 accuracy 32%(ep3) ->
   85%(ep9) (see `modarith_multiplication_mechanism.py`'s `mul_r0_acc`
   below). No sudden jump has been observed at any point -- this looks
   like continuous capacity reallocation, not a phase transition (see #6).

5. **Multiplication does NOT use a discrete-log/Fourier mechanism --
   this refutes the "architecture forces Fourier, multiplication needs
   discrete-log" hypothesis from an earlier revision of this document.**
   Measured across `modarith-baseline-v2` epochs 1-9 by re-indexing hidden
   units and the query by `(log_5(a) + log_5(b)) mod 96` (5 is a primitive
   root of 97, so this reindexing turns multiplication into addition if the
   network has found an equivalent trick):
   - Collapse ratio onto the discrete-log index stays at 0.85-0.99 the
     entire time (compare: addition's collapse onto `(a+b) mod 97` is
     0.29-0.62) -- essentially no collapse, i.e. the query does NOT become
     a function of `log(a)+log(b)` alone.
   - The fraction of hidden units (dense1, dense2, query) with a single
     dominant FFT frequency over the discrete-log index is ~0.000 at
     *every* epoch measured -- literally no unit shows the periodic
     signature that a discrete-log/Fourier mechanism would produce.
   - A weak, transient Fourier-like signature DOES appear in the *naive*
     `a*b mod 97` linear index (`mul_naive_collapse` query-clean-fraction:
     0.73 @ ep3), but it decays monotonically as accuracy rises (0.62 ->
     0.40 -> 0.22 -> 0.16 -> 0.08 -> 0.06 by ep9) -- i.e. whatever the model
     is actually using to solve multiplication is becoming LESS visible to
     a 1-D circular-index probe as it gets better, the opposite of what
     grokking a periodic mechanism would look like.
   - Conclusion: multiplication's improving accuracy is not explained by
     either the addition-style Fourier mechanism or its natural
     discrete-log analogue. Most likely it is being solved via some other
     representation the current probes don't target -- e.g. digit-level
     partial-product/carry logic operating directly on `a0,a1,b0,b1` -- but
     that has not been verified, see open hypotheses below.

6. **Addition's own mechanism is not stable once multiplication starts
   learning -- it visibly degrades while its behavior stays intact.**
   Same `modarith-baseline-v2` epochs 1-9, revisiting the addition probe:
   `add_r0_acc` stays ~99% throughout (behavior is fine), but the query's
   fraction of single-frequency units drops from 0.99 (ep1) to 0.38-0.62
   (ep8-9), and the collapse ratio onto `(a+b) mod 97` drifts UP (gets
   worse) from 0.29 (ep1) to 0.54-0.62 (ep7-9) -- the clean Fourier circuit
   addition built early is being partially dismantled/repurposed as the
   shared hidden layer (512+256 units, used by both operators in this mixed
   dataset) gets pulled toward also supporting multiplication. This is
   consistent with #4 being continuous capacity renegotiation across tasks
   rather than each operator solved by an independent, stable circuit.
   See `runs/modarith-baseline-v2/reports/multiplication_mechanism_over_time.html`.

7. **Digit-level partial products are also refuted as the multiplication
   mechanism.** Tested (`modarith_multiplication_digit_terms.py`,
   `modarith-baseline-v2` @ epoch 10, r0_acc=0.862): collapse ratio and
   query/hidden clean-FFT-fraction against each of `a0`, `a1`, `b0`, `b1`
   alone, each pairwise digit product (`a0*b0`, `a1*b0`, `a0*b1`, `a1*b1`
   mod 97), their cross-term sum `a1*b0+a0*b1 mod 97` (the middle term of
   `a*b = 100*a1*b1 + 10*(a1*b0+a0*b1) + a0*b0`), and `a0+b0`/`a1+b1` mod
   97. Every single-term index gives collapse ratio 0.91-0.99 (i.e.
   essentially no collapse) and clean-fraction ~0.00-0.11 across the board
   -- none of these simple bilinear digit terms is what the query has
   organized around.
   Also re-confirmed: as of epoch 10, addition's original Fourier circuit
   is still functioning (r0_acc=0.992) but continues eroding in purity
   (collapse 0.663, query clean-fraction 0.523, both worse than epoch 1's
   0.291/0.992) -- same circuit, increasingly diluted, not yet replaced.

8. **The visible sawtooth/periodic pattern in individual `dense1` units'
   activation over the `(a, b)` multiplication grid is real, not an
   eyeballing illusion.** Tested with a permutation-null + matched-noise
   control (`modarith_multiplication_pattern_significance.py`,
   `modarith-baseline-v2` @ latest, 3 highest-variance dense1 units): for
   each unit, reshaped its 9409 activations back into the genuine `(97,
   97)` `(a, b)` grid (not the flattened 1D line, which could manufacture
   spurious period-97 artifacts from the plotting layout itself), took a 2D
   FFT, and compared the peak spectral-energy fraction against (a) 200
   shuffles of which `(a, b)` label is attached to each activation value
   (same values, randomized assignment) and (b) Gaussian noise matched to
   the unit's own mean/variance, through the identical pipeline.
   - All three units' real peak energy fraction (0.048-0.121) sits at the
     **100th percentile** of 200 shuffles (whose max was 0.0013-0.0018) --
     50-120x above anything the shuffle ever produced.
   - Matched-variance Gaussian noise scores ~0.001 through the same
     pipeline, 70-100x weaker than the real signal.
   - All three peaks are of the form `(ka=0, kb=k)` -- i.e. these
     particular units are periodic in `b` alone (period ~97/10 ~= 10,
     or ~97/1 = 97-ish depending on the unit) and roughly constant across
     `a`, not a mixed diagonal `(a,b)` pattern.
   - Practical implication: this is a stronger positive signal than
     anything the 1-D collapse-ratio probes found for multiplication so
     far (findings #5, #7), and it points at a *specific new candidate*
     for multiplication's mechanism: something indexed by `b` alone (or
     `b mod 10` / the tens-digit `b1`, given the ~10-ish period), not by
     `a*b mod 97`, `log(a)+log(b)`, or any of the bilinear digit terms
     tried so far. Re-running the collapse-ratio/clean-fraction probes
     indexed by `b mod 10` (and separately by `a mod 10`, `b1` the tens
     digit alone) is the natural next step, not yet done.

9. **A single shared oscillatory mode explains most of the cross-channel
   variation across ALL 512 `dense1` units, for every value of `a`.**
   Multichannel Singular Spectrum Analysis (`modarith_multiplication_mssa.py`,
   `modarith-baseline-v2` @ latest): for each of the 97 `a` rows
   independently, built a per-channel Hankel trajectory matrix (window
   length L=48) over the 512 dense1 channels' length-97 series in `b`,
   stacked them, and took the SVD.
   - The top singular value alone explains 67-81% of total spectral energy
     in every single one of the 97 rows (mean 74.2%), and the top-5 explain
     79-88% (mean 84.2%) -- both bands are tight across all 97
     independently-computed rows, which itself argues against coincidence
     (independent noise slices would show much more row-to-row scatter in
     how much a "leading mode" explains).
   - Combined with finding #8 (2D-FFT significance test, `ka=0` peaks),
     this confirms the periodicity in `b` is a genuine POPULATION-LEVEL
     circuit signature -- most of the 512 channels are phase-locked/scaled
     copies of a small number of shared drivers, not 512 independent
     coincidentally-periodic units.
   - Still open: what exactly that shared driver's frequency/phase encodes
     as a function of `b` (and whether/how it varies with `a`) has not been
     extracted from the leading singular vectors yet -- this establishes
     THAT a shared mode exists, not WHAT it computes.
   See `runs/modarith-baseline-v2/reports/multiplication_mssa_per_row.html`.

## Incident: `modarith-baseline-v2` diverged to NaN at epoch 25

Training loss/val_loss went to `nan` starting epoch 25 (`Epoch 25/100:
loss: nan ... equation_exact_match_*: 0.0205`, collapsing every metric back
toward chance) and stayed NaN through at least epoch 27 before the run was
stopped. **Epoch 24 is the last good state**
(`checkpoints/checkpoint_00024.keras`) -- it's also the epoch quoted
throughout findings #4-9 above (88.3% `*` exact-match, 99.7%/99.8% `+`/`-`).
`model.keras` (what `load_checkpoint(..., "latest")` reads) is now the
corrupted post-divergence state, so **every `ad hoc scripts/modarith_*`
script that targets `modarith-baseline-v2` has been pinned to
`CHECKPOINT = 24` explicitly** rather than `"latest"`. Root cause not yet
investigated (lr=0.1 the whole run is a plausible suspect -- see the open
hypothesis below -- but this hasn't been confirmed against, e.g., a specific
weight/gradient blowing up first).

One concrete side-effect worth recording: at checkpoint 24, `scipy.linalg
.svd`'s default `gesdd` driver (and `scipy.sparse.linalg.svds`/ARPACK) both
fail outright on the per-`a` MSSA trajectory matrix (`LinAlgError: SVD did
not converge` / `ArpackError: Could not build an Arnoldi factorization`) --
a genuine Hankel trajectory matrix has heavily correlated
(overlapping-window) rows, which is numerically pathological for those
drivers. `modarith_multiplication_mssa_waveforms.py` (which needs actual
singular VECTORS to reconstruct waveforms, not just values) works around
this with `scipy.linalg.svd(..., lapack_driver="gesvd")`, the slower
classical driver. `modarith_multiplication_mssa.py`'s energy-only sweep
still uses `svds` for speed since it only ever needed singular values.

## Open hypotheses (NOT yet verified -- reasoning, not measurement)

1. Multiplication's real mechanism is unknown. Ruled out so far: addition's
   Fourier circle (that's addition's own, distinct signature), the
   discrete-log/Fourier analogue (finding #5), and simple bilinear
   digit-partial-product terms (finding #7). Findings #8-9 give a concrete,
   now well-supported lead: a shared, population-level periodic mode in `b`
   alone (period ~10, `ka=0`). Not yet done: extract the leading MSSA
   singular vector's actual shape (is it a clean sinusoid at a specific
   period? does it shift with `a`?) and check whether reconstructing `q`
   from just that leading component recovers most of `r0`'s accuracy --
   which would confirm not just that a shared mode exists but that it is
   causally what the model uses to solve `*`, not a side effect.
   Remaining candidates beyond that: some higher-dimensional or genuinely
   nonlinear interaction a 1-D scalar index can't capture (e.g. a 2-D torus
   over `(a mod k, b mod k)` for small `k`); or no clean interpretable
   mechanism at all (memorization-adjacent, gradually smoothed by weight
   decay -- now less likely given findings #8-9's clean, consistent shared
   structure across all 97 rows, but not fully ruled out).
2. Whether the addition/multiplication interference in finding #6 is a
   genuine shared-capacity trade-off (freeing capacity for `*` measurably
   should degrade `+`'s clean-frequency purity further) or a coincidental
   correlation -- would need an ablation (e.g. train on `+`/`-` only, no
   `*`, and check whether addition's clean-frequency fraction stays near
   1.0 throughout instead of decaying).
3. Whether a non-spherical readout (e.g. plain dense-to-softmax, no cosine
   constraint) still finds a Fourier representation for addition, which
   would show the mechanism isn't specific to this architecture's
   cosine-sphere readout.
4. Whether multiplication, given much more training, eventually undergoes
   its own sharp transition into whatever its real mechanism is, or
   continues the smooth climb indefinitely and plateaus short of 100% --
   now moot for this exact run past epoch 24 given the NaN divergence
   above; would need a fresh run (same config or with a lower/warmed-up lr)
   to keep testing this.
5. What caused the epoch-25 NaN divergence, and whether lr=0.1 fixed for
   the entire run (as opposed to e.g. warmup + decay) is the cause -- not
   yet investigated.

## Finding #10: zero-lag cross-correlation between channels is near-zero
(this REFINES, not contradicts, finding #9)

`modarith_multiplication_dense1_cross_correlation.py`: fixed `b=48`, swept
`a` over 0..96, collected the resulting `(97, 512)` dense1 activations, and
computed the full 512x512 Pearson and Spearman correlation matrices across
those 97 samples (checkpoint 24).

- Mean off-diagonal correlation is ~0 for both (Pearson 0.0093, Spearman
  0.0234), and effectively 0% of the 130,816 channel pairs exceed |r|=0.9
  for either metric (0.0000 for Pearson, 0.0001 for Spearman).
- At first glance this looks like it contradicts finding #9 (a single MSSA
  mode explains 67-81% of cross-channel variance) -- it doesn't. Pearson/
  Spearman at zero lag only detect channels that move together AT THE SAME
  TIME as `a` varies. MSSA's Hankel/windowed trajectory matrix instead
  captures shared structure across a whole window of lags, so two channels
  oscillating at the identical frequency but with a phase offset (e.g. one
  behaving like `cos(k*a)`, another like `sin(k*a)` -- exactly what a
  Fourier-style representation would produce) score ~0 on zero-lag
  correlation while still loading strongly onto the same MSSA mode.
- So findings #9 and #10 together narrow the picture: the 512 channels are
  NOT simple scaled/copied versions of one another (ruled out by #10), but
  can still share a common frequency with channel-specific phase shifts
  (consistent with #9) -- a genuine multi-phase Fourier-like basis, not a
  single "everyone does the same thing" signal.
- Not yet done (at the time): lagged cross-correlation (or phase difference
  at each channel's own dominant FFT frequency) would directly test the
  phase-shifted-common-frequency hypothesis this contrast implies.
See `runs/modarith-baseline-v2/reports/multiplication_dense1_cross_correlation_b48.html`.

## Finding #11: the phase-shifted-common-frequency hypothesis (from #10) is REFUTED

`modarith_multiplication_dense1_phase_correlation.py`: same setup (`b=48`
fixed, `a` swept 0..96, checkpoint 24). Found the population's shared
dominant FFT frequency empirically (k=1, i.e. period ~97 in `a` -- one full
cycle across the whole sweep), extracted every channel's phase angle at
that frequency, and measured the population's Phase-Locking Value (PLV,
`|mean(exp(i*phase))|` -- the correct circular-statistics analogue of
"how correlated/concentrated is this population", since ordinary Pearson
correlation is invalid on angles due to wraparound).

- PLV (unweighted) = 0.024, PLV (magnitude-weighted by how strongly each
  channel actually carries that frequency) = 0.015 -- both consistent with
  512 statistically INDEPENDENT, uniformly random phases (a true
  phase-locked population -- e.g. genuine cos/sin pairs of one shared
  oscillator -- would score well above 0.3-0.5, typically approaching 1.0).
  Phases span the full circle with no visible clustering.
- This refutes the specific hypothesis raised at the end of finding #10:
  the channels are NOT a set of phase-shifted copies of one clean sinusoid
  at a single shared frequency.
- Reconciling with finding #9 (MSSA: one mode explains 67-81% of
  cross-channel trajectory-matrix variance): that result is still valid,
  but it captures a broader notion of "shared structure" than "one sine
  wave, many phases" -- MSSA operates on the full Hankel/windowed
  trajectory, which can pick up a shared NON-sinusoidal waveform shape
  (e.g. a shared sawtooth or other broadband periodic shape smeared across
  many harmonics) as one dominant singular vector, without that shape
  having a single clean FFT frequency or channels being phase-locked at
  any one frequency. This is consistent with the ORIGINAL visual
  observation that started this whole investigation -- the raw `y0`/`y2`/
  `y5` line plots showed sawtooth-like shapes, not clean sinusoids, and a
  sawtooth's harmonic energy is spread across many frequencies rather than
  concentrated at one, which is exactly why single-frequency phase-locking
  fails here even though real, strong shared structure (finding #9) exists.
- Net picture so far: dense1 has real, statistically confirmed
  (findings #8, #9) shared periodic-ish structure across channels tied to
  `b`/`a`, but it is NOT a clean multi-phase Fourier/cos-sin basis the way
  addition's mechanism is (finding #1) -- it looks more like a shared
  broadband/non-sinusoidal waveform. What that waveform's actual shape
  encodes (a genuine algorithm vs. some other structured-but-not-classic
  representation) is still open.
See `runs/modarith-baseline-v2/reports/multiplication_dense1_phase_correlation_b48.html`.

Treat items in this section as hypotheses to test with a follow-up script,
not as established results.
