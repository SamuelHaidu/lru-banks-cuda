# How does the model actually recall a name: retrieval, attention, or the final FFN?

Run analyzed: `transformer-alibi-retriver` (architecture `alibi-retriever-sphere`),
checkpoint `latest` (epoch 200, bpc 0.921, top-1 accuracy 43.8%).

## Motivation

The earlier study (`studies/small-table-as-compression-of-big-table.md`)
established *what* the small and big embedding tables encode: the small
table stores coarse group membership, the big table carries fine-grained,
individually-discriminated identity. This study asks *how* the model gets
from context to a name prediction mechanically -- specifically, whether the
architecture's named mechanism (softmax retrieval against the big table) is
actually doing the retrieving, or whether recall happens some other way.

## Method

`model scripts query-recall-trace --run-name transformer-alibi-retriver`
(`model/scripts/query_recall_trace.py`) runs 20 prompts, each introducing a
name once in context, selected so the *trained checkpoint's own* greedy
next-token prediction is verified to already be that name reappearing (a
genuine in-context recall case, not an assumed target). For each prompt it
replicates the model's forward pass using its own trained sub-layers exactly
(`attn1`, `trans1_mlp1/2`, `attn2`, `trans2_mlp1/2`, `seq_proj`,
`dense1/2/3` -- verified byte-identical to the real layer outputs, max
diff 5.2e-06 from float rounding only) and extracts:

- Block 1's full causal self-attention matrix, per head.
- The 128-D query vector Block 1 produces (the one used for softmax
  retrieval against the big table).
- The softmax retrieval weights over the full 10,000-token vocabulary, their
  entropy (normalized to `[0, 1]` by `log(vocab_size)` -- 1.0 = exactly
  uniform), and the resulting pooled/retrieved vector.
- An ablation: Block 2 and the FFN readout are re-run, unchanged, with the
  real retrieved vector swapped for the vector a perfectly uniform softmax
  would produce, to isolate what the retrieval step itself contributes to
  the final prediction.

Report: `runs/transformer-alibi-retriver/reports/query_recall_trace.html`.

## Finding 1: attention does not "look up" the name

Attention from the final context position (the position whose output feeds
the retrieval query) is dominated by ALiBi's recency bias, not a targeted
look-back at the name. For `"Once there was a kind girl named Emma. Her
teacher said, ""`, attention weight on " Emma" is ~0.024-0.025 -- no higher
than several unrelated tokens near it, and far below the weight on the
immediately preceding punctuation (`,` at 0.124, `"` at 0.201). There is no
sharp, name-specific attention spike in any of the 20 traced prompts.

## Finding 2: the softmax retrieval step has collapsed to near-uniform

This is the central, unambiguous result. Across **all 20 prompts, without
exception**, the retrieval softmax's entropy ratio is **0.996-0.997** (1.0 =
exactly uniform) and its cosine similarity to the vector a perfectly uniform
softmax would produce ("retrieval invariance") is **0.995-0.998**. There is
no prompt among the 20 where retrieval is meaningfully more content-dependent
than any other -- the distribution is uniformly (and almost completely)
flat everywhere it was tested, so there is no "most content-dependent"
subset to single out; the intended selection of top least-uniform prompts
turned up nothing to differentiate, which is itself the finding.

Concretely, for the Emma prompt the top retrieval weight is 0.000245 against
a uniform baseline of 0.0001 -- the single most-favored candidate is only
2.4x the uniform weight, and none of the softmax's nominal "top candidates"
(`dollhouse`, `distracted`, `trampoline`, `fingers`, `own`, ...) are
name-like at all. The retrieved vector for the Emma prompt has cosine
similarity 0.9998 with the retrieved vector for a completely unrelated
prompt ("There was a little boy named Tom...''), and 0.9979 with the plain
mean of the entire (L2-normalized) big table.

**The softmax retrieval mechanism, at this checkpoint, is not retrieving
anything query-specific. It has learned to output an almost-constant
"average token" vector regardless of the query.**

## Finding 3: recall survives the retrieval step being replaced by noise-free uniform averaging

The ablation isolates what retrieval actually contributes. Replacing the
real retrieved vector with the *exact* uniform-softmax vector, holding every
other trained weight fixed, and re-running Block 2 + the FFN readout:

| name | entropy ratio | rank, real retrieval | rank, uniform-substituted | top-1 unaffected? |
|---|---|---|---|---|
| Lily | 0.996 | 1 | 1 | yes |
| Tom | 0.996 | 1 | 2 | no (small shift) |
| Emma | 0.996 | 1 | 3 | no (small shift) |
| Timmy | 0.996 | 1 | 1 | yes |
| Sarah | 0.996 | 1 | 2 | no (small shift) |
| Jane | 0.997 | 2 | 2 | n/a (not top-1 either way) |
| Max | 0.997 | 1 | 1 | yes |
| Amy | 0.997 | 1 | 1 | yes |
| Kate | 0.997 | 107 | 129 | n/a (weak in both) |
| Maria | 0.997 | 88 | 82 | n/a (weak in both) |

(full 20-prompt table in the HTML report). Across all 20 prompts, the
target name's rank changes by at most a handful of positions when the
retrieval step's actual content is deleted and replaced by pure
vocabulary-averaging; **4 of 20 prompts keep the exact same top-1 prediction
with zero query-specific retrieval content at all**, and none of the 20
collapse to a dramatically different, name-unrelated top prediction as a
direct result of the ablation (the model still outputs a name, or the same
generic function word "Let" it already favored, in nearly every case).

## Finding 4: the collapse is not specific to names or hand-crafted prompts

The 20 prompts above were all hand-crafted name-recall cases. To check
whether the retrieval collapse is an artifact of that (e.g. names being
poorly represented, as the earlier compression study found), a second script,
`model scripts retrieval-behavior-trace --run-name transformer-alibi-retriver`
(`model/scripts/retrieval_behavior_trace.py`), samples real
`(context, target)` pairs directly from the training corpus using the exact
same random-document, random-cut-point scheme `VariablePrefixDataset` uses
for training -- no hand-crafted text at all -- across three context-length
buckets (short: 1-8 tokens, medium: 9-32, long: 33-128 tokens), split into
cases the model gets exactly right (target is its own top-1 prediction) and
cases it gets badly wrong (target falls outside its own top-10).

The result is the same regardless of length or correctness. Across hit cases
whose targets are ordinary words -- "a", "Once", "They", "he", "other",
"his", "was", ".", "the", "to", "," -- entropy ratio is **0.996-0.997** in
every single length bucket (short/medium/long alike). Miss cases (targets
like "Jill", "boat", "light", "Lisa" that the model ranks 28th-255th) show
**the identical entropy ratio, 0.996**, indistinguishable from the hits.
Aggregated across all sampled real-data cases: mean entropy ratio 0.997,
mean retrieval invariance 0.996 -- matching Finding 2's name-recall numbers
almost exactly.

**The retrieval collapse is not a name-specific or prompt-specific artifact.
It holds uniformly across context lengths, token categories (names, common
nouns, function words, punctuation), and both success and failure cases.**
If retrieval were doing real, content-dependent work anywhere in this
checkpoint's behavior, the miss cases -- where the model is visibly
struggling -- would be the most likely place to see a sharper, more
concentrated softmax attempting (and failing) to find the right content.
Instead they look exactly like the hits: uniform.

## Finding 5: a learned temperature explains the collapse, and the loss cost of corrupting retrieval is small -- as long as the corruption stays in-distribution

The retrieval softmax's temperature, `retriever_tau`, is a learnable scalar
initialized at 10.0 and trained down to **3.26** at this checkpoint. Since
both tables are L2-normalized, cosine similarities are bounded in `[-1, 1]`;
at `retriever_tau = 3.26`, even a fairly large similarity gap of 0.1 only
produces a `exp(0.1 * 3.26) ~ 1.39x` weight ratio -- nowhere near enough to
concentrate probability mass across a 10,000-way softmax. The model appears
to have actively learned *toward* a smoother retrieval softmax, not a
sharper one (contrast the final cosine-readout's own temperature, `tau`,
which trained to **21.95** -- the stage that does discriminate sharply is
the final readout, not retrieval). This is a plausible mechanistic cause of
the near-uniform entropy ratio measured in Findings 2 and 4.

`model scripts retrieval-ablation-loss --run-name transformer-alibi-retriver`
makes the model-level, causal version of this claim: it re-scores 5,120
real validation examples (20 batches of 256), replicating the exact trained
forward pass, under four conditions for the retrieved vector spliced into
Block 2, with every other weight held fixed:

| condition | what it is | loss | &Delta; loss | &Delta; loss % | top-1 accuracy |
|---|---|---|---|---|---|
| normal | the real forward pass | 2.6150 | -- | -- | 43.65% |
| uniform | exact uniform-softmax vector (in-distribution, query-independent) | 2.6282 | +0.0133 | +0.51% | 43.63% |
| shuffled | another example's real retrieved vector (in-distribution, wrong identity) | 2.6337 | +0.0188 | +0.72% | 43.36% |
| random | Gaussian noise (off-distribution) | 5.0807 | +2.4657 | +94.29% | 18.50% |

The `uniform` vs. `shuffled` comparison is the decisive one. Both discard
the query-specific retrieved content -- `uniform` replaces it with no
content at all (the global average), `shuffled` replaces it with *someone
else's* real, well-formed retrieved content -- yet they cost almost exactly
the same (+0.0133 vs. +0.0188 nats). If the retrieved vector's specific
identity carried meaningful information the model uses, `shuffled` (wrong
but real content) should cost noticeably more than `uniform` (no content);
it does not. Only `random`, which pushes Block 2 and the FFN readout
off the input distribution they were ever trained on, produces a large
loss increase -- and that increase is best read as a **distributional-shock
artifact**, not evidence that ~2.47 nats of useful information normally
flows through retrieval.

## Finding 6: attention's positional routing, not the final FFN, is where recall actually lives

With retrieval established as near-inert, two candidates remained for the
mechanism actually behind recall: the attention blocks (`attn1`/`attn2`)
doing genuine positional routing, or the final readout FFN
(`dense1 -> dense2 -> dense3`, a fixed function of one flattened vector `u`)
acting as a key-value memory over aggregate context content, indifferent to
*where* that content sits. Nothing else in this architecture mixes
information across sequence positions -- ALiBi's positional bias depends
only on the absolute index gap, not token identity -- so this is a clean,
weight-free test: `model scripts order-sensitivity-loss` re-scores the same
5,120 real validation examples under two corruptions, unmodified weights:

| condition | what it is | loss | &Delta; loss | &Delta; loss % | top-1 accuracy |
|---|---|---|---|---|---|
| normal | the real forward pass | 2.6150 | -- | -- | 43.65% |
| shuffled_order | the valid context tokens randomly permuted among themselves (same tokens, same length, scrambled sequence) | 7.8502 | +5.2352 | +200.20% | 5.64% |
| masked_target | every earlier occurrence of the target token replaced by a fixed placeholder, same position | 2.8116 | +0.1967 | +7.52% | 41.02% |

Shuffling word order **triples the loss and collapses top-1 accuracy from
43.65% to 5.64%** -- an effect two orders of magnitude larger than anything
found for retrieval (whose worst in-distribution ablation cost +0.019 nats).
Since shuffled tokens are still real, in-distribution token embeddings, just
reordered, this is not a distributional-shock artifact the way the `random`
retrieval condition was -- it is a direct measurement of how much the model
depends on sequence order, which only the attention blocks can exploit.
Masking just the target's earlier occurrence (keeping order and every other
token intact) costs much less (+0.197 nats, -2.6pp top-1) -- real but modest,
consistent with the model using redundant contextual cues beyond one exact
earlier token, not a single hard-coded position lookup.

**This rules out the final FFN being a pure bag-of-tokens key-value memory.**
If `dense1/2/3` were reading only aggregate, order-insensitive content from
`u`, scrambling word order should cost little, the same way scrambling the
retrieved vector's identity (the `shuffled` retrieval condition) cost little.
Instead it is catastrophic.

Caveat, found by re-running the identical test on `baseline`
(architecture `dual-sphere`: embed -> flatten -> normalize -> FFN -> cosine
readout, **no attention at all**): that model is *also* dramatically
order-sensitive.

| run | arch | normal loss | shuffled loss | shuffled &Delta;% | shuffled top-1 |
|---|---|---|---|---|---|
| transformer-alibi-retriver | alibi-retriever-sphere (attention) | 2.6150 | 7.8502 | +200.2% | 5.64% |
| baseline | dual-sphere (no attention) | 3.3438 | 7.6956 | +130.2% | 5.18% |

Both collapse to nearly the same absolute loss (~7.7-7.85 nats) and the same
near-floor top-1 accuracy (~5.2-5.6%) when shuffled, regardless of whether
attention exists. This is because flattening `(window_size, dim_in)` into
one fixed-slot vector before a Dense layer is *not* permutation-invariant on
its own -- each position occupies a distinct slice of the flattened input,
so even a pure FFN with zero sequence-mixing depends heavily on order. **This
is the classic Bengio-style fixed-window neural language model**: a
position-wise FFN over a flattened context necessarily has to learn *both*
"what is at each position" and "how positions combine" in the same set of
weights, because nothing else in that architecture separates the two jobs.

That reframes what the two architectures' near-identical parameter counts
buy them. Mechanism parameters are almost equal (1,658,370 for
`transformer-alibi-retriver` vs. 1,638,401 for `baseline` -- 1.2% more), yet
the attention model reaches substantially lower loss and higher accuracy
(2.6150 vs. 3.3438 nats; 43.65% vs. 35.43% top-1, both scored identically).
**The performance gap is a specialization gap, not a capacity gap.**
`baseline`'s FFN has to spend its weights on two jobs at once: figuring out
*where* useful information sits in the window, and *what* to infer from it.
Attention is comparatively parameter-cheap (`attn1`/`attn2` are small
bias-free projections) but computationally does the positional
contextualization -- mixing each position's representation with the right
others, guided by ALiBi's distance bias -- so that by the time the final FFN
runs, position has already been resolved into content. The final FFN in the
attention model is therefore free to specialize on pure inference from
already-contextualized representations, rather than splitting its capacity
between position-decoding and inference the way `baseline`'s single FFN
must. Attention buys more effective computation per parameter, not more
parameters.

## Finding 7: the model is not copying -- it falls back to whichever name is most frequent in training

`model scripts name-swap-trace --run-name transformer-alibi-retriver`
tests whether the model performs content-sensitive copying: each of the 20
recall templates has its earlier name swapped for every name in the same
20-name pool (position, length, and every other token held fixed), and the
top prediction is checked against the swapped-in name. If the model copies
the specific name's identity forward, the prediction should track the swap
for every name; if it does not, whatever explains which names *do* track
should not be "the model found this specific name in context."

Tracking splits sharply by name and is *not* explained by anything
context-specific -- every template and position is identical, only the name
differs:

| swapped-in name | tracked (predicted the same name back) | training-corpus occurrences |
|---|---|---|
| Lily | 20/20 | 2,359,584 |
| Timmy | 20/20 | 794,135 |
| Amy | 10/20 | 72,908 |
| Molly | 10/20 | 59,380 |
| Max | 9/20 | 404,365 |
| Sarah | 5/20 | 129,109 |
| Mia | 3/20 | 273,162 |
| Tom | 2/20 | 848,565 |
| Emma | 1/20 | 48,242 |
| Anna, Daisy, Emily, Jane, Jenny, Jill, Kate, Lucy, Maria, Mary, Sue | 0/20 | 8,973 - 331,278 |

Occurrence counts come from a full scan of the 467.2M-token training corpus
(`tinystories-10k`, train split). Overall, only 80/400 (20.0%) of swaps are
tracked, and when a swap is *not* tracked, the model does not fall back to
the *original* name either (only 2/380 non-tracking cases predict the
original) -- it falls back to a small set of names, overwhelmingly the
dialogue-starter "Let" (211/400 predictions across the whole matrix, not
itself a name but the model's generic template completion) or "Lily" itself
(124/400), regardless of which name was actually swapped in.

**This is not copying: the correlation is with training-corpus frequency,
not with anything about the specific context.** Pearson correlation between
log-frequency and swap-tracking rate across the 20 names is **r = 0.60**
(Spearman rho = 0.53, p = 0.015); small-table neighbor overlap from the
compression study, tested as an alternative explanation, correlates far more
weakly (r = 0.30) and adds negligible predictive power in combination
(R^2 rises from 0.36 to only 0.37). `Lily` is not just the best-preserved
name in the small table -- at 2.36M occurrences it is also by a wide margin
the single most common name in the entire corpus (`Tom`, the second most
frequent at 848K, tracks only 2/20, ruling out frequency-plus-discriminability
as a clean joint story too, but frequency alone is still the dominant single
predictor found). The model is doing **inference, not retrieval-by-copying**:
when the context under-determines the answer (because word order and
identity only partially resolve which name is meant, per Finding 6's own
partial-fix from masking), the model's learned prior over next tokens --
shaped overwhelmingly by how often each name appeared during training --
takes over, and it outputs whichever name (or generic filler) has the
highest unconditional probability given the surrounding template. Occasional
apparent "copying" of a specific name (Lily, Timmy) is exactly what that
same frequency-driven inference predicts when the discriminating context is
weak: the model defaults to its prior, and for those two names, the prior
happens to already point at the correct answer.

## Addendum: token-anchored retrieval (`gpu-retriver-tokenid`) is not uniform the same way

Findings 1-7 above are all about `transformer-alibi-retriver`
(`alibi-retriever-sphere`), whose query is compared against the *entire
vocabulary* table. `gpu-retriver-tokenid` (architecture
`alibi-token-retriever-sphere`, batch_size 73728, lr 0.01, trained on GPU) is
mechanically different: its query is compared only against the tokens
*already present in the context window* (`window_size` = 128 candidates, not
10,000), and the resulting per-position softmax weights are scattered into a
vocabulary-sized bucket by token id before pooling the big table (see
`model/scripts/_alibi_token_retriever_trace.py::trace_block1`). A uniform
softmax here is bounded by however many distinct token ids actually occur in
a given 128-token window, not by vocabulary size, so Finding 2's
"near-uniform over the full vocabulary" framing does not directly transfer.

`ad_hoc_scripts/gpu_retriver_tokenid_softmax_mass.py` measures concentration
directly: for each of 2,048 real validation examples (8 batches of 256,
sampled the same way `build_full_eval_dataset` samples for `model
evaluate`), it scatters the trace's per-position softmax weights by token id
exactly as the trained forward pass does, sorts the resulting per-example
vocabulary-weight bucket descending, and counts how many distinct token ids
are needed to reach 80% of the cumulative softmax mass.

| metric | value |
|---|---|
| examples analyzed | 2,048 |
| vocabulary size / window size | 10,000 / 128 |
| mean tokens for 80% mass | 26.74 |
| median | 28 |
| min | 1 |
| max | 51 |
| std | 11.23 |

Since a window can contain at most 128 distinct ids (and typically far fewer,
given repeated function words), needing ~27 of them on average to reach 80%
mass is meaningfully concentrated relative to the window, but still far from
a sharp, few-token spike -- it is closer to "the softmax leans on roughly a
fifth to a quarter of the in-context vocabulary" than to either extreme
(one dominant token, or fully uniform over everything present). This softmax
is measurably more content-selective than `transformer-alibi-retriver`'s
full-vocabulary retrieval (Finding 2's entropy ratio ~0.996-0.997), which is
expected given it is normalizing over a much smaller candidate pool
(in-context tokens only) rather than the full 10,000-token vocabulary -- the
two numbers are not directly comparable and this result should not be read
as contradicting Findings 1-7, which are scoped to the free-query
architecture only. Whether this measured concentration translates into
retrieval actually mattering for `gpu-retriver-tokenid`'s loss (the way
Finding 5's ablation settled it for the other architecture) is untested here
and would need the token-id equivalent, `retrieval-ablation-loss-tokenid`.

### Eyeball check: 5 sampled generations at temperature 0.5

`ad_hoc_scripts/gpu_retriver_tokenid_generate_sample.py` generates from the
same checkpoint (`gpu-retriver-tokenid`, `latest`, eval loss 2.1969, top-1
49.04%) by calling `model.generation.generate` directly rather than the
`model generate` CLI, because the CLI does not expose `stop_tokens` or a
`<bos>` prefix. Each of the 5 prompts is explicitly prefixed with the literal
`<bos>` token (id 0) before encoding, `temperature=0.5`, and generation runs
until `<eos>` (id 1) is sampled naturally (`stop_tokens=["<eos>"]`), with
`max_new_tokens=1024` only as a safety cap against a runaway non-terminating
sample -- consistent with how the model was trained on document-delimited
windows, and none of the 5 samples came close to hitting that cap.

Prompts and outcomes (all 5 reached `<eos>` naturally this time, unlike the
first pass at a 128-token cap where 3/5 were cut off mid-loop):

| # | prompt | tokens to `<eos>` |
|---|---|---|
| 1 | "Once upon a time, there was a little girl named Lily." | ~95 |
| 2 | "Tom and his dog went to the park to play." | ~150 |
| 3 | "One day, a boy found a strange box in the forest." | ~150 |
| 4 | "The sun was shining and the birds were singing in the" | ~185 |
| 5 | "Mom said it was time to clean up the toys before" | ~165 |

Eyeball read of the transcripts:

- **Grammar and local coherence are solid.** Every sentence is
  well-formed English, subject-verb agreement holds throughout, and dialogue
  tags/quotation marks are used correctly in all 5 samples -- this is the
  same quality bar the model's ~49% top-1 accuracy and 2.20 eval loss would
  suggest.
- **Repetition is still the dominant failure mode, and removing the token
  cap makes this clearer, not better.** Given room to keep going instead of
  being cut off, the model does not recover into new content -- it loops
  longer before finally reaching a resolution beat. Prompt 2 cycles through
  "the tree" (climb it, run to it, "ate the tree") for several extra
  sentences past where the capped run was cut off, then abruptly resolves
  with "They were good friends." Prompt 4 is the clearest case: "The birds
  sang and sang" / "singing and singing" repeats across essentially the
  entire ~185-token sample with almost no new content introduced, before
  ending on "it was time to go home." Prompt 5 loops on "swing"/"so happy"
  for most of its length. This confirms the capped run's hypothesis directly:
  these were not going to self-correct with more tokens, they were going to
  keep looping until the model happened to sample its way into an ending.
- **`<eos>` is still coupled to a narrative closure beat, not emitted
  arbitrarily** -- now confirmed on all 5 samples instead of 2/5. Every
  sample ends on a plausible closing line ("I love you!"; "They were good
  friends."; "They had a lot of fun!"; "it was time to go home."; "she was
  the best way to swing." -- the last is semantically garbled but
  structurally still a wrap-up line), even the heavily-repetitive ones. The
  model reliably finds an exit eventually; it just may take a long,
  repetitive detour to get there.
- **Local topic drift persists.** Prompt 3 changes what's found in the
  forest from "a strange box" to "a big, shiny rock" to "a mineral" across
  the sample without acknowledging the shift, and prompt 2 still drops its
  "big, brown bear" as soon as it's mentioned. Short-range object
  permanence breaking down under a repetitive scene is unaffected by
  removing the token cap -- consistent with Finding 6's result that this
  family of models leans on recent positional context rather than a stable
  global state.

This is a small, unscored, single-seed eyeball pass on one temperature
(N=5) -- useful for a fast qualitative sanity check of `gpu-retriver-tokenid`
generation quality, not a substitute for the quantitative retrieval findings
above or for a scored generation-quality study (e.g. repetition rate,
distinct-n, or human eval across seeds/temperatures).

## Conclusion

The name given to this architecture's mechanism ("retriever") describes its
intended design, not its learned behavior at this checkpoint: for in-context
name recall *and* for ordinary next-token prediction on real training data
of every length tested, **the softmax retrieval step against the big table
has degenerated into an almost input-independent constant** (entropy ratio
~0.996-0.997, retrieval-invariance ~0.995-0.998, uniformly across every case
tested -- hand-crafted or real, short or long, hit or miss), and the
model-level loss cost of removing its query-specific content is small and
almost entirely explained by leaving the retrieved vector on-distribution
or not, rather than by the vector's actual content (+0.013-0.019 nats for
in-distribution substitutions vs. +2.47 nats only once the input is pushed
off-distribution entirely).

Word order matters enormously (Finding 6: +5.24 nats, top-1 to near-floor
when shuffled), and re-running the identical test on `baseline`
(architecture `dual-sphere`, no attention at all) showed this is not
exclusive proof of attention's contribution by itself -- an attention-free
flatten-then-FFN readout is *also* dramatically order-sensitive, because it
has no other way to resolve "where" information sits in the window. What the
comparison actually reveals is a **specialization gap, not a copying
mechanism**: with almost identical mechanism parameter counts (1,658,370 vs.
1,638,401 -- 1.2% more), the attention model reaches meaningfully lower loss
and higher accuracy (2.6150 vs. 3.3438 nats; 43.65% vs. 35.43% top-1). This
is the classic Bengio-style fixed-window neural language model's limitation:
a bare FFN over a flattened context must spend its weights on *both*
decoding position and inferring content from it, in the same parameters.
Attention is parameter-cheap but does real, computationally substantial work
disentangling the two -- resolving "where" via ALiBi-guided mixing across
positions -- so the final FFN can specialize purely on content-based
inference from an already-contextualized representation, rather than
splitting its capacity the way `baseline`'s single FFN must.

Finding 7 rules out the remaining alternative that this specialized
inference amounts to copying a specific in-context name forward. Swapping
the earlier name's identity while holding context fixed shows tracking rates
that correlate with **training-corpus frequency** (r = 0.60 for
log-frequency vs. tracking rate across 20 names), not with anything about
the specific prompt -- the two names that track perfectly (`Lily`, `Timmy`)
are also, respectively, the single most and third-most frequent names in the
467M-token training corpus, while `Tom` (second-most frequent, 848K
occurrences) tracks only 2/20, and small-table discriminability (the
compression study's own explanation) correlates far more weakly (r = 0.30)
and adds negligible power combined with frequency. The model is performing
**inference from an ambiguous, under-resolved context**, not
content-addressed retrieval: when the surrounding structure does not pin
down a specific name strongly enough, the model falls back to its learned
prior over next tokens, dominated by how often each candidate appeared
during training, and outputs the name (or generic filler, "Let") with the
highest unconditional probability given the template -- not the name that
was actually present.

The most accurate description of this checkpoint's mechanism, combining all
seven findings: the small table supplies a coarse per-token sketch; the
attention blocks perform the computationally substantial job of resolving
position into content (a job a same-sized bare FFN would otherwise have to
share its capacity with), letting the final FFN specialize on inference; and
that specialized inference is frequency-driven pattern completion over an
imperfectly-resolved context, not verbatim copying -- while the purpose-built
softmax retrieval pathway contributes almost nothing beyond serving as a
stable, in-distribution filler vector for Block 2 to condition on. "The
model recalls a name from context" is, more precisely, "the model infers the
most probable name given a partially-resolved context" -- which happens to
look like recall whenever the most probable name and the actually-present
name coincide, and looks like a generic guess whenever they don't.
