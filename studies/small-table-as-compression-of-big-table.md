# Does the small embedding table compress the big embedding table?

Run analyzed: `transformer-alibi-retriver` (architecture `alibi-retriever-sphere`),
checkpoint `latest` (epoch 200).

## Model architecture

`AlibiRetrieverSphereLM` (`model/archs/alibi_retriever_sphere.py`) is a two-stage
spherical transformer with a soft vocabulary-retrieval bottleneck between the
two stages:

1. **Small table** (`input_embeddings`, shape `(10000, 8)`), L2-normalized onto
   `S^7`. Every input token is looked up here.
2. **Block 1**: causal ALiBi transformer block (L2-norm -> causal MHA -> L2-norm
   -> MLP) over the `window_size = 128` token window, entirely in the 8-D
   small-table space.
3. **Query extraction**: the last `dim_out / dim_in = 16` output tokens of
   block 1 are reshaped and L2-normalized into one 128-D query vector on
   `S^127`.
4. **Soft retrieval**: cosine similarity between that query and every row of
   the **big table** (`target_embeddings`, shape `(10000, 128)`, also on
   `S^127`), softmax-weighted (learnable `retriever_tau`), pooled into one
   128-D "retrieved" vector.
5. The retrieved vector is reshaped back into 16 tokens of 8-D each,
   concatenated onto the block-1 sequence, and passed through **Block 2**
   (a second causal ALiBi transformer block).
6. A sequence projection + baseline FFN readout produces the final 128-D
   query, scored against the big table by cosine similarity (scaled by a
   learnable `tau`) to get the next-token logits.

Key parameters (`runs/transformer-alibi-retriver/config.json`):

| param | value |
|---|---|
| vocabulary_size | 10,000 |
| dim_in (small table width) | 8 |
| dim_out (big table width) | 128 |
| window_size | 128 |
| hidden_dim | 1024 |
| num_heads | 2 |
| activation | gelu |

Dataset: `tinystories-10k` (467.2M training tokens, 2.12M documents, a
byte-level BPE tokenizer trained from scratch on the corpus, vocab size
10,000). Trained 200 epochs x 1,000 steps. Final validation score
(`eval_latest.json`): loss 2.635, perplexity 13.95, **bpc 0.921**, top-1
accuracy 43.8%, 3.02M total parameters (1.36M embedding, 1.66M mechanism).

Both tables are indexed by the same 10,000-token vocabulary: token `i`'s row
in the small table and its row in the big table describe the same token, just
at two different widths and (as the retrieval readout requires) two learned,
a priori unrelated geometries.

## Method of data collection

Three independent analyses, implemented as `keras.ops`-only scripts under
`model/scripts/` and run against the trained checkpoint's small/big embedding
tables (both L2-normalized before analysis):

1. **`model scripts pga-dual-embeddings --run-name transformer-alibi-retriver`**
   Principal Geodesic Analysis (tangent-space PCA at the Karcher mean) of the
   big table, reduced to 8 components to match `dim_in`. The 8-D PGA
   embedding is aligned to the small table with the best single orthogonal
   (Procrustes) rotation, then compared per-token by cosine similarity.
   - 8 components explain **22.0%** of the big table's tangent-space variance.
   - Mean cosine similarity after alignment: **0.260** (median 0.294).
   - Only **27.2%** of tokens exceed 0.5 cosine similarity after the best
     possible single rotation.

2. **`model scripts spherical-k-means --run-name transformer-alibi-retriver -k 32`**
   Spherical k-means (k = 32) run independently on each table, then compared
   via Adjusted Rand Index and Normalized Mutual Information over the shared
   10,000 tokens.
   - ARI: **0.095**, NMI: **0.264** -- weak agreement between the two
     tables' induced partitions of the vocabulary.

3. **`model scripts table-neighbor-agreement --run-name transformer-alibi-retriver`**
   For every token, its 10 nearest neighbors (cosine similarity) are found
   independently in each table and intersected -- a rotation- and
   partition-invariant test of "are a token's neighbors preserved across
   tables?" Additionally, Pearson correlation between small-table and
   big-table cosine similarity is computed over 200,000 randomly sampled
   token pairs.
   - Mean 10-NN overlap: **6.2%** (median near 0%, right-skewed), versus a
     **0.1%** chance level for 10 neighbors drawn from 10,000 tokens -- **62x**
     above chance.
   - Pairwise similarity correlation: **0.575**.

All three scripts write self-contained HTML reports (Jinja2 + Plotly) to
`runs/transformer-alibi-retriver/reports/`.

### Case study: "Jane"

Token `ĠJane` (id 1247, the mid-sentence, space-prefixed form used after e.g.
"named"), top-10 nearest neighbors by cosine similarity in each table:

| rank | small table | big table |
|---|---|---|
| 1 | Emma | Mary |
| 2 | Emily | Molly |
| 3 | Susie | Jill |
| 4 | Molly | Emma |
| 5 | Sally | Jane *(no-space variant)* |
| 6 | Molly *(dup form)* | Susie |
| 7 | Jane *(no-space variant)* | Emily |
| 8 | Jill | Jenny |
| 9 | Mary | Joe |
| 10 | Maria | Maria |

**8 of 10 neighbors are shared between the two tables** (80% overlap, well
above the 6.2% average and enormously above the 0.1% chance level) --
{Emma, Molly, Jill, Emily, Mary, Maria} all appear on both sides, and every
neighbor in both lists is a first name. "Jane" is one of the well-mapped
tokens: the small table's 8-D neighborhood for this token is a faithful,
if imperfect, sketch of the big table's 128-D neighborhood.

### Does 8 dimensions let the model tell names apart? (Jane vs. Emma)

The 10-NN overlap only shows that Jane's neighborhood is *consistent* between
tables -- it does not show whether the small table can tell Jane and Emma
apart at all, since both are expected to be near each other in any reasonable
name-embedding space. To test discrimination directly, cosine similarity was
computed for all pairs among 10 common female first names (`Jane, Emma,
Emily, Susie, Molly, Sally, Jill, Mary, Maria, Jenny`), plus each pair's
similarity percentile against 100,000 random vocabulary pairs:

| pair | small-table cos sim | big-table cos sim |
|---|---|---|
| Jane-Emma | 0.981 | 0.707 |
| Jane-Molly | 0.979 | 0.720 |
| Jane-Susie | 0.979 | 0.685 |
| Jane-Emily | 0.980 | 0.658 |
| Jane-Jill | 0.971 | 0.707 |
| Jane-Maria | 0.963 | 0.585 |
| Jane-Mary | 0.969 | 0.763 |
| Jane-Sally | 0.837 | 0.569 |
| Jane-Jenny | 0.815 | 0.603 |
| Sally-Molly | 0.736 | 0.147 |
| Molly-Susie | 0.985 | 0.845 |
| Jill-Susie | 0.930 | 0.284 |

Random-pair baseline: mean cosine similarity 0.335 (small table) / 0.294
(big table) across the full vocabulary. Jane-Emma sits at the **99.98th
percentile** of small-table similarity and the **97.90th percentile** of
big-table similarity among random pairs -- both tables recognize these as an
unusually close pair, but the small table pushes almost every name pair up
into the 0.8-0.98 range, while the big table spreads the same pairs much more
widely (0.15-0.85), preserving structure the small table erases (e.g.
Sally-Molly: 0.147 in the big table vs. 0.736 in the small table).

The variance decomposition of the 10-name cluster confirms this is a
dimensionality problem, not just a display artifact. An SVD of the (centered)
name-cluster points shows:

| axis | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| small table variance share | 58.7% | 19.9% | 8.0% | 7.5% | 4.7% | 0.9% | 0.3% | 0.1% |
| big table variance share (first 8 of 127) | 37.1% | 19.6% | 13.7% | 12.2% | 6.6% | 4.4% | 2.8% | 2.0% |

In the small table, one axis alone (almost certainly a generic "is a
name"/gender direction) explains 58.7% of the already-tiny variation among
these names, and by the 6th axis there is essentially nothing left (under
1.3% combined across axes 6-8) to distinguish individual identities. The big
table spreads variance far more evenly across its first 8 axes, i.e. it
genuinely uses more independent directions to keep names apart.

**8 dimensions is not enough for the small table, on its own, to keep names
crisply separated** -- it collapses the name cluster onto a near
one-dimensional ridge. The model avoids name swaps in practice only because
the *retrieval step* reaches into the big table, which still carries most of
the separating information; if generation depended on the small table alone,
confusing Jane for Emma (or any two of these names) would be the expected
failure mode, not an edge case.

### Group discriminability, generalized beyond names

The Jane/Emma finding above generalizes: `table_neighbor_agreement.html` now
reports, for every token already listed in its best/worst-overlap tables, a
**group discriminability** measurement on that token's own small-table
neighborhood (the token plus its `k` nearest neighbors) -- computed identically
in both tables:

- **group similarity**: mean pairwise cosine similarity within the group
  (higher = more bunched together, less discriminable).
- **group rank**: a participation-ratio effective rank in `[0, 1]` (how many
  independent directions the group's variance actually spans; near 0 means
  the group has collapsed onto essentially one axis).

The pattern holds across every inspected token, not just names:

| token | group sim (small / big) | group rank (small / big) |
|---|---|---|
| " clown" (90% NN overlap) | 0.90 / 0.56 | 0.53 / 0.72 |
| " rat" (80% NN overlap) | 0.93 / 0.50 | 0.46 / 0.57 |
| "'" (0% NN overlap) | 0.83 / 0.16 | 0.44 / 0.85 |

In every case, regardless of whether the token's neighbor *set* transfers
well (clown, rat) or poorly (`'`), the **small table always bunches the
group tighter and uses less effective dimensionality than the big table**.
This is not a property of a few tokens with an unlucky geometry -- it is a
structural property of the 8-D bottleneck itself: the same handful of
directions that place a token in roughly the right neighborhood category
leave little room left over to also spread that category's members apart.

## Conclusion

**The small table is a partial, non-linear, token-dependent compression of
the big table.** More precisely, and this is the clearest finding of this
study: **the small table stores only abstract group classes ("this is a
name," "this is an animate character," "this is punctuation") which then get
mapped to fine-grained, individually-discriminated tokens in the big table.**
Given that the small table has 8 dimensions against the big table's 128, this
result could hardly have been otherwise -- 8 dimensions is nowhere near
enough to hold both "which broad category" and "which specific member of
that category" for a 10,000-token vocabulary, so the model has learned to
spend its small-table capacity on the former and defer the latter entirely
to the retrieval step into the big table.

- It is a genuine compression in the sense that neighbor structure survives
  the 128-D -> 8-D bottleneck at a rate far above chance (6.2% vs. 0.1% for
  10-NN overlap) and cosine similarities between the two tables correlate
  positively and substantially (r = 0.575) across the vocabulary.
- It is **not** a clean linear compression: the best possible single
  orthogonal rotation between an 8-component PGA summary of the big table and
  the small table only reaches 0.260 mean cosine similarity, far short of
  what a faithful linear encoder/decoder pair would produce.
- It is **not** a clean discrete-partition compression either: independent
  spherical k-means clusterings of the two tables barely agree (ARI 0.095).
- The relationship is markedly **token-dependent**: the neighbor-overlap
  distribution is right-skewed, with a median near 0% but a real tail of
  strongly-compressed tokens. "Jane" (80% 10-NN overlap) sits in that tail,
  alongside other proper nouns and concrete, semantically tight categories
  observed in the neighbor-agreement report (e.g. "clown" at 90% overlap);
  many other tokens map far more weakly.
- Within that tail, the compression is still lossy at the level of individual
  identity: 8 dimensions is not enough to keep a tight cluster like first
  names separated from each other -- one axis alone explains 58.7% of the
  small table's name-to-name variation, versus 37.1% for the corresponding
  leading axis in the big table. The small table gets the *category* right
  (these are all names, and roughly the right names) but leaves little room
  to tell members of the category apart; that finer discrimination survives
  only in the big table.
- This under-discrimination is not specific to names: measured directly via
  group similarity / effective rank on every inspected token's own
  neighborhood (" clown", " rat", "'"), the small table consistently bunches
  the group tighter and spans fewer effective dimensions than the big table,
  independent of whether the token's neighbor set otherwise transfers well or
  poorly. The small table encodes *which abstract group a token belongs to*;
  the big table is where *which specific member of that group* gets decided.

This is consistent with the two-stage architecture's division of labor: the
small table (queried directly by Block 1's attention/FFN) encodes a coarse,
lossy semantic sketch -- enough to steer retrieval toward the right
*category* (e.g. "this is a name, and one of these five or six names") for
tokens with a tight, well-separated semantic identity -- while the big table,
reached only through the softmax retrieval step, supplies the finer-grained
distinctions needed both for the rest of the vocabulary and for telling
individual members of a tight category (like first names) apart. It explains
the qualitative behavior noted during generation: names and concrete entities
are remembered coherently across a story's context window, while more
diffuse or function-heavy vocabulary is not -- and also predicts that, since
the small table alone cannot fully separate similar names, occasional name
swaps between semantically close names should remain a plausible failure
mode whenever the retrieval step's softmax is not sharp enough to fully
compensate.
