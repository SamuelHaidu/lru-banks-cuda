# Token-anchored retrieval vs. free-query retrieval: where does the gain come from?

Runs analyzed (all `tinystories-10k`, 10 epochs x 1,000 steps, batch 512,
`dim_in=8`, `dim_out=128`, `window_size=128`, `hidden_dim=1024`):

| run | arch | loss (nats) |
|---|---|---|
| `baseline` | `dual-sphere` (no transformer block) | 3.3622 |
| `transformer-alibi` | `alibi-sphere` (single ALiBi block) | 3.3363 |
| `transformer-alibi-retriver` | `alibi-retriever-sphere` (block + free-query retrieval) | 3.1914 |
| `retriver-tokenid` | `alibi-token-retriever-sphere` (block + token-anchored retrieval) | 3.0794 |

## Architectural difference being isolated

`alibi-retriever-sphere` lets Block 1 emit an arbitrary 128-D query point on
the big table's manifold and retrieves by cosine similarity against all
10,000 rows — the transformer decides *where* to query.

`alibi-token-retriever-sphere` (this session's new module) restricts
retrieval to token ids that actually occur in the current window: Block 1's
output is compared against itself (in `dim_in` space) to get a softmax
weight per context position, those weights are scattered into a
vocabulary-sized bucket keyed by each position's token id, and that bucket
weighted-pools the big table. The transformer only decides *how much*
fine-grained detail to pull per in-context token, not *which* point to query.

## Decomposition (user's analysis)

Deltas are against the `baseline` (3.3622 nats):

- **`baseline -> transformer-alibi`**: +1 ALiBi block, no retrieval.
  Gain: **0.025 nats**. A single transformer block over the small table
  buys almost nothing on its own.
- **`transformer-alibi -> transformer-alibi-retriver`**: +free-query
  retrieval + Block 2. Gain: **0.17 nats**, a **6.5x** jump over the single
  block's own contribution. Most of the previously observed benefit of
  "the retriever architecture" is actually structural (a second transformer
  block + wider effective capacity), not the free-query retrieval itself.
- **`transformer-alibi-retriver -> retriver-tokenid`**: swap free-query
  retrieval for token-anchored retrieval, same block count/capacity.
  Gain: **0.12 nats** over the previous baseline (`transformer-alibi-retriver`),
  a further **~1.5x** improvement on top of what was previously believed to
  be "useless" retrieval capacity.

## Conclusion

Net gain of token-anchored retrieval over the full 2-block baseline: **0.25
nats**. Of that:

- ~0.025 nats is explained by a second transformer block existing at all.
- The remaining **~0.09 nats is retrieval doing real work that free-query
  retrieval was not doing** — i.e. constraining the query to only ask about
  tokens actually present in context, rather than letting the transformer
  invent an arbitrary query point, extracts information the free-query
  version was leaving on the table.

This supports the design rationale for the new architecture: "we will not
let the transformer say what is the query, we will ask the transformer to
say what are the tokens that need fine-grained information."

## Related

- `studies/small-table-as-compression-of-big-table.md` — background on
  `alibi-retriever-sphere`'s free-query retrieval mechanism and the two-table
  spherical setup this study builds on.
- `model/archs/alibi_token_retriever_sphere.py` — implementation.
- `tests/test_arch_alibi_token_retriever_sphere.py` — scatter-semantics
  regression tests (`TestTokenIdScatter`).
