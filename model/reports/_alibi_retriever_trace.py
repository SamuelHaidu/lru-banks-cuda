"""Forward-pass replication for `AlibiRetrieverSphereLM`, exposed for tracing.

Both `query-recall-trace` and `retrieval-behavior-trace` need to read
intermediate values (attention weights, the retrieval softmax, an ablated
retrieved vector) that the trained `call()` doesn't return. Every function
here re-runs a piece of the real forward pass using the model's own trained
sub-layers directly -- verified byte-identical to the real layer outputs
(max diff ~5e-6, float rounding only) -- so this is inspection of the trained
checkpoint, not an approximate re-implementation.
"""

from typing import Any, Dict

from model.device import setup_backend

setup_backend()

from keras import ops  # noqa: E402


def trace_block1(model: Any, token_ids: Any) -> Dict[str, Any]:
    """Replicates `AlibiRetrieverSphereLM.call` through query extraction.

    `token_ids`: `(batch, window_size)`. Returns the block-1 attention
    weights `(batch, num_heads, seq_len, seq_len)`, the L2-normalized 128-D
    query vector `(batch, dim_out)`, and the block-1 output sequence `x`
    `(batch, seq_len, dim_in)` used to continue into Block 2.
    """
    batch_size = ops.shape(token_ids)[0]
    norm_in = ops.normalize(model.input_embeddings, axis=-1, order=2)
    flat = ops.take(norm_in, ops.reshape(token_ids, (-1,)), axis=0)
    x = ops.reshape(flat, (batch_size, model.window_size, model.dim_in))

    attn1 = model.attn1
    h = ops.normalize(x, axis=-1, order=2)

    seq_len = ops.shape(h)[1]
    q = attn1.q_proj(h)
    k = attn1.k_proj(h)
    v = attn1.v_proj(h)
    q = ops.transpose(ops.reshape(q, (batch_size, seq_len, attn1.num_heads, attn1.head_dim)), (0, 2, 1, 3))
    k = ops.transpose(ops.reshape(k, (batch_size, seq_len, attn1.num_heads, attn1.head_dim)), (0, 2, 1, 3))
    v = ops.transpose(ops.reshape(v, (batch_size, seq_len, attn1.num_heads, attn1.head_dim)), (0, 2, 1, 3))

    scale = 1.0 / (float(attn1.head_dim) ** 0.5)
    scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale
    alibi, mask = attn1._get_alibi_and_mask(seq_len, scores.dtype)
    scores = scores + alibi + mask
    attn_weights = ops.softmax(scores, axis=-1)

    attn_out = ops.matmul(attn_weights, v)
    attn_out = ops.reshape(ops.transpose(attn_out, (0, 2, 1, 3)), (batch_size, seq_len, attn1.num_heads * attn1.head_dim))
    attn_out = attn1.out_proj(attn_out)

    x = x + attn_out
    h = ops.normalize(x, axis=-1, order=2)
    x = x + model.trans1_mlp2(model.trans1_mlp1(h, training=False), training=False)

    last_tokens = x[:, -model.num_query_tokens :, :]
    query = ops.reshape(last_tokens, (batch_size, model.dim_out))
    query = ops.normalize(query, axis=-1, order=2)

    return {"attn_weights": attn_weights, "query": query, "x": x}


def finish_forward(model: Any, x: Any, retrieved_tokens: Any) -> Any:
    """Replicates `AlibiRetrieverSphereLM.call` from Block 2 onward.

    `x`: `(1, window_size, dim_in)` Block-1 output. `retrieved_tokens`:
    `(1, num_query_tokens, dim_in)`, either the model's real softmax-pooled
    vector or a substitute -- lets an ablation swap in an arbitrary retrieved
    vector and see the effect on the final logits, using every other weight
    exactly as trained.
    """
    batch_size = ops.shape(x)[0]
    norm_tgt = ops.normalize(model.target_embeddings, axis=-1, order=2)

    x2 = ops.concatenate([x, retrieved_tokens], axis=1)
    h2 = ops.normalize(x2, axis=-1, order=2)
    x2 = x2 + model.attn2(h2, training=False)
    h2 = ops.normalize(x2, axis=-1, order=2)
    x2 = x2 + model.trans2_mlp2(model.trans2_mlp1(h2, training=False), training=False)

    x2_t = ops.transpose(x2, (0, 2, 1))
    y_t = model.seq_proj(x2_t, training=False)
    y = ops.transpose(y_t, (0, 2, 1))
    flat_y = ops.reshape(y, (batch_size, model.window_size * model.dim_in))
    u = ops.normalize(flat_y, axis=-1, order=2)

    h_ffn = model.dense1(u, training=False)
    h_ffn = model.dense2(h_ffn, training=False)
    z = model.dense3(h_ffn, training=False)
    q_final = ops.normalize(z, axis=-1, order=2)
    cos_sim = ops.matmul(q_final, ops.transpose(norm_tgt))
    return cos_sim * ops.cast(model.tau, cos_sim.dtype)


def retrieval_weights(model: Any, query: Any) -> Any:
    """The softmax retrieval weights `(batch, V)` and per-example entropy ratio `(batch,)`.

    `query` is `(dim_out,)` or `(batch, dim_out)`. Entropy ratio is
    `H(weights) / log(V)` in `[0, 1]`: 1.0 means the distribution is exactly
    uniform (retrieval carries no information about the query -- it always
    returns the same average vector), values near 0 mean the softmax is
    sharply concentrated on a few tokens (retrieval is actually
    content-dependent for this query).
    """
    query = ops.reshape(query, (-1, ops.shape(query)[-1]))
    norm_tgt = ops.normalize(model.target_embeddings, axis=-1, order=2)
    sim = ops.matmul(query, ops.transpose(norm_tgt)) * ops.cast(model.retriever_tau, query.dtype)
    weights = ops.softmax(sim, axis=-1)
    vocab_size = ops.cast(ops.shape(weights)[-1], weights.dtype)
    entropy = -ops.sum(weights * ops.log(weights + 1e-12), axis=-1)
    entropy_ratio = entropy / ops.log(vocab_size)
    return weights, entropy_ratio


def retrieve(model: Any, query: Any) -> Any:
    """Reproduces the model's softmax retrieval step: query -> pooled tokens.

    `query`: `(dim_out,)` or `(batch, dim_out)`. Returns
    `(batch, num_query_tokens, dim_in)`.
    """
    weights, _ = retrieval_weights(model, query)
    batch_size = ops.shape(weights)[0]
    norm_tgt = ops.normalize(model.target_embeddings, axis=-1, order=2)
    pooled = ops.matmul(weights, norm_tgt)
    pooled = ops.normalize(pooled, axis=-1, order=2)
    retrieved = ops.reshape(pooled, (batch_size, model.num_query_tokens, model.dim_in))
    return ops.normalize(retrieved, axis=-1, order=2)


def vocab_mean_retrieved_tokens(model: Any, batch_size: int = 1) -> Any:
    """The retrieved-token vector a fully uniform softmax would produce.

    If the real retrieval weights are close to uniform, the real retrieved
    vector and this one should be nearly identical -- a direct test of
    whether retrieval is content-dependent at all. Broadcasts to
    `(batch_size, num_query_tokens, dim_in)`.
    """
    norm_tgt = ops.normalize(model.target_embeddings, axis=-1, order=2)
    mean_tgt = ops.normalize(ops.mean(norm_tgt, axis=0, keepdims=True), axis=-1, order=2)
    retrieved = ops.reshape(mean_tgt, (1, model.num_query_tokens, model.dim_in))
    retrieved = ops.tile(retrieved, (batch_size, 1, 1))
    return ops.normalize(retrieved, axis=-1, order=2)


def random_retrieved_tokens(model: Any, batch_size: int, seed: Any = None) -> Any:
    """Genuinely random unit-norm retrieved tokens -- a harsher corruption.

    Unlike the uniform-softmax substitute (which is at least a real,
    in-distribution average of the big table), this is off-distribution
    noise: the retrieved vector's specific content is destroyed, not just
    made query-independent.
    """
    from keras import random as keras_random

    noise = keras_random.normal((batch_size, model.num_query_tokens, model.dim_in), seed=seed)
    return ops.normalize(noise, axis=-1, order=2)
