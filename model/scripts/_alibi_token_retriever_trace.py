"""Forward-pass replication for `AlibiTokenRetrieverSphereLM`, exposed for tracing.

Mirrors `_alibi_retriever_trace.py` but follows this architecture's actual
retrieval mechanism: a query projected from the *flattened* Block-1 output,
compared against Block-1's own token outputs (not the vocabulary table), then
scattered by token id into a vocabulary-weight vector that pools the big
table. Every function here re-runs a piece of the real forward pass using the
model's own trained sub-layers directly, so this is inspection of the trained
checkpoint, not an approximate re-implementation.
"""

from typing import Any, Dict

from model.device import setup_backend

setup_backend()

from keras import ops  # noqa: E402


def trace_block1(model: Any, token_ids: Any) -> Dict[str, Any]:
    """Replicates `AlibiTokenRetrieverSphereLM.call` through the token-weight scatter.

    `token_ids`: `(batch, window_size)`. Returns the block-1 output sequence
    `x` `(batch, window_size, dim_in)`, the L2-normalized query `(batch,
    dim_in)`, the per-position softmax weights `(batch, window_size)`, and the
    scattered vocabulary-weight vector `(batch, vocabulary_size)`.
    """
    token_ids = ops.cast(token_ids, "int32")
    batch_size = ops.shape(token_ids)[0]
    norm_in = ops.normalize(model.input_embeddings, axis=-1, order=2)
    flat = ops.take(norm_in, ops.reshape(token_ids, (-1,)), axis=0)
    x = ops.reshape(flat, (batch_size, model.window_size, model.dim_in))

    h = ops.normalize(x, axis=-1, order=2)
    x = x + model.attn1(h, training=False)
    h = ops.normalize(x, axis=-1, order=2)
    x = x + model.trans1_mlp2(model.trans1_mlp1(h, training=False), training=False)

    flat_x = ops.reshape(x, (batch_size, model.window_size * model.dim_in))
    query = model.query_proj(flat_x, training=False)
    query = ops.normalize(query, axis=-1, order=2)

    x_norm = ops.normalize(x, axis=-1, order=2)
    token_sim = ops.einsum("bd,bwd->bw", query, x_norm) * ops.cast(model.retriever_tau, query.dtype)
    token_weights = ops.softmax(token_sim, axis=-1)

    batch_idx = ops.reshape(ops.arange(batch_size, dtype="int32"), (batch_size, 1))
    offset_ids = token_ids + batch_idx * model.vocabulary_size
    flat_ids = ops.reshape(offset_ids, (-1,))
    flat_weights = ops.reshape(token_weights, (-1,))
    vocab_weights = ops.segment_sum(flat_weights, flat_ids, num_segments=batch_size * model.vocabulary_size)
    vocab_weights = ops.reshape(vocab_weights, (batch_size, model.vocabulary_size))

    return {"x": x, "query": query, "token_weights": token_weights, "vocab_weights": vocab_weights}


def retrieve_from_vocab_weights(model: Any, vocab_weights: Any) -> Any:
    """Vocabulary-weight vector -> pooled, split, normalized retrieved tokens.

    `vocab_weights`: `(batch, vocabulary_size)`. Returns `(batch,
    num_query_tokens, dim_in)`.
    """
    batch_size = ops.shape(vocab_weights)[0]
    norm_tgt = ops.normalize(model.target_embeddings, axis=-1, order=2)
    v_pooled = ops.matmul(vocab_weights, norm_tgt)
    v_pooled = ops.normalize(v_pooled, axis=-1, order=2)
    retrieved = ops.reshape(v_pooled, (batch_size, model.num_query_tokens, model.dim_in))
    return ops.normalize(retrieved, axis=-1, order=2)


def finish_forward(model: Any, x: Any, retrieved_tokens: Any) -> Any:
    """Replicates `AlibiTokenRetrieverSphereLM.call` from Block 2 onward.

    `x`: `(batch, window_size, dim_in)` Block-1 output. `retrieved_tokens`:
    `(batch, num_query_tokens, dim_in)`, either the model's real scatter-pooled
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


def vocab_mean_retrieved_tokens(model: Any, batch_size: int = 1) -> Any:
    """The retrieved-token vector a fully uniform token-weight vector would produce.

    If the real vocabulary weights are close to uniform, the real retrieved
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

    Unlike the uniform substitute (which is at least a real, in-distribution
    average of the big table), this is off-distribution noise: the retrieved
    vector's specific content is destroyed, not just made query-independent.
    """
    from keras import random as keras_random

    noise = keras_random.normal((batch_size, model.num_query_tokens, model.dim_in), seed=seed)
    return ops.normalize(noise, axis=-1, order=2)
