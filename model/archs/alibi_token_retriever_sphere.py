"""ALiBi Token-Retriever Sphere LM: two-stage transformer with token-anchored retrieval.

Unlike `AlibiRetrieverSphereLM` (`alibi_retriever_sphere.py`), which lets Block 1
invent a free query point anywhere on the big table's manifold, this
architecture restricts retrieval to tokens that actually occurred in the
current context window. Block 1 only decides *how much* fine-grained detail to
pull for each in-context token, not *which point in embedding space* to query.

Architecture flow:
    1. Input table ``(V, dim_in)``, L2-normalized onto S^(dim_in - 1).
    2. Gather ``window_size`` causal ids -> ``(B, window_size, dim_in)``.
    3. Block 1: Causal ALiBi Transformer block (L2-norm + causal MHA + L2-norm + MLP)
       -> transformed sequence ``x``, ``(B, window_size, dim_in)``.
    4. Query projection: flatten ``x`` to ``(B, window_size * dim_in)``, project
       down to ``dim_in`` with a learned matrix, L2-normalize -> query on
       S^(dim_in - 1).
    5. In-context similarity: cosine similarity between the query and each of
       the ``window_size`` transformed tokens in ``x`` (dim_in space, not
       vocabulary space) -> ``(B, window_size)`` scores, scaled by learnable
       ``retriever_tau`` and softmax'd -> one weight per window position.
    6. Token-id scatter: route each position's softmax weight into a
       ``(B, vocabulary_size)`` vector keyed by that position's token id
       (positions sharing a token id accumulate their weights). Implemented
       with a batch-offset ``segment_sum`` rather than a dense one-hot, since
       a ``(B, window_size, vocabulary_size)`` one-hot tensor is several GiB
       at real batch/window/vocab sizes; no renormalization needed since it
       is a regrouping of a distribution that already sums to 1.
    7. Soft vocabulary retrieval: use the vocabulary-weight vector to pool
       ``target_embeddings`` (weighted sum over all ``V`` rows), L2-normalize
       to obtain a continuous semantic summary vector on S^(dim_out - 1).
    8. Token split: reshape the pooled ``dim_out``-D summary directly to
       ``(B, num_query_tokens, dim_in)`` and L2-normalize each token onto
       S^(dim_in - 1).
    9. Sequence augmentation: concatenate original transformed sequence and
       retrieved tokens along sequence axis -> ``(B, window_size + num_query_tokens, dim_in)``.
   10. Block 2: Causal ALiBi Transformer block over extended sequence.
   11. Sequence projection: project along sequence axis from
       ``window_size + num_query_tokens`` to ``window_size``, flatten to
       ``window_size * dim_in``, and L2-normalize.
   12. Cosine readout with ``target_embeddings`` scaled by learnable ``tau``.
"""

from typing import Any, Dict, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import initializers, layers, ops  # noqa: E402

from model.interfaces import BaseLM  # noqa: E402
from model.archs.alibi_sphere import AlibiCausalSelfAttention  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class AlibiTokenRetrieverSphereLM(BaseLM):
    """Spherical LM with ALiBi attention, token-anchored soft retrieval, and FFN readout."""

    def __init__(
        self,
        vocabulary_size: int = 10000,
        dim_in: int = 8,
        dim_out: int = 128,
        window_size: int = 128,
        hidden_dim: int = 1024,
        num_heads: int = 2,
        initial_tau: float = 10.0,
        initial_retriever_tau: float = 10.0,
        learnable_tau: bool = True,
        activation: str = "gelu",
        embeddings_initializer: str = "normal",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if vocabulary_size <= 0:
            raise ValueError(f"`vocabulary_size` must be positive, got {vocabulary_size}")
        if dim_in <= 0 or dim_out <= 0:
            raise ValueError(f"`dim_in`/`dim_out` must be positive, got {dim_in}/{dim_out}")
        if dim_out % dim_in != 0:
            raise ValueError(f"`dim_out` ({dim_out}) must be divisible by `dim_in` ({dim_in})")
        if window_size <= 0:
            raise ValueError(f"`window_size` must be positive, got {window_size}")

        num_query_tokens = dim_out // dim_in
        if num_query_tokens > window_size:
            raise ValueError(
                f"`dim_out // dim_in` ({num_query_tokens}) cannot exceed `window_size` ({window_size})"
            )
        if hidden_dim < 2:
            raise ValueError(f"`hidden_dim` must be >= 2, got {hidden_dim}")
        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be positive, got {num_heads}")
        if dim_in % num_heads != 0:
            raise ValueError(f"`dim_in` ({dim_in}) must be divisible by `num_heads` ({num_heads})")
        if activation not in ("gelu", "tanh"):
            raise ValueError(f"`activation` must be 'gelu' or 'tanh', got {activation!r}")

        self._vocabulary_size = int(vocabulary_size)
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.window_size = int(window_size)
        self.num_query_tokens = num_query_tokens
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.initial_tau = float(initial_tau)
        self.initial_retriever_tau = float(initial_retriever_tau)
        self.learnable_tau = bool(learnable_tau)
        self.activation = str(activation)
        self.embeddings_initializer = initializers.get(embeddings_initializer)

        # Block 1
        self.attn1 = AlibiCausalSelfAttention(dim=self.dim_in, num_heads=self.num_heads, name="trans1_attn")
        self.trans1_mlp1 = layers.Dense(self.dim_in * 4, activation=self.activation, use_bias=False, name="trans1_mlp1")
        self.trans1_mlp2 = layers.Dense(self.dim_in, activation=None, use_bias=False, name="trans1_mlp2")

        # Query projection: flattened Block-1 output -> dim_in query vector
        self.query_proj = layers.Dense(self.dim_in, activation=None, use_bias=False, name="query_proj")

        # Block 2
        self.attn2 = AlibiCausalSelfAttention(dim=self.dim_in, num_heads=self.num_heads, name="trans2_attn")
        self.trans2_mlp1 = layers.Dense(self.dim_in * 4, activation=self.activation, use_bias=False, name="trans2_mlp1")
        self.trans2_mlp2 = layers.Dense(self.dim_in, activation=None, use_bias=False, name="trans2_mlp2")

        # Sequence projection from (window_size + num_query_tokens) to window_size
        self.seq_proj = layers.Dense(self.window_size, activation=None, use_bias=False, name="seq_proj")

        # Baseline Readout FFN
        self.dense1 = layers.Dense(self.hidden_dim, activation=self.activation, use_bias=False, name="ffn_dense1")
        self.dense2 = layers.Dense(self.hidden_dim // 2, activation=self.activation, use_bias=False, name="ffn_dense2")
        self.dense3 = layers.Dense(self.dim_out, activation=None, use_bias=False, name="ffn_dense3")

        self.build()

    # -- BaseLM contract ------------------------------------------------------

    @property
    def context_length(self) -> int:
        return self.window_size

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    # -- weights --------------------------------------------------------------

    def build(self, input_shape: Optional[Any] = None) -> None:
        if self.built:
            return
        self.input_embeddings = self.add_weight(
            shape=(self._vocabulary_size, self.dim_in),
            initializer=self.embeddings_initializer,
            trainable=True,
            name="input_emb",
        )
        self.target_embeddings = self.add_weight(
            shape=(self._vocabulary_size, self.dim_out),
            initializer=self.embeddings_initializer,
            trainable=True,
            name="target_emb",
        )
        self.tau = self.add_weight(
            shape=(),
            initializer=initializers.Constant(self.initial_tau),
            trainable=self.learnable_tau,
            name="tau",
        )
        self.retriever_tau = self.add_weight(
            shape=(),
            initializer=initializers.Constant(self.initial_retriever_tau),
            trainable=self.learnable_tau,
            name="retriever_tau",
        )

        self.attn1.build((None, self.window_size, self.dim_in))
        self.trans1_mlp1.build((None, self.window_size, self.dim_in))
        self.trans1_mlp2.build((None, self.window_size, self.dim_in * 4))

        self.query_proj.build((None, self.window_size * self.dim_in))

        self.attn2.build((None, self.window_size + self.num_query_tokens, self.dim_in))
        self.trans2_mlp1.build((None, self.window_size + self.num_query_tokens, self.dim_in))
        self.trans2_mlp2.build((None, self.window_size + self.num_query_tokens, self.dim_in * 4))

        self.seq_proj.build((None, self.dim_in, self.window_size + self.num_query_tokens))
        self.dense1.build((None, self.window_size * self.dim_in))
        self.dense2.build((None, self.hidden_dim))
        self.dense3.build((None, self.hidden_dim // 2))
        self.built = True

    def normalized_input_embeddings(self) -> Any:
        """Input table projected onto S^(dim_in - 1)."""
        return ops.normalize(self.input_embeddings, axis=-1, order=2)

    def normalized_target_embeddings(self) -> Any:
        """Target table projected onto S^(dim_out - 1)."""
        return ops.normalize(self.target_embeddings, axis=-1, order=2)

    # -- forward --------------------------------------------------------------

    def call(self, inputs: Any, training: bool = False) -> Any:
        token_ids = ops.cast(inputs, "int32")
        batch_size = ops.shape(token_ids)[0]

        norm_in = self.normalized_input_embeddings()
        norm_tgt = self.normalized_target_embeddings()

        flat = ops.take(norm_in, ops.reshape(token_ids, (-1,)), axis=0)
        x = ops.reshape(flat, (batch_size, self.window_size, self.dim_in))

        # Block 1
        h = ops.normalize(x, axis=-1, order=2)
        x = x + self.attn1(h, training=training)
        h = ops.normalize(x, axis=-1, order=2)
        x = x + self.trans1_mlp2(self.trans1_mlp1(h, training=training), training=training)

        # Query projection: flattened Block-1 output -> dim_in query vector
        flat_x = ops.reshape(x, (batch_size, self.window_size * self.dim_in))
        query = self.query_proj(flat_x, training=training)
        query = ops.normalize(query, axis=-1, order=2)

        # In-context similarity: query vs. each transformed token in x (dim_in space)
        x_norm = ops.normalize(x, axis=-1, order=2)
        token_sim = ops.einsum("bd,bwd->bw", query, x_norm) * ops.cast(self.retriever_tau, query.dtype)
        token_weights = ops.softmax(token_sim, axis=-1)

        # Scatter per-position weights into a (batch_size, vocabulary_size) vector by token id.
        # Offsetting ids by `batch_index * vocabulary_size` keeps every batch row's segment_sum
        # independent while avoiding a dense (batch, window_size, vocabulary_size) one-hot tensor,
        # which at real batch/window/vocab sizes is multiple GiB and blows out device memory.
        batch_idx = ops.reshape(ops.arange(batch_size, dtype="int32"), (batch_size, 1))
        offset_ids = token_ids + batch_idx * self._vocabulary_size
        flat_ids = ops.reshape(offset_ids, (-1,))
        flat_weights = ops.reshape(token_weights, (-1,))
        vocab_weights = ops.segment_sum(flat_weights, flat_ids, num_segments=batch_size * self._vocabulary_size)
        vocab_weights = ops.reshape(vocab_weights, (batch_size, self._vocabulary_size))

        # Soft vocabulary retrieval: weighted pool of the big target table
        v_pooled = ops.matmul(vocab_weights, norm_tgt)
        v_pooled = ops.normalize(v_pooled, axis=-1, order=2)

        # Split pooled vector into num_query_tokens of dimension dim_in
        retrieved_tokens = ops.reshape(v_pooled, (batch_size, self.num_query_tokens, self.dim_in))
        retrieved_tokens = ops.normalize(retrieved_tokens, axis=-1, order=2)

        # Concatenate retrieved tokens onto sequence -> (batch_size, window_size + num_query_tokens, dim_in)
        x2 = ops.concatenate([x, retrieved_tokens], axis=1)

        # Block 2
        h2 = ops.normalize(x2, axis=-1, order=2)
        x2 = x2 + self.attn2(h2, training=training)
        h2 = ops.normalize(x2, axis=-1, order=2)
        x2 = x2 + self.trans2_mlp2(self.trans2_mlp1(h2, training=training), training=training)

        # Sequence projection: (batch_size, window_size + num_query_tokens, dim_in) -> (batch_size, window_size, dim_in)
        x2_t = ops.transpose(x2, (0, 2, 1))
        y_t = self.seq_proj(x2_t, training=training)
        y = ops.transpose(y_t, (0, 2, 1))
        flat_y = ops.reshape(y, (batch_size, self.window_size * self.dim_in))
        u = ops.normalize(flat_y, axis=-1, order=2)

        # Baseline Readout FFN
        h_ffn = self.dense1(u, training=training)
        h_ffn = self.dense2(h_ffn, training=training)
        z = self.dense3(h_ffn, training=training)

        q_final = ops.normalize(z, axis=-1, order=2)
        cos_sim = ops.matmul(q_final, ops.transpose(norm_tgt))
        return cos_sim * ops.cast(self.tau, cos_sim.dtype)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "vocabulary_size": self._vocabulary_size,
            "dim_in": self.dim_in,
            "dim_out": self.dim_out,
            "window_size": self.window_size,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "initial_tau": self.initial_tau,
            "initial_retriever_tau": self.initial_retriever_tau,
            "learnable_tau": self.learnable_tau,
            "activation": self.activation,
            "embeddings_initializer": initializers.serialize(self.embeddings_initializer),
        }
        return {**super().get_config(), **config}
