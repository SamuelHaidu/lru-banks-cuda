"""ALiBi Retriever Sphere LM: Two-stage transformer with soft vocabulary retrieval.

Architecture flow:
    1. Input table ``(V, dim_in)``, L2-normalized onto S^(dim_in - 1).
    2. Gather ``window_size`` causal ids -> ``(B, window_size, dim_in)``.
    3. Block 1: Causal ALiBi Transformer block (L2-norm + causal MHA + L2-norm + MLP).
    4. Query extraction: take the last ``num_query_tokens = dim_out // dim_in`` tokens
       (shape ``(B, num_query_tokens, dim_in)``), reshape directly to ``(B, dim_out)``,
       and L2-normalize to obtain a continuous query vector on S^(dim_out - 1).
    5. Soft vocabulary retrieval: compute cosine similarity with ``target_embeddings``,
       scale by learnable ``retriever_tau``, apply softmax over vocabulary, and pool
       the target embeddings to get a continuous semantic summary vector on S^(dim_out - 1).
    6. Token split: reshape the pooled 128-D summary directly to ``(B, num_query_tokens, dim_in)``
       and L2-normalize each token onto S^(dim_in - 1).
    7. Sequence augmentation: concatenate original transformed sequence and retrieved
       tokens along sequence axis -> ``(B, window_size + num_query_tokens, dim_in)``.
    8. Block 2: Causal ALiBi Transformer block over extended sequence.
    9. Sequence projection: project along sequence axis from ``window_size + num_query_tokens``
       to ``window_size``, flatten to ``window_size * dim_in``, and L2-normalize.
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
class AlibiRetrieverSphereLM(BaseLM):
    """Spherical LM with ALiBi attention, soft vocabulary retrieval, and FFN readout."""

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

        # Query big target table using the last dim_out // dim_in tokens
        last_tokens = x[:, -self.num_query_tokens:, :]
        q = ops.reshape(last_tokens, (batch_size, self.dim_out))
        q_norm = ops.normalize(q, axis=-1, order=2)

        # Softmax retrieval over target embeddings with learnable retriever_tau
        sim = ops.matmul(q_norm, ops.transpose(norm_tgt)) * ops.cast(self.retriever_tau, q_norm.dtype)
        weights = ops.softmax(sim, axis=-1)
        v_pooled = ops.matmul(weights, norm_tgt)
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
