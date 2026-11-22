"""ALiBi-Sphere-Two-Block LM: S^(dim_in-1) context -> two stacked ALiBi
Transformer blocks -> S^(dim_out-1) readout.

Identical to `alibi-sphere` except the single ALiBi Transformer block is
replaced by TWO of them, stacked sequentially (Pre-RMSNorm + causal MHA +
Pre-RMSNorm + MLP, twice), each with its own independent weights. No
retrieval mechanism -- unlike `alibi-retriever-sphere`, block 2's output goes
straight to the flatten-and-FFN readout, it never queries a target/token
table mid-forward-pass.
"""

from typing import Any, Dict, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import initializers, layers, ops  # noqa: E402

from model.archs.alibi_sphere import AlibiCausalSelfAttention  # noqa: E402
from model.interfaces import BaseLM  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class AlibiSphereTwoBlockLM(BaseLM):
    """Bias-free LM: S^(dim_in-1) tokens -> two ALiBi Transformer blocks -> S^(dim_out-1) readout.

    Forward pass:
        1. Input table ``(V, dim_in)``, L2-normalized -> S^(dim_in - 1)
        2. Gather ``window_size`` causal ids -> ``(B, window_size, dim_in)``
        3. ALiBi Transformer block 1 (Pre-RMSNorm + causal MHA + Pre-RMSNorm + MLP)
        4. ALiBi Transformer block 2, independent weights, same shape as block 1
        5. Flatten transformed tokens -> ``(B, window_size * dim_in)``
        6. L2-normalize flattened representation
        7. Baseline FFN: ``hidden_dim`` GELU -> ``hidden_dim // 2`` GELU -> ``dim_out`` linear
        8. L2-normalize -> query on S^(dim_out - 1)
        9. Target table ``(V, dim_out)``, L2-normalized
        10. ``logits = (q @ target.T) * tau``

    Args:
        vocabulary_size: Token vocabulary size (width of the logits).
        dim_in: Dimensionality of the context manifold S^(dim_in - 1).
        dim_out: Dimensionality of the readout manifold S^(dim_out - 1).
        window_size: Causal lookback in tokens.
        hidden_dim: Width of the first FFN layer; second is half of it.
        num_heads: Number of attention heads in EACH ALiBi Transformer block.
        initial_tau: Initial cosine-readout temperature.
        learnable_tau: Whether ``tau`` trains.
        embeddings_initializer: Initializer for both tables.
    """

    def __init__(
        self,
        vocabulary_size: int = 10000,
        dim_in: int = 8,
        dim_out: int = 128,
        window_size: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 2,
        initial_tau: float = 9.6158,
        learnable_tau: bool = True,
        embeddings_initializer: Any = "uniform",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if vocabulary_size <= 0:
            raise ValueError(f"`vocabulary_size` must be positive, got {vocabulary_size}")
        if dim_in <= 0 or dim_out <= 0:
            raise ValueError(f"`dim_in`/`dim_out` must be positive, got {dim_in}/{dim_out}")
        if window_size <= 0:
            raise ValueError(f"`window_size` must be positive, got {window_size}")
        if hidden_dim < 2:
            raise ValueError(f"`hidden_dim` must be >= 2, got {hidden_dim}")
        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be positive, got {num_heads}")

        self._vocabulary_size = int(vocabulary_size)
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.initial_tau = float(initial_tau)
        self.learnable_tau = bool(learnable_tau)
        self.embeddings_initializer = initializers.get(embeddings_initializer)

        self.attn1 = AlibiCausalSelfAttention(dim=self.dim_in, num_heads=self.num_heads, name="trans1_attn")
        self.trans1_mlp1 = layers.Dense(self.dim_in * 4, activation="gelu", use_bias=False, name="trans1_mlp1")
        self.trans1_mlp2 = layers.Dense(self.dim_in, activation=None, use_bias=False, name="trans1_mlp2")

        self.attn2 = AlibiCausalSelfAttention(dim=self.dim_in, num_heads=self.num_heads, name="trans2_attn")
        self.trans2_mlp1 = layers.Dense(self.dim_in * 4, activation="gelu", use_bias=False, name="trans2_mlp1")
        self.trans2_mlp2 = layers.Dense(self.dim_in, activation=None, use_bias=False, name="trans2_mlp2")

        self.dense1 = layers.Dense(self.hidden_dim, activation="gelu", use_bias=False, name="ffn_dense1")
        self.dense2 = layers.Dense(self.hidden_dim // 2, activation="gelu", use_bias=False, name="ffn_dense2")
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
        self.attn1.build((None, self.window_size, self.dim_in))
        self.trans1_mlp1.build((None, self.window_size, self.dim_in))
        self.trans1_mlp2.build((None, self.window_size, self.dim_in * 4))
        self.attn2.build((None, self.window_size, self.dim_in))
        self.trans2_mlp1.build((None, self.window_size, self.dim_in))
        self.trans2_mlp2.build((None, self.window_size, self.dim_in * 4))
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
        flat = ops.take(norm_in, ops.reshape(token_ids, (-1,)), axis=0)
        x = ops.reshape(flat, (batch_size, self.window_size, self.dim_in))

        # Block 1
        h = ops.normalize(x, axis=-1, order=2)
        x = x + self.attn1(h, training=training)
        h = ops.normalize(x, axis=-1, order=2)
        x = x + self.trans1_mlp2(self.trans1_mlp1(h, training=training), training=training)

        # Block 2 -- independent weights, same shape as block 1.
        h = ops.normalize(x, axis=-1, order=2)
        x = x + self.attn2(h, training=training)
        h = ops.normalize(x, axis=-1, order=2)
        x = x + self.trans2_mlp2(self.trans2_mlp1(h, training=training), training=training)

        # Concat / flatten transformed tokens, L2-normalize
        u = ops.reshape(x, (batch_size, self.window_size * self.dim_in))
        u = ops.normalize(u, axis=-1, order=2)

        # Baseline FFN
        h = self.dense1(u, training=training)
        h = self.dense2(h, training=training)
        z = self.dense3(h, training=training)

        q = ops.normalize(z, axis=-1, order=2)
        cos_sim = ops.matmul(q, ops.transpose(self.normalized_target_embeddings()))
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
            "learnable_tau": self.learnable_tau,
            "embeddings_initializer": initializers.serialize(self.embeddings_initializer),
        }
        return {**super().get_config(), **config}
