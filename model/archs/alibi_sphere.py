"""ALiBi-Sphere LM: S^(dim_in-1) context -> minimal ALiBi Transformer -> S^(dim_out-1) readout.

Combines an input spherical embedding lookup with a lightweight causal
Transformer block using Attention with Linear Biases (ALiBi, Press et al.),
followed by flattening, L2 spherical normalization, and the baseline
bias-free 3-layer FFN cosine readout.
"""

from typing import Any, Dict, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import initializers, layers, ops  # noqa: E402

from model.interfaces import BaseLM  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class AlibiCausalSelfAttention(layers.Layer):
    """Causal multi-head self-attention with linear positional biases (ALiBi).

    Bias-free projections; adds non-learnable geometric distance penalties
    to the attention matrix, eliminating the need for additive positional
    embeddings while supporting length extrapolation.
    """

    def __init__(self, dim: int, num_heads: int = 2, **kwargs: Any):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = max(1, self.dim // self.num_heads)

        self.q_proj = layers.Dense(self.num_heads * self.head_dim, use_bias=False, name="q_proj")
        self.k_proj = layers.Dense(self.num_heads * self.head_dim, use_bias=False, name="k_proj")
        self.v_proj = layers.Dense(self.num_heads * self.head_dim, use_bias=False, name="v_proj")
        self.out_proj = layers.Dense(self.dim, use_bias=False, name="out_proj")

        # Cache for `_get_alibi_and_mask`, keyed by `(seq_len, dtype)`. The bias
        # depends only on `seq_len`/`num_heads`/`dtype`, all fixed for the
        # lifetime of a model, so recomputing it on every call is pure waste.
        self._bias_cache: Dict[Any, Any] = {}

    def build(self, input_shape: Optional[Any] = None) -> None:
        self.q_proj.build((None, None, self.dim))
        self.k_proj.build((None, None, self.dim))
        self.v_proj.build((None, None, self.dim))
        self.out_proj.build((None, None, self.num_heads * self.head_dim))
        self.built = True

    def _get_alibi_and_mask(self, seq_len: int, dtype: str) -> Any:
        # ALiBi geometric slope sequence: 2^(-8 * (h+1) / H)
        slopes = [2.0 ** (-8.0 * (h + 1) / self.num_heads) for h in range(self.num_heads)]
        slopes_t = ops.convert_to_tensor(slopes, dtype=dtype)

        pos_i = ops.arange(seq_len, dtype=dtype)
        pos_j = ops.arange(seq_len, dtype=dtype)
        # rel_pos[i, j] = j - i (non-positive for causal positions j <= i)
        rel_pos = ops.expand_dims(pos_j, 0) - ops.expand_dims(pos_i, 1)

        # alibi bias: (1, num_heads, seq_len, seq_len)
        alibi = ops.expand_dims(slopes_t, (1, 2)) * ops.expand_dims(rel_pos, 0)
        alibi = ops.expand_dims(alibi, 0)

        # Causal mask: 0 for j <= i, -1e9 for future positions j > i
        mask = ops.triu(ops.full((seq_len, seq_len), -1e9, dtype=dtype), k=1)
        mask = ops.reshape(mask, (1, 1, seq_len, seq_len))
        return alibi, mask

    def _get_combined_bias(self, seq_len: int, dtype: str) -> Any:
        """`alibi + causal mask`, memoized per `(seq_len, dtype)`.

        Neither operand carries gradients (ALiBi slopes are a fixed geometric
        sequence, the mask is a constant), so caching the sum is safe to reuse
        across calls and across training steps.
        """
        key = (int(seq_len), str(dtype))
        cached = self._bias_cache.get(key)
        if cached is None:
            alibi, mask = self._get_alibi_and_mask(seq_len, dtype)
            cached = ops.stop_gradient(alibi + mask)
            self._bias_cache[key] = cached
        return cached

    def call(self, x: Any, training: bool = False) -> Any:
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]

        # Single matmul against the concatenated q/k/v kernels instead of
        # three separate Dense dispatches -- same weights, same math, fewer
        # GPU kernel launches for a `dim` this small.
        qkv_kernel = ops.concatenate([self.q_proj.kernel, self.k_proj.kernel, self.v_proj.kernel], axis=-1)
        qkv = ops.matmul(x, qkv_kernel)
        width = self.num_heads * self.head_dim
        q, k, v = qkv[..., :width], qkv[..., width : 2 * width], qkv[..., 2 * width :]

        # `ops.dot_product_attention` expects (batch, seq_len, num_heads, head_dim).
        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        scale = 1.0 / (float(self.head_dim) ** 0.5)
        # Bias must be broadcastable to (batch, num_heads, seq_len, seq_len).
        bias = self._get_combined_bias(seq_len, q.dtype)

        out = ops.dot_product_attention(q, k, v, bias=bias, is_causal=False, scale=scale)

        # Recombine heads -> (batch, seq_len, num_heads * head_dim)
        out = ops.reshape(out, (batch_size, seq_len, self.num_heads * self.head_dim))
        return self.out_proj(out)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "dim": self.dim,
            "num_heads": self.num_heads,
        }
        return {**super().get_config(), **config}


@keras.saving.register_keras_serializable(package="model")
class AlibiSphereLM(BaseLM):
    """Bias-free LM: S^(dim_in-1) tokens -> ALiBi Transformer -> S^(dim_out-1) readout.

    Forward pass:
        1. Input table ``(V, dim_in)``, L2-normalized -> S^(dim_in - 1)
        2. Gather ``window_size`` causal ids -> ``(B, window_size, dim_in)``
        3. Lightweight ALiBi Transformer block (Pre-RMSNorm + causal MHA + Pre-RMSNorm + MLP)
        4. Flatten transformed tokens -> ``(B, window_size * dim_in)``
        5. L2-normalize flattened representation
        6. Baseline FFN: ``hidden_dim`` GELU -> ``hidden_dim // 2`` GELU -> ``dim_out`` linear
        7. L2-normalize -> query on S^(dim_out - 1)
        8. Target table ``(V, dim_out)``, L2-normalized
        9. ``logits = (q @ target.T) * tau``

    Args:
        vocabulary_size: Token vocabulary size (width of the logits).
        dim_in: Dimensionality of the context manifold S^(dim_in - 1).
        dim_out: Dimensionality of the readout manifold S^(dim_out - 1).
        window_size: Causal lookback in tokens.
        hidden_dim: Width of the first FFN layer; second is half of it.
        num_heads: Number of attention heads in the ALiBi Transformer block.
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

        self.attn = AlibiCausalSelfAttention(dim=self.dim_in, num_heads=self.num_heads, name="trans_attn")
        self.trans_mlp1 = layers.Dense(self.dim_in * 4, activation="gelu", use_bias=False, name="trans_mlp1")
        self.trans_mlp2 = layers.Dense(self.dim_in, activation=None, use_bias=False, name="trans_mlp2")

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
        self.attn.build((None, self.window_size, self.dim_in))
        self.trans_mlp1.build((None, self.window_size, self.dim_in))
        self.trans_mlp2.build((None, self.window_size, self.dim_in * 4))
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

        # Minimal causal transformer block with ALiBi
        h = ops.normalize(x, axis=-1, order=2)
        x = x + self.attn(h, training=training)
        h = ops.normalize(x, axis=-1, order=2)
        x = x + self.trans_mlp2(self.trans_mlp1(h, training=training), training=training)

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
