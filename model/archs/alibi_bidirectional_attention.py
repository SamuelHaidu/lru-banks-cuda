"""Bidirectional variant of `AlibiCausalSelfAttention` (`alibi_sphere.py`).

Identical in every respect -- same ALiBi slopes, same bias-free projections,
same fused qkv matmul, same `(seq_len, dtype)`-keyed bias cache -- except the
causal `-1e9` triangular mask is dropped, so every position can attend to
every other position (ALiBi's distance bias still applies, unmasked).

Why this exists: this project's training objective supervises a single
target token per window (`VariablePrefixDataset` in `model/data.py`), not a
dense per-position next-token loss. The target token is never placed inside
the context window at all, so there is no "seeing the answer early" leakage
risk that causal masking exists to prevent -- causal masking's usual
justification does not apply to this training setup. This module makes that
choice explicit and swappable without touching the causal layer or any
architecture already built on it.
"""

from typing import Any, Dict, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import layers, ops  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class AlibiBidirectionalSelfAttention(layers.Layer):
    """Bidirectional multi-head self-attention with linear positional biases (ALiBi).

    Bias-free projections; adds non-learnable geometric distance penalties
    to the attention matrix (same as `AlibiCausalSelfAttention`), but every
    position may attend to every other position -- no causal mask.
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

        # Cache for `_get_alibi_bias`, keyed by `(seq_len, dtype)`. The bias
        # depends only on `seq_len`/`num_heads`/`dtype`, all fixed for the
        # lifetime of a model, so recomputing it on every call is pure waste.
        self._bias_cache: Dict[Any, Any] = {}

    def build(self, input_shape: Optional[Any] = None) -> None:
        self.q_proj.build((None, None, self.dim))
        self.k_proj.build((None, None, self.dim))
        self.v_proj.build((None, None, self.dim))
        self.out_proj.build((None, None, self.num_heads * self.head_dim))
        self.built = True

    def _get_alibi_bias(self, seq_len: int, dtype: str) -> Any:
        # ALiBi geometric slope sequence: 2^(-8 * (h+1) / H)
        slopes = [2.0 ** (-8.0 * (h + 1) / self.num_heads) for h in range(self.num_heads)]
        slopes_t = ops.convert_to_tensor(slopes, dtype=dtype)

        pos_i = ops.arange(seq_len, dtype=dtype)
        pos_j = ops.arange(seq_len, dtype=dtype)
        # rel_pos[i, j] = j - i, unrestricted (no causal j <= i requirement)
        rel_pos = ops.expand_dims(pos_j, 0) - ops.expand_dims(pos_i, 1)

        # alibi bias: (1, num_heads, seq_len, seq_len). Unlike the causal
        # variant, distances are used unsigned so the penalty for attending
        # to a position `d` steps away is symmetric in both directions --
        # ALiBi's original bias is only ever applied to non-positive
        # (backward-looking) relative positions, so extending it verbatim to
        # positive (forward-looking) ones would make it asymmetric for no
        # principled reason.
        alibi = ops.expand_dims(slopes_t, (1, 2)) * -ops.abs(ops.expand_dims(rel_pos, 0))
        alibi = ops.expand_dims(alibi, 0)
        return alibi

    def _get_combined_bias(self, seq_len: int, dtype: str) -> Any:
        """ALiBi bias only, memoized per `(seq_len, dtype)` -- no mask term."""
        key = (int(seq_len), str(dtype))
        cached = self._bias_cache.get(key)
        if cached is None:
            alibi = self._get_alibi_bias(seq_len, dtype)
            cached = ops.stop_gradient(alibi)
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
