"""RoPE kernel attention: continuous-kernel context, then FFN-query attention.

    sequence  v_k = R_k E[t_{n-k}]                     tokens -> RoPE, computed once
    context   h0  = sum_k a_k v_k                      fixed learned kernel
    alpha         = MLP_1(h0)                          basis amplitudes (`kernel_basis`)
    refined   h1  = h0 + sum_k w_k(alpha) v_k          function-generated kernel
    query     q   = MLP_2(h1)                          (dim,), split into `num_heads`
    attention p   = softmax_k(tau_h * cos(q_h, v_{k,h}))   per head; keys = values = sequence
              a   = concat_h sum_k p_{h,k} v_{k,h}
    logits        = W_out^T a                          or, with `attention_residual`,
                  = W_out^T (h1 + g a)                 g: learned scalar, 0 at init (ReZero)

Stages 1-2 are `RopeLogGaussianSumLM`'s (`kernels()`), so `h1` is exactly that
model's pre-readout vector. There are no key/value projections: the rotated
token sequence is both. Because keys are RoPE-rotated by distance from the
predicted position, the query matches "this content at about this distance".

Scores use QK-norm: query and keys are L2-normalized per head and scaled by a
learned per-head temperature `tau_h` (stored as log, initialized to
`log(10)`). Unnormalized `q . v / sqrt(d_h)` let the query norm explode
(~24k after one epoch), which saturated the softmax into fixed one-hot heads
with no gradient; with QK-norm the scores are bounded by `tau_h`.

`key_value_projection=True` adds per-read key and value projections
(`dim x dim`, identity at init): keys `R_k W_K e`, values `W_V e` (no
rotation), so the query is matched against keys and the attention
aggregates the values. Key norms are then taken from the projected keys.

Heads split the embedding along RoPE pairs: RoPE rotates dimension `i` with
`i + dim/2`, so head `h` takes the same slice from both halves. The slices
follow RoPE's frequency order, so head 0 sees the fastest-rotating (most
position-sensitive) pairs and the last head the slowest (nearly
position-free, content matching).
"""

from typing import Any, Dict, Optional, Tuple

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402

from model.archs._dense import rotate  # noqa: E402
from model.archs.rope_log_gaussian_sum import RopeLogGaussianSumLM  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class RopeKernelAttentionLM(RopeLogGaussianSumLM):
    """`RopeLogGaussianSumLM` + one FFN-query attention read over the token sequence.

    Args (beyond `RopeLogGaussianSumLM`'s):
        num_heads: Attention heads; `dim / 2` must be divisible by it.
        query_hidden: Hidden width of the query SwiGLU MLP.
        attention_residual: If True, logits read `h1 + g * a` (learned gate `g`
            starting at 0, so training starts as the kernel-only model)
            instead of `a` alone. Without it the kernel stages only get gradient
            through the query, and they barely train.
        key_value_projection: If True, each attention read projects the tokens
            to keys (then RoPE) and values with its own `dim x dim` matrices.
    """

    _defer_build = True  # parent must not build before the query layers exist
    _defer_build_attention = False  # set by subclasses that add their own layers

    def __init__(
        self,
        vocabulary_size: int = 10000,
        dim: int = 128,
        window_size: int = 128,
        rope_base: float = 10000.0,
        mlp_hidden: int = 128,
        num_centers: int = 8,
        kernel_basis: str = "damped-oscillator",
        basis_window: Optional[int] = None,
        readout_hidden: Optional[int] = None,
        num_heads: int = 4,
        query_hidden: int = 128,
        attention_residual: bool = False,
        key_value_projection: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            vocabulary_size=vocabulary_size,
            dim=dim,
            window_size=window_size,
            rope_base=rope_base,
            mlp_hidden=mlp_hidden,
            num_centers=num_centers,
            kernel_basis=kernel_basis,
            basis_window=basis_window,
            readout_hidden=readout_hidden,
            **kwargs,
        )
        if num_heads <= 0 or (self.dim // 2) % num_heads != 0:
            raise ValueError(f"`dim / 2` ({self.dim // 2}) must be divisible by `num_heads` ({num_heads})")
        if query_hidden <= 0:
            raise ValueError(f"`query_hidden` must be positive, got {query_hidden}")
        self.num_heads = int(num_heads)
        self.query_hidden = int(query_hidden)
        self.attention_residual = bool(attention_residual)
        self.key_value_projection = bool(key_value_projection)
        # One (key, value) pair per attention read; subclasses with more reads
        # append theirs before build().
        self.key_proj: list = []
        self.value_proj: list = []
        if self.key_value_projection:
            self._add_key_value("")
        self._pair_chunk = self.dim // (2 * self.num_heads)  # dims per head per RoPE half

        # Query MLP: same 3-layer SwiGLU shape as the kernel MLP, output = dim.
        self.query_gate_1 = layers.Dense(self.query_hidden, use_bias=False, name="query_gate_1")
        self.query_up_1 = layers.Dense(self.query_hidden, use_bias=False, name="query_up_1")
        self.query_gate_2 = layers.Dense(self.query_hidden, use_bias=False, name="query_gate_2")
        self.query_up_2 = layers.Dense(self.query_hidden, use_bias=False, name="query_up_2")
        self.query_out = layers.Dense(self.dim, use_bias=False, name="query_out")
        if not self._defer_build_attention:
            self.build()

    def _add_key_value(self, prefix: str) -> None:
        dense = lambda name: layers.Dense(self.dim, use_bias=False, kernel_initializer="identity", name=name)
        self.key_proj.append(dense(f"{prefix}key_proj"))
        self.value_proj.append(dense(f"{prefix}value_proj"))

    def build(self, input_shape: Any = None) -> None:
        if self.built:
            return
        for layer in self.key_proj + self.value_proj:
            layer.build((None, self.dim))
        self.log_temperature = self.add_weight(
            name="log_temperature",
            shape=(self.num_heads,),
            initializer=keras.initializers.Constant(float(np.log(10.0))),
        )
        if self.attention_residual:
            self.attention_gate = self.add_weight(name="attention_gate", shape=(), initializer="zeros")
        self.query_gate_1.build((None, self.state_dim))
        self.query_up_1.build((None, self.state_dim))
        self.query_gate_2.build((None, self.query_hidden))
        self.query_up_2.build((None, self.query_hidden))
        self.query_out.build((None, self.query_hidden))
        super().build(input_shape)  # parent layers + `distance_weights`; sets `built`

    # -- forward --------------------------------------------------------------

    def _head_major(self, x: Any) -> Any:
        """`(B, L, dim)` -> `(B, H, L, 2c)`: head `h` takes its slice of both RoPE halves."""
        ops = keras.ops
        H, c = self.num_heads, self._pair_chunk
        L = x.shape[1]
        return ops.reshape(ops.transpose(ops.reshape(x, (-1, L, 2, H, c)), (0, 3, 1, 2, 4)), (-1, H, L, 2 * c))

    def _head_norms(self, k: Any) -> Any:
        """Per-head L2 norms `(B, H, L)` of `(B, L, dim)` keys (rotation-invariant)."""
        ops = keras.ops
        kh = ops.reshape(k, (-1, k.shape[1], 2, self.num_heads, self._pair_chunk))
        return ops.transpose(ops.sqrt(ops.sum(kh * kh, axis=(2, 4))), (0, 2, 1)) + 1e-6

    def _key_values(self, read: int, inputs: Any) -> Tuple[Any, Any, Any]:
        """Single-target `(keys (B, H, T, 2c), values (B, H, T, 2c), key norms (B, H, T))`
        for attention read `read` (0 = hop 1) with key/value projections."""
        x = self.token_embedding(keras.ops.cast(inputs, "int32"))
        cos, sin, _ = self._constants(x)
        k = rotate(self.key_proj[read](x), cos, sin)
        return self._head_major(k), self._head_major(self.value_proj[read](x)), self._head_norms(k)

    def _attend(self, q: Any, rotated: Any, inputs: Any, log_temperature: Any, read: int = 0) -> Tuple[Any, Any]:
        """One FFN-query attention read over the rotated window.

        `q` `(B, dim)`, `rotated` `(B, T, dim)`; returns `(a (B, dim), probs (B, H, T))`.
        `read` picks the key/value projections when `key_value_projection` is on.
        """
        ops = keras.ops
        T, H, c = self.window_size, self.num_heads, self._pair_chunk
        eps = 1e-6
        if self.key_value_projection:
            seq, values, k_norm = self._key_values(read, inputs)
        else:
            # Head-major layout (B, H, T, 2c), copied once, so scores and pooling are
            # plain batched matmuls instead of einsums that copy `seq` internally.
            seq = values = self._head_major(rotated)
        qh = ops.reshape(q, (-1, 2, H, c))
        # QK-norm: per-head unit query; key norms divided out of the scores.
        q_unit = qh / (ops.sqrt(ops.sum(qh * qh, axis=(1, 3), keepdims=True)) + eps)
        q_unit = ops.reshape(ops.transpose(q_unit, (0, 2, 1, 3)), (-1, H, 2 * c, 1))
        # RoPE preserves each pair's norm and a head holds whole pairs, so a key's
        # per-head norm equals its raw embedding row's: computed on the (V, H)
        # table instead of normalizing the (B, T, dim) sequence.
        if not self.key_value_projection:
            table = ops.reshape(self.token_embedding.embeddings, (-1, 2, H, c))
            row_norms = ops.sqrt(ops.sum(table * table, axis=(1, 3))) + eps  # (V, H)
            k_norm = ops.transpose(ops.take(row_norms, ops.cast(inputs, "int32"), axis=0), (0, 2, 1))  # (B, H, T)
        tau = ops.exp(ops.cast(log_temperature, q.dtype))[None, :, None]  # (1, H, 1)
        scores = ops.squeeze(ops.matmul(seq, q_unit), -1) / k_norm * tau
        probs = ops.softmax(scores, axis=-1)  # (B, H, T)
        a = ops.matmul(probs[:, :, None, :], values)  # (B, H, 1, 2c)
        a = ops.transpose(ops.reshape(a, (-1, H, 2, c)), (0, 2, 1, 3))  # back to (B, half, head, chunk)
        return ops.reshape(a, (-1, self.dim)), probs

    def attention(self, inputs: Any) -> Tuple[Any, Any, Any]:
        """Returns `(a, probs, h1)`: attention vector `(B, dim)`, weights `(B, H, T)`
        aligned with the context axis (last index = n-1), and the refined context."""
        ops = keras.ops
        rotated, h0, _, adaptive_w = self.kernels(inputs)
        h1 = h0 + ops.einsum("bt,btd->bd", adaptive_w, rotated)
        q = self._query(h1)  # (B, dim)

        a, probs = self._attend(q, rotated, inputs, self.log_temperature)
        return a, probs, h1

    def _combine(self, h1: Any, a: Any) -> Any:
        if self.attention_residual:
            return h1 + keras.ops.cast(self.attention_gate, a.dtype) * a
        return a

    def _query(self, h1: Any) -> Any:
        ops = keras.ops
        z = ops.silu(self.query_gate_1(h1)) * self.query_up_1(h1)
        z = ops.silu(self.query_gate_2(z)) * self.query_up_2(z)
        return self.query_out(z)

    def hidden(self, inputs: Any) -> Any:
        """Pre-readout vector `(B, dim)` for single-target windows."""
        a, _, h1 = self.attention(inputs)
        return self._combine(h1, a)

    def call(self, inputs: Any, training: bool = False) -> Any:
        return self.readout(self.readout_input(self.hidden(inputs)))

    def _dense_attend(self, q: Any, u: Any, inputs: Any, rope: Any, log_temperature: Any, read: int = 0) -> Tuple[Any, Any]:
        """Dense counterpart of `_attend`: `q` `(B, T, dim)` in the query frame, `u`
        `(B, 2T-1, dim)` unrotated keys. Returns `(a (B, T, dim), probs (B, H, T, 2T-1))`."""
        ops = keras.ops
        T, H, c = self.window_size, self.num_heads, self._pair_chunk
        S = 2 * T - 1
        eps = 1e-6
        q = rope.from_query_frame(q)  # (B, T, dim)
        if self.key_value_projection:
            x = self.token_embedding(ops.cast(inputs, "int32"))
            k = self.key_proj[read](x)
            keys = self._head_major(rope.unrotate_keys(k))
            values = self._head_major(self.value_proj[read](x))
            k_norm = self._head_norms(k)[:, :, None, :]
        else:
            keys = values = self._head_major(u)
        qh = ops.reshape(q, (-1, T, 2, H, c))
        qh = qh / (ops.sqrt(ops.sum(qh * qh, axis=(2, 4), keepdims=True)) + eps)
        q_unit = ops.reshape(ops.transpose(qh, (0, 3, 1, 2, 4)), (-1, H, T, 2 * c))
        if not self.key_value_projection:
            table = ops.reshape(self.token_embedding.embeddings, (-1, 2, H, c))
            row_norms = ops.sqrt(ops.sum(table * table, axis=(1, 3))) + eps  # (V, H)
            k_norm = ops.transpose(ops.take(row_norms, ops.cast(inputs, "int32"), axis=0), (0, 2, 1))[:, :, None, :]
        tau = ops.exp(ops.cast(log_temperature, q.dtype))[None, :, None, None]
        scores = ops.matmul(q_unit, ops.transpose(keys, (0, 1, 3, 2))) / k_norm * tau  # (B, H, T, S)
        scores = ops.where(rope.mask[None, None], scores, ops.cast(float("-inf"), scores.dtype))
        probs = ops.softmax(scores, axis=-1)
        a = ops.matmul(probs, values)  # (B, H, T, 2c)
        a = ops.reshape(ops.transpose(ops.reshape(a, (-1, H, T, 2, c)), (0, 2, 3, 1, 4)), (-1, T, self.dim))
        # Unprojected values are the rotated tokens: back to the query frame.
        # Projected values carry no rotation.
        return (a if self.key_value_projection else rope.to_query_frame(a)), probs

    def dense_attention(self, inputs: Any) -> Tuple[Any, Any, Any]:
        """Dense counterpart of `attention` for a `(B, 2T-1)` input.

        Returns `(a, probs, h1)`: attention vectors `(B, T, dim)`, banded
        weights `(B, H, T, 2T-1)` (query `m` sees inputs `m..m+T-1`), and the
        refined context `(B, T, dim)`. Scores use `(R_{-(i+1)} q) . u_j`, the
        absolute-position form of the single-target `q . R_k e_j`.
        """
        u, h1, rope = self.dense_kernels(inputs)
        a, probs = self._dense_attend(self._query(h1), u, inputs, rope, self.log_temperature)
        return a, probs, h1

    def dense_hidden(self, inputs: Any) -> Any:
        """`(B, T, dim)` pre-readout vectors for all `T` queries of a `(B, 2T-1)` input."""
        a, _, h1 = self.dense_attention(inputs)
        return self._combine(h1, a)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "num_heads": self.num_heads,
            "query_hidden": self.query_hidden,
            "attention_residual": self.attention_residual,
            "key_value_projection": self.key_value_projection,
        }
        return {**super().get_config(), **config}
