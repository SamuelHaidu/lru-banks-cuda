"""RoPE logit sum: a single linear layer, weighted additive context.

    logits = sum_k  a_k  W_out^T  R_k  E[t_{n-k}],   k = 1..window_size

Every context token is looked up in a shared input table `E`, rotated by RoPE
according to its distance `k` from the predicted position (`k = 1` is the
n-1 / Markov-1 token), and read out through a shared output table `W_out`.
The per-token logit vectors are summed with one learned scalar `a_k` per
distance (initialized to `1/k`, so the model starts recency-biased): no gating,
no normalization, no nonlinearity before the softmax.

Because everything is linear, summing logits equals summing the rotated
embeddings and projecting once, so the sum is done in `dim` space and only
one `(dim, vocab)` matmul is paid per sample.

`per_dim_kernels=True` learns one kernel per embedding dimension
(`a_{k,d}`, a `(window, dim)` table): a depthwise causal convolution of the
rotated tokens. A per-dimension weight does not commute with the RoPE
rotation, so the dense path splits each pair `(p, p + dim/2)` into four
banded sums (both kernels over both unrotated halves), twice the work of the
shared kernel.

`kernel_param` picks how the shared kernel is made: `"free"` (one weight per
distance), or `num_oscillators` terms of

    "exp-oscillators":  a_n e^{-alpha_n k} cos(omega_n k + phi_n)        (SSM-style)
    "log-oscillators":  a_n k^{-sigma_n} cos(gamma_n log k + phi_n)      (explicit-formula style)

with every `alpha / sigma, omega / gamma, a, phi` learned; amplitudes start
as a least-squares fit to `1/k`. The parameters do not depend on the window.

`memory_hidden=N` adds a key-value memory MLP after the sum, per position:

    h = c + W_2 gelu(W_1 RMSNorm(c) + b_1) + b_2       (dim -> N -> dim)

`W_2`, `b_2` start at zero, so training starts as the linear model. It is
pointwise, so dense supervision stays one parallel pass.
"""

from typing import Any, Dict, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402

from model.archs import _context  # noqa: E402
from model.archs._dense import DenseRope, DenseSupervisionMixin, dense_kernel_sum, skew_band  # noqa: E402
from model.archs._norm import RMSNorm  # noqa: E402
from model.interfaces import BaseLM  # noqa: E402


KERNEL_PARAMS = ("free", "exp-oscillators", "log-oscillators")


def _oscillator_basis(kind: str, k: np.ndarray, rate: np.ndarray, freq: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """`(len(k), n)` oscillator terms (unit amplitude) at distances `k`."""
    k = k[:, None]
    if kind == "exp-oscillators":
        return np.exp(-rate * k) * np.cos(freq * k + phase)
    return k ** (-rate) * np.cos(freq * np.log(k) + phase)


@keras.saving.register_keras_serializable(package="model")
class RopeLogitSumLM(DenseSupervisionMixin, BaseLM):
    """Shared skip-gram-style tables with RoPE distance rotation and a weighted sum.

    Args:
        vocabulary_size: Token vocabulary size (width of the logits).
        dim: Embedding dimension (must be even for RoPE).
        window_size: Context length.
        rope_base: RoPE frequency base.
        per_dim_kernels: One distance kernel per embedding dimension instead of
            one shared by all.
        memory_hidden: Hidden width of the residual GELU memory MLP (None: off).
        kernel_param: `"free"`, `"exp-oscillators"` or `"log-oscillators"`.
        num_oscillators: Terms in an oscillator kernel.
    """

    def __init__(
        self,
        vocabulary_size: int = 10000,
        dim: int = 512,
        window_size: int = 64,
        rope_base: float = 10000.0,
        per_dim_kernels: bool = False,
        memory_hidden: Optional[int] = None,
        kernel_param: str = "free",
        num_oscillators: int = 8,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if vocabulary_size <= 0:
            raise ValueError(f"`vocabulary_size` must be positive, got {vocabulary_size}")
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"`dim` must be positive and even, got {dim}")
        if window_size <= 0:
            raise ValueError(f"`window_size` must be positive, got {window_size}")

        self._vocabulary_size = int(vocabulary_size)
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.rope_base = float(rope_base)
        self.per_dim_kernels = bool(per_dim_kernels)
        if memory_hidden is not None and int(memory_hidden) <= 0:
            raise ValueError(f"`memory_hidden` must be positive, got {memory_hidden}")
        self.memory_hidden = int(memory_hidden) if memory_hidden is not None else None
        if kernel_param not in KERNEL_PARAMS:
            raise ValueError(f"`kernel_param` must be one of {KERNEL_PARAMS}, got {kernel_param!r}")
        if kernel_param != "free" and self.per_dim_kernels:
            raise ValueError("oscillator kernels are shared-channel only (per_dim_kernels=False)")
        if int(num_oscillators) <= 0:
            raise ValueError(f"`num_oscillators` must be positive, got {num_oscillators}")
        self.kernel_param = str(kernel_param)
        self.num_oscillators = int(num_oscillators)

        # Distance from the predicted position: context index j -> k = T - j,
        # so the last context token (n-1) gets k = 1.
        half = self.dim // 2
        inv_freq = 1.0 / (self.rope_base ** (np.arange(half, dtype=np.float64) / half))
        distance = np.arange(self.window_size, 0, -1, dtype=np.float64)
        angles = distance[:, None] * inv_freq[None, :]  # (T, dim/2)
        self._cos = np.cos(angles).astype("float32")
        self._sin = np.sin(angles).astype("float32")
        self._distance = distance.astype("float32")  # (T,), k for each context index

        self.token_embedding = layers.Embedding(self._vocabulary_size, self.dim, name="token_emb")
        self.readout = layers.Dense(self._vocabulary_size, use_bias=False, name="readout")
        if self.memory_hidden is not None:
            self.memory_norm = RMSNorm(name="memory_norm")
            self.memory_in = layers.Dense(self.memory_hidden, activation="gelu", name="memory_in")
            self.memory_out = layers.Dense(self.dim, kernel_initializer="zeros", name="memory_out")

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
        self.token_embedding.build((None, self.window_size))
        self.readout.build((None, self.dim))
        if self.memory_hidden is not None:
            self.memory_norm.build((None, self.dim))
            self.memory_in.build((None, self.dim))
            self.memory_out.build((None, self.memory_hidden))
        if self.kernel_param != "free":
            self._build_oscillators()
            self.built = True
            return
        # One learned scalar per distance (per dimension with `per_dim_kernels`),
        # aligned with the context axis.
        init = (1.0 / self._distance).astype("float32")
        shape = (self.window_size, self.dim) if self.per_dim_kernels else (self.window_size,)
        init = np.broadcast_to(init[:, None], shape).copy() if self.per_dim_kernels else init
        self.distance_weights = self.add_weight(
            name="distance_weights",
            shape=shape,
            initializer=lambda shape, dtype=None: keras.ops.convert_to_tensor(init, dtype=dtype),
        )
        self.built = True

    def _build_oscillators(self) -> None:
        n, T = self.num_oscillators, self.window_size
        # Frequencies start nonzero: at freq = phase = 0 their gradients vanish.
        if self.kernel_param == "exp-oscillators":
            tau = np.geomspace(1.0, T, n)  # timescales 1..T tokens
            rate, freq = 1.0 / tau, 0.5 / tau  # alpha; omega (rad / token), half a radian per timescale
        elif n == 1:
            rate, freq = np.array([1.0]), np.array([0.1])  # one term: 1/k with a slight log-wobble
        elif n <= 3:  # a few terms: all near 1/k, the wide spread below fits it badly
            rate, freq = np.linspace(0.9, 1.1, n), np.linspace(0.1, 0.4, n)
        else:
            rate = np.linspace(0.5, 1.5, n)  # sigma: power-law exponents
            freq = np.linspace(0.25, 2.0, n)  # gamma, rad / unit of log k
        k = np.arange(1, T + 1, dtype=np.float64)
        # Amplitudes: ridge fit to 1/k in relative error (rows scaled by k).
        A = _oscillator_basis(self.kernel_param, k, rate, freq, np.zeros(n)) * k[:, None]
        amp = np.linalg.solve(A.T @ A + 1e-3 * np.eye(n), A.T @ np.ones(T))
        const = lambda v: (lambda shape, dtype=None: keras.ops.convert_to_tensor(np.asarray(v, "float32"), dtype=dtype))
        self.osc_rate = self.add_weight(name="osc_rate", shape=(n,), initializer=const(np.log(np.expm1(rate))))  # softplus^-1
        self.osc_freq = self.add_weight(name="osc_freq", shape=(n,), initializer=const(freq))
        self.osc_amp = self.add_weight(name="osc_amp", shape=(n,), initializer=const(amp))
        self.osc_phase = self.add_weight(name="osc_phase", shape=(n,), initializer="zeros")

    def distance_kernel(self) -> Any:
        """The kernel aligned with the context axis (last entry = distance 1):
        `(T,)`, or `(T, dim)` with `per_dim_kernels`."""
        if self.kernel_param == "free":
            return self.distance_weights
        ops = keras.ops
        k = ops.convert_to_tensor(self._distance)[:, None]  # (T, 1), context-aligned distances
        rate = ops.softplus(self.osc_rate)[None]
        freq, phase = self.osc_freq[None], self.osc_phase[None]
        if self.kernel_param == "exp-oscillators":
            terms = ops.exp(-rate * k) * ops.cos(freq * k + phase)
        else:
            logk = ops.log(k)
            terms = ops.exp(-rate * logk) * ops.cos(freq * logk + phase)
        return ops.sum(terms * self.osc_amp[None], axis=1)  # (T,)

    # -- forward --------------------------------------------------------------

    def hidden(self, inputs: Any) -> Any:
        """Pre-readout vector `(B, dim)` for single-target windows."""
        token_ids = keras.ops.cast(inputs, "int32")
        x = self.token_embedding(token_ids)  # (B, T, dim)

        half = self.dim // 2
        cos = keras.ops.convert_to_tensor(self._cos, dtype=x.dtype)
        sin = keras.ops.convert_to_tensor(self._sin, dtype=x.dtype)
        x1, x2 = x[..., :half], x[..., half:]
        rotated = keras.ops.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

        weights = keras.ops.cast(self.distance_kernel(), x.dtype)
        weights = weights[None] if self.per_dim_kernels else weights[None, :, None]  # (1, T, dim | 1)
        return keras.ops.sum(rotated * weights, axis=1)  # (B, dim) == weighted sum of per-token logits

    def readout_input(self, c: Any) -> Any:
        """Residual key-value memory MLP on the summed context (identity when off).

        Applied on the readout side so dense training runs it on valid rows only."""
        if self.memory_hidden is None:
            return c
        return c + self.memory_out(self.memory_in(self.memory_norm(c)))

    def call(self, inputs: Any, training: bool = False) -> Any:
        return self.readout(self.readout_input(self.hidden(inputs)))

    def dense_hidden(self, inputs: Any) -> Any:
        """`(B, T, dim)` pre-readout vectors for all `T` queries of a `(B, 2T-1)` input
        (see `model/archs/_dense.py`); query `m` equals `hidden` on `inputs[:, m:m+T]`."""
        ops = keras.ops
        x = self.token_embedding(ops.cast(inputs, "int32"))
        rope = DenseRope.get(self, x)
        u = rope.unrotate_keys(x)
        w = ops.cast(self.distance_kernel(), x.dtype)
        if not self.per_dim_kernels:
            return rope.to_query_frame(dense_kernel_sum(w, u))
        # Pair (p, q = p + dim/2), query angle A = (i+1)θ, unrotated halves u1, u2:
        #   h_p = cos A (w_p * u1) - sin A (w_p * u2)
        #   h_q = sin A (w_q * u1) + cos A (w_q * u2)
        # where `*` is the per-channel banded sum: one batched matmul of the
        # (2, half, T, S) bands [w_p; w_q] against the pair's two halves
        # (half, S, 2B) = [u1 | u2].
        T, half = self.window_size, self.dim // 2
        B = ops.shape(u)[0]
        wt = ops.reshape(ops.transpose(w), (2, half, 1, T))  # [w_p; w_q] per pair
        band = skew_band(ops.broadcast_to(wt, (2, half, T, T)))  # (2, half, T, S)
        uh = ops.transpose(ops.reshape(u, (B, -1, 2, half)), (3, 1, 2, 0))  # (half, S, 2, B)
        out = ops.matmul(band, ops.reshape(uh, (half, -1, 2 * B))[None])  # (2, half, T, 2B)
        out = ops.transpose(ops.reshape(out, (2, half, T, 2, B)), (4, 2, 0, 3, 1))  # (B, T, kernel, half-of-u, pair)
        cos, sin = rope.query_cos, rope.query_sin  # (T, half)
        h_p = cos * out[:, :, 0, 0] - sin * out[:, :, 0, 1]
        h_q = sin * out[:, :, 1, 0] + cos * out[:, :, 1, 1]
        return ops.concatenate([h_p, h_q], axis=-1)

    def with_context_length(self, context_length: int) -> "RopeLogitSumLM":
        # RoPE works at any distance; `distance_weights` is zero-padded beyond the trained window.
        return _context.with_context_length(self, context_length)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "vocabulary_size": self._vocabulary_size,
            "dim": self.dim,
            "window_size": self.window_size,
            "rope_base": self.rope_base,
            "per_dim_kernels": self.per_dim_kernels,
            "memory_hidden": self.memory_hidden,
            "kernel_param": self.kernel_param,
            "num_oscillators": self.num_oscillators,
        }
        return {**super().get_config(), **config}
