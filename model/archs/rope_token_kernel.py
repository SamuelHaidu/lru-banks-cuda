"""RoPE token kernel: a shared oscillator kernel bent per token.

    weight(j)  = s_t * delta_t(n_j) * w(k^{lambda_t})            t = token at context position j
    w(x)       = sum_{m=1..M} a_m e^{-alpha_m x} cos(omega_m x + phi_m)       shared, M oscillators
    h          = sum_j weight(j) R_k e_t / sum_j |weight(j)|                 (normalized)
    logits     = W_out^T h

- `s_t` (prior, softplus, init 1): how much the token counts when it is new.
- `lambda_t` (forgetting, sigmoid, init 0.95): the token sees the shared kernel at the
  warped distance `k^lambda`. `lambda = 1` forgets like the shared kernel, `lambda = 0`
  keeps `w(1)` forever (constant presence), in between it forgets more slowly. For a
  power-law kernel this is exactly `k^{-lambda p}`; `k^lambda <= k`, so with a
  decreasing `w` a token is never stronger in the past than when it appeared.
- `delta_t(n)` (saturation): the n-th mention of a token *within the window* gets
  `g(n) - g(n-1)` with `g(n) = n (1 + kappa_t) / (n + kappa_t)` (BM25, `kappa` softplus,
  init 20 = nearly additive), so all mentions of a token sum to at most `1 + kappa_t`
  times one mention.
- normalization: divide by the total |weight| (|.| because the oscillator kernel can be
  negative), so `h` is a weighted average of rotated tokens, times a learned gain
  (`output_gain` is its initial value): an average of small random embeddings is tiny,
  and without the gain the logits start near 0 and training crawls.

Every weight depends only on the context token and its distance, so dense supervision is
one banded matmul per sample, like the other RoPE archs. Studies:
`studies/token_coefficient_kernel.html`, `studies/token_kernel_empirical.py`.
"""

from typing import Any, Dict, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402

from model.archs import _context  # noqa: E402
from model.archs._dense import DenseRope, DenseSupervisionMixin  # noqa: E402
from model.archs.rope_logit_sum import _oscillator_basis  # noqa: E402
from model.interfaces import BaseLM  # noqa: E402


def _const(v: Any):
    return lambda shape, dtype=None: keras.ops.convert_to_tensor(np.asarray(v, "float32"), dtype=dtype)


@keras.saving.register_keras_serializable(package="model")
class RopeTokenKernelLM(DenseSupervisionMixin, BaseLM):
    """Kernel and per-token numbers are excluded from weight decay (`no_weight_decay_names`):
    decay on their raw values would pull every token's lambda to 0.5, prior to 0.69, etc.

    Args:
        vocabulary_size, dim, window_size, rope_base: as `rope-logit-sum`.
        num_oscillators: Terms of the shared exponential-oscillator kernel.
        prior, forgetting, saturation, normalize: switch each mechanism on or off.
    """

    no_weight_decay_names = ("token_prior", "token_lambda", "token_kappa",
                             "osc_rate", "osc_freq", "osc_amp", "osc_phase", "log_gain")

    def __init__(
        self,
        vocabulary_size: int = 10000,
        dim: int = 128,
        window_size: int = 128,
        rope_base: float = 10000.0,
        num_oscillators: int = 8,
        prior: bool = True,
        forgetting: bool = True,
        saturation: bool = True,
        normalize: bool = True,
        output_gain: float = 30.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"`dim` must be positive and even, got {dim}")
        if window_size <= 1:
            raise ValueError(f"`window_size` must be > 1, got {window_size}")
        if num_oscillators <= 0:
            raise ValueError(f"`num_oscillators` must be positive, got {num_oscillators}")
        self._vocabulary_size = int(vocabulary_size)
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.rope_base = float(rope_base)
        self.num_oscillators = int(num_oscillators)
        self.prior, self.forgetting = bool(prior), bool(forgetting)
        self.saturation, self.normalize = bool(saturation), bool(normalize)
        self.output_gain = float(output_gain)

        half = self.dim // 2
        inv_freq = 1.0 / (self.rope_base ** (np.arange(half, dtype=np.float64) / half))
        distance = np.arange(self.window_size, 0, -1, dtype=np.float64)  # context-aligned: last = 1
        angles = distance[:, None] * inv_freq[None, :]
        self._cos = np.cos(angles).astype("float32")
        self._sin = np.sin(angles).astype("float32")
        self._log_distance = np.log(distance).astype("float32")

        self.token_embedding = layers.Embedding(self._vocabulary_size, self.dim, name="token_emb")
        self.readout = layers.Dense(self._vocabulary_size, use_bias=False, name="readout")
        self.build()

    @property
    def context_length(self) -> int:
        return self.window_size

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    def build(self, input_shape: Optional[Any] = None) -> None:
        if self.built:
            return
        self.token_embedding.build((None, self.window_size))
        self.readout.build((None, self.dim))
        # Shared kernel: amplitudes start as a ridge fit to 1/k (as rope-logit-sum's exp-oscillators).
        n, T = self.num_oscillators, self.window_size
        tau = np.geomspace(1.0, T, n)
        rate, freq = 1.0 / tau, 0.5 / tau
        k = np.arange(1, T + 1, dtype=np.float64)
        A = _oscillator_basis("exp-oscillators", k, rate, freq, np.zeros(n)) * k[:, None]
        amp = np.linalg.solve(A.T @ A + 1e-3 * np.eye(n), A.T @ np.ones(T))
        self.osc_rate = self.add_weight(name="osc_rate", shape=(n,), initializer=_const(np.log(np.expm1(rate))))
        self.osc_freq = self.add_weight(name="osc_freq", shape=(n,), initializer=_const(freq))
        self.osc_amp = self.add_weight(name="osc_amp", shape=(n,), initializer=_const(amp))
        self.osc_phase = self.add_weight(name="osc_phase", shape=(n,), initializer="zeros")
        V = self._vocabulary_size
        if self.prior:
            self.token_prior = self.add_weight(name="token_prior", shape=(V,), initializer=_const(np.full(V, np.log(np.expm1(1.0)))))
        if self.forgetting:
            self.token_lambda = self.add_weight(name="token_lambda", shape=(V,), initializer=_const(np.full(V, np.log(0.95 / 0.05))))
        if self.normalize:
            self.log_gain = self.add_weight(name="log_gain", shape=(), initializer=_const(np.log(self.output_gain)))
        if self.saturation:
            self.token_kappa = self.add_weight(name="token_kappa", shape=(V,), initializer=_const(np.full(V, np.log(np.expm1(20.0)))))
        self.built = True

    # -- pieces ---------------------------------------------------------------

    def shared_kernel(self, x: Any) -> Any:
        """`w(x)` at (warped) distances `x >= 1`, any shape."""
        ops = keras.ops
        rate = ops.softplus(self.osc_rate)
        out = 0.0
        for m in range(self.num_oscillators):
            out = out + self.osc_amp[m] * ops.exp(-rate[m] * x) * ops.cos(self.osc_freq[m] * x + self.osc_phase[m])
        return out

    def token_numbers(self, ids: Any) -> Dict[str, Any]:
        """Per-token `s`, `lambda`, `kappa` gathered for `ids` (None when the mechanism is off)."""
        ops = keras.ops
        take = lambda w: ops.take(w, ids, axis=0)
        return {
            "s": ops.softplus(take(self.token_prior)) if self.prior else None,
            "lambda": ops.sigmoid(take(self.token_lambda)) if self.forgetting else None,
            "kappa": ops.softplus(take(self.token_kappa)) if self.saturation else None,
        }

    @staticmethod
    def _delta(n: Any, kappa: Any) -> Any:
        g = lambda v: v * (1.0 + kappa) / (v + kappa)
        return g(n) - g(n - 1.0)

    def _weights(self, ids: Any, log_k: Any, mention: Any) -> Any:
        """Weights from token ids (broadcast to `log_k`'s shape), log distances and mention index."""
        ops = keras.ops
        num = self.token_numbers(ids)
        lam = num["lambda"]
        warped = ops.exp(log_k * lam) if lam is not None else ops.exp(log_k)  # k^lambda
        w = self.shared_kernel(warped)
        if num["s"] is not None:
            w = w * num["s"]
        if num["kappa"] is not None:
            w = w * self._delta(mention, num["kappa"])
        return w

    def _finish(self, w: Any, summed: Any) -> Any:
        if not self.normalize:
            return summed
        ops = keras.ops
        return summed * ops.exp(self.log_gain) / (ops.sum(ops.abs(w), axis=-1, keepdims=True) + 1e-6)

    # -- forward --------------------------------------------------------------

    def hidden(self, inputs: Any) -> Any:
        """Pre-readout vector `(B, dim)` for single-target windows `(B, T)`."""
        ops = keras.ops
        ids = ops.cast(inputs, "int32")
        x = self.token_embedding(ids)
        half = self.dim // 2
        cos = ops.convert_to_tensor(self._cos, dtype=x.dtype)
        sin = ops.convert_to_tensor(self._sin, dtype=x.dtype)
        x1, x2 = x[..., :half], x[..., half:]
        rotated = ops.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        log_k = ops.convert_to_tensor(self._log_distance, dtype=x.dtype)[None]  # (1, T)
        mention = None
        if self.saturation:  # n_j = mentions of t_j in the window up to and including j
            T = self.window_size
            same = ops.cast(ops.equal(ids[:, :, None], ids[:, None, :]), x.dtype)  # (B, j, j')
            mention = ops.sum(same * ops.convert_to_tensor(np.tril(np.ones((T, T), "float32")), dtype=x.dtype)[None], axis=-1)
        w = self._weights(ids, log_k, mention)  # (B, T)
        return self._finish(w, ops.einsum("bt,btd->bd", w, rotated))

    def call(self, inputs: Any, training: bool = False) -> Any:
        return self.readout(self.hidden(inputs))

    def dense_hidden(self, inputs: Any) -> Any:
        """`(B, T, dim)` for all `T` queries of a `(B, 2T-1)` input; query `m` sees inputs
        `m .. m+T-1` and equals `hidden(inputs[:, m:m+T])`."""
        ops = keras.ops
        T = self.window_size
        S = 2 * T - 1
        ids = ops.cast(inputs, "int32")
        x = self.token_embedding(ids)
        rope = DenseRope.get(self, x)
        u = rope.unrotate_keys(x)  # (B, S, dim)
        consts = self.__dict__.setdefault("_token_dense_cache", {})
        key = (str(getattr(x, "device", "")), str(x.dtype))
        if key not in consts:
            m = np.arange(T)[:, None]
            s = np.arange(S)[None, :]
            k = (m + T - s).astype("float32")  # distance of input s from query m
            band = (s >= m) & (s < m + T)
            consts[key] = (
                ops.convert_to_tensor(np.where(band, np.log(np.maximum(k, 1.0)), 0.0).astype("float32"), dtype=x.dtype),
                ops.convert_to_tensor(band.astype("float32"), dtype=x.dtype),
            )
        log_k, band = consts[key]
        mention = None
        if self.saturation:
            # Mentions of t_j inside query m's window [m, j]: (count up to j) - (count before m).
            same = ops.cast(ops.equal(ids[:, :, None], ids[:, None, :]), x.dtype)  # (B, j, j')
            upto = ops.cumsum(same, axis=-1)  # [b, j, c] = mentions of t_j among positions <= c
            before = ops.transpose(upto, (0, 2, 1))[:, : T - 1]  # [b, m-1, j] for m = 1..T-1
            before = ops.concatenate([ops.zeros_like(before[:, :1]), before], axis=1)  # (B, T, S)
            diag = ops.diagonal(upto, axis1=1, axis2=2)  # (B, S): mentions of t_j up to j itself
            # Outside the band the count can be <= 0, where g(n - 1) divides by ~0 and the
            # masked inf * 0 becomes NaN: clamp (in-band counts are always >= 1).
            mention = ops.maximum(diag[:, None, :] - before, 1.0)
        w = self._weights(ids[:, None, :], log_k[None], mention) * band[None]  # (B, T, S)
        return rope.to_query_frame(self._finish(w, ops.matmul(w, u)))

    def with_context_length(self, context_length: int) -> "RopeTokenKernelLM":
        return _context.with_context_length(self, context_length)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "vocabulary_size": self._vocabulary_size, "dim": self.dim,
                "window_size": self.window_size, "rope_base": self.rope_base,
                "num_oscillators": self.num_oscillators, "prior": self.prior, "forgetting": self.forgetting,
                "saturation": self.saturation, "normalize": self.normalize,
                "output_gain": self.output_gain}
