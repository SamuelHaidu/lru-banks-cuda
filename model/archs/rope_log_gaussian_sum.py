"""RoPE log-Gaussian sum: input-adaptive distance weights, two passes.

    v_k   = R_k E[t_{n-k}]                         rotated context, computed once
    h0    = sum_k a_k v_k                          pass 1: static learned a_k (bag of context)
    alpha = softplus(MLP(h0))                      (M,) amplitudes, 3-layer SwiGLU MLP
    w_k   = sum_j alpha_j exp(-(log k - mu_j)^2 / (2 sigma^2))
    h     = sum_k w_k v_k                          pass 2: per-input weights
    logits = W_out^T h0 + W_out^T h = W_out^T (h0 + h)

The final logits are the sum of both passes' logits: the static kernel `a_k`
plus the adaptive kernel `w_k`. Because the readout is linear, both sums and
their addition are done in `dim` space and projected once.

The MLP reads `h0` rather than the first-pass logits `W_out^T h0`: those are a
linear map of `h0`, so they carry the same information at 10k-wide cost.

`kernel_basis` picks the fixed basis the MLP's amplitudes are applied to:

- `"log-gaussian"` (default): `M` Gaussians in log-distance with fixed, evenly
  spaced centers `mu_j` over `[0, log T]` and width `sigma` equal to the
  spacing. Amplitudes go through softplus, so `w_k >= 0`: a power-law-like
  decay, a bump at some distance, or both.
- `"damped-oscillator"`: SSM-style damped springs. For `M` timescales `tau_j`
  log-spaced over `[1, T]`, two basis functions
  `exp(-(k-1)/tau_j) * cos(omega_j (k-1))` and the same with `sin`, with
  `omega_j = pi / (2 tau_j)` (a quarter swing per decay time). That is `2M`
  functions with signed amplitudes (no softplus), so `w_k` can go negative,
  like an S4/LRU kernel, except that the amplitudes are chosen per input.

The basis is built for `basis_window` (default: `window_size`). Keeping it
fixed while changing `window_size` evaluates the same functions at more
distances, which is what context extrapolation and longer-window
fine-tuning need: the learned amplitudes keep their meaning.

Only amplitudes are predicted, so the MLP's cost does not depend on `T`.
Amplitudes start (zero-init last layer) at a least-squares fit of `1/k`, so
the second pass starts out close to the static first pass.
"""

from typing import Any, Dict, Optional, Tuple

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402

from model.archs import _context  # noqa: E402
from model.archs._dense import DenseRope, DenseSupervisionMixin, dense_kernel_sum  # noqa: E402
from model.interfaces import BaseLM  # noqa: E402


KERNEL_BASES = ("log-gaussian", "damped-oscillator", "fourier")


def _build_basis(kind: str, distance: np.ndarray, m: int, basis_window: int) -> Tuple[np.ndarray, list]:
    """Returns `(basis (len(distance), n_functions) float32, labels)`.

    Centers / timescales span `[1, basis_window]`; `distance` may extend
    beyond it (extrapolation), in which case the same functions are evaluated
    further out.
    """
    T = float(basis_window)
    if kind == "log-gaussian":
        spacing = np.log(T) / (m - 1)
        centers = np.arange(m) * spacing
        basis = np.exp(-((np.log(distance)[:, None] - centers[None, :]) ** 2) / (2 * spacing**2))
        return basis.astype("float32"), [f"k~{np.exp(c):.0f}" for c in centers]
    if kind == "fourier":
        # cos/sin of frequencies pi*j/T on the linear lag axis (j = 0..m-1; the
        # always-zero sin at j = 0 is dropped). A bump at lag k0 is
        # sum_j cos(w_j k0) cos(w_j k) + sin(w_j k0) sin(w_j k): its amplitudes are
        # the same cos/sin phases RoPE writes into attended vectors.
        omegas = np.pi * np.arange(m) / T
        lag = distance[:, None] - 1.0
        basis = np.concatenate([np.cos(omegas * lag), np.sin(omegas[1:] * lag)], axis=1)
        labels = [f"cos {j}" for j in range(m)] + [f"sin {j}" for j in range(1, m)]
        return basis.astype("float32"), labels
    taus = np.exp(np.arange(m) * np.log(T) / (m - 1))  # 1 .. T
    omegas = np.pi / (2 * taus)
    lag = distance[:, None] - 1.0  # k = 1 is lag 0
    envelope = np.exp(-lag / taus[None, :])
    basis = np.concatenate([envelope * np.cos(omegas * lag), envelope * np.sin(omegas * lag)], axis=1)
    labels = [f"tau~{t:.0f} cos" for t in taus] + [f"tau~{t:.0f} sin" for t in taus]
    return basis.astype("float32"), labels


def _initial_amplitudes(kind: str, basis: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Bias init for the amplitude layer: least-squares fit of `basis @ alpha` to
    `1/k` in relative error (rows scaled by k), in the layer's pre-activation space."""
    scaled = basis * distance[:, None]
    if kind == "log-gaussian":
        target = np.linalg.lstsq(scaled, np.ones_like(distance), rcond=None)[0]
        target = np.maximum(target, 1e-4)  # floored to stay positive under softplus
        return np.log(np.expm1(target)).astype("float32")  # inverse softplus
    # Signed bases: ridge-regularized (oscillator cos/sin pairs are nearly collinear
    # at long timescales; Fourier is well conditioned either way).
    gram = scaled.T @ scaled + 1e-3 * np.eye(scaled.shape[1])
    return np.linalg.solve(gram, scaled.T @ np.ones_like(distance)).astype("float32")


@keras.saving.register_keras_serializable(package="model")
class RopeLogGaussianSumLM(DenseSupervisionMixin, BaseLM):
    """Shared tables, RoPE distance rotation, MLP-predicted log-Gaussian distance kernel.

    Args:
        vocabulary_size: Token vocabulary size (width of the logits).
        dim: Embedding dimension (must be even for RoPE).
        window_size: Context length.
        rope_base: RoPE frequency base.
        mlp_hidden: Hidden width of the SwiGLU MLP.
        num_centers: Number of basis timescales `M` (log-Gaussian centers, or
            oscillator timescales -- each gives a cos and a sin function).
        kernel_basis: `"log-gaussian"`, `"damped-oscillator"` or `"fourier"`
            (`num_centers` frequencies on the linear distance axis).
        basis_window: Window the basis centers/timescales are built for
            (default `window_size`). Set it to the training window when
            running or fine-tuning at a different `window_size`.
    """

    _defer_build = False

    def __init__(
        self,
        vocabulary_size: int = 10000,
        dim: int = 128,
        window_size: int = 128,
        rope_base: float = 10000.0,
        mlp_hidden: int = 128,
        num_centers: int = 8,
        kernel_basis: str = "log-gaussian",
        basis_window: Optional[int] = None,
        readout_hidden: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if vocabulary_size <= 0:
            raise ValueError(f"`vocabulary_size` must be positive, got {vocabulary_size}")
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"`dim` must be positive and even, got {dim}")
        if window_size <= 1:
            raise ValueError(f"`window_size` must be > 1, got {window_size}")
        if mlp_hidden <= 0:
            raise ValueError(f"`mlp_hidden` must be positive, got {mlp_hidden}")
        if num_centers < 2:
            raise ValueError(f"`num_centers` must be >= 2, got {num_centers}")
        if basis_window is not None and int(basis_window) <= 1:
            raise ValueError(f"`basis_window` must be > 1, got {basis_window}")
        if readout_hidden is not None and int(readout_hidden) <= 0:
            raise ValueError(f"`readout_hidden` must be positive, got {readout_hidden}")
        if kernel_basis not in KERNEL_BASES:
            raise ValueError(f"`kernel_basis` must be one of {KERNEL_BASES}, got {kernel_basis!r}")

        self._vocabulary_size = int(vocabulary_size)
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.rope_base = float(rope_base)
        self.mlp_hidden = int(mlp_hidden)
        self.num_centers = int(num_centers)
        self.kernel_basis = str(kernel_basis)
        self.basis_window = int(basis_window) if basis_window is not None else self.window_size
        self.readout_hidden = int(readout_hidden) if readout_hidden is not None else None
        # Width of the accumulated context state the MLPs and readout see;
        # subclasses may widen it past the token width `dim`.
        self.state_dim = self.dim

        # Distance from the predicted position: context index j -> k = T - j,
        # so the last context token (n-1) gets k = 1.
        half = self.dim // 2
        inv_freq = 1.0 / (self.rope_base ** (np.arange(half, dtype=np.float64) / half))
        distance = np.arange(self.window_size, 0, -1, dtype=np.float64)
        angles = distance[:, None] * inv_freq[None, :]  # (T, dim/2)
        self._cos = np.cos(angles).astype("float32")
        self._sin = np.sin(angles).astype("float32")
        self._distance = distance.astype("float32")

        # Fixed basis over distance, (T, num_amplitudes), and its initial amplitudes:
        # a least-squares fit of sum_j alpha_j B_j(k) to 1/k in relative error
        # (rows scaled by k).
        self._basis, self._basis_labels = _build_basis(self.kernel_basis, distance, self.num_centers, self.basis_window)
        self._alpha_bias_init = _initial_amplitudes(self.kernel_basis, self._basis, distance)

        self.token_embedding = layers.Embedding(self._vocabulary_size, self.dim, name="token_emb")
        self.readout = layers.Dense(self._vocabulary_size, use_bias=False, name="readout")
        # Optional non-linear projection of the context vector before the readout:
        # logits = W_out^T gelu(W_p h + b), with `readout_hidden` << `dim` possible.
        self.readout_proj = (
            layers.Dense(self.readout_hidden, activation="gelu", name="readout_proj")
            if self.readout_hidden is not None else None
        )
        # 3-layer MLP: two SwiGLU layers, then a linear layer to the basis amplitudes.
        self.mlp_gate_1 = layers.Dense(self.mlp_hidden, use_bias=False, name="mlp_gate_1")
        self.mlp_up_1 = layers.Dense(self.mlp_hidden, use_bias=False, name="mlp_up_1")
        self.mlp_gate_2 = layers.Dense(self.mlp_hidden, use_bias=False, name="mlp_gate_2")
        self.mlp_up_2 = layers.Dense(self.mlp_hidden, use_bias=False, name="mlp_up_2")
        bias_init = self._alpha_bias_init
        self.mlp_out = layers.Dense(
            self._basis.shape[1],
            kernel_initializer="zeros",
            bias_initializer=lambda shape, dtype=None: keras.ops.convert_to_tensor(bias_init, dtype=dtype),
            name="mlp_out",
        )

        # Subclasses that add layers set `_defer_build = True` and call `build()`
        # themselves: keras forbids creating sub-layers once the model is built.
        if not self._defer_build:
            self.build()

    # -- BaseLM contract ------------------------------------------------------

    @property
    def context_length(self) -> int:
        return self.window_size

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    def basis_labels(self) -> list:
        """One label per predicted amplitude, for analysis scripts."""
        return list(self._basis_labels)

    # -- weights --------------------------------------------------------------

    def build(self, input_shape: Optional[Any] = None) -> None:
        if self.built:
            return
        self.token_embedding.build((None, self.window_size))
        if self.readout_proj is not None:
            self.readout_proj.build((None, self.state_dim))
        self.readout.build((None, self.readout_hidden or self.state_dim))
        self.mlp_gate_1.build((None, self.state_dim))
        self.mlp_up_1.build((None, self.state_dim))
        self.mlp_gate_2.build((None, self.mlp_hidden))
        self.mlp_up_2.build((None, self.mlp_hidden))
        self.mlp_out.build((None, self.mlp_hidden))
        # Pass-1 static weight per distance, aligned with the context axis.
        init = (1.0 / self._distance).astype("float32")
        self.distance_weights = self.add_weight(
            name="distance_weights",
            shape=(self.window_size,),
            initializer=lambda shape, dtype=None: keras.ops.convert_to_tensor(init, dtype=dtype),
        )
        self.built = True

    # -- forward --------------------------------------------------------------

    def _constants(self, like: Any) -> Tuple[Any, Any, Any]:
        """`(cos, sin, basis)` as device tensors, converted once per (device, dtype).

        Converting the numpy arrays on every call meant host-to-device copies each
        step. The cache lives in `__dict__` directly so keras does not track it as
        model state.
        """
        cache = self.__dict__.setdefault("_constant_cache", {})
        key = (str(getattr(like, "device", "")), str(like.dtype))
        if key not in cache:
            cache[key] = tuple(
                keras.ops.convert_to_tensor(a, dtype=like.dtype) for a in (self._cos, self._sin, self._basis)
            )
        return cache[key]

    def kernels(self, inputs: Any) -> Tuple[Any, Any, Any, Any]:
        """Returns `(rotated, h0, alpha, adaptive_w)`; shared by `call` and analysis scripts.

        `adaptive_w` is `(B, T)` aligned with the context axis (last index = n-1).
        """
        ops = keras.ops
        rotated, h0 = self._static_pass(inputs)
        basis = self._constants(rotated)[2]
        # MLP -> basis amplitudes -> per-input distance weights.
        alpha = self._amplitudes(self._kernel_input(h0))  # (B, num_amplitudes)
        adaptive_w = ops.einsum("bm,tm->bt", alpha, basis)  # (B, T); basis is (T, M)
        return rotated, h0, alpha, adaptive_w

    def _kernel_input(self, h0: Any) -> Any:
        """What the kernel MLP reads from the pass-1 summary (identity here)."""
        return h0

    def _static_pass(self, inputs: Any) -> Tuple[Any, Any]:
        """`(rotated (B, T, dim), h0 (B, dim))`: the rotated window and its static sum."""
        ops = keras.ops
        token_ids = ops.cast(inputs, "int32")
        x = self.token_embedding(token_ids)  # (B, T, dim)

        half = self.dim // 2
        cos, sin, _ = self._constants(x)
        x1, x2 = x[..., :half], x[..., half:]
        rotated = ops.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)  # reused by both passes

        # Pass 1: static weighted sum -> context summary.
        static_w = ops.cast(self.distance_weights, x.dtype)
        h0 = ops.einsum("t,btd->bd", static_w, rotated)
        return rotated, h0

    def _amplitudes(self, h0: Any) -> Any:
        """Kernel MLP: context summary `(..., dim)` -> basis amplitudes `(..., M)`."""
        ops = keras.ops
        z = ops.silu(self.mlp_gate_1(h0)) * self.mlp_up_1(h0)
        z = ops.silu(self.mlp_gate_2(z)) * self.mlp_up_2(z)
        alpha = self.mlp_out(z)
        return ops.softplus(alpha) if self.kernel_basis == "log-gaussian" else alpha

    def hidden(self, inputs: Any) -> Any:
        """Pre-readout vector `(B, dim)` for single-target windows."""
        rotated, h0, _, adaptive_w = self.kernels(inputs)
        # Pass 2: adaptive weighted sum; logits of both passes summed (in dim space).
        return h0 + keras.ops.einsum("bt,btd->bd", adaptive_w, rotated)

    def readout_input(self, h: Any) -> Any:
        """Context vector `(..., dim)` -> what the linear readout sees."""
        return h if self.readout_proj is None else self.readout_proj(h)

    def call(self, inputs: Any, training: bool = False) -> Any:
        return self.readout(self.readout_input(self.hidden(inputs)))

    def dense_kernels(self, inputs: Any) -> Tuple[Any, Any, Any]:
        """Dense counterpart of `kernels` for a `(B, 2T-1)` input.

        Returns `(u, h1, rope)`: unrotated keys `(B, 2T-1, dim)`, the refined
        context `(B, T, dim)` for every query (equal to `hidden` on each
        window), and the dense RoPE constants.
        """
        ops = keras.ops
        u, s0, rope = self._dense_static_pass(inputs)
        h0 = rope.to_query_frame(s0)  # (B, T, dim), the per-query pass-1 summary
        basis = self._constants(u)[2]  # (T, M)
        adaptive_w = ops.einsum("bmk,tk->bmt", self._amplitudes(self._kernel_input(h0)), basis)  # (B, T, T), context-aligned
        s1 = dense_kernel_sum(adaptive_w, u)
        return u, rope.to_query_frame(s0 + s1), rope

    def _dense_static_pass(self, inputs: Any) -> Tuple[Any, Any, Any]:
        """`(u, s0, rope)`: unrotated keys `(B, 2T-1, dim)`, the static sum per query
        `(B, T, dim)` in the unrotated frame, and the dense RoPE constants."""
        ops = keras.ops
        x = self.token_embedding(ops.cast(inputs, "int32"))
        rope = DenseRope.get(self, x)
        u = rope.unrotate_keys(x)
        return u, dense_kernel_sum(ops.cast(self.distance_weights, x.dtype), u), rope

    def dense_hidden(self, inputs: Any) -> Any:
        """`(B, T, dim)` pre-readout vectors for all `T` queries of a `(B, 2T-1)` input."""
        return self.dense_kernels(inputs)[1]

    def with_context_length(self, context_length: int) -> "RopeLogGaussianSumLM":
        # The basis stays built for `basis_window` (carried in the config), so it
        # is the same functions evaluated at more (or fewer) distances.
        return _context.with_context_length(self, context_length)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "vocabulary_size": self._vocabulary_size,
            "dim": self.dim,
            "window_size": self.window_size,
            "rope_base": self.rope_base,
            "mlp_hidden": self.mlp_hidden,
            "num_centers": self.num_centers,
            "kernel_basis": self.kernel_basis,
            "basis_window": self.basis_window,
            "readout_hidden": self.readout_hidden,
        }
        return {**super().get_config(), **config}
