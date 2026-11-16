"""RoPE token banks: one general state, several selective banks, a query to mix them.

    tokens -> RoPE; every state has its own oscillator kernel w_b (M exponential oscillators, default 5)

    general   g   = gain_0 * sum_j w_0(k) R_k e_j / sum_j |w_0(k)|                   plain curve, no token numbers
    bank b    m_b = gain_b * sum_j s_b,t w_b(k^lambda_b,t) R_k e_j / sum_j |.|      own kernel, prior s_b, forgetting lambda_b
    router    pi  = softmax( W_r g )              (W_r: dim x banks, no bias; zero-init: uniform mix at start)
    final     h   = sum_b pi_b m_b,     logits = W_out^T h

The general state only decides the mix; the final vector is a weighted sum of the banks.

Options:
- `general_pair_kernels`: the general state gets one kernel per RoPE pair (dim / 2
  kernels, each `num_oscillators` terms) instead of one. It has no per-token numbers,
  so this is a per-channel convolution; a pair's kernel commutes with RoPE, so the dense
  path is one batched matmul with a band shared by every sample.
- `general_in_output`: the final vector is `h = g + sum_b pi_b m_b`, so the general state
  is also the base prediction, not only the router's input.
- `memory_heads`: associative memories (bind / unbind). Two more 128 x banks softmax heads
  on the general state give a binder mix and an unbinder mix of the banks, shared by all
  memories. At every position, memory n's binder (own MLP on the binder mix) outputs
  `memory_oscillators` oscillator parameters defining which recent tokens to store (a kernel
  over the last `memory_value_range` tokens), a free write key (unit phasors) and a write gate;
  the value is the kernel-weighted, RoPE-ordered average of projected tokens, bound to the key
  and added to the memory. The unbinder (own MLP on the unbinder mix) outputs a read key;
  `conj(key) * memory` is the retrieved value. Memory n holds the writes of the last
  `memory_window` positions. The final block reads `[h ; r_1 .. r_n]`. A prediction then depends
  on `window_size + memory_window - 1` tokens, which becomes the model's context length.
- `memory_hidden`: a residual feed-forward block before the readout,
  `h' = h + W_2 gelu(W_1 RMSNorm(h) + b_1) + b_2` (dim -> memory_hidden -> dim, `W_2`, `b_2`
  zero-initialized, so training starts as the model without it). It lets the readout combine
  features non-additively and store knowledge in its weights. Applied on the readout side,
  so dense training runs it on valid rows only.

Each bank is normalized on its own, so one bank can spend all its energy on the last token
("tried to") while another keeps a far content word ("climb") or the names. The banks start
with different forgetting (`bank_lambdas`, default 0.95 / 0.7 / 0.4 / 0.1) so they do not stay
copies of each other. The router reads the plain-curve summary of the context and decides
which banks to use.

The kernel is evaluated per (token, distance) at warped distances `k^lambda`; a custom
autograd function recomputes the oscillators in backward instead of storing one
`(B, banks, T, S)` tensor per oscillator.
"""

from typing import Any, Dict, Optional, Sequence

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from keras import layers  # noqa: E402

from model.archs import _context  # noqa: E402
from model.archs._dense import DenseRope, DenseSupervisionMixin, dense_kernel_sum, skew_band  # noqa: E402
from model.archs._norm import RMSNorm  # noqa: E402
from model.archs.rope_logit_sum import _oscillator_basis  # noqa: E402
from model.interfaces import BaseLM  # noqa: E402


class _OscKernel(torch.autograd.Function):
    """Per-state kernels `w_p(x) = sum_m a_pm e^{-r_pm x} cos(f_pm x + p_pm)` elementwise.

    `x` has the state axis first, `(P, ...)`; parameters are `(P, M)`. Backward recomputes
    each term instead of storing `M` full-size intermediates.
    """

    @staticmethod
    def forward(ctx, x, rate, freq, amp, phase):  # type: ignore[override]
        view = lambda v, m: v[:, m].reshape((-1,) + (1,) * (x.dim() - 1))
        out = torch.zeros_like(x)
        for m in range(rate.shape[1]):
            out += view(amp, m) * torch.exp(-view(rate, m) * x) * torch.cos(view(freq, m) * x + view(phase, m))
        ctx.save_for_backward(x, rate, freq, amp, phase)
        return out

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        x, rate, freq, amp, phase = ctx.saved_tensors
        view = lambda v, m: v[:, m].reshape((-1,) + (1,) * (x.dim() - 1))
        per_state = lambda t: t.reshape(t.shape[0], -1).sum(-1)
        dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        grads = [torch.zeros_like(rate), torch.zeros_like(freq), torch.zeros_like(amp), torch.zeros_like(phase)]
        for m in range(rate.shape[1]):
            a, r, f = view(amp, m), view(rate, m), view(freq, m)
            e = torch.exp(-r * x)
            arg = f * x + view(phase, m)
            ge = g * e
            gec, ges = ge * torch.cos(arg), ge * torch.sin(arg)
            grads[2][:, m] = per_state(gec)  # d amp
            grads[0][:, m] = -amp[:, m] * per_state(gec * x)  # d rate
            grads[1][:, m] = -amp[:, m] * per_state(ges * x)  # d freq
            grads[3][:, m] = -amp[:, m] * per_state(ges)  # d phase
            if dx is not None:
                dx += a * (-r * gec - f * ges)
        return (dx, *grads)


def _band_matmul(compact: Any, keys: Any) -> Any:
    """`out[..., m, :] = sum_r compact[..., m, r] * keys[..., m + r, :]`.

    `compact` (..., Q, W), `keys` (..., Q + W - 1, F), leading axes broadcast. Instead of the
    full (Q, Q + W - 1) band (mostly zeros), queries go in blocks of W; block c only needs
    keys cW .. cW + 2W - 1, so it is a (W, 2W) matmul against two adjacent key blocks.
    """
    ops = keras.ops
    Q, W = int(compact.shape[-2]), int(compact.shape[-1])
    S = int(keys.shape[-2])
    nc = -(-Q // W)
    lead_c, lead_k = tuple(compact.shape[:-2]), tuple(keys.shape[:-2])
    c = ops.pad(compact, [[0, 0]] * len(lead_c) + [[0, nc * W - Q], [0, W + 1]])  # rows W + 1 -> 2W + 1
    c = ops.reshape(c, lead_c + (nc, W * (2 * W + 1)))[..., : W * 2 * W]
    c = ops.reshape(c, lead_c + (nc, W, 2 * W))  # block row i holds its weights at columns i .. i + W - 1
    k = ops.pad(keys, [[0, 0]] * len(lead_k) + [[0, (nc + 1) * W - S], [0, 0]])
    k = ops.reshape(k, lead_k + (nc + 1, W, int(keys.shape[-1])))
    win = ops.concatenate([k[..., :-1, :, :], k[..., 1:, :, :]], axis=-2)  # (..., nc, 2W, F)
    out = ops.matmul(c, win)  # (..., nc, W, F)
    lead = tuple(out.shape[:-3])
    return ops.reshape(out, lead + (nc * W, int(out.shape[-1])))[..., :Q, :]


def _crot(v: Any, phase: Any) -> Any:
    """RoPE as one complex multiply: `v` (..., D) with pairs *interleaved* (dims 2j, 2j+1 are the
    real / imaginary parts of pair j), `phase` (..., D/2) complex unit numbers. ~3x faster on the
    Iris Xe than the half-split rotation (4 half-size multiplies, 2 adds and a concatenate)."""
    shape = tuple(v.shape)
    vc = torch.view_as_complex(v.contiguous().reshape(shape[:-1] + (shape[-1] // 2, 2)))
    return torch.view_as_real(vc * phase).reshape(shape)


def _deinterleave(v: Any) -> Any:
    """Interleaved pairs (2j, 2j+1) -> the half-split layout (j, j + D/2) used everywhere else."""
    shape = tuple(v.shape)
    return v.reshape(shape[:-1] + (shape[-1] // 2, 2)).transpose(-1, -2).reshape(shape)


class _TiedReadout:
    """Readout that reuses the token embedding table: logits = h E^T. A plain object, not a Keras
    layer, so the table is tracked (and saved, and counted) once, by the embedding layer. Same
    interface as the Dense readout it replaces: `.kernel` `(D, V)` and `__call__`."""

    def __init__(self, embedding: Any):
        self._embedding = embedding

    @property
    def kernel(self) -> Any:
        return keras.ops.transpose(self._embedding.embeddings)

    def __call__(self, h: Any) -> Any:
        return keras.ops.matmul(h, self.kernel)


def _const(v: Any):
    return lambda shape, dtype=None: keras.ops.convert_to_tensor(np.asarray(v, "float32"), dtype=dtype)


@keras.saving.register_keras_serializable(package="model")
class RopeTokenBanksLM(DenseSupervisionMixin, BaseLM):
    """Args:
        vocabulary_size, dim, window_size, rope_base: as `rope-logit-sum`.
        num_oscillators: Terms of each state's own kernel.
        bank_lambdas: Initial forgetting of each bank (one entry per bank).
        output_gain: Initial value of every state's learned gain.
        general_pair_kernels: One kernel per RoPE pair in the general state.
        general_in_output: Add the general state to the final vector.
        general_concat: The residual block before the readout reads `[RMSNorm(bank mix) ;
            RMSNorm(general)]` (2 x dim, each normalized on its own) instead of their sum.
            The residual path is unchanged (mix, plus general with `general_in_output`).
            Needs `memory_hidden`.
        general_concat_norm: How `general_concat` normalizes the two halves: "separate" (one
            RMSNorm each; drops their per-step size ratio) or "joint" (one RMSNorm over the whole
            2 x dim vector; keeps how big the mix is relative to the general state).
        memory_hidden: Hidden width of the residual GELU block before the readout (None: off).
        memory_heads: Number of associative memories (0: off; needs memory_hidden).
        memory_window: Writes kept in memory, in positions (default window_size).
        memory_value_range: How many recent tokens a write can store.
        memory_oscillators: Oscillators of the binder's value kernel.
        memory_mlp_hidden: Hidden width of each binder / unbinder MLP.
        tie_embeddings: Use the token embedding table as the readout too (logits = h E^T), so a
            wider `dim` costs the vocabulary table once instead of twice.
        bank_projections: A learned D x D projection of the token embeddings per state (general
            first, then each bank), applied before RoPE and pooling, so every state pools its own
            view of the tokens. Starts as the identity (the same model as without it).
    """

    # Implementation switches of the dense path (same math; kept as flags to benchmark them).
    _complex_rope = True  # rotate keys / states as complex multiplies on interleaved pairs
    _stacked_lookup = True  # projected views read from one (states * V, D) table: no transpose copy

    no_weight_decay_names = ("_prior", "_lambda", "osc_", "log_gain", "bank_proj")  # bank_proj: decay would pull it from identity to 0

    def __init__(
        self,
        vocabulary_size: int = 10000,
        dim: int = 128,
        window_size: int = 128,
        rope_base: float = 10000.0,
        num_oscillators: int = 5,
        bank_lambdas: Sequence[float] = (0.95, 0.7, 0.4, 0.1),
        output_gain: float = 30.0,
        general_pair_kernels: bool = False,
        general_in_output: bool = False,
        general_concat: bool = False,
        general_concat_norm: str = "separate",
        memory_hidden: Optional[int] = None,
        memory_heads: int = 0,
        memory_window: Optional[int] = None,
        memory_value_range: int = 16,
        memory_oscillators: int = 8,
        memory_mlp_hidden: int = 256,
        bank_projections: bool = False,
        tie_embeddings: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"`dim` must be positive and even, got {dim}")
        if not bank_lambdas or not all(0.0 < float(v) < 1.0 for v in bank_lambdas):
            raise ValueError(f"`bank_lambdas` must be values in (0, 1), got {bank_lambdas}")
        self._vocabulary_size = int(vocabulary_size)
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.rope_base = float(rope_base)
        self.num_oscillators = int(num_oscillators)
        self.bank_lambdas = [float(v) for v in bank_lambdas]
        self.num_banks = len(self.bank_lambdas)
        self.output_gain = float(output_gain)
        self.general_pair_kernels = bool(general_pair_kernels)
        self.general_in_output = bool(general_in_output)
        self.general_concat = bool(general_concat)
        if general_concat_norm not in ("separate", "joint"):
            raise ValueError(f"`general_concat_norm` must be 'separate' or 'joint', got {general_concat_norm!r}")
        self.general_concat_norm = general_concat_norm
        if memory_hidden is not None and int(memory_hidden) <= 0:
            raise ValueError(f"`memory_hidden` must be positive, got {memory_hidden}")
        self.memory_hidden = int(memory_hidden) if memory_hidden is not None else None
        self.memory_heads = int(memory_heads)
        self.memory_window = int(memory_window) if memory_window is not None else self.window_size
        self.memory_value_range = int(memory_value_range)
        self.memory_oscillators = int(memory_oscillators)
        self.memory_mlp_hidden = int(memory_mlp_hidden)
        self.bank_projections = bool(bank_projections)
        self.tie_embeddings = bool(tie_embeddings)
        if self.general_concat and self.memory_hidden is None:
            raise ValueError("`general_concat` needs `memory_hidden` (the block that reads the concatenation)")
        if self.memory_heads:
            if self.memory_hidden is None:
                raise ValueError("`memory_heads` needs `memory_hidden` (the final block reads the retrieved values)")
            if not 1 <= self.memory_value_range <= self.window_size:
                raise ValueError(f"`memory_value_range` must be in [1, window_size], got {memory_value_range}")
            if self.memory_window < 1:
                raise ValueError(f"`memory_window` must be >= 1, got {memory_window}")

        half = self.dim // 2
        inv_freq = 1.0 / (self.rope_base ** (np.arange(half, dtype=np.float64) / half))
        distance = np.arange(self.window_size, 0, -1, dtype=np.float64)  # context-aligned: last = 1
        angles = distance[:, None] * inv_freq[None, :]
        self._cos = np.cos(angles).astype("float32")
        self._sin = np.sin(angles).astype("float32")
        self._distance = distance.astype("float32")
        self._log_distance = np.log(distance).astype("float32")

        self.token_embedding = layers.Embedding(self._vocabulary_size, self.dim, name="token_emb")
        if self.tie_embeddings:
            self.readout = _TiedReadout(self.token_embedding)
        else:
            self.readout = layers.Dense(self._vocabulary_size, use_bias=False, name="readout")
        self.router = layers.Dense(self.num_banks, use_bias=False, kernel_initializer="zeros", name="bank_router")
        if self.memory_heads:
            self.bind_router = layers.Dense(self.num_banks, use_bias=False, kernel_initializer="zeros", name="bind_router")
            self.unbind_router = layers.Dense(self.num_banks, use_bias=False, kernel_initializer="zeros", name="unbind_router")
        if self.memory_hidden is not None:
            self.memory_norm = RMSNorm(name="memory_norm")
            if self.general_concat and self.general_concat_norm == "separate":
                self.general_norm = RMSNorm(name="general_norm")
            if self.general_concat and self.general_concat_norm == "joint":
                self.concat_norm = RMSNorm(name="concat_norm")  # over [mix ; general]
            if self.memory_heads:
                self.read_norm = RMSNorm(name="read_norm")  # per read (last axis D), scale shared by the reads
            self.memory_in = layers.Dense(self.memory_hidden, activation="gelu", name="memory_in")
            self.memory_out = layers.Dense(self.dim, kernel_initializer="zeros", name="memory_out")
        self.build()

    @property
    def context_length(self) -> int:
        if self.memory_heads:  # a read sees writes up to memory_window back, each with its own bank window
            return self.window_size + self.memory_window - 1
        return self.window_size

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    def build(self, input_shape: Optional[Any] = None) -> None:
        if self.built:
            return
        self.token_embedding.build((None, self.window_size))
        if not self.tie_embeddings:
            self.readout.build((None, self.dim))
        self.router.build((None, self.dim))
        if self.memory_hidden is not None:
            width = self.dim * (1 + int(self.general_concat) + self.memory_heads)  # [h ; (general) ; r_1 .. r_n]
            self.memory_norm.build((None, self.dim))  # h only; general and reads have their own norms
            if self.general_concat and self.general_concat_norm == "separate":
                self.general_norm.build((None, self.dim))
            if self.general_concat and self.general_concat_norm == "joint":
                self.concat_norm.build((None, 2 * self.dim))
            if self.memory_heads:
                self.read_norm.build((None, self.memory_heads, self.dim))
            self.memory_in.build((None, width))
            self.memory_out.build((None, self.memory_hidden))
        if self.memory_heads:
            self.bind_router.build((None, self.dim))
            self.unbind_router.build((None, self.dim))
            self._build_memory()
        n, T, V = self.num_oscillators, self.window_size, self._vocabulary_size
        tau = np.geomspace(1.0, T, n)
        rate, freq = 1.0 / tau, 0.5 / tau
        k = np.arange(1, T + 1, dtype=np.float64)
        A = _oscillator_basis("exp-oscillators", k, rate, freq, np.zeros(n)) * k[:, None]
        amp = np.linalg.solve(A.T @ A + 1e-3 * np.eye(n), A.T @ np.ones(T))  # w(k) ~ 1/k at init
        nb = self.num_banks
        # osc_*: one kernel per state (general first, then banks) -- or, with
        # general_pair_kernels, banks only, plus gen_osc_* with one kernel per RoPE pair.
        # All start as the same 1/k fit.
        P = nb if self.general_pair_kernels else nb + 1
        for prefix, rows in (("osc", P),) + ((("gen_osc", self.dim // 2),) if self.general_pair_kernels else ()):
            tile = lambda v, r=rows: np.tile(np.asarray(v)[None], (r, 1))
            setattr(self, f"{prefix}_rate", self.add_weight(name=f"{prefix}_rate", shape=(rows, n), initializer=_const(tile(np.log(np.expm1(rate))))))
            setattr(self, f"{prefix}_freq", self.add_weight(name=f"{prefix}_freq", shape=(rows, n), initializer=_const(tile(freq))))
            setattr(self, f"{prefix}_amp", self.add_weight(name=f"{prefix}_amp", shape=(rows, n), initializer=_const(tile(amp))))
            setattr(self, f"{prefix}_phase", self.add_weight(name=f"{prefix}_phase", shape=(rows, n), initializer="zeros"))
        self.bank_prior = self.add_weight(name="bank_prior", shape=(nb, V), initializer=_const(np.full((nb, V), np.log(np.expm1(1.0)))))
        lam0 = np.array(self.bank_lambdas)[:, None] * np.ones((nb, V))
        self.bank_lambda = self.add_weight(name="bank_lambda", shape=(nb, V), initializer=_const(np.log(lam0 / (1 - lam0))))
        self.log_gain = self.add_weight(name="log_gain", shape=(nb + 1,), initializer=_const(np.full(nb + 1, np.log(self.output_gain))))
        if self.bank_projections:  # (states, D, D): general first, then the banks; identity at init
            D = self.dim
            self.bank_proj = self.add_weight(name="bank_proj", shape=(nb + 1, D, D), initializer=_const(np.tile(np.eye(D)[None], (nb + 1, 1, 1))))
        self.built = True

    def _build_memory(self) -> None:
        NM, D, H, Mo = self.memory_heads, self.dim, self.memory_mlp_hidden, self.memory_oscillators
        half = D // 2
        O = 4 * Mo + half + 1  # oscillators (a, alpha, omega, phi), key angles, gate
        rng = np.random.default_rng(0)
        w1 = rng.normal(0, np.sqrt(2.0 / (D + H)), (D, NM, H))
        w2 = rng.normal(0, np.sqrt(2.0 / (H + O)), (NM, H, O))
        w2[:, :, : 4 * Mo] *= 0.1  # value kernel and gate start (nearly) context-independent
        w2[:, :, -1] *= 0.1
        b2 = np.zeros((NM, O))
        b2[:, 0] = 1.0  # one oscillator: the current token plus a fast tail
        b2[:, Mo: 2 * Mo] = np.log(np.expm1(np.geomspace(2.0, 0.02, Mo)))  # decay rates
        b2[:, 2 * Mo: 3 * Mo] = np.linspace(0.0, 1.0, Mo)  # frequencies
        b2[:, -1] = -2.0  # write gate starts nearly closed (~0.12)
        self.mem_bind_w1 = self.add_weight(name="mem_bind_w1", shape=(D, NM, H), initializer=_const(w1))
        self.mem_bind_b1 = self.add_weight(name="mem_bind_b1", shape=(NM, H), initializer="zeros")
        self.mem_bind_w2 = self.add_weight(name="mem_bind_w2", shape=(NM, H, O), initializer=_const(w2))
        self.mem_bind_b2 = self.add_weight(name="mem_bind_b2", shape=(NM, O), initializer=_const(b2))
        # The unbinder starts as a copy of the binder's key network: with uniform mixes at init,
        # read keys equal write keys.
        key = slice(4 * Mo, 4 * Mo + half)
        self.mem_unbind_w1 = self.add_weight(name="mem_unbind_w1", shape=(D, NM, H), initializer=_const(w1))
        self.mem_unbind_b1 = self.add_weight(name="mem_unbind_b1", shape=(NM, H), initializer="zeros")
        self.mem_unbind_w2 = self.add_weight(name="mem_unbind_w2", shape=(NM, H, half), initializer=_const(w2[:, :, key]))
        self.mem_unbind_b2 = self.add_weight(name="mem_unbind_b2", shape=(NM, half), initializer=_const(b2[:, key]))
        self.mem_value = self.add_weight(name="mem_value", shape=(NM, D, D), initializer=_const(np.tile(np.eye(D)[None], (NM, 1, 1))))

    # -- pieces ---------------------------------------------------------------

    def kernel(self, x: Any, states: slice = slice(None)) -> Any:
        """Rows `states` of the `osc_*` kernels at distances `x` `(len(states), ...)`."""
        rate = keras.ops.softplus(self.osc_rate)[states]
        return _OscKernel.apply(x, rate, self.osc_freq.value[states], self.osc_amp.value[states], self.osc_phase.value[states])

    def bank_kernel(self, x: Any) -> Any:
        """The banks' kernels at distances `x` `(banks, ...)`."""
        return self.kernel(x, slice(None) if self.general_pair_kernels else slice(1, None))

    def general_kernel(self, like: Any) -> Any:
        """Context-aligned general weights: `(T,)`, or `(D, T)` with one kernel per RoPE pair
        (both dims of a pair share it)."""
        ops = keras.ops
        dist = ops.convert_to_tensor(self._distance, dtype=like.dtype)
        if not self.general_pair_kernels:
            return self.kernel(dist[None], slice(0, 1))[0]
        half = self.dim // 2
        w = _OscKernel.apply(ops.broadcast_to(dist[None], (half, self.window_size)), ops.softplus(self.gen_osc_rate),
                             self.gen_osc_freq.value, self.gen_osc_amp.value, self.gen_osc_phase.value)  # (half, T)
        return ops.concatenate([w, w], axis=0)  # (D, T)

    def bank_numbers(self, ids: Any):
        """`(s, lambda)` per bank for `ids`: shapes `(banks,) + ids.shape`."""
        ops = keras.ops
        s = ops.softplus(ops.take(self.bank_prior, ids, axis=1))
        lam = ops.sigmoid(ops.take(self.bank_lambda, ids, axis=1))
        return s, lam

    def _state_tokens(self, x: Any, ids: Any):
        """Token vectors each state pools: `(general (..., D), banks (banks, ..., D))`. Without
        projections all states share `x` (banks get a broadcast axis of size 1).

        With projections, each state's view is a "virtual embedding table" `E_n = E W_n`, built
        once per forward (once per gradient step) and read by token like an embedding: V x D x D
        per state instead of a D x D matmul for every token in the batch (B*S >> V). One
        `(V, states * D)` table read along its first axis keeps the backward a plain row scatter.
        """
        if not self.bank_projections:
            return x, x[None]
        ops = keras.ops
        P, D = self.num_banks + 1, self.dim
        table = ops.einsum("vd,nde->vne", self.token_embedding.embeddings, self.bank_proj)  # (V, P, D)
        xp = ops.take(ops.reshape(table, (self._vocabulary_size, P * D)), ids, axis=0)  # (..., P * D)
        xp = ops.moveaxis(ops.reshape(xp, tuple(ids.shape) + (P, D)), -2, 0)  # (P, ..., D)
        return xp[0], xp[1:]

    def _mix(self, general: Any, banks: Any):
        """`general` (..., D), `banks` (banks, ..., D) -> the hidden vector and the router's mix
        weights `(..., banks)`. Hidden: `sum_b pi_b bank_b` (+ general with `general_in_output`),
        or `[mix ; general]` (..., 2D) with `general_concat`."""
        ops = keras.ops
        pi = ops.softmax(self.router(general), axis=-1)  # (..., banks)
        h = ops.sum(banks * ops.moveaxis(pi, -1, 0)[..., None], axis=0)
        return self._join(h, general), pi

    def _join(self, mix: Any, general: Any) -> Any:
        if self.general_concat:
            return keras.ops.concatenate([mix, general], axis=-1)
        return mix + general if self.general_in_output else mix

    def residual_base(self, h: Any) -> Any:
        """The `dim`-wide residual part of a hidden vector (what the readout gets besides the block)."""
        D = self.dim
        if self.general_concat:
            mix, general = h[..., :D], h[..., D:2 * D]
            return mix + general if self.general_in_output else mix
        return h[..., :D]

    # -- forward --------------------------------------------------------------

    def states(self, inputs: Any):
        """Single-target `(general (B, D), banks (banks, B, D))`."""
        ops = keras.ops
        ids = ops.cast(inputs, "int32")
        x = self.token_embedding(ids)
        half = self.dim // 2
        cos = ops.convert_to_tensor(self._cos, dtype=x.dtype)
        sin = ops.convert_to_tensor(self._sin, dtype=x.dtype)
        rot = lambda v: ops.concatenate([v[..., :half] * cos - v[..., half:] * sin, v[..., :half] * sin + v[..., half:] * cos], axis=-1)
        x0, xb = self._state_tokens(x, ids)
        r0, rb = rot(x0), rot(xb)  # (B, T, D), (banks or 1, B, T, D)
        gain = ops.exp(self.log_gain)
        w0 = self.general_kernel(x)  # (T,) or (D, T)
        if self.general_pair_kernels:
            general = ops.einsum("dt,btd->bd", w0, r0) / (ops.sum(ops.abs(w0), axis=-1) + 1e-6) * gain[0]
        else:
            general = ops.einsum("t,btd->bd", w0, r0) / (ops.sum(ops.abs(w0)) + 1e-6) * gain[0]
        s, lam = self.bank_numbers(ids)  # (banks, B, T)
        log_k = ops.convert_to_tensor(self._log_distance, dtype=x.dtype)
        w = s * self.bank_kernel(ops.exp(lam * log_k))  # (banks, B, T)
        pooled = ops.einsum("nbt,nbtd->nbd", w, rb) if self.bank_projections else ops.einsum("nbt,btd->nbd", w, rb[0])
        banks = pooled / (ops.sum(ops.abs(w), axis=-1, keepdims=True) + 1e-6)
        return general, banks * gain[1:, None, None]

    def hidden(self, inputs: Any) -> Any:
        if self.memory_heads:  # single target: the window's states, writes, one read at the end
            ids = keras.ops.cast(inputs, "int32")
            x = self.token_embedding(ids)
            general, banks = self._dense_states(ids, x)
            return self._memory_forward(ids, x, general, banks, n_query=1)[:, 0]
        return self._mix(*self.states(inputs))[0]

    def bank_mix(self, inputs: Any) -> Any:
        """Router weights `(B, banks)` for single-target windows."""
        return self._mix(*self.states(inputs))[1]

    def readout_input(self, h: Any) -> Any:
        """Residual feed-forward block before the readout (identity when `memory_hidden` is None)."""
        if self.memory_hidden is None:
            return h
        if not self.memory_heads and not self.general_concat:
            return h + self.memory_out(self.memory_in(self.memory_norm(h)))
        # h = [h ; (general) ; r_1 .. r_n]. The parts have very different scales (bank gains vs
        # gate x embedding scale for a read): each is normalized on its own before the block.
        D = self.dim
        base = self.residual_base(h)
        if self.general_concat and self.general_concat_norm == "joint":
            parts = [self.concat_norm(h[..., :2 * D])]  # one norm: the mix / general size ratio survives
        else:
            parts = [self.memory_norm(h[..., :D])]
            if self.general_concat:
                parts.append(self.general_norm(h[..., D:2 * D]))
        if self.memory_heads:
            reads = h[..., D * (1 + int(self.general_concat)):]
            lead = tuple(reads.shape[:-1])
            reads = self.read_norm(keras.ops.reshape(reads, lead + (self.memory_heads, D)))
            parts.append(keras.ops.reshape(reads, lead + (self.memory_heads * D,)))
        return base + self.memory_out(self.memory_in(keras.ops.concatenate(parts, axis=-1)))

    def call(self, inputs: Any, training: bool = False) -> Any:
        return self.readout(self.readout_input(self.hidden(inputs)))

    def dense_hidden(self, inputs: Any) -> Any:
        """All queries of a `(B, 2C-1)` input, C = context_length (query m sees inputs m..m+C-1):
        `(B, C, D)`, or `(B, C, D * (1 + memory_heads))` = `[h ; r_1 .. r_n]` with memories."""
        ids = keras.ops.cast(inputs, "int32")
        x = self.token_embedding(ids)
        general, banks = self._dense_states(ids, x)
        if self.memory_heads:
            return self._memory_forward(ids, x, general, banks, n_query=self.context_length)
        return self._mix(general, banks)[0]

    def _dense_constants(self, S: int, like: Any):
        cache = self.__dict__.setdefault("_banks_dense_consts", {})
        key = (S, str(getattr(like, "device", "")), str(like.dtype))
        if key not in cache:
            from model.archs._dense import rope_angles

            T = self.window_size
            Q = S - T + 1
            kc, ks = rope_angles(self.dim, self.rope_base, np.arange(S))
            qc, qs = rope_angles(self.dim, self.rope_base, np.arange(T, T + Q))  # i + 1 for query i = T-1+m
            t = lambda a: keras.ops.convert_to_tensor(a, dtype=like.dtype)
            cache[key] = (t(kc), t(-ks), t(qc), t(qs))
        return cache[key]

    def _complex_constants(self, S: int, like: Any):
        """Complex RoPE phases for keys `(S, D/2)` (angle -s) and queries `(Q, D/2)`, and the
        index that interleaves a half-split vector's pairs."""
        cache = self.__dict__.setdefault("_banks_complex_consts", {})
        key = (S, str(getattr(like, "device", "")))
        if key not in cache:
            kc, ks_neg, qc, qs = self._dense_constants(S, like)
            half = self.dim // 2
            perm = torch.stack([torch.arange(half), torch.arange(half) + half], -1).reshape(-1).to(like.device)
            cache[key] = (torch.complex(kc, ks_neg), torch.complex(qc, qs), perm)
        return cache[key]

    def _dense_keys(self, ids: Any, x: Any, S: int):
        """Each state's token vectors rotated back by their own position, `R_{-s} e_s`:
        `(general (B, S, D), banks (banks or 1, B, S, D))`, and `to_query` (rotate to the query
        frame, return to the half-split layout). With `_complex_rope` the keys are interleaved."""
        from model.archs._dense import rotate

        if not self._complex_rope:
            kc, ks_neg, qc, qs = self._dense_constants(S, x)
            x0, xb = self._state_tokens(x, ids)
            u = rotate(x0, kc, ks_neg)
            ub = rotate(xb, kc, ks_neg) if self.bank_projections else u[None]
            return u, ub, (lambda v: rotate(v, qc, qs)), None
        kph, qph, perm = self._complex_constants(S, x)
        E = self.token_embedding.embeddings
        if self.bank_projections:  # virtual tables with interleaved columns
            P, V = self.num_banks + 1, self._vocabulary_size
            W = self.bank_proj[:, :, perm]
            if self._stacked_lookup:  # (P * V, D) table, ids offset per state: rows come out as (P, B, S, D)
                table = torch.einsum("vd,nde->nve", E, W).reshape(P * V, -1)
                offs = (torch.arange(P, device=ids.device) * V).reshape((P,) + (1,) * ids.dim())
                xp = torch.nn.functional.embedding(ids[None].long() + offs, table)
            else:
                table = torch.einsum("vd,nde->vne", E, W).reshape(V, -1)
                xp = torch.nn.functional.embedding(ids.long(), table).reshape(tuple(ids.shape) + (P, -1)).movedim(-2, 0)
            up = _crot(xp, kph)
            return up[0], up[1:], (lambda v: _deinterleave(_crot(v, qph))), perm
        u = _crot(torch.nn.functional.embedding(ids.long(), E[:, perm]), kph)  # interleaved embedding table
        return u, u[None], (lambda v: _deinterleave(_crot(v, qph))), perm

    def _dense_states(self, ids: Any, x: Any):
        """General state and banks at every position with a full window of an `(B, S)` input:
        positions T-1 .. S-1, i.e. `Q = S - T + 1` states. Returns `(B, Q, D)` and `(banks, B, Q, D)`."""
        ops = keras.ops
        T = self.window_size
        S = int(ids.shape[1])
        Q = S - T + 1
        # keys: (B, S, D) for the general state, (banks or 1, B, S, D) for the banks
        u, ub, to_query, perm = self._dense_keys(ids, x, S)
        gain = ops.exp(self.log_gain)
        w0 = self.general_kernel(x)  # (T,) or (D, T): the same for every query
        # Band rows in compact form: [m, r] = weight of input m + r for query m (distance T - r).
        if self.general_pair_kernels:
            w0k = w0 if perm is None else w0[perm]  # in the keys' layout
            band0 = ops.broadcast_to(w0k[:, None, :], (self.dim, Q, T))  # (D, Q, T), shared by the batch
            g = ops.transpose(_band_matmul(band0, ops.transpose(u, (2, 1, 0))), (2, 1, 0))  # (B, Q, D)
            general = to_query(g) / (ops.sum(ops.abs(w0), axis=-1) + 1e-6) * gain[0]
        else:
            band0 = ops.broadcast_to(w0[None], (Q, T))  # (Q, T)
            general = to_query(_band_matmul(band0[None], u)) / (ops.sum(ops.abs(w0)) + 1e-6) * gain[0]
        # A bank weight s_b,t * w_b(k^lambda_b,t) depends only on (token, distance): tabulate it for
        # the whole vocabulary (banks, V, T), look rows up by token, then shift them into the band.
        # Read the table along its first axis, like an embedding: selecting along axis 1 of
        # (banks, V, T) makes the backward scatter into strided rows, ~30x slower on real text
        # (frequent tokens repeat) on the Iris Xe.
        nb = self.num_banks
        table = ops.reshape(ops.transpose(self.weight_table(), (1, 0, 2)), (self._vocabulary_size, nb * T))
        rows = ops.reshape(ops.take(table, ids, axis=0), (-1, S, nb, T))  # (B, S, banks, T)
        rows = ops.transpose(rows, (2, 0, 1, 3))  # (banks, B, S, T): rows[.., s, k-1]
        w = self._rows_to_compact(rows)  # (banks, B, Q, T)
        summed = _band_matmul(w, ub)  # (banks, B, Q, D)
        banks = to_query(summed / (ops.sum(ops.abs(w), axis=-1, keepdims=True) + 1e-6))
        return general, banks * gain[1:, None, None, None]

    def memory_trace(self, inputs: Any) -> Dict[str, Any]:
        """Single-target `(B, context_length)` windows: the memory's intermediate tensors (for
        reports). Write a is at window position T-1+a; the read is at the last position."""
        ids = keras.ops.cast(inputs, "int32")
        x = self.token_embedding(ids)
        general, banks = self._dense_states(ids, x)
        parts: Dict[str, Any] = {}
        parts["z"] = self._memory_forward(ids, x, general, banks, n_query=1, parts=parts)[:, 0]
        return parts

    def _memory_forward(self, ids: Any, x: Any, general: Any, banks: Any, n_query: int,
                        parts: Optional[Dict[str, Any]] = None) -> Any:
        """Writes at every state position, reads at the last `n_query` ones.

        State position a corresponds to input position T-1+a. Returns `(B, n_query, D * (1 + NM))`.
        `parts`, if given, is filled with the intermediates (gates, keys, value kernels, values...).
        """
        ops = keras.ops
        NM, D, T, W, R, Mo = self.memory_heads, self.dim, self.window_size, self.memory_window, self.memory_value_range, self.memory_oscillators
        half = D // 2
        Qb = int(general.shape[1])
        n0 = Qb - n_query
        mix = lambda pi: ops.sum(banks * ops.moveaxis(pi, -1, 0)[..., None], axis=0)  # (B, Qb, D)
        pi_out = ops.softmax(self.router(general[:, n0:]), axis=-1)
        mix_q = ops.sum(banks[:, :, n0:] * ops.moveaxis(pi_out, -1, 0)[..., None], axis=0)
        h = self._join(mix_q, general[:, n0:])
        pi_bind = ops.softmax(self.bind_router(general), axis=-1)
        pi_unbind = ops.softmax(self.unbind_router(general[:, n0:]), axis=-1)
        c_bind = mix(pi_bind)
        c_unbind = ops.sum(banks[:, :, n0:] * ops.moveaxis(pi_unbind, -1, 0)[..., None], axis=0)

        # binder: oscillator value kernel over the last R tokens, write key, gate
        z = ops.gelu(ops.einsum("bqd,dnh->bqnh", c_bind, self.mem_bind_w1) + self.mem_bind_b1)
        out = ops.einsum("bqnh,nho->bqno", z, self.mem_bind_w2) + self.mem_bind_b2  # (B, Qb, NM, O)
        amp, rate, freq, phase = (out[..., i * Mo:(i + 1) * Mo] for i in range(4))
        rate = ops.softplus(rate)
        lag = ops.convert_to_tensor(np.arange(R, dtype="float32"), dtype=x.dtype)  # k - 1
        kern = 0.0
        for m in range(Mo):
            kern = kern + amp[..., m:m + 1] * ops.exp(-rate[..., m:m + 1] * lag) * ops.cos(freq[..., m:m + 1] * lag + phase[..., m:m + 1])
        phi = out[..., 4 * Mo: 4 * Mo + half]
        gate = ops.sigmoid(out[..., -1:])
        # value: kernel-weighted, RoPE-ordered (relative to the write) average of projected tokens
        # The token k-1 before write a (input j = T-k+a) is rotated by R_k = R_{T+a} R_{-j}: rotate
        # tokens back by their own position once, take the R-wide window sum as a banded matmul,
        # then rotate each write forward by T+a.
        from model.archs._dense import rotate

        compact = ops.transpose(ops.flip(kern, axis=-1), (0, 2, 1, 3))  # [a, r] = kern[a, R-1-r]
        if self._complex_rope:  # interleaved pairs: P_n's output columns permuted, rotations as complex multiplies
            kph, qph, perm = self._complex_constants(int(x.shape[1]), x)
            pu = _crot(torch.einsum("bsd,nde->bnse", x, self.mem_value[:, :, perm]), kph)  # (B, NM, S, D)
            summed = _band_matmul(compact, pu[:, :, T - R:])  # (B, NM, Qb, D)
            value = ops.transpose(_deinterleave(_crot(summed, qph)), (0, 2, 1, 3))  # (B, Qb, NM, D)
        else:
            kc, ks_neg, qc, qs = self._dense_constants(int(x.shape[1]), x)
            pe = ops.einsum("bsd,nde->bsne", x, self.mem_value)  # (B, S, NM, D)
            pu = rotate(pe, kc[:, None], ks_neg[:, None])
            keys = ops.transpose(pu[:, T - R:], (0, 2, 1, 3))  # (B, NM, Qb + R - 1, D): inputs T-R ..
            summed = ops.transpose(_band_matmul(compact, keys), (0, 2, 1, 3))  # (B, Qb, NM, D)
            value = rotate(summed, qc[:, None], qs[:, None])
        value = value / (ops.sum(ops.abs(kern), axis=-1, keepdims=True) + 1e-6)
        # bind (complex multiply by the unit-phasor key), gate, bundle over the last W writes
        kr, ki = ops.cos(phi), ops.sin(phi)
        vr, vi = value[..., :half], value[..., half:]
        bound = gate * ops.concatenate([kr * vr - ki * vi, kr * vi + ki * vr], axis=-1)  # (B, Qb, NM, D)
        # Gate-weighted average rather than a sum: divide by the total gate in the window, so the
        # memory stays at value scale however many writes are open (|M| <= max |v|).
        window_sum = lambda t: (lambda cs: cs[:, n0:] - ops.concatenate([ops.zeros_like(cs[:, :W]), cs], axis=1)[:, n0:n0 + n_query])(ops.cumsum(t, axis=1))
        memory = window_sum(bound) / (window_sum(gate) + 1e-6)  # writes a-W+1 .. a
        # unbind with the read key
        z = ops.gelu(ops.einsum("bqd,dnh->bqnh", c_unbind, self.mem_unbind_w1) + self.mem_unbind_b1)
        psi = ops.einsum("bqnh,nho->bqno", z, self.mem_unbind_w2) + self.mem_unbind_b2  # (B, nq, NM, half)
        qr, qi = ops.cos(psi), ops.sin(psi)
        mr, mi = memory[..., :half], memory[..., half:]
        read = ops.concatenate([qr * mr + qi * mi, qr * mi - qi * mr], axis=-1)  # conj(q) * M
        if parts is not None:
            parts.update(pi_out=pi_out, pi_bind=pi_bind, pi_unbind=pi_unbind, gate=gate[..., 0], phi=phi,
                         kern=kern, value=value, psi=psi, read=read)
        read = ops.reshape(read, (-1, n_query, NM * D))
        return ops.concatenate([h, read], axis=-1)

    def weight_table(self) -> Any:
        """`(banks, V, T)`: weight of token v at distance k (index k - 1) in each bank, prior included."""
        ops = keras.ops
        log_k = ops.convert_to_tensor(np.log(np.arange(1, self.window_size + 1, dtype="float32")))
        s = ops.softplus(self.bank_prior)[:, :, None]  # (banks, V, 1)
        lam = ops.sigmoid(self.bank_lambda)[:, :, None]
        return s * self.bank_kernel(ops.exp(lam * log_k))

    def _rows_to_compact(self, rows: Any) -> Any:
        """`rows[.., s, k-1]` (per input position, weight at each distance) -> the band in
        compact form `(.., Q, T)`: `[m, r]` = weight of input `m + r` for query m, i.e.
        `rows[m + r, T - 1 - r]` (distance T - r).

        Flip k so column r holds distance T - r, then read the sheared diagonals
        `C[r, m] = G[r, r + m]` with a pad-and-reshape (row stride S + 1). Only reshapes
        and copies, no gather.
        """
        ops = keras.ops
        T = self.window_size
        S = int(rows.shape[-2])
        Q = S - T + 1
        lead = tuple(rows.shape[:-2])
        g = ops.swapaxes(ops.flip(rows, axis=-1), -1, -2)  # (.., T(r), S): g[r, s] = rows[s, T-1-r]
        flat = ops.reshape(g, lead + (T * S,))
        flat = ops.pad(flat, [[0, 0]] * len(lead) + [[0, T]])
        compact = ops.reshape(flat, lead + (T, S + 1))[..., :Q]  # [r, m] = g[r, r + m]
        return ops.swapaxes(compact, -1, -2)  # [m, r]

    def with_context_length(self, context_length: int) -> "RopeTokenBanksLM":
        return _context.with_context_length(self, context_length)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "vocabulary_size": self._vocabulary_size, "dim": self.dim,
                "window_size": self.window_size, "rope_base": self.rope_base,
                "num_oscillators": self.num_oscillators, "bank_lambdas": self.bank_lambdas,
                "output_gain": self.output_gain, "general_pair_kernels": self.general_pair_kernels,
                "general_in_output": self.general_in_output, "general_concat": self.general_concat,
                "general_concat_norm": self.general_concat_norm,
                "memory_hidden": self.memory_hidden,
                "memory_heads": self.memory_heads, "memory_window": self.memory_window,
                "memory_value_range": self.memory_value_range, "memory_oscillators": self.memory_oscillators,
                "memory_mlp_hidden": self.memory_mlp_hidden, "bank_projections": self.bank_projections,
                "tie_embeddings": self.tie_embeddings}
