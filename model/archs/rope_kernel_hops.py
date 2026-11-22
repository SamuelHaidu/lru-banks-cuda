"""RoPE kernel hops: multi-hop refinement of the kernel-attention read.

Hop 1 is exactly `rope-kernel-attention` (with residual): fixed kernel ->
MLP-generated kernel -> FFN-query attention, giving the state

    s_1 = h1 + g_1 a_1

Every further hop re-reads the same rotated token sequence, conditioned on
what the previous hops found:

    alpha_h = MLP_k,h(s_{h-1})                 new kernel amplitudes ("waves")
    c_h     = sum_k w_k(alpha_h) v_k           kernel read with those waves
    z_h     = s_{h-1} + gk_h c_h
    q_h     = MLP_q,h(z_h)                     new query
    a_h     = attention(q_h)                   QK-norm, own per-head temperature
    s_h     = z_h + g_h a_h

    logits  = W_out^T s_H

`gk_h` and `g_h` are learned scalars starting at 0 (ReZero), so a fresh
model starts exactly as the one-hop model and the hops switch on as they
earn it. The attended values are RoPE-rotated by distance, so `s_{h-1}`
carries the phase of where earlier hops looked; the hop's kernel MLP can use
it to place its waves relative to that position (e.g. just after a found
name), though the 8-timescale basis can only emphasize regions -- the hop's
attention does the precise pick.

`output_mode="refine"` makes hops 2+ pure retrieval refiners: hop 1 is
unchanged (kernel + attention), and each later hop's kernel read is used
only to build its query, never added to the output. The hop-h kernel
generator gets `a_{h-1}`, the hop-h query generator gets that hop's kernel
read `c_h`, and

    logits = W_out^T (h1 + g_1 a_1 + g_2 a_2 + ...)   (output gates start at 0)

`context_dim` (refine mode only) widens the accumulated state past the
token width `dim`: every pooled read (static sum, adaptive kernel read,
each hop's attention read and kernel read) gets its own linear map
`dim -> context_dim`, the kernel and query generators read those
`context_dim` vectors, and the output state is

    s = P_s h0 + P_c c_1 + g_1 P_a1 a_1 + g_2 P_a2 a_2 + ...

Queries and keys stay `dim` wide (the rotated tokens).

`state_norm=True` (needs `context_dim`) RMS-normalizes, pre-norm style,
every `context_dim` vector before it is read: the inputs of each kernel and
query generator and the state before the readout. The accumulated sum
itself stays unnormalized.

Hops 2+ can use their own basis (`hop_kernel_basis`, `hop_num_centers`),
e.g. `"fourier"` with many frequencies: fine enough to aim waves at a
specific distance, which the 8 smooth oscillator timescales cannot.

The same token sequence serves every hop, so a hop costs two small MLPs, one
kernel sum and one attention read; nothing is recomputed per position. The
dense path (`dense_hidden`) applies the same hops to every query position.
"""

from typing import Any, Dict, List, Optional, Tuple

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402

from model.archs._dense import dense_kernel_sum  # noqa: E402
from model.archs._norm import RMSNorm  # noqa: E402
from model.archs.rope_kernel_attention import RopeKernelAttentionLM  # noqa: E402
from model.archs.rope_log_gaussian_sum import KERNEL_BASES, _build_basis, _initial_amplitudes  # noqa: E402


def _swiglu(gate_1, up_1, gate_2, up_2, out, x):
    ops = keras.ops
    z = ops.silu(gate_1(x)) * up_1(x)
    z = ops.silu(gate_2(z)) * up_2(z)
    return out(z)


@keras.saving.register_keras_serializable(package="model")
class RopeKernelHopsLM(RopeKernelAttentionLM):
    """`RopeKernelAttentionLM` (residual) + `num_hops - 1` further kernel/attention hops.

    Args (beyond `RopeKernelAttentionLM`'s):
        num_hops: Total hops, >= 2 (hop 1 is the parent model).
        hop_kernel_basis: Basis for hops 2+ (default: `kernel_basis`).
        hop_num_centers: Basis size for hops 2+ (default: `num_centers`).
        output_mode: `"residual"` (running sum of kernel and attention reads) or
            `"refine"` (hop 1 kernel + attention, later hops add attention only).
        context_dim: Width of the accumulated state (default `dim`: no
            projections). Refine mode only.
        state_norm: RMSNorm (learned gain) on every `context_dim` generator
            input and before the readout. Needs `context_dim`.
    """

    _defer_build_attention = True  # hop layers must exist before build()

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
        attention_residual: bool = True,
        key_value_projection: bool = False,
        num_hops: int = 2,
        hop_kernel_basis: Optional[str] = None,
        hop_num_centers: Optional[int] = None,
        output_mode: str = "residual",
        context_dim: Optional[int] = None,
        state_norm: bool = False,
        **kwargs: Any,
    ):
        if not attention_residual:
            raise ValueError("rope-kernel-hops needs attention_residual=True: each hop adds to a running state.")
        if output_mode not in ("residual", "refine"):
            raise ValueError(f"`output_mode` must be 'residual' or 'refine', got {output_mode!r}")
        self_output_mode = output_mode
        if int(num_hops) < 2:
            raise ValueError(f"`num_hops` must be >= 2 (use rope-kernel-attention for 1), got {num_hops}")
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
            num_heads=num_heads,
            query_hidden=query_hidden,
            attention_residual=True,
            key_value_projection=key_value_projection,
            **kwargs,
        )
        self.num_hops = int(num_hops)
        self.output_mode = self_output_mode
        self.context_dim = int(context_dim) if context_dim is not None else None
        if self.context_dim is not None:
            if self.context_dim <= 0:
                raise ValueError(f"`context_dim` must be positive, got {context_dim}")
            if self.output_mode != "refine":
                raise ValueError("`context_dim` needs output_mode='refine'")
            self.state_dim = self.context_dim
            proj = lambda name: layers.Dense(self.context_dim, use_bias=False, name=name)
            self.state_static = proj("state_static")
            self.state_adaptive = proj("state_adaptive")
            self.state_attention = [proj(f"state_attention_{h}") for h in range(1, self.num_hops + 1)]
            self.hop_query_input = [proj(f"hop{h}_query_input") for h in range(2, self.num_hops + 1)]
        self.state_norm = bool(state_norm)
        if self.state_norm:
            if self.context_dim is None:
                raise ValueError("`state_norm` needs `context_dim`")
            norm = lambda name: RMSNorm(name=name)
            # Inputs of [kernel generator, query generator] per hop, and the readout's.
            self.state_norms = [[norm(f"{p}kernel_norm"), norm(f"{p}query_norm")]
                                for p in [""] + [f"hop{h}_" for h in range(2, self.num_hops + 1)]]
            self.output_norm = norm("output_norm")
        self.hop_kernel_basis = str(hop_kernel_basis or kernel_basis)
        self.hop_num_centers = int(hop_num_centers or num_centers)
        if self.hop_kernel_basis not in KERNEL_BASES:
            raise ValueError(f"`hop_kernel_basis` must be one of {KERNEL_BASES}, got {self.hop_kernel_basis!r}")
        if self.hop_num_centers < 2:
            raise ValueError(f"`hop_num_centers` must be >= 2, got {self.hop_num_centers}")
        self._hop_basis, self._hop_basis_labels = _build_basis(
            self.hop_kernel_basis, self._distance.astype(np.float64), self.hop_num_centers, self.basis_window)
        bias_init = _initial_amplitudes(self.hop_kernel_basis, self._hop_basis, self._distance.astype(np.float64))
        n_amp = self._hop_basis.shape[1]
        if self.key_value_projection:
            for h in range(2, self.num_hops + 1):
                self._add_key_value(f"hop{h}_")
        self.hop_kernel_mlps: List[List[Any]] = []
        self.hop_query_mlps: List[List[Any]] = []
        for h in range(2, self.num_hops + 1):
            self.hop_kernel_mlps.append([
                layers.Dense(self.mlp_hidden, use_bias=False, name=f"hop{h}_kernel_gate_1"),
                layers.Dense(self.mlp_hidden, use_bias=False, name=f"hop{h}_kernel_up_1"),
                layers.Dense(self.mlp_hidden, use_bias=False, name=f"hop{h}_kernel_gate_2"),
                layers.Dense(self.mlp_hidden, use_bias=False, name=f"hop{h}_kernel_up_2"),
                layers.Dense(
                    n_amp,
                    kernel_initializer="zeros",
                    bias_initializer=lambda shape, dtype=None: keras.ops.convert_to_tensor(bias_init, dtype=dtype),
                    name=f"hop{h}_kernel_out",
                ),
            ])
            self.hop_query_mlps.append([
                layers.Dense(self.query_hidden, use_bias=False, name=f"hop{h}_query_gate_1"),
                layers.Dense(self.query_hidden, use_bias=False, name=f"hop{h}_query_up_1"),
                layers.Dense(self.query_hidden, use_bias=False, name=f"hop{h}_query_gate_2"),
                layers.Dense(self.query_hidden, use_bias=False, name=f"hop{h}_query_up_2"),
                layers.Dense(self.dim, use_bias=False, name=f"hop{h}_query_out"),
            ])
        self.build()

    def build(self, input_shape: Any = None) -> None:
        if self.built:
            return
        self.hop_log_temperature = []
        self.hop_kernel_gate = []
        self.hop_attention_gate = []
        for i, h in enumerate(range(2, self.num_hops + 1)):
            k = self.hop_kernel_mlps[i]
            for layer, width in zip(k, (self.state_dim, self.state_dim, self.mlp_hidden, self.mlp_hidden, self.mlp_hidden)):
                layer.build((None, width))
            qm = self.hop_query_mlps[i]
            for layer, width in zip(qm, (self.state_dim, self.state_dim, self.query_hidden, self.query_hidden, self.query_hidden)):
                layer.build((None, width))
            self.hop_log_temperature.append(self.add_weight(
                name=f"hop{h}_log_temperature", shape=(self.num_heads,),
                initializer=keras.initializers.Constant(float(np.log(10.0)))))
            self.hop_kernel_gate.append(self.add_weight(name=f"hop{h}_kernel_gate", shape=(), initializer="zeros"))
            self.hop_attention_gate.append(self.add_weight(name=f"hop{h}_attention_gate", shape=(), initializer="zeros"))
        if self.context_dim is not None:
            for layer in [self.state_static, self.state_adaptive, *self.state_attention, *self.hop_query_input]:
                layer.build((None, self.dim))
        if self.state_norm:
            for layer in [*(n for pair in self.state_norms for n in pair), self.output_norm]:
                layer.build((None, self.context_dim))
        super().build(input_shape)
        if self.output_mode == "refine":
            # Here the kernel gate scales a hop's kernel read into its *query*
            # generator, not into the output: it starts at 1 (a zero query input
            # would sit on the QK-norm singularity). Output gates stay at 0.
            # Loading a checkpoint overwrites these afterwards.
            for g in self.hop_kernel_gate:
                g.assign(1.0)

    # -- forward --------------------------------------------------------------

    def _hop_amplitudes(self, i: int, state: Any) -> Any:
        alpha = _swiglu(*self.hop_kernel_mlps[i], state)
        return keras.ops.softplus(alpha) if self.hop_kernel_basis == "log-gaussian" else alpha

    def _hop_basis_tensor(self, like: Any) -> Any:
        cache = self.__dict__.setdefault("_hop_basis_cache", {})
        key = (str(getattr(like, "device", "")), str(like.dtype))
        if key not in cache:
            cache[key] = keras.ops.convert_to_tensor(self._hop_basis, dtype=like.dtype)
        return cache[key]

    def hop_basis_labels(self) -> list:
        return list(self._hop_basis_labels)

    def _attention_only_hops(self, rotated: Any, h1: Any, w1: Any, read: Any, kernel_read: Any, amps_to_w: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        """`output_mode="refine"`: out = h1 + sum_h g_h a_h; hop h's kernel generator
        reads a_{h-1}, its query generator reads its own kernel read c_h."""
        ops = keras.ops
        a, probs = read(self._query(h1), self.log_temperature, 0)
        out = h1 + ops.cast(self.attention_gate, a.dtype) * a
        trace = [{"kernel": w1, "probs": probs}]
        for i in range(self.num_hops - 1):
            w = amps_to_w(self._hop_amplitudes(i, a))
            c = ops.cast(self.hop_kernel_gate[i], a.dtype) * kernel_read(w)
            a, probs = read(_swiglu(*self.hop_query_mlps[i], c), self.hop_log_temperature[i], i + 1)
            out = out + ops.cast(self.hop_attention_gate[i], a.dtype) * a
            trace.append({"kernel": w, "probs": probs})
        return out, trace

    def _kernel_input(self, h0: Any) -> Any:
        if self.context_dim is None:
            return h0
        return self._norm(0, 0, self.state_static(h0))

    def _norm(self, hop: int, which: int, x: Any) -> Any:
        """`state_norm` on generator input `which` (0 kernel, 1 query) of hop index `hop`."""
        return self.state_norms[hop][which](x) if self.state_norm else x

    def readout_input(self, h: Any) -> Any:
        return super().readout_input(self.output_norm(h) if self.state_norm else h)

    def _projected_hops(self, h0: Any, read: Any, kernel_read: Any, amps_to_w1: Any, amps_to_w: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        """Refine mode with a `context_dim` state: each pooled read is mapped
        `dim -> context_dim` by its own matrix before it is used or accumulated."""
        ops = keras.ops
        s = self.state_static(h0)
        w1 = amps_to_w1(self._amplitudes(self._norm(0, 0, s)))
        s = s + self.state_adaptive(kernel_read(w1))
        a, probs = read(self._query(self._norm(0, 1, s)), self.log_temperature, 0)
        pa = self.state_attention[0](a)
        out = s + ops.cast(self.attention_gate, pa.dtype) * pa
        trace = [{"kernel": w1, "probs": probs}]
        for i in range(self.num_hops - 1):
            w = amps_to_w(self._hop_amplitudes(i, self._norm(i + 1, 0, pa)))
            c = ops.cast(self.hop_kernel_gate[i], pa.dtype) * self.hop_query_input[i](kernel_read(w))
            a, probs = read(_swiglu(*self.hop_query_mlps[i], self._norm(i + 1, 1, c)), self.hop_log_temperature[i], i + 1)
            pa = self.state_attention[i + 1](a)
            out = out + ops.cast(self.hop_attention_gate[i], pa.dtype) * pa
            trace.append({"kernel": w, "probs": probs})
        return out, trace

    def hops(self, inputs: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        """Single-target forward. Returns `(state (B, dim), trace)`; `trace[h]` holds
        hop `h+1`'s `kernel` `(B, T)` (context-aligned, None for hop 1's static
        part) and attention `probs` `(B, H, T)`."""
        ops = keras.ops
        if self.context_dim is not None:
            rotated, h0 = self._static_pass(inputs)
            basis1, basis = self._constants(rotated)[2], self._hop_basis_tensor(rotated)
            return self._projected_hops(
                h0,
                read=lambda q, log_t, r: self._attend(q, rotated, inputs, log_t, r),
                kernel_read=lambda w: ops.einsum("bt,btd->bd", w, rotated),
                amps_to_w1=lambda alpha: ops.einsum("bm,tm->bt", alpha, basis1),
                amps_to_w=lambda alpha: ops.einsum("bm,tm->bt", alpha, basis),
            )
        rotated, h0, _, w1 = self.kernels(inputs)
        h1 = h0 + ops.einsum("bt,btd->bd", w1, rotated)
        if self.output_mode == "refine":
            basis = self._hop_basis_tensor(rotated)
            return self._attention_only_hops(
                rotated, h1, w1,
                read=lambda q, log_t, r: self._attend(q, rotated, inputs, log_t, r),
                kernel_read=lambda w: ops.einsum("bt,btd->bd", w, rotated),
                amps_to_w=lambda alpha: ops.einsum("bm,tm->bt", alpha, basis),
            )
        a, probs = self._attend(self._query(h1), rotated, inputs, self.log_temperature)
        state = h1 + ops.cast(self.attention_gate, a.dtype) * a
        trace = [{"kernel": w1, "probs": probs}]
        basis = self._hop_basis_tensor(rotated)
        for i in range(self.num_hops - 1):
            w = ops.einsum("bm,tm->bt", self._hop_amplitudes(i, state), basis)
            z = state + ops.cast(self.hop_kernel_gate[i], state.dtype) * ops.einsum("bt,btd->bd", w, rotated)
            a, probs = self._attend(_swiglu(*self.hop_query_mlps[i], z), rotated, inputs, self.hop_log_temperature[i], i + 1)
            state = z + ops.cast(self.hop_attention_gate[i], a.dtype) * a
            trace.append({"kernel": w, "probs": probs})
        return state, trace

    def hidden(self, inputs: Any) -> Any:
        return self.hops(inputs)[0]

    def dense_hidden(self, inputs: Any) -> Any:
        """`(B, T, dim)` states for all `T` queries of a `(B, 2T-1)` input."""
        ops = keras.ops
        if self.context_dim is not None:
            u, s0, rope = self._dense_static_pass(inputs)
            basis1, basis = self._constants(u)[2], self._hop_basis_tensor(u)
            return self._projected_hops(
                rope.to_query_frame(s0),
                read=lambda q, log_t, r: self._dense_attend(q, u, inputs, rope, log_t, r),
                kernel_read=lambda w: rope.to_query_frame(dense_kernel_sum(w, u)),
                amps_to_w1=lambda alpha: ops.einsum("bmk,tk->bmt", alpha, basis1),
                amps_to_w=lambda alpha: ops.einsum("bmk,tk->bmt", alpha, basis),
            )[0]
        u, h1, rope = self.dense_kernels(inputs)
        if self.output_mode == "refine":
            basis = self._hop_basis_tensor(u)
            return self._attention_only_hops(
                u, h1, None,
                read=lambda q, log_t, r: self._dense_attend(q, u, inputs, rope, log_t, r),
                kernel_read=lambda w: rope.to_query_frame(dense_kernel_sum(w, u)),
                amps_to_w=lambda alpha: ops.einsum("bmk,tk->bmt", alpha, basis),
            )[0]
        a, _ = self._dense_attend(self._query(h1), u, inputs, rope, self.log_temperature)
        state = h1 + ops.cast(self.attention_gate, a.dtype) * a
        basis = self._hop_basis_tensor(u)
        for i in range(self.num_hops - 1):
            w = ops.einsum("bmk,tk->bmt", self._hop_amplitudes(i, state), basis)  # (B, T, T) context-aligned
            c = rope.to_query_frame(dense_kernel_sum(w, u))
            z = state + ops.cast(self.hop_kernel_gate[i], state.dtype) * c
            a, _ = self._dense_attend(_swiglu(*self.hop_query_mlps[i], z), u, inputs, rope, self.hop_log_temperature[i], i + 1)
            state = z + ops.cast(self.hop_attention_gate[i], a.dtype) * a
        return state

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "num_hops": self.num_hops,
                "hop_kernel_basis": self.hop_kernel_basis, "hop_num_centers": self.hop_num_centers,
                "output_mode": self.output_mode, "context_dim": self.context_dim, "state_norm": self.state_norm}
