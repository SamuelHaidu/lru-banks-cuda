"""`gated-deltanet`: a stack of Gated DeltaNet token mixers (arXiv 2412.06464) and mixture-of-experts FFNs.

One layer (n), pre-norm residual stream x `(B, T, dim)`:

    u        = rms_norm(x) * attn_norm
    q, k, v  = silu(conv(u Wqkv))                    causal depthwise conv (conv_size taps), never across a <bos>
    g        = -exp(A_log) * softplus(u Wa + dt_bias)    per head: log alpha_t (Mamba2 parametrization)
    beta     = sigmoid(u Wb)                          per head
    o        = gated_delta_rule(q, k, v, g, beta)     per head, state K x V (`model.archs.delta_rule`)
    x       += (rms_norm(o) * o_norm * silu(u Wgate)) Wo
    h        = rms_norm(x) * ffn_norm
    x       += p_e(h) * expert_e(h),  e = argmax(h Wr + router_bias)      top-1 of `num_experts`
    expert_e(h) = gelu(h W1_e + b1_e) W2_e + b2_e

logits = rms_norm(x) * final_norm @ readout. Documents restart at `reset_token` (the delta-rule state and the conv
taps). Expert load is balanced without an auxiliary loss (DeepSeek-V3): a per-expert bias, added to the router
logits for the choice only, moves by `router_bias_rate` towards the mean load after every training forward
(`router_bias`, not trained, saved with the weights).

On CUDA the matmuls run in bf16 on the tensor cores (autocast; the residual stream, norms, gates and the loss stay
fp32) and the delta rule is fla's fused Triton kernel. Elsewhere (Iris Xe, CPU) fp32 and the torch chunked delta rule.
Packed rows like `min-gru` (`packed_documents`), dense supervision, stateful generation (`advance`).
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model.archs._dense import DenseSupervisionMixin  # noqa: E402
from model.archs.delta_rule import gated_delta_rule  # noqa: E402
from model.interfaces import BaseLM  # noqa: E402


def _rms(x: torch.Tensor, w: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x if w is None else x * w


# -- the fused CUDA training path (torch.compile -> Triton: each elementwise chain around a matmul is one kernel) ----
# Unfused, 83% of the GPU time went to elementwise passes (conv taps, norms, casts, gelu, adds) and 9% to tensor-core
# matmuls (7.5 TFLOPS of the A5000's 111 bf16 peak). Matmuls in bf16 (tensor cores), everything else fp32 in registers.
FUSED = os.environ.get("GDN_FUSED", "1") != "0"
BF = torch.bfloat16
import torch._dynamo  # noqa: E402

torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
torch._dynamo.config.accumulated_recompile_limit = max(torch._dynamo.config.accumulated_recompile_limit, 1024)
import torch._functorch.config  # noqa: E402

# Memory: what the backward pass keeps. Measured per token and layer (dim 256, 4 x 64 heads, 6 experts of 1024): 29.2 KB
# at first (23 GB at 96 x 512 x 16 layers): the compiler saved the conv's shifted fp32 copies (14.2 KB in `_mixer_pre`)
# and the experts' hidden layer twice plus fp32 copies of the routed rows (10.8 KB in the MoE). Now ~11 KB: the compiled
# graphs recompute their elementwise parts in the backward pass (activation_memory_budget, a global AOTAutograd
# setting), the experts save only their input and pre-activation (bf16, `ExpertMLP`), routed rows and gates are bf16.
torch._functorch.config.activation_memory_budget = min(torch._functorch.config.activation_memory_budget, 0.5)
# the fp32 matmuls left (decay / write-strength gates, router, the cross-entropy readout) on the tensor cores as TF32
torch.backends.cuda.matmul.allow_tf32 = True


@torch.compile(fullgraph=True, dynamic=False)
def _mixer_pre(x, seg, attn_norm, w_qkv, conv, w_ab, dt_bias, A_log, w_gate, H: int, Dh: int):
    """Norm, q/k/v projection, the causal conv (taps of the same document, zeros before the row), silu, the decay
    and write-strength gates (fp32) and the output gate. x `(B, T, D)` fp32, seg `(B, T)` document numbers."""
    B, T, _ = x.shape
    u = _rms(x, attn_norm)
    ub = u.to(BF)
    qkv = (ub @ w_qkv.to(BF)).float()
    S = conv.shape[0]
    out = conv[S - 1] * qkv
    for i in range(1, S):
        shifted = F.pad(qkv, (0, 0, i, 0))[:, :T]
        same = (F.pad(seg, (i, 0), value=-1)[:, :T] == seg).to(qkv.dtype)[..., None]
        out = out + conv[S - 1 - i] * shifted * same
    q, k, v = F.silu(out).to(BF).reshape(B, T, 3, H, Dh).unbind(2)
    a, b = (u @ w_ab).split([H, H], -1)
    g = -torch.exp(A_log) * F.softplus(a + dt_bias)
    gate = F.silu((ub @ w_gate.to(BF)).float()).to(BF)
    return q, k, v, g, torch.sigmoid(b), gate


@torch.compile(fullgraph=True, dynamic=False)
def _mixer_proj(x, attn_norm, w_qkv, w_ab, dt_bias, A_log, w_gate, H: int):
    """`_mixer_pre` without the conv (the Triton kernel `gdn_kernels.conv_silu` does it): qkv (bf16), the decay /
    write-strength gates (fp32) and the output gate (bf16)."""
    u = _rms(x, attn_norm)
    ub = u.to(BF)
    qkv = ub @ w_qkv.to(BF)
    a, b = (u @ w_ab).split([H, H], -1)
    g = -torch.exp(A_log) * F.softplus(a + dt_bias)
    gate = F.silu((ub @ w_gate.to(BF)).float()).to(BF)
    return qkv, g, torch.sigmoid(b), gate


try:  # the Triton conv kernel (CUDA)
    from model.archs.gdn_kernels import conv_silu
except Exception:  # noqa: BLE001
    conv_silu = None
TRITON_CONV = os.environ.get("GDN_TRITON_CONV", "1") != "0"


@torch.compile(fullgraph=True, dynamic=False)
def _mixer_post(x, o, o_norm, gate, w_o):
    """x + (rms_norm(o) per head * o_norm * gate) @ w_o."""
    B, T, D = x.shape
    o = _rms(o, o_norm).reshape(B, T, -1) * gate.float()
    return x + (o.to(BF) @ w_o.to(BF)).float()


@torch.compile(fullgraph=True)
def _router(x, ffn_norm, router, router_bias):
    """h = rms_norm(x) in bf16 (the experts' input), the expert choice (with the balancing bias) and its probability
    (fp32)."""
    h = _rms(x, ffn_norm)
    logits = h @ router
    choice = torch.argmax(logits + router_bias, -1)
    return h.to(BF), choice, torch.softmax(logits, -1).gather(1, choice[:, None])


@torch.compile(fullgraph=True, dynamic=True)
def _gelu_fwd(a, b1):
    return F.gelu(a.float() + b1, approximate="tanh").to(BF)


@torch.compile(fullgraph=True, dynamic=True)
def _gelu_bwd(dz, a, b1):
    """(the GELU output again, d pre-activation) from the saved pre-activation a (bf16)."""
    pre = a.float() + b1
    return F.gelu(pre, approximate="tanh").to(BF), torch.ops.aten.gelu_backward(dz.float(), pre, approximate="tanh")


@torch.compile(fullgraph=True, dynamic=True)
def _gelu_bwd_sum(dz, a, b1):
    """(the GELU output again in bf16, d pre-activation in bf16, its column sum: d b1)."""
    pre = a.float() + b1
    da = torch.ops.aten.gelu_backward(dz.float(), pre, approximate="tanh")
    return F.gelu(pre, approximate="tanh").to(BF), da.to(BF), da.sum(0)


@torch.compile(fullgraph=True, dynamic=True)
def _bias_out(y, b2):
    return y.float() + b2


class MoEExperts(torch.autograd.Function):
    """All experts of a layer at once on rows sorted by expert (`counts` rows each): gelu(rows W1_e + b1_e) W2_e + b2_e
    with bf16 matmuls. The up-projection is one grouped Triton GEMM with bias + GELU in its epilogue
    (`gdn_kernels.moe_up_gelu`), and its backward one grouped GEMM with the GELU derivative, the recomputed GELU and the
    bias gradient in the epilogue (`moe_dz_gelu_bwd`); the down-projection and the weight gradients are cuBLAS calls
    per expert, written in place (autograd through weight slices adds a zero-filled full-size gradient per expert).
    Saves the rows, the pre-activations and the bf16 weights."""

    @staticmethod
    def forward(ctx: Any, xs: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor,
                counts: List[int]) -> torch.Tensor:
        from model.archs.gdn_kernels import moe_tiles, moe_up_gelu

        w1b, w2b = w1.to(BF), w2.to(BF)
        tiles = moe_tiles(counts, xs.device)
        pre, z = moe_up_gelu(xs.contiguous(), w1b, b1, tiles)
        out = torch.empty(xs.shape[0], w2.shape[2], device=xs.device, dtype=torch.float32)
        s = 0
        for e, n in enumerate(counts):
            if n:
                out[s:s + n] = _bias_out(z[s:s + n] @ w2b[e], b2[e])
            s += n
        ctx.save_for_backward(xs, pre, w1b, w2b, tiles)
        ctx.counts = counts
        return out

    @staticmethod
    def backward(ctx: Any, d_out: torch.Tensor):  # type: ignore[override]
        from model.archs.gdn_kernels import moe_dz_gelu_bwd

        xs, pre, w1b, w2b, tiles = ctx.saved_tensors
        d_outb = d_out.to(BF).contiguous()
        da, z, db1 = moe_dz_gelu_bwd(d_outb, w2b, pre, tiles)
        dxs = torch.empty_like(xs)
        dw1, dw2 = torch.zeros(w1b.shape, device=xs.device), torch.zeros(w2b.shape, device=xs.device)
        db2 = torch.zeros(w2b.shape[0], w2b.shape[2], device=xs.device)
        s = 0
        for e, n in enumerate(ctx.counts):
            if n:
                r = slice(s, s + n)
                dw2[e] = (z[r].t() @ d_outb[r]).float()
                db2[e] = d_out[r].sum(0)
                dw1[e] = (xs[r].t() @ da[r]).float()
                torch.mm(da[r], w1b[e].t(), out=dxs[r])
            s += n
        return dxs, dw1, db1, dw2, db2, None


class PermuteRows(torch.autograd.Function):
    """x[order] for a permutation `order`: the backward is the gather by the inverse permutation (index_select's
    backward is an atomic scatter-add)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, order: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inverse)
        return x.index_select(0, order)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor):  # type: ignore[override]
        (inverse,) = ctx.saved_tensors
        return g.index_select(0, inverse), None, None


def _const(v: Any):
    return lambda shape, dtype=None: keras.ops.convert_to_tensor(np.asarray(v, "float32"), dtype=dtype)


def causal_conv(x: torch.Tensor, w: torch.Tensor, reset: torch.Tensor, carry: Optional[torch.Tensor]):
    """Depthwise causal conv over time: out[t] = sum_i w[S-1-i] x[t-i] over the taps of the same document. x `(B, T,
    C)`, w `(S, C)`, reset `(B, T)`, carry `(B, S-1, C)`: the inputs before the row (zeros: none). Returns (out, the
    carry after the row: its last S-1 inputs, zeroed where they belong to an earlier document than the last token)."""
    B, T, C = x.shape
    S = w.shape[0]
    if carry is None:
        carry = x.new_zeros(B, S - 1, C)
    xx = torch.cat([carry.to(x.dtype), x], 1)
    seg = reset.long().cumsum(1)
    segx = torch.cat([seg.new_zeros(B, S - 1), seg], 1)
    out = 0
    for i in range(S):  # tap i reads position t - i
        a = S - 1 - i
        same = (segx[:, a:a + T] == seg).to(x.dtype)[..., None]
        out = out + w[S - 1 - i].to(x.dtype) * xx[:, a:a + T] * same
    tail = xx[:, -(S - 1):] * (segx[:, -(S - 1):] == seg[:, -1:]).to(x.dtype)[..., None]
    return out, tail


@keras.saving.register_keras_serializable(package="model")
class GatedDeltaNetLM(DenseSupervisionMixin, BaseLM):
    """Args:
        vocabulary_size: Vocabulary size.
        dim: Width of the residual stream and the token embedding.
        num_layers: Stacked layers (token mixer + MoE each).
        num_heads, head_dim: Delta-rule heads; each keeps a head_dim x head_dim state (keys and values head_dim).
        conv_size: Taps of the short causal convolution on q, k, v.
        num_experts: Expert FFNs per layer (top-1 routing).
        expert_hidden: Hidden width of each expert (GELU).
        window_size: Training row length (packed documents). The state has no window.
        reset_token: Document marker id (the state restarts there).
        embedding_std: Token embedding init std (the other matrices: 0.02, output projections 0.02 / sqrt(2 layers)).
        router_bias_rate: Step of the load-balancing router bias per training forward.
        recompute: Recompute each layer in the backward pass (less memory, ~1.3x time).
    """

    no_weight_decay_names = ("norm", "bias", "A_log", "conv")
    config_defaults = {
        "training": {"learning_rate": 0.001, "warmup_steps": 200, "dense_supervision": True, "dense_validation": True,
                     "weight_decay": 0.1, "global_clipnorm": 1.0},
        "data": {"batch_size": 32, "random_windows": True},
    }
    packed_documents = True
    stateful_generation = True

    def __init__(self, vocabulary_size: int = 1024, dim: int = 256, num_layers: int = 16, num_heads: int = 4,
                 head_dim: int = 64, conv_size: int = 4, num_experts: int = 6, expert_hidden: int = 1024,
                 window_size: int = 512, reset_token: int = 0, embedding_std: float = 0.02,
                 router_bias_rate: float = 0.001, recompute: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self._vocabulary_size = int(vocabulary_size)
        self.dim, self.num_layers = int(dim), int(num_layers)
        self.num_heads, self.head_dim, self.conv_size = int(num_heads), int(head_dim), int(conv_size)
        self.num_experts, self.expert_hidden = int(num_experts), int(expert_hidden)
        self.window_size, self.reset_token = int(window_size), int(reset_token)
        self.embedding_std, self.router_bias_rate = float(embedding_std), float(router_bias_rate)
        self.recompute = bool(recompute)
        self.token_embedding = layers.Embedding(
            self._vocabulary_size, self.dim, name="token_embedding",
            embeddings_initializer=keras.initializers.RandomNormal(stddev=self.embedding_std))
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
        D, H, Dh, E, Fh = self.dim, self.num_heads, self.head_dim, self.num_experts, self.expert_hidden
        HD = H * Dh
        self.token_embedding.build((None,))
        self.readout.build((None, D))
        self.final_norm = self.add_weight(name="final_norm", shape=(D,), initializer="ones")
        normal = keras.initializers.RandomNormal(stddev=0.02)
        out_init = keras.initializers.RandomNormal(stddev=0.02 / math.sqrt(2 * self.num_layers))
        rng = np.random.default_rng(0)
        self._params: List[Dict[str, Any]] = []
        for n in range(self.num_layers):
            w = lambda name, shape, init, trainable=True: self.add_weight(
                name=f"l{n}_{name}", shape=shape, initializer=init, trainable=trainable)
            A = rng.uniform(1, 16, H)  # Mamba2: alpha = exp(-A softplus(.)), dt in [1e-3, 0.1] at init
            dt = np.exp(rng.uniform(np.log(1e-3), np.log(0.1), H))
            self._params.append({
                "attn_norm": w("attn_norm", (D,), "ones"),
                "w_qkv": w("w_qkv", (D, 3 * HD), normal),
                "conv": w("conv", (self.conv_size, 3 * HD), keras.initializers.RandomUniform(
                    -1 / math.sqrt(self.conv_size), 1 / math.sqrt(self.conv_size))),
                "w_ab": w("w_ab", (D, 2 * H), normal),
                "dt_bias": w("dt_bias", (H,), _const(dt + np.log(-np.expm1(-dt)))),  # softplus^-1(dt)
                "A_log": w("A_log", (H,), _const(np.log(A))),
                "w_gate": w("w_gate", (D, HD), normal),
                "o_norm": w("o_norm", (Dh,), "ones"),
                "w_o": w("w_o", (HD, D), out_init),
                "ffn_norm": w("ffn_norm", (D,), "ones"),
                "router": w("router", (D, E), normal),
                "router_bias": w("router_bias", (E,), "zeros", trainable=False),
                "w1": w("w1", (E, D, Fh), normal),
                "b1": w("b1", (E, Fh), "zeros"),
                "w2": w("w2", (E, Fh, D), out_init),
                "b2": w("b2", (E, D), "zeros"),
            })
        self.built = True

    # -- one layer --------------------------------------------------------------------------------------------------

    def _moe(self, p: Dict[str, torch.Tensor], h: torch.Tensor, bias_var: Any) -> torch.Tensor:
        """Top-1 mixture of experts on rows h `(N, dim)` (fp32 in, fp32 out; expert matmuls under the caller's
        autocast). Tokens are sorted by expert, each expert runs one matmul pair on its rows."""
        if not h.shape[0]:
            return h
        with torch.autocast("cuda", enabled=False):  # the choice in fp32
            logits = h.float() @ p["router"].float()
        probs = torch.softmax(logits, -1)
        choice = torch.argmax(logits + p["router_bias"].float(), -1)
        gate = probs.gather(1, choice[:, None])
        counts_t = torch.bincount(choice, minlength=self.num_experts)
        self._balance(bias_var, counts_t)
        order = torch.argsort(choice)
        xs = h.index_select(0, order)
        outs, unused = [], 0.0
        for e, rows in enumerate(xs.split(counts_t.tolist(), 0)):
            if rows.shape[0]:
                z = F.gelu(rows @ p["w1"][e] + p["b1"][e], approximate="tanh")
                outs.append(z @ p["w2"][e] + p["b2"][e])
            else:  # no token chose it: in the graph with a zero contribution (its gradient is 0, not missing)
                unused = unused + sum(p[w][e].sum() for w in ("w1", "b1", "w2", "b2")) * 0.0
        y = torch.cat(outs, 0).float().index_select(0, torch.argsort(order))
        return gate * y + unused

    def _balance(self, bias_var: Any, counts: torch.Tensor) -> None:
        """Training: move the router bias towards balanced expert loads. The step is queued and applied at the start
        of the next forward pass (`_apply_balance`): a layer recomputed in the backward pass (`recompute`) must route
        exactly as in its forward pass, and it is queued once per forward pass."""
        if not (torch.is_grad_enabled() and self.router_bias_rate):
            return
        pending = self.__dict__.setdefault("_pending_bias", {})
        if bias_var.path in pending:
            return
        with torch.no_grad():
            c = counts.float()
            pending[bias_var.path] = (bias_var, self.router_bias_rate * torch.sign(c.mean() - c))

    def _apply_balance(self) -> None:
        pending = self.__dict__.pop("_pending_bias", {})
        with torch.no_grad():
            for var, step in pending.values():
                var.value.add_(step)

    @staticmethod
    def _fused(x: torch.Tensor, state: Any, need_state: bool) -> bool:
        """The fused CUDA path: training / evaluation rows (no carried state)."""
        return FUSED and x.is_cuda and state is None and not need_state

    def _moe_fused(self, p: Dict[str, torch.Tensor], x: torch.Tensor, bias_var: Any) -> torch.Tensor:
        """`_moe` with the compiled router and expert MLP (x: the stream rows, not normalized)."""
        if not x.shape[0]:
            return x
        h, choice, gate = _router(x, p["ffn_norm"], p["router"], p["router_bias"])
        counts_t = torch.bincount(choice, minlength=self.num_experts)
        self._balance(bias_var, counts_t)
        order = torch.argsort(choice)
        inverse = torch.argsort(order)
        xs = PermuteRows.apply(h, order, inverse)
        y = MoEExperts.apply(xs, p["w1"], p["b1"], p["w2"], p["b2"], counts_t.tolist())
        return gate * PermuteRows.apply(y, inverse, order)

    def _layer(self, n: int, x: torch.Tensor, reset: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]],
               rows: Optional[torch.Tensor] = None, need_state: bool = False):
        """Layer n on the stream x `(B, T, dim)`; state: (delta-rule state `(B, H, K, V)`, conv carry). `rows`: return
        the stream at those flat positions only `(R, dim)` (the MoE runs there only). Returns (stream, state)."""
        p = {k: v.value for k, v in self._params[n].items()}
        B, T, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        if self._fused(x, state, need_state):
            seg = reset.long().cumsum(1)
            if TRITON_CONV and conv_silu is not None:
                qkv, g, beta, gate = _mixer_proj(x, p["attn_norm"], p["w_qkv"], p["w_ab"], p["dt_bias"], p["A_log"],
                                                 p["w_gate"], H)
                q, k, v = conv_silu(qkv, p["conv"], seg).reshape(B, T, 3, H, Dh).unbind(2)
            else:
                q, k, v, g, beta, gate = _mixer_pre(x, seg, p["attn_norm"], p["w_qkv"], p["conv"], p["w_ab"],
                                                    p["dt_bias"], p["A_log"], p["w_gate"], H, Dh)
            o, _ = gated_delta_rule(q, k, v, g, beta, reset)
            x = _mixer_post(x, o, p["o_norm"], gate, p["w_o"])
            flat = x.reshape(-1, D) if rows is None else x.reshape(-1, D).index_select(0, rows)
            out = flat + self._moe_fused(p, flat, self._params[n]["router_bias"])
            return (out if rows is not None else out.reshape(B, T, D)), (None, None)
        S0, carry = (None, None) if state is None else state
        u = _rms(x, p["attn_norm"])
        qkv, carry = causal_conv(u @ p["w_qkv"], p["conv"], reset, carry)
        q, k, v = F.silu(qkv).reshape(B, T, 3, H, Dh).unbind(2)
        with torch.autocast("cuda", enabled=False):  # decay and write strength in fp32 (small matmul)
            a, b = (u.float() @ p["w_ab"].float()).split([H, H], -1)
        g = -torch.exp(p["A_log"].float()) * F.softplus(a + p["dt_bias"].float())
        beta = torch.sigmoid(b)
        o, S = gated_delta_rule(q, k, v, g, beta, reset, S0, need_state=need_state)
        o = _rms(o, p["o_norm"].float()).reshape(B, T, H * Dh) * F.silu((u @ p["w_gate"]).float())
        x = x + (o @ p["w_o"]).float()
        if rows is not None:
            x = x.reshape(-1, D).index_select(0, rows)
        flat = x.reshape(-1, D)
        h = _rms(flat, p["ffn_norm"])
        x = (flat + self._moe(p, h, self._params[n]["router_bias"])).reshape(x.shape)
        return x, (S, carry)

    def _run(self, ids: Any, initial: Optional[List[Any]] = None, valid: Optional[torch.Tensor] = None,
             need_state: bool = False):
        """`(B, T)` ids through every layer from `initial` (one (state, conv carry) per layer, None: zeros).
        `valid` `(B, T)` bool: the stream at those positions only `(R, dim)`, else `(B, T, dim)`."""
        E = self.token_embedding.embeddings.value
        self._apply_balance()  # the router bias steps of the previous training forward pass (see `_balance`)
        ids = torch.as_tensor(keras.ops.cast(ids, "int32")).to(E.device).long()
        B, T = ids.shape
        x = E.index_select(0, ids.reshape(-1)).reshape(B, T, self.dim).float()
        reset = ids == self.reset_token
        rows = None if valid is None else torch.as_tensor(valid, device=E.device).reshape(-1).nonzero().squeeze(1)
        ends = []
        cuda = x.device.type == "cuda"
        fused = self._fused(x, None if initial is None else initial[0], need_state)  # explicit bf16 matmuls inside
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda and not fused):
            for n in range(self.num_layers):
                last = rows if n == self.num_layers - 1 else None
                init = None if initial is None else initial[n]
                if self.recompute and torch.is_grad_enabled():
                    x, end = torch.utils.checkpoint.checkpoint(self._layer, n, x, reset, init, last, need_state,
                                                               use_reentrant=False)
                else:
                    x, end = self._layer(n, x, reset, init, last, need_state)
                ends.append(end)
        return x.float(), ends

    # -- the model API ----------------------------------------------------------------------------------------------

    def hidden(self, inputs: Any) -> torch.Tensor:
        B, T = inputs.shape[0], inputs.shape[1]
        last = torch.zeros(B, T, dtype=torch.bool)
        last[:, -1] = True
        return self._run(inputs, valid=last)[0]

    def dense_hidden(self, inputs: Any) -> torch.Tensor:
        return self._run(inputs)[0]

    def dense_hidden_valid(self, inputs: Any, valid: Any, carry: bool = False) -> torch.Tensor:
        if carry:
            raise ValueError("gated-deltanet does not support training.stateful")
        return self._run(inputs, valid=valid)[0]

    def readout_input(self, h: torch.Tensor) -> torch.Tensor:
        return _rms(h, self.final_norm.value)

    def call(self, inputs: Any, training: bool = False) -> Any:
        return self.readout(self.readout_input(self.hidden(inputs)))

    def advance(self, ids: Any, state: Optional[List[Any]] = None):
        """Feeds `(B, T)` ids after `state` (per layer: delta-rule state, conv carry): (logits of the last position,
        the states after it)."""
        B, T = ids.shape[0], ids.shape[1]
        last = torch.zeros(B, T, dtype=torch.bool)
        last[:, -1] = True
        out, ends = self._run(ids, state, valid=last, need_state=True)
        return self.readout(self.readout_input(out)), ends

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "vocabulary_size": self._vocabulary_size, "dim": self.dim,
                "num_layers": self.num_layers, "num_heads": self.num_heads, "head_dim": self.head_dim,
                "conv_size": self.conv_size, "num_experts": self.num_experts, "expert_hidden": self.expert_hidden,
                "window_size": self.window_size, "reset_token": self.reset_token, "embedding_std": self.embedding_std,
                "router_bias_rate": self.router_bias_rate, "recompute": self.recompute}
