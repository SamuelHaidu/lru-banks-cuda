"""`lru-banks`: LRU + slot-memory ("banks") layers with two reads per layer (`model/archs/bank_memory.py`).

Factorized tied embedding: ids -> E (vocab x embed_dim) -> W_in -> the residual stream h (dim). Layer n:

    x      = rms_norm(h) * norm1
    LRU    h_t = lambda_t h_{t-1} + gamma_t B x_t     lru_dim complex channels, s_t = [Re h_t, Im h_t] (2 lru_dim)
           layers >= gate_from: input gate g_t = sigmoid(x_t w + b): lambda_t = lambda^g_t, gamma_t =
           sqrt(1 - |lambda_t|^2) (g = 0 skips the token); below: g = 1 (the plain LRU)
    banks  heads x slots, keys / values head_dim; key k_t = normalize(W_k s_{t-1} + b_k) (the context before the
           token), query q_t = normalize(W_q s_t), value W_v x_t; written top-2 into the slots chosen by a router on
           the key (renormalized), each slot an average ("mass" normalization) with a fixed decay (half-lives
           log-spaced over `slot_half_life`); read softmax(tau (Kb q) / z) over the slots, then the average values
    hop 1  h += W_o read(q);   h += FFN(rms_norm(h))
    hop 2  q2 = normalize(W_q2 rms_norm(h));  h += W_o2 read(q2) (the same banks: reads never write);
           h += FFN(rms_norm(h))   (the same FFN: one per layer; GELU, ffn_mult x dim)

logits = (rms_norm(h) * final_norm) W_out E^T (the embedding table, transposed). Every state restarts at
`reset_token` (<bos>, right after each <eos>). Packed rows, dense supervision, stateful generation (`advance`).

CUDA training path (`LRU_BANKS_FUSED`, default on): bf16 matmuls on the tensor cores, the LRU and the bank
recurrences in fused Triton kernels (`model/archs/bank_kernels.py`: the LRU scan with the gate inside; the bank
read on fla's gated-slot-attention kernels plus the mass normalization). Elsewhere the torch chunked forms of
`bank_memory.py` (the reference the kernels are tested against).
"""

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model.archs._dense import DenseSupervisionMixin  # noqa: E402
from model.archs.bank_memory import (EPS, bank_read, bank_write_chunk, lru_chunk, route)  # noqa: E402
from model.archs.rope_state_memory import _TiedReadout  # noqa: E402
from model.interfaces import BaseLM  # noqa: E402

FUSED = os.environ.get("LRU_BANKS_FUSED", "1") != "0"
try:  # Triton + fla: CUDA only
    from model.archs import bank_kernels  # noqa: E402
except Exception:  # pragma: no cover - no Triton / fla here
    bank_kernels = None


def _rms(x: torch.Tensor, w: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w.float()


def _const(v: Any):
    return lambda shape, dtype=None: np.asarray(v, dtype="float32").reshape(shape)


def lru_states(proj: torch.Tensor, nu: torch.Tensor, theta: torch.Tensor, gate: Optional[torch.Tensor],
               reset: torch.Tensor, h0: Optional[torch.Tensor] = None, chunk: int = 16):
    """The LRU (torch): proj `(B, T, 2N)` (B x, real then imaginary parts), nu / theta `(N,)`, gate `(B, T, 1)` or None.
    Returns s `(B, T, 2N)` ([Re h, Im h]) and the last state h `(B, N)` complex."""
    dt = torch.float64 if proj.dtype == torch.float64 else torch.float32  # float64 in tests
    rate = nu.to(dt).exp()  # |lambda| = exp(-rate)
    if gate is not None:
        rate = gate.to(dt) * rate
    gamma = torch.sqrt((-torch.expm1(-2 * rate)).clamp_min(1e-12))
    re, im = proj.to(dt).chunk(2, -1)
    b = gamma * torch.complex(re, im)
    log_lam = torch.complex(-nu.to(dt).exp(), theta.to(dt))
    if gate is not None:
        log_lam = gate.to(dt) * log_lam
    h, hT = lru_chunk(b, log_lam, reset, h0, chunk=chunk)
    return torch.cat([h.real, h.imag], -1), hT


# -- the fused CUDA path: everything between the recurrence kernels compiled (inductor fuses the norms, casts, routing
# and residual adds into few kernels and the matmul epilogues), the matmuls in bf16 on the tensor cores -----------------

BF = torch.bfloat16


def _top2(lg: torch.Tensor) -> torch.Tensor:
    """The top-2 of softmax(lg) renormalized == a softmax over the 2 largest logits: (sigmoid(l1 - l2), 1 - that) at
    their slots, 0 elsewhere. Two max passes and one select: no softmax over every slot, no topk kernel, no scatter."""
    M = lg.shape[-1]
    ar = torch.arange(M, device=lg.device)
    l1, i1 = lg.max(-1, keepdim=True)
    l2, i2 = torch.where(ar == i1, torch.full_like(lg, -1e30), lg).max(-1, keepdim=True)
    p1 = torch.sigmoid(l1 - l2)
    return torch.where(ar == i1, p1, torch.where(ar == i2, 1 - p1, torch.zeros_like(lg)))


@torch.compile(dynamic=False)
def _f_pre(h, norm1, w_lru):
    x = _rms(h, norm1).to(BF)
    return x, x @ w_lru.to(BF)


@torch.compile(dynamic=False)
def _f_pre_gated(h, norm1, w_lru, w_gate, gate_bias):
    x = _rms(h, norm1).to(BF)
    gate = torch.sigmoid((x @ w_gate.to(BF)).float() + gate_bias.float())
    return x, x @ w_lru.to(BF), gate


@torch.compile(dynamic=False)
def _f_addr(s, reset, x, w_k, k_bias, w_q, w_v, router, H: int, D: int):
    """Keys (state before the token), queries (after), values, top-2 routing: bf16 `(B, T, H, .)` contiguous."""
    B, T, _ = s.shape
    sb = s.to(BF)
    s_prev = torch.cat([torch.zeros_like(sb[:, :1]), sb[:, :-1]], 1) * (~reset)[..., None].to(BF)
    k = F.normalize((s_prev @ w_k.to(BF)).float().view(B, T, H, D) + k_bias.float().view(H, D), dim=-1)
    q = F.normalize((sb @ w_q.to(BF)).float().view(B, T, H, D), dim=-1)
    v = (x @ w_v.to(BF)).view(B, T, H, D)
    lg = torch.einsum("bthd,hdm->bthm", k, router.float())
    return k.to(BF), q.to(BF), v.contiguous(), _top2(lg).to(BF)


@torch.compile(dynamic=False)
def _f_post(h, o, w_o, ffn_norm, w1, b1, w2, b2):
    """h += W_o o; h += FFN(rms(h)) (GELU), fp32 stream, bf16 matmuls."""
    B, T, _ = h.shape
    h = h + (o.reshape(B, T, -1) @ w_o.to(BF)).float()
    z = F.gelu((_rms(h, ffn_norm).to(BF) @ w1.to(BF)).float() + b1.float(), approximate="tanh").to(BF)
    return h + (z @ w2.to(BF)).float() + b2.float()


@torch.compile(dynamic=False)
def _f_q2(h, q2_norm, w_q2, H: int, D: int):
    B, T, _ = h.shape
    return F.normalize((_rms(h, q2_norm).to(BF) @ w_q2.to(BF)).float().view(B, T, H, D), dim=-1).to(BF)


@keras.saving.register_keras_serializable(package="model")
class LRUBanksLM(DenseSupervisionMixin, BaseLM):
    """Args:
        vocabulary_size: Vocabulary size.
        embed_dim: Width of the token embedding table (tied with the readout).
        dim: Width of the residual stream.
        num_layers: Stacked layers.
        lru_dim: Complex LRU channels per layer (its state s has 2 lru_dim reals).
        num_heads, head_dim: Bank heads; keys and values head_dim each.
        slots: Bank slots per head and layer.
        topk: Slots written per token and head.
        hops: Reads per layer (1 or 2).
        ffn_mult: FFN hidden width / dim (GELU).
        gate_from: First layer with the LRU input gate (layers below it: no gate).
        lru_half_life, slot_half_life: (min, max) half-lives in tokens at init, log-spaced.
        tau: Initial read temperature per head.
        gate_bias: Initial input-gate bias (sigmoid 0.95 at 3: nearly the plain LRU).
        window_size: Training row length (packed documents). The states have no window.
        reset_token: Document marker id (every state restarts there).
        chunk: Chunk length of the torch recurrences.
    """

    no_weight_decay_names = ("norm", "bias", "nu", "theta", "tau", "loga")
    config_defaults = {
        "training": {"learning_rate": 0.002, "warmup_steps": 500, "dense_supervision": True,
                     "dense_validation": True, "weight_decay": 0.1, "global_clipnorm": 1.0},
        "data": {"batch_size": 64, "random_windows": True},
    }
    packed_documents = True
    stateful_generation = True

    def __init__(self, vocabulary_size: int = 32768, embed_dim: int = 128, dim: int = 512, num_layers: int = 4,
                 lru_dim: int = 256, num_heads: int = 16, head_dim: int = 32, slots: int = 128, topk: int = 2,
                 hops: int = 2, ffn_mult: int = 4, gate_from: int = 1,
                 lru_half_life: Sequence[float] = (1.0, 16.0), slot_half_life: Sequence[float] = (1.0, 1024.0),
                 tau: float = 8.0, gate_bias: float = 3.0, window_size: int = 512, reset_token: int = 0,
                 chunk: int = 16, **kwargs: Any):
        super().__init__(**kwargs)
        self._vocabulary_size = int(vocabulary_size)
        self.embed_dim, self.dim, self.num_layers = int(embed_dim), int(dim), int(num_layers)
        self.lru_dim, self.num_heads, self.head_dim = int(lru_dim), int(num_heads), int(head_dim)
        self.slots, self.topk, self.hops, self.ffn_mult = int(slots), int(topk), int(hops), int(ffn_mult)
        assert self.hops in (1, 2), "hops: 1 or 2"
        self.gate_from = int(gate_from)
        self.lru_half_life = [float(v) for v in lru_half_life]
        self.slot_half_life = [float(v) for v in slot_half_life]
        self.tau, self.gate_bias = float(tau), float(gate_bias)
        self.window_size, self.reset_token, self.chunk = int(window_size), int(reset_token), int(chunk)
        self.token_embedding = layers.Embedding(
            self._vocabulary_size, self.embed_dim, name="token_embedding",
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.readout = _TiedReadout(self.token_embedding, name="readout")
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
        D, E0, N, H, Dh, M = self.dim, self.embed_dim, self.lru_dim, self.num_heads, self.head_dim, self.slots
        HD, Fh = H * Dh, self.ffn_mult * self.dim
        self.token_embedding.build((None,))
        normal = keras.initializers.RandomNormal(stddev=0.02)
        out_init = keras.initializers.RandomNormal(stddev=0.02 / math.sqrt(2 * self.num_layers * self.hops))
        self.w_in = self.add_weight(name="w_in", shape=(E0, D), initializer=normal)
        self.final_norm = self.add_weight(name="final_norm", shape=(D,), initializer="ones")
        self.w_out = self.add_weight(name="w_out", shape=(D, E0), initializer=normal)
        hl = np.exp(np.linspace(np.log(self.lru_half_life[0]), np.log(self.lru_half_life[1]), N))
        shl = np.exp(np.linspace(np.log(self.slot_half_life[0]), np.log(self.slot_half_life[1]), M))
        rng = np.random.default_rng(0)
        self._params: List[Dict[str, Any]] = []
        for n in range(self.num_layers):
            w = lambda name, shape, init, trainable=True: self.add_weight(
                name=f"l{n}_{name}", shape=shape, initializer=init, trainable=trainable)
            p = {
                "norm1": w("norm1", (D,), "ones"),
                "w_lru": w("w_lru", (D, 2 * N), normal),
                "nu": w("nu", (N,), _const(np.log(math.log(2) / hl))),
                "theta": w("theta", (N,), _const(rng.uniform(0, 2 * math.pi, N))),
                "w_k": w("w_k", (2 * N, HD), normal),
                "k_bias": w("k_bias", (HD,), "zeros"),
                "w_q": w("w_q", (2 * N, HD), normal),
                "w_v": w("w_v", (D, HD), normal),
                "router": w("router", (H, Dh, M), keras.initializers.RandomNormal(stddev=1 / math.sqrt(Dh))),
                "loga": w("loga", (H, M), _const(np.tile(math.log(0.5) / shl, (H, 1))), trainable=False),
                "log_tau": w("log_tau", (H,), _const(np.full(H, math.log(self.tau)))),
                "w_o": w("w_o", (HD, D), out_init),
                "ffn_norm1": w("ffn_norm1", (D,), "ones"),
                "w1": w("w1", (D, Fh), normal),
                "b1_bias": w("b1_bias", (Fh,), "zeros"),
                "w2": w("w2", (Fh, D), out_init),
                "b2_bias": w("b2_bias", (D,), "zeros"),
            }
            if n >= self.gate_from:
                p["w_gate"] = w("w_gate", (D, 1), "zeros")
                p["gate_bias"] = w("gate_bias", (1,), _const([self.gate_bias]))
            if self.hops > 1:
                p |= {"q2_norm": w("q2_norm", (D,), "ones"), "w_q2": w("w_q2", (D, HD), normal),
                      "log_tau2": w("log_tau2", (H,), _const(np.full(H, math.log(self.tau)))),
                      "w_o2": w("w_o2", (HD, D), out_init), "ffn_norm2": w("ffn_norm2", (D,), "ones")}
            self._params.append(p)
        self.built = True

    # -- one layer --------------------------------------------------------------------------------------------------

    def _ffn(self, p: Dict[str, torch.Tensor], h: torch.Tensor, norm: str) -> torch.Tensor:
        z = F.gelu(_rms(h, p[norm]) @ p["w1"] + p["b1_bias"], approximate="tanh")
        return (z @ p["w2"] + p["b2_bias"]).float()

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(t.float().reshape(*t.shape[:-1], self.num_heads, self.head_dim), dim=-1)

    def _layer_fused(self, n: int, h: torch.Tensor, reset: torch.Tensor, bctx: Any) -> torch.Tensor:
        """Layer n on CUDA: compiled pieces around the LRU scan and the bank kernels (same math as `_layer`)."""
        p = {k: v.value for k, v in self._params[n].items()}
        H, D = self.num_heads, self.head_dim
        if "w_gate" in p:
            x, proj, gate = _f_pre_gated(h, p["norm1"], p["w_lru"], p["w_gate"], p["gate_bias"])
        else:
            (x, proj), gate = _f_pre(h, p["norm1"], p["w_lru"]), None
        s = bank_kernels.lru_scan(proj, p["nu"], p["theta"], gate, reset)
        k, q, v, r = _f_addr(s, reset, x, p["w_k"], p["k_bias"], p["w_q"], p["w_v"], p["router"], H, D)
        mem = bank_kernels.BankWrite(k, v, r, p["loga"].float(), reset, bctx)
        h = _f_post(h, mem.read(q, p["log_tau"].float().exp()), p["w_o"], p["ffn_norm1"], p["w1"], p["b1_bias"],
                    p["w2"], p["b2_bias"])
        if self.hops > 1:
            q2 = _f_q2(h, p["q2_norm"], p["w_q2"], H, D)
            h = _f_post(h, mem.read(q2, p["log_tau2"].float().exp()), p["w_o2"], p["ffn_norm2"], p["w1"],
                        p["b1_bias"], p["w2"], p["b2_bias"])
        return h

    def _layer(self, n: int, h: torch.Tensor, reset: torch.Tensor, state: Optional[Tuple[Any, Any]],
               fused: bool, bctx: Any = None):
        """Layer n on the stream h `(B, T, dim)` fp32; state: (LRU h `(B, N)` complex, banks (Kb, Vb, z)) or None.
        Returns (stream, state after the last position)."""
        p = {k: v.value for k, v in self._params[n].items()}
        B, T, _ = h.shape
        x = _rms(h, p["norm1"])
        proj = x @ p["w_lru"]
        gate = torch.sigmoid((x @ p["w_gate"]).float() + p["gate_bias"].float()) if "w_gate" in p else None
        h0, banks0 = (None, None) if state is None else state
        if fused:
            s = bank_kernels.lru_scan(proj, p["nu"], p["theta"], gate, reset)
            hT = None
        else:
            s, hT = lru_states(proj, p["nu"], p["theta"], gate, reset, h0, self.chunk)
        s0 = s.new_zeros(B, 1, s.shape[-1]) if h0 is None else torch.cat([h0.real, h0.imag], -1)[:, None].to(s.dtype)
        s_prev = torch.cat([s0, s[:, :-1]], 1) * (~reset)[..., None].to(s.dtype)
        k = self._heads(s_prev @ p["w_k"] + p["k_bias"])
        q = self._heads(s @ p["w_q"])
        v = (x @ p["w_v"]).float().reshape(B, T, self.num_heads, self.head_dim)
        r = route(k, p["router"].float(), self.topk)
        loga = p["loga"].float()
        if fused:
            mem = bank_kernels.BankWrite(k, v, r, loga, reset, bctx)
            read = mem.read
            banks = None
        else:
            mem, banks = bank_write_chunk(k, v, r, loga, reset, banks0, self.chunk, "mass")
            read = lambda q_, tau: bank_read(mem, q_, tau)
        o = read(q, p["log_tau"].float().exp())
        h = h + (o.reshape(B, T, -1) @ p["w_o"]).float()
        h = h + self._ffn(p, h, "ffn_norm1")
        if self.hops > 1:
            q2 = self._heads(_rms(h, p["q2_norm"]) @ p["w_q2"])
            o2 = read(q2, p["log_tau2"].float().exp())
            h = h + (o2.reshape(B, T, -1) @ p["w_o2"]).float()
            h = h + self._ffn(p, h, "ffn_norm2")
        return h, (hT, banks)

    def _run(self, ids: Any, initial: Optional[List[Any]] = None, valid: Optional[torch.Tensor] = None,
             need_state: bool = False):
        """`(B, T)` ids through every layer from `initial` (one state per layer, None: zeros). `valid` `(B, T)` bool:
        the stream at those positions only `(R, dim)`, else `(B, T, dim)`."""
        E = self.token_embedding.embeddings.value
        ids = torch.as_tensor(keras.ops.cast(ids, "int32")).to(E.device).long()
        B, T = ids.shape
        reset = ids == self.reset_token
        cuda = E.device.type == "cuda"
        fused = (FUSED and cuda and bank_kernels is not None and initial is None and not need_state
                 and bank_kernels.supported(self.head_dim, self.head_dim, self.slots))
        ends = []
        # the slot decays are fixed and identical in every layer: one set of cumulative decays per forward pass
        bctx = bank_kernels.BankContext(reset, self._params[0]["loga"].value.float()) if fused else None
        # bf16 matmuls only with the fused kernels: the torch chunked forms multiply factors up to e^30 and their
        # inverses (exact in fp32, not in bf16), so on CUDA generation runs them in fp32
        h = (E.index_select(0, ids.reshape(-1)).reshape(B, T, self.embed_dim) @ self.w_in.value).float()
        if fused and self.topk == 2:  # compiled pieces with explicit bf16 matmuls
            for n in range(self.num_layers):
                h = self._layer_fused(n, h, reset, bctx)
                ends.append((None, None))
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=fused):
                for n in range(self.num_layers):
                    h, end = self._layer(n, h, reset, None if initial is None else initial[n], fused, bctx)
                    ends.append(end)
        if valid is not None:
            rows = torch.as_tensor(valid, device=E.device).reshape(-1).nonzero().squeeze(1)
            h = h.reshape(-1, self.dim).index_select(0, rows)
        return h, ends

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
            raise ValueError("lru-banks does not support training.stateful")
        return self._run(inputs, valid=valid)[0]

    def readout_input(self, h: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.as_tensor(h).is_cuda):
            return (_rms(h, self.final_norm.value) @ self.w_out.value).float()

    def call(self, inputs: Any, training: bool = False) -> Any:
        return self.readout(self.readout_input(self.hidden(inputs)))

    def advance(self, ids: Any, state: Optional[List[Any]] = None):
        """Feeds `(B, T)` ids after `state` (per layer: LRU state, banks): (logits of the last position, the states
        after it)."""
        B, T = ids.shape[0], ids.shape[1]
        last = torch.zeros(B, T, dtype=torch.bool)
        last[:, -1] = True
        out, ends = self._run(ids, state, valid=last, need_state=True)
        return self.readout(self.readout_input(out)), ends

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "vocabulary_size": self._vocabulary_size, "embed_dim": self.embed_dim,
                "dim": self.dim, "num_layers": self.num_layers, "lru_dim": self.lru_dim, "num_heads": self.num_heads,
                "head_dim": self.head_dim, "slots": self.slots, "topk": self.topk, "hops": self.hops,
                "ffn_mult": self.ffn_mult, "gate_from": self.gate_from, "lru_half_life": self.lru_half_life,
                "slot_half_life": self.slot_half_life, "tau": self.tau, "gate_bias": self.gate_bias,
                "window_size": self.window_size, "reset_token": self.reset_token, "chunk": self.chunk}
