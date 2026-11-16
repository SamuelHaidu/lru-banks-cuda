"""CUDA kernels for `lru-banks` (`model/archs/lru_banks.py`): the same math as the torch reference in
`model/archs/bank_memory.py`, fused so each recurrence reads its inputs once and writes its outputs once.

- `lru_scan`: the complex diagonal LRU, h_t = lambda_t h_{t-1} + gamma_t (x_r + i x_i), with the input gate g_t,
  lambda_t = exp(g_t (-e^nu + i theta)), gamma_t = sqrt(1 - |lambda_t|^2), computed inside the kernel (no per-token
  lambda tensor in memory); one Triton program per (row, block of channels) walks the row; the backward pass walks it
  in reverse from the saved states and accumulates d nu, d theta (per channel) and d g (per position) with atomics.
- `MassScan`: the written mass z_t = a z_{t-1} + r_t per (head, slot), fp32, exact (the read divides by it).
- `BankWrite(...).read(q, tau)`: the bank read with fla's gated-slot-attention kernels (fla 0.6, `fla.ops.gsa`):
  scores over the slots ok = Kb q with `chunk_gsa_fwd_k` (keys k, slot weights r, per-slot log decays), the mass
  normalization and temperature in one compiled elementwise pass, p = softmax(tau ok / z) over the non-empty slots,
  w = p / z, then the values o = sum_m w_m Vb_m with `chunk_gla_fwd` (keys r, values v, the same decays). Backward:
  `chunk_gla_bwd`, the normalization's backward (compiled), `chunk_gsa_bwd_k`. Rows are flattened into one sequence
  cut at row and document starts (`cu_seqlens`), as `delta_rule_fla` does. bf16 operands on the tensor cores, decays
  and the normalization in fp32. The per-slot decays are the same in every layer, so their chunk-local cumulative sums
  are computed once per forward (`BankContext`).
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from fla.ops.gla.chunk import chunk_gla_bwd, chunk_gla_fwd
from fla.ops.gsa.chunk_bwd import chunk_gsa_bwd_k
from fla.ops.gsa.chunk_fwd import chunk_gsa_fwd_k
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2

EPS = 1e-6
CHUNK = 64


def supported(key_dim: int, value_dim: int, slots: int) -> bool:
    """fla's chunk kernels use tensor-core dots: key / value sizes and slots in multiples of 16."""
    return key_dim % 16 == 0 and value_dim % 16 == 0 and slots % 16 == 0 and min(key_dim, value_dim, slots) >= 16


# -- the LRU ------------------------------------------------------------------------------------------------------------

@triton.jit
def _gamma2(rate):
    """1 - exp(-2 rate), accurate for small rates."""
    return tl.where(rate < 1e-3, 2 * rate - 2 * rate * rate, 1 - tl.exp(-2 * rate))


@triton.jit
def _lru_fwd_kernel(P, NU, TH, G, R, S, T, N: tl.constexpr, BN: tl.constexpr, GATED: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = n < N
    er = tl.exp(tl.load(NU + n, mask=mask, other=0.0).to(tl.float32))
    th = tl.load(TH + n, mask=mask, other=0.0).to(tl.float32)
    hr = tl.zeros([BN], dtype=tl.float32)
    hi = tl.zeros([BN], dtype=tl.float32)
    for t in range(T):
        off = (b * T + t) * 2 * N
        xr = tl.load(P + off + n, mask=mask, other=0.0).to(tl.float32)
        xi = tl.load(P + off + N + n, mask=mask, other=0.0).to(tl.float32)
        keep = 1.0 - tl.load(R + b * T + t).to(tl.float32)
        if GATED:
            g = tl.load(G + b * T + t).to(tl.float32)
        else:
            g = 1.0
        rate = g * er
        mag = tl.exp(-rate)
        ang = g * th
        lr = mag * tl.cos(ang)
        li = mag * tl.sin(ang)
        gam = tl.sqrt(tl.maximum(_gamma2(rate), 1e-12))
        nr = keep * (lr * hr - li * hi) + gam * xr
        hi = keep * (lr * hi + li * hr) + gam * xi
        hr = nr
        tl.store(S + off + n, hr, mask=mask)
        tl.store(S + off + N + n, hi, mask=mask)


@triton.jit
def _lru_bwd_kernel(P, NU, TH, G, R, S, DS, DP, DG, DNU, DTH, T, N: tl.constexpr, BN: tl.constexpr,
                    GATED: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = n < N
    er = tl.exp(tl.load(NU + n, mask=mask, other=0.0).to(tl.float32))
    th = tl.load(TH + n, mask=mask, other=0.0).to(tl.float32)
    cr = tl.zeros([BN], dtype=tl.float32)  # d loss / d h_t carried from t + 1 (conj(lambda_{t+1}) G_{t+1})
    ci = tl.zeros([BN], dtype=tl.float32)
    dnu = tl.zeros([BN], dtype=tl.float32)
    dth = tl.zeros([BN], dtype=tl.float32)
    for tt in range(T):
        t = T - 1 - tt
        off = (b * T + t) * 2 * N
        gr = tl.load(DS + off + n, mask=mask, other=0.0).to(tl.float32) + cr
        gi = tl.load(DS + off + N + n, mask=mask, other=0.0).to(tl.float32) + ci
        xr = tl.load(P + off + n, mask=mask, other=0.0).to(tl.float32)
        xi = tl.load(P + off + N + n, mask=mask, other=0.0).to(tl.float32)
        keep = 1.0 - tl.load(R + b * T + t).to(tl.float32)
        if GATED:
            g = tl.load(G + b * T + t).to(tl.float32)
        else:
            g = 1.0
        rate = g * er
        mag = tl.exp(-rate)
        ang = g * th
        ca = tl.cos(ang)
        sa = tl.sin(ang)
        lr = mag * ca
        li = mag * sa
        gam = tl.sqrt(tl.maximum(_gamma2(rate), 1e-12))
        tl.store(DP + off + n, gam * gr, mask=mask)
        tl.store(DP + off + N + n, gam * gi, mask=mask)
        dgam = gr * xr + gi * xi
        prev = mask & (t > 0)
        hpr = keep * tl.load(S + off - 2 * N + n, mask=prev, other=0.0)
        hpi = keep * tl.load(S + off - N + n, mask=prev, other=0.0)
        dlr = gr * hpr + gi * hpi
        dli = gi * hpr - gr * hpi
        dmag = dlr * ca + dli * sa
        dang = mag * (dli * ca - dlr * sa)
        drate = -mag * dmag + dgam * tl.exp(-2 * rate) / gam
        dnu += drate * rate
        dth += dang * g
        if GATED:
            tl.atomic_add(DG + b * T + t, tl.sum(tl.where(mask, drate * er + dang * th, 0.0), 0))
        cr = keep * (lr * gr + li * gi)
        ci = keep * (lr * gi - li * gr)
    tl.atomic_add(DNU + n, dnu, mask=mask)
    tl.atomic_add(DTH + n, dth, mask=mask)


def _lru_launch(N: int):
    BN = 32 if N >= 128 else 16
    return BN, triton.cdiv(N, BN)


class LRUScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proj, nu, theta, gate, reset):
        B, T, N2 = proj.shape
        N = N2 // 2
        proj = proj.contiguous()
        rst = reset.to(torch.uint8).contiguous()
        g = gate.reshape(B, T).float().contiguous() if gate is not None else proj.new_empty(1, dtype=torch.float32)
        s = torch.empty(B, T, N2, device=proj.device, dtype=torch.float32)
        BN, nb = _lru_launch(N)
        _lru_fwd_kernel[(B, nb)](proj, nu.float().contiguous(), theta.float().contiguous(), g, rst, s, T, N=N, BN=BN,
                                 GATED=gate is not None)
        ctx.save_for_backward(proj, nu, theta, g, rst, s)
        ctx.gated = gate is not None
        return s

    @staticmethod
    def backward(ctx, ds):
        proj, nu, theta, g, rst, s = ctx.saved_tensors
        B, T, N2 = proj.shape
        N = N2 // 2
        dp = torch.empty(B, T, N2, device=proj.device, dtype=torch.float32)
        dg = torch.zeros(B, T, device=proj.device, dtype=torch.float32) if ctx.gated else g
        dnu = torch.zeros(N, device=proj.device, dtype=torch.float32)
        dth = torch.zeros(N, device=proj.device, dtype=torch.float32)
        BN, nb = _lru_launch(N)
        _lru_bwd_kernel[(B, nb)](proj, nu.float().contiguous(), theta.float().contiguous(), g, rst, s,
                                 ds.float().contiguous(), dp, dg, dnu, dth, T, N=N, BN=BN, GATED=ctx.gated)
        return (dp.to(proj.dtype), dnu.to(nu.dtype), dth.to(theta.dtype),
                dg.reshape(B, T, 1) if ctx.gated else None, None)


def lru_scan(proj: torch.Tensor, nu: torch.Tensor, theta: torch.Tensor, gate: Optional[torch.Tensor],
             reset: torch.Tensor) -> torch.Tensor:
    """proj `(B, T, 2N)` (real parts, then imaginary), nu / theta `(N,)`, gate `(B, T, 1)` or None, reset `(B, T)`:
    the states s `(B, T, 2N)` = [Re h, Im h], fp32. Same as `lru_banks.lru_states`."""
    return LRUScan.apply(proj, nu, theta, gate, reset)


# -- the written mass --------------------------------------------------------------------------------------------------

@triton.jit
def _mass_fwd_kernel(X, A, R, Y, T, C: tl.constexpr, BC: tl.constexpr):
    b = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = c < C
    a = tl.load(A + c, mask=mask, other=0.0)
    y = tl.zeros([BC], dtype=tl.float32)
    for t in range(T):
        keep = 1.0 - tl.load(R + b * T + t).to(tl.float32)
        y = keep * a * y + tl.load(X + (b * T + t) * C + c, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + (b * T + t) * C + c, y, mask=mask)


@triton.jit
def _mass_bwd_kernel(DY, A, R, DX, T, C: tl.constexpr, BC: tl.constexpr):
    b = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = c < C
    a = tl.load(A + c, mask=mask, other=0.0)
    carry = tl.zeros([BC], dtype=tl.float32)
    for tt in range(T):
        t = T - 1 - tt
        gr = tl.load(DY + (b * T + t) * C + c, mask=mask, other=0.0) + carry
        tl.store(DX + (b * T + t) * C + c, gr, mask=mask)
        keep = 1.0 - tl.load(R + b * T + t).to(tl.float32)
        carry = keep * a * gr


class MassScan(torch.autograd.Function):
    """z_t = a z_{t-1} + r_t (a per channel, restart at resets): r `(B, T, H, M)` -> z fp32."""

    @staticmethod
    def forward(ctx, r, a, reset):
        B, T, H, M = r.shape
        C = H * M
        x = r.reshape(B, T, C).contiguous()
        rst = reset.to(torch.uint8).contiguous()
        av = a.reshape(C).float().contiguous()
        z = torch.empty(B, T, C, device=r.device, dtype=torch.float32)
        BC = 128
        _mass_fwd_kernel[(B, triton.cdiv(C, BC))](x, av, rst, z, T, C=C, BC=BC)
        ctx.save_for_backward(av, rst)
        ctx.shape, ctx.dtype = r.shape, r.dtype
        return z.reshape(B, T, H, M)

    @staticmethod
    def backward(ctx, dz):
        av, rst = ctx.saved_tensors
        B, T, H, M = ctx.shape
        C = H * M
        dx = torch.empty(B, T, C, device=dz.device, dtype=torch.float32)
        BC = 128
        _mass_bwd_kernel[(B, triton.cdiv(C, BC))](dz.reshape(B, T, C).float().contiguous(), av, rst, dx, T, C=C, BC=BC)
        return dx.reshape(B, T, H, M).to(ctx.dtype), None, None


# -- the read ----------------------------------------------------------------------------------------------------------

@torch.compile(dynamic=False)
def _normalize(ok, z, tau):
    """p = softmax(tau ok / z) over the slots with z > EPS, w = p / z (0 elsewhere). ok `(1, N, H, M)`, z fp32,
    tau `(H,)`."""
    valid = z > EPS
    zc = z.clamp_min(EPS)
    sc = torch.where(valid, tau[:, None] * ok.float() / zc, torch.full_like(zc, -1e30))
    p = torch.softmax(sc, -1)
    return torch.where(valid, p / zc, torch.zeros_like(p))


@torch.compile(dynamic=False)
def _normalize_bwd(dw, ok, z, tau):
    """The gradients of `_normalize` w.r.t. ok, z and tau (p recomputed: nothing extra saved)."""
    valid = z > EPS
    zc = z.clamp_min(EPS)
    okf = ok.float()
    sc = torch.where(valid, tau[:, None] * okf / zc, torch.full_like(zc, -1e30))
    p = torch.softmax(sc, -1)
    dwf = dw.float()
    dp = torch.where(valid, dwf / zc, torch.zeros_like(zc))
    dsc = p * (dp - (p * dp).sum(-1, keepdim=True))
    dsc = torch.where(valid, dsc, torch.zeros_like(dsc))
    dok = dsc * tau[:, None] / zc
    dz = torch.where(valid, -(dwf * p + dsc * tau[:, None] * okf) / (zc * zc), torch.zeros_like(zc))
    dtau = (dsc * okf / zc).sum((0, 1, 3))
    return dok, dz, dtau


class BankContext:
    """What every layer's banks share in one forward pass: the document cuts and the chunk-local cumulative log
    decays (the slot decays are fixed and the same in every layer)."""

    def __init__(self, reset: torch.Tensor, loga: torch.Tensor):
        B, T = reset.shape
        H, M = loga.shape
        starts = reset.clone()
        starts[:, 0] = True
        cu = torch.nonzero(starts.reshape(-1)).squeeze(1)
        self.cu = torch.cat([cu, cu.new_tensor([B * T])])
        self.ci = prepare_chunk_indices(self.cu, CHUNK)
        g = loga.float()[None, None].expand(1, B * T, H, M).contiguous()
        self.g = chunk_local_cumsum(g, chunk_size=CHUNK, scale=RCP_LN2, cu_seqlens=self.cu, chunk_indices=self.ci)
        self.a = loga.float().exp()
        self.reset, self.B, self.T = reset, B, T


class BankRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, r, z, tau, bctx):
        kw = dict(cu_seqlens=bctx.cu, chunk_size=CHUNK, chunk_indices=bctx.ci)
        hk, _, ok = chunk_gsa_fwd_k(q=q, k=k, v=r, g=bctx.g, h0=None, output_final_state=False, scale=1.0, **kw)
        w = _normalize(ok, z, tau).to(q.dtype)
        _, Av, hv, _, o = chunk_gla_fwd(q=w, k=r, v=v, g=None, g_cumsum=bctx.g, scale=1.0, initial_state=None,
                                        output_final_state=False, **kw)
        ctx.save_for_backward(q, k, v, r, z, tau, ok, w, Av, hk, hv)  # w kept: no recompute in the backward
        ctx.bctx = bctx
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, r, z, tau, ok, w, Av, hk, hv = ctx.saved_tensors
        bctx = ctx.bctx
        kw = dict(cu_seqlens=bctx.cu, chunk_size=CHUNK, chunk_indices=bctx.ci)
        dw, dr1, dv, dg, _ = chunk_gla_bwd(q=w, k=r, v=v, g=None, g_cumsum=bctx.g, scale=1.0, initial_state=None,
                                           h=hv, A=Av, do=do.contiguous(), dht=None, **kw)
        dok, dz, dtau = _normalize_bwd(dw, ok, z, tau)
        dq, dk, dr2, _, _ = chunk_gsa_bwd_k(q=q, k=k, v=r, g=bctx.g, h0=None, h=hk, o=ok, do=dok.to(ok.dtype),
                                            dht=None, dg=dg, scale=1.0, **kw)
        return dq, dk, dv, (dr1.float() + dr2.float()).to(r.dtype), dz, dtau, None


class BankWrite:
    """The banks of one layer (CUDA): k, v `(B, T, H, D)`, r `(B, T, H, M)` (routing weights), loga `(H, M)`,
    reset `(B, T)`; `read(q, tau)` -> `(B, T, H, E)`, any number of times (hops share everything here)."""

    def __init__(self, k, v, r, loga, reset, bctx: Optional[BankContext] = None, dtype=torch.bfloat16):
        self.bctx = bctx if bctx is not None else BankContext(reset, loga)
        B, T, H, _ = k.shape
        self.shape = (B, T, H, v.shape[-1])
        self.z = MassScan.apply(r, self.bctx.a, reset).reshape(1, B * T, H, -1)
        flat = lambda x: x.reshape(1, B * T, *x.shape[2:]).to(dtype).contiguous()  # fp32 in tests
        self.k, self.v, self.r = flat(k), flat(v), flat(r)
        self.flat = flat

    def read(self, q: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        o = BankRead.apply(self.flat(q), self.k, self.v, self.r, self.z, tau.float(), self.bctx)
        return o.reshape(self.shape)
