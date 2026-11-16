"""Triton kernels for `gated-deltanet` on CUDA.

`conv_silu(x, w, seg)`: the short causal depthwise convolution over time followed by SiLU, never across a document
start, for x `(B, T, C)` bf16, w `(S, C)` fp32 taps, seg `(B, T)` int32 document numbers:

    pre[b, t, c] = sum_i w[S-1-i, c] x[b, t-i, c] [t-i >= 0, seg[b, t-i] == seg[b, t]],   y = silu(pre)  (bf16)

Forward: one pass reading x once (fp32 accumulation). Backward: kernel 1 recomputes pre, d_pre = dy silu'(pre)
(stored bf16) and the weight gradient (each tile sums its tokens, one atomic add per tap and channel); kernel 2 the
input gradient dx[t] = sum_i w[S-1-i] d_pre[t+i] [same document]. Replaces the compiled version, whose shifted fp32
copies, masks and token reductions were the largest elementwise kernels of the step.
"""

from typing import Any

import torch
import triton
import triton.language as tl

BT, BC = 64, 128  # tile: tokens x channels


@triton.jit
def _conv_fwd(X, W, SEG, Y, T, C, S: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    nt = tl.cdiv(T, BT)
    b, tb = tl.program_id(0) // nt, tl.program_id(0) % nt
    t = tb * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    tm, cm = t < T, c < C
    seg_t = tl.load(SEG + b * T + t, mask=tm, other=-1)
    acc = tl.zeros((BT, BC), dtype=tl.float32)
    for i in tl.static_range(S):
        tt = t - i
        ok = tm & (tt >= 0)
        seg_s = tl.load(SEG + b * T + tt, mask=ok, other=-2)
        m = ok & (seg_s == seg_t)
        x = tl.load(X + (b * T + tt)[:, None] * C + c[None, :], mask=m[:, None] & cm[None, :], other=0.0)
        w = tl.load(W + (S - 1 - i) * C + c, mask=cm, other=0.0)
        acc += x.to(tl.float32) * w[None, :]
    y = acc * tl.sigmoid(acc)
    tl.store(Y + (b * T + t)[:, None] * C + c[None, :], y.to(Y.dtype.element_ty), mask=tm[:, None] & cm[None, :])


@triton.jit
def _conv_bwd_pre(X, W, SEG, DY, DP, DW, T, C, S: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    """d_pre = dy silu'(pre) at the tile's tokens, and the tile's share of the weight gradient."""
    nt = tl.cdiv(T, BT)
    b, tb = tl.program_id(0) // nt, tl.program_id(0) % nt
    t = tb * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    tm, cm = t < T, c < C
    seg_t = tl.load(SEG + b * T + t, mask=tm, other=-1)
    acc = tl.zeros((BT, BC), dtype=tl.float32)
    for i in tl.static_range(S):
        tt = t - i
        ok = tm & (tt >= 0)
        seg_s = tl.load(SEG + b * T + tt, mask=ok, other=-2)
        m = ok & (seg_s == seg_t)
        x = tl.load(X + (b * T + tt)[:, None] * C + c[None, :], mask=m[:, None] & cm[None, :], other=0.0)
        w = tl.load(W + (S - 1 - i) * C + c, mask=cm, other=0.0)
        acc += x.to(tl.float32) * w[None, :]
    mask2 = tm[:, None] & cm[None, :]
    dy = tl.load(DY + (b * T + t)[:, None] * C + c[None, :], mask=mask2, other=0.0).to(tl.float32)
    sg = tl.sigmoid(acc)
    dp = dy * sg * (1 + acc * (1 - sg))
    tl.store(DP + (b * T + t)[:, None] * C + c[None, :], dp.to(DP.dtype.element_ty), mask=mask2)
    for i in tl.static_range(S):  # dW[S-1-i, c] += sum_t dp[t] x[t-i] [same document]
        tt = t - i
        ok = tm & (tt >= 0)
        seg_s = tl.load(SEG + b * T + tt, mask=ok, other=-2)
        m = ok & (seg_s == seg_t)
        x = tl.load(X + (b * T + tt)[:, None] * C + c[None, :], mask=m[:, None] & cm[None, :], other=0.0)
        tl.atomic_add(DW + (S - 1 - i) * C + c, tl.sum(dp * x.to(tl.float32), 0), mask=cm)


@triton.jit
def _conv_bwd_x(DP, W, SEG, DX, T, C, S: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    """dx[t] = sum_i w[S-1-i] d_pre[t+i] [t+i < T, same document]."""
    nt = tl.cdiv(T, BT)
    b, tb = tl.program_id(0) // nt, tl.program_id(0) % nt
    t = tb * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    tm, cm = t < T, c < C
    seg_t = tl.load(SEG + b * T + t, mask=tm, other=-1)
    acc = tl.zeros((BT, BC), dtype=tl.float32)
    for i in tl.static_range(S):
        tt = t + i
        ok = tm & (tt < T)
        seg_s = tl.load(SEG + b * T + tt, mask=ok, other=-2)
        m = ok & (seg_s == seg_t)
        dp = tl.load(DP + (b * T + tt)[:, None] * C + c[None, :], mask=m[:, None] & cm[None, :], other=0.0)
        w = tl.load(W + (S - 1 - i) * C + c, mask=cm, other=0.0)
        acc += dp.to(tl.float32) * w[None, :]
    tl.store(DX + (b * T + t)[:, None] * C + c[None, :], acc.to(DX.dtype.element_ty), mask=tm[:, None] & cm[None, :])


class ConvSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, w: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        x, seg = x.contiguous(), seg.to(torch.int32).contiguous()
        w32 = w.float().contiguous()
        B, T, C = x.shape
        y = torch.empty_like(x)
        grid = (B * triton.cdiv(T, BT), triton.cdiv(C, BC))
        _conv_fwd[grid](x, w32, seg, y, T, C, S=w.shape[0], BT=BT, BC=BC)
        ctx.save_for_backward(x, w32, seg)
        return y

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor):  # type: ignore[override]
        x, w32, seg = ctx.saved_tensors
        B, T, C = x.shape
        S = w32.shape[0]
        grid = (B * triton.cdiv(T, BT), triton.cdiv(C, BC))
        dp = torch.empty_like(x)
        dw = torch.zeros_like(w32)
        _conv_bwd_pre[grid](x, w32, seg, dy.contiguous(), dp, dw, T, C, S=S, BT=BT, BC=BC)
        dx = torch.empty_like(x)
        _conv_bwd_x[grid](dp, w32, seg, dx, T, C, S=S, BT=BT, BC=BC)
        return dx, dw, None


def conv_silu(x: torch.Tensor, w: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    return ConvSiLU.apply(x, w, seg)


# -- the experts' up-projection with GELU in the epilogue (grouped over experts: one launch) ---------------------------
# gelu_tanh(x) = x sigmoid(2u), u = sqrt(2/pi) (x + 0.044715 x^3); d/dx = s + x s (1 - s) 2 u',  s = sigmoid(2u)

MOE_BM = 64  # rows per tile (fixed: the tile list is built for it)
_C0, _C1 = tl.constexpr(0.7978845608028654), tl.constexpr(0.044715)


@triton.autotune(configs=[triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
                          for bn, bk, w, st in ((128, 64, 4, 3), (128, 32, 4, 4), (64, 64, 4, 3), (256, 64, 8, 3))],
                 key=["K", "F"])
@triton.jit
def _moe_up_gelu(X, W, B1, PRE, Z, TE, TR0, TR1, K, F, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """PRE = X[rows] W[e] + b1[e], Z = gelu(PRE) (bf16) for the tile's rows of expert e."""
    t = tl.program_id(0)
    e, r0, r1 = tl.load(TE + t), tl.load(TR0 + t), tl.load(TR1 + t)
    rows = r0 + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    rm, cm = rows < r1, cols < F
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        km = kk < K
        x = tl.load(X + rows[:, None] * K + kk[None, :], mask=rm[:, None] & km[None, :], other=0.0)
        w = tl.load(W + e * K * F + kk[:, None] * F + cols[None, :], mask=km[:, None] & cm[None, :], other=0.0)
        acc += tl.dot(x, w)
    pre = acc + tl.load(B1 + e * F + cols, mask=cm, other=0.0)[None, :]
    s = tl.sigmoid(2 * _C0 * (pre + _C1 * pre * pre * pre))
    m2 = rm[:, None] & cm[None, :]
    tl.store(PRE + rows[:, None] * F + cols[None, :], pre.to(tl.bfloat16), mask=m2)
    tl.store(Z + rows[:, None] * F + cols[None, :], (pre * s).to(tl.bfloat16), mask=m2)


@triton.autotune(configs=[triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
                          for bn, bk, w, st in ((128, 64, 4, 3), (128, 32, 4, 4), (64, 64, 4, 3), (256, 64, 8, 3))],
                 key=["D", "F"], reset_to_zero=["DB1"])  # the benchmark runs accumulate into DB1
@triton.jit
def _moe_dz_gelu_bwd(DY, W2, PRE, DA, Z, DB1, TE, TR0, TR1, D, F, BM: tl.constexpr, BN: tl.constexpr,
                     BK: tl.constexpr):
    """dz = dY[rows] W2[e]^T; epilogue: dA = dz gelu'(PRE), Z = gelu(PRE) (recomputed), db1[e] += column sums of dA."""
    t = tl.program_id(0)
    e, r0, r1 = tl.load(TE + t), tl.load(TR0 + t), tl.load(TR1 + t)
    rows = r0 + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    rm, cm = rows < r1, cols < F
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, D, BK):
        kk = k0 + tl.arange(0, BK)
        km = kk < D
        dy = tl.load(DY + rows[:, None] * D + kk[None, :], mask=rm[:, None] & km[None, :], other=0.0)
        w = tl.load(W2 + e * F * D + cols[None, :] * D + kk[:, None], mask=km[:, None] & cm[None, :], other=0.0)
        acc += tl.dot(dy, w)
    m2 = rm[:, None] & cm[None, :]
    pre = tl.load(PRE + rows[:, None] * F + cols[None, :], mask=m2, other=0.0).to(tl.float32)
    s = tl.sigmoid(2 * _C0 * (pre + _C1 * pre * pre * pre))
    dgelu = s + pre * s * (1 - s) * 2 * _C0 * (1 + 3 * _C1 * pre * pre)
    da = tl.where(m2, acc * dgelu, 0.0)
    tl.store(DA + rows[:, None] * F + cols[None, :], da.to(tl.bfloat16), mask=m2)
    tl.store(Z + rows[:, None] * F + cols[None, :], (pre * s).to(tl.bfloat16), mask=m2)
    tl.atomic_add(DB1 + e * F + cols, tl.sum(da, 0), mask=cm)


def moe_tiles(counts, device) -> torch.Tensor:
    """(3, tiles) int32: each tile's expert, first row and end row (rows sorted by expert, MOE_BM rows per tile)."""
    te, r0, r1, s = [], [], [], 0
    for e, n in enumerate(counts):
        for r in range(s, s + n, MOE_BM):
            te.append(e)
            r0.append(r)
            r1.append(s + n)
        s += n
    return torch.tensor([te, r0, r1], dtype=torch.int32).to(device, non_blocking=True)


def moe_up_gelu(x: torch.Tensor, w1b: torch.Tensor, b1: torch.Tensor, tiles: torch.Tensor):
    """(pre-activations, GELU outputs) `(N, F)` bf16 for rows x `(N, K)` bf16 sorted by expert."""
    N, K = x.shape
    F = w1b.shape[2]
    pre, z = x.new_empty(N, F), x.new_empty(N, F)
    if tiles.shape[1]:
        grid = lambda meta: (tiles.shape[1], triton.cdiv(F, meta["BN"]))
        _moe_up_gelu[grid](x, w1b, b1.float().contiguous(), pre, z, tiles[0], tiles[1], tiles[2], K, F, BM=MOE_BM)
    return pre, z


def moe_dz_gelu_bwd(dy: torch.Tensor, w2b: torch.Tensor, pre: torch.Tensor, tiles: torch.Tensor):
    """(dA, Z, db1) for dY `(N, D)` bf16 sorted by expert, W2 `(E, F, D)` bf16, the saved pre-activations."""
    N, D = dy.shape
    E, F = w2b.shape[0], w2b.shape[1]
    da, z = pre.new_empty(N, F), pre.new_empty(N, F)
    db1 = torch.zeros(E, F, device=dy.device, dtype=torch.float32)
    if tiles.shape[1]:
        grid = lambda meta: (tiles.shape[1], triton.cdiv(F, meta["BN"]))
        _moe_dz_gelu_bwd[grid](dy, w2b, pre, da, z, db1, tiles[0], tiles[1], tiles[2], D, F, BM=MOE_BM)
    return da, z, db1
