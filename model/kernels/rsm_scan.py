"""Custom XPU kernels of `rope-state-memory` (`rsm_scan.sycl`), built on first use, and their autograd.

`available(like)` says whether the kernels can run for a tensor; the arch falls back to its chunked torch path
otherwise (CPU, CUDA, a failed build, or `RSM_KERNELS=0`).

Gradients. Both scans are linear in their inputs, so every input gradient is the same kind of scan:

    state    s_t = gamma sum_{j <= t} lambda^(t - j) R_{t - j} x_j
             dx_j = gamma sum_{t >= j} lambda^(t - j) R_{-(t - j)} ds_t           reverse rot_scan
    memory   o_t = sum_{j <= t} sum_a q_ta mu_a^(t - j) k_ja v_j       (a decay per key row a)
             dq_ta = sum_{j <= t} mu_a^(t - j) k_ja (do_t . v_j)                   outer_scan(v, k, do), column decay
             dk_ja = sum_{t >= j} mu_a^(t - j) q_ta (do_t . v_j)                   reverse outer_scan(do, q, v), columns
             dv_j  = sum_{t >= j} sum_a mu_a^(t - j) q_ta k_ja do_t                reverse outer_scan(q, do, k), rows

(sums within a document). The decays need no extra pass. With w_tj the (t, j) term of sum_t ds_t . s_t,
d/d ln(rate) of sum w_tj rate^(t - j) is sum (t - j) w_tj = sum_t t (sum_j w_tj) - sum_j j (sum_t w_tj), and
sum_j w_tj = ds_t . s_t, sum_t w_tj = x_j . dx_j: two dot products per position. The LRU factor adds
d/d rate of gamma (or c). These hold for a zero initial state, which is what training uses (the arch only takes
this path with zero carries when gradients are on).
"""

import os
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_ext: Any = None
_error: Optional[str] = None


def _icpx() -> Optional[str]:
    """icpx whose SYCL runtime ABI matches torch's (`libsycl.so.8`): oneAPI 2025.x."""
    found = shutil.which("icpx")
    if found:
        return found
    for root in sorted(Path("/opt/intel/oneapi/compiler").glob("2025.*"), reverse=True):
        if (root / "bin" / "icpx").exists():
            return str(root / "bin" / "icpx")
    return None


def load() -> Any:
    """The compiled extension (cached in `.kernel_build/`), or None if it cannot be built here."""
    global _ext, _error
    if _ext is not None or _error is not None:
        return _ext
    try:
        if os.environ.get("RSM_KERNELS", "1") == "0":
            raise RuntimeError("disabled by RSM_KERNELS=0")
        if not torch.xpu.is_available():
            raise RuntimeError("no XPU")
        icpx = _icpx()
        if icpx is None:
            raise RuntimeError("icpx (oneAPI 2025.x DPC++) not found")
        os.environ["PATH"] = str(Path(icpx).parent) + os.pathsep + os.environ.get("PATH", "")
        if "TORCH_XPU_ARCH_LIST" not in os.environ and "Iris(R) Xe" in torch.xpu.get_device_name(0):
            os.environ["TORCH_XPU_ARCH_LIST"] = "tgllp"  # ahead-of-time code for Tiger Lake Xe-LP
        from torch.utils.cpp_extension import load as _load

        build = _ROOT / ".kernel_build" / "rsm_scan"
        build.mkdir(parents=True, exist_ok=True)
        _ext = _load(name="rsm_scan", sources=[str(_HERE / "rsm_scan.sycl")], build_directory=str(build),
                     extra_sycl_cflags=["-O3", "-ffast-math"], verbose=False)
    except Exception as e:  # noqa: BLE001  any failure means: use the torch path
        _error = f"{type(e).__name__}: {e}"
    return _ext


def error() -> Optional[str]:
    return _error


def available(like: torch.Tensor) -> bool:
    return like.device.type == "xpu" and load() is not None


def _f(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().float()


_EMPTY: dict = {}
# row widths `outer_scan` takes (its register tiling); `outer_scan_cols` takes rows 16, 32, 64, 128 and any even width
OUTER_ROWS = (16, 32, 48, 64, 80, 128, 144)
ROWS = (16, 32, 64, 128)


def _empty(like: torch.Tensor) -> torch.Tensor:
    key = like.device
    if key not in _EMPTY:
        _EMPTY[key] = torch.empty(0, device=like.device, dtype=torch.float32)
    return _EMPTY[key]


def _positions(n: int, like: torch.Tensor) -> torch.Tensor:
    return torch.arange(n, device=like.device, dtype=like.dtype)[None, :]


# Time-segmented memory scans. One packed row of N tokens is one scan of N dependent steps, and at a batch of one or
# two rows the scan kernels occupy a few hundred threads each waiting on its own step chain. The memory is linear
# in its state, so a row is cut into P segments scanned as independent rows from zero (B P rows of N / P steps),
# then each segment gets the state carried in from the ones before it (after it, for the reversed scans):
#     carry_p = Zend_{p-1} + live_{p-1} mu^L carry_{p-1}     (live: no document reset inside the segment)
#     out_i  += scale [ (gamma_i (.) mu^(i+1) (.) live_i) @ carry_p ]   (positions i of segment p; the reset drops it)
# a batched matmul per segment. The result is the single scan's (the same sums, in another order).
SEGMENT_ROWS = int(os.environ.get("RSM_SEGMENT_ROWS", "8"))  # split until the batch has this many rows
SEGMENT_MIN = 256  # but no segment shorter than this


def _parts(B: int, N: int) -> int:
    P = 1
    while B * P < SEGMENT_ROWS and N % (2 * P) == 0 and N // (2 * P) >= SEGMENT_MIN:
        P *= 2
    return P


def _segmented(kind: str, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, reset: torch.Tensor,
               dec: torch.Tensor, scale: torch.Tensor, reverse: bool, dec_rows: bool, want_end: bool):
    """`kind` "rows": `outer_scan` (out (B, N, C) = scale Z^T gamma); "cols": `outer_scan_cols` (out (B, N, A) =
    scale Z gamma, a decay per row). From a zero state. Returns (out, the memory after the scan or empty)."""
    B, N, A = alpha.shape
    C = beta.shape[2]
    e = _empty(alpha)
    P = _parts(B, N)
    if P == 1:
        if kind == "rows":
            return _ext.outer_scan(alpha, beta, gamma, reset, dec, scale, e, reverse, want_end, dec_rows)
        out, z = _ext.outer_scan_cols(alpha, beta, gamma, reset, dec, scale, reverse, want_end)
        return out, z
    L = N // P
    seg = lambda x: x.reshape(B * P, L, x.shape[-1])
    if kind == "rows":
        out, zl = _ext.outer_scan(seg(alpha), seg(beta), seg(gamma), reset.reshape(B * P, L), dec, scale, e, reverse,
                                  True, dec_rows)
    else:
        out, zl = _ext.outer_scan_cols(seg(alpha), seg(beta), seg(gamma), reset.reshape(B * P, L), dec, scale,
                                       reverse, True)
    zl = zl.view(B, P, A, C)
    r = reset.to(alpha.dtype)
    if reverse:  # step t of a reversed scan keeps its state unless position t + 1 starts a document
        keep = 1 - torch.cat([r[:, 1:], r.new_zeros(B, 1)], 1)
        live = torch.cumprod(keep.view(B, P, L).flip(-1), -1).flip(-1)  # no reset from i to the segment's end
        live_end = live[..., 0]
    else:
        live = torch.cumprod((1 - r).view(B, P, L), -1)  # no reset from the segment's start to i
        live_end = live[..., -1]
    rows_decay = kind == "cols" or dec_rows
    pw = torch.exp(torch.arange(1, L + 1, device=alpha.device, dtype=alpha.dtype)[:, None] * torch.log(dec)[None])
    posf = pw.flip(0) if reverse else pw  # mu^(L - i) reversed, mu^(i + 1) forward: (L, decay width)
    decL = pw[-1]
    dk = (lambda w: decL[None, :, None] * w[:, None, None]) if rows_decay else (lambda w: decL[None, None, :] * w[:, None, None])
    carries = [None] * P
    order = range(P - 2, -1, -1) if reverse else range(1, P)
    first = P - 1 if reverse else 0
    carries[first] = zl.new_zeros(B, A, C)
    for p in order:
        q = p + 1 if reverse else p - 1  # the neighbour the state comes from
        carries[p] = zl[:, q] + dk(live_end[:, q]) * carries[q]
    carry = torch.stack(carries, 1).view(B * P, A, C)
    lv = live.reshape(B * P, L, 1)
    g = seg(gamma)
    if kind == "rows" and dec_rows:  # the factors on the narrow side (A), added in place by the matmul
        out.baddbmm_(g * (posf[None] * lv * scale), carry)
    elif kind == "rows":
        out.add_(torch.bmm(g, carry).mul_(lv * posf[None] * scale))
    else:
        out.add_(torch.bmm(g, carry.transpose(1, 2)).mul_(lv * posf[None] * scale))
    end = e
    if want_end:
        last = 0 if reverse else P - 1
        end = zl[:, last] + dk(live_end[:, last]) * carries[last]
    return out.view(B, N, out.shape[-1]), end


class StateScan(torch.autograd.Function):
    """s_t = lambda R_1 s_{t-1} keep_t + gamma x_t, lambda = e^{-alpha}, gamma = sqrt(1 - lambda^2)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: torch.Tensor, reset: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor, s0: Optional[torch.Tensor]):  # type: ignore[override]
        lam = torch.exp(-alpha)
        gam = torch.sqrt(-torch.expm1(-2 * alpha))
        one = torch.ones_like(lam)
        scal = torch.stack([lam, gam, one]).float().contiguous()
        x = _f(x)
        s, s_end = _ext.rot_scan(x, reset, cos, sin, scal, _empty(x) if s0 is None else _f(s0), False, 1.0)
        ctx.save_for_backward(x, s, alpha, reset, cos, sin)
        return s, s_end

    @staticmethod
    def backward(ctx: Any, ds: torch.Tensor, ds_end: torch.Tensor):  # type: ignore[override]
        x, s, alpha, reset, cos, sin = ctx.saved_tensors
        lam = torch.exp(-alpha)
        gam = torch.sqrt(-torch.expm1(-2 * alpha))
        ds = _f(ds)
        scal = torch.stack([lam, torch.ones_like(lam), gam]).float().contiguous()
        dx, _ = _ext.rot_scan(ds, reset, cos, sin, scal, _empty(ds), True, -1.0)
        pos = _positions(x.shape[1], x)
        w_out = (ds * s).sum(-1)  # (B, N): sum_j w_tj
        w_in = (x * dx).sum(-1)  # sum_t w_tj
        dlog = (pos * w_out).sum() - (pos * w_in).sum()  # d / d ln(lambda), lambda^(t - j) only
        # d ln(lambda) / d alpha = -1; gamma: d gamma / d alpha = lambda^2 / gamma, and dL/d gamma = sum w_out / gamma
        dalpha = -dlog + (lam * lam / (gam * gam)) * w_out.sum()
        return dx, dalpha.reshape(alpha.shape), None, None, None, None


class MemoryScan(torch.autograd.Function):
    """o_t = M_t^T q_t, M_t[a] = mu_a M_{t-1}[a] keep_t + k_t[a] v_t: a decay per key row a (one value repeated for
    the linear memory, one per bank for the bank memory).

    Backward: three scans of the same kernel, with the decay along the matching axis, and per-row decay gradients
    d/d ln mu_a = sum_t t q_ta dq_ta - sum_j j k_ja dk_ja (the closed form of the module docstring, per row;
    summed over a it is sum_t t do_t . o_t - sum_j j v_j . dv_j)."""

    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mu: torch.Tensor,
                reset: torch.Tensor, m0: Optional[torch.Tensor], want_end: bool):  # type: ignore[override]
        q, k, v = _f(q), _f(k), _f(v)
        mu_c = mu.contiguous().float()
        one = torch.ones(1, device=q.device)
        if m0 is None:  # from zero: time-segmented
            o, m_end = _segmented("rows", k, v, q, reset, mu_c, one, False, True, want_end)
        else:
            o, m_end = _ext.outer_scan(k, v, q, reset, mu_c, one, _f(m0), False, want_end, True)
        ctx.save_for_backward(q, k, v, mu_c, reset)
        return o, m_end

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor, dm_end: torch.Tensor):  # type: ignore[override]
        q, k, v, mu, reset = ctx.saved_tensors
        do = _f(do)
        e = _empty(do)
        one = torch.ones(1, device=do.device)
        if v.shape[-1] in OUTER_ROWS:  # the value width on the rows of the scan (a decay per column)
            dq, _ = _segmented("rows", v, k, do, reset, mu, one, False, False, False)
            dk, _ = _segmented("rows", do, q, v, reset, mu, one, True, False, False)
        else:  # wide values: the key rows stay on the rows, the read-out sums the columns
            dq, _ = _segmented("cols", k, v, do, reset, mu, one, False, True, False)  # dq_t = M_t do_t, M the forward memory
            dk, _ = _segmented("cols", q, do, v, reset, mu, one, True, True, False)  # dk_j = G_j v_j, G = sum mu^(t-j) q_t do_t^T
        dv, _ = _segmented("rows", q, do, k, reset, mu, one, True, True, False)
        pos = _positions(q.shape[1], q)[..., None]  # (1, N, 1)
        dlog = (pos * q * dq).sum((0, 1)) - (pos * k * dk).sum((0, 1))  # (A,)
        return dq, dk, dv, dlog / mu, None, None, None


def state_scan(x: torch.Tensor, alpha: torch.Tensor, reset: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               s0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """States `(B, N, D)` and the last state `(B, D)`. `reset` `(B, N)` uint8, cos / sin `(D/2,)` of theta."""
    return StateScan.apply(x, alpha, reset, cos, sin, s0)


def memory_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mu: torch.Tensor, reset: torch.Tensor,
                m0: Optional[torch.Tensor] = None, want_end: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Memory reads o `(B, N, dim)` and the last memory `(B, A, dim)` (empty unless `want_end`); `mu` `(A,)` the
    decay of each key row."""
    return MemoryScan.apply(q, k, v, mu, reset, m0, want_end)


class FusedNormMLP(torch.autograd.Function):
    """y = MLP(RMSNorm(x)) = gelu_tanh(z W1 + b1) W2 + b2, z = x inv w; with `l2_out`, y = normalize(that).

    Every pass over the activations is one fused kernel or one oneDNN `mm`: the row norms (`row_norm`) keep a
    single float per row for the backward pass, and the bias + GELU (`bias_gelu_`, `gelu_backward_`) run in place
    on the matmul outputs. The `hidden`-wide layer is recomputed in the backward pass, never stored; z is kept
    (one dim-wide write, cheaper than recomputing it). Rows go in blocks of `rows` to bound the temporaries."""

    rows = 16384

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scale: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor,
                b2: torch.Tensor, l2_out: bool, keep_hidden: bool = False, half: bool = False):  # type: ignore[override]
        mm = _matmul(half)
        ctx.half = half
        x = _f(x)
        e = _empty(x)
        z, inv = _ext.row_norm(0, x, scale.contiguous(), e, e, 1e-6)
        b1c = b1.contiguous()
        out = x.new_empty(x.shape[0], w2.shape[1])
        # keep_hidden: the pre-activations z W1 stay for the backward pass (rows x hidden floats), which then skips
        # recomputing them (one rows x dim x hidden matmul)
        kept = x.new_empty(x.shape[0], w1.shape[1]) if keep_hidden else None
        for a in range(0, x.shape[0], FusedNormMLP.rows):
            if keep_hidden:
                pre = mm(z[a:a + FusedNormMLP.rows], w1, out=kept[a:a + FusedNormMLP.rows])
                hid = _ext.bias_gelu(pre, b1c)
            else:
                hid = mm(z[a:a + FusedNormMLP.rows], w1)
                _ext.bias_gelu_(hid, b1c)
            mm(hid, w2, out=out[a:a + FusedNormMLP.rows])
        ctx.kept = kept
        if l2_out:
            out, n = _ext.row_norm(2, out, b2.contiguous(), e, e, 1e-6)
            ctx.save_for_backward(x, scale, inv, z, w1, b1, w2, b2, out, n)
        else:
            out += b2
            ctx.save_for_backward(x, scale, inv, z, w1, b1, w2, b2)
        ctx.l2_out = l2_out
        return out

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor):  # type: ignore[override]
        if ctx.l2_out:
            x, scale, inv, z, w1, b1, w2, b2, y, n = ctx.saved_tensors
            do, db2 = _ext.row_norm(3, y, b2.contiguous(), _f(dy), n, 1e-6)
        else:
            x, scale, inv, z, w1, b1, w2, b2 = ctx.saved_tensors
            do = _f(dy)
            db2 = do.sum(0)
        b1c = b1.contiguous()
        mm = _matmul(ctx.half)
        dz = torch.empty_like(z)
        dw1t = dw2 = db1 = None
        w2t = w2.T
        kept, ctx.kept = ctx.kept, None
        for a in range(0, z.shape[0], FusedNormMLP.rows):
            zb, dob = z[a:a + FusedNormMLP.rows], do[a:a + FusedNormMLP.rows]
            pre = kept[a:a + FusedNormMLP.rows] if kept is not None else mm(zb, w1)  # becomes h below, in place
            dh = mm(dob, w2t, scale_a=True)
            gb1 = _ext.gelu_backward_(pre, dh, b1c)  # pre <- h, dh <- dpre
            g2 = mm(pre.T, dob, scale_b=True)
            g1t = mm(dh.T, zb, scale_a=True)  # dW1^T (H, D): the faster layout
            mm(dh, w1.T, scale_a=True, out=dz[a:a + FusedNormMLP.rows])
            dw1t, dw2, db1 = (g1t, g2, gb1) if dw2 is None else (dw1t + g1t, dw2 + g2, db1 + gb1)
        dx, dscale = _ext.row_norm(1, x, scale.contiguous(), dz, inv, 1e-6)
        return dx, dscale, dw1t.T, db1, dw2, db2, None, None, None


def _matmul(half: bool):
    """`mm(a, b, scale_a=False, scale_b=False, out=None)`: the assembly fp16 kernel (`hgemm.mm`: a gradient operand
    is scaled into fp16's range) or torch's fp32 `mm` (the scale flags ignored)."""
    if half:
        from model.kernels import hgemm

        if hgemm.load() is None:
            raise RuntimeError(f"mlp_fp16: the fp16 matmul is unavailable: {hgemm.error()}")
        return hgemm.mm

    def mm(a: torch.Tensor, b: torch.Tensor, scale_a: bool = False, scale_b: bool = False,
           out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.mm(a, b, out=out) if out is not None else a @ b

    return mm


def norm_mlp_ok(like: torch.Tensor, dim_in: int, hidden: int, dim_out: int, l2_out: bool) -> bool:
    return dim_in % 64 == 0 and hidden % 4 == 0 and (not l2_out or dim_out % 64 == 0) and available(like)


def norm_mlp(x: torch.Tensor, scale: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor,
             b2: torch.Tensor, l2_out: bool = False, keep_hidden: bool = False, half: bool = False) -> torch.Tensor:
    """`(R, dim_in)` rows -> `(R, dim_out)`: MLP(RMSNorm(x)), L2-normalized if `l2_out`; `keep_hidden` stores the
    pre-activations for the backward pass instead of recomputing them; `half`: the matmuls in fp16 (`hgemm`)."""
    return FusedNormMLP.apply(x, scale, w1, b1, w2, b2, l2_out, keep_hidden, half)


class RowRMS(torch.autograd.Function):
    """y = x inv w, inv = (mean x^2 + 1e-6)^(-1/2), per row: one fused pass forward, one backward (one float per
    row kept)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scale: torch.Tensor):  # type: ignore[override]
        x = _f(x)
        e = _empty(x)
        y, inv = _ext.row_norm(0, x, scale.contiguous(), e, e, 1e-6)
        ctx.save_for_backward(x, scale, inv)
        return y

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor):  # type: ignore[override]
        x, scale, inv = ctx.saved_tensors
        dx, dscale = _ext.row_norm(1, x, scale.contiguous(), _f(dy), inv, 1e-6)
        return dx, dscale


def rms_ok(like: torch.Tensor, dim: int) -> bool:
    return dim % 64 == 0 and available(like)


def rms(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """RMSNorm over the last axis of `x` `(..., dim)`."""
    return RowRMS.apply(x.reshape(-1, x.shape[-1]), scale).reshape(x.shape)


class JoinRead(torch.autograd.Function):
    """h = a s + g o, o = m[:, :D] / max(m[:, D], 1e-12): the bank memory's read (values' sum m[:, :D], mass m[:, D])
    joined to the state, one fused pass each way. Rows (R, D) / (R, M); g, a (R, 1); a None means 1."""

    @staticmethod
    def forward(ctx: Any, s: torch.Tensor, m: torch.Tensor, g: torch.Tensor, a: Optional[torch.Tensor]):  # type: ignore[override]
        s, m, gg = _f(s), _f(m), _f(g).reshape(-1)
        aa = _f(a).reshape(-1) if a is not None else _empty(s)
        ctx.save_for_backward(s, m, gg, aa)
        ctx.has_a = a is not None
        return _ext.join_read(s, m, gg, aa)

    @staticmethod
    def backward(ctx: Any, dh: torch.Tensor):  # type: ignore[override]
        s, m, g, a = ctx.saved_tensors
        dh = _f(dh)
        ds, dm, dg, da = _ext.join_read_backward(s, m, g, a, dh)
        return (ds if ctx.has_a else dh), dm, dg[:, None], (da[:, None] if ctx.has_a else None)


def join_ok(like: torch.Tensor, dim: int, cols: int) -> bool:
    return dim % 4 == 0 and cols % 4 == 0 and available(like)


def join_read(s: torch.Tensor, m: torch.Tensor, g: torch.Tensor, a: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`(..., D)` state, `(..., M)` memory read (values then mass), `(..., 1)` gates -> h `(..., D)`."""
    R, D = s.numel() // s.shape[-1], s.shape[-1]
    h = JoinRead.apply(s.reshape(R, D), m.reshape(R, m.shape[-1]), g.reshape(R, 1),
                       None if a is None else a.reshape(R, 1))
    return h.reshape(s.shape)
