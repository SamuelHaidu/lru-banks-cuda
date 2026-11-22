r"""A slot memory with content-addressed reads ("banks"), fed by an LRU: plain torch, a token loop (the definition)
and a chunked parallel form (what a GPU kernel runs), proven equal below and checked by `tests/test_bank_memory.py`.

The layer, per position t (one document; every state restarts at <bos>):

    LRU      h_t = lambda * h_{t-1} + gamma * (B_r u_t + i B_i u_t)      complex diagonal, lambda = exp(-e^nu + i theta)
             s_t = [Re h_t, Im h_t]                                         gamma = sqrt(1 - |lambda|^2)
    key      k_t = normalize(W_k s_{t-1} + b_k)      the state BEFORE the token: the context it followed
    query    q_t = normalize(W_q s_t)                the state AFTER the token: the context now
    value    v_t = W_v u_t                           the token itself
    route    r_t = topK(softmax(k_t R_h))            per head: which of the M slots this key's pattern is filed in
                                                     (the K largest probabilities renormalized, the rest 0)
    store    Kb[j] <- a_j Kb[j] + r_tj k_t           bank of keys        a_j = exp(loga_j): a fixed decay per slot
             Vb[j] <- a_j Vb[j] + r_tj v_t           bank of values
             z[j]  <- a_j z[j]  + r_tj               written mass
    read     score_j = tau * (Kb[j] . q_t) / z[j]    the query against each slot's AVERAGE key (slots with z <= eps
             p_t = softmax_j(score)                  are left out)
             o_t = sum_j p_tj Vb[j] / z[j]           the matching slots' AVERAGE values
    out      y_t = W_o o_t

The keys are unit vectors, so an average key is shorter when its slot mixes different patterns: a blurred slot
scores lower than a clean one with the same direction.

Write normalization (`norm`): every write r_tj is multiplied by a constant per slot c_j:
    "mass" (default)  c_j = 1, and the read divides by z: each slot is the AVERAGE of what it holds, however old
                      (a per-slot c_j would cancel here: it scales the bank and z alike);
    "ema"             c_j = 1 - a_j, no division: each slot is an exponential moving average, bounded by |k|, |v|;
                      a slot written once long ago fades (its key shrinks), so freshness enters the score; fast slots
                      take big writes, slow slots small ones;
    "lru"             c_j = sqrt(1 - a_j^2) (the LRU's normalization: keeps the slot's variance ~1 for uncorrelated
                      writes; for repeated similar keys a slow slot grows to ~sqrt((1 + a) / (1 - a))), no division;
    "none"            c_j = 1, no division: raw decayed sums (slow slots grow with every write).
In every mode z (the mass, with the same c_j) only decides which slots are empty (z <= eps: left out of the read).
c_j is a constant: the coefficients stay known in advance and the proof below is unchanged.

Why it is parallel (the proof). Write X_t for any of the three banks (Kb, Vb with a vector per slot, z with a scalar)
and x_t for what token t writes into it (k_t, v_t or 1). With rho_t = 1 at a document start (else 0):

    X_t[j] = (1 - rho_t) a_j X_{t-1}[j] + r_tj x_t

(1) Every coefficient is known before the recurrence runs: a_j is a parameter, r_tj depends only on k_t (the LRU
    output at t - 1, itself computed first), rho_t on the tokens. Nothing in the update reads the bank: the read
    (score, softmax, o) produces the output and never feeds back. So the update is a LINEAR recurrence in X with
    precomputed coefficients, and unrolls to

        X_t[j] = sum_{i <= t, same document} a_j^(t - i) r_ij x_i                                              (A)

(2) For a chunk of C positions starting after state X_0 (the state after the previous chunk), split (A) at the chunk
    start:  X_t[j] = gamma_tj X_0[j] + sum_{i in chunk, i <= t, same document} a_j^(t - i) r_ij x_i,
    gamma_tj = a_j^(tau + 1) if no document starts in the chunk up to t (tau = t's index in the chunk), else 0. The
    sum is a matrix product inside the chunk ((C x C) decay masks, the same for every t), and only X_0 crosses chunk
    boundaries: X_end[j] = gamma_{C-1,j} X_0[j] + sum_i a_j^(C-1-i) r_ij x_i [same document as the last position].
    Hops: a second read at t with a query made from the first read at t (through an MLP, position by position)
    reads the same X_t: reads never change the banks, so any number of hops reuses one write pass (`bank_write_*`
    then `bank_read` per hop). What would break the recurrence is a read result entering a later write or the LRU.
(3) The read needs only Kb_t q_t (an M-vector), z_t and the weighted sum of Vb_t, at the same t:
        Kb_t q_t   = gamma_t (Kb_0 q_t) + sum_i D_ti r_i (k_i . q_t)           ((C x C) scores times decays)
        sum_j w_tj Vb_t[j] = sum_j w_tj gamma_tj Vb_0[j] + sum_i (sum_j w_tj D_tij r_ij) v_i
    with w_t = p_t / z_t computed position by position from the first line. Everything is matmuls inside the chunk,
    plus one state per chunk boundary: the same structure as chunked linear attention (GLA / GSA), and equal to the
    loop by (A). Chunk boundaries cost O(T / C) sequential steps (themselves an associative scan of affine maps,
    if needed). Hence exact equality for any chunk size C, including C = 1 (the loop) and C = T (fully parallel).
(4) The LRU is the same argument with a complex scalar per channel: h_t = (1 - rho_t) lambda h_{t-1} + b_t.

The torch chunked form here materializes (C x C) decay masks per slot ((B, H, M, C, C)): the reference for a fused
kernel (fla's `chunk_gsa` computes this structure tiled); fine for tests and small models, not for 512 slots at
training scale.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


# -- LRU: h_t = (1 - rho_t) lambda h_{t-1} + b_t, complex diagonal -------------------------------------------------------

def lru_loop(b: torch.Tensor, log_lam: torch.Tensor, reset: torch.Tensor, h0: Optional[torch.Tensor] = None):
    """b `(B, T, N)` complex, log_lam `(N,)` complex (real part < 0) or `(B, T, N)` (a decay per token: the input
    gate), reset `(B, T)` bool. Returns h `(B, T, N)`, h_T."""
    lam = log_lam.exp()
    h = b.new_zeros(b.shape[0], b.shape[2]) if h0 is None else h0
    out = []
    for t in range(b.shape[1]):
        h = (lam if lam.dim() == 1 else lam[:, t]) * h * (~reset[:, t, None]).to(b.dtype) + b[:, t]
        out.append(h)
    return torch.stack(out, 1), h


def _chunk_masks(reset_c: torch.Tensor):
    """reset_c `(B, C)` bool -> lower & same-document mask `(B, C, C)` (t, i), distances `(C, C)`, and carry `(B, C)`:
    position t still sees the state before the chunk (no document start in the chunk up to t)."""
    C = reset_c.shape[1]
    seg = reset_c.long().cumsum(1)
    idx = torch.arange(C, device=reset_c.device)
    lower = idx[:, None] >= idx[None, :]
    same = (seg[:, :, None] == seg[:, None, :]) & lower
    dist = (idx[:, None] - idx[None, :]).clamp_min(0)
    return same, dist, seg == 0


MAX_EXP = 30.0  # chunks are cut so that a^(-i) stays below e^30 (see `_safe_chunk`)


def _safe_chunk(chunk: int, T: int, fastest: float) -> int:
    """The chunk length: at most `chunk` and T, and short enough that the fastest decay's inverse powers a^(-i),
    i < C, stay below e^MAX_EXP (fastest: the largest |log a|)."""
    return max(1, min(chunk, T, int(MAX_EXP / max(fastest, 1e-9)) + 1))


def lru_chunk(b: torch.Tensor, log_lam: torch.Tensor, reset: torch.Tensor, h0: Optional[torch.Tensor] = None,
              chunk: int = 16):
    """`lru_loop` in chunks (proof (2) / (4) in the module doc). lambda^(t - i) = lambda^t lambda^(-i): a chunk's own
    part is one matmul per chunk (batched), then the chunk-start states one after the other (a multiply-add per
    chunk), then their contributions at once. A decay per token (log_lam `(B, T, N)`): lambda^(t - i) becomes
    exp(L_t - L_i), L the cumulative log decay inside the chunk."""
    B, T, N = b.shape
    C = _safe_chunk(chunk, T, float(log_lam.real.detach().abs().max()))
    n = -(-T // C)
    pad = n * C - T
    if pad:  # positions after the row: outputs dropped
        b = F.pad(b, (0, 0, 0, pad))
        reset = F.pad(reset, (0, pad))
        if log_lam.dim() == 3:
            log_lam = torch.cat([log_lam, log_lam.new_zeros(B, pad, N)], 1)
    same, _, carry = _chunk_masks(reset.reshape(B * n, C))
    same, carry = same.view(B, n, C, C).to(b.dtype), carry.view(B, n, C).to(b.dtype)
    if log_lam.dim() == 1:
        idx = torch.arange(C, device=b.device)[:, None]
        L = ((idx + 1) * log_lam)[None, None]  # (1, 1, t, N)
    else:
        L = log_lam.reshape(B, n, C, N).cumsum(2)  # (B, n, t, N)
    inner = torch.exp(L) * torch.einsum("bcti,bcin->bctn", same, b.reshape(B, n, C, N) * torch.exp(-L))
    g = torch.exp(L) * carry[..., None]  # (B, n, t, N)
    h = b.new_zeros(B, N) if h0 is None else h0
    starts = []
    for c in range(n):
        starts.append(h)
        h = g[:, c, -1] * h + inner[:, c, -1]
    out = (inner + g * torch.stack(starts, 1)[:, :, None]).reshape(B, n * C, N)[:, :T]
    return out, out[:, -1]


# -- the banks -----------------------------------------------------------------------------------------------------------

def write_scale(loga: torch.Tensor, norm: str) -> torch.Tensor:
    """The constant c_j multiplying every write into slot j (`(H, M)`), see the module doc."""
    if norm == "mass":
        return torch.ones_like(loga)
    if norm == "ema":
        return -torch.expm1(loga)  # 1 - a
    if norm == "lru":
        return torch.sqrt(-torch.expm1(2 * loga))  # sqrt(1 - a^2)
    if norm == "none":
        return torch.ones_like(loga)
    raise ValueError(f"norm {norm!r}: mass, ema, lru or none")


def _read(s: torch.Tensor, z: torch.Tensor, tau: torch.Tensor, divide: bool = True):
    """Read weights from the slot scores Kb.q `(..., M)` and masses z, over the slots with z > EPS (0 elsewhere):
    divide (mass): w = softmax(tau s / z) / z (averages); else w = softmax(tau s). tau broadcasts over the leading dims."""
    valid = z > EPS
    zc = z.clamp_min(EPS) if divide else torch.ones_like(z)
    score = torch.where(valid, tau * s / zc, torch.full_like(s, -1e30))  # finite: a row without valid slots gives 0
    p = torch.softmax(score, -1)
    return torch.where(valid, p / zc, torch.zeros_like(p))


def bank_write_loop(k, v, r, loga, reset, state=None, norm: str = "mass", erase=None):
    """The writes, token by token (the definition): the banks after every position. k `(B, T, H, D)` (unit vectors),
    v `(B, T, H, E)`, r `(B, T, H, M)` (routing weights), loga `(H, M)` (log decay per slot, < 0), reset `(B, T)`
    bool, state (Kb `(B, H, M, D)`, Vb `(B, H, M, E)`, z `(B, H, M)`) or None, norm: the write normalization,
    erase: `(B, T, H, M)` log factors <= 0 added to the decay at each position, before its write (known in advance:
    the recurrence stays linear), or None. Returns the memory to read (`bank_read`) and the state after the last
    position."""
    B, T, H, D = k.shape
    M, E = r.shape[-1], v.shape[-1]
    if state is None:
        state = (k.new_zeros(B, H, M, D), k.new_zeros(B, H, M, E), k.new_zeros(B, H, M))
    Kb, Vb, z = state
    r = r * write_scale(loga, norm)
    Ks, Vs, zs = [], [], []
    for t in range(T):
        keep = (~reset[:, t]).to(k.dtype)[:, None, None]  # (B, 1, 1)
        a = (loga if erase is None else loga + erase[:, t]).exp()  # (H, M) or (B, H, M)
        rt = r[:, t]  # (B, H, M)
        Kb = (keep * a)[..., None] * Kb + rt[..., None] * k[:, t, :, None, :]
        Vb = (keep * a)[..., None] * Vb + rt[..., None] * v[:, t, :, None, :]
        z = keep * a * z + rt
        Ks.append(Kb)
        Vs.append(Vb)
        zs.append(z)
    mem = {"mode": "loop", "norm": norm, "Kb": torch.stack(Ks, 1), "Vb": torch.stack(Vs, 1), "z": torch.stack(zs, 1)}
    return mem, (Kb, Vb, z)


def _to_chunks(x: torch.Tensor, mem: dict) -> torch.Tensor:
    """`(B, T, H, M)` -> the chunk layout `(B, n, H, t, M)` of a chunk memory."""
    if mem["pad"]:
        x = F.pad(x, (0, 0, 0, 0, 0, mem["pad"]))
    return x.reshape(mem["B"], mem["n"], mem["C"], *x.shape[2:]).transpose(2, 3)


def _from_chunks(y: torch.Tensor, mem: dict) -> torch.Tensor:
    """`(B, n, H, t, M)` -> `(B, T, H, M)`."""
    y = y.transpose(2, 3)
    return y.reshape(mem["B"], mem["n"] * mem["C"], *y.shape[3:])[:, :mem["T"]]


def bank_mass(mem: dict) -> torch.Tensor:
    """The written mass z at every position `(B, T, H, M)`."""
    return mem["z"] if mem["mode"] == "loop" else _from_chunks(mem["zt"], mem)


def bank_weights(mem: dict, q: torch.Tensor, tau: torch.Tensor):
    """The read's addressing at every position, queries q `(B, T, H, D)`: the weights w (= p / z under "mass", applied
    to the summed values) and probabilities p over the slots `(B, T, H, M)`, and the best match per head `(B, T, H)`
    (max over the non-empty slots of q . average key; 0 when all are empty)."""
    if mem["mode"] == "loop":
        s, z = torch.einsum("bthmd,bthd->bthm", mem["Kb"], q), mem["z"]
    else:
        B, n, C = mem["B"], mem["n"], mem["C"]
        qc = q if not mem["pad"] else F.pad(q, (0, 0, 0, 0, 0, mem["pad"]))
        qc = qc.reshape(B, n, C, *q.shape[2:])
        same, kc, ra, pw, gam = mem["same"], mem["kc"], mem["ra"], mem["pw"], mem["gam"]
        qk = torch.einsum("bcthd,bcihd->bchti", qc, kc) * same[:, :, None]
        sc = pw * torch.einsum("bchti,bcihm->bchtm", qk, ra) + gam * torch.einsum("bchmd,bcthd->bchtm", mem["K0"], qc)
        s, z = _from_chunks(sc, mem), _from_chunks(mem["zt"], mem)
    mass = mem["norm"] == "mass"
    w = _read(s, z, tau[:, None], mass)
    valid = z > EPS
    zc = z.clamp_min(EPS) if mass else torch.ones_like(z)
    p = w * zc
    best = torch.where(valid, s / zc, torch.full_like(s, -1e30)).amax(-1)
    return w, p, torch.where(valid.any(-1), best, torch.zeros_like(best))


def bank_aggregate(mem: dict, w: torch.Tensor, which: str = "v") -> torch.Tensor:
    """sum_j w_tj X_t[j] at every position: the banks of values (which "v": `(B, T, H, E)`) or keys ("k")."""
    if mem["mode"] == "loop":
        return torch.einsum("bthm,bthme->bthe", w, mem["Vb" if which == "v" else "Kb"])
    wc = _to_chunks(w, mem)
    A = torch.einsum("bchtm,bcihm->bchti", wc * mem["pw"], mem["ra"]) * mem["same"][:, :, None]  # each earlier position
    xc, x0 = (mem["vc"], mem["V0"]) if which == "v" else (mem["kc"], mem["K0"])
    o = torch.einsum("bchti,bcihe->bcthe", A, xc) + torch.einsum("bchtm,bchme->bcthe", wc * mem["gam"], x0)
    return o.reshape(mem["B"], mem["n"] * mem["C"], *o.shape[3:])[:, :mem["T"]]


def bank_read(mem: dict, q: torch.Tensor, tau: torch.Tensor, record: Optional[list] = None) -> torch.Tensor:
    """Read the banks at every position with the queries q `(B, T, H, D)` (unit vectors), tau `(H,)`: o `(B, T, H,
    E)`. The banks never change by a read, so any number of reads (hops) can use one `bank_write_*`, with queries
    computed from earlier reads at the same position. record: a list that gets the read probabilities over the slots
    `(B, T, H, M)`. A "split" memory (`LRUBankLayer(erase="read")`) addresses with its key bank and reads the values
    from its erasable value bank."""
    if mem["mode"] == "split":
        _, p, _ = bank_weights(mem["K"], q, tau)
        zv = bank_mass(mem["V"])
        w = torch.where(zv > EPS, p / zv.clamp_min(EPS), torch.zeros_like(p))
        if record is not None:
            record.append(p.detach())
        return bank_aggregate(mem["V"], w)
    w, p, _ = bank_weights(mem, q, tau)
    if record is not None:
        record.append(p.detach())
    return bank_aggregate(mem, w)


def bank_loop(q, k, v, r, loga, tau, reset, state=None, norm: str = "mass", record: Optional[list] = None):
    """The definition, token by token: `bank_write_loop`, then one read. Returns o `(B, T, H, E)` and the state after
    the last position."""
    mem, state = bank_write_loop(k, v, r, loga, reset, state, norm)
    return bank_read(mem, q, tau, record), state


def bank_chunk(q, k, v, r, loga, tau, reset, state=None, chunk: int = 16, norm: str = "mass"):
    """`bank_loop` in chunks: `bank_write_chunk`, then one read."""
    mem, state = bank_write_chunk(k, v, r, loga, reset, state, chunk, norm)
    return bank_read(mem, q, tau), state


def bank_write_chunk(k, v, r, loga, reset, state=None, chunk: int = 16, norm: str = "mass", erase=None):
    """`bank_write_loop` in chunks (proof (2), (3) in the module doc), the same arguments; the memory holds what a read
    needs, not the banks at every position. a^(t - i) = a^t a^(-i), so every in-chunk term of a read is a matmul ((C x
    C) by (C x M)), batched over the chunks; the banks at each chunk start are computed here one after the other (a
    multiply-add per chunk). With `erase` the decays vary per position: a^(t - i) becomes exp(L_t - L_i), L the
    cumulative log decay inside the chunk (a GLA-style forget gate); the chunk is cut so exp(-L) stays below e^30."""
    B, T, H, D = k.shape
    M, E = r.shape[-1], v.shape[-1]
    if state is None:
        state = (k.new_zeros(B, H, M, D), k.new_zeros(B, H, M, E), k.new_zeros(B, H, M))
    Kb, Vb, z = state
    r = r * write_scale(loga, norm)
    la = None if erase is None else loga + erase  # (B, T, H, M)
    C = _safe_chunk(chunk, T, float((loga if la is None else la).detach().abs().max()))
    n = -(-T // C)
    pad = n * C - T
    if pad:  # past the row: no write (r = 0), no reset, no decay; outputs dropped, the last state read at the last real position
        k, v, r = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (k, v, r))
        reset = F.pad(reset, (0, pad))
        la = None if la is None else F.pad(la, (0, 0, 0, 0, 0, pad))
    kc, vc, rc = (x.reshape(B, n, C, *x.shape[2:]) for x in (k, v, r))
    same, _, carry = _chunk_masks(reset.reshape(B * n, C))
    same, carry = same.view(B, n, C, C).to(k.dtype), carry.view(B, n, C).to(k.dtype)
    idx = torch.arange(C, device=k.device)
    if la is None:  # L_t = (t + 1) log a, the same for every row: (1, 1, t, H, M)
        L = ((idx + 1)[:, None, None] * loga)[None, None]
    else:
        L = la.reshape(B, n, C, H, M).cumsum(2)  # (B, n, t, H, M): log of the decays a_0 .. a_t of the chunk
    pw = L.exp().transpose(2, 3)  # (., ., H, t, M): exp(L_t)
    ra = rc * torch.exp(-L)  # (B, n, i, H, M): r_im exp(-L_i)
    gam = pw * carry[:, :, None, :, None]  # (B, n, H, t, M): what is left of the chunk-start banks at t
    # chunk-end contributions (exp(L_{C-1} - L_i) r_im, same document as the last position), then the chunk-start banks
    lw = rc * torch.exp(L[:, :, -1:] - L) * same[:, :, -1, :, None, None]  # (B, n, i, H, M)
    cK = torch.einsum("bcihm,bcihd->bchmd", lw, kc)
    cV = torch.einsum("bcihm,bcihe->bchme", lw, vc)
    cz = lw.sum(2)  # (B, n, H, M)
    gend = gam[:, :, :, -1]
    K0, V0, z0 = [], [], []
    for c in range(n):
        K0.append(Kb)
        V0.append(Vb)
        z0.append(z)
        Kb = gend[:, c, ..., None] * Kb + cK[:, c]
        Vb = gend[:, c, ..., None] * Vb + cV[:, c]
        z = gend[:, c] * z + cz[:, c]
    K0, V0, z0 = torch.stack(K0, 1), torch.stack(V0, 1), torch.stack(z0, 1)
    if pad:  # the banks at the last real position of the last chunk
        t = C - 1 - pad
        wt = rc[:, -1] * torch.exp(L[:, -1, t:t + 1] - L[:, -1]) * same[:, -1, t, :, None, None]
        g_t = gam[:, -1, :, t]
        Kb = g_t[..., None] * K0[:, -1] + torch.einsum("bihm,bihd->bhmd", wt, kc[:, -1])
        Vb = g_t[..., None] * V0[:, -1] + torch.einsum("bihm,bihe->bhme", wt, vc[:, -1])
        z = g_t * z0[:, -1] + wt.sum(1)
    zt = pw * torch.einsum("bcti,bcihm->bchtm", same, ra) + gam * z0[:, :, :, None]  # the mass at every position
    mem = {"mode": "chunk", "norm": norm, "B": B, "T": T, "n": n, "C": C, "pad": pad, "same": same, "kc": kc,
           "vc": vc, "ra": ra, "pw": pw, "gam": gam, "K0": K0, "V0": V0, "zt": zt}
    return mem, (Kb, Vb, z)


def route(k: torch.Tensor, router: torch.Tensor, topk: int, renorm: bool = True) -> torch.Tensor:
    """r = the top-K of softmax(k R) per head, renormalized to sum 1 (renorm) or the raw probabilities, scattered back
    to `(B, T, H, M)` (0 elsewhere). With K = 1 and renorm the weight is always 1: the router gets no gradient."""
    p = torch.softmax(torch.einsum("bthd,hdm->bthm", k, router), -1)
    top, ix = p.topk(topk, -1)
    return torch.zeros_like(p).scatter(-1, ix, top / top.sum(-1, keepdim=True) if renorm else top)


def erase_address(q: torch.Tensor, router: torch.Tensor, topk: int) -> torch.Tensor:
    """Where a query would be filed: the top-K of softmax(q R) divided by the largest (so the first slot gets 1 and a
    single gate can erase it fully), 0 elsewhere: `(B, T, H, M)`. The same hash as the writes (`route`)."""
    p = torch.softmax(torch.einsum("bthd,hdm->bthm", q, router), -1)
    top, ix = p.topk(topk, -1)
    return torch.zeros_like(p).scatter(-1, ix, top / top[..., :1])


# -- the layer -----------------------------------------------------------------------------------------------------------

class LRUBankLayer(nn.Module):
    """LRU (key / query state) + the banks. `forward(x, reset, mode)` on rows `(B, T, dim)`; `step(x_t, reset_t,
    state)` one position at a time (generation). mode "chunk" (parallel) or "loop" (the definition)."""

    def __init__(self, dim: int, lru_dim: int = 64, heads: int = 2, key_dim: int = 32, value_dim: int = 32,
                 slots: int = 64, topk: int = 2, slot_half_life=(1.0, 1024.0), lru_half_life=(1.0, 16.0),
                 tau: float = 8.0, chunk: int = 16, norm: str = "mass", route_renorm: bool = True, hops: int = 1,
                 erase=False, erase_max: float = 0.95, erase_bias: float = -4.0, input_gate: bool = False,
                 input_gate_bias: float = 3.0):
        super().__init__()
        self.N, self.H, self.D, self.E, self.M, self.topk, self.chunk = lru_dim, heads, key_dim, value_dim, slots, topk, chunk
        self.norm, self.route_renorm = norm, route_renorm
        hl = torch.exp(torch.linspace(math.log(lru_half_life[0]), math.log(lru_half_life[1]), lru_dim))
        self.nu = nn.Parameter(torch.log(math.log(2) / hl))  # |lambda| = exp(-e^nu): half-life hl
        self.theta = nn.Parameter(torch.rand(lru_dim) * 2 * math.pi)
        self.in_proj = nn.Linear(dim, 2 * lru_dim, bias=False)
        self.wk = nn.Linear(2 * lru_dim, heads * key_dim)  # with a bias: the state before a document's first token is 0
        self.wq = nn.Linear(2 * lru_dim, heads * key_dim, bias=False)
        self.wv = nn.Linear(dim, heads * value_dim, bias=False)
        self.router = nn.Parameter(torch.randn(heads, key_dim, slots) / math.sqrt(key_dim))
        shl = torch.exp(torch.linspace(math.log(slot_half_life[0]), math.log(slot_half_life[1]), slots))
        self.register_buffer("loga", (math.log(0.5) / shl).repeat(heads, 1))  # fixed per slot, the same in every head
        self.log_tau = nn.Parameter(torch.full((heads,), math.log(tau)))
        self.wo = nn.Linear(heads * value_dim, dim, bias=False)
        # hops 2..: a query from the stream after the earlier hops, its own temperature and output projection
        self.wq_hop = nn.ModuleList([nn.Linear(dim, heads * key_dim, bias=False) for _ in range(hops - 1)])
        self.wo_hop = nn.ModuleList([nn.Linear(heads * value_dim, dim, bias=False) for _ in range(hops - 1)])
        self.log_tau_hop = nn.Parameter(torch.full((hops - 1, heads), math.log(tau))) if hops > 1 else None
        # LRU input gate (selective decay): g_t = sigmoid(w x_t + b) from the layer's input; lambda_t = lambda^g_t,
        # gamma_t = sqrt(1 - |lambda_t|^2): g = 0 skips the token (no decay, no rotation, no input), g = 1 is the plain
        # LRU. A coefficient per token known before the recurrence: still linear (chunks with cumulative log decays).
        self.wg = None
        if input_gate:
            self.wg = nn.Linear(dim, 1)
            nn.init.zeros_(self.wg.weight)
            nn.init.constant_(self.wg.bias, input_gate_bias)
        # query-routed erase: before the write at t, the slots where the query of t - 1 would be filed (route of
        # normalize(W_q s_{t-1}), the same state the key of t comes from) shrink by e_t = erase_max sigmoid(w s_{t-1} + b)
        # per head. Every factor comes from the LRU state, never from a read: the banks stay a linear recurrence.
        # erase "read" (two banks by role): the key bank (keys, mass) is the address memory, never erased; the value
        # bank (values, mass) is erased. The read addresses with the key bank (p_t); before the write at t + 1 the
        # value bank shrinks where that read landed, by e_t p_tj / max_j p_tj, e_t = erase_max sigmoid(w [s_t, the
        # matched average key, the best match] + b): a function of the key bank's read, which the value bank never
        # feeds, so both stay linear recurrences (the key bank first, then the value bank with known decays).
        self.erase = "query" if erase is True else (erase or None)
        assert self.erase in (None, "query", "read"), erase
        self.erase_max = erase_max
        self.we = None
        if self.erase:
            self.we = nn.Linear(2 * lru_dim + (heads * key_dim + heads if self.erase == "read" else 0), heads)
            nn.init.zeros_(self.we.weight)
            nn.init.constant_(self.we.bias, erase_bias)

    def log_lam(self) -> torch.Tensor:
        return torch.complex(-self.nu.exp(), self.theta)

    def lru_input(self, x: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gamma B u; g `(B, T, 1)`: the input gate (gamma of the gated decay lambda^g)."""
        rate = self.nu.exp() if g is None else g * self.nu.exp()
        gamma = torch.sqrt((-torch.expm1(-2 * rate)).clamp_min(1e-12))
        re, im = self.in_proj(x).chunk(2, -1)
        return gamma * torch.complex(re, im)

    def lru_coeffs(self, x: torch.Tensor):
        """The LRU's input b `(B, T, N)` and log decay (`(N,)`, or `(B, T, N)` with the input gate) for inputs x, and the
        gate `(B, T, 1)` or None."""
        if self.wg is None:
            return self.lru_input(x), self.log_lam(), None
        g = torch.sigmoid(self.wg(x))
        return self.lru_input(x, g), g * self.log_lam(), g

    def heads(self, s_prev: torch.Tensor, s: torch.Tensor, x: torch.Tensor):
        sh = lambda t, d: t.reshape(*t.shape[:-1], self.H, d)
        k = F.normalize(sh(self.wk(s_prev), self.D), dim=-1)
        q = F.normalize(sh(self.wq(s), self.D), dim=-1)
        v = sh(self.wv(x), self.E)
        return q, k, v

    def write(self, x: torch.Tensor, reset: torch.Tensor, mode: str = "chunk", state=None, info: Optional[dict] = None):
        """The LRU and the writes for rows x `(B, T, dim)`: the memory to read (`read`), the first-hop queries
        `(B, T, H, D)` and the state after the last position ((h `(B, N)` complex, banks); `state` carries one in).
        mode "chunk" (parallel) or "loop" (the definition)."""
        B, T, _ = x.shape
        h0, banks = (None, None) if state is None else state
        lru = lru_chunk if mode == "chunk" else lru_loop
        kw = {"chunk": self.chunk} if mode == "chunk" else {}
        b_in, log_lam, gate = self.lru_coeffs(x)
        h, hT = lru(b_in, log_lam, reset, h0, **kw)
        s = torch.cat([h.real, h.imag], -1)
        s0 = s.new_zeros(B, 1, s.shape[-1]) if h0 is None else torch.cat([h0.real, h0.imag], -1)[:, None]
        s_prev = torch.cat([s0, s[:, :-1]], 1) * (~reset)[..., None].to(s.dtype)
        q, k, v = self.heads(s_prev, s, x)
        r = route(k, self.router, self.topk, self.route_renorm)
        if self.erase == "read":
            mem, banks = self._write_split(s, q, k, v, r, reset, mode, banks, info)
            if info is not None:
                info.update(route=r.detach(), q=q.detach(), k=k.detach(), s=s.detach())
            return mem, q, (hT, banks)
        erase = None
        if self.we is not None:  # log(1 - e_t rho_t), from s_{t-1} only
            q_prev = F.normalize(self.wq(s_prev).reshape(B, T, self.H, self.D), dim=-1)
            e = self.erase_max * torch.sigmoid(self.we(s_prev))  # (B, T, H)
            erase = torch.log1p(-e[..., None] * erase_address(q_prev, self.router, self.topk))
        if mode == "chunk":
            mem, banks = bank_write_chunk(k, v, r, self.loga, reset, banks, self.chunk, self.norm, erase)
        else:
            mem, banks = bank_write_loop(k, v, r, self.loga, reset, banks, self.norm, erase)
        if info is not None:
            info.update(route=r.detach(), q=q.detach(), k=k.detach(), s=s.detach())
            if gate is not None:
                info.update(input_gate=gate[..., 0].detach())
            if erase is not None:
                info.update(erase=(-torch.expm1(erase)).detach())  # the fraction erased per slot
        return mem, q, (hT, banks)

    def _write_split(self, s, q, k, v, r, reset, mode, banks, info):
        """erase "read": the key bank, its read at every position, the erase it decides, then the value bank. banks:
        (key bank state, value bank state, the erase pending for the next position `(B, H, M)` log factors) or None."""
        B, T = k.shape[:2]
        stK, stV, pending = (None, None, None) if banks is None else banks
        write = (lambda *a, **kw: bank_write_chunk(*a, chunk=self.chunk, **kw)) if mode == "chunk" else bank_write_loop
        memK, stK = write(k, k[..., :0], r, self.loga, reset, stK, norm=self.norm)  # keys + mass only
        _, p, best = bank_weights(memK, q, self.log_tau.exp())
        kmatch = bank_aggregate(memK, torch.where(bank_mass(memK) > EPS, p / bank_mass(memK).clamp_min(EPS),
                                                  torch.zeros_like(p)), "k")  # the matched average key
        feats = torch.cat([s, kmatch.reshape(B, T, -1), best], -1)
        e = self.erase_max * torch.sigmoid(self.we(feats))  # (B, T, H)
        rho = p / p.amax(-1, keepdim=True).clamp_min(EPS)
        nxt = torch.log1p(-e[..., None] * rho)  # applied at the next position, before its write
        first = nxt.new_zeros(B, 1, *nxt.shape[2:]) if pending is None else pending[:, None]
        erase = torch.cat([first, nxt[:, :-1]], 1)
        memV, stV = write(k[..., :0], v, r, self.loga, reset, stV, norm=self.norm, erase=erase)  # values + mass
        if info is not None:
            info.update(erase=(-torch.expm1(erase)).detach())
        return {"mode": "split", "K": memK, "V": memV}, (stK, stV, nxt[:, -1])

    def query(self, stream: torch.Tensor, hop: int) -> torch.Tensor:
        """The query of hop `hop` >= 1 (0 is the LRU's) from the residual stream `(B, T, dim)` after the earlier hops:
        position by position, so the banks stay a linear recurrence (see `bank_read`)."""
        q = self.wq_hop[hop - 1](stream)
        return F.normalize(q.reshape(*q.shape[:-1], self.H, self.D), dim=-1)

    def read(self, mem: dict, q: torch.Tensor, hop: int = 0, info: Optional[dict] = None) -> torch.Tensor:
        """Read the banks with the queries of hop `hop`: `(B, T, dim)` for the residual stream."""
        tau = (self.log_tau if hop == 0 else self.log_tau_hop[hop - 1]).exp()
        rec = [] if info is not None else None
        o = bank_read(mem, q, tau, rec)
        if info is not None:
            sfx = "" if hop == 0 else f"_hop{hop}"
            info.update({"o" + sfx: o.detach(), "q" + sfx: q.detach()}, **({"read" + sfx: rec[0]} if rec else {}))
        wo = self.wo if hop == 0 else self.wo_hop[hop - 1]
        return wo(o.reshape(*o.shape[:2], self.H * self.E))

    def forward(self, x: torch.Tensor, reset: torch.Tensor, mode: str = "chunk", info: Optional[dict] = None) -> torch.Tensor:
        """One hop: write, then the LRU's query. info: a dict that gets the internals (the loop runs): route `(B, T, H,
        M)` (write weights), read `(B, T, H, M)` (read probabilities), q, k `(B, T, H, D)`, lru state s `(B, T, 2N)`,
        bank output o `(B, T, H, E)`. More hops: `write`, `read`, then `query` / `read` per hop (the caller decides
        what the stream holds between them, e.g. an MLP)."""
        if info is not None:
            mode = "loop"
        mem, q, _ = self.write(x, reset, mode, info=info)
        return self.read(mem, q, 0, info)

    def step(self, x_t: torch.Tensor, reset_t: torch.Tensor, state=None):
        """One position: x_t `(B, dim)`, reset_t `(B,)`. state: (h `(B, N)` complex, banks) or None."""
        mem, q, state = self.write(x_t[:, None], reset_t[:, None], "loop", state)
        return self.read(mem, q)[:, 0], state


class StreamBankLayer(LRUBankLayer):
    """A bank layer without an LRU, for layers above the first: one projection phi of the residual stream gives both
    the key and the query, one position apart (key_t = normalize(phi(x_{t-1})), "what the stream said before this
    token"; query_t = normalize(phi(x_t))), so keys and queries live in the same space by construction, as the LRU's
    state before / after the token does in `LRUBankLayer`. Later hops query with the same phi. The writes are
    position-wise functions of the input: the banks stay a linear recurrence (the module doc's proof)."""

    def __init__(self, dim: int, heads: int = 2, key_dim: int = 32, value_dim: int = 32, slots: int = 64,
                 topk: int = 2, slot_half_life=(1.0, 1024.0), tau: float = 8.0, chunk: int = 16, norm: str = "mass",
                 route_renorm: bool = True, hops: int = 1):
        nn.Module.__init__(self)
        self.H, self.D, self.E, self.M, self.topk, self.chunk = heads, key_dim, value_dim, slots, topk, chunk
        self.norm, self.route_renorm, self.erase, self.we = norm, route_renorm, None, None
        self.phi = nn.Linear(dim, heads * key_dim)  # with a bias: the key before a document's first token is phi(0)
        self.wv = nn.Linear(dim, heads * value_dim, bias=False)
        self.router = nn.Parameter(torch.randn(heads, key_dim, slots) / math.sqrt(key_dim))
        shl = torch.exp(torch.linspace(math.log(slot_half_life[0]), math.log(slot_half_life[1]), slots))
        self.register_buffer("loga", (math.log(0.5) / shl).repeat(heads, 1))
        self.log_tau = nn.Parameter(torch.full((heads,), math.log(tau)))
        self.wo = nn.Linear(heads * value_dim, dim, bias=False)
        self.wo_hop = nn.ModuleList([nn.Linear(heads * value_dim, dim, bias=False) for _ in range(hops - 1)])
        self.log_tau_hop = nn.Parameter(torch.full((hops - 1, heads), math.log(tau))) if hops > 1 else None

    def _addr(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.phi(x).reshape(*x.shape[:-1], self.H, self.D), dim=-1)

    def write(self, x: torch.Tensor, reset: torch.Tensor, mode: str = "chunk", state=None, info: Optional[dict] = None):
        """As `LRUBankLayer.write`; state: (the last input `(B, dim)`, banks) or None."""
        B = x.shape[0]
        x0, banks = (None, None) if state is None else state
        x0 = x.new_zeros(B, 1, x.shape[-1]) if x0 is None else x0[:, None]
        x_prev = torch.cat([x0, x[:, :-1]], 1) * (~reset)[..., None].to(x.dtype)
        k, q = self._addr(x_prev), self._addr(x)
        v = self.wv(x).reshape(*x.shape[:-1], self.H, self.E)
        r = route(k, self.router, self.topk, self.route_renorm)
        if mode == "chunk":
            mem, banks = bank_write_chunk(k, v, r, self.loga, reset, banks, self.chunk, self.norm)
        else:
            mem, banks = bank_write_loop(k, v, r, self.loga, reset, banks, self.norm)
        if info is not None:
            info.update(route=r.detach(), q=q.detach(), k=k.detach())
        return mem, q, (x[:, -1], banks)

    def query(self, stream: torch.Tensor, hop: int) -> torch.Tensor:
        return self._addr(stream)
