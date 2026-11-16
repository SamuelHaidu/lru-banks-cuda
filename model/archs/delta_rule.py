"""The gated delta rule (Gated DeltaNet, Yang, Kautz, Hatamizadeh 2024, arXiv 2412.06464), three ways:

    S_t = alpha_t S_{t-1} + k_t u_t^T,   u_t = beta_t (v_t - alpha_t S_{t-1}^T k_t),   o_t = S_t^T q_t

per head, with S `(K, V)`, alpha_t = exp(g_t) (g <= 0, a decay chosen by the token), beta_t in (0, 1) the writing
strength: the token overwrites what the state returns for its key k_t (the delta rule) after decaying the whole state.
q and k are L2-normalized here, q is scaled by K^-0.5. `reset` `(B, T)` marks document starts: the state before them
is dropped.

- `delta_rule_loop`: token by token (the reference).
- `delta_rule_chunk`: chunks of C tokens, matmuls only plus one unit-triangular solve per chunk (exact; any device;
  the Iris Xe and generation use it). Inside a chunk with the state S0 at its start and G the decays' cumulative sum:
      gamma_t = exp(G_t) [no reset in the chunk up to t],  Gam_tj = exp(G_t - G_j) [same document], j <= t
      (I + diag(beta) (K K^T o Gam o strict lower)) U = diag(beta) (V - diag(gamma) K S0)
      O = diag(gamma) Q S0 + (Q K^T o Gam o lower) U,   S_end = gamma_last S0 + (K o Gam_last)^T U
- CUDA: flash-linear-attention's fused Triton kernel `chunk_gated_delta_rule` (bf16 matmuls on the tensor cores), the
  rows flattened into one sequence cut at every row start and document start (`cu_seqlens`).

`gated_delta_rule` picks fla on CUDA (no carried state), else the torch chunks (the loop for <= 16 tokens: generation).
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:  # flash-linear-attention (pip: flash-linear-attention, fla-core, einops); CUDA only
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk
except Exception:  # noqa: BLE001 (not installed, or no triton backend here)
    _fla_chunk = None

CHUNK = 64


def _qk(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q = F.normalize(q, dim=-1) * q.shape[-1] ** -0.5
    return q, F.normalize(k, dim=-1)


def delta_rule_loop(q, k, v, g, beta, reset, initial=None):
    """q, k `(B, T, H, K)`, v `(B, T, H, V)`, g, beta `(B, T, H)`, reset `(B, T)` bool, initial `(B, H, K, V)`.
    Returns o `(B, T, H, V)` and the final state `(B, H, K, V)`."""
    q, k = _qk(q, k)
    B, T, H, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1]) if initial is None else initial.to(q.dtype)
    out = []
    for t in range(T):
        S = S * (~reset[:, t]).to(S.dtype)[:, None, None, None]
        a = g[:, t].exp()[..., None, None]
        kt, vt, b = k[:, t], v[:, t], beta[:, t][..., None]
        u = b * (vt - a[..., 0] * torch.einsum("bhkv,bhk->bhv", S, kt))
        S = a * S + kt[..., :, None] * u[..., None, :]
        out.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t]))
    return torch.stack(out, 1), S


def delta_rule_chunk(q, k, v, g, beta, reset, initial=None, chunk: int = CHUNK):
    """`delta_rule_loop` in chunks of `chunk` tokens (see the module doc). Same arguments and results."""
    q, k = _qk(q, k)
    B, T, H, K = q.shape
    V = v.shape[-1]
    pad = (-T) % chunk
    if pad:  # padding after the row: g = 0 (keep), beta = 0 (no write): nothing changes
        q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
        g, beta = F.pad(g, (0, 0, 0, pad)), F.pad(beta, (0, 0, 0, pad))
        reset = F.pad(reset, (0, pad))
    N = (T + pad) // chunk
    # (B, H, N, C, .)
    q, k, v = (x.reshape(B, N, chunk, H, -1).permute(0, 3, 1, 2, 4) for x in (q, k, v))
    g, beta = (x.reshape(B, N, chunk, H).permute(0, 3, 1, 2) for x in (g, beta))
    r = reset.reshape(B, 1, N, chunk).expand(B, H, N, chunk)
    G = g.cumsum(-1)
    seg = r.long().cumsum(-1)  # document number inside the chunk (0: the one open at the chunk start)
    same = seg[..., :, None] == seg[..., None, :]
    lower = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    strict = lower & ~torch.eye(chunk, dtype=torch.bool, device=q.device)
    Gam = torch.exp((G[..., :, None] - G[..., None, :]).masked_fill(~(same & lower), float("-inf")))
    gamma = G.exp() * (seg == 0).to(G.dtype)
    A = (k @ k.transpose(-1, -2)) * Gam * strict  # (B, H, N, C, C)
    L = torch.eye(chunk, dtype=q.dtype, device=q.device) + beta[..., None] * A
    attn = (q @ k.transpose(-1, -2)) * Gam
    S = q.new_zeros(B, H, K, V) if initial is None else initial.to(q.dtype)
    out = []
    for n in range(N):
        kn, qn, gn = k[:, :, n], q[:, :, n], gamma[:, :, n]
        rhs = beta[:, :, n, :, None] * (v[:, :, n] - gn[..., None] * (kn @ S))
        U = torch.linalg.solve_triangular(L[:, :, n], rhs, upper=False, unitriangular=True)
        out.append(gn[..., None] * (qn @ S) + attn[:, :, n] @ U)
        S = gn[..., -1, None, None] * S + (kn * Gam[:, :, n, -1, :, None]).transpose(-1, -2) @ U
    o = torch.stack(out, 2).permute(0, 2, 3, 1, 4).reshape(B, N * chunk, H, V)
    return o[:, :T], S


def delta_rule_fla(q, k, v, g, beta, reset):
    """The fused Triton kernel (CUDA): rows flattened into one sequence cut at row and document starts."""
    B, T, H, K = q.shape
    starts = reset.clone()
    starts[:, 0] = True
    cu = torch.nonzero(starts.reshape(-1)).squeeze(1)
    cu = torch.cat([cu, cu.new_tensor([B * T])])
    flat = lambda x: x.reshape(1, B * T, *x.shape[2:])
    o, _ = _fla_chunk(flat(q), flat(k), flat(v), flat(g).float(), flat(beta).to(q.dtype), cu_seqlens=cu,
                      use_qk_l2norm_in_kernel=True)
    return o.reshape(B, T, H, -1), None


def delta_rule_read_weights(q, k, g, beta, reset, t: int) -> Tuple[torch.Tensor, int]:
    """The read at position t unrolled into an exact attention over the past values (no carried state):

        o_t = sum_i c_i v_i,   c_i = beta_i k_i^T A_{i+1} ... A_t q_t,   A_s = alpha_s (I - beta_s k_s k_s^T)

    (S_t = A_t S_{t-1} + beta_t k_t v_t^T, A_s symmetric). c_i is what token i's value is worth in the read now: its
    write strength times how much of its key the later decays and overwrites left in the direction q_t asks for; it is
    negative where a later write to a similar key took more than i wrote. One sweep from t back to the document start
    (O(t K) per head). q, k `(B, T, H, K)` (normalized here, as in the model), g, beta `(B, T, H)`, reset `(T,)` (the
    same for every row: e.g. one text's layers as rows). Returns c `(B, H, t + 1 - start)` and start, the position of
    the document start (c is 0 before it)."""
    q, k = _qk(q, k)
    reset = torch.as_tensor(reset, dtype=torch.bool).reshape(-1)
    starts = torch.nonzero(reset[:t + 1]).reshape(-1)
    start = int(starts[-1]) if len(starts) else 0
    w = q[:, t].clone()  # (B, H, K): A_{i+1} ... A_t q_t
    c = q.new_zeros(q.shape[0], q.shape[2], t + 1 - start)
    for i in range(t, start - 1, -1):
        ki, bi = k[:, i], beta[:, i]
        kw = (ki * w).sum(-1)
        c[..., i - start] = bi * kw
        w = g[:, i].exp()[..., None] * (w - (bi * kw)[..., None] * ki)
    return c, start


def gated_delta_rule(q, k, v, g, beta, reset, initial: Optional[torch.Tensor] = None, need_state: bool = False):
    """The fla kernel on CUDA (training, no carried state), else the torch chunks."""
    if _fla_chunk is not None and q.is_cuda and initial is None and not need_state:
        return delta_rule_fla(q, k, v, g, beta, reset)
    dt = torch.float32 if q.dtype in (torch.bfloat16, torch.float16) else q.dtype
    f = delta_rule_loop if q.shape[1] <= 16 else delta_rule_chunk  # generation steps: no 64-token chunk padding
    o, S = f(q.to(dt), k.to(dt), v.to(dt), g.to(dt), beta.to(dt), reset, initial)
    return o.to(q.dtype), S
