"""Training throughput of small sequence mixers on the XPU: tokens/s of a full train step (forward, backward, AdamW)
at rows of 1024 tokens and dim 128. Pure torch, chunkwise forms (matmuls per chunk, a sequential loop over chunks
only for the state).

    uv run python "ad hoc scripts/bench_linear_attention.py" [--models attn linear gla gdn mingru mingru-tanh] [--batch 16] [--seq 1024]

Every model: token embedding, `layers` x (pre-norm mixer + pre-norm GELU MLP 4x), RMSNorm, untied readout, vocab 1024.
Mixers (heads of 64):
  attn    causal softmax attention (`scaled_dot_product_attention`)
  linear  linear attention, S += k v^T (L2-normalized q, k)
  gla     + a data-dependent scalar decay per head and token, S = a_t S + k v^T (Mamba-2 / GLA scalar gate)
  gdn     Gated DeltaNet, S = a_t S (I - b_t k k^T) + b_t k v^T (chunkwise WY form)
  mingru  minGRU (Feng et al. 2024, "Were RNNs All We Needed?"): h_t = (1 - z_t) h_{t-1} + z_t g(h~_t), z and h~
          linear in x only, so the whole row is one parallel scan; hidden `--mingru-expand` x dim, then an output
          projection (lucidrains/minGRU-pytorch layout), no conv. The recurrence runs in linear space by log2(T)
          doubling steps (Hillis-Steele), stable since every factor is in (0, 1). (The paper's log-space scan with
          logcumsumexp is ~100x slower than cumsum on this XPU; the `min-gru` arch has the fast chunked scan.)
  mingru-tanh  minGRU layers without MLPs: per layer residual += scale * tanh(out(LayerNorm(scan(RMSNorm(residual))))),
          scale per channel, init 0.1
The attention-like recurrent mixers (linear, gla, gdn) have the standard short causal conv (width 4) + SiLU on q, k, v
and an output gate.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.device import setup_device

setup_device("xpu")


class LayerNorm(nn.Module):
    """(x - mean) / std * w + b, written out like RMSNorm below."""

    def __init__(self, d):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))

    def forward(self, x):
        x = x - x.mean(-1, keepdim=True)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w + self.b


class RMSNorm(nn.Module):
    """x / rms(x) * w. (nn.RMSNorm's backward fails on this XPU: "tensor does not have a device".)"""

    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


def chunk_linear(q, k, v, g, L):
    """o_i = sum_{j<=i} exp(G_i - G_j) (q_i . k_j) v_j, G the cumulative log decay (g None: no decay).
    q, k, v (B, H, N, D); g (B, H, N)."""
    B, H, N, D = q.shape
    C = N // L
    q, k, v = (x.reshape(B, H, C, L, -1) for x in (q, k, v))
    G = torch.zeros(B, H, C, L, device=q.device, dtype=q.dtype) if g is None else g.reshape(B, H, C, L).cumsum(-1)
    tri = torch.ones(L, L, device=q.device, dtype=torch.bool).tril()
    dec = torch.exp((G[..., :, None] - G[..., None, :]).masked_fill(~tri, float("-inf")))
    intra = ((q @ k.transpose(-1, -2)) * dec) @ v
    kv = (k * torch.exp(G[..., -1:] - G)[..., None]).transpose(-1, -2) @ v  # (B, H, C, D, Dv)
    qd = q * torch.exp(G)[..., None]
    keep = torch.exp(G[..., -1])  # (B, H, C)
    S = q.new_zeros(B, H, D, v.shape[-1])
    outs = []
    for c in range(C):
        outs.append(qd[:, :, c] @ S)
        S = keep[:, :, c, None, None] * S + kv[:, :, c]
    return (intra + torch.stack(outs, 2)).reshape(B, H, N, -1)


def chunk_gdn(q, k, v, beta, g, L):
    """Gated delta rule S_i = a_i S_{i-1} + k_i e_i^T, e_i = b_i (v_i - a_i S_{i-1}^T k_i), o_i = S_i^T q_i.
    Inside a chunk from its start state S0 (G the cumulative log decay, A_ij = exp(G_i - G_j) k_i . k_j, j < i):
    (I + diag(b) A) E = diag(b) (V - diag(e^G) K S0), so E = U - W S0 with both solves done for all chunks at once."""
    B, H, N, D = q.shape
    C = N // L
    q, k, v = (x.reshape(B, H, C, L, -1) for x in (q, k, v))
    beta = beta.reshape(B, H, C, L, 1)
    G = g.reshape(B, H, C, L).cumsum(-1)
    tri = torch.ones(L, L, device=q.device, dtype=torch.bool).tril()
    dec = torch.exp((G[..., :, None] - G[..., None, :]).masked_fill(~tri, float("-inf")))
    A = (k @ k.transpose(-1, -2)) * dec * (~torch.eye(L, device=q.device, dtype=torch.bool))
    T = torch.eye(L, device=q.device, dtype=q.dtype) + beta * A
    U = torch.linalg.solve_triangular(T, beta * v, upper=False, unitriangular=True)
    W = torch.linalg.solve_triangular(T, beta * torch.exp(G)[..., None] * k, upper=False, unitriangular=True)
    Aq = (q @ k.transpose(-1, -2)) * dec
    qg = q * torch.exp(G)[..., None]
    kend = (k * torch.exp(G[..., -1:] - G)[..., None]).transpose(-1, -2)  # (B, H, C, D, L)
    keep = torch.exp(G[..., -1])
    S = q.new_zeros(B, H, D, v.shape[-1])
    outs = []
    for c in range(C):
        E = U[:, :, c] - W[:, :, c] @ S
        outs.append(qg[:, :, c] @ S + Aq[:, :, c] @ E)
        S = keep[:, :, c, None, None] * S + kend[:, :, c] @ E
    return torch.stack(outs, 2).reshape(B, H, N, -1)


def recurrent_gdn(q, k, v, beta, g):
    """The step-by-step reference of `chunk_gdn`."""
    B, H, N, D = q.shape
    S = q.new_zeros(B, H, D, v.shape[-1])
    outs = []
    for t in range(N):
        a, b = torch.exp(g[..., t])[..., None, None], beta[..., t, None]
        S = a * S
        e = b * (v[:, :, t] - torch.einsum("bhk,bhkv->bhv", k[:, :, t], S))
        S = S + k[:, :, t, :, None] * e[:, :, None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
    return torch.stack(outs, 2)


def scan_doubling(keep, write):
    """state[t] = keep[t] * state[t-1] + write[t] along dim 1, state[-1] = 0, in log2(T) steps: after the step of
    shift s, position t holds the composition of the maps over (t - 2s, t]."""
    s, N = 1, keep.shape[1]
    while s < N:
        write = torch.cat([write[:, :s], write[:, s:] + keep[:, s:] * write[:, :-s]], 1)
        keep = torch.cat([keep[:, :s], keep[:, s:] * keep[:, :-s]], 1)
        s *= 2
    return write


class MinGRU(nn.Module):
    """state_after_update = (1 - update_gate) * state_before_update + update_gate * positive_candidate.
    Returns the states (B, T, inner)."""

    def __init__(self, d, expand, out=True):
        super().__init__()
        self.inner = int(d * expand)
        self.hg = nn.Linear(d, 2 * self.inner, bias=False)
        self.out = nn.Linear(self.inner, d, bias=False) if out else nn.Identity()

    def states(self, x):
        h, k = self.hg(x).chunk(2, -1)
        z = torch.sigmoid(k)
        return scan_doubling(1 - z, z * torch.where(h >= 0, h + 0.5, torch.sigmoid(h)))

    def forward(self, x):
        return self.out(self.states(x))


class MinGRUTanh(nn.Module):
    """One layer of the MLP-free stack: residual + scale * tanh(out(LayerNorm(minGRU states of RMSNorm(residual))))."""

    def __init__(self, d, expand):
        super().__init__()
        self.norm_in = RMSNorm(d)
        self.gru = MinGRU(d, expand, out=False)
        self.norm_state = LayerNorm(self.gru.inner)
        self.out = nn.Linear(self.gru.inner, d, bias=False)
        nn.init.normal_(self.out.weight, std=0.02)
        self.scale = nn.Parameter(torch.full((d,), 0.1))

    def forward(self, residual):
        state = self.gru.states(self.norm_in(residual))
        return residual + self.scale * torch.tanh(self.out(self.norm_state(state)))


class Mixer(nn.Module):
    def __init__(self, kind, d, heads, chunk):
        super().__init__()
        self.kind, self.h, self.dh, self.chunk = kind, heads, d // heads, chunk
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        if kind != "attn":
            self.conv = nn.Conv1d(3 * d, 3 * d, 4, groups=3 * d, padding=3)
            self.ogate = nn.Linear(d, d, bias=False)
            self.onorm = RMSNorm(self.dh)
        if kind in ("gla", "gdn"):
            self.decay = nn.Linear(d, heads)
            nn.init.constant_(self.decay.bias, -3.5)  # a = exp(-softplus(.)) ~ 0.97
        if kind == "gdn":
            self.beta = nn.Linear(d, heads)

    def forward(self, x):
        B, N, d = x.shape
        qkv = self.qkv(x)
        if self.kind != "attn":
            qkv = F.silu(self.conv(qkv.transpose(1, 2))[..., :N].transpose(1, 2))
        q, k, v = (t.reshape(B, N, self.h, self.dh).transpose(1, 2) for t in qkv.chunk(3, -1))
        if self.kind == "attn":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.out(o.transpose(1, 2).reshape(B, N, d))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        g = -F.softplus(self.decay(x)).transpose(1, 2) if self.kind in ("gla", "gdn") else None  # (B, H, N)
        if self.kind == "gdn":
            o = chunk_gdn(q, k, v, torch.sigmoid(self.beta(x)).transpose(1, 2), g, self.chunk)
        else:
            o = chunk_linear(q, k, v, g, self.chunk)
        o = self.onorm(o).transpose(1, 2).reshape(B, N, d) * F.silu(self.ogate(x))
        return self.out(o)


class LM(nn.Module):
    def __init__(self, kind, vocab, d, layers, heads, chunk, mingru_expand=1.5):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(vocab, d)
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        if kind == "mingru-tanh":
            self.layers = nn.ModuleList(MinGRUTanh(d, mingru_expand) for _ in range(layers))
            return
        self.mix = nn.ModuleList((MinGRU(d, mingru_expand) if kind == "mingru"
                                  else Mixer(kind, d, heads, chunk)) for _ in range(layers))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
                                 for _ in range(layers))
        self.n1 = nn.ModuleList(RMSNorm(d) for _ in range(layers))
        self.n2 = nn.ModuleList(RMSNorm(d) for _ in range(layers))

    def forward(self, ids):
        x = self.emb(ids)
        if self.kind == "mingru-tanh":
            for layer in self.layers:
                x = layer(x)
            return self.head(self.nf(x))
        for mix, mlp, n1, n2 in zip(self.mix, self.mlp, self.n1, self.n2):
            x = x + mix(n1(x))
            x = x + mlp(n2(x))
        return self.head(self.nf(x))


def check_gdn():
    """chunk_gdn against the step-by-step recurrence (float64, CPU)."""
    g = torch.Generator().manual_seed(0)
    B, H, N, D = 2, 2, 64, 16
    q, k = (F.normalize(torch.randn(B, H, N, D, generator=g, dtype=torch.float64), dim=-1) for _ in range(2))
    v = torch.randn(B, H, N, D, generator=g, dtype=torch.float64)
    beta = torch.rand(B, H, N, generator=g, dtype=torch.float64)
    dec = -torch.rand(B, H, N, generator=g, dtype=torch.float64) * 0.2
    err = (chunk_gdn(q, k, v, beta, dec, 16) - recurrent_gdn(q, k, v, beta, dec)).abs().max().item()
    assert err < 1e-9, err
    return err


def check_mingru():
    """The minGRU scan against the step-by-step recurrence (float64, CPU)."""
    torch.manual_seed(0)
    errs = []
    for _ in range(2):
        m = MinGRU(8, 1.0).double()
        x = torch.randn(2, 50, 8, dtype=torch.float64) * 3
        h, k = m.hg(x).chunk(2, -1)
        z, cand = torch.sigmoid(k), torch.where(h >= 0, h + 0.5, torch.sigmoid(h))
        state, ref = torch.zeros(2, 8, dtype=torch.float64), []
        for t in range(50):
            state = (1 - z[:, t]) * state + z[:, t] * cand[:, t]
            ref.append(state)
        errs.append((m.states(x) - torch.stack(ref, 1)).abs().max().item())
    assert max(errs) < 1e-9, errs
    return max(errs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["attn", "linear", "gla", "gdn", "mingru", "mingru-tanh"])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--mingru-expand", type=float, default=1.5, help="minGRU hidden / dim (1.5: ~attn's mixer params)")
    a = p.parse_args()
    print(f"chunk_gdn vs recurrence: max err {check_gdn():.1e}; minGRU scan vs recurrence: {check_mingru():.1e}")
    print(f"rows {a.batch} x {a.seq}, dim {a.dim}, {a.layers} layers, {a.heads} heads, chunk {a.chunk}")
    dev = torch.device("xpu")
    for kind in a.models:
        torch.manual_seed(0)
        model = LM(kind, a.vocab, a.dim, a.layers, a.heads, a.chunk, a.mingru_expand).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids = torch.randint(0, a.vocab, (a.batch, a.seq + 1), device=dev)

        def step():
            logits = model(ids[:, :-1])
            # log_softmax + gather: F.cross_entropy's XPU kernel asks for work-groups over 512 here
            loss = -logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        for _ in range(3):
            step()
        torch.xpu.synchronize()
        torch.xpu.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            step()
        torch.xpu.synchronize()
        dt = (time.perf_counter() - t0) / a.steps
        params = sum(p.numel() for p in model.parameters())
        print(f"{kind:7s} {params / 1e6:5.2f}M params  {dt * 1e3:8.1f} ms/step  {a.batch * a.seq / dt:10,.0f} tokens/s  "
              f"peak {torch.xpu.max_memory_allocated() / 2**30:.2f} GiB")
        del model, opt


if __name__ == "__main__":
    main()
