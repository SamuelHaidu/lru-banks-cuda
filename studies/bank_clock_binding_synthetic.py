"""Conceptual test: banks that rotate by their own clock, summed into one vector, read back by a decoder.

Mechanism under test (from the token-banks work):
  - each bank b has a prior per token (selectivity) and a fixed decay gamma;
  - inside a bank, time is the bank's own clock: tau_j = sum_{i<=j} min(prior_b[tok_i], 1). Tokens the bank
    ignores barely move it, so the elements it selects sit at consecutive clock positions however far apart
    they are in the text ([cold][climb][icy][hill] at 1, 2, 3, 4);
  - each token is rotated by its clock distance to the query (RoPE angles on the clock), weighted by
    prior * gamma^age, normalized; each bank is bound to its own random key and all banks are summed:
        h = sum_b key_b * m_b,   m_b = sum_j w_bj R(omega * age_bj) e_j / sum_j w_bj
  - a decoder gets h and a bank id and must return that bank's last 4 elements, most recent first.

Data: 64 categories x 6 tokens + 128 filler tokens. Each 64-token window holds 8 active categories
(2-6 tokens each, random positions) on filler. Bank b selects category b (fixed priors: 1 vs 0.02), or
learns its priors from the decoding loss (learned: all priors start equal). The queried bank is active
80% of the time; otherwise its slots are "none".

Earlier smoke runs (removed here) showed RoPE by position and no rotation are worse: rotation by the clock
is what keeps order, and it is insensitive to gaps.

Usage: python studies/bank_clock_binding_synthetic.py [--steps 3000] [--device xpu]
"""

import argparse
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--device", default="xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
dev = torch.device(args.device)

T, SLOTS, NCAT, PER_CAT, NFILL, ACTIVE = 64, 4, 64, 6, 128, 8
VCAT = NCAT * PER_CAT
V = VCAT + NFILL
NONE = V
GAMMA, EPS = 0.9, 0.02


def sample(B: int, nb: int, g: torch.Generator):
    """Windows (B, T), queried bank (B,), labels (B, SLOTS)."""
    x = VCAT + torch.randint(0, NFILL, (B, T), device=dev, generator=g)
    n_active = min(ACTIVE, nb)
    active = torch.argsort(torch.rand(B, nb, device=dev, generator=g), dim=1)[:, :n_active]  # categories present
    for a in range(n_active):
        m = torch.randint(2, 7, (B, 1), device=dev, generator=g)
        rank = torch.argsort(torch.argsort(torch.rand(B, T, device=dev, generator=g), dim=1), dim=1)
        mask = rank < m
        tok = active[:, a:a + 1] * PER_CAT + torch.randint(0, PER_CAT, (B, T), device=dev, generator=g)
        x = torch.where(mask, tok, x)
    # query an active bank 80% of the time, any bank otherwise
    pick = active[torch.arange(B, device=dev), torch.randint(0, n_active, (B,), device=dev, generator=g)]
    anyb = torch.randint(0, nb, (B,), device=dev, generator=g)
    q = torch.where(torch.rand(B, device=dev, generator=g) < 0.8, pick, anyb)
    cat = torch.where(x < VCAT, x // PER_CAT, torch.full_like(x, -1))
    mine = cat == q[:, None]
    from_right = torch.flip(torch.cumsum(torch.flip(mine.float(), [1]), 1), [1]).round().long()  # float cumsum: int64 cumsum fails on the XPU
    labels = torch.full((B, SLOTS), NONE, device=dev)
    for r in range(SLOTS):
        hit = mine & (from_right == r + 1)
        idx = hit.float().argmax(1)
        labels[:, r] = torch.where(hit.any(1), x.gather(1, idx[:, None])[:, 0], torch.full_like(idx, NONE))
    return x, q, labels


class Banks(torch.nn.Module):
    def __init__(self, nb: int, D: int, learn: bool):
        super().__init__()
        self.nb, self.D, self.learn = nb, D, learn
        gen = torch.Generator().manual_seed(1)
        self.register_buffer("E", torch.polar(torch.ones(V, D), torch.rand(V, D, generator=gen) * 2 * torch.pi))  # token values
        self.register_buffer("keys", torch.polar(torch.ones(nb, D), torch.rand(nb, D, generator=gen) * 2 * torch.pi))
        self.register_buffer("omega", 100.0 ** (-torch.arange(D) / D))
        S = torch.full((nb, V), EPS)
        for b in range(nb):
            S[b, b * PER_CAT:(b + 1) * PER_CAT] = 1.0
        self.register_buffer("S_fixed", S)
        self.prior_raw = torch.nn.Parameter(torch.zeros(nb, V)) if learn else None  # softplus(0) = 0.69 for every token

    def priors(self):
        return torch.nn.functional.softplus(self.prior_raw) if self.learn else self.S_fixed

    def forward(self, x, chunk: int = 16):
        S = self.priors()
        h = 0
        ex = self.E[x]  # (B, T, D)
        for b0 in range(0, self.nb, chunk):  # banks in chunks: (chunk, B, T, D) at a time
            s = S[b0:b0 + chunk][:, x]  # (c, B, T)
            tau = torch.cumsum(torch.clamp(s, max=1.0), dim=-1)  # the bank's clock: a selected token adds <= 1
            age = tau[..., -1:] - tau
            w = s * GAMMA ** age
            rot = torch.polar(torch.ones_like(age)[..., None].expand(*age.shape, self.D), age[..., None] * self.omega)
            m = (w[..., None] * rot * ex[None]).sum(-2) / (w.sum(-1, keepdim=True) + 1e-6)  # (c, B, D)
            h = h + (self.keys[b0:b0 + chunk, None, :] * m).sum(0)
        return h  # (B, D) complex


class Decoder(torch.nn.Module):
    def __init__(self, nb: int, D: int, hidden: int = 1024):
        super().__init__()
        self.bank = torch.nn.Embedding(nb, 64)
        self.net = torch.nn.Sequential(torch.nn.Linear(2 * D + 64, hidden), torch.nn.GELU(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, SLOTS * (V + 1)))

    def forward(self, h, q):
        return self.net(torch.cat([h.real, h.imag, self.bank(q)], -1)).view(-1, SLOTS, V + 1)


def run(nb: int, D: int, learn: bool):
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    banks, dec = Banks(nb, D, learn).to(dev), Decoder(nb, D).to(dev)
    opt = torch.optim.Adam(list(dec.parameters()) + list(banks.parameters()), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.steps, pct_start=0.1)
    t0, curve = time.time(), []
    for step in range(1, args.steps + 1):
        x, q, y = sample(args.batch, nb, g)
        loss = torch.nn.functional.cross_entropy(dec(banks(x), q).reshape(-1, V + 1), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % (args.steps // 4) == 0:
            curve.append(f"{loss.item():.2f}")
    with torch.no_grad():
        accs, all4, sel = [], [], ""
        for _ in range(8):  # 8 x 512 test windows
            x, q, y = sample(512, nb, g)
            active = y[:, 0] != NONE
            pred = dec(banks(x), q).argmax(-1)
            ok = pred == y
            accs.append(ok[active].float().mean(0))
            all4.append(ok[active].all(-1).float().mean())
        acc = torch.stack(accs).mean(0).tolist()
        a4 = torch.stack(all4).mean().item()
        if learn:
            P = banks.priors()
            own = torch.stack([P[b, b * PER_CAT:(b + 1) * PER_CAT].mean() for b in range(nb)]).mean().item()
            other = ((P.sum(1) - torch.stack([P[b, b * PER_CAT:(b + 1) * PER_CAT].sum() for b in range(nb)])) / (V - PER_CAT)).mean().item()
            sel = f"  learned priors: own category {own:.2f}, other tokens {other:.3f}"
    return acc, a4, curve, sel, time.time() - t0


print(f"device {dev}; window {T}, {ACTIVE} active categories per window, decode the queried bank's last {SLOTS} elements; "
      f"chance per slot ~{1 / PER_CAT:.2f} within the category, {1 / V:.4f} over the vocab")
print(f"{'priors':7s} {'banks':>5s} {'D':>4s} | {'slot1':>6s} {'slot2':>6s} {'slot3':>6s} {'slot4':>6s} | {'all 4':>6s} | loss curve (4 points)")
for D in (64, 128):
    for nb in (4, 16, 64):
        if D == 128 and nb != 64:
            continue
        for learn in (False, True):
            acc, a4, curve, sel, secs = run(nb, D, learn)
            print(f"{'learned' if learn else 'fixed':7s} {nb:5d} {D:4d} | " + " ".join(f"{v:6.3f}" for v in acc) +
                  f" | {a4:6.3f} | {' '.join(curve)}{sel}  ({secs:.0f}s)", flush=True)
