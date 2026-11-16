"""Synthetic test: can a model learn to BIND a token to a context-made key and UNBIND it later?

Task (needs role -> filler binding, not just "which tokens were present"):

    ... noise  FRIEND Tom  noise ...  PET cat  noise ...  TOY ball ... PLACE park ...
    ... noise  ASK_PET  -> cat   noise ...  ASK_FRIEND -> Tom  ...

- 4 roles, each followed once by a random filler (40 possible fillers, distinct per sequence);
- 6 distractor fillers per sequence appear as noise, so knowing *which fillers occurred* is
  not enough (about 10% accuracy);
- queries ASK_r come 10-159 tokens later; the target is the filler bound to role r.

Context for every write/read network: c_t = [e_t ; e_{t-1}]. Value v_t = W_v c_t (starts as
e_t), read r_t projected by W_r (zero-init) and added to h before the readout.

Baselines: `local` (last two tokens) and `bank+mlp` (normalized per-token prior/forgetting
kernel state + residual MLP: our current model family).

Memory variants (keys are unit phasors, 32 complex numbers; bind = multiply, unbind = conj):
    free        write key = exp(i MLP_bind(c)), read key = exp(i MLP_unbind(c))
    codebook    K shared role keys; binder / unbinder output a distribution over them
                (softmax, 1.5-entmax, sparsemax or straight-through Gumbel on the write side);
                key = phase-normalized mix of the chosen role keys
    slots       K separate memory vectors; write / read weights are the same distributions
                (no binding at all: zero crosstalk, K x D state)
    fpe         fractional power encoding: binder / unbinder output a role coordinate z (1-D
                or 2-D), key = exp(i z . Omega) with fixed random frequencies Omega

Reports query accuracy by distance, and for codebook / slot models: how many roles a fact was
written to (effective count), whether each role always uses the same slot across sequences,
and whether the read picks the slot the fact was written to.

Run: .venv/bin/python studies/assoc_memory_synthetic.py      (CPU, ~10 minutes)
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(max(1, torch.get_num_threads()))

R, NF, NN = 4, 40, 30  # roles, fillers, noise tokens
ROLE0 = 1
ASK0 = ROLE0 + R
FILL0 = ASK0 + R
NOISE0 = FILL0 + NF
V = NOISE0 + NN
L = 160
D = 64
HALF = D // 2


def make_batch(B, rng):
    x = rng.integers(NOISE0, V, size=(B, L))
    is_q = np.zeros((B, L), bool)
    dist = np.zeros((B, L), np.int64)
    for b in range(B):
        for p in rng.choice(L, size=6, replace=False):  # distractor fillers
            x[b, p] = FILL0 + rng.integers(NF)
        fillers = rng.choice(NF, size=R, replace=False)
        pos = np.sort(rng.choice(np.arange(1, 70, 2), size=R, replace=False))
        where = {}
        for r, p in zip(rng.permutation(R), pos):
            x[b, p], x[b, p + 1] = ROLE0 + r, FILL0 + fillers[r]
            where[r] = p + 1
        p = 80
        while p < L - 1:
            if rng.random() < 0.12:
                r = rng.integers(R)
                x[b, p], x[b, p + 1] = ASK0 + r, FILL0 + fillers[r]
                is_q[b, p + 1] = True
                dist[b, p + 1] = p + 1 - where[r]
                p += 2
            else:
                p += 1
    return torch.tensor(x), torch.tensor(is_q), torch.tensor(dist)


# ---------------------------------------------------------------- sparse distributions
def sparsemax(z):
    zs, _ = torch.sort(z, dim=-1, descending=True)
    k = torch.arange(1, z.shape[-1] + 1, dtype=z.dtype)
    cs = zs.cumsum(-1)
    support = (1 + k * zs) > cs
    kz = support.sum(-1, keepdim=True)
    tau = (cs.gather(-1, kz - 1) - 1) / kz
    return torch.clamp(z - tau, min=0)


def entmax15(z):
    z = z / 2
    zs, _ = torch.sort(z, dim=-1, descending=True)
    k = torch.arange(1, z.shape[-1] + 1, dtype=z.dtype)
    mean = zs.cumsum(-1) / k
    mean_sq = (zs ** 2).cumsum(-1) / k
    ss = k * (mean_sq - mean ** 2)
    delta = torch.clamp((1 - ss) / k, min=0)
    tau = mean - torch.sqrt(delta)
    support = (tau <= zs).sum(-1, keepdim=True).clamp_min(1)  # float rounding can leave it empty
    tau_star = tau.gather(-1, support - 1)
    return torch.clamp(z - tau_star, min=0) ** 2


def gumbel_st(z, training):
    if not training:
        return F.one_hot(z.argmax(-1), z.shape[-1]).to(z.dtype)
    g = -torch.log(-torch.log(torch.rand_like(z).clamp(1e-9, 1 - 1e-9)))
    soft = F.softmax(z + g, -1)
    hard = F.one_hot(soft.argmax(-1), z.shape[-1]).to(z.dtype)
    return hard - soft.detach() + soft


def distribution(kind, z, training):
    return {"softmax": lambda: F.softmax(z, -1), "entmax15": lambda: entmax15(z), "sparsemax": lambda: sparsemax(z),
            "gumbel": lambda: gumbel_st(z, training)}[kind]()


# ---------------------------------------------------------------- model
def mlp(i, h, o):
    return torch.nn.Sequential(torch.nn.Linear(i, h), torch.nn.GELU(), torch.nn.Linear(h, o))


class Model(torch.nn.Module):
    def __init__(self, memory=None, K=8, write="softmax", read="softmax", fpe_dim=1, bank=False, use_mlp=False):
        super().__init__()
        self.memory, self.K, self.write, self.read_kind, self.bank, self.use_mlp = memory, K, write, read, bank, use_mlp
        self.emb = torch.nn.Embedding(V, D)
        torch.nn.init.normal_(self.emb.weight, std=0.5)
        self.local = torch.nn.Linear(2 * D, D)
        self.out = torch.nn.Linear(D, V)
        pos = torch.arange(L)
        k = (pos[:, None] - pos[None, :] + 1).float()
        self.register_buffer("logk", torch.log(k.clamp_min(1)))
        self.register_buffer("causal", (k >= 1).float())
        inv = 1.0 / (10000 ** (torch.arange(HALF).float() / HALF))
        self.register_buffer("ang", pos[:, None].float() * inv[None])
        if bank:
            self.s_raw = torch.nn.Parameter(torch.zeros(V))
            self.lam_raw = torch.nn.Parameter(torch.full((V,), 2.0))
            self.p_raw = torch.nn.Parameter(torch.tensor(0.5))
            self.bank_out = torch.nn.Linear(D, D)
        if memory:
            out_dim = {"free": HALF, "codebook": K, "slots": K, "fpe": fpe_dim}[memory]
            self.bind = mlp(2 * D, 128, out_dim)
            self.unbind = mlp(2 * D, 128, out_dim)
            self.gate = mlp(2 * D, 64, 1)
            torch.nn.init.constant_(self.gate[-1].bias, -2.0)
            self.value = torch.nn.Linear(2 * D, D, bias=False)
            with torch.no_grad():
                self.value.weight.zero_(); self.value.weight[:, :D] = torch.eye(D)
            self.readp = torch.nn.Linear(D, D)
            torch.nn.init.zeros_(self.readp.weight); torch.nn.init.zeros_(self.readp.bias)
            if memory == "codebook":
                self.role_angles = torch.nn.Parameter(torch.rand(K, HALF) * 2 * np.pi)
            if memory == "fpe":
                self.register_buffer("omega", torch.randn(fpe_dim, HALF) * 3.0)
        if use_mlp:
            self.ff = mlp(D, 256, D)
            torch.nn.init.zeros_(self.ff[-1].weight); torch.nn.init.zeros_(self.ff[-1].bias)

    def _codebook_key(self, p):
        """Phase-normalized mix of role keys: p (B, L, K) -> (cos, sin) of the key phases."""
        cr = p @ torch.cos(self.role_angles)
        ci = p @ torch.sin(self.role_angles)
        n = torch.sqrt(cr ** 2 + ci ** 2) + 1e-6
        return cr / n, ci / n

    def forward(self, x, trace=False):
        e = self.emb(x)
        prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], 1)
        c = torch.cat([e, prev], -1)
        h = self.local(c)
        info = {}
        if self.bank:
            lam = torch.sigmoid(self.lam_raw)[x]
            w = F.softplus(self.s_raw)[x][:, None, :] * torch.exp(-F.softplus(self.p_raw) * lam[:, None, :] * self.logk) * self.causal
            zr, zi = e[..., :HALF], e[..., HALF:]
            c_, s_ = torch.cos(self.ang), torch.sin(self.ang)
            ur, ui = zr * c_ + zi * s_, zi * c_ - zr * s_
            sr, si = w @ ur, w @ ui
            hb = torch.cat([sr * c_ - si * s_, sr * s_ + si * c_], -1) / (w.sum(-1, keepdim=True) + 1e-6)
            h = h + self.bank_out(hb)
        if self.memory:
            g = torch.sigmoid(self.gate(c))
            v = self.value(c)
            zb, zu = self.bind(c), self.unbind(c)
            if self.memory == "slots":
                p = distribution(self.write, zb, self.training)
                a = distribution(self.read_kind, zu, self.training)
                M = torch.cumsum((g * p)[..., None] * v[:, :, None, :], 1)  # (B, L, K, D)
                r = (a[..., None] * M).sum(2)
                info = {"p": p, "a": a, "g": g}
            else:
                if self.memory == "free":
                    kr, ki, qr, qi = torch.cos(zb), torch.sin(zb), torch.cos(zu), torch.sin(zu)
                elif self.memory == "fpe":
                    kr, ki = torch.cos(zb @ self.omega), torch.sin(zb @ self.omega)
                    qr, qi = torch.cos(zu @ self.omega), torch.sin(zu @ self.omega)
                else:  # codebook
                    p = distribution(self.write, zb, self.training)
                    a = distribution(self.read_kind, zu, self.training)
                    (kr, ki), (qr, qi) = self._codebook_key(p), self._codebook_key(a)
                    info = {"p": p, "a": a, "g": g}
                vr, vi = v[..., :HALF], v[..., HALF:]
                Mr = torch.cumsum(g * (kr * vr - ki * vi), 1)
                Mi = torch.cumsum(g * (kr * vi + ki * vr), 1)
                r = torch.cat([qr * Mr + qi * Mi, qr * Mi - qi * Mr], -1)
                info.setdefault("g", g)
                info.update(kc=(kr, ki), qc=(qr, qi))
            h = h + self.readp(r)
        if self.use_mlp:
            h = h + self.ff(h)
        return (self.out(h), info) if trace else self.out(h)


def train(model, steps, rng, lr=3e-3, B=32):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        x, _, _ = make_batch(B, rng)
        loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def evaluate(model, data):
    x, is_q, dist = data
    with torch.no_grad():
        pred = model(x)[:, :-1].argmax(-1)
    ok = pred == x[:, 1:]
    q, d = is_q[:, 1:], dist[:, 1:]
    res = {"acc": ok[q].float().mean().item()}
    for lo, hi in ((10, 60), (61, 110), (111, 160)):
        sel = q & (d >= lo) & (d <= hi)
        res[f"d{lo}-{hi}"] = ok[sel].float().mean().item() if sel.any() else float("nan")
    return res


def role_stats(model, data):
    """For codebook / slot models: write sparsity, role -> slot consistency, read hits the written slot."""
    x, _, _ = data
    with torch.no_grad():
        _, info = model(x, trace=True)
    if "p" not in info:
        return None
    p, a = info["p"], info["a"]
    eff_w, slot_of = [], {r: [] for r in range(R)}
    hit, eff_r = [], []
    for b in range(x.shape[0]):
        filler_slot = {}
        for t in range(1, L):
            if ROLE0 <= x[b, t - 1] < ASK0:
                r = int(x[b, t - 1]) - ROLE0
                eff_w.append(1.0 / float((p[b, t] ** 2).sum()))
                filler_slot[r] = int(p[b, t].argmax())
                slot_of[r].append(filler_slot[r])
        for t in range(L):
            if ASK0 <= x[b, t] < FILL0:
                r = int(x[b, t]) - ASK0
                eff_r.append(1.0 / float((a[b, t] ** 2).sum()))
                hit.append(int(a[b, t].argmax()) == filler_slot.get(r, -1))
    consistency = np.mean([np.bincount(v).max() / len(v) for v in slot_of.values()])
    distinct = len({int(np.bincount(v).argmax()) for v in slot_of.values()})
    return {"write roles/fact": np.mean(eff_w), "read roles/query": np.mean(eff_r), "role->slot consistency": consistency,
            "distinct slots for 4 roles": distinct, "read hits written slot": np.mean(hit)}


def main():
    test = make_batch(400, np.random.default_rng(99))
    print(f"vocab {V}: {R} roles, {R} ask tokens, {NF} fillers, {NN} noise; seq {L}; dim {D} ({HALF} complex)")
    print("facts at positions 1-70, queries at 80-160 (distance 10-159); presence-only guessing ~0.10\n")
    configs = [
        ("local", dict()),
        ("bank+mlp", dict(bank=True, use_mlp=True)),
        ("free keys", dict(memory="free")),
        ("codebook8 softmax", dict(memory="codebook", K=8, write="softmax")),
        ("codebook8 entmax1.5", dict(memory="codebook", K=8, write="entmax15")),
        ("codebook8 sparsemax", dict(memory="codebook", K=8, write="sparsemax")),
        ("codebook8 gumbel-ST", dict(memory="codebook", K=8, write="gumbel")),
        ("codebook16 entmax1.5", dict(memory="codebook", K=16, write="entmax15")),
        ("slots8 softmax", dict(memory="slots", K=8, write="softmax")),
        ("slots8 entmax1.5", dict(memory="slots", K=8, write="entmax15")),
        ("fpe 1-D", dict(memory="fpe", fpe_dim=1)),
        ("fpe 2-D", dict(memory="fpe", fpe_dim=2)),
    ]
    import sys
    only = sys.argv[1:]  # optional: run only configs whose name contains one of these words
    if only:
        configs = [c for c in configs if any(w in c[0] for w in only)]
    rows = []
    for name, kw in configs:
        torch.manual_seed(0)
        t0 = time.time()
        m = train(Model(**kw), 2000, np.random.default_rng(1))
        rows.append((name, evaluate(m, test), role_stats(m, test)))
        print(f"  trained {name:22s} {time.time() - t0:5.1f}s   acc {rows[-1][1]['acc']:.3f}", flush=True)
    keys = list(rows[0][1])
    print("\n  query accuracy")
    print("  " + "model".ljust(24) + "".join(k.rjust(11) for k in keys))
    for name, r, _ in rows:
        print("  " + name.ljust(24) + "".join(f"{r[k]:11.3f}" for k in keys))
    print("\n  role usage (codebook / slots)")
    skeys = None
    for name, _, st in rows:
        if st is None:
            continue
        if skeys is None:
            skeys = list(st)
            print("  " + "model".ljust(24) + "".join(k.rjust(24) for k in skeys))
        print("  " + name.ljust(24) + "".join(f"{st[k]:24.3f}" for k in skeys))


if __name__ == "__main__":
    main()
