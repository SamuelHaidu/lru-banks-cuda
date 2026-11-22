"""Empirical checks of the token-prior / forgetting / saturation kernel idea.

    weight(i, j) = s_t * delta_t(n_j) * a * k^(-p * lambda_t)       (log-mix on a shared power law)
    h_i          = R_{i+1} sum_j weight(i, j) R_{-j} e_j            (RoPE, as in rope-logit-sum)
                   optionally divided by sum_j weight(i, j)         (normalized)

    t = token at j, k = i - j + 1, n_j = which mention of t it is, delta(n) = g(n) - g(n-1),
    g(n) = n (1 + kappa_t) / (n + kappa_t)  (BM25 saturation)

Part 1 checks the math (exact identities and guarantees).
Part 2 trains small models on a synthetic language whose right answer is known:

    - a protagonist NAME is mentioned early; a cue token Q later asks for it (long-range recall)
    - function words TO_i are always followed by VERB_i (purely local: only the last token matters)
    - everything else is NOISE (unpredictable filler)
    - some sequences repeat the name many times (tests saturation)

and checks what each model learns: does the name get lambda ~ 0 (kept), do filler and function
words keep lambda ~ 1 (forget), does recall stay good at long distance, does saturation matter.

Run:  .venv/bin/python studies/token_kernel_empirical.py   (CPU, a few minutes)
"""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(max(1, torch.get_num_threads()))
DEV = "cpu"

# ----------------------------------------------------------------------------- synthetic language
N_NAME, N_TO, N_NOISE = 8, 8, 24
PAD, Q = 0, 1
NAME0 = 2
TO0 = NAME0 + N_NAME
VERB0 = TO0 + N_TO
NOISE0 = VERB0 + N_TO
V = NOISE0 + N_NOISE  # 50
L, T = 256, 128  # sequence length, kernel window


def category(tok):
    if tok >= NOISE0:
        return "noise"
    if tok >= VERB0:
        return "verb"
    if tok >= TO0:
        return "to"
    if tok >= NAME0:
        return "name"
    return "Q" if tok == Q else "pad"


def make_batch(B, rng, repeat_frac=0.4):
    """Returns tokens (B, L) and per-target labels: kind (B, L) in {0 other, 1 recall, 2 verb},
    recall distance (B, L) = distance from the cue to the latest name mention, and the name count."""
    x = rng.integers(NOISE0, V, size=(B, L))
    kind = np.zeros((B, L), np.int64)
    rdist = np.zeros((B, L), np.int64)
    n_mentions = np.zeros(B, np.int64)
    for b in range(B):
        name = NAME0 + rng.integers(N_NAME)
        first = rng.integers(0, 60)
        x[b, first] = name
        mentions = [first]
        if rng.random() < repeat_frac:  # many-mention sequence
            for p in rng.choice(np.arange(first + 1, L - 1), size=rng.integers(4, 15), replace=False):
                x[b, p] = name
                mentions.append(int(p))
        p = first + 2
        while p < L - 1:
            r = rng.random()
            if r < 0.08 and x[b, p] != name and x[b, p + 1] != name:  # function word -> its verb
                i = rng.integers(N_TO)
                x[b, p], x[b, p + 1] = TO0 + i, VERB0 + i
                kind[b, p + 1] = 2
                p += 2
                continue
            if r < 0.12 and x[b, p] != name:  # cue -> name (recall), if the name is in the window
                last = max(m for m in mentions if m < p)
                d = p - last
                if d < T - 2:
                    x[b, p], x[b, p + 1] = Q, name
                    kind[b, p + 1] = 1
                    rdist[b, p + 1] = d
                    mentions.append(p + 1)
                    p += 2
                    continue
            p += 1
        n_mentions[b] = len(mentions)
    return torch.tensor(x), torch.tensor(kind), torch.tensor(rdist), torch.tensor(n_mentions)


# ----------------------------------------------------------------------------- model
def rope_tables(D, positions, base=10000.0):
    half = D // 2
    inv = 1.0 / (base ** (torch.arange(half, dtype=torch.float64) / half))
    ang = positions.double()[:, None] * inv[None]
    return torch.cos(ang).float(), torch.sin(ang).float()


def rotate(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def bm25(n, kappa):
    return n * (1 + kappa) / (n + kappa)


def mention_index(x):
    """n_j: which mention of its token x[:, j] is (1-based), counting from the sequence start."""
    eq = (x[:, :, None] == x[:, None, :])  # (B, L, L): [j, j'] same token
    causal = torch.tril(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool))
    return (eq & causal).sum(-1)  # (B, L)


class KernelLM(torch.nn.Module):
    """variant: 'free' (per-distance weights), 'shared' (a k^-p), 'prior' (+ s_t),
    'forget' (+ lambda_t), 'saturate' (+ kappa_t)."""

    def __init__(self, variant, D=16, normalize=False, lam_init=0.95):
        super().__init__()
        self.variant, self.normalize = variant, normalize
        self.emb = torch.nn.Embedding(V, D)
        self.out = torch.nn.Linear(D, V)
        torch.nn.init.normal_(self.emb.weight, std=0.5)
        if variant == "free":
            self.w_free = torch.nn.Parameter(1.0 / torch.arange(1, T + 1).float())
        self.log_a = torch.nn.Parameter(torch.zeros(()))
        self.p_raw = torch.nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))  # softplus^-1(1)
        self.s_raw = torch.nn.Parameter(torch.full((V,), math.log(math.expm1(1.0))))
        self.lam_raw = torch.nn.Parameter(torch.full((V,), math.log(lam_init / (1 - lam_init))))
        self.kap_raw = torch.nn.Parameter(torch.full((V,), math.log(math.expm1(20.0))))  # ~additive at init
        pos = torch.arange(L)
        self.register_buffer("k", (pos[:, None] - pos[None, :] + 1).float())  # (L, L) distance
        self.register_buffer("mask", (self.k >= 1) & (self.k <= T))
        c, s = rope_tables(D, pos)
        cq, sq = rope_tables(D, pos + 1)
        self.register_buffer("cu", c); self.register_buffer("su", -s)
        self.register_buffer("cq", cq); self.register_buffer("sq", sq)

    # learned per-token numbers
    def s(self): return F.softplus(self.s_raw)
    def lam(self): return torch.sigmoid(self.lam_raw)
    def kappa(self): return F.softplus(self.kap_raw)
    def p(self): return F.softplus(self.p_raw)

    def weights(self, x):
        """(B, L, L) weight of context position j for query i (0 outside the causal window)."""
        logk = torch.log(self.k.clamp_min(1.0))
        if self.variant == "free":
            idx = (self.k.long() - 1).clamp(0, T - 1)
            W = self.w_free[idx][None].expand(x.shape[0], -1, -1)
        else:
            lam = self.lam()[x] if self.variant in ("forget", "saturate") else torch.ones_like(x, dtype=torch.float)
            W = torch.exp(self.log_a - self.p() * lam[:, None, :] * logk[None])  # a k^(-p lambda_t)
            if self.variant != "shared":
                W = W * self.s()[x][:, None, :]
            if self.variant == "saturate":
                n = mention_index(x).float()
                kap = self.kappa()[x]
                W = W * (bm25(n, kap) - bm25(n - 1, kap))[:, None, :]
        return W * self.mask[None]

    def hidden(self, x):
        W = self.weights(x)
        u = rotate(self.emb(x), self.cu, self.su)  # R_{-j} e_j
        h = rotate(W @ u, self.cq, self.sq)  # R_{i+1} sum_j W u_j
        if self.normalize:
            h = h / (W.sum(-1, keepdim=True) + 1e-6)
        return h

    def forward(self, x):
        return self.out(self.hidden(x))


def evaluate(model, data):
    x, kind, rdist, nm = data
    with torch.no_grad():
        pred = model(x)[:, :-1].argmax(-1)
    tgt, kd, rd = x[:, 1:], kind[:, 1:], rdist[:, 1:]
    ok = pred == tgt
    rep = (nm >= 5)[:, None].expand_as(kd)
    res = {"verb": ok[kd == 2].float().mean().item()}
    for lo, hi in ((1, 16), (17, 48), (49, 96), (97, 127)):
        sel = (kd == 1) & (rd >= lo) & (rd <= hi)
        res[f"recall d{lo}-{hi}"] = ok[sel].float().mean().item() if sel.any() else float("nan")
    res["recall (1 mention)"] = ok[(kd == 1) & ~rep].float().mean().item()
    res["recall (5+ mentions)"] = ok[(kd == 1) & rep].float().mean().item()
    res["verb (5+ mentions)"] = ok[(kd == 2) & rep].float().mean().item()
    return res


def train(variant, normalize=False, steps=1200, B=32, seed=0, lr=0.03):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = KernelLM(variant, normalize=normalize)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for step in range(steps):
        x, *_ = make_batch(B, rng)
        logits = m(x)[:, :-1]
        loss = F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m


# ----------------------------------------------------------------------------- part 1: math
def part1():
    print("=" * 78)
    print("PART 1: mathematical properties (exact checks)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    results = []

    def check(name, cond, detail=""):
        results.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    k = np.arange(1, 513, dtype=np.float64)
    p = 0.84
    # 1. monotone non-increasing for every lambda in [0, 1], both mixes
    lam = rng.random(1000)
    env = k ** -p
    logmix = env[None] ** lam[:, None]
    linmix = (1 - lam[:, None]) + lam[:, None] * env[None]
    check("log-mix never grows with distance (1000 random lambda)", bool((np.diff(logmix, axis=1) <= 1e-15).all()))
    check("linear mix never grows with distance", bool((np.diff(linmix, axis=1) <= 1e-15).all()))
    check("peak is at k = 1 and equals 1 (so the prior s is the maximum)", bool(np.allclose(logmix[:, 0], 1) and np.allclose(linmix[:, 0], 1)))
    # 2. limits
    check("lambda = 0 -> exactly constant", bool(np.allclose(env ** 0.0, 1.0)))
    check("lambda = 1 -> exactly the shared kernel", bool(np.allclose(env ** 1.0, env)))
    # 3. saturation bound
    for kap in (0.1, 0.5, 2.0, 10.0):
        n = np.arange(1, 10001, dtype=np.float64)
        d = bm25(n, kap) - bm25(n - 1, kap)
        tot = np.cumsum(d)
        check(f"BM25 kappa={kap}: n mentions sum to g(n) <= 1+kappa, each mention <= the previous",
              bool(tot.max() <= 1 + kap + 1e-9 and (np.diff(d) <= 1e-12).all() and abs(d[0] - 1) < 1e-12),
              f"g(10)={tot[9]:.3f}  g(10000)={tot[-1]:.3f}")
    # 4. parallel (matrix) form == sequential loop
    m = KernelLM("saturate", normalize=True)
    with torch.no_grad():
        m.lam_raw.normal_(); m.s_raw.normal_(); m.kap_raw.normal_()
    x, *_ = make_batch(2, rng)
    with torch.no_grad():
        h_par = m.hidden(x)
        e = m.emb(x)
        s, lam_t, kap = m.s()[x], m.lam()[x], m.kappa()[x]
        n = mention_index(x).float()
        a, pp = torch.exp(m.log_a), m.p()
        D = e.shape[-1]
        h_seq = torch.zeros_like(h_par)
        for b in range(2):
            for i in (5, 77, 200, 255):
                acc, z = torch.zeros(D), 0.0
                for j in range(max(0, i - T + 1), i + 1):
                    kk = i - j + 1
                    w = s[b, j] * (bm25(n[b, j], kap[b, j]) - bm25(n[b, j] - 1, kap[b, j])) * a * kk ** (-pp * lam_t[b, j])
                    c, sn = rope_tables(D, torch.tensor([kk]))  # rotate e_j by its distance directly
                    acc = acc + w * rotate(e[b, j][None], c, sn)[0]
                    z = z + w
                h_seq[b, i] = acc / (z + 1e-6)
        idx = [5, 77, 200, 255]
        err = (h_par[:, idx] - h_seq[:, idx]).abs().max().item()
    check("parallel matrix form == token-by-token loop (with RoPE, saturation, normalization)", err < 1e-4, f"max err {err:.2e}")
    # 5. linear-mix decomposition: cumulative sum + shared kernel convolution
    S_ = 300
    e = rng.normal(size=(S_, 4)); s_t = rng.random(S_) + 0.5; l_t = rng.random(S_)
    direct = np.zeros((S_, 4)); split = np.zeros((S_, 4))
    for i in range(S_):
        j = np.arange(max(0, i - T + 1), i + 1); kk = i - j + 1
        wsh = kk ** -p
        direct[i] = ((s_t[j] * ((1 - l_t[j]) + l_t[j] * wsh))[:, None] * e[j]).sum(0)
        split[i] = ((s_t[j] * (1 - l_t[j]))[:, None] * e[j]).sum(0) + ((s_t[j] * l_t[j] * wsh)[:, None] * e[j]).sum(0)
    check("linear mix == windowed cumulative sum + shared-kernel conv", bool(np.allclose(direct, split)))
    # 6. what the family cannot do (by design)
    print("  [INFO] cannot express: weight rising with distance (delayed effect), or forgetting FASTER than the")
    print("         shared kernel (lambda <= 1). The shared kernel is the fastest decay any token can have.")
    print(f"  -> {sum(results)}/{len(results)} checks passed\n")


# ----------------------------------------------------------------------------- part 2: learning
def part2():
    print("=" * 78)
    print("PART 2: what models learn on the synthetic language")
    print("=" * 78)
    print(f"  vocab {V}: {N_NAME} names, {N_TO} function words TO_i -> VERB_i, {N_NOISE} noise, cue Q -> name")
    print(f"  seq {L}, window {T}, dim 16, 1200 steps x 32 sequences, Adam lr 0.03, 40% of sequences repeat the name 5-15x\n")
    rng = np.random.default_rng(123)
    test = make_batch(512, rng)
    configs = [("free", False), ("shared", False), ("prior", False), ("forget", False),
               ("forget", True), ("saturate", True)]
    models, rows = {}, []
    for variant, norm in configs:
        t0 = time.time()
        m = train(variant, norm)
        name = variant + (" +norm" if norm else "")
        models[name] = m
        r = evaluate(m, test)
        rows.append((name, r))
        print(f"  trained {name:16s} in {time.time() - t0:5.1f}s")

    keys = list(rows[0][1].keys())
    print("\n  accuracy on test sequences (argmax == target)")
    print("  " + "model".ljust(17) + "".join(k[:14].rjust(15) for k in keys))
    for name, r in rows:
        print("  " + name.ljust(17) + "".join(f"{r[k]:15.3f}" for k in keys))

    print("\n  learned per-token numbers (mean over each token group)")
    groups = {"name": range(NAME0, TO0), "to (local)": range(TO0, VERB0), "verb": range(VERB0, NOISE0),
              "noise": range(NOISE0, V), "Q": [Q]}
    for name in ("forget", "forget +norm", "saturate +norm"):
        m = models[name]
        with torch.no_grad():
            s, lam, kap = m.s().numpy(), m.lam().numpy(), m.kappa().numpy()
        print(f"  {name}: shared exponent p = {m.p().item():.2f}")
        for g, idx in groups.items():
            idx = list(idx)
            line = f"    {g:11s} s={s[idx].mean():6.2f}  lambda={lam[idx].mean():5.2f}  -> decay k^-{m.p().item() * lam[idx].mean():.2f}"
            if name.startswith("saturate"):
                line += f"  kappa={kap[idx].mean():6.2f}"
            print(line)
    return rows, models


if __name__ == "__main__":
    part1()
    part2()
