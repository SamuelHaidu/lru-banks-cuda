"""Does the bank layer's LRU state encode the sequence (order) or only a bag of words?

The state is linear in the inputs: h_t = sum_j lambda^j b_{t-j} (b = gamma B u, per complex channel). A bag of words
is the case lambda = 1 (every past token weighted the same, order lost). Swapping the tokens at offsets i < j changes
the state by (lambda^i - lambda^j)(b_x - b_y): non-zero in every channel where lambda^i != lambda^j, so with any decay
or rotation the state is order-sensitive. The question is how much order survives in practice (decays, float noise,
the key projection), measured three ways on a trained checkpoint and on a fresh layer:

  1. swaps: |s(swapped) - s| relative to |s(w) - s(w')| for an unrelated window w' (a bag gives exactly 0);
  2. permutations of a window after a reset: how close the nearest other permutation's state / key comes
     (a bag maps all k! orders to the same point);
  3. a linear probe (ridge, held out) from the state to the token at offset j. After a reset with a window of k
     distinct tokens, any order-blind (bag) feature scores at most 1/k at every offset: above 1/k is order.
     Mid-sequence (random letters with repeats, as in the copy task), with bag-of-last-W-tokens features as the
     baseline.

    uv run python "ad hoc scripts/lru_order_probe.py" [--ckpt runs/_synthetic/copy/bank_1layer_64slots_90s.pt]
"""

import argparse
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.archs.bank_memory import LRUBankLayer, lru_loop

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="runs/_synthetic/copy/bank_1layer_64slots_90s.pt")
p.add_argument("--n", type=int, default=20000, help="probe samples")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
torch.manual_seed(a.seed)


def load(path):
    """(name, embedding table, input fn, LRU layer, token ids to sample) from a bank_recall_demo / twohop checkpoint."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    pre = "layers.0." if "layers.0.nu" in sd else "blocks.0.bank."
    lay = LRUBankLayer(64, lru_dim=sd[pre + "nu"].shape[0], heads=2, key_dim=32, value_dim=32,
                       slots=sd[pre + "loga"].shape[1])
    lay.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)})
    emb = sd["emb.weight"]
    if pre.startswith("blocks"):  # two-hop: pre-norm LayerNorm before the layer, entities 2..
        ln = nn.LayerNorm(64)
        ln.load_state_dict({"weight": sd["blocks.0.n1.weight"], "bias": sd["blocks.0.n1.bias"]})
        return "trained (two-hop layer 0)", emb, ln, lay, torch.arange(2, emb.shape[0])
    return "trained (copy, 1 layer)", emb, (lambda x: x), lay, torch.arange(2, 34)  # copy letters 2..33


def fresh():
    lay = LRUBankLayer(64, lru_dim=32, heads=2, key_dim=32, value_dim=32, slots=64, lru_half_life=(0.5, 4.0))
    return "fresh init", nn.Embedding(129, 64).weight.detach(), (lambda x: x), lay, torch.arange(2, 34)


@torch.no_grad()
def states(model, ids, bag=False):
    """ids `(B, T)` (a reset before position 0) -> LRU state s `(B, T, 2N)` and the key the next token would get
    `(B, T, H, D)` (the key is built from the state before the token). bag: lambda = 1 (no decay, no rotation)."""
    _, emb, inp, lay, _ = model
    x = inp(emb[ids])
    reset = torch.zeros(ids.shape, dtype=torch.bool)
    reset[:, 0] = True
    log_lam = torch.zeros_like(lay.log_lam()) if bag else lay.log_lam()
    h, _ = lru_loop(lay.lru_input(x).to(torch.complex128), log_lam.to(torch.complex128), reset)
    s = torch.cat([h.real, h.imag], -1)
    k = F.normalize(lay.wk(s.float()).reshape(*s.shape[:-1], lay.H, lay.D), dim=-1)
    return s, k


def ridge_acc(X, y, classes, lam=1e-3):
    """Held-out accuracy of a linear (ridge to one-hot) probe; features standardized, last 20% held out."""
    X = X.double()
    n = int(0.8 * len(X))
    mu, sd = X[:n].mean(0), X[:n].std(0) + 1e-9
    X = torch.cat([(X - mu) / sd, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    Y = F.one_hot(y, classes).double()
    W = torch.linalg.solve(X[:n].T @ X[:n] + lam * n * torch.eye(X.shape[1], dtype=X.dtype), X[:n].T @ Y[:n])
    return float(((X[n:] @ W).argmax(1) == y[n:]).double().mean())


def report(model):
    name, emb, _, lay, toks = model
    A = len(toks)
    hl = math.log(2) / lay.nu.detach().exp()
    print(f"\n=== {name}: {A} tokens, LRU {lay.N} complex channels, half-lives (tokens) min {hl.min():.2f} "
          f"median {hl.median():.2f} max {hl.max():.2f}; |rotation| median {lay.theta.detach().remainder(2 * math.pi).sub(math.pi).abs().median():.2f} rad from pi")

    # 1. swaps of two tokens in a window of 8 after a reset
    k = 8
    w = toks[torch.rand(a.n // 10, A).argsort(1)[:, :k]]
    w2 = toks[torch.rand(a.n // 10, A).argsort(1)[:, :k]]
    s, _ = states(model, w)
    s2, _ = states(model, w2)
    sb, _ = states(model, w, bag=True)
    base = (s[:, -1] - s2[:, -1]).norm(dim=-1).mean()
    print("1. swap two tokens (offsets from the newest; |change| / |distance to an unrelated window|; a bag: 0)")
    for i, j in [(0, 1), (1, 2), (2, 3), (0, 4), (3, 4), (5, 6), (6, 7)]:
        ws = w.clone()
        ws[:, k - 1 - i], ws[:, k - 1 - j] = w[:, k - 1 - j], w[:, k - 1 - i]
        ss, _ = states(model, ws)
        sbs, _ = states(model, ws, bag=True)
        print(f"   offsets {i},{j}: {float((ss[:, -1] - s[:, -1]).norm(dim=-1).mean() / base):.3f}"
              f"   (same layer with lambda = 1: {float((sbs[:, -1] - sb[:, -1]).norm(dim=-1).max()):.1e})")

    # 2. all orders of a window of k distinct tokens after a reset
    print("2. all k! orders of k distinct tokens (after a reset): nearest OTHER order, relative state distance and "
          "key cosine (a bag: 0 and 1)")
    for k in (2, 3, 4, 5, 6):
        perms = torch.tensor(list(itertools.permutations(range(k))))
        w = toks[torch.rand(200, A).argsort(1)[:, :k]]
        allw = w[:, perms]  # (200, k!, k)
        s, key = states(model, allw.reshape(-1, k))
        s = s[:, -1].reshape(200, len(perms), -1)
        key = key[:, -1].reshape(200, len(perms), -1) / math.sqrt(lay.H)  # both heads: cosine of the concatenation
        d = torch.cdist(s, s) / s.norm(dim=-1, keepdim=True)
        d.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        c = key @ key.transpose(1, 2)
        c.diagonal(dim1=1, dim2=2).fill_(-1)
        dm, cm = d.min(-1).values.flatten(), c.max(-1).values.flatten()
        print(f"   k={k} ({len(perms):3d} orders): state distance median {float(dm.median()):.3f}, 1% {float(dm.quantile(0.01)):.3f}"
              f";  key cosine median {float(cm.median()):.3f}, 99% {float(cm.quantile(0.99)):.3f}")

    # 3a. probe: token at offset j after a reset, k distinct tokens (bag ceiling exactly 1/k)
    print("3a. linear probe, token at offset j from the state after k distinct tokens (bag ceiling = 1/k)")
    for k in (4, 8):
        w = toks[torch.rand(a.n, A).argsort(1)[:, :k]]
        s, key = states(model, w)
        y = w - toks[0]
        accs = [ridge_acc(s[:, -1], y[:, k - 1 - j], A) for j in range(k)]
        kacc = [ridge_acc(key[:, -1].flatten(1), y[:, k - 1 - j], A) for j in range(k)]
        print(f"   k={k}, bag ceiling {1 / k:.3f}:  state " + " ".join(f"{x:.3f}" for x in accs))
        print(f"   {'':22s}key   " + " ".join(f"{x:.3f}" for x in kacc))

    # 3b. probe mid-sequence: random letters with repeats (the copy task's strings), positions 24..63
    print(f"3b. linear probe mid-sequence (random tokens with repeats, chance {1 / A:.3f}): offset j = 0..9")
    T = 64
    w = toks[torch.randint(0, A, (a.n // 40 + 1, T))]
    s, key = states(model, w)
    pos = torch.arange(24, T)
    S = s[:, pos].reshape(-1, s.shape[-1])
    Kf = key[:, pos].flatten(2).reshape(-1, lay.H * lay.D)
    oh = F.one_hot(w - toks[0], A).double()
    rows = {"state": S, "key": Kf}
    for W in (4, 8):  # bag of the last W tokens: counts, order-blind
        cs = torch.cat([torch.zeros_like(oh[:, :1]), oh.cumsum(1)], 1)
        rows[f"bag of last {W}"] = (cs[:, pos + 1] - cs[:, pos + 1 - W]).reshape(-1, A)
    for nm, X in rows.items():
        accs = [ridge_acc(X, (w[:, pos - j] - toks[0]).reshape(-1), A) for j in range(10)]
        print(f"   {nm:14s} " + " ".join(f"{x:.3f}" for x in accs))


for m in (load(a.ckpt), fresh()):
    report(m)
