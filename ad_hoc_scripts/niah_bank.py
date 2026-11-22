"""Needle in a haystack for the bank layer (fast, CPU). A haystack of random filler tokens (their own vocabulary) with
needles "KEY k v" at random depths, then questions "ASK k" -> v.

  --task single   one needle per chain:      KEY k v                       ASK k -> v
  --task twohop   two needles per chain:     KEY k1 k2 ... KEY k2 v        ASK k1 -> v   (two lookups)

Several chains per row (the other chains are distractor needles: guessing among them is the floor); with --hide-ends
(default on for twohop) every chain also gets a never-asked needle "KEY v x", so the answer also appears as a key and
"the token never seen as a key" is not a shortcut. Trained on short haystacks (--train-len), evaluated by haystack
length (also far longer than trained) and by the depth of the asked chain's first needle. One bank layer with --hops
reads (`LRUBankLayer`: read, FFN, query from the stream, second read of the same banks, the FFN again).

    uv run python ad_hoc_scripts/niah_bank.py [--task twohop] [--hops 2] [--steps 800] [--save PATH]
"""

import argparse
import math
import time

p = argparse.ArgumentParser()
p.add_argument("--task", default="twohop", choices=["single", "twohop"])
p.add_argument("--hops", type=int, default=2, help="reads per bank layer")
p.add_argument("--layers", type=int, default=1, help="bank layers (each its own FFN, reused by its hops)")
p.add_argument("--gate-second", action="store_true", help="LRU input gate (selective decay from the layer's input, "
               "i.e. the previous layer's output) in every layer after the first")
p.add_argument("--chains", type=int, default=4, help="chains (needle groups) per row; the others are distractors")
p.add_argument("--queries", type=int, default=2, help="questions per row (distinct chains)")
p.add_argument("--hide-ends", type=int, default=None, help="add 'KEY v x' per chain (default: 1 for twohop)")
p.add_argument("--entities", type=int, default=64, help="key / value tokens")
p.add_argument("--noise", type=int, default=64, help="filler tokens (never keys or values)")
p.add_argument("--train-len", nargs=2, type=int, default=[16, 96], help="haystack lengths trained (min max)")
p.add_argument("--eval-lens", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048])
p.add_argument("--slots", type=int, default=64)
p.add_argument("--steps", type=int, default=800)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=6e-3)
p.add_argument("--log-every", type=int, default=5)
p.add_argument("--eval-every", type=int, default=200)
p.add_argument("--save", default=None)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
if a.hide_ends is None:
    a.hide_ends = 1 if a.task == "twohop" else 0

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from model.archs.bank_memory import LRUBankLayer  # noqa: E402

torch.manual_seed(a.seed)
BOS, ASK, KEY = 0, 1, 2
ENT = 3  # entities 3 .. 3 + entities - 1
NOISE = ENT + a.entities
V = NOISE + a.noise
PER = (2 if a.task == "twohop" else 1) + a.hide_ends  # needles per chain


def make_rows(n, L, g=None):
    """n rows with a haystack of L filler tokens. Returns x, y, scored mask, depth of each scored question's chain
    (relative position of its first needle in the haystack, 0 = start)."""
    C, Q = a.chains, a.queries
    ne = 3 if a.task == "twohop" else 2  # entities per chain: k1 k2 v / k v (+ x when hiding ends)
    ne += a.hide_ends
    ents = torch.rand(n, a.entities, generator=g).argsort(1)[:, :C * ne].reshape(n, C, ne) + ENT
    if a.task == "twohop":
        needles = [ents[:, :, [0, 1]], ents[:, :, [1, 2]]]  # k1 k2, k2 v
        if a.hide_ends:
            needles.append(ents[:, :, [2, 3]])  # v x
    else:
        needles = [ents[:, :, [0, 1]]] + ([ents[:, :, [1, 2]]] if a.hide_ends else [])
    needles = torch.stack(needles, 2).reshape(n, C * PER, 2)  # chain c's needles at c * PER ..
    answer = ents[:, :, 2] if a.task == "twohop" else ents[:, :, 1]
    hay = torch.randint(NOISE, V, (n, L), generator=g)
    rows, scored_pos, depths = [], [], []
    ask = torch.rand(n, C, generator=g).argsort(1)[:, :Q]
    for i in range(n):
        cuts = torch.sort(torch.randint(0, L + 1, (C * PER,), generator=g)).values  # insertion points in the haystack
        order = torch.randperm(C * PER, generator=g)  # which needle goes at which point
        parts, prev, first = [torch.tensor([BOS])], 0, {}
        for slot, nid in enumerate(order.tolist()):
            parts.append(hay[i, prev:cuts[slot]])
            prev = int(cuts[slot])
            c = nid // PER
            first[c] = min(first.get(c, L), prev)
            parts.append(torch.tensor([KEY, int(needles[i, nid, 0]), int(needles[i, nid, 1])]))
        parts.append(hay[i, prev:])
        body = torch.cat(parts)
        qs, pos, dep = [], [], []
        for qi, c in enumerate(ask[i].tolist()):
            qs.append(torch.tensor([ASK, int(ents[i, c, 0]), int(answer[i, c])]))
            pos.append(len(body) + 3 * qi + 1)  # the asked key: predicts the answer
            dep.append(first[c] / max(L, 1))
        rows.append(torch.cat([body] + qs))
        scored_pos.append(pos)
        depths.append(dep)
    seq = torch.stack(rows)  # same length for every row
    x, y = seq[:, :-1], seq[:, 1:]
    s = torch.zeros_like(y, dtype=torch.bool)
    s[torch.arange(n)[:, None], torch.tensor(scored_pos)] = True
    return x, y, s, torch.tensor(depths)


class Block(nn.Module):
    """A bank layer with --hops reads and its own FFN (after every read)."""

    def __init__(self, dim, gate=False):
        super().__init__()
        self.n1, self.n2, self.n3, self.nq = (nn.LayerNorm(dim) for _ in range(4))
        self.bank = LRUBankLayer(dim, lru_dim=32, heads=2, key_dim=32, value_dim=32, slots=a.slots, topk=2,
                                 slot_half_life=(1.0, 1024.0), lru_half_life=(0.5, 4.0), chunk=16, norm="mass",
                                 hops=a.hops, input_gate=gate)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, h, reset, info=None):
        mem, q, _ = self.bank.write(self.n1(h), reset, "loop" if info is not None else "chunk", info=info)
        h = h + self.bank.read(mem, q, 0)
        h = h + self.ffn(self.n2(h))
        for j in range(1, a.hops):
            h = h + self.bank.read(mem, self.bank.query(self.nq(h), j), j)
            h = h + self.ffn(self.n3(h))
        return h


class Net(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.emb = nn.Embedding(V, dim)
        self.blocks = nn.ModuleList([Block(dim, gate=a.gate_second and i > 0) for i in range(a.layers)])
        self.norm, self.out = nn.LayerNorm(dim), nn.Linear(dim, V)

    def forward(self, x, infos=None):
        reset = x == BOS
        h = self.emb(x)
        for i, b in enumerate(self.blocks):
            h = b(h, reset, None if infos is None else infos[i])
        return self.out(self.norm(h))


def xent(logits, y, s):
    idx = s.reshape(-1).nonzero().squeeze(1)
    lg = logits.reshape(-1, logits.shape[-1]).index_select(0, idx)
    return -lg.log_softmax(-1).gather(1, y.reshape(-1).index_select(0, idx)[:, None]).mean()


def evaluate(m, lens, n_rows=None):
    """Accuracy per haystack length, overall and by depth fifth (of the asked chain's first needle)."""
    out = {}
    with torch.no_grad():
        for L in lens:
            n = n_rows or max(16, min(128, 16384 // (L + 32)))
            x, y, s, dep = make_rows(n, L, torch.Generator().manual_seed(1000 + L))
            hit = (m(x).argmax(-1)[s] == y[s]).float().reshape(n, -1)
            b = (dep * 5).clamp(max=4.999).long()
            out[L] = {"acc": float(hit.mean()),
                      "by_depth": [round(float(hit[b == d].mean()), 3) if (b == d).any() else None for d in range(5)],
                      "rows": n}
    return out


m = Net()
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
floor = 1 / a.chains
print(f"NIAH {a.task}: {a.chains} chains x {PER} needles per row (hide ends {a.hide_ends}), {a.queries} questions, "
      f"haystack {a.train_len[0]}-{a.train_len[1]} trained; {a.layers} bank layer(s) x {a.hops} hops, {a.slots} slots"
      f"{', input gate after layer 0' if a.gate_second else ''}; "
      f"{sum(p_.numel() for p_ in m.parameters())} params. Guess among the chains: {floor:.3f}, chance "
      f"{1 / a.entities:.3f}", flush=True)
t0 = time.time()
for step in range(1, a.steps + 1):
    L = int(torch.randint(a.train_len[0], a.train_len[1] + 1, ()))
    x, y, s, _ = make_rows(a.batch, L)
    lg = m(x)
    loss = xent(lg, y, s)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if a.log_every and step % a.log_every == 0:
        acc = float((lg.argmax(-1)[s] == y[s]).float().mean())
        print(f"  step {step}: haystack {L}, loss {loss.item():.3f}, batch accuracy {acc:.3f} ({time.time() - t0:.0f} s)",
              flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        ev = evaluate(m, [a.train_len[1]], 128)[a.train_len[1]]
        print(f"  == step {step}: accuracy at haystack {a.train_len[1]} {ev['acc']:.3f}", flush=True)

print(f"\nevaluation by haystack length (trained {a.train_len[0]}-{a.train_len[1]}); depth fifths of the asked "
      f"chain's first needle (0 = start of the haystack, far from the question)", flush=True)
res = evaluate(m, a.eval_lens)
for L, r in res.items():
    dd = "  ".join("  -  " if v is None else f"{v:.3f}" for v in r["by_depth"])
    print(f"  L {L:5d}: {r['acc']:.3f}   by depth: {dd}   ({r['rows']} rows)", flush=True)
if a.gate_second and a.layers > 1:  # what the gates learned: filler vs needle tokens vs questions
    with torch.no_grad():
        x, y, s, _ = make_rows(64, a.train_len[1], torch.Generator().manual_seed(7))
        infos = [dict() for _ in range(a.layers)]
        m(x, infos)
        kinds = {"filler": (x >= NOISE), "KEY marker": x == KEY, "needle entity": (x >= ENT) & (x < NOISE),
                 "ASK": x == ASK}
        for i in range(1, a.layers):
            gt = infos[i]["input_gate"]
            print(f"  layer {i} input gate (1 = enters the LRU, 0 = skipped): " + ", ".join(
                f"{k} {float(gt[v].mean()):.3f}" for k, v in kinds.items()), flush=True)
if a.save:
    from pathlib import Path

    Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"args": vars(a), "state_dict": m.state_dict(), "results": res}, a.save)
    print(f"saved {a.save}")
