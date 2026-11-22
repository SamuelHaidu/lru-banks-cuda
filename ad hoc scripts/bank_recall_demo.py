"""Does the LRU + bank layer learn associative recall? Multi-query associative recall (MQAR, as in Zoology / Based):

    <bos> k1 v1 k2 v2 ... kN vN  ka va  kb vb ...      (N random key -> value pairs, then Q of the keys asked again)

scored only on the answers (the value after each asked key). Pure lookup: guessing gets 1 / values. One layer on top
of an embedding, the readout from the stream; the same model with the bank switched off (LRU only) is the baseline.
`--task induction`: the textbook induction test instead: <bos> s1 .. sL s1 .. sL (L distinct random tokens, then the
same sequence again); the first copy is unpredictable (chance), the second is predictable only by induction (find
where the current token appeared and copy what followed it); accuracy on both copies.
`--task copy`: the copying benchmark ("Repeat After Me", Jelassi et al. 2024): <bos> s <sep> s, s a random string of
length L over `--alphabet` letters (repeats allowed: the next letter is found by the last few, not the last one),
trained on L in [--min-len, --max-len] for `--seconds`, then token accuracy and exact-string accuracy per length,
also past the trained lengths.
`--arch gdn`: one layer of the `gated-deltanet` model instead (Gated DeltaNet mixer + its FFN, the torch delta rule).
CPU, a few minutes.

    uv run python "ad hoc scripts/bank_recall_demo.py" [--arch bank|lru|gdn|all] [--steps 800] [--pairs 16] [--norm mass]
"""

import argparse
import math
import time

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=800)
p.add_argument("--pairs", type=int, default=16)
p.add_argument("--queries", type=int, default=8)
p.add_argument("--keys", type=int, default=64)
p.add_argument("--values", type=int, default=64)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--norm", default="mass")
p.add_argument("--slots", type=int, default=64)
p.add_argument("--topk", type=int, default=2, help="slots written per token")
p.add_argument("--raw-route", action="store_true", help="write with the raw top-K probabilities (not renormalized)")
p.add_argument("--chunk", type=int, default=16)
p.add_argument("--heads", type=int, default=2)
p.add_argument("--head-dim", type=int, default=32, help="key and value size per head")
p.add_argument("--layers", type=int, default=1, help="bank layers; --slots is split between them; an FFN per layer")
p.add_argument("--shared-ffn", action="store_true", help="one FFN shared by all layers (how runs before the per-layer "
               "FFN were trained)")
p.add_argument("--log-every", type=int, default=5, help="print the training loss and batch accuracy every N steps")
p.add_argument("--erase", nargs="?", const="query", default=None, choices=["query", "read"],
               help="erase before each write: 'query' (slots the query is routed to) or 'read' (a key bank addresses, "
               "the value bank is erased where its read landed); LRUBankLayer(erase=...)")
p.add_argument("--overwrites", type=int, default=0, help="MQAR: this many keys get a new value after the pairs; "
               "every one is asked (the latest value is the answer), the other queries ask unchanged keys")
p.add_argument("--gap", type=int, default=0, help="MQAR: this many filler pairs (fresh keys, never asked) between "
               "the overwrites and the queries")
p.add_argument("--hops", type=int, default=1, help="reads per bank layer: hop n+1 queries from the stream after hop n "
               "and the shared FFN (the same banks, still parallel)")
p.add_argument("--bench", type=int, default=0, help="only time this many training steps (at --max-len)")
p.add_argument("--save", default=None, help="save weights, arguments and the evaluation here (.pt)")
p.add_argument("--arch", default="all", choices=["bank", "lru", "gdn", "all"])
p.add_argument("--task", default="mqar", choices=["mqar", "induction", "copy"])
p.add_argument("--alphabet", type=int, default=32)
p.add_argument("--min-len", type=int, default=8)
p.add_argument("--max-len", type=int, default=64)
p.add_argument("--eval-lens", nargs="+", type=int, default=[8, 16, 32, 48, 64, 96, 128])
p.add_argument("--seconds", type=float, default=0, help="train for this long instead of --steps")
p.add_argument("--length", type=int, default=24, help="induction: tokens per copy")
p.add_argument("--device", default="cpu", help="cpu or cuda")
p.add_argument("--lr", type=float, default=6e-3)
p.add_argument("--warmup", type=int, default=0, help="linear warmup steps")
a = p.parse_args()

from model.device import setup_device  # noqa: E402

setup_device(a.device)

import numpy as np  # noqa: E402
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.archs.bank_memory import LRUBankLayer

DEV = torch.device(a.device)
torch.set_default_device(DEV)  # every new tensor (batches, masks) directly on the device


def gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEV).manual_seed(seed)


def xent(logits: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Cross-entropy at the positions s (bool `(B, T)`): picked with index_select and computed as log_softmax + gather.
    The Iris Xe fails on cross_entropy over a boolean-mask selection at batch 256+ (work-groups over 512)."""
    idx = s.reshape(-1).nonzero().squeeze(1)
    lg = logits.reshape(-1, logits.shape[-1]).index_select(0, idx)
    return -lg.log_softmax(-1).gather(1, y.reshape(-1).index_select(0, idx)[:, None]).mean()


def make_opt(model):
    """AdamW at --lr with a linear warmup over --warmup steps."""
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    step, k = opt.step, [0]

    def warm_step(*args, **kw):
        for group in opt.param_groups:
            group["lr"] = a.lr * min(1.0, (k[0] + 1) / max(1, a.warmup))
        k[0] += 1
        return step(*args, **kw)

    opt.step = warm_step
    return opt


torch.manual_seed(0)
V = 1 + a.keys + a.values


def batch(n, split=False):
    """<bos> pairs [overwrites] queries. split: also the masks of the answers to overwritten keys and the old values
    (what a blended or stale read would give)."""
    U = a.overwrites
    G = a.gap
    keys_all = torch.rand(n, a.keys).argsort(1)[:, :a.pairs + G] + 1  # distinct keys per row
    keys, gap_keys = keys_all[:, :a.pairs], keys_all[:, a.pairs:]
    raw = torch.randint(0, a.values, (n, a.pairs))
    perm = torch.rand(n, a.pairs).argsort(1)
    ow, others = perm[:, :U], perm[:, U:U + a.queries - U]  # overwritten keys (all asked), then unchanged ones
    new = (raw.gather(1, ow) + torch.randint(1, a.values, (n, U))) % a.values  # a different value
    latest = raw.scatter(1, ow, new)
    order = torch.rand(n, a.queries).argsort(1)
    ask = torch.cat([ow, others], 1).gather(1, order)
    is_ow = torch.cat([torch.ones(n, U, dtype=torch.bool), torch.zeros(n, a.queries - U, dtype=torch.bool)], 1).gather(1, order)
    off = 1 + a.keys
    seq = torch.cat([torch.zeros(n, 1, dtype=torch.long),
                     torch.stack([keys, raw + off], 2).reshape(n, -1),
                     torch.stack([keys.gather(1, ow), new + off], 2).reshape(n, -1),
                     torch.stack([gap_keys, torch.randint(0, a.values, (n, G)) + off], 2).reshape(n, -1),
                     torch.stack([keys.gather(1, ask), latest.gather(1, ask) + off], 2).reshape(n, -1)], 1)
    x, y = seq[:, :-1], seq[:, 1:]
    scored = torch.zeros_like(y, dtype=torch.bool)
    first = 2 * a.pairs + 2 * U + 2 * G + 1
    scored[:, first::2] = True  # the answers: the value after each asked key
    if not split:
        return x, y, scored
    ow_mask = torch.zeros_like(scored)
    ow_mask[:, first::2] = is_ow
    old = torch.zeros_like(y)
    old[:, first::2] = raw.gather(1, ask) + off  # the first value of each asked key
    return x, y, scored, ow_mask, old


def induction_batch(n):
    """<bos> s s: L distinct random tokens twice. Scored: copy 1 (from its 2nd token: chance) and copy 2 (from its
    2nd token: induction); returns x, y, copy-1 mask, copy-2 mask."""
    L = a.length
    s = torch.rand(n, V - 1).argsort(1)[:, :L] + 1
    seq = torch.cat([torch.zeros(n, 1, dtype=torch.long), s, s], 1)
    x, y = seq[:, :-1], seq[:, 1:]
    t = torch.arange(x.shape[1])
    c1 = ((t >= 1) & (t <= L - 1)).expand_as(y)
    c2 = ((t >= L + 1) & (t <= 2 * L - 1)).expand_as(y)
    return x, y, c1, c2


def copy_batch(n, L, g=None):
    """<bos> s <sep> s (sep = 1, letters 2..). Scored: the copy (from the <sep>, which predicts s_1)."""
    s = torch.randint(2, 2 + a.alphabet, (n, L), generator=g)
    seq = torch.cat([torch.zeros(n, 1, dtype=torch.long), s, torch.ones(n, 1, dtype=torch.long), s], 1)
    x, y = seq[:, :-1], seq[:, 1:]
    t = torch.arange(x.shape[1])
    return x, y, ((t >= L + 1) & (t <= 2 * L)).expand_as(y)


class Model(nn.Module):
    def __init__(self, bank: bool, dim=64):
        super().__init__()
        self.emb = nn.Embedding(V, dim)
        mk = lambda: LRUBankLayer(dim, lru_dim=32, heads=a.heads, key_dim=a.head_dim, value_dim=a.head_dim,
                                  slots=a.slots // a.layers,
                                  lru_half_life=(0.5, 4.0), chunk=a.chunk, norm=a.norm, topk=a.topk,
                                  route_renorm=not a.raw_route, hops=a.hops, erase=a.erase or False)
        self.layers = nn.ModuleList([mk() for _ in range(a.layers)])
        self.layer = self.layers[0]
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(a.layers)]) if a.layers > 1 else None
        self.bank = bank
        mlp = lambda: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.mlp = mlp()  # layer 0's FFN (the only one with --shared-ffn)
        self.mlps_more = nn.ModuleList([mlp() for _ in range(a.layers - 1)]) if not a.shared_ffn else None
        self.out = nn.Linear(dim, V)

    def ffn(self, i: int) -> nn.Module:
        return self.mlp if i == 0 or self.mlps_more is None else self.mlps_more[i - 1]

    def forward(self, x):
        reset = x == 0
        h = self.emb(x)
        if self.bank:
            for i, layer in enumerate(self.layers):  # bank, then the layer's FFN (reused by its hops)
                mem, q, _ = layer.write(h if self.norms is None else self.norms[i](h), reset)
                h = h + layer.read(mem, q, 0)
                for j in range(1, a.hops):  # FFN, then a query from the stream into the same banks
                    h = h + self.ffn(i)(h)
                    h = h + layer.read(mem, layer.query(h, j), j)
                h = h + self.ffn(i)(h)
            return self.out(h)
        # the LRU alone: its state through the query projection, no banks
        hh, _ = __import__("model.archs.bank_memory", fromlist=["lru_chunk"]).lru_chunk(
            self.layer.lru_input(h), self.layer.log_lam(), reset)
        h = h + self.layer.wo(self.layer.wq(torch.cat([hh.real, hh.imag], -1)))
        h = h + self.mlp(h)
        return self.out(h)


class GDN(nn.Module):
    """One `gated-deltanet` layer (mixer + FFN) on the embedding, its own readout."""

    def __init__(self, dim=64):
        super().__init__()
        from model.archs.gated_deltanet import GatedDeltaNetLM

        self.m = GatedDeltaNetLM(vocabulary_size=V, dim=dim, num_layers=1, num_heads=2, head_dim=32, num_experts=1,
                                 expert_hidden=4 * dim, window_size=256, router_bias_rate=0.0)
        self.m.build()
        self.ps = nn.ParameterList([w.value for w in self.m.trainable_weights])

    def forward(self, x):
        h = self.m.dense_hidden(x.int())
        return torch.as_tensor(self.m.readout(self.m.readout_input(h)))


def log_step(step, loss, logits, y, s, t0):
    """Every --log-every steps: the training loss and the accuracy on this batch's scored positions (no extra pass)."""
    if a.log_every and step % a.log_every == 0:
        with torch.no_grad():
            acc = float((logits.argmax(-1)[s] == y[s]).float().mean())
        print(f"    step {step}: loss {loss.item():.3f}, batch accuracy {acc:.3f} ({time.time() - t0:.0f} s)", flush=True)


def train(arch: str):
    m = GDN() if arch == "gdn" else Model(arch == "bank")
    bank = arch
    m = m.to(DEV)
    opt = make_opt(m)
    if a.bench:
        x, y, s = copy_batch(a.batch, a.max_len)
        for i in range(a.bench + 2):
            if i == 2:
                t0 = time.time()
            loss = xent(m(x), y, s)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  {bank}, {a.layers} layer(s) x {a.hops} hop(s) x {a.slots // a.layers} slots, {a.heads} heads x {a.head_dim}: {1000 * (time.time() - t0) / a.bench:.0f} ms "
              f"per step (batch {a.batch}, length {a.max_len})", flush=True)
        return
    t0 = time.time()
    if a.task == "copy":
        step = 0
        while time.time() - t0 < a.seconds if a.seconds else step < a.steps:
            step += 1
            x, y, s = copy_batch(a.batch, int(torch.randint(a.min_len, a.max_len + 1, ())))
            lg = m(x)
            loss = xent(lg, y, s)
            opt.zero_grad()
            loss.backward()
            opt.step()
            log_step(step, loss, lg, y, s, t0)
        print(f"  {bank}: {step} steps in {time.time() - t0:.0f} s, last loss {loss.item():.3f}; per length "
              f"(trained {a.min_len}-{a.max_len}): token accuracy / exact string", flush=True)
        results = {}
        with torch.no_grad():
            for L in a.eval_lens:
                x, y, s = copy_batch(256, L, gen(L))
                hit = (m(x).argmax(-1) == y) | ~s
                acc = float(hit[s.any(0).expand_as(s)].float().mean())
                # bits recovered: L * (log2 A - H(error)), the wrong guesses spread over the other A - 1 letters
                A = a.alphabet
                h = -(acc * math.log2(acc) if acc > 0 else 0) - ((1 - acc) * math.log2((1 - acc) / (A - 1)) if acc < 1 else 0)
                bits = L * (math.log2(A) - h)
                results[L] = {"token_accuracy": acc, "exact": float(hit.all(1).float().mean()), "bits": bits}
                print(f"    L {L:4d}: {acc:.3f} / {hit.all(1).float().mean():.3f}   {bits:6.1f} bits", flush=True)
        if a.save:
            from pathlib import Path

            Path(a.save).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"args": vars(a), "steps": step, "state_dict": m.state_dict(), "results": results}, a.save)
            print(f"  saved {a.save}")
        return
    for step in range(1, a.steps + 1):
        if a.task == "induction":
            x, y, c1, s = induction_batch(a.batch)
            s = c1 | s  # trained on every predictable-or-not position of both copies
        else:
            x, y, s = batch(a.batch)
        lg = m(x)
        loss = xent(lg, y, s)
        opt.zero_grad()
        loss.backward()
        opt.step()
        log_step(step, loss, lg, y, s, t0)
        if step % 200 == 0 or step == a.steps:
            with torch.no_grad():
                if a.task == "induction":
                    x, y, c1, c2 = induction_batch(512)
                    hit = m(x).argmax(-1) == y
                    print(f"  {bank} step {step}: loss {loss.item():.3f}, copy 1 accuracy {hit[c1].float().mean():.3f} "
                          f"(chance ~{1 / (V - 1):.3f}), copy 2 (induction) {hit[c2].float().mean():.3f}, "
                          f"{time.time() - t0:.0f} s", flush=True)
                    continue
                x, y, s, owm, old = batch(512, split=True)
                pred = m(x).argmax(-1)
                acc = (pred[s] == y[s]).float().mean()
                extra = ""
                if a.overwrites:
                    keep = s & ~owm
                    extra = (f"; overwritten keys: latest {(pred[owm] == y[owm]).float().mean():.3f}, old value "
                             f"{(pred[owm] == old[owm]).float().mean():.3f}; unchanged keys {(pred[keep] == y[keep]).float().mean():.3f}")
            print(f"  {bank} step {step}: loss {loss.item():.3f}, recall accuracy {acc:.3f} "
                  f"(chance {1 / a.values:.3f}){extra}, {time.time() - t0:.0f} s", flush=True)


print(f"copy: strings over {a.alphabet} letters" if a.task == "copy" else
      f"induction: {a.length} distinct random tokens of {V - 1}, then the same again" if a.task == "induction" else
      f"MQAR: {a.pairs} pairs, {a.queries} queries, {a.keys} keys, {a.values} values, slots {a.slots}, norm {a.norm}")
for arch in (["bank", "lru", "gdn"] if a.arch == "all" else [a.arch]):
    train(arch)
