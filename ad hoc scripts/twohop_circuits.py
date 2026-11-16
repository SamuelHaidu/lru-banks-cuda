"""Two-hop recall: do two layers compose? Chains a -> b -> c of distinct tokens, written as shuffled pairs, then queries:

    <bos> (x y)*16 [the pairs a b and b c of 8 chains, shuffled]  ? a c  ? a c ...      (4 queries)

scored at each queried `a` (answer c): two lookups in sequence (a -> b, then b -> c); one lookup layer cannot do it.
Variants: "bank" (one LRU + bank block), "bank+bank", "bank+gdn", "gdn+gdn" (one `gated-deltanet` layer = its mixer
and FFN); each block is pre-norm with a 4x GELU FFN (the GDN layer its own 1-expert MoE). CPU.

Per trained model (`--seeds`): accuracy, how often it answers b instead (one hop), accuracy with each layer's mixer
removed (the bank read, or the GDN w_o set to 0), and for each bank layer: slot usage (entropy / log M, dead slots),
temperature, read sharpness at the queries, and which hop its read retrieves: whether the slot it reads most at the
query is one of the slots the pair (a b) wrote b into (hop 1) or the pair (b c) wrote c into (hop 2).
Weights, arguments and metrics are saved to runs/_synthetic/twohop/<variant>_s<seed>.pt to study the mechanisms.

    uv run python "ad hoc scripts/twohop_circuits.py" [--variants bank bank+bank bank+gdn gdn+gdn] [--seeds 0 1]
        [--steps 1500]
"""

import argparse
import json
import math
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--variants", nargs="+", default=["bank", "bank+bank", "bank+gdn", "gdn+gdn"])
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--chains", type=int, default=8)
p.add_argument("--log-every", type=int, default=5, help="print the training loss and batch accuracy every N steps")
p.add_argument("--chain-hops", type=int, default=2, help="hops per chain: 2 = a b, b c (answer c); 3 = a b, b c, c d")
p.add_argument("--queries", type=int, default=4)
p.add_argument("--entities", type=int, default=48)
p.add_argument("--dim", type=int, default=64)
p.add_argument("--slots", type=int, default=32)
p.add_argument("--out", default="runs/_synthetic/twohop")
p.add_argument("--resume", default=None, help="start from this saved model (.pt; fresh optimizer)")
p.add_argument("--tag", default="", help="suffix of the saved file name")
p.add_argument("--study", default=None, help="only study the circuit of this saved model (.pt)")
p.add_argument("--device", default="cpu", help="cpu or cuda")
p.add_argument("--lr", type=float, default=6e-3)
p.add_argument("--warmup", type=int, default=0, help="linear warmup steps")
a = p.parse_args()

from model.device import setup_device  # noqa: E402

setup_device(a.device)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model.archs.bank_memory import LRUBankLayer, StreamBankLayer  # noqa: E402

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


BOS, ASK = 0, 1
V = 2 + a.entities


def batch(n, g=None):
    """K-hop chains (K = --chain-hops): entities e0 -> e1 -> ... -> eK per chain, the K C pairs (e_j e_j+1) shuffled,
    then queries "? e0 eK". x, y `(n, T)`; scored `(n, T)` (the queried e0's); for each query: its position, b (= e1),
    c (= eK, the answer), pos_b / pos_c (where pair (e0 e1) put e1 and the last pair put eK), pos_hops `(n, Q, K)` (where
    pair j put e_j+1), and the chain entities."""
    C, Q, K = a.chains, a.queries, a.chain_hops
    ents = torch.rand(n, a.entities, generator=g).argsort(1)[:, :(K + 1) * C].reshape(n, C, K + 1) + 2
    pairs = torch.cat([ents[:, :, j:j + 2] for j in range(K)], 1)  # (n, K C, 2): hop j of chain i at j C + i
    order = torch.rand(n, K * C, generator=g).argsort(1)
    pairs = pairs.gather(1, order[..., None].expand(-1, -1, 2))
    ask = torch.rand(n, C, generator=g).argsort(1)[:, :Q]
    qs = ents.gather(1, ask[..., None].expand(-1, -1, K + 1))  # (n, Q, K + 1)
    qtok = torch.stack([torch.full_like(qs[..., 0], ASK), qs[..., 0], qs[..., K]], 2).view(n, -1)
    seq = torch.cat([torch.zeros(n, 1, dtype=torch.long), pairs.reshape(n, -1), qtok], 1)
    x, y = seq[:, :-1], seq[:, 1:]
    qpos = 1 + 2 * K * C + 3 * torch.arange(Q) + 1  # the queried e0's
    scored = torch.zeros_like(y, dtype=torch.bool)
    scored[:, qpos] = True
    inv = order.argsort(1)
    pos_hops = torch.stack([1 + 2 * inv.gather(1, ask + j * C) + 1 for j in range(K)], -1)  # (n, Q, K)
    return x, y, scored, {"qpos": qpos, "b": qs[..., 1], "c": qs[..., K], "pos_b": pos_hops[..., 0],
                          "pos_c": pos_hops[..., -1], "pos_hops": pos_hops, "ask": ask, "ents": ents}


class BankBlock(nn.Module):
    """A bank layer + FFN. hops N (variant "bankN"): read, FFN, then N - 1 times a query from the stream, another read
    of the SAME banks and the FFN again (shared). ablate: no reads at all; ablate_hops: the hop indices (1..) skipped.
    The hops-2 names (nq, n3) are kept so saved bank2 models load."""

    def __init__(self, dim, hops=1, stream_keys=False):
        super().__init__()
        self.hops = hops
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        if stream_keys:  # variant "phiN": keys / queries from one projection of the stream, no LRU
            self.bank = StreamBankLayer(dim, heads=2, key_dim=32, value_dim=32, slots=a.slots, topk=2, chunk=16,
                                        norm="mass", hops=hops)
        else:
            self.bank = LRUBankLayer(dim, lru_dim=32, heads=2, key_dim=32, value_dim=32, slots=a.slots, topk=2,
                                     lru_half_life=(0.5, 4.0), chunk=16, norm="mass", hops=hops)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        if hops > 1:
            self.nq, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        if hops > 2:
            self.more_q = nn.ModuleList([nn.LayerNorm(dim) for _ in range(hops - 2)])
            self.more_n = nn.ModuleList([nn.LayerNorm(dim) for _ in range(hops - 2)])
        self.ablate_hops = set()

    def forward(self, h, reset, ablate=False, info=None):
        if self.hops == 1 and not isinstance(self.bank, StreamBankLayer):
            if not ablate:
                h = h + self.bank(self.n1(h), reset, info=info)
            return h + self.ffn(self.n2(h))
        if self.hops == 1:
            if not ablate:
                h = h + self.bank(self.n1(h), reset, info=info)
            return h + self.ffn(self.n2(h))
        ffn_norms = [self.n2, self.n3] + (list(self.more_n) if self.hops > 2 else [])
        q_norms = [self.nq] + (list(self.more_q) if self.hops > 2 else [])
        mem, q, _ = self.bank.write(self.n1(h), reset, "loop" if info is not None else "chunk", info=info)
        if not ablate:
            h = h + self.bank.read(mem, q, 0, info)
        h = h + self.ffn(ffn_norms[0](h))
        for j in range(1, self.hops):
            if not (ablate or j in self.ablate_hops):
                h = h + self.bank.read(mem, self.bank.query(q_norms[j - 1](h), j), j, info)
            h = h + self.ffn(ffn_norms[j](h))
        return h


class GDNBlock(nn.Module):
    """One `gated-deltanet` layer (mixer + its 1-expert MoE FFN, residual inside)."""

    def __init__(self, dim):
        super().__init__()
        from model.archs.gated_deltanet import GatedDeltaNetLM

        self.m = GatedDeltaNetLM(vocabulary_size=V, dim=dim, num_layers=1, num_heads=2, head_dim=32, num_experts=1,
                                 expert_hidden=4 * dim, window_size=256, router_bias_rate=0.0)
        self.m.build()
        self.names = [w.path for w in self.m.trainable_weights]
        self.ps = nn.ParameterList([w.value for w in self.m.trainable_weights])

    def forward(self, h, reset, ablate=False, info=None):
        w = self.m._params[0]["w_o"]
        if ablate:
            saved = w.value.detach().clone()
            with torch.no_grad():
                w.value.zero_()
        out, _ = self.m._layer(0, h, reset, None)
        if ablate:
            with torch.no_grad():
                w.value.copy_(saved)
        return out


class Net(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.emb = nn.Embedding(V, a.dim)
        self.blocks = nn.ModuleList([BankBlock(a.dim, hops=int(t[4:] or 1)) if t.startswith("bank") else
                                     BankBlock(a.dim, hops=int(t[3:] or 1), stream_keys=True) if t.startswith("phi")
                                     else GDNBlock(a.dim) for t in variant.split("+")])
        self.norm, self.out = nn.LayerNorm(a.dim), nn.Linear(a.dim, V)

    def forward(self, x, ablate=None, infos=None):
        reset = x == BOS
        h = self.emb(x)
        for i, b in enumerate(self.blocks):
            h = b(h, reset, ablate == i, None if infos is None else infos[i])
        return self.out(self.norm(h))


def evaluate(m, variant):
    g = gen(1234)
    x, y, s, gt = batch(512, g)
    res = {}
    with torch.no_grad():
        pred = m(x).argmax(-1)
        res["accuracy"] = float((pred[s] == y[s]).float().mean())
        res["answers_b_one_hop"] = float((pred[:, gt["qpos"]] == gt["b"]).float().mean())
        for i in range(len(m.blocks)):
            res[f"accuracy_without_layer{i}_mixer"] = float((m(x, ablate=i).argmax(-1)[s] == y[s]).float().mean())
            for j in range(1, getattr(m.blocks[i], "hops", 1)):
                m.blocks[i].ablate_hops = {j}
                pa = m(x).argmax(-1)
                m.blocks[i].ablate_hops = set()
                res[f"accuracy_without_layer{i}_read{j + 1}"] = float((pa[s] == y[s]).float().mean())
        infos = [dict() if isinstance(b, BankBlock) else None for b in m.blocks]
        xs, gs = x[:128], {k: (v[:128] if v.dim() >= 2 else v) for k, v in gt.items()}
        m(xs, infos=infos)
        n = torch.arange(128)[:, None]
        for i, info in enumerate(infos):
            if not info:
                continue
            route = info["route"]  # (B, T, H, M)
            use = route.sum((0, 1))  # (H, M)
            share = use / use.sum(-1, keepdim=True)
            ent = -(share.clamp_min(1e-12) * share.clamp_min(1e-12).log()).sum(-1) / math.log(a.slots)
            dead = (share < 0.1 / a.slots).float().mean(-1)
            res[f"layer{i}_bank"] = {
                "slot_usage_entropy": [round(float(e), 3) for e in ent], "dead_slots": [round(float(d), 3) for d in dead],
                "tau": [round(float(t), 2) for t in m.blocks[i].bank.log_tau.exp()]}
            reads = ["read"] + [f"read_hop{j}" for j in range(1, 8) if f"read_hop{j}" in info]
            for r_, rk in enumerate(reads):  # each read: how sharp, and which chain hop its top slot holds
                rq = info[rk][n, gs["qpos"][None].expand(128, -1)]  # (B, Q, H, M)
                top = rq.argmax(-1)  # (B, Q, H)
                lands = {}
                for j in range(gs["pos_hops"].shape[-1]):
                    wrote = route[n, gs["pos_hops"][..., j]] > 0  # (B, Q, H, M): slots pair j wrote its value into
                    lands[f"hop{j + 1}"] = round(float(wrote.gather(-1, top[..., None])[..., 0].float().mean()), 3)
                res[f"layer{i}_bank"][f"read{r_ + 1}"] = {
                    "max_prob": round(float(rq.max(-1).values.mean()), 3), "top_slot_holds": lands}
    return res


def study_hops(m, ck, path):
    """The circuit of a one-layer two-hop bank ("bank2"): expected, at the queried a: read 1 finds the slot filed under
    "context a" and returns b; the FFN turns b into the query for "context b"; read 2 finds the slot holding c."""
    blk = m.blocks[0]
    bank = blk.bank
    g = gen(4321)
    n = 256
    x, y, s, gt = batch(n, g)
    Q, C = a.queries, a.chains
    rows = torch.arange(n)[:, None]
    qp = gt["qpos"][None].expand(n, -1)
    pct = lambda t: f"{100 * float(t):5.1f}%"

    def stages(xx, info=None):
        reset = xx == BOS
        h0 = m.emb(xx)
        mem, q, _ = bank.write(blk.n1(h0), reset, "loop", info=info)
        h1 = h0 + bank.read(mem, q, 0, info)
        h2 = h1 + blk.ffn(blk.n2(h1))
        h3 = h2 + bank.read(mem, bank.query(blk.nq(h2), 1), 1, info)
        h4 = h3 + blk.ffn(blk.n3(h3))
        return [h0, h1, h2, h3, h4]

    with torch.no_grad():
        info = {}
        hs = stages(x, info)
        print(f"{path}: {ck['variant']}, trained {ck.get('steps', '?')} steps")
        print("\n1. logit lens at the queried a, inside the layer (b = one hop, c = the answer): top-1 rate, mean p")
        for name, hh in zip(["embedding", "+ read 1", "+ FFN", "+ read 2", "+ FFN (final)"], hs):
            pr = torch.softmax(m.out(m.norm(hh)), -1)[rows, qp]
            top = pr.argmax(-1)
            pb, pc = pr.gather(-1, gt["b"][..., None])[..., 0], pr.gather(-1, gt["c"][..., None])[..., 0]
            print(f"   {name:14s} top-1 = b {pct((top == gt['b']).float().mean())}  top-1 = c "
                  f"{pct((top == gt['c']).float().mean())}   p(b) {float(pb.mean()):.3f}  p(c) {float(pc.mean()):.3f}")
        route = info["route"]
        print("\n2. read mass at the queried a on the slots that hold b (pair a b) / c (pair b c), per head")
        for rk, nm in (("read", "read 1"), ("read_hop1", "read 2")):
            rq = info[rk][rows, qp]
            for name, pos in (("b", gt["pos_b"]), ("c", gt["pos_c"])):
                held = (route[rows, pos] > 0).float()
                mass = (rq * held).sum(-1).mean((0, 1))
                print(f"   {nm} on slots holding {name}: " + "  ".join(f"head {h_} {pct(v)}" for h_, v in enumerate(mass)))
        print(f"\n3. each query against every key of the row ({x.shape[1]} positions): mean rank of the key stored with"
              " b (filed under 'context a') and with c ('context b'); rank 1 = the best match")
        k = info["k"]
        for qk_, nm in (("q", "query 1"), ("q_hop1", "query 2")):
            sim = torch.einsum("nqhd,nthd->nqht", info[qk_][rows, qp], k)
            line = []
            for name, pos in (("b", gt["pos_b"]), ("c", gt["pos_c"])):
                own = sim.gather(-1, pos[:, :, None, None].expand(-1, -1, sim.shape[2], 1))
                rank = (sim > own).sum(-1).float() + 1
                line.append(f"key of {name}: " + " ".join(f"h{h_} {float(v):4.1f}" for h_, v in enumerate(rank.mean((0, 1))))
                            + f" (top-1 {pct((rank == 1).float().mean())})")
            print(f"   {nm}: " + "   ".join(line))
        # the query of hop 2 vs the query the LRU itself makes right after b (in pair b c): does the FFN rebuild it?
        q2 = info["q_hop1"][rows, qp]
        q_after_b = info["q"][rows, gt["pos_c"] - 1]  # the LRU's query at the b of pair (b c)
        cos = (q2 * q_after_b).sum(-1)
        cos_other = (q2 * info["q"][rows, (gt["pos_c"] - 1).roll(1, 1)]).sum(-1)  # another chain's b
        print(f"   cosine(query 2, the LRU's own query right after b) {float(cos.mean()):.3f}, vs after another "
              f"chain's b {float(cos_other.mean()):.3f}")

        acc = lambda xx, target: float((m(xx).argmax(-1)[rows, qp] == target).float().mean())
        base = acc(x, gt["c"])
        used = torch.zeros(n, V, dtype=torch.bool)
        used.scatter_(1, x, True)
        fresh = torch.stack([torch.nonzero(~used[i] & (torch.arange(V) >= 2))[:, 0][torch.randperm(int((~used[i] &
                             (torch.arange(V) >= 2)).sum()), generator=g, device=DEV)[:Q]] for i in range(n)])
        print(f"\n4. counterfactuals at the queried a: accuracy {pct(base)}")
        x2 = x.clone()
        x2[rows, gt["pos_c"]] = fresh
        print(f"   pair (b c) -> (b c'), c' a fresh token: answers c' {pct(acc(x2, fresh))}, still c {pct(acc(x2, gt['c']))}")
        x3 = x.clone()
        x3[rows, gt["pos_b"]] = fresh
        print(f"   pair (a b) -> (a b'), b' fresh (chain broken): still c {pct(acc(x3, gt['c']))}")
        # redirect: (a b) -> (a b_j), b_j the middle of another chain j: composition must answer c_j
        j = (gt["ask"] + 1 + torch.randint(0, C - 1, gt["ask"].shape, generator=g)) % C
        bj = gt["ents"][rows, j, 1]
        cj = gt["ents"][rows, j, 2]
        x4 = x.clone()
        x4[rows, gt["pos_b"]] = bj
        print(f"   pair (a b) -> (a b_j), b_j from another chain (b_j c_j): answers c_j {pct(acc(x4, cj))}, "
              f"still c {pct(acc(x4, gt['c']))}, answers b_j {pct(acc(x4, bj))}")
        # ablate read 2 / give read 2 the query of another chain's b
        blk.ablate_hops = {1}
        print(f"   without read 2: {pct(acc(x, gt['c']))}")
        blk.ablate_hops = set()


def study(path):
    """The circuit of a saved model at the queried a's (see the module doc of the analysis in the printout)."""
    ck = torch.load(path, weights_only=False, map_location=DEV)
    m = Net(ck["variant"]).to(DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    if ck["variant"] == "bank2":
        return study_hops(m, ck, path)
    g = gen(4321)
    n = 256
    x, y, s, gt = batch(n, g)
    Q = a.queries
    rows = torch.arange(n)[:, None]
    qp = gt["qpos"][None].expand(n, -1)
    pct = lambda t: f"{100 * float(t):5.1f}%"
    with torch.no_grad():
        # 1. logit lens: the stream after the embedding and after each layer, through the final norm and readout
        reset = x == BOS
        h = m.emb(x)
        lens = [h]
        infos = [dict() if isinstance(b, BankBlock) else None for b in m.blocks]
        for i, b in enumerate(m.blocks):
            h = b(h, reset, False, infos[i])
            lens.append(h)
        print(f"{path}: {ck['variant']}, trained {ck.get('steps', '?')} steps")
        print("\n1. logit lens at the queried a (b = one hop, c = the answer): top-1 rate and mean probability")
        for name, hh in zip(["embedding"] + [f"after layer {i}" for i in range(len(m.blocks))], lens):
            pr = torch.softmax(m.out(m.norm(hh)), -1)[rows, qp]  # (n, Q, V)
            top = pr.argmax(-1)
            pb, pc = pr.gather(-1, gt["b"][..., None])[..., 0], pr.gather(-1, gt["c"][..., None])[..., 0]
            print(f"   {name:14s} top-1 = b {pct((top == gt['b']).float().mean())}  top-1 = c "
                  f"{pct((top == gt['c']).float().mean())}   p(b) {float(pb.mean()):.3f}  p(c) {float(pc.mean()):.3f}")
        # 2. where each layer's read goes: its read mass on the slots holding b (pair a b) / c (pair b c)
        print("\n2. read mass at the queried a on the slots that hold b (written by pair a b) / c (pair b c), per head;"
              " the rest goes elsewhere")
        for i, info in enumerate(infos):
            if not info:
                continue
            route, read = info["route"], info["read"]  # (n, T, H, M)
            rq = read[rows, qp]  # (n, Q, H, M)
            for name, pos in (("b", gt["pos_b"]), ("c", gt["pos_c"])):
                held = (route[rows, pos] > 0).float()  # (n, Q, H, M)
                mass = (rq * held).sum(-1).mean((0, 1))
                print(f"   layer {i} slots holding {name}: " + "  ".join(f"head {h_} {pct(v)}" for h_, v in enumerate(mass)))
            # 3. the query against the keys: rank of the key written with b (resp. c) among all keys of the row
            q, k = info["q"][rows, qp], info["k"]  # (n, Q, H, D), (n, T, H, D)
            sim = torch.einsum("nqhd,nthd->nqht", q, k)
            for name, pos in (("b", gt["pos_b"]), ("c", gt["pos_c"])):
                own = sim.gather(-1, pos[:, :, None, None].expand(-1, -1, sim.shape[2], 1))
                rank = (sim > own).sum(-1).float() + 1  # among all positions of the row
                print(f"   layer {i} query vs the key stored with {name}: mean rank " + "  ".join(
                    f"head {h_} {float(v):.1f}" for h_, v in enumerate(rank.mean((0, 1)))) + f"  (of {x.shape[1]})")
        # 4. counterfactuals
        acc = lambda xx, target: float((m(xx).argmax(-1)[rows, qp] == target).float().mean())
        base = acc(x, gt["c"])
        used = torch.zeros(n, V, dtype=torch.bool)
        used.scatter_(1, x, True)
        fresh = torch.stack([torch.nonzero(~used[i] & (torch.arange(V) >= 2))[:, 0][torch.randperm(int((~used[i] &
                             (torch.arange(V) >= 2)).sum()), generator=g, device=DEV)[:Q]] for i in range(n)])  # (n, Q) unused tokens
        x2 = x.clone()
        x2[rows, gt["pos_c"]] = fresh  # pair (b c) -> (b c'): the answer should follow c'
        follows = acc(x2, fresh)
        still_c = acc(x2, gt["c"])
        x3 = x.clone()
        x3[rows, gt["pos_b"]] = fresh  # pair (a b) -> (a b'): the chain is broken
        broken = acc(x3, gt["c"])
        print(f"\n4. counterfactuals at the queried a: accuracy {pct(base)}")
        print(f"   pair (b c) -> (b c'), c' a fresh token: answers c' {pct(follows)}, still c {pct(still_c)}")
        print(f"   pair (a b) -> (a b'), chain broken: still answers c {pct(broken)}")


def main():
    if a.study:
        study(a.study)
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for variant in a.variants:
        for seed in a.seeds:
            torch.manual_seed(seed)
            m = Net(variant)
            done = 0
            if a.resume:
                ck = torch.load(a.resume, weights_only=False, map_location=DEV)
                m.load_state_dict(ck["state_dict"])
                done = ck.get("steps", ck["args"]["steps"])
                print(f"  resumed {a.resume} (trained {done} steps; fresh optimizer)", flush=True)
            m = m.to(DEV)
            opt = make_opt(m)
            g = gen(seed + 1000 * (done > 0))
            t0, curve = time.time(), []
            for step in range(1, a.steps + 1):
                x, y, s, _ = batch(a.batch, g)
                lg = m(x)
                loss = xent(lg, y, s)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if a.log_every and step % a.log_every == 0:
                    with torch.no_grad():
                        bacc = float((lg.argmax(-1)[s] == y[s]).float().mean())
                    print(f"    step {done + step}: loss {loss.item():.3f}, batch accuracy {bacc:.3f} "
                          f"({time.time() - t0:.0f} s)", flush=True)
                if step % 250 == 0 or step == a.steps:
                    with torch.no_grad():
                        x, y, s, _ = batch(256, gen(99))
                        acc = float((m(x).argmax(-1)[s] == y[s]).float().mean())
                    curve.append((step, round(loss.item(), 4), round(acc, 4)))
                    print(f"  {variant} seed {seed} step {done + step}: loss {loss.item():.3f}, accuracy {acc:.3f} "
                          f"({time.time() - t0:.0f} s)", flush=True)
            res = evaluate(m, variant)
            res["curve"] = curve
            summary[f"{variant}_s{seed}"] = res
            torch.save({"variant": variant, "seed": seed, "args": vars(a), "steps": done + a.steps,
                        "state_dict": m.state_dict(),
                        "gdn_weight_names": {i: b.names for i, b in enumerate(m.blocks) if isinstance(b, GDNBlock)},
                        "metrics": res}, out / f"{variant}_s{seed}{a.tag}.pt")
            print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=1), flush=True)
    old = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    (out / "summary.json").write_text(json.dumps({**old, **summary}, indent=1))  # earlier variants kept
    print(f"\nchance {1 / a.entities:.3f}")
    for k, r in summary.items():
        abl = ", ".join(f"-L{i} {r[f'accuracy_without_layer{i}_mixer']:.3f}" for i in range(k.count("+") + 1))
        print(f"{k:14s} accuracy {r['accuracy']:.3f}  answers b {r['answers_b_one_hop']:.3f}  ablations: {abl}")


if __name__ == "__main__":
    main()
