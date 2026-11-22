"""Induction diagnosis (layer 0, hop 1), from analyze_copy.py's replay: on documents <bos> s s <eos>, for every second-copy step
and head, whether the successor was written, the read attention on its slots, its share of their averages and the
fraction of the read that is the successor.

(Originally: What a model trained on the copy task does, step by step: a float64 replay of the C++ model (the gradient check's
reference math) on rows dumped by `lbc dump`, recording every bank write and every read's attention over the slots.

    build/release/lbc dump RUN --split DATA/validation_L512.npy --rows 2 --out /tmp/lbc_inspect
    python3 scripts/analyze_copy.py /tmp/lbc_inspect [--ablate]

Reports:
  1. the replay's loss / accuracy against the C++ (the replay is the model);
  2. copy accuracy by position in the copied half;
  3. writes: how a layer's routing spreads the L source tokens over its slots (per head: distinct slots, tokens per slot);
  4. reads: for every copy step (input x_j, target x_{j+1}) and every read (layer, hop, head), the source position whose
     write dominates the slot the read attends to most, as an offset from the target's own source position (0: the read
     fetches x_{j+1}'s write: an induction lookup; -1: x_j's), and how concentrated the attention is;
  5. with --ablate: copy accuracy with each read path (layer, hop) silenced.
"""

import math
import sys
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DUMP = Path(ARGS[0] if ARGS else "/tmp/lbc_inspect")
ABLATE = "--ablate" in sys.argv
BOS, EOS, SEP = 0, 1, 2


def read(name, dtype):
    return torch.frombuffer(bytearray((DUMP / name).read_bytes()), dtype=dtype)


cfg = {}
for line in (DUMP / "config.txt").read_text().split("\n"):
    if line.strip():
        k, v = line.split()
        cfg[k] = [int(x) for x in v.split(",")] if k in ("hops", "bank_slots") else (
            float(v) if any(c in v for c in ".e") else int(v))
V, D, L, N, H, B, T, DH, SLOTS = (cfg[k] for k in ("V", "D", "L", "N", "H", "B", "T", "DH", "SLOTS"))
HD, HOPS, NSLOT = H * DH, cfg["hops"], cfg["bank_slots"]
FACT, TIED = bool(cfg["factorized"]), bool(cfg["tied"])

params = read("params.bin", torch.float32).double()
P = {}
for line in (DUMP / "manifest.txt").read_text().split("\n"):
    if line.strip():
        name, off, rows, cols = line.split()
        off, rows, cols = int(off), int(rows), int(cols)
        P[name] = params[off:off + rows * cols].reshape(rows, cols)
ids = read("ids.bin", torch.int32).long().reshape(B, T)
y = read("y.bin", torch.int32).long().reshape(B, T)
reset = read("reset.bin", torch.uint8).bool().reshape(B, T)
dec = []
for n in range(L):
    d = {k: read(f"l{n}_{k}.bin", torch.uint8).long() for k in ("ridx", "gw", "gr", "gr2")}
    d["ridx"] = d["ridx"].reshape(B, T, H, 2)
    for k in ("gw", "gr", "gr2"):
        d[k] = d[k].reshape(B, T, H)
    dec.append(d)


def slot_decays(ns):
    hl = torch.exp(torch.linspace(math.log(cfg["slot_hl_min"]), math.log(cfg["slot_hl_max"]), ns))
    hl = torch.cat([hl, hl[-1:].expand(SLOTS - ns)])
    return torch.exp((math.log(0.5) / hl).float().double())


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def gelu(z):
    return 0.5 * z * (1 + torch.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))


def lru(Pin, nu, theta):
    rate = nu.exp()
    lr_, li = torch.exp(-rate) * torch.cos(theta), torch.exp(-rate) * torch.sin(theta)
    gam = torch.sqrt(-torch.expm1(-2 * rate))
    hr, hi = torch.zeros(B, N), torch.zeros(B, N)
    out = []
    for t in range(T):
        keep = (~reset[:, t]).double()[:, None]
        xr, xi = Pin[:, t, :N], Pin[:, t, N:]
        hr, hi = keep * (lr_ * hr - li * hi) + gam * xr, keep * (lr_ * hi + li * hr) + gam * xi
        out.append(torch.cat([hr, hi], -1))
    return torch.stack(out, 1)



LSEQ = int(ARGS[1]) if len(ARGS) > 1 else 100
n = 0
p_ = lambda key: P[f"l{n}_{key}"]
d = dec[0]
h = P["E"][ids] @ P["W_in"] if FACT else P["E"][ids].clone()
PV = rms(h) @ (p_("Wa") * p_("norm1").T)
s = lru(PV[..., :2 * N], p_("nu")[0], p_("theta")[0])
G = s @ p_("Ws")
kprev = torch.cat([torch.zeros(B, 1, HD), G[..., :HD][:, :-1]], 1) * (~reset).double()[..., None]
k = normalize((kprev + p_("bk")[0]).reshape(B, T, H, DH))
q = normalize(G[..., HD:2 * HD].reshape(B, T, H, DH))
R = p_("R").reshape(H, DH, SLOTS)
lr_ = torch.einsum("bthd,hdm->bthm", k, R)
p1 = torch.sigmoid(lr_.gather(-1, d["ridx"][..., :1])[..., 0] - lr_.gather(-1, d["ridx"][..., 1:])[..., 0])
gw = d["gw"]
v = PV[..., 2 * N:].reshape(B, T, H, DH)
decay = slot_decays(NSLOT[0])
tau = p_("log_tau")[0].exp()
stats = {"written": [], "attn_on_succ_slots": [], "succ_share": [], "signal": [], "writes_in_slot": [], "entropy": []}
for b in range(B):
    starts = [t for t in range(T) if reset[b, t]]
    targets = {}  # query time -> successor write time
    for s0 in starts:
        if s0 + 2 * LSEQ + 1 < T:
            for j in range(LSEQ - 1):
                targets[s0 + LSEQ + 1 + j] = s0 + 2 + j
    for hh in range(H):
        K, Vb, Z = torch.zeros(SLOTS, DH), torch.zeros(SLOTS, DH), torch.zeros(SLOTS)
        hist = [[] for _ in range(SLOTS)]  # per slot: (time, weight) of its writes since the last reset
        for t in range(T):
            if reset[b, t]:
                K, Vb, Z = K * 0, Vb * 0, Z * 0
                hist = [[] for _ in range(SLOTS)]
            else:
                K, Vb, Z = K * decay[:, None], Vb * decay[:, None], Z * decay
            if gw[b, t, hh]:
                m1, m2 = int(d["ridx"][b, t, hh, 0]), int(d["ridx"][b, t, hh, 1])
                for m, w in ((m1, float(p1[b, t, hh])), (m2, 1 - float(p1[b, t, hh]))):
                    K[m] += w * k[b, t, hh]
                    Vb[m] += w * v[b, t, hh]
                    Z[m] += w
                    hist[m].append((t, w))
            if t in targets and d["gr"][b, t, hh]:
                ts = targets[t]
                valid = Z > 1e-6
                Zs = torch.where(valid, Z, torch.ones_like(Z))
                sc = torch.where(valid, tau[hh] * (K @ q[b, t, hh]) / Zs, torch.full_like(Z, -1e300))
                pr = torch.softmax(sc, 0) * valid.double()
                written = bool(gw[b, ts, hh])
                stats["written"].append(float(written))
                stats["entropy"].append(float(-(pr.clamp_min(1e-12).log() * pr).sum()))
                share = torch.zeros(SLOTS)
                for m in range(SLOTS):
                    for (tw, w) in hist[m]:
                        if tw == ts:
                            share[m] = w * float(decay[m]) ** (t - tw) / float(Z[m])
                succ_slots = share > 0
                stats["attn_on_succ_slots"].append(float(pr[succ_slots].sum()))
                stats["succ_share"].append(float(share[succ_slots].max()) if written and succ_slots.any() else 0.0)
                stats["signal"].append(float((pr * share).sum()))
                if succ_slots.any():
                    m = int(share.argmax())
                    stats["writes_in_slot"].append(len(hist[m]))
mean = lambda x: sum(x) / max(1, len(x))
print(f"layer 0 hop 1, {len(stats['written'])} (second-copy step, head) pairs")
print(f"  successor written in this head:                 {mean(stats['written']):.0%}")
print(f"  read attention on the successor's slots:        {mean(stats['attn_on_succ_slots']):.2f}  (read entropy {mean(stats['entropy']):.2f} nats)")
print(f"  successor's share of its best slot's average:   {mean(stats['succ_share']):.2f}  (writes in that slot so far: {mean(stats['writes_in_slot']):.1f})")
print(f"  fraction of the read output that is the successor: {mean(stats['signal']):.3f}")
ok = [x for x, w in zip(stats['signal'], stats['written']) if w]
print(f"     when it was written: {mean(ok):.3f}")
