"""Which memory timescales each read of an lbc model uses (generated from analyze_copy.py's replay): for every
(layer, hop), the attention mass over slot half-life bands, the attention entropy and the mean age of what is read
(attention-weighted tokens since the slot's dominant write), on rows dumped by `lbc dump --rows 2`.

    python3 scripts/analyze_reads.py /tmp/lbc_inspect

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


def bank_reads(k, v, r1, r2, ridx, queries, gates, taus, decay, record):
    """The banks of one layer and its reads (one per query). record: per read, the attention p [B, T, H, SLOTS] and
    each slot's dominant writer position at read time [B, T, H, SLOTS] (the write with the largest decayed weight)."""
    outs = [torch.zeros(B, T, H, DH) for _ in queries]
    rec_p = [torch.zeros(B, T, H, SLOTS) for _ in queries]
    rec_w = torch.full((B, T, H, SLOTS), -1, dtype=torch.long)
    for b in range(B):
        for h in range(H):
            K, Vb, Z = torch.zeros(SLOTS, DH), torch.zeros(SLOTS, DH), torch.zeros(SLOTS)
            dom_pos = torch.full((SLOTS,), -1, dtype=torch.long)
            dom_w = torch.zeros(SLOTS)
            for t in range(T):
                if reset[b, t]:
                    K, Vb, Z = K * 0, Vb * 0, Z * 0
                    dom_pos[:], dom_w[:] = -1, 0
                else:
                    K, Vb, Z, dom_w = K * decay[:, None], Vb * decay[:, None], Z * decay, dom_w * decay
                m1, m2 = int(ridx[b, t, h, 0]), int(ridx[b, t, h, 1])
                for m, w in ((m1, r1[b, t, h]), (m2, r2[b, t, h])):
                    if w > 0:
                        K[m] += w * k[b, t, h]
                        Vb[m] += w * v[b, t, h]
                        Z[m] += w
                        if w > dom_w[m]:
                            dom_w[m], dom_pos[m] = w, t
                rec_w[b, t, h] = dom_pos
                for i, (q, g, tau) in enumerate(zip(queries, gates, taus)):
                    if not bool(g[b, t, h]):
                        continue
                    valid = Z > 1e-6
                    if not bool(valid.any()):
                        continue
                    Zs = torch.where(valid, Z, torch.ones_like(Z))
                    sc = torch.where(valid, tau[h] * (K @ q[b, t, h]) / Zs, torch.full_like(Z, -1e300))
                    p = torch.softmax(sc, 0) * valid.double()
                    outs[i][b, t, h] = (p[:, None] * Vb / Zs[:, None]).sum(0)
                    rec_p[i][b, t, h] = p
    if record is not None:
        record.append((rec_p, rec_w))
    return [o.reshape(B, T, HD) for o in outs]


def forward(silence=(), record=None):
    """Logits [B, T, V]; silence: (layer, hop) reads set to 0; record: per layer (attention per read, dominant writers)."""
    h = P["E"][ids] @ P["W_in"] if FACT else P["E"][ids].clone()
    for n in range(L):
        p_ = lambda key: P[f"l{n}_{key}"]
        d = dec[n]
        PV = rms(h) @ (p_("Wa") * p_("norm1").T)
        s = lru(PV[..., :2 * N], p_("nu")[0], p_("theta")[0])
        G = s @ p_("Ws")
        kprev = torch.cat([torch.zeros(B, 1, HD), G[..., :HD][:, :-1]], 1) * (~reset).double()[..., None]
        k = normalize((kprev + p_("bk")[0]).reshape(B, T, H, DH))
        q = normalize(G[..., HD:2 * HD].reshape(B, T, H, DH))
        R = p_("R").reshape(H, DH, SLOTS)
        lr_ = torch.einsum("bthd,hdm->bthm", k, R)
        p1 = torch.sigmoid(lr_.gather(-1, d["ridx"][..., :1])[..., 0] - lr_.gather(-1, d["ridx"][..., 1:])[..., 0])
        gw = d["gw"].double()
        v = PV[..., 2 * N:].reshape(B, T, H, DH)
        decay = slot_decays(NSLOT[n])
        rec = [] if record is not None else None
        (o1,) = bank_reads(k, v, gw * p1, gw * (1 - p1), d["ridx"], [q], [d["gr"]], [p_("log_tau")[0].exp()], decay,
                           rec)
        if (n, 1) in silence:
            o1 = o1 * 0
        h = h + o1 @ p_("Wo")
        h = h + gelu(rms(h) @ (p_("W1") * p_("ffn_norm").T) + p_("b1")[0]) @ p_("W2") + p_("b2")[0]
        if HOPS[n] == 2:
            G2 = rms(h) @ (p_("Wq2") * p_("q2_norm").T)
            q2 = normalize(G2[..., :HD].reshape(B, T, H, DH))
            (o2,) = bank_reads(k, v, gw * p1, gw * (1 - p1), d["ridx"], [q2], [d["gr2"]], [p_("log_tau2")[0].exp()],
                               decay, rec)
            if (n, 2) in silence:
                o2 = o2 * 0
            h = h + o2 @ p_("Wo2")
        if record is not None:
            record.append((rec, d["ridx"]))
    hr = rms(h) @ (P["W_out"] * P["final_norm"].T) if FACT else rms(h) * P["final_norm"][0]
    return hr @ (P["E"] if TIED else P["E_out"]).T



record = []
logits = forward(record=record)
mask = y >= 0
ce = torch.nn.functional.cross_entropy(logits[mask], y[mask]).item()
print(f"replay vs C++: loss {ce:.4f} vs {cfg['loss']:.4f}")
bands = [(1, 4), (4, 16), (16, 64), (64, 256), (256, 1025)]
print("attention mass by the half-life of the slot read (tokens), entropy (nats), mean age of the read content (tokens)")
for n in range(L):
    rec, _ = record[n]
    ns = NSLOT[n]
    hl = torch.exp(torch.linspace(math.log(cfg["slot_hl_min"]), math.log(cfg["slot_hl_max"]), ns))
    hl = torch.cat([hl, hl[-1:].expand(SLOTS - ns)])
    for hop, (rp, rw) in enumerate(rec, 1):
        p = rp[0]  # B T H SLOTS
        tot = p.sum()
        if tot <= 0:
            continue
        mass = [float(p[..., (hl >= lo) & (hl < hi)].sum() / tot) for lo, hi in bands]
        ent = -(p.clamp_min(1e-12).log() * p).sum(-1)
        valid = p.sum(-1) > 0
        t_idx = torch.arange(T).view(1, T, 1, 1).expand_as(rw)
        age = (t_idx - rw).clamp_min(0).double()
        age = torch.where(rw >= 0, age, torch.zeros_like(age))
        mean_age = float((p * age).sum() / tot)
        print(f"  layer {n} hop {hop}: " + "  ".join(f"hl {lo}-{hi - 1 if hi > 1000 else hi}: {m:.2f}" for (lo, hi), m in zip(bands, mass))
              + f"   entropy {float(ent[valid].mean()):.2f}   mean age {mean_age:.0f}")
