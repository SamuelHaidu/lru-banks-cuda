"""Checks the C++ model's loss and gradients against an independent float64 PyTorch implementation of the same math.

    build/release/train_bench --data TOKENS.npy --tiny --gate-w 0 --gate-r 0 --dump /tmp/lbc_dump
    python3 tests/check_grads.py /tmp/lbc_dump

The reference replays the discrete decisions of the C++ run (the two routed slots, the binary gates), so the
comparison is of the continuous math: bf16 activations and GEMM operands in C++ against float64 here. Gradients of
the binary gates are straight-through: the write gate's for every non-<bos> token, the read gate's for open reads
only (a closed read is never computed). Exit status 1 if a tensor's gradient misses the tolerance.
"""

import math
import sys
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DUMP = Path(ARGS[0] if ARGS else "/tmp/lbc_dump")
BF16 = "--bf16" in sys.argv  # round the forward to bf16 where the C++ stores bf16 (gradients pass straight through)


def bf(x):
    return x + (x.to(torch.bfloat16).double() - x).detach() if BF16 else x

REL_TOL, COS_TOL = 0.05, 0.998


def read(name, dtype):
    return torch.frombuffer(bytearray((DUMP / name).read_bytes()), dtype=dtype)


cfg = {}
for line in (DUMP / "config.txt").read_text().split("\n"):
    if line.strip():
        k, v = line.split()
        cfg[k] = float(v) if "." in v or "e" in v else int(v)
V, E0, D, L, N, H, F = (cfg[k] for k in ("V", "E0", "D", "L", "N", "H", "F"))
B, T, DH, SLOTS = cfg["B"], cfg["T"], cfg["DH"], cfg["SLOTS"]
HD = H * DH

params = read("params.bin", torch.float32).double()
grads_c = read("grads.bin", torch.float32).double()
P, Gc = {}, {}
for line in (DUMP / "manifest.txt").read_text().split("\n"):
    if line.strip():
        name, off, rows, cols = line.split()
        off, rows, cols = int(off), int(rows), int(cols)
        P[name] = params[off:off + rows * cols].reshape(rows, cols).clone().requires_grad_(True)
        Gc[name] = grads_c[off:off + rows * cols].reshape(rows, cols)

ids = read("ids.bin", torch.int32).long().reshape(B, T)
y = read("y.bin", torch.int32).long().reshape(B, T)
reset = read("reset.bin", torch.uint8).bool().reshape(B, T)
dec = []
for n in range(L):
    dec.append({k: read(f"l{n}_{k}.bin", torch.uint8).long() for k in ("ridx", "gw", "gr", "gr2")})
    dec[n]["ridx"] = dec[n]["ridx"].reshape(B, T, H, 2)
    for k in ("gw", "gr", "gr2"):
        dec[n][k] = dec[n][k].reshape(B, T, H)

hl = torch.exp(torch.linspace(math.log(cfg["slot_hl_min"]), math.log(cfg["slot_hl_max"]), SLOTS))
slot_decay = torch.exp((math.log(0.5) / hl).float().double())  # the C++ keeps log decays in fp32


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def gelu(z):
    return 0.5 * z * (1 + torch.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))


def ste(open_, logit, mask):
    """Forward: the binary decision; backward: d sigmoid(logit), where mask."""
    s = torch.sigmoid(logit)
    return open_.double() + (s - s.detach()) * mask.double()


def lru(Pin, nu, theta):
    rate = nu.exp()
    lr_, li = torch.exp(-rate) * torch.cos(theta), torch.exp(-rate) * torch.sin(theta)
    gam = torch.sqrt(-torch.expm1(-2 * rate))
    hr = torch.zeros(B, N)
    hi = torch.zeros(B, N)
    out = []
    for t in range(T):
        keep = (~reset[:, t]).double()[:, None]
        xr, xi = Pin[:, t, :N], Pin[:, t, N:]
        hr, hi = keep * (lr_ * hr - li * hi) + gam * xr, keep * (lr_ * hi + li * hr) + gam * xi
        out.append(torch.cat([hr, hi], -1))
    return torch.stack(out, 1)


def bank_reads(k, v, r1, r2, ridx, queries, gates, taus):
    """The banks of one layer (token loop) and its reads: queries / gates / taus, one per hop."""
    outs = [torch.zeros(B, T, H, DH) for _ in queries]
    outs = [list(torch.unbind(o.reshape(B * T * H, DH))) for o in outs]
    for b in range(B):
        for h in range(H):
            K = torch.zeros(SLOTS, DH)
            Vb = torch.zeros(SLOTS, DH)
            Z = torch.zeros(SLOTS)
            a = slot_decay
            for t in range(T):
                if reset[b, t]:
                    K, Vb, Z = K * 0, Vb * 0, Z * 0
                else:
                    K, Vb, Z = K * a[:, None], Vb * a[:, None], Z * a
                m1, m2 = int(ridx[b, t, h, 0]), int(ridx[b, t, h, 1])
                w = torch.zeros(SLOTS)
                w = w.index_put((torch.tensor([m1]),), r1[b, t, h].reshape(1))
                w = w.index_put((torch.tensor([m2]),), r2[b, t, h].reshape(1))
                K = K + w[:, None] * k[b, t, h][None]
                Vb = Vb + w[:, None] * v[b, t, h][None]
                Z = Z + w
                for i, (q, g, tau) in enumerate(zip(queries, gates, taus)):
                    if not bool(g[b, t, h] > 0.5):
                        continue
                    valid = Z.detach() > 1e-6
                    if not bool(valid.any()):
                        continue
                    Zs = torch.where(valid, Z, torch.ones_like(Z))
                    sc = torch.where(valid, tau[h] * (K @ q[b, t, h]) / Zs, torch.full_like(Z, -1e300))
                    p = torch.softmax(sc, 0) * valid.double()
                    rd = (p[:, None] * Vb / Zs[:, None]).sum(0)
                    outs[i][(b * T + t) * H + h] = g[b, t, h] * rd
    return [torch.stack(o).reshape(B, T, H * DH) for o in outs]


h = bf(P["E"])[ids] @ bf(P["W_in"])
for n in range(L):
    p_ = lambda k: P[f"l{n}_{k}"]
    d = dec[n]
    PV = bf(bf(rms(h)) @ bf(p_("Wa") * p_("norm1").T))
    s = lru(PV[..., :2 * N], p_("nu")[0], p_("theta")[0])
    G = bf(s) @ bf(p_("Ws"))
    kraw = G[..., :HD]
    kprev = torch.cat([torch.zeros(B, 1, HD), kraw[:, :-1]], 1) * (~reset).double()[..., None]
    k = bf(normalize((kprev + p_("bk")[0]).reshape(B, T, H, DH)))
    q = bf(normalize(G[..., HD:2 * HD].reshape(B, T, H, DH)))
    ar = G[..., 2 * HD:2 * HD + H] + p_("brg")[0]
    aw = G[..., 2 * HD + H:2 * HD + 2 * H] + p_("bwg")[0]
    R = p_("R").reshape(H, DH, SLOTS)
    logits_r = torch.einsum("bthd,hdm->bthm", k, R)
    l1 = logits_r.gather(-1, d["ridx"][..., :1])[..., 0]
    l2 = logits_r.gather(-1, d["ridx"][..., 1:])[..., 0]
    p1 = torch.sigmoid(l1 - l2)
    first = torch.zeros(B, T, dtype=torch.bool)
    first[:, 0] = True
    notreset = (~(reset | first))[..., None].expand(B, T, H)
    gw = ste(d["gw"], aw, notreset)
    gr = ste(d["gr"], ar, d["gr"].bool())
    v = PV[..., 2 * N:].reshape(B, T, H, DH)  # already bf16 (PV is)
    # hop 1, the FFN, then hop 2 (its query from the stream after the FFN)
    (o1,) = bank_reads(k, v, gw * p1, gw * (1 - p1), d["ridx"], [q], [gr], [p_("log_tau")[0].exp()])
    h = h + bf(o1) @ bf(p_("Wo"))
    z = bf(bf(bf(rms(h)) @ bf(p_("W1") * p_("ffn_norm").T)) + p_("b1")[0])
    h = h + bf(gelu(z)) @ bf(p_("W2")) + p_("b2")[0]
    G2 = bf(rms(h)) @ bf(p_("Wq2") * p_("q2_norm").T)
    q2 = bf(normalize(G2[..., :HD].reshape(B, T, H, DH)))
    ar2 = G2[..., HD:HD + H] + p_("brg2")[0]
    gr2 = ste(d["gr2"], ar2, d["gr2"].bool())
    (o2,) = bank_reads(k, v, gw * p1, gw * (1 - p1), d["ridx"], [q2], [gr2], [p_("log_tau2")[0].exp()])
    h = h + bf(o2) @ bf(p_("Wo2"))

hr = bf(bf(rms(h)) @ bf(P["W_out"] * P["final_norm"].T))
logits = hr @ bf(P["E"]).T
mask = y >= 0
loss = torch.nn.functional.cross_entropy(logits[mask], y[mask])
loss.backward()

print(f"loss: C++ {cfg['loss']:.6f}  reference {loss.item():.6f}  (diff {abs(cfg['loss'] - loss.item()):.2e})")
bad = []
print(f"{'tensor':16s} {'|grad| ref':>11s} {'rel err':>8s} {'cosine':>8s}")
for name, t in P.items():
    gr_ = t.grad if t.grad is not None else torch.zeros_like(t)
    gc = Gc[name]
    if "Ws" in name or "Wq2" in name:  # padding columns past the real outputs carry no gradient anywhere
        width = 2 * HD + 2 * H if "Ws" in name else HD + H
        gr_, gc = gr_[:, :width], gc[:, :width]
    nr = gr_.norm().item()
    if nr < 1e-12 and gc.norm().item() < 1e-12:
        print(f"{name:16s} {0.0:11.3e} {'-':>8s} {'-':>8s}")
        continue
    rel = (gc - gr_).norm().item() / max(nr, 1e-30)
    cos = torch.nn.functional.cosine_similarity(gc.flatten(), gr_.flatten(), 0).item()
    ok = rel < REL_TOL and cos > COS_TOL
    flag = "" if ok else "   <-- MISMATCH"
    if not ok:
        bad.append(name)
    if gc.numel() <= 16:  # small tensors: the values themselves (a near-cancelling sum shows as noise here)
        flag += "   C++ " + " ".join(f"{v:+.3e}" for v in gc.flatten().tolist()) + "  ref " + \
            " ".join(f"{v:+.3e}" for v in gr_.flatten().tolist())
    print(f"{name:16s} {nr:11.3e} {rel:8.4f} {cos:8.5f}{flag}")
print("PASS" if not bad else f"FAIL: {', '.join(bad)}")
sys.exit(1 if bad else 0)
