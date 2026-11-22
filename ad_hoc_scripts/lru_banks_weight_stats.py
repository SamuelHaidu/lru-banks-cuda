"""Weight statistics of an `lru-banks` checkpoint, layer by layer, looking for patterns:
- matrices: std, excess kurtosis (Gaussian 0), max |w| / std, spectrum (effective rank = exp(entropy of the normalized
  singular values), stable rank = |W|_F^2 / s_max^2, top singular value's share of the energy);
- stream channels: per matrix the norm of each stream dim it reads (rows) or writes (columns); the channels far above
  the median, and whether the same channels recur across layers and matrices (outlier / "massive" dims);
- norms' gains; the LRU's learned half-lives and rotation periods; read temperatures; the input gate; FFN biases
  (and neurons whose input weights or output weights are near 0); the router's column norms against the slots'
  half-lives; the embedding's row norms.

    uv run python ad_hoc_scripts/lru_banks_weight_stats.py --run-name X [--checkpoint best]
"""

import argparse
import math

import numpy as np


def spectrum(W):
    s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    e = s ** 2 / (s ** 2).sum()
    return math.exp(-(e * np.log(e + 1e-30)).sum()), (s ** 2).sum() / s[0] ** 2, e[0], min(W.shape)


def kurt(x):
    x = x.ravel().astype(np.float64)
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2 - 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--top", type=int, default=8)
    a = p.parse_args()
    from model.device import setup_device

    setup_device("cpu")
    from model.evaluate import load_checkpoint

    m = load_checkpoint(a.run_name, int(a.checkpoint) if a.checkpoint.isdigit() else a.checkpoint)
    P = [{k: v.value.detach().float().cpu().numpy() for k, v in p.items()} for p in m._params]
    L, H, Dh, M = m.num_layers, m.num_heads, m.head_dim, m.slots
    print(f"{a.run_name} ({a.checkpoint}): {L} layers, hops {m.layer_hops}")

    print("\n== matrices: std, kurtosis, max/std, effective rank / dims, stable rank, top singular share ==")
    mats = ["w_lru", "w_k", "w_q", "w_v", "w_o", "w_q2", "w_o2", "w1", "w2"]
    for k in mats:
        for n in range(L):
            if k not in P[n]:
                continue
            W = P[n][k]
            er, sr, top, d = spectrum(W)
            print(f"  {k:6s} L{n + 1} {str(W.shape):12s} std {W.std():.4f}  kurt {kurt(W):6.2f}  max/std "
                  f"{np.abs(W).max() / W.std():5.1f}  eff.rank {er:6.1f}/{d}  stable {sr:6.1f}  top {top:.3f}")

    print(f"\n== stream channels (dim {m.dim}): norm per channel / median; channels > 3x the median ==")
    reads = {k: 0 for k in ("w_lru", "w_v", "w1", "w_q2")}  # rows = stream dims read (after the norm's gain)
    writes = ("w_o", "w_o2", "w2")  # columns = stream dims written
    gains = {"w_lru": "norm1", "w_v": "norm1", "w1": "ffn_norm1", "w_q2": "q2_norm"}
    hot = {}
    for n in range(L):
        for k in list(reads) + list(writes):
            if k not in P[n]:
                continue
            W = P[n][k]
            c = np.linalg.norm(W, axis=1) * np.abs(P[n][gains[k]]) if k in reads else np.linalg.norm(W, axis=0)
            r = c / np.median(c)
            big = np.flatnonzero(r > 3)
            hot[(n, k)] = set(big.tolist())
            order = np.argsort(-r)[:a.top]
            print(f"  L{n + 1} {k:5s} {'reads' if k in reads else 'writes'}: max {r.max():5.1f}x, "
                  f"{len(big):3d} > 3x; top dims {', '.join(f'{i}({r[i]:.1f})' for i in order)}")
    counts = np.zeros(m.dim, int)
    for s in hot.values():
        for i in s:
            counts[i] += 1
    rec = np.argsort(-counts)[:16]
    print(f"  dims above 3x in the most (layer, matrix) pairs (of {len(hot)}): "
          + ", ".join(f"{i}:{counts[i]}" for i in rec if counts[i] > 1))
    fn = m.final_norm.value.detach().float().cpu().numpy()
    wout = m.w_out.value.detach().float().cpu().numpy()
    ro = np.linalg.norm(wout, axis=1) * np.abs(fn)
    print(f"  readout (final_norm x W_out rows): max {ro.max() / np.median(ro):.1f}x; top dims "
          + ", ".join(f"{i}({ro[i] / np.median(ro):.1f})" for i in np.argsort(-ro)[:a.top]))

    print("\n== norm gains: mean, std, min, max ==")
    for k in ("norm1", "ffn_norm1", "q2_norm", "ffn_norm2"):
        for n in range(L):
            if k in P[n]:
                g = P[n][k]
                print(f"  {k:9s} L{n + 1}: mean {g.mean():.3f} std {g.std():.3f} min {g.min():.3f} max {g.max():.3f}")
    print(f"  final_norm   : mean {fn.mean():.3f} std {fn.std():.3f} min {fn.min():.3f} max {fn.max():.3f}")

    print("\n== LRU: half-lives (tokens) and rotation periods (tokens; 2 pi / theta) ==")
    for n in range(L):
        hl = math.log(2) / np.exp(P[n]["nu"])
        th = np.abs((P[n]["theta"] + np.pi) % (2 * np.pi) - np.pi)
        per = 2 * np.pi / np.maximum(th, 1e-9)
        q = lambda x: " ".join(f"{v:7.2f}" for v in np.percentile(x, [0, 10, 50, 90, 100]))
        print(f"  L{n + 1} half-life p0/10/50/90/100 {q(hl)}  | period {q(per)}  | channels with half-life > 16: "
              f"{int((hl > 16).sum())}, near-pi rotation (period < 2.2): {int((per < 2.2).sum())}")

    print("\n== reads: temperatures tau per head ==")
    for n in range(L):
        t1 = np.exp(P[n]["log_tau"])
        t2 = np.exp(P[n]["log_tau2"]) if "log_tau2" in P[n] else None
        print(f"  L{n + 1} hop 1: " + " ".join(f"{v:5.1f}" for v in t1)
              + ("" if t2 is None else "\n     hop 2: " + " ".join(f"{v:5.1f}" for v in t2)))

    print("\n== input gate: bias, |w|, sigmoid(bias) ==")
    for n in range(L):
        if "w_gate" in P[n]:
            b, w = float(P[n]["gate_bias"][0]), P[n]["w_gate"][:, 0] * P[n]["norm1"]
            print(f"  L{n + 1}: bias {b:.3f} (sigmoid {1 / (1 + math.exp(-b)):.3f}), |w| {np.linalg.norm(w):.3f}, "
                  f"top dims {', '.join(str(i) for i in np.argsort(-np.abs(w))[:a.top])}")

    print("\n== FFN neurons (4 x dim): bias b1 percentiles; neurons with input or output norm < 10% of the median ==")
    for n in range(L):
        w1, b1, w2 = P[n]["w1"], P[n]["b1_bias"], P[n]["w2"]
        cin, cout = np.linalg.norm(w1, axis=0), np.linalg.norm(w2, axis=1)
        dead_in, dead_out = (cin < 0.1 * np.median(cin)).sum(), (cout < 0.1 * np.median(cout)).sum()
        corr = np.corrcoef(cin, cout)[0, 1]
        print(f"  L{n + 1}: b1 p1/50/99 {np.percentile(b1, 1):.2f} {np.median(b1):.2f} {np.percentile(b1, 99):.2f}; "
              f"in-norm spread (p99/p1) {np.percentile(cin, 99) / np.percentile(cin, 1):.1f}, out {np.percentile(cout, 99) / np.percentile(cout, 1):.1f}; "
              f"tiny in {dead_in}, tiny out {dead_out}; corr(in, out norm) {corr:.2f}")

    print("\n== router: column norm per slot (sharpness of that slot's logit) vs slot half-life ==")
    shl = np.log(0.5) / P[0]["loga"][0]
    groups = [(0, 8), (8, 32), (32, 128), (128, 2000)]
    for n in range(L):
        cn = np.linalg.norm(P[n]["router"], axis=1)  # (H, M)
        parts = [f"hl {lo}-{hi}: {cn[:, (shl >= lo) & (shl < hi)].mean():.2f}" for lo, hi in groups]
        print(f"  L{n + 1}: mean {cn.mean():.2f}, per head min/max of mean {cn.mean(1).min():.2f}/{cn.mean(1).max():.2f}; "
              + ", ".join(parts) + f"; corr(norm, log half-life) {np.corrcoef(cn.mean(0), np.log(shl))[0, 1]:.2f}")
        kb = P[n]["k_bias"].reshape(H, Dh)
        print(f"       key bias norm per head: {' '.join(f'{v:.2f}' for v in np.linalg.norm(kb, axis=1))}")

    E = m.token_embedding.embeddings.value.detach().float().cpu().numpy()
    en = np.linalg.norm(E, axis=1)
    print(f"\n== embedding ({E.shape}): row norm p1/50/99 {np.percentile(en, 1):.3f} {np.median(en):.3f} "
          f"{np.percentile(en, 99):.3f}; kurtosis {kurt(E):.2f}; eff. rank {spectrum(E)[0]:.1f}/{E.shape[1]}")


if __name__ == "__main__":
    main()
