"""`rope-state-memory`: learned retention (state / memory decays) and read-gate selectivity of a checkpoint.

    uv run python ad_hoc_scripts/rsm_gate_retention.py --run-name rsm-1L-prenorm [--checkpoint 1] [--rows 32]

Runs on the CPU (does not compete with a training run on the XPU). Reads `rows` validation rows of `window_size`
tokens, each starting at a document start (packed documents, as in training), and reports per layer:
  - lambda / mu and their half-lives (vs. the init);
  - the read gate g = sigmoid(w_g . q + b) over every position: distribution, by position in the document,
    by current token; its weight norm and bias;
  - the size of the gated memory read |g o| against the state |s|;
  - the loss with the memory read removed (g = 0) and with the gate forced open (g = 1), against the model.
"""

import argparse
import math

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()
setup_device("cpu")

import torch  # noqa: E402

from model import config as cfg_mod  # noqa: E402
import model.archs  # noqa: E402,F401
from model.evaluate import load_checkpoint  # noqa: E402
from model.scripts._model_io import load_tokenizer  # noqa: E402


def rows_from_validation(config, n_rows, length, seed=0):
    ds = config["data"]["dataset"]
    meta = cfg_mod.load_dataset_metadata(ds) if hasattr(cfg_mod, "load_dataset_metadata") else None
    import json

    meta = meta or json.loads((cfg_mod.DATASETS_DIR / ds / "metadata.json").read_text())
    tokens = np.load(cfg_mod.DATASETS_DIR / ds / meta["validation"]["file"], mmap_mode="r")
    bos = int(meta["doc_token_id"])
    starts = np.flatnonzero(np.asarray(tokens[: min(len(tokens), 5_000_000)]) == bos)
    rng = np.random.default_rng(seed)
    pick = rng.choice(starts[starts + length + 1 < len(tokens)], n_rows, replace=False)
    seq = np.stack([np.asarray(tokens[p:p + length + 1]) for p in pick]).astype(np.int64)
    return seq[:, :-1], seq[:, 1:], bos


def dense_loss(m, x, y, bos):
    with torch.no_grad():
        h = m.dense_hidden(torch.as_tensor(x, dtype=torch.int32))
        logits = m.readout(m.readout_input(h))
        lp = torch.log_softmax(torch.as_tensor(logits), -1)
        tgt = torch.as_tensor(y)
        nll = -lp.gather(-1, tgt[..., None])[..., 0]
        valid = tgt != bos
    return nll, valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--rows", type=int, default=32)
    a = p.parse_args()
    config = cfg_mod.load(a.run_name)
    m = load_checkpoint(a.run_name, a.checkpoint)
    setup_device("cpu")
    m.use_kernels = False
    tok = load_tokenizer(config)
    T = m.window_size
    x, y, bos = rows_from_validation(config, a.rows, T)
    print(f"run {a.run_name}  checkpoint {a.checkpoint}  layers {m.num_layers}  rows {a.rows} x {T}")

    for n in range(m.num_layers):
        lam, mu = (v.detach().cpu().numpy().astype("float64") for v in m.decays(n))
        lam = float(lam)
        hl = lambda r: np.log(2) / -np.log(r)
        print(f"\n== layer {n + 1}")
        print(f"state  decay lambda {lam:.5f}  half-life {hl(lam):7.2f} tokens   (init {m.state_half_life})")
        if m.memory == "linear":
            mu = float(mu)
            print(f"memory decay mu     {mu:.5f}  half-life {hl(mu):7.2f} tokens   (init {m.memory_half_life})")
            for d, name in ((1, "next"), (10, "10 back"), (50, "50 back"), (200, "200 back")):
                print(f"   weight of a token {name:>8}: state {lam ** d:.3f}  memory {mu ** d:.3f}")
        else:
            h = np.sort(hl(mu))
            init = np.geomspace(m.memory_half_life_min, m.memory_half_life_max, m.num_banks)
            q = lambda a: " ".join(f"{v:7.1f}" for v in np.percentile(a, [0, 10, 25, 50, 75, 90, 100]))
            print(f"bank half-lives min/10/25/50/75/90/max: {q(h)}")
            print(f"               init                     {q(init)}")
            nrm = lambda name: float(np.linalg.norm(m._w(name, n).value.detach().cpu().numpy()))
            print(f"address projections |W_q| {nrm('address_q'):.3f}  |W_k| {nrm('address_k'):.3f}  (init 0: uniform)")

    # gates over real text
    with torch.no_grad():
        ids = torch.as_tensor(x, dtype=torch.int32)
        E = m.token_embedding.embeddings.value
        reset = ids == m.reset_token
        xs = torch.nn.functional.embedding(ids.long(), E)
        B = xs.shape[0]
        for n in range(m.num_layers):
            u = m._norm(m._w("in_norm", n), xs) if m.pre_norm else xs
            zero = (E.new_zeros(B, m.dim), E.new_zeros(B, m.memory_rows, m.memory_cols))
            h, _, (s, o) = m._layer(n, u, reset, zero, collect=True, fresh=True)
            read = s @ m._w("address_q", n).value
            g = m._gate(n, m.read_gate_in(s, read))[..., 0].numpy()  # (B, T)
            if m.memory == "banks":  # how peaked / how spread the bank addressing is
                prev = torch.cat([zero[0][:, None], s[:, :-1]], 1)
                write = torch.where(reset[..., None], 0.0, prev) @ m._w("address_k", n).value
                r, w, mu_b = m._addresses(n, read, write)
                w = w / (1 - mu_b)
                ent = lambda p: float((-(p * torch.log(p + 1e-12)).sum(-1)).exp().mean())
                use = w.reshape(-1, w.shape[-1]).mean(0).numpy()
                print(f"\n== layer {n + 1} banks: effective banks per read {ent(r):.1f}, per write {ent(w):.1f} "
                      f"(of {m.num_banks}); banks holding 90% of the writes: "
                      f"{int(np.searchsorted(np.cumsum(np.sort(use)[::-1]), 0.9) + 1)}")
            so = (o.norm(dim=-1) / s.norm(dim=-1).clamp_min(1e-9)).numpy()  # o is already gated
            w = m._w("read_gate_w", n).value.numpy()[:, 0]
            b = float(m._w("read_gate_b", n).value.numpy()[0])
            print(f"\n== layer {n + 1} read gate   |w_g| {np.linalg.norm(w):.3f}  b_g {b:+.3f}  "
                  f"(init 0, {m.read_gate_bias:+.1f}; sigmoid(b) = {1 / (1 + math.exp(-b)):.4f})")
            q = np.percentile(g, [1, 10, 25, 50, 75, 90, 99])
            print("   g percentiles 1/10/25/50/75/90/99: " + " ".join(f"{v:.3f}" for v in q))
            print(f"   mean {g.mean():.3f}  std {g.std():.3f}   g<0.1 {np.mean(g < 0.1):.1%}  g>0.5 {np.mean(g > 0.5):.1%}"
                  f"  g>0.9 {np.mean(g > 0.9):.1%}")
            print(f"   |g o| / |s|: median {np.median(so):.3f}  90% {np.percentile(so, 90):.3f}")
            # position since the last document start
            pos = np.zeros_like(x)
            for r in range(B):
                last = -1
                for t in range(T):
                    if x[r, t] == bos:
                        last = t
                    pos[r, t] = t - last if last >= 0 else t + 1000
            for lo, hi in ((0, 1), (1, 5), (5, 20), (20, 50), (50, 100), (100, 256)):
                sel = (pos >= lo) & (pos < hi)
                if sel.any():
                    print(f"   doc position {lo:>3}-{hi:<3}: mean g {g[sel].mean():.3f}  (n={sel.sum()})")
            # by current token
            stats = {}
            for tid, gv in zip(x.ravel(), g.ravel()):
                stats.setdefault(int(tid), []).append(gv)
            agg = [(np.mean(v), len(v), t) for t, v in stats.items() if len(v) >= 15]
            agg.sort()
            name = lambda t: repr(tok.id_to_token(t)) if tok else str(t)
            print("   lowest-gate tokens:  " + ", ".join(f"{name(t)} {mg:.2f}" for mg, _, t in agg[:10]))
            print("   highest-gate tokens: " + ", ".join(f"{name(t)} {mg:.2f}" for mg, _, t in agg[-10:][::-1]))
            within = np.mean([np.var(v) for v in stats.values() if len(v) >= 15])
            print(f"   variance of g: total {g.var():.4f}, within the same token {within:.4f} "
                  f"({within / max(g.var(), 1e-12):.0%} of it depends on context, not on the token)")

    # what reading is worth
    nll, valid = dense_loss(m, x, y, bos)
    base = float(nll[valid].mean())
    orig = m._gate
    res = {}
    for tag, val in (("memory off (g = 0)", 0.0), ("gate forced open (g = 1)", 1.0)):
        m._gate = (lambda v: (lambda n, q: torch.full(q.shape[:-1] + (1,), v)))(val)
        l2, _ = dense_loss(m, x, y, bos)
        res[tag] = l2
        m._gate = orig
    print(f"\n== loss over {int(valid.sum())} targets: model {base:.4f}")
    for tag, l2 in res.items():
        print(f"   {tag}: {float(l2[valid].mean()):.4f}  ({float(l2[valid].mean()) - base:+.4f})")
    gain = (res["memory off (g = 0)"] - nll).numpy()
    vm = valid.numpy()
    g_all = g if m.num_layers == 1 else None
    if g_all is not None:
        for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)):
            sel = vm & (g_all >= lo) & (g_all < hi)
            if sel.any():
                print(f"   where g in [{lo:.1f}, {hi:.1f}): loss saved by reading {gain[sel].mean():+.4f} nats/token (n={sel.sum()})")


if __name__ == "__main__":
    main()
