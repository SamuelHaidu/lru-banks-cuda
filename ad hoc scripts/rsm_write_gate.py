"""`rope-state-memory` banks: is the write gate selective, and how concentrated are the writes, the reads and the
read's attention over tokens? Compares runs on the same validation rows.

    uv run python "ad hoc scripts/rsm_write_gate.py" --runs rsm-1L-wgate:2 rsm-1L-wgate:1 rsm-1L-banks:1 [--rows 32]

`run:checkpoint` (checkpoint = epoch number, `latest` or `best`). CPU only. Layer 1, packed validation rows starting
at a document start (as in training). Per run:
  - write gate beta_t (if any): only ratios between tokens matter (the read divides by the written mass), so it is
    reported relative to the row's median: spread, the share of tokens effectively skipped, the share of all write
    mass carried by the top 10% of tokens, the effective fraction of writing tokens (sum beta)^2 / (N sum beta^2);
    which tokens write most / least; and the loss with the gate made uniform (= no gate);
  - write / read softmax over banks: effective banks (exp entropy), top-1 weight;
  - the read as attention a_tj over past tokens (exact, from the banks): effective tokens attended, top-1 weight,
    mean distance, mass on the last 1 / 4 tokens;
  - sink banks: banks whose read share is < 0.2 x their (effective) write share.
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()
setup_device("cpu")

import torch  # noqa: E402

from model import config as cfg_mod  # noqa: E402
import model.archs  # noqa: E402,F401
from model.evaluate import load_checkpoint  # noqa: E402
from model.scripts._model_io import load_tokenizer  # noqa: E402


def rows(config, n, length, seed=0):
    ds = config["data"]["dataset"]
    meta = json.loads((cfg_mod.DATASETS_DIR / ds / "metadata.json").read_text())
    tokens = np.load(cfg_mod.DATASETS_DIR / ds / meta["validation"]["file"], mmap_mode="r")
    head = np.asarray(tokens[: min(len(tokens), 5_000_000)])
    starts = np.flatnonzero(head == int(meta["doc_token_id"]))
    pick = np.random.default_rng(seed).choice(starts[starts + length + 1 < len(head)], n, replace=False)
    seq = np.stack([head[p:p + length + 1] for p in pick]).astype(np.int64)
    return seq[:, :-1], seq[:, 1:]


def loss(m, x, y):
    with torch.no_grad():
        lp = torch.log_softmax(torch.as_tensor(m.readout(m.readout_input(m.dense_hidden(torch.as_tensor(x, dtype=torch.int32))))), -1)
        t = torch.as_tensor(y)
        ok = t != m.reset_token
        return float(-lp.gather(-1, t[..., None])[..., 0][ok].mean())


def analyze(run, ckpt, x, y, tok):
    m = load_checkpoint(run, ckpt)
    setup_device("cpu")
    m.use_kernels = False
    name = lambda t: repr(tok.id_to_token(int(t)).replace("Ġ", " ")) if tok else str(int(t))
    B, N = x.shape
    S = m.num_banks
    with torch.no_grad():
        ids = torch.as_tensor(x, dtype=torch.int32)
        E = m.token_embedding.embeddings.value
        reset = ids == m.reset_token
        u = torch.nn.functional.embedding(ids.long(), E)
        u = m._norm(m._w("in_norm", 0), u) if m.pre_norm else u
        s, _ = m._state_torch(0, u, reset, E.new_zeros(B, m.dim))
        prev = torch.cat([E.new_zeros(B, 1, m.dim), s[:, :-1]], 1)
        prev0 = torch.where(reset[..., None], 0.0, prev)
        a_w = prev0 @ m._w("address_k", 0).value
        g_in = prev0 if m.write_gate_input == "state" else a_w
        w = torch.softmax(a_w, -1).double()
        r = torch.softmax(s @ m._w("address_q", 0).value, -1).double()
        beta = m._write_gate(0, g_in)[..., 0].double() if m.write_gate else torch.ones(B, N, dtype=torch.float64)
        mu = m.decays(0)[1].detach().double()
    valid = ~reset.numpy()
    ent = lambda p: torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum(-1))
    out = {"loss": loss(m, x, y)}
    print(f"\n=== {run} @ {ckpt}: loss {out['loss']:.4f} on these rows")

    if m.write_gate:
        b = beta.numpy()
        rel = np.stack([row / np.median(row[v]) for row, v in zip(b, valid)])  # per row: relative to its median
        rv = rel[valid]
        print(f"write gate  |w_w| {float(m.write_gate_w.value.norm()):.2f}  b_w {float(m.write_gate_b.value[0]):+.2f}  "
              f"(init 0, {m.write_gate_bias:+.1f});  absolute beta median {np.median(b[valid]):.2e}")
        print("   beta / row median, percentiles 1/10/25/50/75/90/99: "
              + " ".join(f"{v:.3g}" for v in np.percentile(rv, [1, 10, 25, 50, 75, 90, 99])))
        print(f"   tokens effectively skipped (beta < 0.1 x median): {np.mean(rv < 0.1):.1%};  < 0.01 x: {np.mean(rv < 0.01):.1%};"
              f"  > 10 x median: {np.mean(rv > 10):.1%}")
        tops, effs = [], []
        for row, v in zip(b, valid):
            g = np.sort(row[v])[::-1]
            tops.append(g[: max(1, len(g) // 10)].sum() / g.sum())
            effs.append(g.sum() ** 2 / (len(g) * (g ** 2).sum()))
        print(f"   write mass carried by the top 10% of tokens: {np.mean(tops):.0%} (uniform 10%);  effective fraction of "
              f"writing tokens (sum b)^2 / (N sum b^2): {np.mean(effs):.0%}")
        stats = defaultdict(list)
        for t, g in zip(x[valid], rv):
            stats[int(t)].append(np.log10(g))
        agg = sorted((np.mean(v), len(v), t) for t, v in stats.items() if len(v) >= 15)
        print("   writes least (mean log10 beta/median):  " + ", ".join(f"{name(t)} {m_:+.1f}" for m_, _, t in agg[:12]))
        print("   writes most:                            " + ", ".join(f"{name(t)} {m_:+.1f}" for m_, _, t in agg[-12:][::-1]))
        within = np.mean([np.var(v) for v in stats.values() if len(v) >= 15])
        print(f"   variance of log10 beta: total {np.var(np.log10(rv)):.3f}, within the same token {within:.3f} "
              f"({within / np.var(np.log10(rv)):.0%} depends on context, not on the token)")
        keep = m._write_gate
        m._write_gate = lambda n, a: torch.full(a.shape[:-1] + (1,), 0.5, dtype=a.dtype)
        out["loss_uniform_gate"] = loss(m, x, y)
        m._write_gate = keep
        print(f"   loss with a uniform gate (= no gate): {out['loss_uniform_gate']:.4f} ({out['loss_uniform_gate'] - out['loss']:+.4f})")

    if getattr(m, "read_mix", "add") == "gates" or getattr(m, "mlp_residual", "input") == "gates":
        mix_gates(m, x, y, valid, name, out)

    wv, rv_ = w[torch.as_tensor(valid)], r[torch.as_tensor(valid)]
    print(f"write softmax: effective banks median {ent(wv).median():.1f}, top-1 weight median {wv.max(-1).values.median():.2f};  "
          f"read softmax: effective banks median {ent(rv_).median():.1f}, top-1 median {rv_.max(-1).values.median():.2f}")

    # the read as attention over tokens: a_tj = sum_b r_tb c_jb(t) / sum_b r_tb z_b(t), c_jb(t) = (1 - mu_b) mu_b^(t-j) beta_j w_jb
    pos = torch.arange(N)
    d = (pos[:, None] - pos[None, :]).double()
    causal = d >= 0
    eff, top, dist, last1, last4 = [], [], [], [], []
    copy_all, copy_far = [], []
    sharp = []  # per read: (top weight, top token == the next token, top token is the current token, distance of top)  # (mass on earlier occurrences of the next token, the same under uniform attention)
    zsum = torch.zeros(S, dtype=torch.float64)
    wsum = torch.zeros(S, dtype=torch.float64)
    for i in range(B):
        seg = np.cumsum(x[i] == m.reset_token)
        same = torch.as_tensor(seg[:, None] == seg[None, :]) & causal
        we = w[i] * beta[i][:, None] * (1 - mu)  # (N, S) effective writes
        z = torch.zeros(S, dtype=torch.float64)
        A = torch.zeros(N, N, dtype=torch.float64)
        for t in range(N):
            c = we * mu[None] ** d[t, :, None].clamp_min(0) * same[t, :, None]  # (N, S)
            num = c @ r[i, t]
            A[t] = num / num.sum().clamp_min(1e-300)
            if valid[i, t]:
                zt = c.sum(0)
                zsum += r[i, t] * zt / (r[i, t] * zt).sum()  # read share per bank
                wsum += w[i, t] * beta[i, t]
        v = torch.as_tensor(valid[i]) & torch.as_tensor(np.r_[False, seg[1:] == seg[:-1]])  # not a doc's first token
        for t in torch.nonzero(v)[:, 0].tolist():
            a = A[t, : t + 1]
            eff.append(float(ent(a)))
            top.append(float(a.max()))
            dist.append(float((a * d[t, : t + 1]).sum()))
            last1.append(float(a[t]))
            last4.append(float(a[max(0, t - 3): t + 1].sum()))
            jt = int(a.argmax())
            sharp.append((float(a[jt]), int(x[i, jt] == y[i, t]), int(jt == t), t - jt))
            if y[i, t] == m.reset_token:
                continue
            j0 = int(np.flatnonzero(seg[: t + 1] == seg[t])[0])
            hit = torch.as_tensor(x[i, j0: t + 1] == y[i, t])  # earlier positions holding the token that comes next
            if hit.any():
                aa = a[j0:]
                copy_all.append((float(aa[hit].sum()), float(hit.double().mean())))
                far = hit.clone()
                far[-4:] = False  # only occurrences more than 3 tokens back
                if far.any():
                    copy_far.append((float(aa[far].sum()), float(far.sum()) / len(far)))
    q = lambda v: " ".join(f"{p:.3g}" for p in np.percentile(v, [10, 50, 90]))
    print(f"read as attention over tokens (10/50/90 pct): effective tokens {q(eff)};  top-1 weight {q(top)};  "
          f"mean distance {q(dist)};  on the current token {q(last1)};  on the last 4 {q(last4)}")
    out.update(eff=np.median(eff), top=np.median(top))
    sh = np.asarray(sharp, dtype=np.float64)
    for lo in (0.2, 0.3, 0.5):
        sel = sh[:, 0] >= lo
        print(f"reads with top weight >= {lo}: {sel.mean():.1%} of reads;  their top token is the next token "
              f"{sh[sel, 1].mean():.1%}, the current token {sh[sel, 2].mean():.1%}, median distance {np.median(sh[sel, 3]):.0f}")
    print(f"all reads: top token is the next token {sh[:, 1].mean():.1%}")
    for tag, c in (("any distance", copy_all), ("more than 3 back", copy_far)):
        c = np.asarray(c)
        print(f"copy precision, next token seen earlier ({tag}, {len(c)} positions): read weight on its earlier "
              f"occurrences mean {c[:, 0].mean():.3f} vs {c[:, 1].mean():.3f} under uniform attention "
              f"(lift {c[:, 0].mean() / c[:, 1].mean():.2f}x)")
    keep_g = m._gate
    for tag, val in (("memory off (g = 0)", 0.0), ("read gate forced open (g = 1)", 1.0)):
        m._gate = (lambda v: lambda n, q_: torch.full(q_.shape[:-1] + (1,), v, dtype=q_.dtype))(val)
        l2 = loss(m, x, y)
        print(f"loss with {tag}: {l2:.4f} ({l2 - out['loss']:+.4f})")
    m._gate = keep_g
    wshare, rshare = (wsum / wsum.sum()).numpy(), (zsum / zsum.sum()).numpy()
    hl = np.log(0.5) / np.log(mu.numpy())
    sink = rshare < 0.2 * wshare
    print(f"sink banks (read share < 0.2 x write share): {sink.sum()} of {S}, taking {wshare[sink].sum():.1%} of the writes, "
          f"giving {rshare[sink].sum():.1%} of the reads; their half-lives: "
          + ", ".join(f"{h:.0f}" for h in sorted(hl[sink]))[:200])
    print(f"bank half-lives now, 10/50/90/max: {q(hl)} {hl.max():.0f};  banks with half-life > 300: {(hl > 300).sum()}, "
          f"their read share {rshare[hl > 300].sum():.1%}")
    return out


def mlp_out(m, n, h):
    """The MLP output m = MLP(RMSNorm(h)) (torch, no fused kernel)."""
    w1, w2 = m._w("mlp_in", n), m._w("mlp_out", n)
    z = m._norm(m._w("mlp_norm", n), h)
    return torch.nn.functional.gelu(z @ w1.kernel.value + w1.bias.value, approximate="tanh") @ w2.kernel.value + w2.bias.value


def mix_gates(m, x, y, valid, name, out):
    """The keep / add gates of `read_mix="gates"` (h = a s + g o) and `mlp_residual="gates"` (y = a2 x + g2 m), layer 1:
    their values, which tokens move them, the size of what each side contributes, and the loss when a gate is
    replaced by its mean over these rows (what its selectivity is worth)."""
    B, N = x.shape
    with torch.no_grad():
        ids = torch.as_tensor(x, dtype=torch.int32)
        E = m.token_embedding.embeddings.value
        reset = ids == m.reset_token
        xe = torch.nn.functional.embedding(ids.long(), E)
        u = m._norm(m._w("in_norm", 0), xe) if m.pre_norm else xe
        zero = (E.new_zeros(B, m.dim), E.new_zeros(B, m.memory_rows, m.memory_cols))
        h, _, (s, go) = m._layer(0, u, reset, zero, collect=True, fresh=True)
        q = s @ m._w("address_q", 0).value
        gates, sizes = {}, {}
        if m.read_mix == "gates":
            gates["keep state a"] = m._keep_state(0, m.read_gate_in(s, q))[..., 0]
            gates["memory read g"] = m._gate(0, m.read_gate_in(s, q))[..., 0]
            o = go / gates["memory read g"][..., None]
            sizes.update({"|a s|": (gates["keep state a"][..., None] * s).norm(dim=-1), "|g o|": go.norm(dim=-1),
                     "|s|": s.norm(dim=-1), "|o| (ungated)": o.norm(dim=-1)})
        if m.mlp_residual == "gates":
            mo = mlp_out(m, 0, h)
            a2, g2 = m._out_gates(0, mo)
            gates["keep input a2"], gates["MLP g2"] = a2[..., 0], g2[..., 0]
            sizes.update({"|a2 x|": (a2 * xe).norm(dim=-1), "|g2 m|": (g2 * mo).norm(dim=-1), "|x|": xe.norm(dim=-1),
                          "|m| (ungated)": mo.norm(dim=-1)})
    pct = lambda v: " ".join(f"{p:.3g}" for p in np.percentile(v, [1, 10, 50, 90, 99]))
    print("mixing gates (layer 1), percentiles 1/10/50/90/99:")
    for k, g in gates.items():
        gv = g.numpy()[valid]
        stats = defaultdict(list)
        for t, val in zip(x[valid], gv):
            stats[int(t)].append(val)
        agg = sorted((np.mean(v), t) for t, v in stats.items() if len(v) >= 15)
        within = np.mean([np.var(v) for v in stats.values() if len(v) >= 15])
        print(f"   {k:14s} {pct(gv)}   std {gv.std():.3f}  ({within / max(gv.var(), 1e-12):.0%} of its variance within a token)")
        print(f"      lowest:  " + ", ".join(f"{name(t)} {v:.2f}" for v, t in agg[:8]))
        print(f"      highest: " + ", ".join(f"{name(t)} {v:.2f}" for v, t in agg[-8:][::-1]))
    print("   sizes (median): " + ",  ".join(f"{k} {float(np.median(v.numpy()[valid])):.3g}" for k, v in sizes.items()))
    # each gate replaced by its mean over these rows
    base = out["loss"]
    means = {k: float(g.numpy()[valid].mean()) for k, g in gates.items()}
    keep_ks, keep_g, keep_o = m._keep_state if m.read_mix == "gates" else None, m._gate, getattr(m, "_out_gates", None)
    const = lambda v: (lambda n, a: torch.full(a.shape[:-1] + (1,), v, dtype=a.dtype))
    for k in gates:
        if k == "keep state a":
            m._keep_state = const(means[k])
        elif k == "memory read g":
            m._gate = const(means[k])
        elif k == "keep input a2":
            m._out_gates = lambda n, mo, f=keep_o: (torch.full(mo.shape[:-1] + (1,), means["keep input a2"], dtype=mo.dtype), f(n, mo)[1])
        else:
            m._out_gates = lambda n, mo, f=keep_o: (f(n, mo)[0], torch.full(mo.shape[:-1] + (1,), means["MLP g2"], dtype=mo.dtype))
        l2 = loss(m, x, y)
        print(f"   loss with {k} fixed at its mean {means[k]:.3f}: {l2:.4f} ({l2 - base:+.4f})")
        if keep_ks is not None:
            m._keep_state = keep_ks
        m._gate = keep_g
        if keep_o is not None:
            m._out_gates = keep_o
    for tag, ka, kg in (("memory off (g = 0)", None, 0.0), ("state off (a = 0)", 0.0, None)):
        if m.read_mix != "gates" and ka is not None:
            continue
        if ka is not None:
            m._keep_state = const(ka)
        if kg is not None:
            m._gate = const(kg)
        l2 = loss(m, x, y)
        print(f"   loss with {tag}: {l2:.4f} ({l2 - base:+.4f})")
        if keep_ks is not None:
            m._keep_state = keep_ks
        m._gate = keep_g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run:checkpoint ...")
    p.add_argument("--rows", type=int, default=32)
    a = p.parse_args()
    first = a.runs[0].split(":")[0]
    config = cfg_mod.load(first)
    m0 = load_checkpoint(first, "latest")
    x, y = rows(config, a.rows, m0.window_size)
    tok = load_tokenizer(config)
    print(f"{a.rows} validation rows x {x.shape[1]} tokens")
    for spec in a.runs:
        run, _, ck = spec.partition(":")
        analyze(run, ck or "latest", x, y, tok)


if __name__ == "__main__":
    main()
