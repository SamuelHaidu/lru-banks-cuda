"""`model scripts token-kernel-report`: what a `rope-token-kernel` run learned per token.

For every vocabulary token: prior `s`, forgetting `lambda`, saturation `kappa`, how far each
moved from its initial value, and its training-corpus frequency. Writes an interactive
report and prints a text summary:

- distributions of s / lambda / kappa (frequent tokens vs all) against their init;
- lambda vs s scatter (hover = token), colored by log frequency;
- movement from init vs frequency (are rare tokens just untouched?);
- extremes among frequent tokens (most kept, fastest forgotten, highest / lowest prior,
  most saturating) and group means (names, function words, punctuation, other);
- the learned shared kernel w(k), and the effective curves s * w(k^lambda) of example tokens.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np

from model.scripts._model_io import load_full_model, load_tokenizer

FUNCTION_WORDS = ["the", "a", "an", "to", "and", "was", "has", "had", "of", "in", "on", "at", "it", "is",
                  "that", "with", "for", "he", "she", "they", "his", "her", "but", "so", "then"]
EXAMPLES = [" Lily", " Tom", " Ben", " Mom", " dog", " ball", " the", " to", " has", " and", ".", ","]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model scripts token-kernel-report",
                                     description="Per-token prior / forgetting / saturation learned by rope-token-kernel.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--min-count", type=int, default=2000, help="'Frequent' = at least this many training occurrences.")
    return parser


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def main(argv: List[str]) -> None:
    args = build_parser().parse_args(argv)
    import plotly.graph_objects as go
    import torch
    from keras import ops

    from model import config as cfg_mod
    from model.data import load_tokens

    model, config, ckpt = load_full_model(args.run_name, args.checkpoint)
    if config["arch"] != "rope-token-kernel":
        raise SystemExit(f"Run {args.run_name!r} is {config['arch']!r}, not rope-token-kernel.")
    tok = load_tokenizer(config)
    V = model.vocabulary_size
    labels = [(tok.id_to_token(i) or f"<{i}>").replace("Ġ", " ").replace("Ċ", "\\n") for i in range(V)]
    to_np = lambda w: np.asarray(ops.convert_to_numpy(w), dtype=np.float64)

    s = _softplus(to_np(model.token_prior)) if model.prior else np.ones(V)
    lam = 1 / (1 + np.exp(-to_np(model.token_lambda))) if model.forgetting else np.ones(V)
    kap = _softplus(to_np(model.token_kappa)) if model.saturation else np.full(V, np.inf)
    s0, lam0, kap0 = 1.0, 0.95, 20.0

    print("counting training-token frequencies ...")
    counts = np.bincount(np.asarray(load_tokens(str(cfg_mod.dataset_split_path(config["data"]["dataset"], "train")))), minlength=V)[:V]
    freq_mask = counts >= args.min_count
    logc = np.log10(counts + 1)

    T = model.window_size
    k = np.arange(1, T + 1, dtype=np.float64)
    with torch.no_grad():
        w_shared = to_np(model.shared_kernel(ops.convert_to_tensor(k.astype("float32"))))
        def eff(i):
            return s[i] * to_np(model.shared_kernel(ops.convert_to_tensor((k ** lam[i]).astype("float32"))))

    def group(i):
        t = labels[i].strip()
        if not t:
            return "other"
        if t.lower() in FUNCTION_WORDS:
            return "function word"
        if all(not c.isalnum() for c in t):
            return "punctuation"
        if labels[i].startswith(" ") and t[0].isupper() and t[1:].islower() and len(t) > 2:
            return "capitalized (names...)"
        return "other"

    groups = np.array([group(i) for i in range(V)])
    figs = []

    # 1. distributions
    for name, vals, init in (("prior s", s, s0), ("forgetting lambda", lam, lam0), ("saturation kappa", np.minimum(kap, 100), kap0)):
        f = go.Figure()
        f.add_trace(go.Histogram(x=vals, name="all tokens", opacity=0.5, nbinsx=80))
        f.add_trace(go.Histogram(x=vals[freq_mask], name=f"frequent (>= {args.min_count})", opacity=0.7, nbinsx=80))
        f.add_vline(x=init, line_dash="dash", annotation_text="init")
        f.update_layout(title=f"Learned {name}", barmode="overlay", height=330)
        figs.append(f)

    # 2. scatter lambda vs s
    sel = np.where(counts > 0)[0]
    sc = go.Figure(go.Scattergl(x=lam[sel], y=s[sel], mode="markers", text=[f"{labels[i]!r} n={counts[i]}" for i in sel],
                                marker=dict(size=5, color=logc[sel], colorscale="Viridis", showscale=True, colorbar=dict(title="log10 count")),
                                hovertemplate="%{text}<br>lambda=%{x:.3f}<br>s=%{y:.3f}<extra></extra>"))
    sc.update_layout(title="Forgetting lambda vs prior s per token (0 = kept forever, 1 = forgets like the shared kernel)",
                     xaxis_title="lambda", yaxis_title="s", height=550)
    figs.append(sc)

    # 3. movement vs frequency
    move = np.abs(lam - lam0) + np.abs(np.log(s / s0)) + (np.abs(np.log(kap / kap0)) if model.saturation else 0)
    mv = go.Figure(go.Scattergl(x=counts[sel] + 1, y=move[sel], mode="markers", text=[repr(labels[i]) for i in sel],
                                marker=dict(size=4), hovertemplate="%{text}<br>count %{x}<br>moved %{y:.3f}<extra></extra>"))
    mv.update_layout(title="How far each token's numbers moved from init vs its training frequency",
                     xaxis=dict(title="training count", type="log"), yaxis_title="|d lambda| + |d log s| + |d log kappa|", height=420)
    figs.append(mv)

    # 4. kernel and example curves
    kc = go.Figure()
    kc.add_trace(go.Scatter(x=k, y=w_shared, name="shared kernel w(k)", line=dict(color="black", width=3, dash="dash")))
    for t in EXAMPLES:
        ids = [i for i in range(V) if labels[i] == t]
        if ids:
            i = ids[0]
            kc.add_trace(go.Scatter(x=k, y=eff(i), name=f"{t!r} s={s[i]:.2f} lambda={lam[i]:.2f}"))
    kc.update_layout(title="Effective curve s * w(k^lambda) of example tokens", xaxis=dict(title="distance k", type="log"),
                     yaxis_title="weight (before normalization)", height=480)
    figs.append(kc)

    # text summary
    fi = np.where(freq_mask)[0]
    lines = [f"run {args.run_name} ({ckpt.name}): {V} tokens, {len(fi)} with >= {args.min_count} training occurrences",
             f"shared kernel w(k): w(1)={w_shared[0]:.3f} w(8)={w_shared[7]:.3f} w(32)={w_shared[31]:.3f} w(128)={w_shared[-1]:.3f}"
             + (f"   output gain {float(np.exp(to_np(model.log_gain))):.2f}" if model.normalize else "")]
    for name, vals, init in (("s", s, s0), ("lambda", lam, lam0), ("kappa", kap, kap0)):
        v = vals[fi]
        lines.append(f"{name:7s} frequent tokens: mean {v.mean():.3f}  std {v.std():.3f}  min {v.min():.3f}  max {v.max():.3f}"
                     f"  (init {init}); moved > 5%: {np.mean(np.abs(v - init) > 0.05 * abs(init)) * 100:.0f}%")
    lines.append("group means over frequent tokens:")
    for g in ("capitalized (names...)", "function word", "punctuation", "other"):
        idx = fi[groups[fi] == g]
        if len(idx):
            lines.append(f"  {g:24s} n={len(idx):5d}  s={s[idx].mean():.3f}  lambda={lam[idx].mean():.3f}  kappa={np.median(kap[idx]):.2f} (median)")

    def top(vals, n=12, reverse=False):
        order = fi[np.argsort(vals[fi])]
        order = order[::-1] if reverse else order
        return ", ".join(f"{labels[i]!r} {vals[i]:.2f}" for i in order[:n])

    lines += [f"most kept (lowest lambda): {top(lam)}",
              f"fastest forgotten (highest lambda): {top(lam, reverse=True)}",
              f"highest prior s: {top(s, reverse=True)}",
              f"lowest prior s: {top(s)}"]
    if model.saturation:
        lines.append(f"most saturating (lowest kappa): {top(kap)}")
    lines.append("examples: " + "; ".join(f"{t!r} n={counts[i]} s={s[i]:.2f} lambda={lam[i]:.2f} kappa={kap[i]:.1f}"
                                          for t in EXAMPLES for i in [j for j in range(V) if labels[j] == t][:1]))
    summary = "\n".join(lines)
    print(summary)

    out = ckpt.parent.parent / "reports" / "token_kernel_report.html" if ckpt.parent.name == "checkpoints" else ckpt.parent / "reports" / "token_kernel_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, f in enumerate(figs))
    import html as _html
    out.write_text(f"<html><head><meta charset='utf-8'><title>token kernel report</title></head><body style='font-family:sans-serif;max-width:1200px;margin:20px'>"
                   f"<h2>{args.run_name}: learned per-token numbers</h2><pre style='background:#f4f4f6;padding:10px;white-space:pre-wrap'>{_html.escape(summary)}</pre>{body}</body></html>")
    print(f"Wrote {out}")
