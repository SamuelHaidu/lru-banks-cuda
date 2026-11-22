"""`model scripts conv-kernels`: the distance kernels a `rope-logit-sum` run learned.

Reads `distance_weights` (shared `(T,)` or per-dimension `(T, dim)`) and
writes one interactive HTML report:

- heatmap: weight per channel and distance (raw, or each channel scaled to
  its max |w|), channels in embedding order with their RoPE pair and period;
- every channel's curve vs distance (log/linear axes, signed or |w|);
- per channel: the distance holding 50% / 90% of its total |w| ("reach")
  against its RoPE rotation period, and the weight at distance 1.

Channel `d` and `d + dim/2` form one RoPE pair (same rotation frequency).
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np

from model.scripts._model_io import load_full_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model scripts conv-kernels",
        description="Plot the per-channel distance kernels of a rope-logit-sum run.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="latest", help="latest | best | <epoch>")
    parser.add_argument("--out", default=None, help="Output HTML (default: runs/<run>/reports/conv_kernels.html).")
    return parser


def _reach(w_abs: np.ndarray, frac: float) -> np.ndarray:
    """Per channel: smallest distance whose cumulative |w| (from distance 1) reaches `frac` of the total."""
    c = np.cumsum(w_abs, axis=0) / np.maximum(w_abs.sum(axis=0, keepdims=True), 1e-12)
    return (c < frac).sum(axis=0) + 1


def main(argv: List[str]) -> None:
    args = build_parser().parse_args(argv)
    import plotly.graph_objects as go
    from keras import ops

    model, config, ckpt = load_full_model(args.run_name, args.checkpoint)
    if not hasattr(model, "distance_kernel"):
        raise SystemExit(f"Run {args.run_name!r} ({config['arch']}) has no distance kernel.")
    import torch
    with torch.no_grad():
        w = np.asarray(ops.convert_to_numpy(model.distance_kernel()), dtype=np.float64)
    if w.ndim == 1:
        w = w[:, None]
    w = w[::-1]  # context-aligned (last = distance 1) -> row i = distance i + 1
    T, C = w.shape
    dist = np.arange(1, T + 1)
    dim = int(getattr(model, "dim", C))
    half = dim // 2
    base = float(getattr(model, "rope_base", 10000.0))
    inv_freq = 1.0 / (base ** (np.arange(half) / half))
    period = 2 * np.pi / inv_freq  # tokens per full rotation
    pair = np.arange(C) % half if C == dim else np.zeros(C, int)
    labels = [f"d{c} (pair {pair[c]}, period {period[pair[c]]:.0f})" if C == dim else "shared" for c in range(C)]

    figs = []

    # 1. Heatmap, raw and per-channel normalized.
    scaled = w / np.maximum(np.abs(w).max(axis=0, keepdims=True), 1e-12)
    vmax = float(np.quantile(np.abs(w), 0.995))
    heat = go.Figure()
    heat.add_trace(go.Heatmap(z=w.T, x=dist, y=labels, colorscale="RdBu", zmid=0, zmin=-vmax, zmax=vmax,
                              colorbar=dict(title="w"), name="raw"))
    heat.add_trace(go.Heatmap(z=scaled.T, x=dist, y=labels, colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                              colorbar=dict(title="w / max|w|"), name="scaled", visible=False))
    heat.update_layout(
        title="Weight per channel and distance", height=max(500, 6 * C + 150),
        xaxis=dict(title="distance k", type="log"), yaxis=dict(title="channel", autorange="reversed"),
        updatemenus=[dict(type="buttons", direction="right", x=0, y=1.08, xanchor="left", buttons=[
            dict(label="raw", method="update", args=[{"visible": [True, False]}]),
            dict(label="per-channel scaled", method="update", args=[{"visible": [False, True]}]),
            dict(label="x log", method="relayout", args=[{"xaxis.type": "log"}]),
            dict(label="x linear", method="relayout", args=[{"xaxis.type": "linear"}]),
        ])],
    )
    figs.append(heat)

    # 2. Curves: one per channel, colored by RoPE pair (fast -> slow).
    import plotly.colors as pc
    colors = pc.sample_colorscale("Viridis", [p / max(half - 1, 1) for p in pair]) if C == dim else ["#1f77b4"]
    curves = go.Figure()
    for c in range(C):
        curves.add_trace(go.Scatter(x=dist, y=w[:, c], mode="lines", name=labels[c],
                                    line=dict(color=colors[c], width=1, dash="solid" if c < half or C != dim else "dot"),
                                    customdata=np.abs(w[:, c]), hovertemplate="k=%{x}<br>w=%{y:.4g}<extra>" + labels[c] + "</extra>"))
    curves.add_trace(go.Scatter(x=dist, y=1.0 / dist, mode="lines", name="1/k (init)",
                                line=dict(color="red", width=2, dash="dash")))
    n = C + 1
    curves.update_layout(
        title="Kernel curves (color: RoPE pair, fast = dark; dotted = second half of the pair)",
        height=650, showlegend=False,
        xaxis=dict(title="distance k", type="log"), yaxis=dict(title="w"),
        updatemenus=[dict(type="buttons", direction="right", x=0, y=1.1, xanchor="left", buttons=[
            dict(label="signed", method="update",
                 args=[{"y": [w[:, c] for c in range(C)] + [1.0 / dist]}, {"yaxis.type": "linear", "yaxis.title.text": "w"}]),
            dict(label="|w| log-log", method="update",
                 args=[{"y": [np.abs(w[:, c]) for c in range(C)] + [1.0 / dist]}, {"yaxis.type": "log", "yaxis.title.text": "|w|"}]),
            dict(label="x log", method="relayout", args=[{"xaxis.type": "log"}]),
            dict(label="x linear", method="relayout", args=[{"xaxis.type": "linear"}]),
        ])],
    )
    figs.append(curves)

    # 3. Per-channel reach vs RoPE period.
    if C == dim:
        a = np.abs(w)
        r50, r90 = _reach(a, 0.5), _reach(a, 0.9)
        reach = go.Figure()
        for name, r, sym in (("50% of |w| within", r50, "circle"), ("90% of |w| within", r90, "diamond")):
            reach.add_trace(go.Scatter(x=period[pair], y=r, mode="markers", name=name, marker=dict(symbol=sym, color=colors, size=8),
                                       text=labels, hovertemplate="%{text}<br>reach %{y}<extra>" + name + "</extra>"))
        reach.update_layout(title="Reach per channel vs its RoPE rotation period", height=500,
                            xaxis=dict(title="RoPE period (tokens per rotation)", type="log"),
                            yaxis=dict(title="distance", type="log"))
        figs.append(reach)
        w1 = go.Figure(go.Bar(x=labels, y=w[0], marker_color=colors))
        w1.update_layout(title="Weight at distance 1 per channel", height=400, xaxis=dict(showticklabels=False), yaxis=dict(title="w(k=1)"))
        figs.append(w1)

    out = Path(args.out) if args.out else ckpt.parent.parent / "reports" / "conv_kernels.html" if ckpt.parent.name == "checkpoints" \
        else ckpt.parent / "reports" / "conv_kernels.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    head = (f"<h2>{args.run_name}: learned distance kernels</h2><p>checkpoint <code>{ckpt.name}</code>, "
            f"{C} channel(s) x {T} distances. Channels d and d+{half} are one RoPE pair.</p>")
    body = "".join(f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, f in enumerate(figs))
    out.write_text(f"<html><head><meta charset='utf-8'><title>conv kernels</title></head><body style='font-family:sans-serif'>{head}{body}</body></html>")
    print(f"Wrote {out}")
