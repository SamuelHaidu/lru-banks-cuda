"""`model report pga-dual-embeddings`: does the big table reduce to the small one?

Runs Principal Geodesic Analysis (tangent-space PCA at the Karcher mean) on
the big table (`target_embeddings`, S^(dim_out-1)) with `--components`
components (default: `dim_in`, so it matches the small table's dimension by
construction). The resulting `num_components`-D PGA embedding is aligned to
the small table (`input_embeddings`, S^(dim_in-1)) with an orthogonal
Procrustes rotation -- the two live in different, arbitrarily-rotated bases,
so only a rotation-invariant comparison is meaningful -- and per-token cosine
similarity after alignment measures how much of the small table a low-rank
summary of the big table already reconstructs.

All PGA/Procrustes math is `keras.ops`; only report assembly (token strings,
Jinja2, JSON for Plotly) touches plain Python.
"""

ARCHS = ('dual-tables',)

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict

from model.device import setup_backend

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.reports import _spherical_ops as so  # noqa: E402
from model.reports._model_io import load_dual_embedding_tables, load_tokenizer  # noqa: E402
from model.reports._report import plotly_div, render_report, token_labels  # noqa: E402


def _variance_bar(variance_ratio: Any) -> Any:
    import plotly.graph_objects as go

    values = ops.convert_to_numpy(variance_ratio)
    cumulative = values.cumsum()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"g{i + 1}" for i in range(len(values))],
            y=values,
            marker_color="#2f5fa3",
            name="per-component",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[f"g{i + 1}" for i in range(len(values))],
            y=cumulative,
            mode="lines+markers",
            marker_color="#2f7d4f",
            name="cumulative",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Geodesic variance explained per principal component (big table)",
        margin=dict(l=50, r=50, t=40, b=40),
        height=380,
        plot_bgcolor="white",
        yaxis=dict(title="variance ratio"),
        yaxis2=dict(title="cumulative", overlaying="y", side="right", range=[0, 1.05]),
        legend=dict(orientation="h", y=1.15),
    )
    return fig


def _similarity_histogram(cos_sim: Any) -> Any:
    import plotly.graph_objects as go

    values = ops.convert_to_numpy(cos_sim)
    fig = go.Figure(
        data=go.Histogram(x=values, nbinsx=60, marker_color="#2f5fa3")
    )
    fig.update_layout(
        title="Per-token cosine similarity: aligned PGA(big) vs. small table",
        margin=dict(l=50, r=20, t=40, b=40),
        height=360,
        plot_bgcolor="white",
        xaxis_title="cosine similarity",
        yaxis_title="tokens",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f2", range=[-1, 1])
    return fig


def _scatter_2d(coords: Any, color: Any, title: str, colorbar_title: str) -> Any:
    import plotly.graph_objects as go

    coords = ops.convert_to_numpy(coords)
    color = ops.convert_to_numpy(color)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(size=4, color=color, colorscale="RdBu", cmid=0, opacity=0.7, colorbar=dict(title=colorbar_title)),
            hoverinfo="none",
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=40, b=40),
        height=420,
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f2", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef0f2", zeroline=False)
    return fig


def run(
    run_name: str,
    components: int = 8,
    checkpoint: str = "latest",
    top_n: int = 15,
) -> Path:
    small_table, big_table, config, ckpt_path = load_dual_embedding_tables(run_name, checkpoint)
    tokenizer = load_tokenizer(config)

    dim_in = small_table.shape[1]
    vocab_size = small_table.shape[0]

    mean, basis, variance_ratio, pga_coords = so.pga(big_table, num_components=components)
    pga_unit = so.normalize(pga_coords)

    # Rotation-invariant comparison only makes sense at matching dimension.
    aligned_available = components == dim_in
    if aligned_available:
        rotation = so.orthogonal_procrustes(pga_unit, small_table)
        aligned = so.normalize(ops.matmul(pga_unit, rotation))
        cos_sim = ops.sum(aligned * small_table, axis=-1)
        mean_cos_sim = float(ops.mean(cos_sim))
        median_cos_sim = float(ops.median(cos_sim))
        frac_above_0_5 = float(ops.mean(ops.cast(cos_sim > 0.5, "float32")))

        _, best_idx = so.top_k(cos_sim, k=min(top_n, vocab_size))
        _, worst_idx = so.top_k(-cos_sim, k=min(top_n, vocab_size))
        best_tokens = [
            {"id": int(i), "token": token_labels(tokenizer, [int(i)])[0] if tokenizer else str(int(i)), "cos_sim": float(cos_sim[i])}
            for i in best_idx
        ]
        worst_tokens = [
            {"id": int(i), "token": token_labels(tokenizer, [int(i)])[0] if tokenizer else str(int(i)), "cos_sim": float(cos_sim[i])}
            for i in worst_idx
        ]
        hist_fig = _similarity_histogram(cos_sim)
    else:
        mean_cos_sim = median_cos_sim = frac_above_0_5 = None
        best_tokens = worst_tokens = []
        hist_fig = None

    var_fig = _variance_bar(variance_ratio)
    scatter_pga_fig = _scatter_2d(
        pga_coords[:, :2],
        pga_coords[:, 0],
        "Big table -- leading 2 PGA axes",
        "PGA axis 1",
    )

    plots = {
        "variance": plotly_div(var_fig, "variance-plot", include_js=True),
        "scatter_pga": plotly_div(scatter_pga_fig, "scatter-pga-plot"),
    }
    if hist_fig is not None:
        plots["histogram"] = plotly_div(hist_fig, "histogram-plot")

    context: Dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "checkpoint_path": str(ckpt_path),
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "components": components,
        "dim_in": dim_in,
        "dim_out": big_table.shape[1],
        "vocab_size": vocab_size,
        "variance_ratio": [round(float(v), 4) for v in variance_ratio],
        "cumulative_variance": round(float(ops.sum(variance_ratio)), 4),
        "aligned_available": aligned_available,
        "mean_cos_sim": mean_cos_sim,
        "median_cos_sim": median_cos_sim,
        "frac_above_0_5": frac_above_0_5,
        "best_tokens": best_tokens,
        "worst_tokens": worst_tokens,
        "plots": plots,
    }

    out_path = cfg_mod.run_dir(run_name) / "reports" / "pga_dual_embeddings.html"
    render_report("pga_dual_embeddings.html", context, out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model report pga-dual-embeddings",
        description="Principal Geodesic Analysis of the big table vs. the small table.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--components",
        type=int,
        default=None,
        help="Number of principal geodesic components (default: the run's dim_in).",
    )
    parser.add_argument("--checkpoint", default="latest", help="latest | best | <epoch number>")
    parser.add_argument("--top-n", type=int, default=15, help="Best/worst-aligned tokens to list.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    components = args.components
    if components is None:
        config = cfg_mod.load(args.run_name)
        components = config["arch_params"].get("dim_in", 8)
    out_path = run(args.run_name, components=components, checkpoint=args.checkpoint, top_n=args.top_n)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
