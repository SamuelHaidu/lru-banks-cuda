"""`model report spherical-kmeans`: do the small and big tables agree?

Runs spherical k-means independently on the small table (`input_embeddings`,
S^(dim_in-1)) and the big table (`target_embeddings`, S^(dim_out-1)) of a
dual-table run, then asks whether the two clusterings of the *same* 10k
tokens agree: same contingency matrix, Adjusted Rand Index, and Normalized
Mutual Information. High agreement means the small table's coarse geometry
already reflects the fine-grained semantic neighborhoods of the big table.

All clustering and metric math is `keras.ops`; only report assembly (token
strings, Jinja2, JSON for Plotly) touches plain Python.
"""

ARCHS = ('dual-tables',)

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

from model.device import setup_backend

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.reports import _spherical_ops as so  # noqa: E402
from model.reports._model_io import load_dual_embedding_tables, load_tokenizer  # noqa: E402
from model.reports._report import plotly_div, render_report, token_labels  # noqa: E402


def _top_tokens_per_cluster(
    points: Any, centroids: Any, assignments: Any, top_n: int
) -> List[List[int]]:
    """`top_n` token ids closest (highest cosine similarity) to each centroid."""
    sims = ops.matmul(points, ops.transpose(centroids))  # (N, k)
    k = centroids.shape[0]
    result = []
    for c in range(k):
        mask = ops.equal(assignments, c)
        cluster_sims = ops.where(mask, sims[:, c], ops.full_like(sims[:, c], -2.0))
        n_take = min(top_n, int(ops.sum(ops.cast(mask, "int32"))))
        if n_take == 0:
            result.append([])
            continue
        _, top_idx = so.top_k(cluster_sims, k=n_take)
        result.append([int(i) for i in top_idx])
    return result


def _cluster_sizes(assignments: Any, k: int) -> List[int]:
    one_hot = ops.equal(ops.reshape(assignments, (-1, 1)), ops.arange(k, dtype="int32"))
    return [int(x) for x in ops.sum(ops.cast(one_hot, "int32"), axis=0)]


def _scatter_2d(points: Any, assignments: Any, title: str) -> str:
    """2D visualization via geodesic-distance MDS, colored by cluster.

    Classical MDS on great-circle distances is used instead of a tangent-plane
    PGA projection because it directly minimizes distortion of the full
    pairwise distance matrix rather than only local variance near one base
    point -- a more faithful 2D read of the table's cluster geometry.
    """
    import plotly.graph_objects as go

    coords = so.geodesic_mds(points, num_components=2)
    coords = ops.convert_to_numpy(coords)
    labels = ops.convert_to_numpy(assignments)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                size=4,
                color=labels,
                colorscale="Viridis",
                opacity=0.65,
                line=dict(width=0),
            ),
            text=[f"cluster {int(c)}" for c in labels],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=40, b=40),
        height=420,
        plot_bgcolor="white",
        xaxis_title="MDS axis 1 (geodesic-distance preserving)",
        yaxis_title="MDS axis 2 (geodesic-distance preserving)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f2", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef0f2", zeroline=False)
    return fig


def _agreement_heatmap(contingency: Any, k: int) -> Any:
    import plotly.graph_objects as go

    matrix = ops.convert_to_numpy(contingency)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            colorscale="Blues",
            colorbar=dict(title="tokens"),
        )
    )
    fig.update_layout(
        title="Small-table cluster vs. big-table cluster (token co-occurrence)",
        xaxis_title="big-table cluster",
        yaxis_title="small-table cluster",
        margin=dict(l=60, r=20, t=40, b=50),
        height=440,
        plot_bgcolor="white",
    )
    return fig


def run(
    run_name: str,
    k: int = 32,
    checkpoint: str = "latest",
    iterations: int = 100,
    seed: int = 0,
    top_n: int = 12,
) -> Path:
    small_table, big_table, config, ckpt_path = load_dual_embedding_tables(run_name, checkpoint)
    tokenizer = load_tokenizer(config)

    small_centroids, small_assign, small_iters = so.spherical_kmeans(
        small_table, k=k, max_iterations=iterations, seed=seed
    )
    big_centroids, big_assign, big_iters = so.spherical_kmeans(
        big_table, k=k, max_iterations=iterations, seed=seed
    )

    contingency = so.contingency_matrix(small_assign, big_assign, k, k)
    ari = float(so.adjusted_rand_index(contingency))
    nmi = float(so.normalized_mutual_info(contingency))

    small_top = _top_tokens_per_cluster(small_table, small_centroids, small_assign, top_n)
    big_top = _top_tokens_per_cluster(big_table, big_centroids, big_assign, top_n)

    small_sizes = _cluster_sizes(small_assign, k)
    big_sizes = _cluster_sizes(big_assign, k)

    clusters = []
    for c in range(k):
        clusters.append(
            {
                "id": c,
                "small_size": small_sizes[c],
                "big_size": big_sizes[c],
                "small_tokens": token_labels(tokenizer, small_top[c]) if tokenizer else [str(i) for i in small_top[c]],
                "big_tokens": token_labels(tokenizer, big_top[c]) if tokenizer else [str(i) for i in big_top[c]],
            }
        )

    fig_small = _scatter_2d(small_table, small_assign, "Small table (dim_in) -- geodesic MDS projection")
    fig_big = _scatter_2d(big_table, big_assign, "Big table (dim_out) -- geodesic MDS projection")
    fig_heat = _agreement_heatmap(contingency, k)

    plots = {
        "small_scatter": plotly_div(fig_small, "small-scatter", include_js=True),
        "big_scatter": plotly_div(fig_big, "big-scatter"),
        "heatmap": plotly_div(fig_heat, "heatmap"),
    }

    context: Dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "checkpoint_path": str(ckpt_path),
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "k": k,
        "vocab_size": small_table.shape[0],
        "dim_in": small_table.shape[1],
        "dim_out": big_table.shape[1],
        "small_iters": int(small_iters),
        "big_iters": int(big_iters),
        "ari": ari,
        "nmi": nmi,
        "clusters": clusters,
        "plots": plots,
    }

    out_path = cfg_mod.run_dir(run_name) / "reports" / "spherical_kmeans.html"
    render_report("spherical_kmeans.html", context, out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model report spherical-kmeans",
        description="Cluster the small and big embedding tables independently and measure agreement.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("-k", "--clusters", type=int, default=32, help="Number of clusters (default: 32).")
    parser.add_argument("--checkpoint", default="latest", help="latest | best | <epoch number>")
    parser.add_argument("--iterations", type=int, default=100, help="Max k-means iterations.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=12, help="Example tokens shown per cluster.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_path = run(
        args.run_name,
        k=args.clusters,
        checkpoint=args.checkpoint,
        iterations=args.iterations,
        seed=args.seed,
        top_n=args.top_n,
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
