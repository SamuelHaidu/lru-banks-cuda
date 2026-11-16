"""`model scripts table-neighbor-agreement`: does the small table map to the big one?

The PGA and k-means reports each answer a narrower question -- "does a single
global rotation of an 8-D summary of the big table match the small table"
and "do independent clusterings of each table agree" -- both of which can
under-state agreement if the true relationship isn't a rigid rotation or a
clean partition. This script asks the more direct, rotation- and
partition-invariant question: for a given token, are the tokens *near it* in
the small table also the tokens near it in the big table?

Two measurements, both over the same 10k tokens in both tables:
  1. k-nearest-neighbor overlap: for each token, the fraction of its
     small-table neighbors that are also big-table neighbors.
  2. Pairwise-similarity correlation: Pearson correlation between small-table
     cosine similarity and big-table cosine similarity, over a random sample
     of token pairs (a distance-preservation / Mantel-style correlation).

All neighbor/overlap/correlation math is `keras.ops`; only report assembly
(token strings, Jinja2, JSON for Plotly) touches plain Python.
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict

from model.device import setup_backend

setup_backend()

from keras import ops, random  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.scripts import _spherical_ops as so  # noqa: E402
from model.scripts._model_io import load_dual_embedding_tables, load_tokenizer  # noqa: E402
from model.scripts._report import plotly_div, render_report, token_labels  # noqa: E402


def _overlap_histogram(overlap: Any, num_neighbors: int) -> Any:
    import plotly.graph_objects as go

    values = ops.convert_to_numpy(overlap)
    fig = go.Figure(data=go.Histogram(x=values, nbinsx=num_neighbors + 1, marker_color="#2f5fa3"))
    fig.update_layout(
        title=f"Per-token {num_neighbors}-NN overlap: small table vs. big table",
        margin=dict(l=50, r=20, t=40, b=40),
        height=360,
        plot_bgcolor="white",
        xaxis_title="fraction of neighbors shared",
        yaxis_title="tokens",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f2", range=[0, 1])
    return fig


def _similarity_scatter(sim_small: Any, sim_big: Any, correlation: float) -> Any:
    import plotly.graph_objects as go

    x = ops.convert_to_numpy(sim_small)
    y = ops.convert_to_numpy(sim_big)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=3, color="#2f5fa3", opacity=0.35, line=dict(width=0)),
            hoverinfo="none",
        )
    )
    fig.update_layout(
        title=f"Pairwise cosine similarity: small table vs. big table (r = {correlation:.3f})",
        margin=dict(l=50, r=20, t=40, b=40),
        height=420,
        plot_bgcolor="white",
        xaxis_title="cosine similarity -- small table",
        yaxis_title="cosine similarity -- big table",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f2", range=[-1, 1], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef0f2", range=[-1, 1], zeroline=False)
    return fig


def run(
    run_name: str,
    num_neighbors: int = 10,
    checkpoint: str = "latest",
    sample_pairs: int = 200_000,
    top_n: int = 15,
    seed: int = 0,
) -> Path:
    small_table, big_table, config, ckpt_path = load_dual_embedding_tables(run_name, checkpoint)
    tokenizer = load_tokenizer(config)
    vocab_size = small_table.shape[0]

    small_neighbors = so.nearest_neighbors(small_table, num_neighbors)
    big_neighbors = so.nearest_neighbors(big_table, num_neighbors)
    overlap = so.neighbor_overlap(small_neighbors, big_neighbors, vocab_size)

    mean_overlap = float(ops.mean(overlap))
    median_overlap = float(ops.median(overlap))
    chance_overlap = num_neighbors / vocab_size

    seed_gen = random.SeedGenerator(seed)
    sample_indices = random.randint((sample_pairs, 2), minval=0, maxval=vocab_size, seed=seed_gen)
    correlation, sim_small, sim_big = so.pairwise_similarity_correlation(small_table, big_table, sample_indices)
    correlation = float(correlation)

    _, best_idx = so.top_k(overlap, k=min(top_n, vocab_size))
    _, worst_idx = so.top_k(-overlap, k=min(top_n, vocab_size))

    def _describe(idx):
        out = []
        for i in idx:
            i = int(i)
            # Discriminability within this token's own small-table neighborhood
            # (the token plus its k nearest neighbors): how well-separated is
            # that little group in each table? Reuses the neighbor ids already
            # computed above -- no extra sampling or lookups.
            group_ids = ops.concatenate([ops.reshape(ops.convert_to_tensor(i, dtype="int32"), (1,)), small_neighbors[i]])
            small_sim, small_pr = so.group_discriminability(small_table, group_ids)
            big_sim, big_pr = so.group_discriminability(big_table, group_ids)
            out.append(
                {
                    "id": i,
                    "token": token_labels(tokenizer, [i])[0] if tokenizer else str(i),
                    "overlap": float(overlap[i]),
                    "small_neighbors": token_labels(tokenizer, [int(n) for n in small_neighbors[i]])
                    if tokenizer
                    else [str(int(n)) for n in small_neighbors[i]],
                    "big_neighbors": token_labels(tokenizer, [int(n) for n in big_neighbors[i]])
                    if tokenizer
                    else [str(int(n)) for n in big_neighbors[i]],
                    "small_group_sim": float(small_sim),
                    "big_group_sim": float(big_sim),
                    "small_group_rank": float(small_pr),
                    "big_group_rank": float(big_pr),
                }
            )
        return out

    best_tokens = _describe(best_idx)
    worst_tokens = _describe(worst_idx)

    hist_fig = _overlap_histogram(overlap, num_neighbors)
    scatter_fig = _similarity_scatter(sim_small, sim_big, correlation)

    plots = {
        "histogram": plotly_div(hist_fig, "overlap-histogram", include_js=True),
        "scatter": plotly_div(scatter_fig, "similarity-scatter"),
    }

    context: Dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "checkpoint_path": str(ckpt_path),
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "num_neighbors": num_neighbors,
        "vocab_size": vocab_size,
        "dim_in": small_table.shape[1],
        "dim_out": big_table.shape[1],
        "mean_overlap": mean_overlap,
        "median_overlap": median_overlap,
        "chance_overlap": chance_overlap,
        "correlation": correlation,
        "sample_pairs": sample_pairs,
        "best_tokens": best_tokens,
        "worst_tokens": worst_tokens,
        "plots": plots,
    }

    out_path = cfg_mod.run_dir(run_name) / "reports" / "table_neighbor_agreement.html"
    render_report("table_neighbor_agreement.html", context, out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model scripts table-neighbor-agreement",
        description="Does the small table's neighborhood structure match the big table's?",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("-n", "--num-neighbors", type=int, default=10, help="k in k-NN overlap (default: 10).")
    parser.add_argument("--checkpoint", default="latest", help="latest | best | <epoch number>")
    parser.add_argument("--sample-pairs", type=int, default=200_000, help="Random token pairs for the correlation.")
    parser.add_argument("--top-n", type=int, default=15, help="Best/worst-overlap tokens to list.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_path = run(
        args.run_name,
        num_neighbors=args.num_neighbors,
        checkpoint=args.checkpoint,
        sample_pairs=args.sample_pairs,
        top_n=args.top_n,
        seed=args.seed,
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
