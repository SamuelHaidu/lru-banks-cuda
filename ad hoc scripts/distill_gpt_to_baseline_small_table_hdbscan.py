"""Ad hoc: non-parametric clustering of `gpu-retriver-tokenid`'s small table.

Motivation: an observed failure mode where the model correctly tracks *that*
an entity slot should hold a name, and copies a name there, but swaps *which*
name goes in *which* slot (e.g. a Timmy/Billy story where the names get
exchanged between sentences while everything else stays coherent). Before
digging into slot-binding, the first question is purely geometric: how many
distinct clusters does the small `input_embeddings` table (10k tokens,
dim_in=8, L2-normalized) actually form, with no `k` chosen in advance --
does something like "a names cluster" fall out on its own?

HDBSCAN (`sklearn.cluster.HDBSCAN`, already available, no new dependency) is
used because it is non-parametric (no `k` to guess), runs on 10k x 8-D points
in well under a minute, and -- unlike k-means -- explicitly labels tokens
that don't belong to any coherent cluster as noise (-1) rather than forcing
every token into one.

Results (labels, cluster sizes, per-cluster nearest-centroid token ids) are
cached to disk so this doesn't need to be recomputed on every follow-up
question about the clustering.

Usage:
    .venv/bin/python "ad hoc scripts/gpu_retriver_tokenid_small_table_hdbscan.py"
"""

import json
from pathlib import Path

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.scripts._model_io import load_dual_embedding_tables, load_tokenizer  # noqa: E402
from model.scripts._report import token_labels  # noqa: E402

RUN_NAME = "distill-gpt-to-baseline"
CHECKPOINT = "latest"
OUT_DIR = Path(__file__).resolve().parent / "results"
MIN_CLUSTER_SIZE = 5  # smallest group HDBSCAN will call a cluster rather than noise


def main() -> int:
    from sklearn.cluster import HDBSCAN

    small_table, _big_table, config, ckpt_path = load_dual_embedding_tables(RUN_NAME, CHECKPOINT)
    tokenizer = load_tokenizer(config)

    points = ops.convert_to_numpy(small_table).astype(np.float64)  # (10000, dim_in)

    clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="cosine", n_jobs=-1)
    labels = clusterer.fit_predict(points)

    unique, counts = np.unique(labels, return_counts=True)
    n_clusters = int((unique >= 0).sum())
    n_noise = int(counts[unique == -1][0]) if -1 in unique else 0

    # Nearest-to-centroid tokens per cluster, for a quick eyeball of what each
    # cluster represents (reuses the report module's token-label cleanup).
    cluster_summaries = []
    for c in sorted(unique[unique >= 0]):
        members = np.where(labels == c)[0]
        centroid = points[members].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        sims = points[members] @ centroid
        order = np.argsort(-sims)[: min(15, len(members))]
        top_ids = members[order].tolist()
        cluster_summaries.append(
            {
                "cluster": int(c),
                "size": int(len(members)),
                "top_token_ids": top_ids,
                "top_tokens": token_labels(tokenizer, top_ids) if tokenizer else None,
            }
        )
    cluster_summaries.sort(key=lambda c: -c["size"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = OUT_DIR / "distill_gpt_to_baseline_small_table_hdbscan_labels.npy"
    summary_path = OUT_DIR / "distill_gpt_to_baseline_small_table_hdbscan_summary.json"

    np.save(labels_path, labels)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_name": RUN_NAME,
                "checkpoint": CHECKPOINT,
                "checkpoint_path": str(ckpt_path),
                "algorithm": "sklearn.cluster.HDBSCAN",
                "params": {"min_cluster_size": MIN_CLUSTER_SIZE, "metric": "cosine"},
                "vocabulary_size": int(points.shape[0]),
                "dim_in": int(points.shape[1]),
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "cluster_sizes": {int(c): int(n) for c, n in zip(unique, counts) if c != -1},
                "clusters": cluster_summaries,
            },
            f,
            indent=2,
        )

    print(f"vocabulary_size={points.shape[0]}  dim_in={points.shape[1]}")
    print(f"clusters found: {n_clusters}  noise points: {n_noise} ({n_noise / points.shape[0]:.1%})")
    print(f"labels saved to {labels_path}")
    print(f"summary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
