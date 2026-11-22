"""Ad hoc: hierarchical clustering of `gpu-retriver-tokenid`'s small table.

Follow-up to `gpu_retriver_tokenid_small_table_hdbscan.py`. HDBSCAN's
noise/cluster split turned out to be highly sensitive to `min_cluster_size`
(539 clusters at min_cluster_size=3 down to 2 at min_cluster_size=25, noise
fraction 35-78% throughout) -- there is no single natural flat cut, because
10k points packed into an 8-D sphere (`dim_in=8`) have a smoothly decaying,
near-continuous density rather than a small number of well-separated blobs.
Agglomerative (hierarchical) clustering sidesteps committing to one flat
partition: it produces a full dendrogram, so any level of granularity (a
handful of coarse groups down to many fine ones, e.g. splitting the names
cluster into individual name-like sub-groups) can be inspected after the
fact without recomputing anything.

Average-linkage is used (not single/complete) because it's the more robust
choice against the near-uniform cosine-similarity background found on this
table (mean pairwise cosine ~0.49) -- single linkage chains through dense
backgrounds, complete linkage is overly sensitive to a handful of extreme
pairs.

Cost: cosine distance is computed as a full condensed pairwise matrix
(10000 choose 2 ~= 50M floats, ~400MB) -- `scipy.cluster.hierarchy.linkage`
on that is the expensive step but still well under 5 minutes at n=10k.

Outputs (cached so nothing here needs recomputing for follow-up questions):
  - `results/gpu_retriver_tokenid_small_table_linkage.npy`: the raw
    `(n-1, 4)` SciPy linkage matrix `Z` -- the single artifact that encodes
    the entire dendrogram; any flat cut can be derived from it directly via
    `scipy.cluster.hierarchy.fcluster(Z, ...)` with no re-clustering.
  - `results/gpu_retriver_tokenid_small_table_hierarchical_summary.json`:
    flat cuts at a handful of `k` (num clusters) levels, with per-cluster
    size and top-15 nearest-to-centroid tokens, for a first eyeball pass at
    each granularity.

Usage:
    .venv/bin/python ad_hoc_scripts/gpu_retriver_tokenid_small_table_hierarchical.py
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

RUN_NAME = "gpu-retriver-tokenid"
CHECKPOINT = "latest"
OUT_DIR = Path(__file__).resolve().parent / "results"

# Cluster-count levels to materialize as flat cuts for a first eyeball pass.
# Coarse -> fine, so both "few big groups" and "names split apart" are visible.
K_LEVELS = [4, 8, 16, 32, 64, 128]


def _cluster_summary(points: np.ndarray, labels: np.ndarray, tokenizer) -> list:
    summaries = []
    for c in sorted(set(labels.tolist())):
        members = np.where(labels == c)[0]
        centroid = points[members].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        sims = points[members] @ centroid
        order = np.argsort(-sims)[: min(15, len(members))]
        top_ids = members[order].tolist()
        summaries.append(
            {
                "cluster": int(c),
                "size": int(len(members)),
                "top_token_ids": top_ids,
                "top_tokens": token_labels(tokenizer, top_ids) if tokenizer else None,
            }
        )
    summaries.sort(key=lambda c: -c["size"])
    return summaries


def main() -> int:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    small_table, _big_table, config, ckpt_path = load_dual_embedding_tables(RUN_NAME, CHECKPOINT)
    tokenizer = load_tokenizer(config)

    points = ops.convert_to_numpy(small_table).astype(np.float64)  # (10000, dim_in), L2-normalized

    condensed = pdist(points, metric="cosine")  # (n*(n-1)/2,)
    Z = linkage(condensed, method="average")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    linkage_path = OUT_DIR / "gpu_retriver_tokenid_small_table_linkage.npy"
    summary_path = OUT_DIR / "gpu_retriver_tokenid_small_table_hierarchical_summary.json"

    np.save(linkage_path, Z)

    levels = {}
    for k in K_LEVELS:
        labels = fcluster(Z, t=k, criterion="maxclust")
        levels[str(k)] = {
            "n_clusters_actual": int(len(set(labels.tolist()))),
            "clusters": _cluster_summary(points, labels, tokenizer),
        }

    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_name": RUN_NAME,
                "checkpoint": CHECKPOINT,
                "checkpoint_path": str(ckpt_path),
                "algorithm": "scipy.cluster.hierarchy.linkage",
                "params": {"method": "average", "metric": "cosine"},
                "vocabulary_size": int(points.shape[0]),
                "dim_in": int(points.shape[1]),
                "linkage_path": str(linkage_path),
                "levels": levels,
            },
            f,
            indent=2,
        )

    print(f"vocabulary_size={points.shape[0]}  dim_in={points.shape[1]}")
    print(f"linkage matrix saved to {linkage_path}  shape={Z.shape}")
    print(f"flat-cut summary (k={K_LEVELS}) saved to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
