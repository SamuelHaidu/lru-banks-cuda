"""Ad hoc: same HDBSCAN clustering methodology as
`gpu_retriver_tokenid_small_table_hdbscan.py`, applied to
`simple-gpt-baseline`'s single dense token embedding table
(`token_embedding`, dim=128) instead of a thin dim_in=8 sphere.

Motivation: `gpu-retriver-tokenid`'s small table collapsed hundreds of
distinct character names into just two coarse "a name" clusters (sizes 224
and 86), with 78% of the vocabulary landing in no cluster at all -- the
direct structural cause of the observed entity-swap failure mode. This
script checks whether a well-trained, full-capacity (dim=128) dense
embedding table shows the same collision pattern, or genuinely separates
individual names/entities from each other.

Usage:
    .venv/bin/python ad_hoc_scripts/simple_gpt_token_embedding_hdbscan.py
"""

import json
from pathlib import Path

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
import model.archs  # noqa: E402,F401  (registers custom classes for keras.saving.load_model)
from model.evaluate import load_checkpoint  # noqa: E402
from model.reports._report import token_labels  # noqa: E402

RUN_NAME = "simple-gpt-baseline"
CHECKPOINT = "latest"
OUT_DIR = Path(__file__).resolve().parent / "results"
MIN_CLUSTER_SIZE = 5  # same threshold as the small-table analysis, for a fair comparison


def load_tokenizer(config):
    tokenizer_file = cfg_mod.tokenizer_path(config["data"]["dataset"])
    if not tokenizer_file.exists():
        return None
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_file))


def main() -> int:
    from sklearn.cluster import HDBSCAN

    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))
    model = load_checkpoint(RUN_NAME, CHECKPOINT)
    tokenizer = load_tokenizer(config)

    # `token_embedding` is a plain `keras.layers.Embedding` -- L2-normalize
    # for a cosine-metric clustering comparable to the small table's own
    # (already-normalized) representation.
    raw = ops.convert_to_numpy(model.token_embedding.embeddings)
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    points = (raw / np.clip(norms, 1e-12, None)).astype(np.float64)  # (vocab, dim)

    clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="cosine", n_jobs=-1)
    labels = clusterer.fit_predict(points)

    unique, counts = np.unique(labels, return_counts=True)
    n_clusters = int((unique >= 0).sum())
    n_noise = int(counts[unique == -1][0]) if -1 in unique else 0

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
    labels_path = OUT_DIR / "simple_gpt_token_embedding_hdbscan_labels.npy"
    summary_path = OUT_DIR / "simple_gpt_token_embedding_hdbscan_summary.json"

    np.save(labels_path, labels)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_name": RUN_NAME,
                "checkpoint": CHECKPOINT,
                "algorithm": "sklearn.cluster.HDBSCAN",
                "params": {"min_cluster_size": MIN_CLUSTER_SIZE, "metric": "cosine"},
                "vocabulary_size": int(points.shape[0]),
                "dim": int(points.shape[1]),
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "cluster_sizes": {int(c): int(n) for c, n in zip(unique, counts) if c != -1},
                "clusters": cluster_summaries,
            },
            f,
            indent=2,
        )

    print(f"vocabulary_size={points.shape[0]}  dim={points.shape[1]}")
    print(f"clusters found: {n_clusters}  noise points: {n_noise} ({n_noise / points.shape[0]:.1%})")
    print(f"labels saved to {labels_path}")
    print(f"summary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
