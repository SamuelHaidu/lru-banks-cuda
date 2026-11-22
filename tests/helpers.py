"""Shared test fixtures.

Imported first by every test module, because it pins the device to CPU before
anything touches keras -- the suite must pass on a machine with no XPU, and
tiny models are faster on CPU anyway.
"""

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

os.environ.setdefault("KERAS_DEVICE", "cpu")

import numpy as np

import model  # noqa: F401  (pins KERAS_BACKEND=torch before keras is imported)
from model import config as cfg_mod

VOCAB = 50
DOC_TOKEN_ID = 0


def make_corpus(
    doc_lengths: List[int],
    vocab: int = VOCAB,
    seed: int = 0,
    doc_token_id: int = DOC_TOKEN_ID,
    body_low: int = 1,
) -> np.ndarray:
    """Builds a flat token stream of documents, each starting with the marker.

    Body tokens are drawn from `[body_low, vocab)`, so the marker appears only
    at genuine document starts and the expected boundaries are unambiguous.

    `doc_token_id`/`body_low` exist because the real corpus uses `<bos>` (id 0)
    as its marker, which is *also* a legitimate context token -- meaning `x == 0`
    cannot distinguish left-padding from a real `<bos>`. Tests that need to
    identify padding positions unambiguously use a non-zero marker and reserve
    0 for padding alone.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for length in doc_lengths:
        body = rng.integers(body_low, vocab, size=max(length - 1, 0), dtype=np.int32)
        parts.append(np.concatenate([[doc_token_id], body]).astype(np.int32))
    return np.concatenate(parts).astype(np.int32)


@contextlib.contextmanager
def temp_workspace(
    train_doc_lengths: Optional[List[int]] = None,
    validation_doc_lengths: Optional[List[int]] = None,
    dataset_name: str = "tiny",
    avg_chars_per_token: float = 4.0,
    doc_token_id: int = DOC_TOKEN_ID,
    body_low: int = 1,
) -> Iterator[Path]:
    """Isolated `datasets/` + `runs/` rooted in a temp directory.

    Repoints `model.config`'s module-level path constants so tests never read
    or write the real corpora and run directories.
    """
    train_doc_lengths = train_doc_lengths or [40, 25, 60, 12]
    validation_doc_lengths = validation_doc_lengths or [30, 18]

    tmp = Path(tempfile.mkdtemp(prefix="rnn-test-"))
    datasets = tmp / "datasets"
    runs = tmp / "runs"
    tokenizers = tmp / "tokenizers"
    ds_dir = datasets / dataset_name
    ds_dir.mkdir(parents=True)
    runs.mkdir()
    tokenizers.mkdir()

    train = make_corpus(train_doc_lengths, seed=1, doc_token_id=doc_token_id, body_low=body_low)
    validation = make_corpus(validation_doc_lengths, seed=2, doc_token_id=doc_token_id, body_low=body_low)
    np.save(ds_dir / "train.npy", train)
    np.save(ds_dir / "validation.npy", validation)

    metadata: Dict[str, Any] = {
        "name": dataset_name,
        "tokenizer": dataset_name,
        "vocabulary_size": VOCAB,
        "doc_token_id": doc_token_id,
        "avg_chars_per_token": avg_chars_per_token,
        "train": {"file": "train.npy", "tokens": int(train.size), "documents": len(train_doc_lengths)},
        "validation": {
            "file": "validation.npy",
            "tokens": int(validation.size),
            "documents": len(validation_doc_lengths),
        },
    }
    (ds_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    originals = (cfg_mod.DATASETS_DIR, cfg_mod.RUNS_DIR, cfg_mod.TOKENIZERS_DIR)
    cfg_mod.DATASETS_DIR, cfg_mod.RUNS_DIR, cfg_mod.TOKENIZERS_DIR = datasets, runs, tokenizers
    try:
        yield tmp
    finally:
        cfg_mod.DATASETS_DIR, cfg_mod.RUNS_DIR, cfg_mod.TOKENIZERS_DIR = originals
        shutil.rmtree(tmp, ignore_errors=True)


def tiny_arch_params(arch: str = "dual-sphere") -> Dict[str, Any]:
    """Smallest sensible parameters for `arch`, for fast tests.

    Reads the architecture's own defaults and shrinks the size-related ones,
    so a newly registered architecture works here without editing this helper
    as long as it uses the conventional parameter names.
    """
    from model.archs import arch_defaults

    params = arch_defaults(arch)
    params.pop("embeddings_initializer", None)
    shrink = {
        "vocabulary_size": VOCAB,
        "window_size": 8,
        "hidden_dim": 16,
        "dim_out": 12,
        "dim_in": 4,
    }
    for key, value in shrink.items():
        if key in params:
            params[key] = value
    return params


def tiny_config(run_name: str = "t", arch: str = "dual-sphere", dataset: str = "tiny") -> Dict[str, Any]:
    """A complete, valid run config sized for tests."""
    cfg = json.loads(json.dumps(cfg_mod.DEFAULT_CONFIG))
    cfg["run_name"] = run_name
    cfg["arch"] = arch
    cfg["arch_params"] = tiny_arch_params(arch)
    cfg["data"].update({"dataset": dataset, "batch_size": 4, "steps_per_epoch": 3, "corrupt_prob": 0.0})
    cfg["training"].update({"epochs": 1, "validation_steps": 2})
    cfg["device"] = "cpu"
    return cfg
