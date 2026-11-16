"""Evaluation on the full, deterministic validation set.

Every derived metric is computed from one measured loss plus dataset
metadata. In particular `avg_chars_per_token` comes from
`datasets/<name>/metadata.json`, never from a run config -- when it lived
per-run, two runs on the same corpus silently used different values (3.6 vs
4.127) and their BPC numbers were not comparable.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

from model.device import setup_backend, setup_device

setup_backend()

import keras  # noqa: E402

from model import config as cfg_mod  # noqa: E402

# Imported for its side effect: importing the package runs every architecture's
# `@keras.saving.register_keras_serializable` decorator. Without it,
# `load_model` on a saved run fails with "Could not locate class ...".
import model.archs  # noqa: E402,F401
from model.data import build_full_eval_dataset  # noqa: E402
from model.losses import SparseCategoricalCrossentropyFromLogits  # noqa: E402


def resolve_checkpoint(run_name: str, checkpoint: Union[str, int] = "latest") -> Path:
    """Resolves `latest` | `best` | an epoch number to a `.keras` path."""
    run_path = cfg_mod.run_dir(run_name)
    ckpt_dir = run_path / "checkpoints"

    if checkpoint == "latest":
        path = run_path / "model.keras"
    elif checkpoint == "best":
        path = ckpt_dir / "best.keras"
    else:
        try:
            epoch = int(checkpoint)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid checkpoint {checkpoint!r}. Use 'latest', 'best', or an epoch number.")
        path = ckpt_dir / f"checkpoint_{epoch:05d}.keras"

    if not path.exists():
        available = sorted(p.name for p in ckpt_dir.glob("*.keras")) if ckpt_dir.exists() else []
        raise FileNotFoundError(f"No checkpoint at {path}. Available: {available}")
    return path


def load_checkpoint(run_name: str, checkpoint: Union[str, int] = "latest") -> keras.Model:
    return keras.saving.load_model(resolve_checkpoint(run_name, checkpoint))


def evaluate(
    run_name: str,
    checkpoint: Union[str, int] = "latest",
    batch_size: Optional[int] = None,
    verbose: int = 1,
    context_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Scores a checkpoint over the entire validation split.

    Returns loss, perplexity, bits-per-character and top-1 accuracy, plus the
    parameter breakdown, and writes the same dict to
    `runs/<run_name>/eval_<checkpoint>.json`.
    """
    config = cfg_mod.load(run_name)
    setup_device(config.get("device"))
    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])

    model = load_checkpoint(run_name, checkpoint)
    trained_context = model.context_length
    if context_length and int(context_length) != trained_context:
        model = model.with_context_length(int(context_length))  # uncompiled copy, compiled below
    if not model.compiled:
        model.compile(
            loss=SparseCategoricalCrossentropyFromLogits(),
            metrics=["sparse_categorical_accuracy"],
            # `.get` with a default: run configs written before this key
            # existed don't have it, and should evaluate exactly as before.
            jit_compile=config["training"].get("jit_compile", False),
        )

    eval_ds = build_full_eval_dataset(config, model.context_length, batch_size=batch_size)
    scores = model.evaluate(eval_ds, verbose=verbose, return_dict=True)

    loss = float(scores["loss"])
    chars_per_token = float(meta["avg_chars_per_token"])
    results = {
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "checkpoint_path": str(resolve_checkpoint(run_name, checkpoint)),
        "dataset": config["data"]["dataset"],
        "samples": int(eval_ds.num_docs),
        "context_length": int(model.context_length),
        "trained_context_length": int(trained_context),
        "loss": round(loss, 4),
        "perplexity": round(math.exp(loss), 4),
        # nats -> bits, then amortized over the characters a token represents.
        "bpc": round(loss / math.log(2) / chars_per_token, 4),
        "top1_accuracy": round(float(scores.get("sparse_categorical_accuracy", float("nan"))), 4),
        "avg_chars_per_token": chars_per_token,
        "parameters": model.parameter_breakdown(),
    }

    suffix = f"_ctx{model.context_length}" if model.context_length != trained_context else ""
    out = cfg_mod.run_dir(run_name) / f"eval_{checkpoint}{suffix}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    return results


def format_results(results: Dict[str, Any]) -> str:
    p = results["parameters"]
    return (
        f"\n{results['run_name']} @ {results['checkpoint']}  "
        f"({results['dataset']}, {results['samples']:,} samples)\n"
        f"  loss           {results['loss']:.4f}\n"
        f"  perplexity     {results['perplexity']:.2f}\n"
        f"  bpc            {results['bpc']:.4f}\n"
        f"  top-1 accuracy {results['top1_accuracy'] * 100:.2f}%\n"
        f"  parameters     {p['total']:,} "
        f"(embeddings {p['embeddings']:,} / mechanism {p['mechanism']:,})"
    )
