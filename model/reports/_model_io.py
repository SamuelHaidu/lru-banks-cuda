"""Loads a run's checkpoint and its two embedding tables for analysis.

Both analysis scripts need the same three things: the trained
`AlibiRetrieverSphereLM` (or any architecture exposing `input_embeddings` /
`target_embeddings`), the run's tokenizer, and its config. Centralized here so
neither script duplicates the loading/validation logic.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

from model.device import setup_backend, setup_device

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
import model.archs  # noqa: E402,F401  (registers custom classes for keras.saving.load_model)
from model.evaluate import load_checkpoint, resolve_checkpoint  # noqa: E402


class UnsupportedArchitectureError(ValueError):
    """Raised when a run's architecture has no dual embedding tables."""


def load_dual_embedding_tables(
    run_name: str,
    checkpoint: str = "latest",
) -> Tuple[Any, Any, Dict[str, Any], Path]:
    """Returns `(small_table, big_table, config, checkpoint_path)`.

    `small_table` is `input_embeddings` (shape `(V, dim_in)`), `big_table` is
    `target_embeddings` (shape `(V, dim_out)`) -- both L2-normalized onto
    their respective spheres. Requires the architecture to expose both
    (currently `alibi-retriever-sphere`).
    """
    config = cfg_mod.load(run_name)
    setup_device(config.get("device"))

    model = load_checkpoint(run_name, checkpoint)
    if not (hasattr(model, "input_embeddings") and hasattr(model, "target_embeddings")):
        raise UnsupportedArchitectureError(
            f"Architecture {config['arch']!r} has no `input_embeddings`/`target_embeddings` "
            "pair -- this analysis only applies to dual-table architectures "
            "(e.g. 'alibi-retriever-sphere')."
        )

    import torch

    with torch.no_grad():
        small_table = ops.normalize(ops.convert_to_tensor(model.input_embeddings), axis=-1, order=2)
        big_table = ops.normalize(ops.convert_to_tensor(model.target_embeddings), axis=-1, order=2)
    return small_table, big_table, config, resolve_checkpoint(run_name, checkpoint)


def load_full_model(run_name: str, checkpoint: str = "latest") -> Tuple[Any, Dict[str, Any], Path]:
    """Returns `(model, config, checkpoint_path)` -- the whole model, not just its tables.

    For analyses that need to re-run part of the forward pass (e.g. reading
    intermediate attention weights) rather than only the embedding tables.
    """
    config = cfg_mod.load(run_name)
    setup_device(config.get("device"))
    model = load_checkpoint(run_name, checkpoint)
    return model, config, resolve_checkpoint(run_name, checkpoint)


def load_tokenizer(config: Dict[str, Any]):
    tokenizer_file = cfg_mod.tokenizer_path(config["data"]["dataset"])
    if not tokenizer_file.exists():
        return None
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_file))
