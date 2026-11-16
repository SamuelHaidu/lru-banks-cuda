"""`model scripts order-sensitivity-loss`: does word order matter, or is the final FFN a bag-of-tokens lookup?

The retrieval ablation showed the softmax retrieval step contributes almost
nothing query-specific. That leaves two remaining candidates for how the
model actually recalls a name from context: the attention blocks doing
genuine positional routing (finding the name at its earlier position and
copying it forward), or the final readout FFN (`dense1/2/3`, a fixed
function of one flattened vector `u`) acting as a key-value memory over
aggregate context content, indifferent to *where* in the window that content
sits. Since nothing else in this architecture mixes information across
sequence positions except `attn1`/`attn2` (ALiBi's positional bias only
depends on the absolute index gap, not token identity), this is a clean,
weight-free test:

  1. shuffled -- randomly permute the order of the *valid* (non-padded)
     context tokens, re-score the exact same tokens the model would
     otherwise see, unmodified weights. If loss/top-1 survive, the specific
     sequence order (and hence what attention could exploit about it) does
     not matter much -- consistent with a bag-of-embeddings key-value
     readout. If loss/top-1 collapse, sequence order matters, which only
     attention's routing can be responsible for.
  2. name-masked -- the target's occurrence(s) earlier in context are
     replaced by a fixed placeholder token id, at the *same position*,
     leaving everything else (including order) untouched. This isolates
     whether recall depends on the *specific identity* of the earlier
     occurrence rather than merely something occupying that slot.

Both conditions are compared against the unmodified baseline over the same
real validation examples used by `retrieval-ablation-loss`.
"""

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

from model.device import setup_backend

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.data import build_full_eval_dataset  # noqa: E402
from model.scripts._model_io import load_full_model  # noqa: E402
from model.scripts._report import render_report  # noqa: E402

PLACEHOLDER_TOKEN_TEXT = "the"  # a common, semantically neutral filler


def _cross_entropy(logits: Any, targets: Any) -> Any:
    labels = ops.cast(ops.reshape(targets, (-1, 1)), "int32")
    log_probs = ops.log_softmax(logits, axis=-1)
    picked = ops.take_along_axis(log_probs, labels, axis=-1)
    return -ops.squeeze(picked, axis=-1)


def _shuffle_valid_suffix(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permutes each row's non-zero (non-padding) tokens among themselves.

    Padding always occupies a contiguous prefix (see `model.data._fill_windows`),
    so the valid tokens are exactly the trailing non-zero run; shuffling only
    within that run preserves window length and avoids moving real content
    into the always-zero padding positions.
    """
    out = x.copy()
    for i in range(x.shape[0]):
        nonzero = np.flatnonzero(x[i])
        if nonzero.size <= 1:
            continue
        start = nonzero[0]
        segment = out[i, start:]
        perm = rng.permutation(segment.shape[0])
        out[i, start:] = segment[perm]
    return out


def _mask_target_occurrences(x: np.ndarray, y: np.ndarray, placeholder_id: int) -> np.ndarray:
    """Replaces every context occurrence of the target token with `placeholder_id`.

    Position is untouched -- only identity changes -- so any change in
    prediction is attributable to the earlier occurrence's specific content,
    not to where it sits in the window.
    """
    out = x.copy()
    for i in range(x.shape[0]):
        out[i][out[i] == y[i]] = placeholder_id
    return out


def _evaluate(model: Any, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    import torch

    with torch.no_grad():
        token_ids = ops.convert_to_tensor(x, dtype="int32")
        targets = ops.convert_to_tensor(y, dtype="int32")
        logits = model(token_ids, training=False)
        loss = _cross_entropy(logits, targets)
        predicted = ops.argmax(logits, axis=-1)
        correct = ops.sum(ops.cast(ops.equal(ops.cast(predicted, "int32"), targets), "int32"))

    return {
        "loss": float(ops.sum(loss)),
        "correct": int(correct),
        "count": int(x.shape[0]),
    }


def _accumulate(model: Any, x_batches, y_batches) -> Dict[str, float]:
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for x, y in zip(x_batches, y_batches):
        scores = _evaluate(model, x, y)
        total_loss += scores["loss"]
        total_correct += scores["correct"]
        total_count += scores["count"]
    mean_loss = total_loss / total_count
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "top1_accuracy": total_correct / total_count,
        "num_examples": total_count,
    }


def run(
    run_name: str,
    checkpoint: str = "latest",
    num_batches: int = 20,
    batch_size: int = 256,
    seed: int = 0,
) -> Path:
    model, config, ckpt_path = load_full_model(run_name, checkpoint)

    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
    chars_per_token = float(meta["avg_chars_per_token"])

    eval_ds = build_full_eval_dataset(config, model.context_length, batch_size=batch_size)
    n = min(num_batches, len(eval_ds))
    x_batches, y_batches = [], []
    for i in range(n):
        x, y = eval_ds[i]
        x_batches.append(x)
        y_batches.append(y)

    rng = np.random.default_rng(seed)

    from tokenizers import Tokenizer

    tokenizer_file = cfg_mod.tokenizer_path(config["data"]["dataset"])
    placeholder_id = 0
    if tokenizer_file.exists():
        tok = Tokenizer.from_file(str(tokenizer_file))
        ids = tok.encode(" " + PLACEHOLDER_TOKEN_TEXT).ids
        placeholder_id = ids[-1] if ids else 0

    shuffled_x = [_shuffle_valid_suffix(x, rng) for x in x_batches]
    masked_x = [_mask_target_occurrences(x, y, placeholder_id) for x, y in zip(x_batches, y_batches)]

    results = {}
    results["normal"] = _accumulate(model, x_batches, y_batches)
    results["shuffled_order"] = _accumulate(model, shuffled_x, y_batches)
    results["masked_target"] = _accumulate(model, masked_x, y_batches)

    for scores in results.values():
        scores["bpc"] = scores["loss"] / math.log(2) / chars_per_token

    baseline = results["normal"]["loss"]
    for scores in results.values():
        scores["loss_delta"] = scores["loss"] - baseline
        scores["loss_delta_pct"] = (scores["loss"] - baseline) / baseline * 100.0

    context: Dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "checkpoint_path": str(ckpt_path),
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "num_examples": results["normal"]["num_examples"],
        "batch_size": batch_size,
        "placeholder_token": PLACEHOLDER_TOKEN_TEXT,
        "results": results,
    }

    out_path = cfg_mod.run_dir(run_name) / "reports" / "order_sensitivity_loss.html"
    render_report("order_sensitivity_loss.html", context, out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model scripts order-sensitivity-loss",
        description="Test whether token order / earlier-occurrence identity matters, vs. a bag-of-tokens FFN readout.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="latest", help="latest | best | <epoch number>")
    parser.add_argument("--num-batches", type=int, default=20, help="Validation batches to score per condition.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_path = run(
        args.run_name,
        checkpoint=args.checkpoint,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
