"""`model scripts retrieval-ablation-loss-tokenid`: is token-anchored retrieval load-bearing?

Same causal-ablation methodology as `retrieval-ablation-loss`
(`model/scripts/retrieval_ablation_loss.py`), applied to
`alibi-token-retriever-sphere` instead of `alibi-retriever-sphere`. That
script's trace helpers replicate the *free-query* retrieval mechanism
(query vs. the whole vocab table) and do not match this architecture's
actual forward pass (query vs. in-context tokens, scatter by token id) --
running it against a token-retriever checkpoint would silently score the
wrong computation. This script uses `_alibi_token_retriever_trace.py`
instead, which replicates the real mechanism, and scores four conditions
over the full validation set:

  1. normal   -- the real forward pass, unmodified.
  2. uniform  -- the retrieved vector is replaced by the exact vector a
                 perfectly uniform vocabulary-weight vector would produce
                 (removes whatever query-dependence remains, keeps an
                 in-distribution vector).
  3. shuffled -- each example gets another example's real retrieved vector
                 (on-distribution -- a genuine retrieved output -- but
                 mismatched content, isolating "wrong content" from
                 "off-distribution").
  4. random   -- the retrieved vector is replaced by genuine random noise
                 (destroys the retrieved content entirely, off-distribution).

Every other trained weight is left untouched in every condition; only the
`(batch, num_query_tokens, dim_in)` block spliced into Block 2 changes. This
isolates exactly what token-anchored retrieval contributes to the model's
actual loss, as opposed to a hand-picked set of tokens.
"""

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

from model.device import setup_backend

setup_backend()

from keras import ops, random as keras_random  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.data import build_full_eval_dataset  # noqa: E402
from model.scripts._alibi_token_retriever_trace import (  # noqa: E402
    finish_forward,
    random_retrieved_tokens,
    retrieve_from_vocab_weights,
    trace_block1,
    vocab_mean_retrieved_tokens,
)
from model.scripts._model_io import load_full_model  # noqa: E402
from model.scripts._report import render_report  # noqa: E402


def _cross_entropy(logits: Any, targets: Any) -> Any:
    """Per-example cross-entropy from logits, XPU-safe (see `model.losses`)."""
    labels = ops.cast(ops.reshape(targets, (-1, 1)), "int32")
    log_probs = ops.log_softmax(logits, axis=-1)
    picked = ops.take_along_axis(log_probs, labels, axis=-1)
    return -ops.squeeze(picked, axis=-1)


def _evaluate_condition(model: Any, x_batches, y_batches, condition: str, seed_gen) -> Dict[str, float]:
    import torch

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for x, y in zip(x_batches, y_batches):
            token_ids = ops.convert_to_tensor(x, dtype="int32")
            targets = ops.convert_to_tensor(y, dtype="int32")
            batch_size = ops.shape(token_ids)[0]

            trace = trace_block1(model, token_ids)
            block1_x = trace["x"]

            if condition == "normal":
                retrieved = retrieve_from_vocab_weights(model, trace["vocab_weights"])
            elif condition == "uniform":
                retrieved = vocab_mean_retrieved_tokens(model, batch_size=int(batch_size))
            elif condition == "random":
                retrieved = random_retrieved_tokens(model, batch_size=int(batch_size), seed=seed_gen)
            elif condition == "shuffled":
                # Each example gets another example's real retrieved vector:
                # on-distribution (it is a genuine retrieved output), but
                # mismatched content -- isolates "wrong content" from
                # "off-distribution", unlike the `random` (Gaussian) condition.
                real_retrieved = retrieve_from_vocab_weights(model, trace["vocab_weights"])
                perm = ops.convert_to_tensor(
                    np.roll(np.arange(int(batch_size)), 1), dtype="int32"
                )
                retrieved = ops.take(real_retrieved, perm, axis=0)
            else:
                raise ValueError(f"Unknown condition {condition!r}")

            logits = finish_forward(model, block1_x, retrieved)
            loss = _cross_entropy(logits, targets)
            predicted = ops.argmax(logits, axis=-1)

            total_loss += float(ops.sum(loss))
            total_correct += int(ops.sum(ops.cast(ops.equal(ops.cast(predicted, "int32"), targets), "int32")))
            total_count += int(batch_size)

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
    if not hasattr(model, "query_proj"):
        raise ValueError(
            f"Architecture {config['arch']!r} is not supported by this ablation "
            "-- expected 'alibi-token-retriever-sphere' (or any arch exposing `query_proj`)."
        )

    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
    chars_per_token = float(meta["avg_chars_per_token"])

    eval_ds = build_full_eval_dataset(config, model.context_length, batch_size=batch_size)
    x_batches = []
    y_batches = []
    n = min(num_batches, len(eval_ds))
    for i in range(n):
        x, y = eval_ds[i]
        x_batches.append(x)
        y_batches.append(y)

    seed_gen = keras_random.SeedGenerator(seed)

    results = {}
    for condition in ("normal", "uniform", "shuffled", "random"):
        scores = _evaluate_condition(model, x_batches, y_batches, condition, seed_gen)
        scores["bpc"] = scores["loss"] / math.log(2) / chars_per_token
        results[condition] = scores

    baseline = results["normal"]["loss"]
    for condition, scores in results.items():
        scores["loss_delta"] = scores["loss"] - baseline
        scores["loss_delta_pct"] = (scores["loss"] - baseline) / baseline * 100.0

    context: Dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "checkpoint_path": str(ckpt_path),
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "num_examples": results["normal"]["num_examples"],
        "batch_size": batch_size,
        "results": results,
    }

    out_path = cfg_mod.run_dir(run_name) / "reports" / "retrieval_ablation_loss_tokenid.html"
    render_report("retrieval_ablation_loss_tokenid.html", context, out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model scripts retrieval-ablation-loss-tokenid",
        description="Measure the validation-loss cost of corrupting token-anchored retrieval.",
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
