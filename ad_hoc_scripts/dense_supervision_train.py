"""Ad hoc: train `dual-sphere-hyper-gate` with DENSE supervision -- one loss
per WINDOW POSITION (128 targets per document) instead of the one
target-per-example scheme `VariablePrefixDataset`/`model train` normally
uses -- via `model.fused_cosine_loss.fused_cosine_cross_entropy`, so the
`(batch*window, vocabulary_size)` logits tensor never has to be retained for
backward (see that module's docstring; `tests/test_fused_cosine_loss.py`
verifies the loss/gradients are correct).

This is NOT wired into `model train`/`model/train.py` -- it's a standalone,
one-off training loop, because dense supervision is fundamentally a
different (and ~128x heavier per step, at window_size=128) training regime
from every other run in this project, not a drop-in option for the normal
one-target-per-example CLI path.

Per training step, for `batch_size` documents:
  1. Pick a random document, take (or randomly cut) a `window_size`-token
     span from it.
  2. Build ALL `window_size` causal-prefix windows for that span (position i
     sees tokens [0..i], left-padded) -- `(batch_size, window_size,
     window_size)` input, `(batch_size, window_size)` targets (`targets[i] =
     span[i]`, i.e. this is exactly the FullEvalDataset trick of "one seeded
     position" generalized to "every position").
  3. Flatten to `(batch_size*window_size, window_size)`, run
     `model.call_query()` ONCE on the whole flattened batch (a single big
     matmul-batch, not `window_size` separate forward calls), and feed the
     resulting `(batch_size*window_size, dim_out)` queries plus the
     `(batch_size*window_size,)` targets into `fused_cosine_cross_entropy`.

Usage:
    .venv/bin/python ad_hoc_scripts/dense_supervision_train.py --run-name hyper-gate-baseline --steps 200
"""

import argparse
import time
from pathlib import Path

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

import torch  # noqa: E402
from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.archs import build_model  # noqa: E402
from model.data import _document_bounds, load_tokens  # noqa: E402
from model.fused_cosine_loss import fused_cosine_cross_entropy  # noqa: E402


def build_dense_batch(tokens: np.ndarray, doc_starts: np.ndarray, doc_ends: np.ndarray,
                       window_size: int, batch_size: int, rng: np.random.Generator):
    """`(batch_size, window_size, window_size)` input, `(batch_size,
    window_size)` targets -- position `i`'s row is the causal prefix ending
    at (and predicting) the `i`-th token of a randomly chosen `window_size`
    -token span."""
    doc_idx = rng.integers(0, len(doc_starts), size=batch_size)
    x = np.zeros((batch_size, window_size, window_size), dtype=np.int32)
    y = np.zeros((batch_size, window_size), dtype=np.int32)

    for b in range(batch_size):
        d = doc_idx[b]
        start, end = int(doc_starts[d]), int(doc_ends[d])
        doc_len = end - start
        span_len = min(window_size, doc_len - 1)  # need >=1 target after the span start
        # Random span start within the document, leaving room for span_len+1 tokens.
        span_start = start + int(rng.integers(0, doc_len - span_len))
        span = np.asarray(tokens[span_start : span_start + span_len + 1])  # +1 for the last target

        # Positions: predict span[1], span[2], ..., span[span_len] from
        # prefixes span[:1], span[:2], ..., span[:span_len] respectively --
        # i.e. `window_size` (context, target) pairs from one (window_size+1)
        # -token span, left-padded per position.
        for i in range(span_len):
            ctx = span[: i + 1]
            r = min(len(ctx), window_size)
            x[b, i, -r:] = ctx[-r:]
            y[b, i] = span[i + 1]
        # Any remaining rows (only possible if span_len < window_size, i.e. a
        # short document) stay all-zero context predicting token id 0 -- rare
        # at TinyStories' typical document lengths, acceptable padding.

    return x.reshape(batch_size * window_size, window_size), y.reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default="hyper-gate-baseline")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = cfg_mod.load(args.run_name)
    setup_device(config.get("device"))

    if config["arch"] != "dual-sphere-hyper-gate":
        raise ValueError(f"Run {args.run_name!r} is arch={config['arch']!r}, expected dual-sphere-hyper-gate "
                          f"(needs call_query()).")
    if config["arch_params"]["window_size"] != args.window_size:
        raise ValueError(f"Run's window_size={config['arch_params']['window_size']} != --window-size={args.window_size}.")

    model = build_model(config)
    # `model.trainable_weights` gives `keras.Variable` wrappers; the torch
    # optimizer needs the real `torch.nn.Parameter` each one wraps
    # (`.value`), not the wrapper itself.
    optimizer = torch.optim.AdamW([w.value for w in model.trainable_weights], lr=args.learning_rate)

    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
    train_path = str(cfg_mod.dataset_split_path(config["data"]["dataset"], "train"))
    tokens = load_tokens(train_path)
    doc_starts, doc_ends = _document_bounds(train_path, tokens, meta["doc_token_id"], use_cache=True)
    print(f"{len(doc_starts)} documents available in {config['data']['dataset']}")

    rng = np.random.default_rng(args.seed)
    print(f"Dense-supervision training: {args.run_name} (dual-sphere-hyper-gate), "
          f"batch_size={args.batch_size}, window_size={args.window_size}, "
          f"{args.batch_size}x{args.window_size}={args.batch_size * args.window_size} targets/step\n")

    step_times = []
    for step in range(1, args.steps + 1):
        t0 = time.time()
        x_np, y_np = build_dense_batch(tokens, doc_starts, doc_ends, args.window_size, args.batch_size, rng)

        optimizer.zero_grad()
        x = ops.convert_to_tensor(x_np, dtype="int32")
        y = ops.convert_to_tensor(y_np, dtype="int64")

        q = model.call_query(x, training=True)
        table = model.normalized_target_embeddings()
        loss = fused_cosine_cross_entropy(q, table, model.tau, y).mean()
        loss.backward()
        optimizer.step()

        step_time = time.time() - t0
        step_times.append(step_time)

        if step == 1 or step % 10 == 0 or step == args.steps:
            with torch.no_grad():
                acc = float((torch.argmax(q @ table.T, dim=-1) == y).float().mean())
            print(f"step {step:4d}/{args.steps}  loss={float(loss.detach()):.4f}  acc={acc:.4f}  "
                  f"step_time={step_time * 1000:.1f}ms")

    times = np.array(step_times)
    print(f"\nPerformance over {args.steps} steps:")
    print(f"  mean step time:   {times.mean() * 1000:.1f} ms")
    print(f"  median step time: {np.median(times) * 1000:.1f} ms")
    print(f"  min/max:          {times.min() * 1000:.1f} / {times.max() * 1000:.1f} ms")
    print(f"  effective examples/sec (targets, not documents): "
          f"{args.batch_size * args.window_size / times.mean():.0f}")
    print(f"  effective documents/sec: {args.batch_size / times.mean():.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
