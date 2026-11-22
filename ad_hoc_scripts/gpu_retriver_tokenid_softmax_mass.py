"""Ad hoc: how concentrated is `alibi-token-retriever-sphere`'s token-anchored
softmax over real validation data for the `gpu-retriver-tokenid` run?

`gpu-retriver-tokenid` (arch `alibi-token-retriever-sphere`, batch_size 73728,
lr 0.01, GPU) scatters its per-position context softmax weights into a
vocabulary-sized bucket by token id before pooling the big embedding table
(see `model/scripts/_alibi_token_retriever_trace.py::trace_block1`). This
script asks: for that scattered per-example distribution, how many distinct
token ids does it take to cover 80% of the softmax mass?

Sampled from real validation batches (not the hand-picked name/noun prompts
used by `model scripts token-retrieval-trace`), so the number reflects
ordinary behavior rather than curated recall cases.

Usage:
    .venv/bin/python ad_hoc_scripts/gpu_retriver_tokenid_softmax_mass.py \
        --run-name gpu-retriver-tokenid --num-batches 4
"""

import argparse

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.data import build_full_eval_dataset  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402
from model.scripts._alibi_token_retriever_trace import trace_block1  # noqa: E402


def tokens_for_mass(sorted_weights: np.ndarray, target_mass: float) -> int:
    """Fewest top-ranked weights whose cumulative sum reaches `target_mass`."""
    cumsum = np.cumsum(sorted_weights)
    idx = np.searchsorted(cumsum, target_mass)
    return int(idx) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="gpu-retriver-tokenid")
    parser.add_argument("--checkpoint", default="latest", help="latest | best | <epoch number>")
    parser.add_argument("--num-batches", type=int, default=4, help="Validation batches to sample.")
    parser.add_argument("--mass", type=float, default=0.80, help="Target cumulative softmax mass.")
    args = parser.parse_args()

    config = cfg_mod.load(args.run_name)
    setup_device(config.get("device"))

    model = load_checkpoint(args.run_name, args.checkpoint)
    if not hasattr(model, "query_proj"):
        raise ValueError(
            f"Architecture {config['arch']!r} has no `query_proj`; this analysis only applies "
            "to alibi-token-retriever-sphere."
        )

    eval_ds = build_full_eval_dataset(config, model.window_size, batch_size=256)

    import torch

    counts = []
    n_examples = 0
    num_batches = min(args.num_batches, len(eval_ds))
    with torch.no_grad():
        for step in range(num_batches):
            x_batch, _y_batch = eval_ds[step]
            token_ids = ops.convert_to_tensor(x_batch, dtype="int32")

            trace = trace_block1(model, token_ids)
            vocab_weights = ops.convert_to_numpy(trace["vocab_weights"])  # (batch, vocab_size)

            for row in vocab_weights:
                sorted_desc = np.sort(row)[::-1]
                counts.append(tokens_for_mass(sorted_desc, args.mass))
            n_examples += vocab_weights.shape[0]

    counts = np.array(counts)
    print(f"run: {args.run_name}  checkpoint: {args.checkpoint}")
    print(f"vocabulary_size: {model.vocabulary_size}  window_size: {model.window_size}")
    print(f"examples analyzed: {n_examples}")
    print(f"target cumulative softmax mass: {args.mass:.0%}")
    print()
    print(f"distinct token ids needed for {args.mass:.0%} mass:")
    print(f"  mean:   {counts.mean():.2f}")
    print(f"  median: {np.median(counts):.1f}")
    print(f"  min:    {counts.min()}")
    print(f"  max:    {counts.max()}")
    print(f"  std:    {counts.std():.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
