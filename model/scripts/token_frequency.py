"""`model scripts token-frequency`: how often does a token occur in training data?

A small, reusable diagnostic: scans the training corpus once and counts
occurrences of a given list of tokens (by surface text, matched with a
leading space the way mid-sentence words tokenize). Used to test whether a
behavior correlates with how often a token was seen during training, as
opposed to any property of the embedding tables or in-context structure.
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from model.device import setup_backend

setup_backend()

from model import config as cfg_mod  # noqa: E402
from model.data import load_tokens  # noqa: E402
from model.scripts._model_io import load_tokenizer  # noqa: E402
from model.scripts._report import render_report  # noqa: E402


def count_token_frequencies(
    dataset: str,
    words: List[str],
    split: str = "train",
    chunk_size: int = 50_000_000,
) -> Dict[str, int]:
    """Counts occurrences of each surface `words` entry in `dataset`'s `split`.

    Each word is tokenized as `" " + word` (the mid-sentence form) to match
    how it actually appears after other text; words that tokenize to more
    than one id are counted by their last sub-token id (a reasonable proxy,
    since one id from an uncommon multi-token split is rarely a false-positive
    match for common names/words in this vocabulary).
    """
    config_meta = cfg_mod.load_dataset_metadata(dataset)
    tokenizer_path = cfg_mod.tokenizer_path(dataset)
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    word_ids: Dict[str, int] = {}
    for w in words:
        ids = tokenizer.encode(" " + w).ids
        if ids:
            word_ids[w] = ids[-1]

    id_to_word = {}
    for w, tid in word_ids.items():
        id_to_word.setdefault(tid, []).append(w)
    target_ids = np.array(sorted(id_to_word.keys()))

    path = str(cfg_mod.dataset_split_path(dataset, split))
    tokens = load_tokens(path)

    counts: Dict[str, int] = {w: 0 for w in word_ids}
    for begin in range(0, len(tokens), chunk_size):
        block = np.asarray(tokens[begin : begin + chunk_size])
        matched = block[np.isin(block, target_ids)]
        if matched.size == 0:
            continue
        vals, cnts = np.unique(matched, return_counts=True)
        for v, c in zip(vals, cnts):
            for w in id_to_word.get(int(v), []):
                counts[w] += int(c)
    return counts


def run(run_name: str, words: List[str], split: str = "train") -> Path:
    config = cfg_mod.load(run_name)
    dataset = config["data"]["dataset"]

    counts = count_token_frequencies(dataset, words, split=split)
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])

    context: Dict[str, Any] = {
        "run_name": run_name,
        "dataset": dataset,
        "split": split,
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "num_words": len(words),
        "ranked": ranked,
    }

    out_path = cfg_mod.run_dir(run_name) / "reports" / "token_frequency.html"
    render_report("token_frequency.html", context, out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model scripts token-frequency",
        description="Count occurrences of a list of words in a run's training corpus.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--words",
        required=True,
        help="Comma-separated list of surface words to count, e.g. 'Lily,Tom,Emma'.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to scan (default: train).")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    words = [w.strip() for w in args.words.split(",") if w.strip()]
    out_path = run(args.run_name, words, split=args.split)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
