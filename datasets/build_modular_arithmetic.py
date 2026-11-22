"""Builds the `modular-arithmetic-97` synthetic dataset.

One doc per example: `<bos> a0 a1 op b0 b1 = r0 r1 <eos>`, where `a`, `b` are
drawn uniformly from `[0, MODULUS)`, `op` is one of `+ - *`, and
`r = (a op b) % MODULUS`. Operands and the result are always written as two
digits (zero-padded), since `MODULUS - 1` is two digits -- fixed-width fields
mean a "compute the result" objective is always the same number of tokens
past `=`.

Digit-level vocabulary (16 tokens):

    0-9    digits
    10     <bos>  (also the doc-boundary marker `doc_token_id`)
    11     <eos>
    12     +
    13     -
    14     *
    15     =

Usage:
    .venv/bin/python datasets/build_modular_arithmetic.py
"""

import json
from pathlib import Path

import numpy as np

MODULUS = 97
TRAIN_EXAMPLES = 100_000
VALIDATION_EXAMPLES = 5_000
SEED = 0

BOS, EOS, PLUS, MINUS, STAR, EQUALS = 10, 11, 12, 13, 14, 15
VOCAB_SIZE = 16
OPS = ((PLUS, lambda a, b: a + b), (MINUS, lambda a, b: a - b), (STAR, lambda a, b: a * b))

OUT_DIR = Path(__file__).parent / "modular-arithmetic-97"


def digits(n: int) -> tuple:
    return (n // 10, n % 10)


def build_split(rng: np.random.Generator, num_examples: int) -> np.ndarray:
    # Doc layout: <bos> a0 a1 op b0 b1 = r0 r1 <eos>  -> 10 tokens/doc.
    doc_len = 10
    tokens = np.empty(num_examples * doc_len, dtype=np.int32)

    a = rng.integers(0, MODULUS, size=num_examples)
    b = rng.integers(0, MODULUS, size=num_examples)
    op_idx = rng.integers(0, len(OPS), size=num_examples)

    pos = 0
    for i in range(num_examples):
        op_token, op_fn = OPS[op_idx[i]]
        r = op_fn(int(a[i]), int(b[i])) % MODULUS
        a0, a1 = digits(int(a[i]))
        b0, b1 = digits(int(b[i]))
        r0, r1 = digits(r)
        tokens[pos : pos + doc_len] = (BOS, a0, a1, op_token, b0, b1, EQUALS, r0, r1, EOS)
        pos += doc_len
    return tokens


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    train = build_split(rng, TRAIN_EXAMPLES)
    validation = build_split(rng, VALIDATION_EXAMPLES)

    np.save(OUT_DIR / "train.npy", train)
    np.save(OUT_DIR / "validation.npy", validation)

    metadata = {
        "name": "modular-arithmetic-97",
        "source": f"synthetic: a op b mod {MODULUS}, op in {{+, -, *}}, digit-level tokens",
        "tokenizer": "modular-arithmetic-97",
        "vocabulary_size": VOCAB_SIZE,
        "doc_token_id": BOS,
        "bos_token_id": BOS,
        "eos_token_id": EOS,
        "modulus": MODULUS,
        # Every token (digit or symbol) renders as exactly one character.
        "avg_chars_per_token": 1.0,
        "dtype": "int32",
        "train": {
            "file": "train.npy",
            "tokens": int(train.size),
            "documents": TRAIN_EXAMPLES,
        },
        "validation": {
            "file": "validation.npy",
            "tokens": int(validation.size),
            "documents": VALIDATION_EXAMPLES,
        },
        "vocab": {
            "0-9": "digits",
            "10": "<bos>",
            "11": "<eos>",
            "12": "+",
            "13": "-",
            "14": "*",
            "15": "=",
        },
        "notes": (
            "Each doc is <bos> a0 a1 op b0 b1 = r0 r1 <eos> (10 tokens), operands "
            "and result zero-padded to 2 digits. `a`, `b` uniform in [0, 97)."
        ),
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"Wrote {OUT_DIR}/ ({TRAIN_EXAMPLES} train, {VALIDATION_EXAMPLES} validation docs)")


if __name__ == "__main__":
    main()
