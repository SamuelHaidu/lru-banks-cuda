"""The synthetic copy task, a standard test of a sequence model's memory: each document is

    <bos> x_1 .. x_L <sep> x_1 .. x_L <eos>

with x drawn uniformly from the data tokens and L uniform in [--min-len, --max-len] (per document). Only the targets
after <sep> are scored (metadata `score_after_token`), so the trainer's accuracy is the copy accuracy: the fraction of
copied tokens (and the closing <eos>) predicted exactly. Documents are packed one after another, as the pretraining
data; a row may start anywhere in a document (state reset at <bos>).

Writes DIR/train.npy, DIR/validation.npy (lengths as in training) and DIR/validation_L{n}.npy for every --sweep length
(fixed L; lengths past --max-len test length generalization), plus DIR/metadata.json. No dependency beyond Python.

    python3 scripts/make_copy_task.py /root/data/copy-v512 [--vocab 512] [--min-len 8] [--max-len 64]
        [--train-tokens 24000000] [--sweep 16,32,64,128,200]

Then:
    lbc create RUN --dataset DIR ...; lbc train RUN
    lbc eval RUN --split DIR/validation_L128.npy      (copy accuracy at L = 128)
"""

import argparse
import array
import json
import os
import random
import struct

BOS, EOS, SEP = 0, 1, 2


def write_npy(path, tokens):
    header = "{'descr': '<u2', 'fortran_order': False, 'shape': (%d,), }" % len(tokens)
    header += " " * (63 - (10 + len(header)) % 64) + "\n"  # data aligned to 64 bytes
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode())
        tokens.tofile(f)


def documents(n_tokens, vocab, lo, hi, seed):
    rng = random.Random(seed)
    out = array.array("H")
    docs = 0
    while len(out) < n_tokens:
        n = rng.randint(lo, hi)
        x = [rng.randrange(3, vocab) for _ in range(n)]
        out.extend([BOS] + x + [SEP] + x + [EOS])
        docs += 1
    return out, docs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--min-len", type=int, default=8)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--train-tokens", type=int, default=24_000_000)
    p.add_argument("--validation-tokens", type=int, default=400_000)
    p.add_argument("--sweep", default="16,32,64,128,200")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    if a.vocab % 8 or a.vocab < 8 or a.vocab > 65536:
        p.error("--vocab must be a multiple of 8 in [8, 65536]")
    os.makedirs(a.out, exist_ok=True)
    meta = {
        "name": os.path.basename(os.path.normpath(a.out)),
        "task": f"copy: <bos> x_1..x_L <sep> x_1..x_L <eos>, L uniform in [{a.min_len}, {a.max_len}], "
                f"x uniform over [3, {a.vocab}); scored after <sep>",
        "vocabulary_size": a.vocab, "doc_token_id": BOS, "bos_token_id": BOS, "eos_token_id": EOS,
        "score_after_token": SEP,
    }
    train, docs = documents(a.train_tokens, a.vocab, a.min_len, a.max_len, a.seed)
    write_npy(f"{a.out}/train.npy", train)
    meta["train"] = {"file": "train.npy", "tokens": len(train), "documents": docs}
    val, docs = documents(a.validation_tokens, a.vocab, a.min_len, a.max_len, a.seed + 1)
    write_npy(f"{a.out}/validation.npy", val)
    meta["validation"] = {"file": "validation.npy", "tokens": len(val), "documents": docs}
    meta["sweep"] = {}
    for i, n in enumerate(int(x) for x in a.sweep.split(",") if x):
        toks, docs = documents(a.validation_tokens, a.vocab, n, n, a.seed + 100 + i)
        write_npy(f"{a.out}/validation_L{n}.npy", toks)
        meta["sweep"][str(n)] = {"file": f"validation_L{n}.npy", "tokens": len(toks), "documents": docs}
    with open(f"{a.out}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "sweep"}, indent=2))
    print("sweep:", ", ".join(meta["sweep"]))


if __name__ == "__main__":
    main()
