"""Synthetic associative-recall tasks, standard tests of a sequence model's memory. Tokens: 0 <bos>, 1 <eos>, 2 <sep>,
3 the query marker `?`, data tokens from 4.

  mqar   multi-query associative recall (Zoology, Arora et al. 2023): N key-value pairs, then every key queried once in
         a random order:   <bos> k1 v1 .. kN vN <sep> ? k v ? k v .. <eos>
         keys and values from disjoint halves of the data tokens; keys distinct within a document.
  hop2   two-hop recall: N chains a -> b -> c of distinct entities; the facts (a b), (b c) of every chain listed as pairs
         in a random order, then every chain queried by its start, answered by its end:   .. <sep> ? a c ..
  hop3   three hops: chains a -> b -> c -> d, query ? a, answer d.

Only the answers are scored (metadata `score_prev_token` = `?`: the target after `? key`), so the trainer's accuracy is
the recall accuracy. N (pairs or chains) is uniform in [--min-n, --max-n] per document; validation_N{n}.npy files hold
fixed N for a sweep (N past --max-n tests length generalization). No dependency beyond Python.

    python3 scripts/make_recall_task.py /root/data/mqar --task mqar [--vocab 512] [--min-n 4] [--max-n 64]
        [--sweep 16,32,64,96]
"""

import argparse
import array
import json
import os
import random
import struct

BOS, EOS, SEP, QRY, FIRST = 0, 1, 2, 3, 4


def write_npy(path, tokens):
    header = "{'descr': '<u2', 'fortran_order': False, 'shape': (%d,), }" % len(tokens)
    header += " " * (63 - (10 + len(header)) % 64) + "\n"  # data aligned to 64 bytes
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode())
        tokens.tofile(f)


def mqar_doc(rng, n, vocab):
    half = FIRST + (vocab - FIRST) // 2
    keys = rng.sample(range(FIRST, half), n)
    vals = [rng.randrange(half, vocab) for _ in range(n)]
    doc = [BOS]
    for k, v in zip(keys, vals):
        doc += [k, v]
    doc.append(SEP)
    for i in rng.sample(range(n), n):
        doc += [QRY, keys[i], vals[i]]
    return doc + [EOS]


def hop_doc(rng, n, vocab, hops):
    ents = rng.sample(range(FIRST, vocab), n * (hops + 1))
    chains = [ents[i * (hops + 1):(i + 1) * (hops + 1)] for i in range(n)]
    facts = [(c[j], c[j + 1]) for c in chains for j in range(hops)]
    rng.shuffle(facts)
    doc = [BOS]
    for a, b in facts:
        doc += [a, b]
    doc.append(SEP)
    for i in rng.sample(range(n), n):
        doc += [QRY, chains[i][0], chains[i][-1]]
    return doc + [EOS]


def documents(task, n_tokens, vocab, lo, hi, seed):
    rng = random.Random(seed)
    out = array.array("H")
    docs = 0
    while len(out) < n_tokens:
        n = rng.randint(lo, hi)
        out.extend(mqar_doc(rng, n, vocab) if task == "mqar" else hop_doc(rng, n, vocab, int(task[3:])))
        docs += 1
    return out, docs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--task", choices=["mqar", "hop2", "hop3"], required=True)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--min-n", type=int, default=4)
    p.add_argument("--max-n", type=int, default=64)
    p.add_argument("--train-tokens", type=int, default=24_000_000)
    p.add_argument("--validation-tokens", type=int, default=400_000)
    p.add_argument("--sweep", default="16,32,64,96")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    if a.vocab % 8 or a.vocab < 64 or a.vocab > 65536:
        p.error("--vocab must be a multiple of 8 in [64, 65536]")
    pool = (a.vocab - FIRST) // 2 if a.task == "mqar" else (a.vocab - FIRST) // (int(a.task[3:]) + 1)
    sweep = [int(x) for x in a.sweep.split(",") if x]
    if max([a.max_n] + sweep) > pool:
        p.error(f"N up to {max([a.max_n] + sweep)} needs more distinct tokens than --vocab {a.vocab} has ({pool})")
    os.makedirs(a.out, exist_ok=True)
    meta = {
        "name": os.path.basename(os.path.normpath(a.out)),
        "task": f"{a.task}: N uniform in [{a.min_n}, {a.max_n}], vocab {a.vocab}; scored: the answer after each `? key`",
        "vocabulary_size": a.vocab, "doc_token_id": BOS, "bos_token_id": BOS, "eos_token_id": EOS,
        "score_after_token": SEP, "score_prev_token": QRY,
    }
    train, docs = documents(a.task, a.train_tokens, a.vocab, a.min_n, a.max_n, a.seed)
    write_npy(f"{a.out}/train.npy", train)
    meta["train"] = {"file": "train.npy", "tokens": len(train), "documents": docs}
    val, docs = documents(a.task, a.validation_tokens, a.vocab, a.min_n, a.max_n, a.seed + 1)
    write_npy(f"{a.out}/validation.npy", val)
    meta["validation"] = {"file": "validation.npy", "tokens": len(val), "documents": docs}
    meta["sweep"] = {}
    for i, n in enumerate(sweep):
        toks, docs = documents(a.task, a.validation_tokens, a.vocab, n, n, a.seed + 100 + i)
        write_npy(f"{a.out}/validation_N{n}.npy", toks)
        meta["sweep"][str(n)] = {"file": f"validation_N{n}.npy", "tokens": len(toks), "documents": docs}
    with open(f"{a.out}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "sweep"}, indent=2))
    print("sweep:", ", ".join(meta["sweep"]))


if __name__ == "__main__":
    main()
