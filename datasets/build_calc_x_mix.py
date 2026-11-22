"""Builds `calc-x-mix-bpe32768`: Calc-X (`datasets/build_calc_x.py`) weighted per source, plus random pretraining
documents (`pretrain-mix-bpe32768` train) as `--pretrain-share` of the tokens, shuffled at the document level.
No `loss_spans`: every token is scored (questions, calls, results, answers), as in pretraining. `"chat": true` in the
metadata tells the web chat that its runs speak the chat format anyway.

    uv run python datasets/build_calc_x_mix.py [--pretrain-share 0.3] [--seed 0] [--name N]
        [--weights '{"gsm8k": [6.5, 1], "ape210k": [0.35, 1], ...}']

Calc-X documents with no calculator call are dropped (AQuA-RAT: fewer than 3 calls); each source is repeated
`WEIGHTS` (or `--weights`) times (a fraction: that share of its documents once more, at random). Pretraining documents are drawn
without replacement at random (their category proportions follow). Validation = the pretraining validation split as
is (val_loss / val_bpc stay comparable with the pretraining runs: the forgetting measure; the task is measured by
answer accuracy on Calc-X's test splits). `avg_chars_per_token` is the pretraining value, so val_bpc matches.
Categories: the pretraining ones, then `<prefix>/<source>` (`--prefix`, default calc-x). Any task dataset in this
layout works (`--calc`, e.g. kodcode-agent-bpe32768 with `--prefix kodcode --weights '{"solve": [1, 0], "fix": [1, 0]}'`);
without a `train.calls.npy` the minimum-calls filter is skipped.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# source: (repeats, minimum calls per document)
WEIGHTS = {"gsm8k": (5, 1), "mawps": (5, 1), "ape210k": (2.5, 1), "math_qa": (3, 1), "aqua_rat": (3, 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calc", default=str(ROOT / "datasets" / "calc-x-bpe32768"))
    p.add_argument("--pretrain", default=str(ROOT / "datasets" / "pretrain-mix-bpe32768"))
    p.add_argument("--name", default="calc-x-mix-bpe32768")
    p.add_argument("--out", default=None)
    p.add_argument("--pretrain-share", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="calc-x", help="category name prefix of the task dataset's sources")
    p.add_argument("--weights", default=None, help="JSON {source: [repeats, min calls]} (default: WEIGHTS)")
    a = p.parse_args()
    weights = {k: tuple(v) for k, v in json.loads(a.weights).items()} if a.weights else WEIGHTS
    rng = np.random.default_rng(a.seed)
    calc, pre = Path(a.calc), Path(a.pretrain)
    out = Path(a.out) if a.out else ROOT / "datasets" / a.name
    out.mkdir(parents=True, exist_ok=True)
    cm, pm = json.loads((calc / "metadata.json").read_text()), json.loads((pre / "metadata.json").read_text())
    assert cm["tokenizer"] == pm["tokenizer"], (cm["tokenizer"], pm["tokenizer"])
    cat_ids = dict(pm["category_ids"])
    off = max(cat_ids.values()) + 1
    for s, i in cm["category_ids"].items():
        cat_ids[f"{a.prefix}/{s}"] = off + i

    # Calc-X: filter, repeat
    ct = np.load(calc / "train.npy", mmap_mode="r")
    ci = np.load(calc / "train.categories.npy")
    calls = np.load(calc / "train.calls.npy") if (calc / "train.calls.npy").exists() else np.full(len(ci), 1 << 14)
    docs = []  # (token slice, category id)
    calc_tokens = 0
    for s, (w, min_calls) in weights.items():
        sel = np.flatnonzero((ci[:, 2] == cm["category_ids"][s]) & (calls >= min_calls))
        reps = [sel] * int(w)
        if w % 1:
            reps.append(rng.choice(sel, int(round(len(sel) * (w % 1))), replace=False))
        for r in reps:
            for k in r:
                docs.append((np.asarray(ct[ci[k, 0]:ci[k, 0] + ci[k, 1]]), off + int(ci[k, 2])))
                calc_tokens += int(ci[k, 1])
        print(f"  {a.prefix}/{s}: {len(sel):,} documents (calls >= {min_calls}) x {w}", flush=True)

    # pretraining: random documents up to the share
    want = int(calc_tokens * a.pretrain_share / (1 - a.pretrain_share))
    pt = np.load(pre / "train.npy", mmap_mode="r")
    pi = np.load(pre / "train.categories.npy")
    order = rng.permutation(len(pi))
    n = int(np.searchsorted(np.cumsum(pi[order, 1]), want)) + 1
    for k in np.sort(order[:n]):  # sorted: sequential reads of the memory-mapped file
        docs.append((np.asarray(pt[pi[k, 0]:pi[k, 0] + pi[k, 1]]), int(pi[k, 2])))
    pre_tokens = int(pi[order[:n], 1].sum())
    print(f"  pretraining: {n:,} documents, {pre_tokens:,} tokens", flush=True)

    # shuffle documents, write
    perm = rng.permutation(len(docs))
    lens = np.array([len(docs[i][0]) for i in perm], np.int64)
    tokens = np.concatenate([docs[i][0] for i in perm]).astype(np.uint16)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    cats = np.array([docs[i][1] for i in perm], np.int64)
    np.save(out / "train.npy", tokens)
    np.save(out / "train.categories.npy", np.stack([starts, lens, cats], 1))
    for f in ("validation.npy", "validation.categories.npy"):
        shutil.copyfile(pre / f, out / f)
    assert all(tokens[s] == pm["bos_token_id"] and tokens[s + n - 1] == pm["eos_token_id"] for s, n in zip(starts, lens))

    names = {v: k for k, v in cat_ids.items()}
    categories = {}
    for c in np.unique(cats):
        sel = cats == c
        categories[names[int(c)]] = {"train": {"documents": int(sel.sum()), "tokens": int(lens[sel].sum())}}
    for k, v in pm["categories"].items():
        if "validation" in v:
            categories.setdefault(k, {})["validation"] = v["validation"]
    meta = {k: pm[k] for k in ("tokenizer", "vocabulary_size", "doc_token_id", "bos_token_id", "eos_token_id",
                               "dtype", "category_index", "avg_chars_per_token")}
    meta.update(
        name=a.name, category_ids=cat_ids, categories=categories, validation=pm["validation"],
        train={"file": "train.npy", "tokens": int(len(tokens)), "documents": int(len(lens)),
               "calc_x_tokens": calc_tokens, "pretrain_tokens": pre_tokens,
               "doc_tokens": {k: int(np.percentile(lens, q)) for k, q in (("p50", 50), ("p90", 90), ("p99", 99))}
               | {"max": int(lens.max())}},
        weights={s: {"repeats": w, "min_calls": m} for s, (w, m) in weights.items()},
        pretrain_share=a.pretrain_share, seed=a.seed,
        chat=True,  # trained on the chat format, though no loss_spans: the web chat offers Chat for its runs
        source=f"{calc.name} (weighted, datasets/build_calc_x_mix.py) + random {pre.name} train documents",
        notes="Every token scored (no loss_spans). Validation = the pretraining validation split; "
              "avg_chars_per_token = the pretraining value (val_bpc comparable).")
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    tot = len(tokens)
    print(f"train: {len(lens):,} documents, {tot:,} tokens: calc-x {calc_tokens:,} ({calc_tokens / tot:.1%}), "
          f"pretraining {pre_tokens:,} ({pre_tokens / tot:.1%}) -> {out}")
    for k, v in sorted(categories.items(), key=lambda kv: -kv[1].get("train", {}).get("tokens", 0)):
        if "train" in v:
            print(f"  {k:28s} {v['train']['documents']:9,d} docs {v['train']['tokens']:12,d} tokens "
                  f"{v['train']['tokens'] / tot:6.1%}")


if __name__ == "__main__":
    sys.exit(main())
