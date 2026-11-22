"""Concatenates pretraining datasets with the same tokenizer (no `loss_spans`) into one, e.g.
`cosmo-code-math-pretrain-bpe1024` = cosmopedia-pretrain-bpe1024 (70%) + stack-edu-bpe1024 (25%) + finemath-bpe1024 (5%):

    uv run python datasets/build_pretrain_mix.py --parts cosmopedia-pretrain-bpe1024 stack-edu-bpe1024 \\
        finemath-bpe1024 --name cosmo-code-math-pretrain-bpe1024 [--out DIR]

Train and validation token files one after the other (with `data.random_windows` the windows are shuffled anyway),
the categories of every part kept (ids renumbered in order; `{split}.categories.npy` starts shifted), validation the
parts' validation splits. `avg_chars_per_token` is the token-weighted mean of the parts' (it sets bpc).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cosmopedia import NpyWriter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COPY = 1 << 26  # tokens per copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    metas = [json.loads((ROOT / "datasets" / d / "metadata.json").read_text()) for d in a.parts]
    for m in metas:
        assert m["tokenizer"] == metas[0]["tokenizer"] and m["vocabulary_size"] == metas[0]["vocabulary_size"]
        assert not m.get("loss_spans"), f"{m['name']} has loss_spans"
    out = Path(a.out) if a.out else ROOT / "datasets" / a.name
    out.mkdir(parents=True, exist_ok=True)
    cat_ids, cats = {}, {}
    meta = {k: metas[0][k] for k in ("tokenizer", "vocabulary_size", "doc_token_id", "bos_token_id", "eos_token_id")}
    meta |= {"name": a.name, "dtype": "uint16", "parts": {}, "category_index": "{split}.categories.npy"}
    for sp in ("train", "validation"):
        w = NpyWriter(out / f"{sp}.npy")
        index, nbytes = [], 0.0
        for d, m in zip(a.parts, metas):
            src = np.load(ROOT / "datasets" / d / m[sp]["file"], mmap_mode="r")
            idx = np.load(ROOT / "datasets" / d / f"{sp}.categories.npy").copy()
            names = {v: k for k, v in m["category_ids"].items()}
            for c in sorted(names):
                cat_ids.setdefault(names[c], len(cat_ids))
                cats.setdefault(names[c], {})[sp] = m["categories"][names[c]][sp]
            idx[:, 0] += w.n
            idx[:, 2] = [cat_ids[names[c]] for c in idx[:, 2]]
            index.append(idx)
            for s in range(0, len(src), COPY):
                w.write(np.asarray(src[s:s + COPY]))
            nbytes += m[sp]["bytes_per_token"] * m[sp]["tokens"]
            print(f"{sp}: {d} {m[sp]['tokens'] / 1e9:.3f}B tokens", flush=True)
        w.close()
        idx = np.concatenate(index)
        np.save(out / f"{sp}.categories.npy", idx)
        lens = idx[:, 1]
        meta[sp] = {"file": f"{sp}.npy", "tokens": int(w.n), "documents": int(len(idx)),
                    "doc_tokens": {k: int(np.percentile(lens, q)) for k, q in (("p50", 50), ("p90", 90), ("p99", 99))}
                    | {"max": int(lens.max())}, "bytes_per_token": round(nbytes / w.n, 4)}
    total = sum(m["train"]["tokens"] for m in metas)
    meta["avg_chars_per_token"] = round(sum(m["avg_chars_per_token"] * m["train"]["tokens"] for m in metas) / total, 4)
    meta["parts"] = {d: {"train_tokens": m["train"]["tokens"], "share": round(m["train"]["tokens"] / total, 4),
                         "source": m["source"]} for d, m in zip(a.parts, metas)}
    meta["categories"], meta["category_ids"] = cats, cat_ids
    meta["source"] = " + ".join(f"{d} ({100 * v['share']:.0f}%)" for d, v in meta["parts"].items()) \
        + " (datasets/build_pretrain_mix.py)"
    meta["notes"] = "Pretraining text and code: <bos>document<eos>, every token scored."
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    check = np.load(out / "train.npy", mmap_mode="r")
    assert check.shape == (total,) and check[0] == 0 and check[-1] == 1
    print(json.dumps({k: meta[k] for k in ("train", "validation", "parts", "avg_chars_per_token")}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
