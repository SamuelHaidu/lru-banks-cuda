"""Cosmopedia for pretraining (`cosmopedia-pretrain-bpe1024`): the generated texts only, no prompt and no chat tags,

    <bos>{text}<eos>

every token scored (no `loss_spans`), tokenized with the agent-mix-v2 BPE (the same 1024 ids: a model pretrained here
fine-tunes on agent-mix-v2 as is). Balanced across the 8 subsets (HuggingFaceTB/cosmopedia rev 0ae6ec6) for a token
budget (default 3B): subsets smaller than an even share go in whole, the rest get equal shares (water-filling on
token estimates from 120 sampled rows per subset: at 3B, khanacademy 37M, openstax 215M, wikihow 372M whole, then
auto_math_text, stanford, stories, web_samples_v1, web_samples_v2 ~475M each). A sampled subset reads a random spread
of its parquet shards (4x the needed share of files) and keeps each row with the matching probability, so no shard
(topic cluster) dominates. Validation: ~1000 documents per subset, drawn from the kept rows. One category per subset.

Parquet files are streamed one at a time (the next downloads while the current one is tokenized) and deleted; token
ids are uint16, streamed into the .npy files.

    uv run --with pyarrow python datasets/build_cosmopedia_pretrain.py [--budget 3e9] [--out DIR] [--threads 5]
"""

import argparse
import json
import math
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cosmopedia import ALL, REPO, REV, TAGS, NpyWriter, api, download, files_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = "agent-mix-v2-bpe1024"
BATCH = 2000


# rows (datasets-server /size) and mean tokens per text with this tokenizer (<bos> / <eos> included; 120 sampled rows
# per subset, measured 2026-11-15)
SIZES = {"auto_math_text": (1949895, 1424), "khanacademy": (24123, 1550), "openstax": (126332, 1703),
         "stanford": (1020024, 2119), "stories": (4992964, 1229), "web_samples_v1": (12426348, 1863),
         "web_samples_v2": (10345867, 1742), "wikihow": (179191, 2074)}


def estimates(subsets) -> dict:
    """(rows, tokens) per subset."""
    return {c: (SIZES[c][0], float(SIZES[c][0] * SIZES[c][1])) for c in subsets}


def allocate(est: dict, budget: float) -> dict:
    """Water-filling: subsets under an even share whole, the rest split evenly."""
    left, alloc = {c: v for c, (_, v) in est.items()}, {}
    while left:
        share = (budget - sum(alloc.values())) / len(left)
        small = {c: v for c, v in left.items() if v <= share}
        if not small:
            alloc.update({c: share for c in left})
            break
        alloc.update(small)
        for c in small:
            left.pop(c)
    return alloc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=3e9, help="train tokens")
    p.add_argument("--name", default="cosmopedia-pretrain-bpe1024")
    p.add_argument("--out", default=None, help="output directory (default datasets/<name>)")
    p.add_argument("--validation-docs", type=int, default=1000, help="per subset")
    p.add_argument("--threads", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subsets", nargs="+", default=ALL, choices=ALL)
    a = p.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(a.threads)
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ROOT / "tokenizers" / TOKENIZER / "tokenizer.json"))
    assert tok.token_to_id("<bos>") == 0 and tok.token_to_id("<eos>") == 1 and tok.get_vocab_size() == 1024
    est = estimates(a.subsets)
    alloc = allocate(est, a.budget)
    rng = np.random.default_rng(a.seed)
    jobs, plan = [], {}
    for c in a.subsets:
        files = files_of(c)
        frac = min(1.0, alloc[c] / est[c][1])
        k = len(files) if frac >= 1 else min(len(files), max(2, math.ceil(4 * frac * len(files))))
        chosen = sorted(rng.choice(len(files), k, replace=False).tolist())
        keep = min(1.0, frac * len(files) / k)
        docs_kept = est[c][0] * frac
        plan[c] = {"tokens": alloc[c], "files": k, "of": len(files), "keep": keep,
                   "p_val": min(0.5, a.validation_docs / max(1.0, docs_kept))}
        jobs += [(c, files[i]) for i in chosen]
    print(json.dumps({c: {k: (round(v / 1e6) if k == "tokens" else round(v, 4) if isinstance(v, float) else v)
                          for k, v in d.items()} for c, d in plan.items()}, indent=1), flush=True)
    raw = ROOT / "datasets" / "raw" / "cosmopedia"
    out = Path(a.out) if a.out else ROOT / "datasets" / a.name
    out.mkdir(parents=True, exist_ok=True)
    writers = {sp: NpyWriter(out / f"{sp}.npy") for sp in ("train", "validation")}
    index = {sp: [] for sp in writers}
    stats = {sp: {c: {"documents": 0, "tokens": 0} for c in a.subsets} for sp in writers}
    chars, utf8, dropped = [0, 0], {sp: 0 for sp in writers}, 0
    cat = {c: i for i, c in enumerate(a.subsets)}
    nxt = [None]

    def prefetch(i):
        nxt[0] = download(jobs[i][1], raw / jobs[i][0] / Path(jobs[i][1]).name) if i < len(jobs) else None

    th = threading.Thread(target=prefetch, args=(0,))
    th.start()
    t0, total = time.time(), 0
    for i, (c, f) in enumerate(jobs):
        th.join()
        path = nxt[0]
        th = threading.Thread(target=prefetch, args=(i + 1,))
        th.start()
        if stats["train"][c]["tokens"] >= plan[c]["tokens"]:  # the subset's budget is reached (estimates are noisy)
            path.unlink()
            continue
        texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
        path.unlink()
        texts = [t for t, r in zip(texts, rng.random(len(texts))) if r < plan[c]["keep"]]
        docs = []
        for t in texts:
            t = (t or "").strip()
            if not t or any(tag in t for tag in TAGS):
                dropped += 1
                continue
            docs.append(t)
        val = rng.random(len(docs)) < plan[c]["p_val"]
        for b in range(0, len(docs), BATCH):
            encs = tok.encode_batch([f"<bos>{d}<eos>" for d in docs[b:b + BATCH]])
            for d, e, v in zip(docs[b:b + BATCH], encs, val[b:b + BATCH]):
                sp = "validation" if v else "train"
                if sp == "train" and stats["train"][c]["tokens"] >= plan[c]["tokens"]:
                    continue
                x = np.asarray(e.ids, dtype=np.uint16)
                index[sp].append((writers[sp].write(x), len(x), cat[c]))
                stats[sp][c]["documents"] += 1
                stats[sp][c]["tokens"] += len(x)
                utf8[sp] += len(d.encode("utf-8")) + 10
                if sp == "train":
                    chars[0] += len(d)
                    chars[1] += len(x) - 2
                total += len(x)
        el = time.time() - t0
        print(f"[{i + 1}/{len(jobs)}] {c} {Path(f).name}: {len(docs)} docs, {c} {stats['train'][c]['tokens'] / 1e6:.0f}"
              f"/{plan[c]['tokens'] / 1e6:.0f}M, {total / 1e9:.3f}B tokens, {total / el / 1e6:.2f}M tokens/s, "
              f"{el / 60:.1f} min", flush=True)
    th.join()
    if nxt[0] is not None and nxt[0].exists():
        nxt[0].unlink()
    meta = {
        "name": a.name,
        "source": f"{REPO} (rev {REV[:7]}), texts only, balanced over the subsets for {a.budget / 1e9:g}B tokens "
                  f"(datasets/build_cosmopedia_pretrain.py); tokenizer {TOKENIZER}",
        "categories": {c: {sp: stats[sp][c] for sp in writers} for c in a.subsets},
        "category_ids": cat, "category_index": "{split}.categories.npy", "plan": plan,
        "tokenizer": TOKENIZER, "vocabulary_size": 1024,
        "doc_token_id": 0, "bos_token_id": 0, "eos_token_id": 1,
        "avg_chars_per_token": round(chars[0] / max(1, chars[1]), 4), "dtype": "uint16", "seed": a.seed,
        "dropped_documents": dropped,
        "notes": "Pretraining text: <bos>text<eos>, every token scored. Token = tag (ids 0-15), byte (16 + byte) or merge.",
    }
    for sp, w in writers.items():
        w.close()
        idx = np.asarray(index[sp], dtype=np.int64).reshape(-1, 3)
        np.save(out / f"{sp}.categories.npy", idx)
        lens = idx[:, 1]
        meta[sp] = {"file": f"{sp}.npy", "tokens": int(w.n), "documents": int(len(idx)),
                    "doc_tokens": {k: int(np.percentile(lens, q)) for k, q in (("p50", 50), ("p90", 90), ("p99", 99))}
                    | {"max": int(lens.max())}, "bytes_per_token": round(utf8[sp] / max(1, w.n), 4)}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    check = np.load(out / "train.npy", mmap_mode="r")
    assert check.shape == (meta["train"]["tokens"],) and check[0] == 0 and check[-1] == 1
    print(json.dumps({k: meta[k] for k in ("train", "validation", "avg_chars_per_token")}, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    sys.exit(main())
