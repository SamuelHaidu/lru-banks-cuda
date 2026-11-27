"""ClimbMix (karpathy/climbmix-400b-shuffle: 6,543 pre-shuffled parquet shards of ~86k web documents, ~62.6M tokens
each in agentic-bpe32768) as text shards for datasets/agentic/pack.py (the same layout as finewiki/convert.py):

    <out>/climbmix/{split}-{file:04d}{part:02d}.txt / .offsets.npy / .categories.npy + metadata.json

The shards are already shuffled, so the first --files of them are a fair sample (80 files: ~5B tokens). Each worker
downloads one parquet file, keeps the `text` column, writes shards of ~--shard-mb MB and deletes the parquet. 1 in
--val-every documents (crc32 of the text, so every build holds out the same ones) go to validation (pack.py splits
them into validation and test).

    uv run --with pyarrow --with numpy --with huggingface_hub python climbmix/convert.py --files 80 --out /work/text
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import zlib
from pathlib import Path

import numpy as np

REPO = "karpathy/climbmix-400b-shuffle"


def convert_file(job):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    i, name, raw, out, val_every, shard_bytes = job
    t0 = time.time()
    path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=raw)
    t_dl = time.time() - t0
    shards = {"train": [], "validation": []}
    writers, empty = {}, 0

    def close(split):
        w = writers.pop(split)
        w["fh"].close()
        n = len(w["offs"]) - 1
        np.save(f"{w['base']}.offsets.npy", np.asarray(w["offs"], dtype=np.int64))
        np.save(f"{w['base']}.categories.npy", np.zeros(n, dtype=np.int16))
        b = w["base"].name
        shards[split].append({"text": f"{b}.txt", "offsets": f"{b}.offsets.npy", "categories": f"{b}.categories.npy",
                              "documents": n, "bytes": w["offs"][-1]})

    parts = {"train": 0, "validation": 0}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if not text or not text.strip():
                empty += 1
                continue
            b = text.encode("utf-8")
            split = "validation" if zlib.crc32(b) % val_every == 0 else "train"
            if split not in writers:
                base = Path(out) / f"{split}-{i:04d}{parts[split]:02d}"
                parts[split] += 1
                writers[split] = {"base": base, "fh": open(f"{base}.txt", "wb"), "offs": [0]}
            w = writers[split]
            w["fh"].write(b)
            w["offs"].append(w["offs"][-1] + len(b))
            if w["offs"][-1] >= shard_bytes:
                close(split)
    for split in list(writers):
        close(split)
    os.remove(path)
    docs = sum(s["documents"] for v in shards.values() for s in v)
    print(f"  {name}: {docs:,} docs, download {t_dl:.0f} s, total {time.time() - t0:.0f} s", flush=True)
    return i, shards, empty


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", type=int, default=80, help="the first N parquet shards")
    ap.add_argument("--out", default="/work/text")
    ap.add_argument("--raw", default="/work/raw/climbmix")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--shard-mb", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    files = [f"shard_{i:05d}.parquet" for i in range(a.files)]
    out = Path(a.out) / "climbmix"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(i, f, a.raw, str(out), a.val_every, a.shard_mb << 20) for i, f in enumerate(files)]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = sorted(pool.imap_unordered(convert_file, jobs))
    splits = {}
    for split in ("train", "validation"):
        v = [s for _, sh, _ in res for s in sh[split]]
        splits[split] = {"documents": sum(x["documents"] for x in v), "bytes": sum(x["bytes"] for x in v), "shards": v}
    stats = {"kept": sum(s["documents"] for s in splits.values()), "bytes": sum(s["bytes"] for s in splits.values()),
             "drop: empty text": sum(e for _, _, e in res), "parquet files": len(files)}
    meta = {"source": f"{REPO} (first {len(files)} shards)", "category_ids": {"climbmix": 0}, "stats": stats,
            "format": "per shard: UTF-8 text, int64 byte offsets (documents + 1), int16 category per document",
            "splits": splits}
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"done in {time.time() - t0:.0f} s: {stats['kept']:,} docs, {stats['bytes'] / 1e9:.2f} GB "
          f"(validation {splits['validation']['documents']:,}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
