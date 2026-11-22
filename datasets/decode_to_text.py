"""A tokenized dataset back to its text (byte-level BPE: lossless; checked by re-encoding a sample), to retrain a
tokenizer and re-encode without downloading the sources again.

Output `<out>/` (default `datasets/raw/<dataset>-text/`): per split, shards written by the workers in parallel,

    {split}-{k:03d}.txt          the documents' UTF-8 text, one after the other (no <bos> / <eos>, no separator)
    {split}-{k:03d}.offsets.npy  int64 byte offsets, documents + 1 (document i is bytes offsets[i]:offsets[i + 1])
    {split}-{k:03d}.categories.npy  int16 category id per document

and `metadata.json` (the shards in order, the dataset's `category_ids`, documents, bytes). Documents keep the
dataset's order. Workers: `--cpu-fraction` (default 0.8) of the CPUs this container may use (the cgroup quota when
there is one, else the affinity mask), one tokenizer thread each; about 8 shards per worker (small files: a
growing file reserves extra disk until it is closed).

    uv run python datasets/decode_to_text.py --dataset cosmo-code-math-pretrain-bpe1024 [--out DIR] [--cpu-fraction 0.8]
        [--splits validation train]
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BATCH = 2000


def container_cpus() -> float:
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return q / p
    except (OSError, ValueError):
        pass
    return float(len(os.sched_getaffinity(0)))


def work(job):
    """Decodes documents [a, b) of a split into one shard."""
    os.environ["RAYON_NUM_THREADS"] = "1"
    from tokenizers import Tokenizer

    tok_path, tokens_path, index_path, a, b, base, bos, eos = job
    tok = Tokenizer.from_file(tok_path)
    tokens = np.load(tokens_path, mmap_mode="r")
    index = np.load(index_path, mmap_mode="r")[a:b]
    offsets, cats, n = [0], [], 0
    with open(base + ".txt", "wb") as f:
        for s in range(0, len(index), BATCH):
            rows = index[s:s + BATCH]
            docs = []
            for start, length, _ in rows:
                ids = np.asarray(tokens[start:start + length])
                assert ids[0] == bos and ids[-1] == eos, (start, ids[:2], ids[-2:])
                docs.append(ids[1:-1].tolist())
            for text in tok.decode_batch(docs, skip_special_tokens=False):
                data = text.encode("utf-8")
                f.write(data)
                n += len(data)
                offsets.append(n)
            cats.extend(int(c) for c in rows[:, 2])
    np.save(base + ".offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(base + ".categories.npy", np.asarray(cats, dtype=np.int16))
    return len(cats), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--cpu-fraction", type=float, default=0.8)
    p.add_argument("--splits", nargs="+", default=["validation", "train"])
    p.add_argument("--check", type=int, default=2000, help="validation documents re-encoded to check")
    a = p.parse_args()
    src = ROOT / "datasets" / a.dataset
    meta = json.loads((src / "metadata.json").read_text())
    tok_path = str(ROOT / "tokenizers" / meta["tokenizer"] / "tokenizer.json")
    out = Path(a.out) if a.out else ROOT / "datasets" / "raw" / f"{a.dataset}-text"
    out.mkdir(parents=True, exist_ok=True)
    cpus = container_cpus()
    workers = max(1, int(cpus * a.cpu_fraction))
    print(f"{cpus:g} CPUs for this container, {workers} workers", flush=True)
    t0 = time.time()
    shards = {}
    for split in a.splits:
        index = np.load(src / meta[split]["file"].replace(".npy", ".categories.npy"), mmap_mode="r")
        lens = np.asarray(index[:, 1], dtype=np.int64)
        # many small shards (~8 per worker): a file is closed soon after it is written, so the filesystem's
        # speculative preallocation for growing files (several GB over 6 files of 1.5 GB on the pod) stays small
        k = max(1, min(8 * workers, len(index) // 500))
        cut = np.searchsorted(np.cumsum(lens), np.arange(1, k) * lens.sum() / k)  # equal tokens per shard
        bounds = [0, *cut.tolist(), len(index)]
        jobs = [(tok_path, str(src / meta[split]["file"]), str(src / f"{split}.categories.npy"), bounds[i],
                 bounds[i + 1], str(out / f"{split}-{i:03d}"), meta["bos_token_id"], meta["eos_token_id"])
                for i in range(k) if bounds[i + 1] > bounds[i]]
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            res = pool.map(work, jobs, chunksize=1)
        shards[split] = [{"text": f"{Path(j[5]).name}.txt", "offsets": f"{Path(j[5]).name}.offsets.npy",
                          "categories": f"{Path(j[5]).name}.categories.npy", "documents": d, "bytes": n}
                         for j, (d, n) in zip(jobs, res)]
        docs, nbytes = sum(r[0] for r in res), sum(r[1] for r in res)
        assert docs == len(index), (docs, len(index))
        print(f"{split}: {docs} documents, {nbytes / 1e9:.2f} GB of text, {len(jobs)} shards, "
              f"{time.time() - t0:.0f} s", flush=True)
    # check: a sample of validation documents re-encodes to the same ids
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    tokens = np.load(src / meta["validation"]["file"], mmap_mode="r")
    index = np.load(src / "validation.categories.npy")
    sh = shards["validation"][0]
    offs = np.load(out / sh["offsets"])
    blob = (out / sh["text"]).read_bytes()
    n = min(a.check, sh["documents"])
    ok = sum(tok.encode("<bos>" + blob[offs[i]:offs[i + 1]].decode("utf-8") + "<eos>").ids
             == np.asarray(tokens[index[i, 0]:index[i, 0] + index[i, 1]]).tolist() for i in range(n))
    assert ok == n, f"only {ok}/{n} documents re-encode identically"
    (out / "metadata.json").write_text(json.dumps({
        "source": a.dataset, "tokenizer": meta["tokenizer"], "category_ids": meta["category_ids"],
        "format": "per shard: UTF-8 text, int64 byte offsets (documents + 1), int16 category per document",
        "splits": {s: {"documents": sum(x["documents"] for x in v), "bytes": sum(x["bytes"] for x in v), "shards": v}
                   for s, v in shards.items()},
    }, indent=1))
    print(f"checked: {ok}/{n} validation documents re-encode to the same ids; done in {time.time() - t0:.0f} s "
          f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
