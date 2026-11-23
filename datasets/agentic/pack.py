"""Tokenizes the converted text (agentic/convert.py) into compressed chunks of ~1B tokens and uploads them to R2.

    s3://<bucket>/corpora/<corpus>/
      tokenizer.json, metadata.json      the tokenizer; sources, chunks (documents, tokens, bytes, sha256)
      <source>/train-0000.npy.zst        whole documents, ~--chunk-tokens tokens: uint16 .npy, zstd --long
      <source>/train-0000.index.npy      int64 (start, length, category) per document
      <source>/validation.npy.zst + .index.npy, test.npy.zst + .index.npy

Each document is <bos> text <eos>. Validation and test are convert.py's validation documents, alternately (a fixed
hash of the first user message picked them, so every future build holds out the same ones). A chunk is a plain .npy
once decompressed: any set of chunks makes a split (agentic/assemble.py joins them into a dataset the engine reads).

R2 credentials come from the pod's environment (REMOTE_POD_S3_*, set by remote-pod up).

    uv run --with tokenizers --with numpy --with boto3 python agentic/pack.py --text /work/text \\
        --tokenizer /work/repo/tokenizers/agentic-bpe32768 --corpus agentic-v1 [--workers 28]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


def encode_shard(job):
    os.environ["RAYON_NUM_THREADS"] = "1"
    from tokenizers import Tokenizer

    tok_path, base = job
    tok = Tokenizer.from_file(tok_path)
    offs = np.load(base + ".offsets.npy")
    cats = np.load(base + ".categories.npy")
    blob = Path(base + ".txt").read_bytes()
    # batches of ~4 MB of text: an Encoding keeps offsets and token strings (~50 bytes per token), so a batch of
    # long SWE trajectories by count alone took gigabytes per worker
    ids, lens, k = [], [], 0
    while k < len(cats):
        j = k + 1
        while j < len(cats) and offs[j + 1] - offs[k] < (4 << 20):
            j += 1
        docs = ["<bos>" + blob[offs[i]:offs[i + 1]].decode("utf-8") + "<eos>" for i in range(k, j)]
        for e in tok.encode_batch(docs, add_special_tokens=False):
            ids.append(np.asarray(e.ids, dtype=np.uint16))
            lens.append(len(e.ids))
        k = j
    # one array per shard: cheap to send back (one per document made the main process run out of memory)
    flat = np.concatenate(ids) if ids else np.zeros(0, np.uint16)
    return flat, np.asarray(lens, dtype=np.int64), cats.astype(np.int64), int(offs[-1])


def ordered(pool, fn, jobs, window):
    """pool.imap with at most `window` jobs in flight (imap runs ahead without limit and piles results up)."""
    pending = collections.deque()
    it = iter(jobs)
    for j in it:
        pending.append(pool.apply_async(fn, (j,)))
        if len(pending) >= window:
            break
    while pending:
        r = pending.popleft().get()
        nxt = next(it, None)
        if nxt is not None:
            pending.append(pool.apply_async(fn, (nxt,)))
        yield r


class Uploader:
    """Compresses and uploads finished chunks in the background (one at a time: zstd uses many threads)."""

    def __init__(self, bucket, prefix, threads):
        import boto3

        self.s3 = boto3.client("s3", endpoint_url=os.environ["REMOTE_POD_S3_ENDPOINT"],
                               aws_access_key_id=os.environ["REMOTE_POD_S3_ACCESS_KEY"],
                               aws_secret_access_key=os.environ["REMOTE_POD_S3_SECRET_KEY"],
                               region_name=os.environ.get("REMOTE_POD_S3_REGION") or "auto")
        self.bucket, self.prefix, self.threads = bucket, prefix, threads
        self.thread, self.results, self.errors = None, [], []

    def put(self, path: Path, key: str):
        self.s3.upload_file(str(path), self.bucket, f"{self.prefix}/{key}")

    def _job(self, npy: Path, index: Path, key: str, info: dict):
        try:
            zst = npy.with_suffix(".npy.zst")
            subprocess.run(["zstd", "-q", "-f", "--rm", "-12", "--long=30", f"-T{self.threads}", str(npy), "-o",
                            str(zst)], check=True)
            h = hashlib.sha256()
            with open(zst, "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    h.update(b)
            self.put(zst, key + ".npy.zst")
            self.put(index, key + ".index.npy")
            info.update(file=key + ".npy.zst", index=key + ".index.npy", bytes_zst=zst.stat().st_size,
                        sha256=h.hexdigest())
            zst.unlink()
            index.unlink()
            self.results.append(info)
            print(f"    uploaded {key}: {info['tokens'] / 1e9:.2f}B tokens -> {info['bytes_zst'] / 1e9:.2f} GB",
                  flush=True)
        except Exception as e:  # noqa: BLE001: reported at the end
            self.errors.append(f"{key}: {e}")

    def submit(self, npy, index, key, info):
        self.wait()
        self.thread = threading.Thread(target=self._job, args=(npy, index, key, info))
        self.thread.start()

    def wait(self):
        if self.thread:
            self.thread.join()
            self.thread = None


def write_chunk(work: Path, key: str, parts, lens, cats):
    """The chunk as a .npy (uint16 ids) and its index (start, length, category per document)."""
    npy, index = work / (key.replace("/", "__") + ".npy"), work / (key.replace("/", "__") + ".index.npy")
    np.save(npy, np.concatenate(parts) if parts else np.zeros(0, np.uint16))
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]) if len(lens) else np.zeros(0, np.int64)
    np.save(index, np.stack([starts, lens, cats], axis=1).astype(np.int64))
    return npy, index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default="/work/text")
    ap.add_argument("--tokenizer", required=True, help="a directory with tokenizer.json")
    ap.add_argument("--corpus", default="agentic-v1")
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--chunk-tokens", type=float, default=1e9)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--zstd-threads", type=int, default=8)
    ap.add_argument("--work", default="/work/pack")
    a = ap.parse_args()
    text, work = Path(a.text), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    tok_path = str(Path(a.tokenizer) / "tokenizer.json")
    sources = a.sources or sorted(p.name for p in text.iterdir() if p.is_dir() and not p.name.startswith("_"))
    up = Uploader(os.environ["REMOTE_POD_S3_BUCKET"], f"corpora/{a.corpus}", a.zstd_threads)
    up.put(Path(tok_path), "tokenizer.json")
    report = Path(a.tokenizer) / "report.json"
    if report.exists():
        up.put(report, "tokenizer_report.json")
    meta = {"corpus": a.corpus, "tokenizer": Path(a.tokenizer).name, "dtype": "uint16", "bos_token_id": 0,
            "eos_token_id": 1, "chunk_tokens": a.chunk_tokens, "sources": {}}
    if (work / "metadata.json").exists():  # a restart: the sources already uploaded are kept
        meta = json.loads((work / "metadata.json").read_text())
        print("resuming; done:", ", ".join(meta["sources"]), flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        for src in sources:
            if src in meta["sources"]:
                continue
            smeta = json.loads((text / src / "metadata.json").read_text())
            meta["category_ids"] = smeta["category_ids"]
            entry = {"stats": smeta.get("stats", {}), "train": [], "chars": 0}
            # validation documents: alternately validation and test
            held = {"validation": ([], [], []), "test": ([], [], [])}
            k = 0
            for sh in smeta["splits"].get("validation", {}).get("shards", []):
                flat, lens, cats, nbytes = encode_shard((tok_path, str(text / src / Path(sh["text"]).stem)))
                ends = np.cumsum(lens)
                for d in range(len(lens)):
                    p = held["validation" if k % 2 == 0 else "test"]
                    p[0].append(flat[ends[d] - lens[d]:ends[d]]); p[1].append(lens[d]); p[2].append(cats[d])
                    k += 1
            for split, (parts, lens, cats) in held.items():
                if not lens:
                    continue
                key = f"{src}/{split}"
                npy, index = write_chunk(work, key, parts, np.asarray(lens), np.asarray(cats))
                info = {"documents": len(lens), "tokens": int(np.sum(lens))}
                entry[split] = info
                up.submit(npy, index, key, info)
            # train: whole documents into chunks of ~chunk_tokens
            jobs = [(tok_path, str(text / src / Path(sh["text"]).stem))
                    for sh in smeta["splits"]["train"]["shards"]]
            parts, lens, cats, n, c = [], [], [], 0, 0
            for flat, ln, ct, nbytes in ordered(pool, encode_shard, jobs, 2 * a.workers):
                parts.append(flat); lens.append(ln); cats.append(ct); n += int(ln.sum()); entry["chars"] += nbytes
                if n >= a.chunk_tokens:
                    key = f"{src}/train-{c:04d}"
                    L, C = np.concatenate(lens), np.concatenate(cats)
                    npy, index = write_chunk(work, key, parts, L, C)
                    info = {"documents": len(L), "tokens": n}
                    entry["train"].append(info)
                    up.submit(npy, index, key, info)
                    print(f"  {key}: {len(L):,} documents, {n / 1e9:.2f}B tokens ({time.time() - t0:.0f} s)",
                          flush=True)
                    parts, lens, cats, n, c = [], [], [], 0, c + 1
            if n:
                key = f"{src}/train-{c:04d}"
                L, C = np.concatenate(lens), np.concatenate(cats)
                npy, index = write_chunk(work, key, parts, L, C)
                info = {"documents": len(L), "tokens": n}
                entry["train"].append(info)
                up.submit(npy, index, key, info)
            up.wait()
            tokens = sum(x["tokens"] for x in entry["train"])
            entry["train_tokens"] = tokens
            entry["chars_per_token"] = entry.pop("chars") / max(1, tokens)  # bytes of UTF-8 per token (train)
            meta["sources"][src] = entry
            print(f"{src}: {tokens / 1e9:.2f}B train tokens in {len(entry['train'])} chunks, "
                  f"{entry['chars_per_token']:.2f} bytes/token ({time.time() - t0:.0f} s)", flush=True)
            # metadata after every source: a partial corpus is usable
            (work / "metadata.json").write_text(json.dumps(meta, indent=1))
            up.put(work / "metadata.json", "metadata.json")
    if up.errors:
        raise SystemExit("upload errors:\n" + "\n".join(up.errors))
    total = sum(s["train_tokens"] for s in meta["sources"].values())
    print(f"done: {total / 1e9:.2f}B train tokens in {time.time() - t0:.0f} s -> corpora/{a.corpus}")


if __name__ == "__main__":
    main()
