"""HuggingFaceTB/stack-edu (rev eeec5ca: The Stack v2 files kept by per-language "educational value" classifiers,
15 languages) for pretraining (`stack-edu-bpe1024`): each file as

    <bos>{file contents}<eos>

every token scored (no `loss_spans`), tokenized with the agent-mix-v2 BPE (the same 1024 ids as
`cosmopedia-pretrain-bpe1024`, so they concatenate: `datasets/build_pretrain_mix.py`). The same number of train
tokens per language (`--budget` / 15; default: 25/70 of cosmopedia-pretrain's train tokens: the mix
`cosmo-code-math-pretrain-bpe1024` is 70% cosmopedia, 25% code, 5% finemath).

The dataset holds only ids and metadata; the contents are Software Heritage blobs, read anonymously over HTTPS
(`https://softwareheritage.s3.amazonaws.com/content/<blob_id>`, gzip). Per language: one random metadata shard
(parquet, millions of files; downloaded and deleted), files of 200 B..50 KB, the highest classifier scores first:
the score threshold is the highest of 4.5, 4, 3.5, 3, 2.5 whose files hold 3x the language's share (in bytes, from
the measured bytes per token), shuffled; then fetched `--workers` at a time, tokenized, written until the share is
reached. The first `--validation-docs` files of each language go to validation. Files that fail to download, have a
NUL byte, or contain one of our tag strings are skipped. One category per language.

    uv run --with pyarrow python datasets/build_stack_edu.py [--budget 1.05e9] [--out DIR] [--workers 128]
        [--languages Python Go ...] [--samples 2]   # --samples: print files of the first language and stop
"""

import argparse
import gzip
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cosmopedia import TAGS, NpyWriter, api  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO, REV = "HuggingFaceTB/stack-edu", "eeec5caac5cc3758a18f1d3ba4416837a9ba814c"
LANGUAGES = ["C", "CSharp", "Cpp", "Go", "Java", "JavaScript", "Markdown", "PHP", "Python", "Ruby", "Rust", "SQL",
             "Shell", "Swift", "TypeScript"]
TOKENIZER = "agent-mix-v2-bpe1024"
BLOB_URL = "https://softwareheritage.s3.amazonaws.com/content/{}"
MIN_BYTES, MAX_BYTES = 200, 50_000
CHUNK = 4000  # files per download batch


def shard_of(lang: str, rng) -> str:
    tree = api(f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}/{lang}")
    files = sorted(f["path"] for f in tree if f["path"].endswith(".parquet"))
    return files[int(rng.integers(len(files)))]


def download_shard(path: str, dest: Path) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{path}", tmp)
            tmp.rename(dest)
            return dest
        except Exception as e:  # noqa: BLE001 (network: retry)
            print(f"  download {path} failed ({e}), retry {attempt + 1}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not download {path}")


_local = threading.local()


def fetch_blob(blob_id: str):
    """The file's text, or None (missing / not text)."""
    import requests

    s = getattr(_local, "session", None)
    if s is None:
        s = _local.session = requests.Session()
    for attempt in range(4):
        try:
            r = s.get(BLOB_URL.format(blob_id), timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            raw = gzip.decompress(r.content)
            if b"\0" in raw:
                return None
            return raw.decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001 (network: retry)
            time.sleep(1 + 2 * attempt)
    return None


def candidates(table, need_bytes: float, rng):
    """blob ids of 200 B..50 KB files, the highest score threshold holding 3x need_bytes, shuffled."""
    blob = np.asarray(table.column("blob_id").to_pylist(), dtype=object)
    size = table.column("length_bytes").to_numpy()
    score = table.column("score").to_numpy()
    ok = (size >= MIN_BYTES) & (size <= MAX_BYTES)
    for th in (4.5, 4.0, 3.5, 3.0, 2.5, -1e9):
        sel = ok & (score >= th)
        if size[sel].sum() >= 3 * need_bytes:
            break
    idx = np.flatnonzero(sel)
    rng.shuffle(idx)
    return blob[idx].tolist(), th, int(sel.sum()), float(size[sel].sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=None, help="train tokens (default: 25/70 of cosmopedia-pretrain)")
    p.add_argument("--name", default="stack-edu-bpe1024")
    p.add_argument("--out", default=None, help="output directory (default datasets/<name>)")
    p.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES)
    p.add_argument("--validation-docs", type=int, default=400, help="per language")
    p.add_argument("--workers", type=int, default=128, help="parallel blob downloads")
    p.add_argument("--threads", type=int, default=16, help="tokenizer threads")
    p.add_argument("--samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(a.threads)
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ROOT / "tokenizers" / TOKENIZER / "tokenizer.json"))
    assert tok.token_to_id("<bos>") == 0 and tok.token_to_id("<eos>") == 1 and tok.get_vocab_size() == 1024
    if a.budget is None:
        cosmo = json.loads((ROOT / "datasets" / "cosmopedia-pretrain-bpe1024" / "metadata.json").read_text())
        a.budget = cosmo["train"]["tokens"] * 25 / 70
    share = a.budget / len(a.languages)
    rng = np.random.default_rng(a.seed)
    raw = ROOT / "datasets" / "raw" / "stack-edu"
    shards = [(lang, shard_of(lang, rng)) for lang in a.languages]
    print(f"{a.budget / 1e9:.3f}B train tokens, {share / 1e6:.1f}M per language; shards: "
          + ", ".join(Path(s).as_posix() for _, s in shards), flush=True)
    nxt = [None]

    def prefetch(i):
        nxt[0] = download_shard(shards[i][1], raw / shards[i][1]) if i < len(shards) else None

    pool = ThreadPoolExecutor(a.workers)
    if a.samples:
        prefetch(0)
        t = pq.read_table(nxt[0], columns=["blob_id", "length_bytes", "score"])
        ids, th, n, _ = candidates(t, 1e6, rng)
        for text in pool.map(fetch_blob, ids[:a.samples]):
            print(f"<bos>{text}<eos>\n{'=' * 80}")
        nxt[0].unlink()
        return
    out = Path(a.out) if a.out else ROOT / "datasets" / a.name
    out.mkdir(parents=True, exist_ok=True)
    writers = {sp: NpyWriter(out / f"{sp}.npy") for sp in ("train", "validation")}
    index = {sp: [] for sp in writers}
    stats = {sp: {c: {"documents": 0, "tokens": 0} for c in a.languages} for sp in writers}
    chars, utf8 = [0, 0], {sp: 0 for sp in writers}
    plan, skipped = {}, {"missing": 0, "tags": 0, "empty": 0}
    cat = {c: i for i, c in enumerate(a.languages)}
    bytes_per_token = 2.2  # first guess, then measured
    th_ = threading.Thread(target=prefetch, args=(0,))
    th_.start()
    t0, total = time.time(), 0
    for i, (lang, shard) in enumerate(shards):
        th_.join()
        path = nxt[0]
        th_ = threading.Thread(target=prefetch, args=(i + 1,))
        th_.start()
        table = pq.read_table(path, columns=["blob_id", "length_bytes", "score"])
        path.unlink()
        ids, thresh, n, nbytes = candidates(table, share * bytes_per_token, rng)
        del table
        plan[lang] = {"shard": shard, "score_threshold": thresh, "candidates": n, "candidate_bytes": nbytes}
        st_tr, st_val = stats["train"][lang], stats["validation"][lang]
        chunks = (ids[c:c + CHUNK] for c in range(0, len(ids), CHUNK))
        submit = lambda b: [pool.submit(fetch_blob, x) for x in b]
        pending = [submit(next(chunks, []))]
        while pending and st_tr["tokens"] < share:
            futs = pending.pop(0)
            nb = next(chunks, None)  # the next chunk downloads while this one is tokenized
            if nb:
                pending.append(submit(nb))
            texts = [f.result() for f in futs]
            docs = []
            for t in texts:
                if t is None:
                    skipped["missing"] += 1
                    continue
                t = t.strip()
                if not t:
                    skipped["empty"] += 1
                elif any(tag in t for tag in TAGS):
                    skipped["tags"] += 1
                else:
                    docs.append(t)
            for d, e in zip(docs, tok.encode_batch([f"<bos>{d}<eos>" for d in docs])):
                sp = "validation" if st_val["documents"] < a.validation_docs else "train"
                if sp == "train" and st_tr["tokens"] >= share:
                    break
                x = np.asarray(e.ids, dtype=np.uint16)
                index[sp].append((writers[sp].write(x), len(x), cat[lang]))
                stats[sp][lang]["documents"] += 1
                stats[sp][lang]["tokens"] += len(x)
                b = len(d.encode("utf-8"))
                utf8[sp] += b + 10
                if sp == "train":
                    chars[0] += len(d)
                    chars[1] += len(x) - 2
                    bytes_per_token = utf8["train"] / max(1, writers["train"].n)
                total += len(x)
        for futs in pending:
            for f in futs:
                f.cancel()
        el = time.time() - t0
        print(f"[{i + 1}/{len(shards)}] {lang}: score >= {thresh} ({n} candidates), {st_tr['documents']} docs, "
              f"{st_tr['tokens'] / 1e6:.1f}/{share / 1e6:.1f}M tokens, {bytes_per_token:.2f} bytes/token, "
              f"{total / 1e9:.3f}B total, {total / el / 1e6:.2f}M tokens/s, {el / 60:.1f} min, skipped {skipped}",
              flush=True)
        if st_tr["tokens"] < share:
            print(f"  WARNING: {lang} ran out of candidates", flush=True)
    th_.join()
    if nxt[0] is not None and nxt[0].exists():
        nxt[0].unlink()
    meta = {
        "name": a.name,
        "source": f"{REPO} (rev {REV[:7]}), contents from Software Heritage, {share / 1e6:.0f}M tokens per language "
                  f"(datasets/build_stack_edu.py); tokenizer {TOKENIZER}",
        "categories": {c: {sp: stats[sp][c] for sp in writers} for c in a.languages},
        "category_ids": cat, "category_index": "{split}.categories.npy", "plan": plan, "skipped": skipped,
        "tokenizer": TOKENIZER, "vocabulary_size": 1024,
        "doc_token_id": 0, "bos_token_id": 0, "eos_token_id": 1,
        "avg_chars_per_token": round(chars[0] / max(1, chars[1]), 4), "dtype": "uint16", "seed": a.seed,
        "notes": "Pretraining code: <bos>file<eos>, every token scored. Token = tag (ids 0-15), byte (16 + byte) or merge.",
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
    print(json.dumps({k: meta[k] for k in ("train", "validation", "avg_chars_per_token", "skipped")}, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    sys.exit(main())
