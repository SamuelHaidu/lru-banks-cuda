"""HuggingFaceTB/finemath (rev e92b25a, config `finemath-4plus`: Common Crawl math pages kept by a classifier at score
>= 4, 6.7M documents) for pretraining (`finemath-bpe1024`): each page as

    <bos>{text}<eos>

every token scored, tokenized with the agent-mix-v2 BPE (concatenates with `cosmopedia-pretrain-bpe1024` and
`stack-edu-bpe1024`: `datasets/build_pretrain_mix.py`). Train tokens `--budget` (default 5/70 of cosmopedia-pretrain's:
5% of the 70 / 25 / 5 mix). Parquet shards in random order (each one is a random slice of the crawl), downloaded one
at a time (the next while the current one is tokenized) and deleted, rows shuffled inside a shard, until the budget is
reached; the first `--validation-docs` documents go to validation. Rows with one of our tag strings are skipped.

    uv run --with pyarrow python datasets/build_finemath.py [--budget 2.1e8] [--out DIR] [--samples 2]
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cosmopedia import TAGS, NpyWriter, api  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO, REV, CONFIG = "HuggingFaceTB/finemath", "e92b25a616738fe95dc186b64dfb19f9c8525594", "finemath-4plus"
TOKENIZER = "agent-mix-v2-bpe1024"
BATCH = 2000


def download(path: str, dest: Path) -> Path:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=None, help="train tokens (default: 5/70 of cosmopedia-pretrain)")
    p.add_argument("--name", default="finemath-bpe1024")
    p.add_argument("--out", default=None)
    p.add_argument("--validation-docs", type=int, default=1000)
    p.add_argument("--threads", type=int, default=16)
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
        a.budget = cosmo["train"]["tokens"] * 5 / 70
    rng = np.random.default_rng(a.seed)
    tree = api(f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}/{CONFIG}")
    files = sorted(f["path"] for f in tree if f["path"].endswith(".parquet"))
    files = [files[i] for i in rng.permutation(len(files))]
    raw = ROOT / "datasets" / "raw" / "finemath"
    if a.samples:
        t = pq.read_table(download(files[0], raw / Path(files[0]).name), columns=["text", "url"])
        for r in t.slice(0, a.samples).to_pylist():
            print(f"[{r['url']}]\n<bos>{r['text'].strip()}<eos>\n{'=' * 80}")
        (raw / Path(files[0]).name).unlink()
        return
    print(f"{a.budget / 1e6:.1f}M train tokens from {CONFIG} ({len(files)} shards)", flush=True)
    out = Path(a.out) if a.out else ROOT / "datasets" / a.name
    out.mkdir(parents=True, exist_ok=True)
    writers = {sp: NpyWriter(out / f"{sp}.npy") for sp in ("train", "validation")}
    index = {sp: [] for sp in writers}
    stats = {sp: {"documents": 0, "tokens": 0} for sp in writers}
    chars, utf8, dropped = [0, 0], {sp: 0 for sp in writers}, 0
    nxt = [None]

    def prefetch(i):
        nxt[0] = download(files[i], raw / Path(files[i]).name) if i < len(files) else None

    th = threading.Thread(target=prefetch, args=(0,))
    th.start()
    t0 = time.time()
    for i, f in enumerate(files):
        th.join()
        path = nxt[0]
        if stats["train"]["tokens"] >= a.budget:
            path.unlink()
            break
        th = threading.Thread(target=prefetch, args=(i + 1,))
        th.start()
        texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
        path.unlink()
        docs = []
        for j in rng.permutation(len(texts)):
            t = (texts[j] or "").strip()
            if not t or any(tag in t for tag in TAGS):
                dropped += 1
                continue
            docs.append(t)
        for b in range(0, len(docs), BATCH):
            if stats["train"]["tokens"] >= a.budget:
                break
            for d, e in zip(docs[b:b + BATCH], tok.encode_batch([f"<bos>{d}<eos>" for d in docs[b:b + BATCH]])):
                sp = "validation" if stats["validation"]["documents"] < a.validation_docs else "train"
                if sp == "train" and stats["train"]["tokens"] >= a.budget:
                    break
                x = np.asarray(e.ids, dtype=np.uint16)
                index[sp].append((writers[sp].write(x), len(x), 0))
                stats[sp]["documents"] += 1
                stats[sp]["tokens"] += len(x)
                utf8[sp] += len(d.encode("utf-8")) + 10
                if sp == "train":
                    chars[0] += len(d)
                    chars[1] += len(x) - 2
        el = time.time() - t0
        print(f"[{i + 1}] {Path(f).name}: {stats['train']['tokens'] / 1e6:.1f}/{a.budget / 1e6:.1f}M tokens, "
              f"{el / 60:.1f} min", flush=True)
    th.join()
    if nxt[0] is not None and nxt[0].exists():
        nxt[0].unlink()
    meta = {
        "name": a.name,
        "source": f"{REPO} (rev {REV[:7]}), {CONFIG}, {a.budget / 1e6:.0f}M tokens (datasets/build_finemath.py); "
                  f"tokenizer {TOKENIZER}",
        "categories": {"finemath": {sp: stats[sp] for sp in writers}}, "category_ids": {"finemath": 0},
        "category_index": "{split}.categories.npy",
        "tokenizer": TOKENIZER, "vocabulary_size": 1024, "doc_token_id": 0, "bos_token_id": 0, "eos_token_id": 1,
        "avg_chars_per_token": round(chars[0] / max(1, chars[1]), 4), "dtype": "uint16", "seed": a.seed,
        "dropped_documents": dropped,
        "notes": "Pretraining math pages: <bos>text<eos>, every token scored.",
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
    print(json.dumps({k: meta[k] for k in ("train", "validation", "avg_chars_per_token", "dropped_documents")}, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    sys.exit(main())
