"""HuggingFaceTB/cosmopedia (rev 0ae6ec6, synthetic textbooks / tutorials written by Mixtral from a prompt) in the
agent format, tokenized with the agent-mix-v2 BPE (the same 1024 ids, so agent-mix-v2 runs and their embeddings
train on it as is):

    <bos><user>{prompt}</user><assistant>{text}</assistant><eos>

The loss covers the assistant span (`loss_spans`, as in agent-mix-v2): the model learns to write the text from the
prompt. Rows whose prompt or text contain one of our tag strings are dropped. Subsets (`--subsets`, default the five
small ones, 3.3M documents, ~7.5B tokens): auto_math_text, khanacademy, openstax, stanford, wikihow (also: stories,
web_samples_v1, web_samples_v2: 28M documents, ~80 GB of parquet). One category per subset (`category_ids`,
`{split}.categories.npy`).

Built for a small disk: parquet files are downloaded one at a time (the next one while the current one is
tokenized) and deleted after use; token ids are uint16 (`dtype` in the metadata), streamed into the .npy files.
Validation: a random sample of each subset (`--validation-docs` per subset, default 2000).

    uv run --with pyarrow python datasets/build_cosmopedia.py [--subsets khanacademy ...] [--threads 5]
        [--samples 2]   # print documents of the first file and stop
"""

import argparse
import json
import os
import struct
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO, REV = "HuggingFaceTB/cosmopedia", "0ae6ec63f91742bd2d1eaef4f02232c55d719385"
SMALL = ["auto_math_text", "khanacademy", "openstax", "stanford", "wikihow"]
ALL = SMALL + ["stories", "web_samples_v1", "web_samples_v2"]
ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = "agent-mix-v2-bpe1024"
TAGS = ["<bos>", "<eos>", "<tools>", "</tools>", "<user>", "</user>", "<assistant>", "</assistant>", "<call>",
        "</call>", "<result>", "</result>", "<system>", "</system>", "<think>", "</think>"]
BATCH = 2000  # documents per encode_batch call
HEADER = 128  # bytes reserved for the .npy header (written last, when the length is known)


def api(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def files_of(subset: str):
    tree = api(f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}/data/{subset}")
    return sorted(f["path"] for f in tree if f["path"].endswith(".parquet"))


def download(path: str, dest: Path) -> Path:
    if dest.exists():  # downloaded by an earlier, interrupted build
        return dest
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


class NpyWriter:
    """A 1-D uint16 .npy written incrementally: the header space is reserved, filled in by `close`."""

    def __init__(self, path: Path):
        self.path, self.n = path, 0
        self.f = open(path, "wb")
        self.f.write(b"\0" * HEADER)

    def write(self, a: np.ndarray) -> int:
        start = self.n
        self.f.write(np.ascontiguousarray(a, dtype="<u2").tobytes())
        self.n += len(a)
        return start

    def close(self) -> None:
        d = "{'descr': '<u2', 'fortran_order': False, 'shape': (%d,), }" % self.n
        d = d.ljust(HEADER - 10 - 1) + "\n"
        self.f.seek(0)
        self.f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(d)) + d.encode("latin1"))
        self.f.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subsets", nargs="+", default=SMALL, choices=ALL)
    p.add_argument("--name", default="cosmopedia-small-bpe1024")
    p.add_argument("--validation-docs", type=int, default=2000, help="per subset")
    p.add_argument("--threads", type=int, default=5, help="tokenizer threads")
    p.add_argument("--samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(a.threads)  # before the tokenizer starts its pool
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ROOT / "tokenizers" / TOKENIZER / "tokenizer.json"))
    ids = {t: tok.token_to_id(t) for t in TAGS}
    assert ids["<bos>"] == 0 and ids["<assistant>"] == 6 and ids["</assistant>"] == 7 and tok.get_vocab_size() == 1024
    raw = ROOT / "datasets" / "raw" / "cosmopedia"
    out = ROOT / "datasets" / a.name
    rows = {c["config"]: c["num_rows"] for c in api(f"https://datasets-server.huggingface.co/size?dataset={REPO}")["size"]["configs"]}
    jobs = [(s, f) for s in a.subsets for f in files_of(s)]
    print(f"{len(jobs)} parquet files from {a.subsets}", flush=True)

    def fetch(i):
        s, f = jobs[i]
        return download(f, raw / s / Path(f).name)

    if a.samples:
        t = pq.read_table(fetch(0), columns=["prompt", "text"]).slice(0, a.samples).to_pylist()
        for r in t:
            print(f"<bos><user>{r['prompt'].strip()}</user><assistant>{r['text'].strip()}</assistant><eos>\n{'=' * 80}")
        return
    out.mkdir(parents=True, exist_ok=True)
    writers = {sp: NpyWriter(out / f"{sp}.npy") for sp in ("train", "validation")}
    index = {sp: [] for sp in writers}  # (start, length, category)
    stats = {sp: {s: {"documents": 0, "tokens": 0, "assistant_tokens": 0} for s in a.subsets} for sp in writers}
    chars = {sp: [0, 0] for sp in writers}  # assistant text: characters, tokens
    utf8 = {sp: 0 for sp in writers}
    dropped = 0
    cat = {s: i for i, s in enumerate(a.subsets)}
    rng = np.random.default_rng(a.seed)
    nxt = [None, None]

    def prefetch(i):
        nxt[0] = fetch(i) if i < len(jobs) else None

    th = threading.Thread(target=prefetch, args=(0,))
    th.start()
    t0, total = time.time(), 0
    for i, (subset, f) in enumerate(jobs):
        th.join()
        path = nxt[0]
        th = threading.Thread(target=prefetch, args=(i + 1,))
        th.start()
        table = pq.read_table(path, columns=["prompt", "text"])
        prompts, texts = table.column("prompt").to_pylist(), table.column("text").to_pylist()
        del table
        docs, bodies = [], []
        for pr, tx in zip(prompts, texts):
            pr, tx = (pr or "").strip(), (tx or "").strip()
            if not pr or not tx or any(t in pr or t in tx for t in TAGS):
                dropped += 1
                continue
            docs.append(f"<bos><user>{pr}</user><assistant>{tx}</assistant><eos>")
            bodies.append(tx)
        p_val = min(1.0, a.validation_docs / max(1, rows.get(subset, len(docs))))
        val = rng.random(len(docs)) < p_val
        # in batches: an Encoding keeps offsets and token strings (~100 bytes a token): a whole file at once
        # (~230M tokens) went over the pod's 50 GB
        encs = (e.ids for b in range(0, len(docs), BATCH) for e in tok.encode_batch(docs[b:b + BATCH]))
        for d, body, e, v in zip(docs, bodies, encs, val):
            sp = "validation" if v else "train"
            x = np.asarray(e, dtype=np.uint16)
            start = writers[sp].write(x)
            index[sp].append((start, len(x), cat[subset]))
            at = int(np.flatnonzero(x == ids["<assistant>"])[0])
            n_assist = len(x) - at - 2  # the text and </assistant> (scored), not <eos>
            st = stats[sp][subset]
            st["documents"] += 1
            st["tokens"] += len(x)
            st["assistant_tokens"] += n_assist
            chars[sp][0] += len(body)
            chars[sp][1] += n_assist - 1
            utf8[sp] += len(d.encode("utf-8"))
            total += len(x)
        path.unlink()
        el = time.time() - t0
        print(f"[{i + 1}/{len(jobs)}] {subset} {Path(f).name}: {len(docs)} docs, {total / 1e9:.3f}B tokens so far, "
              f"{total / el / 1e6:.2f}M tokens/s, {el / 60:.1f} min", flush=True)
    th.join()
    meta = {
        "name": a.name,
        "source": f"{REPO} (rev {REV[:7]}), subsets {', '.join(a.subsets)}: prompt -> <user>, text -> <assistant> "
                  f"(datasets/build_cosmopedia.py); tokenizer {TOKENIZER}",
        "categories": {s: {sp: stats[sp][s] for sp in writers} for s in a.subsets},
        "category_ids": cat,
        "category_index": "{split}.categories.npy",
        "tokenizer": TOKENIZER,
        "vocabulary_size": 1024,
        "doc_token_id": 0, "bos_token_id": 0, "eos_token_id": 1,
        "special_tokens": ids,
        "loss_spans": [[ids["<assistant>"], ids["</assistant>"]]],
        "avg_chars_per_token": round(chars["train"][0] / max(1, chars["train"][1]), 4),
        "dtype": "uint16",
        "seed": a.seed,
        "dropped_documents": dropped,
        "notes": "Token = tag (ids 0-15), one UTF-8 byte (id 16 + byte) or a merge (ids 272..1023).",
    }
    for sp, w in writers.items():
        w.close()
        idx = np.asarray(index[sp], dtype=np.int64).reshape(-1, 3)
        np.save(out / f"{sp}.categories.npy", idx)
        lens = idx[:, 1]
        meta[sp] = {"file": f"{sp}.npy", "tokens": int(w.n), "documents": int(len(idx)),
                    "doc_tokens": {k: int(np.percentile(lens, q)) for k, q in (("p50", 50), ("p90", 90), ("p99", 99))}
                    | {"max": int(lens.max())},
                    "bytes_per_token": round(utf8[sp] / max(1, w.n), 4)}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    check = np.load(out / "train.npy", mmap_mode="r")
    assert check.shape == (meta["train"]["tokens"],) and check[0] == 0, check.shape
    print(json.dumps({k: meta[k] for k in ("train", "validation", "avg_chars_per_token", "dropped_documents")}, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    sys.exit(main())
