"""Instruction data: smol-smoltalk (HuggingFaceTB/smol-smoltalk, Apache 2.0; SmolLM2's SFT mix for small models) as
chat documents in our tags, plus optional extra documents from a jsonl (e.g. a model's own Calc-X solutions):

    <bos>[<system>...</system>]<user>...</user><assistant>...</assistant>...<eos>

Every token is scored (no `loss_spans`: user turns are language data too); `assistant_tokens` in the metadata
counts the assistant spans for reference. Dropped: conversations that do not alternate user / assistant
(after an optional system message) or do not end with the assistant, any content holding one of our tag strings,
non-Latin script over 0.5% of the letters (the tokenizer is English / code), documents over --max-tokens.

    python3 datasets/build_smoltalk.py --download --raw RAW --out OUT --tokenizer TOKENIZER.json
    python3 datasets/build_smoltalk.py --docs self.jsonl --name calcx-self-bpe32768 --out OUT2 --tokenizer ...
      (--docs: {"source", "text"} rows whose text is already in our tags; 2% held out as validation)

Output: train.npy / validation.npy (uint16), {split}.categories.npy (start, length, source id), metadata.json.
"""
import argparse
import json
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = "HuggingFaceTB/smol-smoltalk"
FILES = {"train": [f"data/train-0000{i}-of-00004.parquet" for i in range(4)],
         "validation": ["data/test-00000-of-00001.parquet"]}
TAGS = ["<bos>", "<eos>", "<tools>", "</tools>", "<user>", "</user>", "<assistant>", "</assistant>", "<call>",
        "</call>", "<result>", "</result>", "<system>", "</system>", "<think>", "</think>"]


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}"))
    rev = info["sha"]
    for fs in FILES.values():
        for f in fs:
            dst = raw / Path(f).name
            if not dst.exists():
                urllib.request.urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/{rev}/{f}", dst)
                print(f"downloaded {f} ({dst.stat().st_size / 1e6:.0f} MB)", flush=True)
    (raw / "revision.json").write_text(json.dumps({"repo": REPO, "sha": rev}))


def latin(text: str) -> bool:
    letters = other = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            o = ord(ch)
            other += not (o < 0x250 or 0x1E00 <= o <= 0x1EFF)
    return other <= 0.005 * max(letters, 1)


def convert(messages):
    """messages -> text in our tags, or (None, reason)."""
    out, want = [], "user"
    for i, m in enumerate(messages):
        role, c = m["role"], m["content"]
        if any(t in c for t in TAGS):
            return None, "tag text"
        if role == "system" and i == 0:
            out.append(f"<system>{c}</system>")
            continue
        if role != want:
            return None, "order"
        out.append(f"<{role}>{c}</{role}>")
        want = "assistant" if role == "user" else "user"
    if want != "user":
        return None, "no final answer"
    text = "".join(out)
    if not latin(text):
        return None, "non-Latin"
    return text, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="datasets/raw/smol-smoltalk")
    p.add_argument("--download", action="store_true")
    p.add_argument("--docs", default=None, help="jsonl of ready documents instead of smol-smoltalk")
    p.add_argument("--name", default="smol-smoltalk-bpe32768")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", required=True, help="tokenizer.json (pretrain-mix-bpe32768)")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(a.threads)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)
    ids = {t: tok.token_to_id(t) for t in TAGS}
    assert all(v is not None for v in ids.values())
    bos, eos, a_open, a_close = ids["<bos>"], ids["<eos>"], ids["<assistant>"], ids["</assistant>"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = {}
    drops = {}
    if a.docs:
        # generated text can hold invalid UTF-8 (a token cut inside a character): such documents are dropped
        rows = [json.loads(l) for l in open(a.docs, encoding="utf-8", errors="replace")]
        n0 = len(rows)
        rows = [r for r in rows if "\ufffd" not in r["text"]]
        drops["invalid UTF-8"] = n0 - len(rows)
        random.Random(0).shuffle(rows)
        nv = max(1, len(rows) // 50)
        splits = {"validation": [(r["source"], r["text"]) for r in rows[:nv]],
                  "train": [(r["source"], r["text"]) for r in rows[nv:]]}
        source = {"docs": a.docs}
    else:
        raw = Path(a.raw)
        if a.download:
            download(raw)
        import pyarrow.parquet as pq

        for sp, fs in FILES.items():
            docs = []
            for f in fs:
                for row in pq.read_table(raw / Path(f).name, columns=["messages", "source"]).to_pylist():
                    text, why = convert(row["messages"])
                    if text is None:
                        drops[why] = drops.get(why, 0) + 1
                    else:
                        docs.append((row["source"], text))
            splits[sp] = docs
        source = json.loads((Path(a.raw) / "revision.json").read_text())

    cat_ids = {s: i for i, s in enumerate(sorted({d[0] for docs in splits.values() for d in docs}))}
    meta = {"name": a.name, "tokenizer": "pretrain-mix-bpe32768", "vocabulary_size": tok.get_vocab_size(),
            "doc_token_id": bos, "bos_token_id": bos, "eos_token_id": eos, "dtype": "uint16",
            "category_index": "{split}.categories.npy", "category_ids": cat_ids,
            "chat": True, "source": source}
    t0 = time.time()
    cats = {}
    for sp in ("train", "validation"):
        docs = splits[sp]
        encs = tok.encode_batch(["<bos>" + d[1] + "<eos>" for d in docs], add_special_tokens=False)
        keep = [i for i, e in enumerate(encs) if len(e.ids) <= a.max_tokens]
        drops[f"{sp}: over {a.max_tokens} tokens"] = len(docs) - len(keep)
        bad = sum(tok.decode(encs[i].ids[1:-1], skip_special_tokens=False) != docs[i][1] for i in keep)
        assert bad == 0, f"{sp}: {bad} documents do not decode back"
        lens = np.array([len(encs[i].ids) for i in keep], np.int64)
        tokens = np.concatenate([np.asarray(encs[i].ids, np.uint32) for i in keep])
        assert tokens.max() < 65536
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        np.save(out / f"{sp}.npy", tokens.astype(np.uint16))
        np.save(out / f"{sp}.categories.npy", np.stack([starts, lens, [cat_ids[docs[i][0]] for i in keep]], 1))
        o, c = np.flatnonzero(tokens == a_open), np.flatnonzero(tokens == a_close)
        assert len(o) == len(c) and (c > o).all() and (o[1:] > c[:-1]).all(), "unbalanced assistant tags"
        scored = int((c - o).sum())  # the targets inside the spans (up to and including </assistant>)
        for i in keep:
            c = cats.setdefault(docs[i][0], {}).setdefault(sp, {"documents": 0, "tokens": 0})
            c["documents"] += 1
            c["tokens"] += len(encs[i].ids)
        meta[sp] = {"file": f"{sp}.npy", "tokens": int(len(tokens)), "documents": len(keep), "assistant_tokens": scored,
                    "doc_tokens": {k: int(np.percentile(lens, q)) for k, q in (("p50", 50), ("p90", 90), ("p99", 99))}
                    | {"max": int(lens.max())}}
        print(f"{sp}: {len(keep):,} documents, {len(tokens):,} tokens ({scored:,} scored), "
              f"{time.time() - t0:.0f} s", flush=True)
    meta["categories"] = cats
    meta["dropped"] = drops
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("dropped:", drops)


if __name__ == "__main__":
    main()
