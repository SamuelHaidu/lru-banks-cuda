"""Encodes decoded text (`datasets/decode_to_text.py` output) with another tokenizer into a pretraining dataset, e.g.
the 70 / 25 / 5 mix with the 32k BPE:

    uv run python datasets/encode_text.py --text datasets/raw/cosmo-code-math-pretrain-bpe1024-text \\
        --tokenizer pretrain-mix-bpe32768 --name pretrain-mix-bpe32768 [--out DIR] [--cpu-fraction 0.8]

Each document becomes `<bos>` + text + `<eos>` (the special tokens by name, as the source datasets were built and as
`decode_to_text` checks). Output exactly like `datasets/build_pretrain_mix.py`: `{split}.npy` (uint16 ids, documents
in the source order), `{split}.categories.npy` (int64 start, length, category per document) and `metadata.json`
(the source's `category_ids`; per category documents / tokens; `avg_chars_per_token` = characters / tokens, so bpc
compares across tokenizers). Workers: `--cpu-fraction` of the container's CPU quota, one tokenizer thread each, one
text shard per job; results come back in order and are appended to the output file directly (no temporary files:
the pod's container disk is small). Checks: a sample of validation documents decodes back to the source text byte
for byte; every document starts with <bos> and ends with <eos>.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cosmopedia import NpyWriter  # noqa: E402
from decode_to_text import container_cpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BATCH = 1000


def work(job):
    """Encodes one text shard: (ids uint16, document lengths, categories, characters)."""
    os.environ["RAYON_NUM_THREADS"] = "1"
    from tokenizers import Tokenizer

    tok_path, base = job
    tok = Tokenizer.from_file(tok_path)
    blob = Path(base + ".txt").read_bytes()
    offs = np.load(base + ".offsets.npy")
    cats = np.load(base + ".categories.npy")
    pieces, lens, chars = [], [], 0
    for s in range(0, len(offs) - 1, BATCH):
        docs = [blob[offs[i]:offs[i + 1]].decode("utf-8") for i in range(s, min(s + BATCH, len(offs) - 1))]
        chars += sum(len(d) for d in docs)
        for enc in tok.encode_batch(["<bos>" + d + "<eos>" for d in docs], add_special_tokens=False):
            ids = np.asarray(enc.ids, dtype=np.uint32)
            pieces.append(ids)
            lens.append(len(ids))
    ids = np.concatenate(pieces) if pieces else np.zeros(0, np.uint32)
    assert ids.max(initial=0) < 65536
    return ids.astype(np.uint16), np.asarray(lens, np.int64), cats.astype(np.int64), chars


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True, help="a decode_to_text output directory")
    p.add_argument("--tokenizer", required=True, help="a directory name under tokenizers/")
    p.add_argument("--name", required=True)
    p.add_argument("--out", default=None, help="output directory (default datasets/<name>)")
    p.add_argument("--cpu-fraction", type=float, default=0.8)
    p.add_argument("--check", type=int, default=2000, help="validation documents decoded back to check")
    a = p.parse_args()
    src = Path(a.text)
    tmeta = json.loads((src / "metadata.json").read_text())
    tok_dir = ROOT / "tokenizers" / a.tokenizer
    tok_path = str(tok_dir / "tokenizer.json")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    vocab = tok.get_vocab_size()
    assert vocab <= 65536 and bos is not None and eos is not None
    out = Path(a.out) if a.out else ROOT / "datasets" / a.name
    out.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(container_cpus() * a.cpu_fraction))
    print(f"{workers} workers; {a.tokenizer} ({vocab} ids, <bos> {bos}, <eos> {eos}) -> {out}", flush=True)
    names = {v: k for k, v in tmeta["category_ids"].items()}
    meta = {"name": a.name, "tokenizer": a.tokenizer, "vocabulary_size": vocab, "doc_token_id": bos,
            "bos_token_id": bos, "eos_token_id": eos, "dtype": "uint16", "category_index": "{split}.categories.npy",
            "category_ids": tmeta["category_ids"]}
    cats_meta = {k: {} for k in tmeta["category_ids"]}
    t0, total_chars, total_tokens = time.time(), 0, 0
    splits = [s for s in ("validation", "train") if s in tmeta["splits"]]
    for sp in splits:
        shards = tmeta["splits"][sp]["shards"]
        jobs = [(tok_path, str(src / Path(s["text"]).stem)) for s in shards]
        w = NpyWriter(out / f"{sp}.npy")
        index, nbytes, nchars = [], 0, 0
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            for k, (ids, lens, cats, chars) in enumerate(pool.imap(work, jobs, chunksize=1)):
                start = w.write(ids)
                starts = start + np.concatenate([[0], np.cumsum(lens)[:-1]])
                index.append(np.stack([starts, lens, cats], 1))
                nbytes += shards[k]["bytes"]
                nchars += chars
                print(f"  {sp} shard {k + 1}/{len(jobs)}: {w.n / 1e9:.3f}B tokens, {time.time() - t0:.0f} s", flush=True)
        w.close()
        idx = np.concatenate(index)
        np.save(out / f"{sp}.categories.npy", idx)
        assert len(idx) == tmeta["splits"][sp]["documents"], (len(idx), tmeta["splits"][sp]["documents"])
        for c in np.unique(idx[:, 2]):
            sel = idx[:, 2] == c
            cats_meta[names[int(c)]][sp] = {"documents": int(sel.sum()), "tokens": int(idx[sel, 1].sum())}
        lens = idx[:, 1]
        meta[sp] = {"file": f"{sp}.npy", "tokens": int(w.n), "documents": int(len(idx)),
                    "doc_tokens": {k: int(np.percentile(lens, q)) for k, q in (("p50", 50), ("p90", 90), ("p99", 99))}
                    | {"max": int(lens.max())}, "bytes_per_token": round(nbytes / w.n, 4),
                    "chars_per_token": round(nchars / w.n, 4)}
        if sp == "train" or len(splits) == 1:
            total_chars, total_tokens = nchars, w.n
        print(f"{sp}: {w.n / 1e9:.3f}B tokens, {len(idx)} documents, {nbytes / w.n:.3f} bytes / "
              f"{nchars / w.n:.3f} chars per token, {time.time() - t0:.0f} s", flush=True)
    meta["avg_chars_per_token"] = round(total_chars / total_tokens, 4)
    meta["categories"] = cats_meta
    meta["source"] = f"{tmeta['source']} decoded to text ({src.name}) and re-encoded with {a.tokenizer} " \
                     "(datasets/encode_text.py)"
    meta["notes"] = "Pretraining text and code: <bos>document<eos>, every token scored."
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    # checks: document boundaries, and a sample of validation documents decodes back to the source text
    for sp in splits:
        tokens = np.load(out / f"{sp}.npy", mmap_mode="r")
        idx = np.load(out / f"{sp}.categories.npy")
        pick = np.random.default_rng(0).choice(len(idx), min(5000, len(idx)), replace=False)
        assert all(tokens[idx[i, 0]] == bos and tokens[idx[i, 0] + idx[i, 1] - 1] == eos for i in pick), sp
        assert idx[-1, 0] + idx[-1, 1] == len(tokens)
    tokens = np.load(out / "validation.npy", mmap_mode="r")
    idx = np.load(out / "validation.categories.npy")
    sh = tmeta["splits"]["validation"]["shards"][0]
    offs = np.load(src / sh["offsets"])
    blob = (src / sh["text"]).read_bytes()
    n = min(a.check, sh["documents"])
    ok = sum(tok.decode(np.asarray(tokens[idx[i, 0] + 1:idx[i, 0] + idx[i, 1] - 1]).tolist(),
                        skip_special_tokens=False).encode("utf-8") == blob[offs[i]:offs[i + 1]] for i in range(n))
    assert ok == n, f"only {ok}/{n} validation documents decode back to their text"
    print(f"checked: {ok}/{n} validation documents decode back byte for byte; boundaries ok. "
          f"{meta['avg_chars_per_token']} chars per token; done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    sys.exit(main())
