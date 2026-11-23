"""Checks one downloaded corpus chunk against the corpus metadata: sha256 of the .npy.zst, the decompressed .npy's
token count and dtype, the index (documents, starts, lengths), every document framed by <bos> ... <eos>, and prints
the start of one decoded document.

    python agentic/verify_chunk.py --meta metadata.json --source nemotron-opencode --chunk 0 \\
        --zst chunk.npy.zst --index chunk.index.npy --tokenizer tokenizers/agentic-bpe32768/tokenizer.json
"""

import argparse
import hashlib
import io
import json
import subprocess

import numpy as np

ap = argparse.ArgumentParser()
for k in ("meta", "source", "zst", "index", "tokenizer"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--chunk", type=int, default=0)
a = ap.parse_args()

entry = json.load(open(a.meta))["sources"][a.source]["train"][a.chunk]
h = hashlib.sha256()
with open(a.zst, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
assert h.hexdigest() == entry["sha256"], "sha256 differs"
raw = subprocess.run(["zstd", "-dc", "--long=30", a.zst], check=True, capture_output=True).stdout
ids = np.load(io.BytesIO(raw))
idx = np.load(a.index)
assert ids.dtype == np.uint16 and ids.shape == (entry["tokens"],), (ids.dtype, ids.shape)
assert len(idx) == entry["documents"] and idx[-1, 0] + idx[-1, 1] == len(ids)
assert np.all(ids[idx[:, 0]] == 0) and np.all(ids[idx[:, 0] + idx[:, 1] - 1] == 1), "a document is not <bos> ... <eos>"
print(f"ok: sha256, {len(ids):,} tokens, {len(idx):,} documents, every one <bos> ... <eos>")

from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(a.tokenizer)
s, n = idx[len(idx) // 2, :2]
print(tok.decode(ids[s:s + min(n, 300)].tolist(), skip_special_tokens=False)[:1200])
