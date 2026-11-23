"""Builds a dataset directory the engine reads (metadata.json, train.npy, validation.npy) from chunks of a corpus in
R2 (agentic/pack.py): one source, the train chunks you choose (or the first that reach a token budget), its
validation split. The chunks are streamed from R2 through zstd straight into train.npy (one pass, no temporary copy).

    uv run --with numpy --with boto3 python agentic/assemble.py --corpus agentic-v1 --source toucan \\
        --tokens 2e9 --out /data/agentic-toucan          # the first chunks reaching 2B tokens
    ... --chunks 0 3 5                                  # or exact chunks (e.g. a fresh slice for a later run)
    ... --list                                          # the corpus's sources and chunks

R2 credentials: the pod's REMOTE_POD_S3_* environment (remote-pod up sets it).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def client():
    import boto3

    return boto3.client("s3", endpoint_url=os.environ["REMOTE_POD_S3_ENDPOINT"],
                        aws_access_key_id=os.environ["REMOTE_POD_S3_ACCESS_KEY"],
                        aws_secret_access_key=os.environ["REMOTE_POD_S3_SECRET_KEY"],
                        region_name=os.environ.get("REMOTE_POD_S3_REGION") or "auto")


def stream_tokens(s3, bucket, key, out, tokens):
    """Appends the chunk's tokens (the .npy body after its header) to the open file `out`; returns the count."""
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    z = subprocess.Popen(["zstd", "-dc", "--long=30"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def feed():
        for b in iter(lambda: body.read(1 << 22), b""):
            z.stdin.write(b)
        z.stdin.close()

    import threading

    t = threading.Thread(target=feed)
    t.start()
    version = np.lib.format.read_magic(z.stdout)
    read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, _, dtype = read_header(z.stdout)
    assert dtype == np.uint16 and shape == (tokens,), (key, shape, dtype)
    n = 0
    for b in iter(lambda: z.stdout.read(1 << 24), b""):
        out.write(b)
        n += len(b)
    t.join()
    if z.wait() != 0:
        raise RuntimeError(f"zstd failed on {key}")
    assert n == 2 * tokens, (key, n, tokens)
    return tokens


def write_split(s3, bucket, prefix, entries, path):
    total = sum(e["tokens"] for e in entries)
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": "<u2", "fortran_order": False, "shape": (total,)})
        for e in entries:
            stream_tokens(s3, bucket, f"{prefix}/{e['file']}", f, e["tokens"])
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="agentic-v1")
    ap.add_argument("--source")
    ap.add_argument("--chunks", type=int, nargs="+")
    ap.add_argument("--tokens", type=float, help="the first chunks reaching this many tokens")
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    s3, bucket = client(), os.environ["REMOTE_POD_S3_BUCKET"]
    prefix = f"corpora/{a.corpus}"
    meta = json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}/metadata.json")["Body"].read())
    if a.list:
        for name, s in meta["sources"].items():
            print(f"{name:20s} {s['train_tokens'] / 1e9:6.2f}B train tokens, {len(s['train'])} chunks, "
                  f"{s['chars_per_token']:.2f} bytes/token")
        return
    src = meta["sources"][a.source]
    if a.chunks is not None:
        chosen = [src["train"][i] for i in a.chunks]
    else:
        chosen, n = [], 0
        for e in src["train"]:
            if a.tokens and n >= a.tokens:
                break
            chosen.append(e)
            n += e["tokens"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, f"{prefix}/tokenizer.json", str(out / "tokenizer.json"))
    train = write_split(s3, bucket, prefix, chosen, out / "train.npy")
    val = write_split(s3, bucket, prefix, [src["validation"]], out / "validation.npy")
    (out / "metadata.json").write_text(json.dumps({
        "name": out.name, "corpus": a.corpus, "source": a.source, "chunks": [e["file"] for e in chosen],
        "tokenizer": meta["tokenizer"], "vocabulary_size": 32768, "doc_token_id": meta["bos_token_id"],
        "bos_token_id": meta["bos_token_id"], "eos_token_id": meta["eos_token_id"], "dtype": "uint16",
        "avg_chars_per_token": src["chars_per_token"],
        "train": {"file": "train.npy", "tokens": train}, "validation": {"file": "validation.npy", "tokens": val},
    }, indent=1))
    print(f"{out}: {train / 1e9:.2f}B train tokens from {len(chosen)} chunks, {val / 1e6:.1f}M validation")


if __name__ == "__main__":
    main()
