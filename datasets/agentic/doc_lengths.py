"""Document length distribution per source of a corpus in R2 (from the chunks' index files: exact, no token data
read): percentiles, a log2 histogram, and the share of tokens in documents longer than each context length.

    uv run --no-project --with numpy --with boto3 python agentic/doc_lengths.py [--corpus agentic-v1]
"""

import argparse
import io
import json
import os

import boto3
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="agentic-v1")
a = ap.parse_args()
s3 = boto3.client("s3", endpoint_url=os.environ["REMOTE_POD_S3_ENDPOINT"],
                  aws_access_key_id=os.environ["REMOTE_POD_S3_ACCESS_KEY"],
                  aws_secret_access_key=os.environ["REMOTE_POD_S3_SECRET_KEY"],
                  region_name=os.environ.get("REMOTE_POD_S3_REGION") or "auto")
bucket, prefix = os.environ["REMOTE_POD_S3_BUCKET"], f"corpora/{a.corpus}"
meta = json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}/metadata.json")["Body"].read())

edges = [2 ** k for k in range(6, 18)]  # 64 .. 131072
print(f"{'source':18s} {'docs':>9s} {'mean':>7s} {'p10':>6s} {'p50':>6s} {'p90':>7s} {'p99':>7s} {'max':>8s}  "
      "share of tokens in docs longer than: 512 / 1024 / 2048 / 4096 / 8192")
hists = {}
for src, s in meta["sources"].items():
    lens = np.concatenate([np.load(io.BytesIO(s3.get_object(Bucket=bucket, Key=f"{prefix}/{c['index']}")["Body"].read()))[:, 1]
                           for c in s["train"]])
    p = np.percentile(lens, [10, 50, 90, 99]).astype(int)
    tot = lens.sum()
    share = [lens[lens > L].sum() / tot for L in (512, 1024, 2048, 4096, 8192)]
    print(f"{src:18s} {len(lens):9,d} {lens.mean():7.0f} {p[0]:6d} {p[1]:6d} {p[2]:7d} {p[3]:7d} {lens.max():8d}  "
          + " / ".join(f"{x:4.0%}" for x in share))
    hists[src] = np.histogram(lens, bins=[0] + edges + [1 << 40])[0] / len(lens)

print("\ndocuments per length bin (share of each source's documents)")
labels = ["<=64"] + [f"<={e // 1024}k" if e >= 1024 else f"<={e}" for e in edges[1:]] + [">128k"]
print(f"{'':18s} " + " ".join(f"{x:>6s}" for x in labels))
for src, h in hists.items():
    print(f"{src:18s} " + " ".join(f"{x:6.1%}" if x >= 0.0005 else f"{'.':>6s}" for x in h))
