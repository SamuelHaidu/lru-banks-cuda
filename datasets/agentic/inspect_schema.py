"""Prints each raw source's columns and one row (long strings cut), to write its converter.

    uv run --with pyarrow python agentic/inspect_schema.py /work/raw/terminal-corpus [more dirs]
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def cut(v, n=400):
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + f"... [{len(v)} chars]"
    if isinstance(v, list):
        return [cut(x, n) for x in v[:6]] + ([f"... [{len(v)} items]"] if len(v) > 6 else [])
    if isinstance(v, dict):
        return {k: cut(x, n) for k, x in v.items()}
    return v


for d in sys.argv[1:]:
    files = sorted(Path(d).rglob("*.parquet")) + sorted(Path(d).rglob("*.jsonl"))
    print(f"===== {d}: {len(files)} files")
    if not files:
        continue
    f = files[0]
    print("file:", f)
    if f.suffix == ".parquet":
        t = pq.ParquetFile(f)
        print("schema:", t.schema_arrow)
        print("rows in file:", t.metadata.num_rows)
        row = next(t.iter_batches(batch_size=1)).to_pylist()[0]
    else:
        with open(f) as fh:
            row = json.loads(fh.readline())
    print(json.dumps(cut(row), indent=1, ensure_ascii=False)[:6000])
