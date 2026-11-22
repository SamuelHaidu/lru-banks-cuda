"""Cosmopedia + agent-mix-v2 in one dataset (`cosmo-agent-mix-bpe1024`), in place, for a small disk.

Takes the output directory of `build_cosmopedia.py` (also an interrupted build: its .npy headers are still empty and
the last document may be cut; both are fixed here: the files end after their last complete document), renames it and
appends agent-mix-v2 (same tokenizer, uint16): its train split `--repeat` times (agent-mix-v2 is 50M tokens: 30
copies are ~25% of 4.4B cosmopedia tokens), its validation split once. Categories: agent-mix-v2's (ids 0-5) and
`cosmopedia` (6, the subsets are not told apart).

    python datasets/build_cosmo_agent_mix.py [--cosmo cosmopedia-small-bpe1024] [--repeat 30]
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HEADER = 128  # build_cosmopedia's reserved header
EOS, BOS = 1, 0


def write_header(path: Path, n: int) -> None:
    d = "{'descr': '<u2', 'fortran_order': False, 'shape': (%d,), }" % n
    d = d.ljust(HEADER - 10 - 1) + "\n"
    with open(path, "r+b") as f:
        f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(d)) + d.encode("latin1"))


def complete_length(path: Path) -> int:
    """Tokens up to and including the last <eos> (an interrupted build may end mid-document)."""
    n = (os.path.getsize(path) - HEADER) // 2
    raw = np.memmap(path, dtype="<u2", mode="r", offset=HEADER, shape=(n,))
    step = 1 << 22
    for end in range(n, 0, -step):
        hits = np.flatnonzero(raw[max(0, end - step):end] == EOS)
        if hits.size:
            return max(0, end - step) + int(hits[-1]) + 1
    raise ValueError(f"no <eos> in {path}")


def doc_starts(path: Path, n: int) -> np.ndarray:
    raw = np.memmap(path, dtype="<u2", mode="r", offset=HEADER, shape=(n,))
    parts = [np.flatnonzero(raw[a:a + 50_000_000] == BOS).astype(np.int64) + a for a in range(0, n, 50_000_000)]
    return np.concatenate(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cosmo", default="cosmopedia-small-bpe1024")
    p.add_argument("--agent", default="agent-mix-v2-bpe1024")
    p.add_argument("--name", default="cosmo-agent-mix-bpe1024")
    p.add_argument("--repeat", type=int, default=30)
    a = p.parse_args()
    src, out = ROOT / "datasets" / a.cosmo, ROOT / "datasets" / a.name
    agent_meta = json.loads((ROOT / "datasets" / a.agent / "metadata.json").read_text())
    if out.exists():
        raise SystemExit(f"{out} exists")
    cosmo_meta = json.loads((src / "metadata.json").read_text()) if (src / "metadata.json").exists() else {}
    os.rename(src, out)
    meta = {"name": a.name, "categories": {}, "category_ids": dict(agent_meta["category_ids"])}
    cos_id = len(meta["category_ids"])
    meta["category_ids"]["cosmopedia"] = cos_id
    for split, repeat in (("train", a.repeat), ("validation", 1)):
        path = out / f"{split}.npy"
        n = complete_length(path)
        starts = doc_starts(path, n)
        lens = np.diff(np.append(starts, n))
        rows = [np.stack([starts, lens, np.full(len(starts), cos_id)], 1)]
        meta["categories"].setdefault("cosmopedia", {})[split] = {"documents": int(len(starts)), "tokens": int(n)}
        agent = np.load(ROOT / "datasets" / a.agent / f"{split}.npy", mmap_mode="r")
        assert int(agent.max()) < 1024
        agent16 = np.asarray(agent).astype("<u2")
        acat = np.load(ROOT / "datasets" / a.agent / f"{split}.categories.npy")
        with open(path, "r+b") as f:
            f.truncate(HEADER + 2 * n)
            f.seek(HEADER + 2 * n)
            for r in range(repeat):
                f.write(agent16.tobytes())
                rows.append(acat + np.array([n + r * len(agent16), 0, 0]))
        total = n + repeat * len(agent16)
        write_header(path, total)
        for k, v in agent_meta["categories"].items():
            meta["categories"].setdefault(k, {})[split] = {kk: vv * repeat for kk, vv in v[split].items()}
        idx = np.concatenate(rows).astype(np.int64)
        np.save(out / f"{split}.categories.npy", idx)
        check = np.load(path, mmap_mode="r")
        assert check.shape == (total,) and check[0] == BOS and check[n] == BOS and check[-1] == EOS
        dl = idx[:, 1]
        meta[split] = {"file": f"{split}.npy", "tokens": int(total), "documents": int(len(idx)),
                       "cosmopedia_tokens": int(n), "agent_tokens": int(total - n), "agent_repeat": repeat,
                       "doc_tokens": {k: int(np.percentile(dl, q)) for k, q in (("p50", 50), ("p90", 90), ("p99", 99))}
                       | {"max": int(dl.max())}}
        for cache in out.glob(f"{split}.npy.*"):
            cache.unlink()
        print(split, meta[split], flush=True)
    meta.update({
        "source": f"{a.cosmo} ({cosmo_meta.get('source', 'HuggingFaceTB/cosmopedia, interrupted build_cosmopedia.py')}) "
                  f"+ {a.agent} (train x{a.repeat}, validation x1) (datasets/build_cosmo_agent_mix.py)",
        "category_index": "{split}.categories.npy",
        "tokenizer": agent_meta["tokenizer"], "vocabulary_size": 1024,
        "doc_token_id": 0, "bos_token_id": 0, "eos_token_id": 1,
        "special_tokens": agent_meta["special_tokens"], "loss_spans": agent_meta["loss_spans"],
        "avg_chars_per_token": cosmo_meta.get("avg_chars_per_token", 2.17), "dtype": "uint16",
        "notes": agent_meta.get("notes", ""),
    })
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("done ->", out)


if __name__ == "__main__":
    sys.exit(main())
