"""A small local dataset of held-out documents for the web chat's catalogs (Inspect documents, Complete starters,
Chat examples) of runs trained on the agentic corpus, whose own validation split is only in R2.

Documents up to --max-tokens long (our conversation tags, re-encoded with the run's tokenizer as <bos>text<eos>), one category per source:
  oswe      the oswe validation prompts + reference assistant turn of an evaluation dump (--oswe, optional)
  toucan    Toucan validation documents (rnn-research's toucan-bpe32768.tar.zst)
  kodcode   KodCode agent validation documents (rnn-research's kodcode-agent-bpe32768)
  calc-x    Calc-X test documents (rnn-research's calc-x-bpe32768)

    uv run python ad_hoc_scripts/build_webchat_samples.py [--per-source 40] [--oswe ~/Downloads/oswe_samples.txt]

Writes datasets/agentic-samples/{metadata.json, validation.npy, validation.categories.npy}; point a run at it with
"dataset": "agentic-samples" in runs/RUN/webchat.json.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent


def jsonl_texts(lines, n: int, rng: np.random.Generator, fits) -> list[str]:
    """n random documents of a docs.jsonl that fit the length limit, in file order."""
    texts = [json.loads(line)["text"] for line in lines if line.strip()]
    picked = []
    for i in rng.permutation(len(texts)):
        if fits(texts[i]):
            picked.append(int(i))
        if len(picked) == n:
            break
    return [texts[i] for i in sorted(picked)]


def oswe_texts(path: Path) -> list[str]:
    """PROMPT (starts with <bos>, ends with <assistant>) + REFERENCE (the assistant turn) of every document."""
    out = []
    for doc in re.split(r"^={20,}\nDOCUMENT ", path.read_text(), flags=re.M)[1:]:
        m = re.search(r"^----- PROMPT[^\n]*\n(.*?)\n+^----- REFERENCE[^\n]*\n(.*?)\n+^----- ", doc, re.S | re.M)
        if m:
            out.append(m.group(1).removeprefix("<bos>") + m.group(2).strip())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rnn-research", type=Path, default=Path.home() / "projects/rnn-research/datasets")
    p.add_argument("--oswe", type=Path, default=Path.home() / "Downloads/oswe_samples.txt")
    p.add_argument("--tokenizer", default="agentic-bpe32768")
    p.add_argument("--per-source", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=1024, help="the web chat shows documents up to 1024 tokens")
    p.add_argument("--out", type=Path, default=ROOT / "datasets/agentic-samples")
    a = p.parse_args()

    rng = np.random.default_rng(0)
    src = a.rnn_research
    tok = Tokenizer.from_file(str(ROOT / "tokenizers" / a.tokenizer / "tokenizer.json"))
    fits = lambda text: len(tok.encode(text, add_special_tokens=False).ids) + 2 <= a.max_tokens  # noqa: E731
    sources: dict[str, list[str]] = {}
    if a.oswe.exists():
        sources["oswe"] = [t for t in oswe_texts(a.oswe) if fits(t)]
    toucan = subprocess.run(["tar", "--zstd", "-xOf", str(src / "toucan-bpe32768.tar.zst"), "--occurrence=1",
                             "toucan-bpe32768/validation.docs.jsonl"], capture_output=True, text=True, check=True)
    sources["toucan"] = jsonl_texts(toucan.stdout.splitlines(), a.per_source, rng, fits)
    with open(src / "kodcode-agent-bpe32768/validation.docs.jsonl") as f:
        sources["kodcode"] = jsonl_texts(f, a.per_source, rng, fits)
    with open(src / "calc-x-bpe32768/test.docs.jsonl") as f:
        sources["calc-x"] = jsonl_texts(f, a.per_source, rng, fits)

    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    ids, index, cats = [], [], {}
    at = 0
    for c, (name, texts) in enumerate(sources.items()):
        cats[name] = c
        for text in texts:
            doc = [bos] + tok.encode(text, add_special_tokens=False).ids + [eos]
            ids.append(np.asarray(doc, np.uint16))
            index.append((at, len(doc), c))
            at += len(doc)
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "validation.npy", np.concatenate(ids))
    np.save(a.out / "validation.categories.npy", np.asarray(index, np.int64))
    lens = {n: [l for _, l, c in index if c == cats[n]] for n in cats}
    meta = {
        "name": a.out.name, "tokenizer": a.tokenizer, "vocabulary_size": tok.get_vocab_size(),
        "doc_token_id": bos, "bos_token_id": bos, "eos_token_id": eos, "dtype": "uint16", "chat": True,
        "category_index": "{split}.categories.npy", "category_ids": cats,
        "validation": {"file": "validation.npy", "tokens": at, "documents": len(index)},
        "categories": {n: {"documents": len(v), "tokens_max": max(v)} for n, v in lens.items()},
        "notes": "Held-out documents for the web chat (ad_hoc_scripts/build_webchat_samples.py), not for training.",
    }
    (a.out / "metadata.json").write_text(json.dumps(meta, indent=1))
    for n, v in lens.items():
        print(f"{n:8s} {len(v):3d} documents, tokens median {int(np.median(v))}, max {max(v)}")
    print(f"wrote {a.out} ({at} tokens)")


if __name__ == "__main__":
    main()
