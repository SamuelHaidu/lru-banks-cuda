"""`agent-mix-nemotron-bpe1024`: `agent-mix-bpe1024` + the converted Nemotron terminal trajectories
(`datasets/convert_nemotron_terminal.py`), encoded with the existing `agent-mix-bpe1024` tokenizer (same ids, so a run
on agent-mix-bpe1024 can continue on it). 200 trajectories held out for validation; train documents shuffled.

    .venv/bin/python datasets/build_agent_mix_nemotron.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
SRC, NAME = "agent-mix-bpe1024", "agent-mix-nemotron-bpe1024"
NEMO = HERE / "raw" / "nemotron-terminal" / "easy_converted.jsonl"
HELD_OUT = 200


def split_docs(ids: np.ndarray) -> list:
    starts = np.flatnonzero(ids == 0).tolist() + [ids.size]
    return [ids[a:b] for a, b in zip(starts[:-1], starts[1:])]


def main() -> None:
    meta = json.loads((HERE / SRC / "metadata.json").read_text())
    tok = Tokenizer.from_file(str(HERE.parent / "tokenizers" / meta["tokenizer"] / "tokenizer.json"))
    with open(NEMO) as f:
        texts = [json.loads(line)["text"] for line in f]
    nemo = []
    for i in range(0, len(texts), 1000):  # in chunks: the ids as int32 arrays, not Python lists
        nemo += [np.asarray(e.ids, dtype=np.int32) for e in tok.encode_batch(texts[i:i + 1000])]
    assert tok.decode(nemo[0].tolist(), skip_special_tokens=False) == texts[0]
    assert all(d[0] == 0 and d[-1] == 1 for d in nemo)  # <bos> ... <eos>
    rng = np.random.default_rng(0)
    order = rng.permutation(len(nemo))
    nemo_val, nemo_train = [nemo[i] for i in order[:HELD_OUT]], [nemo[i] for i in order[HELD_OUT:]]

    out = HERE / NAME
    out.mkdir(exist_ok=True)
    q = lambda xs: {k: int(np.percentile(xs, v)) for k, v in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))}
    splits = {}
    for split, extra in (("train", nemo_train), ("validation", nemo_val)):
        docs = split_docs(np.load(HERE / SRC / f"{split}.npy")) + extra
        docs = [docs[i] for i in rng.permutation(len(docs))]
        arr = np.concatenate(docs)
        np.save(out / f"{split}.npy", arr)
        splits[split] = {"file": f"{split}.npy", "tokens": int(arr.size), "documents": len(docs),
                         "doc_tokens": q([len(d) for d in docs]),
                         "nemotron_documents": len(extra), "nemotron_tokens": int(sum(len(d) for d in extra))}
        print(split, splits[split])
    new = {**meta, "name": NAME, **splits,
           "source": meta["source"] + " + nvidia/Nemotron-Terminal-Corpus skill_based_easy (rev 67a3e03, converted by "
                     "datasets/convert_nemotron_terminal.py), same tokenizer (datasets/build_agent_mix_nemotron.py)"}
    new.pop("sources", None)
    (out / "metadata.json").write_text(json.dumps(new, indent=2) + "\n")
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
