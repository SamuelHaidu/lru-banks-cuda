"""A byte-level BPE trained on a decoded dataset (`datasets/decode_to_text.py` output), with the agent tokenizers'
fixed ids: the 16 tags (ids 0-15: <bos> <eos> ... <think> </think>), byte b at 16 + b, then the merges in learned
order (`train_tokenizer` of `build_agent_shell_bpe.py`: GPT-2 byte-level pre-tokenization, merges never cross a
word / space-run / punctuation boundary). Trained on a random sample of documents (`--sample-bytes`).

Then checked on every validation document (decode(encode(text)) == text, the tags keep their ids) and compared with
the dataset's current tokenizer: characters per token per category and overall. Writes
`tokenizers/<name>/tokenizer.json` and `tokenizers/<name>/report.json`.

The process caps its own memory (`--max-memory-gb`, RLIMIT_AS): on the pod an out-of-memory tokenizer must fail by
itself, not make the kernel kill the training run next to it.

    uv run python datasets/train_bpe_from_text.py --text datasets/raw/cosmo-code-math-pretrain-bpe1024-text \\
        --vocab-size 32768 [--name pretrain-mix-bpe32768] [--sample-bytes 1e9] [--threads 6]
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_agent_mix_bpe import SPECIAL  # noqa: E402  (the 16 tags)
from build_agent_shell_bpe import train_tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def documents(text_dir: Path, shard: dict):
    offs = np.load(text_dir / shard["offsets"])
    cats = np.load(text_dir / shard["categories"])
    blob = (text_dir / shard["text"]).read_bytes()
    for i in range(len(cats)):
        yield blob[offs[i]:offs[i + 1]].decode("utf-8"), int(cats[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--name", default=None)
    p.add_argument("--sample-bytes", type=float, default=1e9)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--max-memory-gb", type=float, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(a.threads)
    cap = int(a.max_memory_gb * 2 ** 30)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    from tokenizers import Tokenizer

    text_dir = Path(a.text)
    meta = json.loads((text_dir / "metadata.json").read_text())
    name = a.name or f"pretrain-mix-bpe{a.vocab_size}"
    names = {v: k for k, v in meta["category_ids"].items()}
    t0 = time.time()

    # a random sample of documents: shards in random order, each document kept with probability sample / total
    rng = np.random.default_rng(a.seed)
    train = meta["splits"]["train"]
    keep = min(1.0, a.sample_bytes / train["bytes"])
    sample, nbytes = [], 0
    for k in rng.permutation(len(train["shards"])):
        for text, _ in documents(text_dir, train["shards"][k]):
            if rng.random() < keep:
                sample.append(text)
                nbytes += len(text.encode("utf-8"))
    print(f"sample: {len(sample)} documents, {nbytes / 1e9:.2f} GB ({time.time() - t0:.0f} s)", flush=True)
    tok, n_merges = train_tokenizer(sample, a.vocab_size, SPECIAL)
    del sample
    V = tok.get_vocab_size()
    assert V == a.vocab_size and n_merges == a.vocab_size - len(SPECIAL) - 256, (V, n_merges)
    assert [tok.token_to_id(s) for s in SPECIAL] == list(range(len(SPECIAL)))
    print(f"trained: {V} ids ({n_merges} merges), {time.time() - t0:.0f} s", flush=True)

    old = Tokenizer.from_file(str(ROOT / "tokenizers" / meta["tokenizer"] / "tokenizer.json"))
    per = {}
    for shard in meta["splits"]["validation"]["shards"]:
        docs = list(documents(text_dir, shard))
        texts = [d for d, _ in docs]
        new_ids = tok.encode_batch(texts)
        old_ids = old.encode_batch(texts)
        for (text, c), n, o in zip(docs, new_ids, old_ids):
            assert tok.decode(n.ids, skip_special_tokens=False) == text, "not lossless"
            s = per.setdefault(names[c], [0, 0, 0])
            s[0] += len(text)
            s[1] += len(n.ids)
            s[2] += len(o.ids)
    tagged = "<bos>def f(x):\n    return x  # <user>\n<eos>"
    assert tok.encode(tagged).ids[0] == 0 and tok.encode(tagged).ids[-1] == 1
    out = ROOT / "tokenizers" / name
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    tot = [sum(v[i] for v in per.values()) for i in range(3)]
    report = {"name": name, "vocab_size": V, "merges": n_merges, "sample_bytes": nbytes, "source": str(text_dir),
              "chars_per_token": {k: {"new": round(v[0] / v[1], 3), "old": round(v[0] / v[2], 3),
                                      "tokens_ratio": round(v[1] / v[2], 3)} for k, v in sorted(per.items())},
              "overall": {"new": round(tot[0] / tot[1], 3), "old": round(tot[0] / tot[2], 3),
                          "tokens_ratio": round(tot[1] / tot[2], 3)}}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{'category':16s} chars/token: old {meta['tokenizer']}  new {name}   tokens new/old")
    for k, v in report["chars_per_token"].items():
        print(f"  {k:16s} {v['old']:6.3f}  {v['new']:6.3f}   {v['tokens_ratio']:.3f}")
    o = report["overall"]
    print(f"  {'overall':16s} {o['old']:6.3f}  {o['new']:6.3f}   {o['tokens_ratio']:.3f}   ({time.time() - t0:.0f} s) -> {out}")


if __name__ == "__main__":
    sys.exit(main())
