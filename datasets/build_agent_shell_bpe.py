"""BPE version of `agent-shell-bytes`: the same documents, re-encoded with a byte-level BPE learned on the train split.

Vocabulary (default 1024): the 12 special tags (ids 0-11, as in the byte dataset), the 256 bytes (id 12 + byte, as
in the byte dataset), then the merges in the order they were learned (ids 268..). GPT-2 byte-level pre-tokenization
(merges never cross a word / space / punctuation boundary; the tags are never merged).

    .venv/bin/python datasets/build_agent_shell.py          # the byte dataset first (the documents come from it)
    .venv/bin/python datasets/build_agent_shell_bpe.py      # -> datasets/agent-shell-bpe1024/, tokenizers/agent-shell-bpe1024/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_agent_shell as bytes_ds  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPECIAL, BYTE0 = bytes_ds.SPECIAL, bytes_ds.BYTE0


def documents(path: Path) -> List[str]:
    """The texts of a byte split (each document starts at <bos>)."""
    ids = np.load(path)
    starts = np.flatnonzero(ids == bytes_ds.BOS).tolist() + [ids.size]
    return [bytes_ds.decode(ids[a:b].tolist()) for a, b in zip(starts[:-1], starts[1:])]


def train_tokenizer(texts: List[str], vocab_size: int, special: List[str] = SPECIAL):
    """Byte-level BPE with fixed ids: the tags (0..), byte b at len(special) + b, then the merges in learned order."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    pre = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m)
              for m in json.loads(tok.to_str())["model"]["merges"]]
    # renumber: the byte alphabet in byte order (GPT-2's byte-to-unicode map), as the byte dataset numbers it
    b2u = {b: u for u, b in ((u, b) for b, u in _bytes_to_unicode().items())}
    vocab = {s: i for i, s in enumerate(special)}
    vocab.update({b2u[b]: len(special) + b for b in range(256)})
    for a, b in merges:
        if a + b not in vocab:
            vocab[a + b] = len(vocab)
    out = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    out.pre_tokenizer = pre
    out.decoder = decoders.ByteLevel()
    out.add_special_tokens(special)
    return out, len(merges)


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=1024, help="total: 12 tags + 256 bytes + merges")
    a = p.parse_args()
    name = f"agent-shell-bpe{a.vocab_size}"
    src = Path(__file__).resolve().parent / bytes_ds.NAME
    out_dir, tok_dir = Path(__file__).resolve().parent / name, ROOT / "tokenizers" / name
    src_meta = json.loads((src / "metadata.json").read_text())

    train_txt, val_txt = documents(src / "train.npy"), documents(src / "validation.npy")
    tok, n_merges = train_tokenizer(train_txt, a.vocab_size)
    V = tok.get_vocab_size()
    assert V <= a.vocab_size, V
    for s in train_txt[:200] + val_txt[:200]:  # lossless, and the tags keep their ids
        ids = tok.encode(s).ids
        assert tok.decode(ids, skip_special_tokens=False) == s
        assert [i for i in ids if i < BYTE0] == [i for i in bytes_ds.encode(s) if i < BYTE0]

    out_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(tok_dir / "tokenizer.json"))
    q = lambda xs: {k: int(np.percentile(xs, v)) for k, v in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))}
    splits, span_bytes, span_tokens = {}, 0, 0
    open_id, close_id = SPECIAL.index("<assistant>"), SPECIAL.index("</assistant>")
    for split, texts in (("train", train_txt), ("validation", val_txt)):
        enc = [e.ids for e in tok.encode_batch(texts)]
        arr = np.concatenate([np.asarray(e, dtype=np.int32) for e in enc])
        np.save(out_dir / f"{split}.npy", arr)
        n_bytes = sum(len(bytes_ds.encode(t)) for t in texts)
        splits[split] = {"file": f"{split}.npy", "tokens": int(arr.size), "documents": len(texts),
                         "doc_tokens": q([len(e) for e in enc]), "bytes_per_token": round(n_bytes / arr.size, 4)}
        if split == "validation":  # bytes per scored token (inside the assistant spans: what the loss covers)
            for e in enc:
                inside = False
                for i in e:
                    if inside:
                        span_tokens += 1
                        span_bytes += 1 if i < BYTE0 else len(tok.decode([i]).encode("utf-8"))
                    inside = (inside or i == open_id) and i != close_id
    meta = {
        "name": name,
        "source": f"{bytes_ds.NAME} documents re-encoded with a byte-level BPE (datasets/build_agent_shell_bpe.py)",
        "tokenizer": name,
        "vocabulary_size": V,
        "doc_token_id": src_meta["doc_token_id"],
        "bos_token_id": src_meta["bos_token_id"],
        "eos_token_id": src_meta["eos_token_id"],
        "special_tokens": src_meta["special_tokens"],
        "loss_spans": src_meta["loss_spans"],
        "merges": n_merges,
        # bytes per token over the scored targets (assistant spans): bpc = loss / ln 2 / this (the byte dataset: 1.0)
        "avg_chars_per_token": round(span_bytes / span_tokens, 4),
        "dtype": "int32",
        **splits,
        "notes": f"Token = special tag (ids 0-11), one UTF-8 byte (id 12 + byte, as in {bytes_ds.NAME}) or a merge "
                 f"(ids {BYTE0 + 256}..{V - 1}). Same documents as {bytes_ds.NAME}.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {out_dir}/: vocab {V} ({n_merges} merges); train {splits['train']['tokens']:,} tokens "
          f"({splits['train']['bytes_per_token']} bytes/token), validation {splits['validation']['tokens']:,}; "
          f"assistant spans {meta['avg_chars_per_token']} bytes/token; doc tokens {splits['train']['doc_tokens']}")
    print("longest merges:", sorted((s for s in tok.get_vocab() if s not in SPECIAL), key=len)[-12:])


if __name__ == "__main__":
    main()
