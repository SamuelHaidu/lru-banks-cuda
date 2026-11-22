"""`agent-mix-bpe1024`: our synthetic `agent-shell-bytes` conversations + Atum09/agent-training-dataset (ChatML),
in one format, encoded with a byte-level BPE learned on the mixed train split.

Tags (16): the 12 of `agent-shell-bytes` (same ids 0-11), then <system> </system> <think> </think> (12-15); the
256 bytes at 16 + byte; then the merges (1024 ids in total by default).

A converted Atum09 conversation:

    <bos><system>...</system><tools>
    web_search(query: str, num_results: int): Search the web
    </tools><user>...</user><assistant><think>...</think><call>web_search(query="...", num_results=5)</call></assistant>
    <result>...</result><assistant>...</assistant><eos>

ChatML -> ours: system -> <system>, tool list -> <tools> lines `name(param: type, ...): description`, human ->
<user>, gpt -> <assistant> (its `<think>` block -> the <think> tags, each `<tool_call>{json}</tool_call>` -> <call>name(
arg=<JSON value>, ...)</call>, string arguments as JSON strings like ours), tool -> <result> (the response's
`content`, as text). Several calls in one message: one <assistant> with every <call>, then a <result> each.
Rows repeat in the source (59k rows, ~18.6k distinct): deduplicated; validation rows also in train are dropped.
Loss: inside <assistant> ... </assistant> (the thinking included); system, tools, user and results are context.

    curl -L -o datasets/raw/atum09-agent/train_chatml.jsonl \\
      https://huggingface.co/datasets/Atum09/agent-training-dataset/resolve/8b96a957ae17d1b7f956bc31c4ff459966bbe448/data/chatml/train_chatml.jsonl
    (same for val_chatml.jsonl)
    .venv/bin/python datasets/build_agent_mix_bpe.py --samples 3     # print converted documents
    .venv/bin/python datasets/build_agent_mix_bpe.py                 # -> datasets/agent-mix-bpe1024/
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_agent_shell as shell_ds  # noqa: E402
from build_agent_shell_bpe import documents, train_tokenizer  # noqa: E402

ROOT = HERE.parent
RAW = HERE / "raw" / "atum09-agent"
SPECIAL = shell_ds.SPECIAL + ["<system>", "</system>", "<think>", "</think>"]
BYTE0 = len(SPECIAL)
TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "array": "list", "object": "dict"}
_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.S)
_RESP = re.compile(r"\s*<tool_response>\s*(.*?)\s*</tool_response>\s*", re.S)


def tool_line(t: Dict) -> str:
    params = t.get("parameters") or {}
    if isinstance(params, dict) and params.get("type") == "object" and "properties" in params:
        params = params["properties"]  # JSON-schema form
    ps = []
    for k, v in params.items() if isinstance(params, dict) else []:
        ty = v.get("type") if isinstance(v, dict) else None
        ps.append(f"{k}: {TYPES.get(ty, ty or 'str')}")
    desc = " ".join(str(t.get("description", "")).split())
    return f"{t['name']}({', '.join(ps)}): {desc}"


def render_call(d: Dict) -> str:
    args = d.get("arguments") or {}
    if isinstance(args, str):
        args = json.loads(args)
    return f"{d['name']}(" + ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items()) + ")"


def assistant(v: str, stats: collections.Counter) -> Tuple[str, int]:
    """A gpt message -> the inside of <assistant>: its thinking, its text, its calls (in order). Returns the text
    and the number of calls."""
    out, pos, n = [], 0, 0
    for m in re.finditer(r"<think>.*?</think>|<tool_call>.*?</tool_call>", v, re.S):
        text = v[pos:m.start()].strip()
        if text:
            out.append(text)
        if m.group(0).startswith("<think>"):
            out.append("<think>" + _THINK.fullmatch(m.group(0)).group(1) + "</think>")
            stats["think"] += 1
        else:
            out.append("<call>" + render_call(json.loads(_CALL.fullmatch(m.group(0)).group(1))) + "</call>")
            n += 1
        pos = m.end()
    text = v[pos:].strip()
    if text:
        out.append(text)
    return "".join(out), n


def result(v: str) -> str:
    m = _RESP.fullmatch(v)
    body = m.group(1) if m else v.strip()
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        return body
    c = d.get("content", d) if isinstance(d, dict) else d
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)


def convert(row: Dict, stats: collections.Counter) -> Optional[str]:
    """One ChatML row -> one document in our format (None if it cannot be converted faithfully)."""
    parts = ["<bos>"]
    cv = row["conversations"]
    try:
        tools = "<tools>\n" + "".join(tool_line(t) + "\n" for t in row.get("tools") or []) + "</tools>"
        i = 0
        if cv and cv[0]["from"] == "system":
            parts.append(f"<system>{cv[0]['value'].strip()}</system>")
            i = 1
        parts.append(tools)
        pending = 0  # calls waiting for their result
        for m in cv[i:]:
            role, v = m["from"], m["value"]
            if role == "human":
                parts.append(f"<user>{v.strip()}</user>")
            elif role == "gpt":
                body, n = assistant(v, stats)
                parts.append(f"<assistant>{body}</assistant>")
                pending += n
            elif role == "tool":
                parts.append(f"<result>{result(v)}</result>")
                pending -= 1
            else:
                stats[f"skip: role {role}"] += 1
                return None
        if pending:
            stats["skip: calls and results do not pair"] += 1
            return None
    except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as e:
        stats[f"skip: {type(e).__name__}"] += 1
        return None
    parts.append("<eos>")
    doc = "".join(parts)
    # the source's own text must not contain our tags (they would encode as tags): count of tags = what we wrote
    if len(re.findall("|".join(re.escape(t) for t in SPECIAL), doc)) != _expected_tags(parts):
        stats["skip: source text contains one of our tags"] += 1
        return None
    return doc


def _expected_tags(parts: List[str]) -> int:
    n = 0
    for p in parts:
        if p in ("<bos>", "<eos>"):
            n += 1
        elif p.startswith("<assistant>"):
            body = p[len("<assistant>"):-len("</assistant>")]
            n += 2 + 2 * len(re.findall(r"<think>(?:(?!<think>|</think>).)*</think>", body, re.S)) \
                + 2 * len(re.findall(r"<call>(?:(?!<call>|</call>).)*</call>", body, re.S))
        else:
            n += 2
    return n


def load_atum(split: str, stats: collections.Counter, exclude: set = frozenset()) -> List[str]:
    out, seen = [], set(exclude)
    with open(RAW / f"{split}_chatml.jsonl") as f:
        for line in f:
            row = json.loads(line)
            stats[f"{split} rows"] += 1
            doc = convert(row, stats)
            if doc is None:
                continue
            if doc in seen:
                stats[f"{split} duplicates"] += 1
                continue
            seen.add(doc)
            out.append(doc)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=1024, help="total: 16 tags + 256 bytes + merges")
    p.add_argument("--samples", type=int, default=0, help="print this many converted documents and exit")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    stats = collections.Counter()
    atum_train = load_atum("train", stats)
    if a.samples:
        rng = np.random.default_rng(a.seed)
        for i in rng.choice(len(atum_train), a.samples, replace=False):
            print(atum_train[i], "\n" + "-" * 100)
        print(dict(stats))
        return
    atum_val = load_atum("val", stats, exclude=set(atum_train))
    shell_train = documents(HERE / shell_ds.NAME / "train.npy")
    shell_val = documents(HERE / shell_ds.NAME / "validation.npy")
    rng = np.random.default_rng(a.seed)
    train = [shell_train[i] for i in range(len(shell_train))] + atum_train
    val = shell_val + atum_val
    train = [train[i] for i in rng.permutation(len(train))]
    val = [val[i] for i in rng.permutation(len(val))]

    name = f"agent-mix-bpe{a.vocab_size}"
    out_dir, tok_dir = HERE / name, ROOT / "tokenizers" / name
    tok, n_merges = train_tokenizer(train, a.vocab_size, SPECIAL)
    V = tok.get_vocab_size()
    tag_re = re.compile("|".join(re.escape(t) for t in SPECIAL))
    for s in train[:300] + val[:300]:  # lossless, the tags encode as tags
        ids = tok.encode(s).ids
        assert tok.decode(ids, skip_special_tokens=False) == s
        assert [SPECIAL[i] for i in ids if i < BYTE0] == tag_re.findall(s)

    assert tok.encode("\x00").ids == [BYTE0] and tok.encode("é").ids == [BYTE0 + 0xC3, BYTE0 + 0xA9]
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(tok_dir / "tokenizer.json"))
    q = lambda xs: {k: int(np.percentile(xs, v)) for k, v in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))}
    open_id, close_id = SPECIAL.index("<assistant>"), SPECIAL.index("</assistant>")
    splits, sources, span_bytes, span_tokens = {}, {}, 0, 0
    for split, texts in (("train", train), ("validation", val)):
        enc = [e.ids for e in tok.encode_batch(texts)]
        arr = np.concatenate([np.asarray(e, dtype=np.int32) for e in enc])
        np.save(out_dir / f"{split}.npy", arr)
        n_bytes = sum(len(tag_re.sub("\x00", t).encode("utf-8")) for t in texts)  # a tag counts 1, as in the byte data
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
    for nm, tr, va in (("agent-shell-bytes", shell_train, shell_val), ("atum09", atum_train, atum_val)):
        sources[nm] = {"train_documents": len(tr), "validation_documents": len(va),
                       "train_bytes": sum(len(tag_re.sub("\x00", t).encode("utf-8")) for t in tr)}
    meta = {
        "name": name,
        "source": "agent-shell-bytes (synthetic, datasets/build_agent_shell.py) + Atum09/agent-training-dataset "
                  "(ChatML, rev 8b96a95, deduplicated), one format, byte-level BPE (datasets/build_agent_mix_bpe.py)",
        "sources": sources,
        "tokenizer": name,
        "vocabulary_size": V,
        "doc_token_id": 0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "special_tokens": {s: i for i, s in enumerate(SPECIAL)},
        "loss_spans": [[open_id, close_id]],
        "merges": n_merges,
        # bytes per token over the scored targets (assistant spans): bpc = loss / ln 2 / this
        "avg_chars_per_token": round(span_bytes / span_tokens, 4),
        "dtype": "int32",
        "seed": a.seed,
        **splits,
        "notes": f"Token = tag (ids 0-{BYTE0 - 1}), one UTF-8 byte (id {BYTE0} + byte) or a merge "
                 f"(ids {BYTE0 + 256}..{V - 1}).",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(dict(stats))
    print(f"Wrote {out_dir}/: vocab {V} ({n_merges} merges); train {splits['train']['tokens']:,} tokens "
          f"({len(train):,} docs, {splits['train']['bytes_per_token']} bytes/token), validation "
          f"{splits['validation']['tokens']:,} ({len(val):,} docs); assistant spans {meta['avg_chars_per_token']} "
          f"bytes/token; doc tokens {splits['train']['doc_tokens']}")
    print("sources:", sources)


if __name__ == "__main__":
    main()
