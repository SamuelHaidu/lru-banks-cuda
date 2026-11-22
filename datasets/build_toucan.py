"""Builds `toucan-bpe32768`: Toucan-1.5M (Agent-Ark/Toucan-1.5M, Apache 2.0; arXiv 2510.01179), tool-agent
trajectories over 495 real MCP servers, converted to our tool format, English only.

    python datasets/build_toucan.py --sample 3                       # convert the shards in --raw, print 3 documents
    python datasets/build_toucan.py --stats                          # keep / drop counts and token lengths of --raw
    python datasets/build_toucan.py --build --out DIR [--workers 6]  # every shard: download, convert, delete

Format (the tags are single tokens of the pretrain-mix-bpe32768 tokenizer; kodcode-agent uses the same):
    <tools>
    name(arg: type, opt?: type = default): description
      arg: argument description
    </tools><user>question</user><assistant><think>reasoning</think>text
    <call>name(key=<JSON value>, ...)</call><call>...</call></assistant><result>output</result><result>...</result>
    <assistant>answer</assistant><user>follow-up</user>...
- Calls: `name(k=v)` with v the argument's JSON (a string is JSON-quoted, as in kodcode-agent's `render_call`).
- Consecutive assistant messages (Kimi-K2 splits text and call; OSS splits reasoning and call) are one turn; parallel
  calls are several <call>s in one turn, their results follow in order.
- The OSS (gpt-oss) reasoning is kept as <think>...</think> at the start of its turn.
- loss_spans = the assistant turns, as kodcode-agent.

Filters: artifacts (tool-call arguments that are not JSON, no user turn or no final answer, tag text such as <call>
inside the content), and documents with more than --max-foreign of their letters in a non-Latin script (CJK,
Cyrillic, Arabic, ...). Latin-script languages are kept.

Splits: by a hash of the first user message (the multi-turn extensions keep their question in the same split):
0.1% validation, 0.1% test, the rest train. SFT (a curated subset of the other three) only adds unseen uuids.
Output: train.npy.zst (multi-frame: the .npy header, then each shard's tokens), validation.npy / test.npy,
{split}.categories.npy (start, length, category), {validation,test}.docs.jsonl, metadata.json.
"""

import argparse
import json
import re
import sys
import textwrap
import time
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "Agent-Ark/Toucan-1.5M"
CONFIGS = ["Kimi-K2", "OSS", "Qwen3", "SFT"]
SUBSETS = ["single-turn-original", "irrelevant", "single-turn-diversify", "multi-turn"]
TAGS = ["<bos>", "<eos>", "<tools>", "</tools>", "<user>", "</user>", "<assistant>", "</assistant>", "<call>",
        "</call>", "<result>", "</result>", "<system>", "</system>", "<think>", "</think>"]
FOREIGN = re.compile("[\u0370-\u03ff\u0400-\u052f\u0590-\u05ff\u0600-\u06ff\u0900-\u0dff\u0e00-\u0eff"
                     "\u1100-\u11ff\u3040-\u30ff\u3130-\u318f\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")
LETTERS = re.compile(r"[^\W\d_]")


def foreign_letters(text):
    """(letters, letters in a non-Latin script: Greek, Cyrillic, Hebrew, Arabic, Indic, Thai, CJK, Hangul)."""
    if text.isascii():
        return 1, 0
    return len(LETTERS.findall(text)), len(FOREIGN.findall(text))


# -- conversion -----------------------------------------------------------------------------------------------------

def type_str(s):
    if not isinstance(s, dict):
        return "any"
    if "enum" in s:
        return " | ".join(json.dumps(v, ensure_ascii=False) for v in s["enum"])
    for k in ("anyOf", "oneOf"):
        if k in s:
            parts = [type_str(x) for x in s[k] if not (isinstance(x, dict) and x.get("type") == "null")]
            return " | ".join(dict.fromkeys(parts)) or "any"
    t = s.get("type", "any")
    if isinstance(t, list):
        return " | ".join(x for x in t if x != "null") or "any"
    if t == "array":
        return f"array[{type_str(s.get('items', {}))}]"
    return t


def clean(desc):
    return textwrap.dedent(desc or "").strip()


def render_tools(tools):
    lines = []
    for t in tools:
        f = t.get("function", t)
        params = f.get("parameters") or {}
        props = params.get("properties") or {}
        req = set(params.get("required") or [])
        sig = []
        for k, s in props.items():
            x = f"{k}{'' if k in req else '?'}: {type_str(s)}"
            if isinstance(s, dict) and "default" in s and k not in req:
                x += f" = {json.dumps(s['default'], ensure_ascii=False)}"
            sig.append(x)
        d = clean(f.get("description"))
        lines.append(f"{f['name']}({', '.join(sig)})" + (f": {d}" if d else "").replace("\n", "\n  "))
        for k, s in props.items():
            pd = clean(s.get("description")) if isinstance(s, dict) else ""
            if pd:
                lines.append(f"  {k}: " + pd.replace("\n", "\n    "))
    return "<tools>\n" + "\n".join(lines) + "\n</tools>"


def render_call(name, args):
    if not isinstance(args, dict):
        raise ValueError("arguments are not an object")
    return f"{name}(" + ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items()) + ")"


def parse_args(a):
    if isinstance(a, dict):
        return a
    a = (a or "").strip()
    return json.loads(a) if a else {}


def normalize(config, row):
    """[(role, payload)] with role user / assistant (text, reasoning, calls) / result; tools list."""
    msgs = json.loads(row["messages"])
    tools = json.loads(row["tools"] if config == "SFT" else row["available_tools"])
    out = []
    for m in msgs:
        r = m["role"]
        if r == "system":
            continue
        if r == "user":
            out.append(("user", m["content"]))
        elif r in ("assistant", "tool_call"):
            if r == "tool_call":  # SFT: a Python-literal dict {'name': ..., 'arguments': '<json>'}
                import ast
                d = ast.literal_eval(m["content"])
                calls, text, reason = [(d["name"], parse_args(d["arguments"]))], "", ""
            else:
                fc = m.get("function_call")
                calls = [(fc["name"], parse_args(fc["arguments"]))] if fc else []
                for tc in m.get("tool_calls") or []:
                    calls.append((tc["function"]["name"], parse_args(tc["function"]["arguments"])))
                text, reason = m.get("content") or "", m.get("reasoning_content") or ""
            if out and out[-1][0] == "assistant" and not out[-1][1]["closed"]:
                prev = out[-1][1]
                if text and prev["calls"]:  # text after a call in the same turn: a new turn would need a result
                    prev["text_after"] = prev.get("text_after", "") + text
                else:
                    prev["text"] = (prev["text"] + "\n" + text).strip("\n") if prev["text"] and text else \
                        prev["text"] or text
                prev["reason"] = (prev["reason"] + "\n" + reason).strip("\n") if prev["reason"] and reason else \
                    prev["reason"] or reason
                prev["calls"] += calls
            else:
                out.append(("assistant", {"text": text, "reason": reason, "calls": calls, "closed": False}))
        elif r in ("function", "tool", "tool_response"):
            if out and out[-1][0] == "assistant":
                out[-1][1]["closed"] = True
            out.append(("result", m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])))
        else:
            raise ValueError(f"role {r}")
    return tools, out


def to_text(tools, turns):
    parts = [render_tools(tools)]
    for role, p in turns:
        if role == "user":
            parts.append(f"<user>{p}</user>")
        elif role == "result":
            parts.append(f"<result>{p}</result>")
        else:
            body = (f"<think>{p['reason']}</think>" if p["reason"] else "") + p["text"].strip()
            for name, args in p["calls"]:
                body += ("\n" if body and not body.endswith("\n") else "") + f"<call>{render_call(name, args)}</call>"
            if p.get("text_after"):
                raise ValueError("text after a call in one turn")
            parts.append(f"<assistant>{body}</assistant>")
    return "".join(parts)


def segments(tools, turns):
    """(natural-language segments, result segments) for the English filter."""
    nat, res = [], []
    for t in tools:
        f = t.get("function", t)
        nat.append(clean(f.get("description")))
        for s in ((f.get("parameters") or {}).get("properties") or {}).values():
            if isinstance(s, dict):
                nat.append(clean(s.get("description")))
    for role, p in turns:
        if role == "user":
            nat.append(p)
        elif role == "result":
            res.append(p)
        else:
            nat += [p["text"], p["reason"]]
            res += [json.dumps(a, ensure_ascii=False) for _, a in p["calls"]]
    return [s for s in nat if s], res


def convert(config, row, max_foreign):
    """(text, None) or (None, drop reason)."""
    try:
        tools, turns = normalize(config, row)
        text = to_text(tools, turns)
    except (ValueError, KeyError, SyntaxError, TypeError, json.JSONDecodeError) as e:
        return None, f"malformed: {type(e).__name__}"
    if not any(r == "user" for r, _ in turns) or turns[-1][0] != "assistant":
        return None, "no user turn or does not end with an answer"
    body = text[text.index("</tools>"):]
    if any(t in body.replace("</tools><user>", "", 1) for t in ("<tools>", "</tools>", "<bos>", "<eos>")) or \
            sum(text.count(t) for t in TAGS) != expected_tags(turns):
        return None, "tag text inside the content"
    nat, res = segments(tools, turns)
    n, f = foreign_letters(" ".join(nat + res))
    if n and f / n > max_foreign:
        return None, "non-Latin script"
    return text, None


def expected_tags(turns):
    n = 2  # <tools> </tools>
    for role, p in turns:
        n += 2
        if role == "assistant":
            n += 2 * len(p["calls"]) + (2 if p["reason"] else 0)
    return n


def split_of(turns_first_user):
    h = zlib.crc32(turns_first_user.encode()) % 1000
    return "validation" if h == 0 else "test" if h == 1 else "train"


def first_user(row):
    for m in json.loads(row["messages"]):
        if m["role"] == "user":
            return m["content"]
    return ""


def category(config, subset):
    return CONFIGS.index(config) * len(SUBSETS) + SUBSETS.index(subset)


# -- drivers --------------------------------------------------------------------------------------------------------

def shard_files():
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=1"))
    files = sorted(x["path"] for x in tree if x["path"].endswith(".parquet"))
    return sorted(files, key=lambda p: (CONFIGS.index(p.split("/")[0]), p))


def read_rows(path, config):
    import pyarrow.parquet as pq
    cols = ["uuid", "subset_name", "messages"] + (["tools"] if config == "SFT" else ["available_tools"])
    return pq.read_table(path, columns=cols).to_pylist()


def work(job):
    config, row, mf = job
    text, why = convert(config, row, mf)
    return row["uuid"], row["subset_name"], first_user(row), text, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/root/toucan_raw")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--dropped", type=int, default=0, help="with --stats: print N dropped segments per reason")
    ap.add_argument("--out", default=str(ROOT / "datasets" / "toucan-bpe32768"))
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizers" / "pretrain-mix-bpe32768" / "tokenizer.json"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-foreign", type=float, default=0.005)
    a = ap.parse_args()
    raw = Path(a.raw)

    if a.sample or a.stats:
        from collections import Counter
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
        for path in sorted(raw.rglob("*.parquet")):
            config = path.parent.name
            rows = read_rows(path, config)
            res = Counter()
            kept, shown, lens = [], 0, []
            drops = {}
            for row in rows:
                text, why = convert(config, row, a.max_foreign)
                res[why or "kept"] += 1
                if text:
                    kept.append((row["subset_name"], text))
                elif len(drops.setdefault(why, [])) < a.dropped:
                    drops[why].append(first_user(row)[:200].replace("\n", " "))
            for _, t in kept[:2000]:
                lens.append(len(tok.encode(t).ids))
            import numpy as np
            print(f"== {path.relative_to(raw)}: {len(rows)} rows; " + ", ".join(f"{k} {v}" for k, v in res.most_common()))
            if lens:
                print(f"   tokens per document (2000 kept): mean {np.mean(lens):.0f}, p50 {np.percentile(lens, 50):.0f}, "
                      f"p90 {np.percentile(lens, 90):.0f}, p99 {np.percentile(lens, 99):.0f}, max {max(lens)}")
            for why, ex in drops.items():
                for e in ex:
                    print(f"   [{why}] {e}")
            for sub, t in kept[:a.sample]:
                print(f"\n----- {config} / {sub} -----\n{t[:3000]}{' ...' if len(t) > 3000 else ''}")
        return

    if a.build:
        build(a)


def build(a):
    import os
    import shutil
    from multiprocessing import Pool

    import numpy as np
    import zstandard
    from tokenizers import Tokenizer

    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    tok = Tokenizer.from_file(a.tokenizer)
    bos, eos, a_open, a_close = (tok.token_to_id(t) for t in ("<bos>", "<eos>", "<assistant>", "</assistant>"))
    out = Path(a.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    raw = Path(a.raw) / "build"
    raw.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"  # resumable: shards done, uuids seen, counts
    st = json.loads(state_path.read_text()) if state_path.exists() else \
        {"done": [], "drops": {}, "cats": {}, "train_tokens": 0, "train_docs": 0, "part_bytes": []}
    seen_path = out / "uuids.txt"
    seen = set(seen_path.read_text().split()) if seen_path.exists() else set()
    held = {"validation": [], "test": []}
    held_path = out / "held.jsonl"
    cctx = zstandard.ZstdCompressor(level=6, threads=2)
    t0 = time.time()
    files = shard_files()
    for i, f in enumerate(files):
        if f in st["done"]:
            continue
        config = f.split("/")[0]
        local = raw / f.replace("/", "_")
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/main/{f}", local)
                break
            except Exception as e:  # noqa: BLE001 (network)
                print(f"  download {f} failed ({e}), retry", flush=True)
                time.sleep(10)
        rows = read_rows(local, config)
        if config == "SFT":
            rows = [r for r in rows if r["uuid"] not in seen]
        jobs = [(config, r, a.max_foreign) for r in rows]
        texts, cats = [], []
        with Pool(a.workers) as pool, open(seen_path, "a") as fs, open(held_path, "a") as fh:
            for uuid, subset, q, text, why in pool.imap(work, jobs, chunksize=64):
                fs.write(uuid + "\n")
                seen.add(uuid)
                key = f"{config}/{subset}"
                if text is None:
                    st["drops"][f"{key}: {why}"] = st["drops"].get(f"{key}: {why}", 0) + 1
                    continue
                sp = split_of(q)
                if sp != "train":
                    fh.write(json.dumps({"split": sp, "uuid": uuid, "config": config, "subset": subset,
                                         "text": text}, ensure_ascii=False) + "\n")
                    continue
                texts.append(text)
                cats.append(category(config, subset))
        encs = tok.encode_batch(["<bos>" + t + "<eos>" for t in texts], add_special_tokens=False)
        lens = np.array([len(e.ids) for e in encs], np.int64)
        toks = np.concatenate([np.asarray(e.ids, np.uint16) for e in encs]) if encs else np.zeros(0, np.uint16)
        starts = st["train_tokens"] + np.concatenate([[0], np.cumsum(lens)[:-1]]) if len(lens) else lens
        part = out / "parts" / f"{i:04d}"
        (part.with_suffix(".bin.zst")).write_bytes(cctx.compress(toks.tobytes()))
        np.save(part.with_suffix(".cats.npy"), np.stack([starts, lens, np.array(cats, np.int64)], 1)
                if len(lens) else np.zeros((0, 3), np.int64))
        for c, n in zip(cats, lens):
            k = str(c)
            st["cats"].setdefault(k, [0, 0])
            st["cats"][k][0] += 1
            st["cats"][k][1] += int(n)
        st["train_tokens"] += int(lens.sum())
        st["train_docs"] += len(texts)
        st["done"].append(f)
        state_path.write_text(json.dumps(st))
        local.unlink()
        el = time.time() - t0
        print(f"[{len(st['done'])}/{len(files)}] {f}: {len(rows)} rows, {len(texts)} train docs, "
              f"{int(lens.sum()):,} tokens; total {st['train_tokens']:,} tokens, {el / 60:.0f} min", flush=True)

    # validation / test: the held-out documents, shuffled, as plain .npy
    meta_splits = {}
    rows = list({r["uuid"]: r for r in (json.loads(l) for l in open(held_path))}.values()) \
        if held_path.exists() else []  # by uuid: a shard interrupted and redone appended its rows twice
    for sp in ("validation", "test"):
        ds = sorted((r for r in rows if r["split"] == sp), key=lambda r: r["uuid"])
        encs = tok.encode_batch(["<bos>" + r["text"] + "<eos>" for r in ds], add_special_tokens=False)
        lens = np.array([len(e.ids) for e in encs], np.int64)
        toks = np.concatenate([np.asarray(e.ids, np.uint16) for e in encs])
        np.save(out / f"{sp}.npy", toks)
        np.save(out / f"{sp}.categories.npy", np.stack([np.concatenate([[0], np.cumsum(lens)[:-1]]), lens,
                                                         [category(r["config"], r["subset"]) for r in ds]], 1))
        with open(out / f"{sp}.docs.jsonl", "w") as fo:
            for r in ds:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta_splits[sp] = {"file": f"{sp}.npy", "tokens": int(lens.sum()), "documents": len(ds)}

    # train.npy.zst: a frame with the .npy header, then the parts' frames (zstd decodes concatenated frames as one)
    n = st["train_tokens"]
    header = np.lib.format.header_data_from_array_1_0(np.zeros(0, np.uint16))
    header["shape"] = (n,)
    import io
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(buf, header)
    parts = sorted((out / "parts").glob("*.bin.zst"))
    with open(out / "train.npy.zst", "wb") as fo:
        fo.write(cctx.compress(buf.getvalue()))
        for p in parts:  # appended then deleted one by one: the disk never holds the data twice
            with open(p, "rb") as fi:
                shutil.copyfileobj(fi, fo)
            p.unlink()
    cats = np.concatenate([np.load(p) for p in sorted((out / "parts").glob("*.cats.npy"))])
    np.save(out / "train.categories.npy", cats)
    shutil.rmtree(out / "parts")
    cat_ids = {f"{c}/{s}": category(c, s) for c in CONFIGS for s in SUBSETS}
    meta = {"name": out.name, "tokenizer": "pretrain-mix-bpe32768", "vocabulary_size": tok.get_vocab_size(),
            "doc_token_id": bos, "bos_token_id": bos, "eos_token_id": eos, "dtype": "uint16",
            "loss_spans": [[a_open, a_close]], "category_index": "{split}.categories.npy", "category_ids": cat_ids,
            "source": {"repo": REPO, "license": "apache-2.0", "paper": "arXiv 2510.01179"},
            "filter": {"max_foreign": a.max_foreign},
            "train": {"file": "train.npy", "stored_as": "train.npy.zst", "tokens": n, "documents": st["train_docs"]},
            **meta_splits,
            "categories": {k: {"documents": v[0], "tokens": v[1]} for k, v in
                           ((next(n for n, i in cat_ids.items() if str(i) == c), v) for c, v in st["cats"].items())},
            "drops": st["drops"]}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"done: {n:,} train tokens, {st['train_docs']:,} documents; {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
