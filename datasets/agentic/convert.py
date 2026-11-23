"""Converts the raw agentic sources (agentic/download.sh) to our conversation format, as sharded text per source.

    <system>...</system><tools>
    name(arg: type, opt?: type): description
    </tools><user>task</user><assistant><think>reasoning</think>text
    <call>name(key=<JSON value>, ...)</call></assistant><result>output</result><assistant>...</assistant>

Sources (--raw/<name>):
  terminal-corpus     Terminus-2 terminal agents          datasets/convert_nemotron_terminal.py
  toucan              MCP tool agents (OSS, Qwen3, Kimi)   datasets/build_toucan.py (SFT is a subset: skipped)
  nemotron-agentic    tool-calling assistants             OpenAI messages
  nemotron-swe        OpenHands SWE agents                OpenAI messages
  open-swe-traces     SWE agents of 5 frameworks          OpenAI messages
  nemotron-opencode   opencode CLI agent                  OpenAI messages
  openmath-tir        math with Python                    <tool_call> + ```output blocks -> python(code=...) calls

Every document: our tags must not occur in the source text (dropped), at most --max-foreign of its letters in a
non-Latin script (dropped; Greek is common in math, so openmath-tir allows more), system prompts cut to
--max-system characters and tool descriptions to --max-tool-desc (the same boilerplate repeats in every document).
Splits by a hash of the first user message: 1 in --val-every to validation, the rest to train.

Output, per source, the `datasets/decode_to_text.py` format (train_bpe_from_text.py and encode_text.py read it):
    <out>/<source>/{split}-{job:04d}.txt / .offsets.npy / .categories.npy + metadata.json
    <out>/_all/metadata.json   every source's shards (relative paths), to train one tokenizer on the mix
Categories: one global table "<source>/<sub>" (sub: the framework, the generator, the file).

    uv run --with pyarrow --with numpy python agentic/convert.py --raw /work/raw --out /work/text [--workers 30]
    uv run --with pyarrow --with numpy python agentic/convert.py --raw /work/raw --sample openmath-tir
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # datasets/ (on the pod: /work, next to agentic/)
sys.path.insert(0, str(HERE))
import build_toucan as toucan  # noqa: E402
import convert_nemotron_terminal as terminal  # noqa: E402

TAGS = toucan.TAGS
_TAG = re.compile("|".join(re.escape(t) for t in TAGS))
SOURCES = ["terminal-corpus", "toucan", "nemotron-agentic", "nemotron-swe", "open-swe-traces", "nemotron-opencode",
           "openmath-tir"]


# -- the OpenAI-messages family ---------------------------------------------------------------------------------------

def _tool_spec(t):
    """A tool definition: OpenAI {"function": ...}, Anthropic {"input_schema": ...}, opencode {"id", "inputSchema":
    {"jsonSchema": ...}}, or one of those as a JSON string."""
    if isinstance(t, str):
        t = json.loads(t)
    f = t.get("function", t)
    schema = f.get("parameters") or f.get("input_schema") or (f.get("inputSchema") or {}).get("jsonSchema") or {}
    return {"name": f.get("name") or f["id"], "description": f.get("description") or "", "parameters": schema}


def _cap(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + " [...]"


def _render_tools(specs, max_desc):
    for s in specs:
        s["description"] = _cap(s["description"], max_desc)
        for p in (s["parameters"].get("properties") or {}).values():
            if isinstance(p, dict) and p.get("description"):
                p["description"] = _cap(p["description"], max_desc // 2)
    return toucan.render_tools(specs)


def openai_doc(row, cfg):
    """(text, None) or (None, drop reason) for a row with OpenAI-style `messages` (+ `tools`)."""
    msgs = row["messages"]
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    tools = row.get("tools") or []
    if isinstance(tools, str):
        tools = json.loads(tools)
    try:
        specs = [x for x in (_tool_spec(t) for t in tools) if x["description"] != "Do not use"]  # opencode's "invalid"
    except (KeyError, TypeError, json.JSONDecodeError):
        return None, "malformed tools"
    parts, called, open_turn = [], {}, None
    system = ""

    def close():
        nonlocal open_turn
        if open_turn is not None:
            parts.append(f"<assistant>{open_turn}</assistant>")
            open_turn = None

    for m in msgs:
        role, content = m.get("role"), m.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if _TAG.search(content) or _TAG.search(m.get("reasoning_content") or ""):
            return None, "tag text inside the content"
        if role == "system":
            system = _cap(content, cfg.max_system)
        elif role == "user":
            close()
            parts.append(f"<user>{content.strip()}</user>")
        elif role == "assistant":
            body = ""
            if m.get("reasoning_content"):
                body += f"<think>{m['reasoning_content'].strip()}</think>"
            body += content.strip()
            for tc in m.get("tool_calls") or []:
                f = tc.get("function", tc)
                try:
                    args = toucan.parse_args(f.get("arguments"))
                    call = toucan.render_call(f["name"], args)
                except (ValueError, KeyError, json.JSONDecodeError):
                    return None, "tool-call arguments are not a JSON object"
                if _TAG.search(call):
                    return None, "tag text inside the content"
                called.setdefault(f["name"], set()).update(args)
                body += ("\n" if body and not body.endswith("\n") else "") + f"<call>{call}</call>"
            if open_turn is not None and body:  # consecutive assistant messages: one turn
                open_turn += ("\n" if not open_turn.endswith("\n") else "") + body
            elif body:
                open_turn = body
        elif role == "tool":
            close()
            parts.append(f"<result>{content.strip()}</result>")
        else:
            return None, f"role {role}"
    close()
    if not any(p.startswith("<user>") for p in parts):
        return None, "no user turn"
    if not parts or not parts[-1].startswith("<assistant>"):
        return None, "does not end with an answer"
    if not specs and called:  # tool names only (no schemas in the source): the arguments that were used
        specs = [{"name": n, "description": "", "parameters": {"properties": {k: {} for k in sorted(a)}}}
                 for n, a in called.items()]
    head = (f"<system>{system}</system>" if system else "") + (_render_tools(specs, cfg.max_tool_desc) if specs else "")
    return head + "".join(parts), None


# -- the other formats ------------------------------------------------------------------------------------------------

def terminal_doc(row, cfg):
    stats = collections.Counter()
    text = terminal.convert(row, stats)
    if text is None:
        return None, next(iter(stats))
    return text.removeprefix("<bos>").removesuffix("<eos>"), None


def toucan_doc(row, cfg):
    return toucan.convert(cfg.sub, row, 1.0)  # the script filter runs below, as for every source


# <tool_call>code</tool_call> ```output ...``` [```system Remaining code executions ...```: the harness budget, dropped]
_CODE = re.compile(r"<tool_call>\n(.*?)</tool_call>\s*```output\n(.*?)```(?:\s*```system\n.*?```)?", re.S)
PY_TOOLS = "<tools>\npython(code: str): run Python 3 code (sympy, numpy available); returns what it prints\n</tools>"


def openmath_doc(row, cfg):
    sol, problem = row["generated_solution"], row["problem"]
    if _TAG.search(problem) or _TAG.search(sol.replace("<think>", "", 1).replace("</think>", "", 1)):
        return None, "tag text inside the content"
    think, sep, answer = sol.partition("</think>")
    if not sep:
        return None, "no </think>"
    think = think.replace("<think>", "", 1)
    parts, last = [f"<user>{problem.strip()}</user>"], 0
    for m in _CODE.finditer(think):
        reason = think[last:m.start()].strip()
        call = toucan.render_call("python", {"code": m.group(1).rstrip("\n")})
        parts.append(f"<assistant><think>{reason}</think><call>{call}</call></assistant>")
        parts.append(f"<result>{m.group(2).strip()}</result>")
        last = m.end()
    if len(parts) == 1:
        return None, "no python call"
    parts.append(f"<assistant><think>{think[last:].strip()}</think>{answer.strip()}</assistant>")
    return PY_TOOLS + "".join(parts), None


ADAPTER = {"terminal-corpus": terminal_doc, "toucan": toucan_doc, "openmath-tir": openmath_doc}


# -- jobs: one input file (or a byte range of a big jsonl) each -----------------------------------------------------

def rows(job):
    path = Path(job["path"])
    if path.suffix == ".parquet":
        f = pq.ParquetFile(path)
        for b in f.iter_batches(batch_size=256):
            yield from b.to_pylist()
    else:
        with open(path, "rb") as fh:
            fh.seek(job["start"])
            if job["start"]:
                fh.readline()  # the line in progress belongs to the previous range
            while fh.tell() <= job["end"]:
                line = fh.readline()
                if not line:
                    break
                if line.strip():
                    yield json.loads(line)


def first_user(text):
    a = text.find("<user>")
    return text[a:a + 2000] if a >= 0 else text[:2000]


def run_job(job):
    cfg = argparse.Namespace(**job["cfg"])
    adapter = ADAPTER.get(job["source"], openai_doc)
    out = Path(job["out"])
    writers = {}
    stats = collections.Counter()
    for row in rows(job):
        try:
            text, why = adapter(row, cfg)
        except Exception as e:  # noqa: BLE001: a malformed row is dropped and counted, never fatal
            text, why = None, f"error {type(e).__name__}"
        if text is None:
            stats["drop: " + why] += 1
            continue
        n, f = toucan.foreign_letters(text)
        if n and f / n > cfg.max_foreign:
            stats["drop: non-Latin script"] += 1
            continue
        split = "validation" if zlib.crc32(first_user(text).encode()) % cfg.val_every == 0 else "train"
        if split not in writers:
            base = out / f"{split}-{job['id']:04d}"
            writers[split] = {"base": base, "fh": open(f"{base}.txt", "wb"), "offs": [0]}
        w = writers[split]
        b = text.encode("utf-8")
        w["fh"].write(b)
        w["offs"].append(w["offs"][-1] + len(b))
        stats["kept"] += 1
        stats["bytes"] += len(b)
    shards = {}
    for split, w in writers.items():
        w["fh"].close()
        n = len(w["offs"]) - 1
        np.save(f"{w['base']}.offsets.npy", np.asarray(w["offs"], dtype=np.int64))
        np.save(f"{w['base']}.categories.npy", np.full(n, job["category"], dtype=np.int16))
        name = w["base"].name
        shards[split] = {"text": f"{name}.txt", "offsets": f"{name}.offsets.npy",
                         "categories": f"{name}.categories.npy", "documents": n, "bytes": w["offs"][-1]}
    return job["source"], job["id"], shards, stats


def jobs_for(source, raw, chunk_bytes):
    """(sub-source name, path, start, end) per job; big jsonl files are cut into byte ranges."""
    root = raw / source
    files = sorted(root.rglob("*.parquet")) + sorted(root.rglob("*.jsonl"))
    out = []
    for p in files:
        rel = p.relative_to(root).parts
        if source == "toucan":
            if rel[0] == "SFT":
                continue
            sub = rel[0]
        elif source == "open-swe-traces":
            sub = rel[1]  # data/<framework>/<model>/...
        elif source == "terminal-corpus":
            sub = rel[0] if rel[0] == "dataset_adapters" else "/".join(rel[1:3])
        else:
            sub = p.stem if p.suffix == ".jsonl" else rel[0]
        if p.suffix == ".jsonl" and p.stat().st_size > chunk_bytes:
            size = p.stat().st_size
            for a in range(0, size, chunk_bytes):
                out.append((sub, p, a, min(size, a + chunk_bytes) - 1))
        else:
            out.append((sub, p, 0, 1 << 62))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="/work/raw")
    ap.add_argument("--out", default="/work/text")
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk-mb", type=int, default=256, help="jsonl byte range per job")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--max-system", type=int, default=2000)
    ap.add_argument("--max-tool-desc", type=int, default=600)
    ap.add_argument("--max-foreign", type=float, default=0.005)
    ap.add_argument("--sample", help="print 2 converted documents of this source and exit")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    cfg = {k: getattr(a, k) for k in ("val_every", "max_system", "max_tool_desc", "max_foreign")}

    if a.sample:
        sub, p, s, e = jobs_for(a.sample, raw, a.chunk_mb << 20)[0]
        c = argparse.Namespace(**cfg, sub=sub)
        adapter, shown, why = ADAPTER.get(a.sample, openai_doc), 0, collections.Counter()
        for row in rows({"path": str(p), "start": s, "end": e}):
            text, reason = adapter(row, c)
            if text is None:
                why[reason] += 1
                continue
            print(f"===== {a.sample} / {sub} ({len(text)} chars)\n{text[:6000]}\n")
            shown += 1
            if shown == 2:
                break
        print("dropped before these:", dict(why))
        return

    cats, jobs = {}, []
    for src in a.sources:
        (out / src).mkdir(parents=True, exist_ok=True)
        for sub, p, s, e in jobs_for(src, raw, a.chunk_mb << 20):
            name = f"{src}/{sub}"
            cats.setdefault(name, len(cats))
            c = dict(cfg, sub=sub, max_foreign=0.05 if src == "openmath-tir" else a.max_foreign)
            jobs.append({"source": src, "id": len([j for j in jobs if j["source"] == src]), "path": str(p),
                         "start": s, "end": e, "category": cats[name], "out": str(out / src), "cfg": c})
    print(f"{len(jobs)} jobs over {len(a.sources)} sources, {a.workers} workers", flush=True)
    t0 = time.time()
    shards = collections.defaultdict(lambda: collections.defaultdict(list))
    stats = collections.defaultdict(collections.Counter)
    with ProcessPoolExecutor(a.workers) as pool:
        futs = [pool.submit(run_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            src, jid, sh, st = f.result()
            for split, s in sh.items():
                shards[src][split].append((jid, s))
            stats[src].update(st)
            if (i + 1) % 20 == 0 or i + 1 == len(futs):
                print(f"  {i + 1}/{len(futs)} jobs, {time.time() - t0:.0f} s", flush=True)

    all_splits = collections.defaultdict(list)
    for src in a.sources:
        splits = {}
        for split, lst in shards[src].items():
            v = [s for _, s in sorted(lst, key=lambda x: x[0])]
            splits[split] = {"documents": sum(x["documents"] for x in v), "bytes": sum(x["bytes"] for x in v),
                             "shards": v}
            all_splits[split] += [dict(x, text=f"../{src}/{x['text']}", offsets=f"../{src}/{x['offsets']}",
                                       categories=f"../{src}/{x['categories']}") for x in v]
        meta = {"source": src, "category_ids": cats, "stats": dict(stats[src]),
                "format": "per shard: UTF-8 text, int64 byte offsets (documents + 1), int16 category per document",
                "splits": splits}
        (out / src / "metadata.json").write_text(json.dumps(meta, indent=1))
        st = stats[src]
        drops = {k: v for k, v in st.items() if k.startswith("drop")}
        print(f"{src:20s} kept {st['kept']:>9,d} docs, {st['bytes'] / 1e9:7.2f} GB; dropped {sum(drops.values()):,d} "
              f"{dict(sorted(drops.items(), key=lambda x: -x[1])[:4])}", flush=True)
    (out / "_all").mkdir(exist_ok=True)
    (out / "_all" / "metadata.json").write_text(json.dumps({
        "source": "agentic (every source)", "category_ids": cats,
        "format": "per shard: UTF-8 text, int64 byte offsets (documents + 1), int16 category per document",
        "splits": {s: {"documents": sum(x["documents"] for x in v), "bytes": sum(x["bytes"] for x in v), "shards": v}
                   for s, v in all_splits.items()}}, indent=1))
    print(f"done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
