"""Tool-call validity on the validation splits: per source, the first N documents with an assistant turn that calls a
tool are cut at that turn's <assistant>, and the model continues them twice: free (it may reason first) and direct
(<think></think> pre-filled). With --until call, generation stops at the first </call> or </assistant>; with
--until session (the default) it goes on through further turns: the model writes the tool results itself (nothing is
executed) and calls again, until it opens a <user> turn, ends the document or reaches the token budget. The first call
is scored:

  call      a <call>...</call> before the turn ends
  parses    name(key=<JSON>, ...)
  declared  the name is one of the prompt's <tools>
  args      every argument is a declared parameter and every required parameter is given
  same      the same tool as the reference trace's call

and every call of the output: calls (how many) and valid (how many have declared tools and valid arguments).

    uv run --no-project --with numpy --with tokenizers python scripts/toolcall_eval.py RUN_DIR \
        [--checkpoint latest] [--lbc build/release/lbc] [--n 10] [--max-prompt 8192] [--until session|call] [--out DIR]
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

SOURCES = ["math", "toucan", "nswe", "oswe", "term", "agentic", "opencode"]
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--checkpoint", default="latest")
ap.add_argument("--lbc", default="build/release/lbc")
ap.add_argument("--data", default="/work/data")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--max-prompt", type=int, default=8192)  # longer prompts are skipped
ap.add_argument("--free-tokens", type=int, default=2048)
ap.add_argument("--direct-tokens", type=int, default=2048)
ap.add_argument("--until", choices=["session", "call"], default="session")
ap.add_argument("--out", default="/tmp/toolcall_eval")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

tok = Tokenizer.from_file(f"{a.data}/{SOURCES[0]}/tokenizer.json")
sid = {t: tok.token_to_id(t) for t in ("<assistant>", "</assistant>", "<call>", "</call>", "<think>", "</think>", "<user>")}
assert all(v is not None for v in sid.values()), sid


def decode(ids):
    return tok.decode(ids, skip_special_tokens=False)


# -- the prompts: cut at the first assistant turn that calls a tool -------------------------------------------------
cases = []
for src in SOURCES:
    v = np.load(f"{a.data}/{src}/validation.npy", mmap_mode="r")
    starts = np.flatnonzero(np.asarray(v[: min(len(v), 50_000_000)]) == 0)
    found = 0
    for i, s in enumerate(starts):
        if found == a.n:
            break
        e = int(starts[i + 1]) if i + 1 < len(starts) else len(v)
        ids = np.asarray(v[int(s):e]).astype(int).tolist()
        for p, t in enumerate(ids):
            if t != sid["<assistant>"]:
                continue
            end = next((q for q in range(p + 1, len(ids)) if ids[q] == sid["</assistant>"]), len(ids))
            if sid["<call>"] not in ids[p + 1:end]:
                continue
            prompt = ids[: p + 1]
            if len(prompt) > a.max_prompt:
                break
            ref = re.search(r"<call>\s*([\w\-.]+)\(", decode(ids[p + 1:end]))
            cases.append({"source": src, "doc": i, "prompt": prompt, "ref": ref[1] if ref else None})
            found += 1
            break
    print(f"{src}: {found} prompts", flush=True)


# -- generation ---------------------------------------------------------------------------------------------------
stop = f"{sid['</call>']},{sid['</assistant>']}" if a.until == "call" else str(sid["<user>"])


def generate(mode, suffix, tokens):
    ids_f, gen_f = out / f"{mode}.ids", out / f"{mode}.gen"
    ids_f.write_text("".join(" ".join(map(str, c["prompt"] + suffix)) + "\n" for c in cases))
    subprocess.run([a.lbc, "generate", a.run, "--checkpoint", a.checkpoint, "--ids", str(ids_f), "--out", str(gen_f),
                    "--tokens", str(tokens), "--temperature", "0.7", "--top-k", "40", "--seed", "1",
                    "--stop", stop], check=True)
    return [[int(x) for x in line.split()] for line in gen_f.read_text().splitlines()]


# -- scoring ------------------------------------------------------------------------------------------------------
def declared_tools(prompt_text):
    m = re.search(r"<tools>(.*?)</tools>", prompt_text, re.S)
    tools = {}
    for line in (m[1] if m else "").split("\n"):
        if not (mm := re.match(r"^([\w\-.]+)\((.*?)\)(?::|$)", line)):
            continue
        params = re.findall(r"(?:^|, )([A-Za-z_][\w\-]*)(\??): ", mm[2])
        tools[mm[1]] = {"all": {p for p, _ in params}, "required": {p for p, opt in params if not opt}}
    return tools


def parse_call(body):
    m = re.match(r"\s*([\w\-.]+)\((.*)\)\s*$", body, re.S)
    if not m:
        return None
    name, rest, args, dec = m[1], m[2].strip(), {}, json.JSONDecoder()
    while rest:
        km = re.match(r"([A-Za-z_][\w\-]*)=", rest)
        if not km:
            return None
        try:
            val, n = dec.raw_decode(rest[km.end():])
        except json.JSONDecodeError:
            return None
        args[km[1]] = val
        rest = rest[km.end() + n:].lstrip()
        if rest.startswith(","):
            rest = rest[1:].lstrip()
        elif rest:
            return None
    return name, args


def valid(pc, tools):
    return bool(pc) and pc[0] in tools and set(pc[1]) <= tools[pc[0]]["all"] and tools[pc[0]]["required"] <= set(pc[1])


def score(case, gen):
    text = decode(gen)
    r = {"call": False, "parses": False, "declared": False, "args": False, "same": False, "tokens": len(gen),
         "calls": 0, "valid": 0}
    tools = declared_tools(decode(case["prompt"]))
    calls = [parse_call(c) for c in re.findall(r"<call>(.*?)</call>", text, re.S)]
    r["calls"], r["valid"] = len(calls), sum(valid(pc, tools) for pc in calls)
    m = re.search(r"<call>(.*?)</call>", text, re.S)
    if not m or "</assistant>" in text[: m.start()]:
        return r, text
    r["call"] = True
    pc = parse_call(m[1])
    if not pc:
        return r, text
    r["parses"] = True
    r["same"] = pc[0] == case["ref"]
    r["declared"] = pc[0] in tools
    r["args"] = valid(pc, tools)
    return r, text


modes = {"free": ([], a.free_tokens), "direct": ([sid["<think>"], sid["</think>"]], a.direct_tokens)}
rows, samples = {}, []
for mode, (suffix, tokens) in modes.items():
    gens = generate(mode, suffix, tokens)
    for case, gen in zip(cases, gens):
        r, text = score(case, gen)
        rows.setdefault((case["source"], mode), []).append(r)
        samples.append({"source": case["source"], "doc": case["doc"], "mode": mode, "ref": case["ref"], **r,
                        "prompt_tail": decode(case["prompt"][-60:]), "generated": text})
(out / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))

keys = ["call", "parses", "declared", "args", "same"]
print(f"\n{'source':9s} {'mode':7s} {'n':>3s} " + " ".join(f"{k:>8s}" for k in keys)
      + f" {'tokens':>7s} {'calls':>6s} {'valid':>6s}")
for mode in modes:
    tot = []
    for src in SOURCES:
        rs = rows.get((src, mode), [])
        tot += rs
        if rs:
            print(f"{src:9s} {mode:7s} {len(rs):3d} " + " ".join(f"{np.mean([r[k] for r in rs]):8.0%}" for k in keys)
                  + f" {np.mean([r['tokens'] for r in rs]):7.0f} {np.mean([r['calls'] for r in rs]):6.1f}"
                  + f" {sum(r['valid'] for r in rs) / max(1, sum(r['calls'] for r in rs)):6.0%}")
    print(f"{'all':9s} {mode:7s} {len(tot):3d} " + " ".join(f"{np.mean([r[k] for r in tot]):8.0%}" for k in keys)
          + f" {np.mean([r['tokens'] for r in tot]):7.0f} {np.mean([r['calls'] for r in tot]):6.1f}"
          + f" {sum(r['valid'] for r in tot) / max(1, sum(r['calls'] for r in tot)):6.0%}\n")
print(f"samples: {out / 'samples.jsonl'}")
