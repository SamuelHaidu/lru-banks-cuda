"""Builds `kodcode-agent-bpe32768`: KodCode-V1 (Xu et al. 2025, arXiv 2503.02951; KodCode/KodCode-V1, CC BY-NC 4.0)
coding problems as agent conversations with file and shell tools, every tool result produced by really running it.

    uv run --with pyarrow python datasets/build_kodcode_agent.py --download          # the 15 parquet shards (2.5 GB)
    uv run --with pyarrow python datasets/build_kodcode_agent.py --inspect           # field formats, distributions
    python datasets/build_kodcode_agent.py [--problems 40000] [--workers 4] [--samples 3]   # build (root, see below)

Problems: easy (`gpt_difficulty`), `instruct` and `complete` styles (pytest tests importing from `solution`), not the
`Docs` / `Package` subsets (library APIs), `benchmark_similarity` < 0.8 (away from HumanEval / MBPP), short
(question <= 1200, solution <= 900, test <= 1500 characters). Split by base problem id (`Leetcode_11` for
`Leetcode_11_I` and `Leetcode_11_C`, so the two styles of one problem stay in one split): validation 500 and test 1000
base problems, the rest train (`--problems` of them sampled).

Tools (`TOOLS`, the format of `build_agent_shell.py`: `render_call`, results in `<result>`):
    write_file(path, content) -> "Wrote N bytes to PATH"; read_file(path) -> the text; edit_file(path, old, new) ->
    "Edited PATH" (old must appear exactly once); run(command) -> its output, stdout and stderr, the last 40 lines.

Conversations (category):
- solve: test_solution.py is in the workspace; the user asks for the function in solution.py; the assistant reads the
  tests first (half of them), writes solution.py (KodCode's verified solution), runs `python -m pytest -q --tb=short
  test_solution.py`, then reports.
- fix: the workspace also holds a solution.py with one mutated line (an operator, comparison, boolean or integer
  constant changed) that really fails a test; the user asks to fix it; the assistant runs the tests, reads the file,
  edits the line back (`edit_file`), reruns, reports. The bug always comes from the workspace: the assistant never
  writes wrong code.
Every result is real: the files are written to a fresh directory and the command runs there as the unprivileged user
`--user` (default kodsbx: it cannot read /root or write /workspace) with time / memory / file-size limits, from a
`python` that is `--python` (/usr/bin/python3 with pytest). Problems whose solution does not pass are dropped.

Output in `--out` (default datasets/kodcode-agent-bpe32768): `{split}.npy` (uint16), `{split}.categories.npy`
(start, length, category), `{split}.docs.jsonl` (id, kind, subset, style, text), `{split}.problems.jsonl` (the
problems: id, style, question, solution, test, function names; for evaluation), `metadata.json` (`loss_spans` = the
assistant spans, per category documents / tokens / drops, avg_chars_per_token over the assistant spans).
"""

import argparse
import ast
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_agent_shell import render_call  # noqa: E402

REPO = "KodCode/KodCode-V1"
COLUMNS = ["style", "subset", "question_id", "question", "solution", "test", "test_info", "gpt_difficulty",
           "gpt_pass_percentage", "benchmark_similarity", "benchmark_instruction", "benchmark_task_id"]
TOOLS = ("<tools>\n"
         "write_file(path: str, content: str): create or overwrite a file with the given content\n"
         "read_file(path: str): return the contents of a file\n"
         "edit_file(path: str, old: str, new: str): replace the text old, which must appear exactly once, with new\n"
         "run(command: str): run a shell command in the workspace and return its output (the last 40 lines)\n"
         "</tools>")
TEST_CMD = "python -m pytest -q --tb=short test_solution.py"
KINDS = ["solve", "fix"]
MAX_LINES = 40
# one-token mutations of a code line: (pattern, replacement); applied to lines without string literals or comments
MUTATIONS = [(r" <= ", " < "), (r" < ", " <= "), (r" >= ", " > "), (r" > ", " >= "), (r" == ", " != "),
             (r" != ", " == "), (r" \+ ", " - "), (r" - ", " + "), (r" \* ", " + "), (r" and ", " or "),
             (r" or ", " and "), (r"\bTrue\b", "False"), (r"\bFalse\b", "True"), (r"\+= 1\b", "-= 1"),
             (r"\b0\b", "1"), (r"\b1\b", "2"), (r"\b1\b", "0"), (r"\bmax\(", "min("), (r"\bmin\(", "max(")]


def download(raw: Path) -> None:
    info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}"))
    files = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/parquet"))["default"]
    raw.mkdir(parents=True, exist_ok=True)
    for split, urls in files.items():
        for i, url in enumerate(urls):
            dst = raw / f"{split}-{i:02d}.parquet"
            if not dst.exists():
                urllib.request.urlretrieve(url, dst)
            print(f"  {split} {i}: {dst.stat().st_size / 1e6:.1f} MB", flush=True)
    (raw / "revision.json").write_text(json.dumps({"repo": REPO, "sha": info.get("sha"),
                                                   "license": (info.get("cardData") or {}).get("license")}, indent=2))


def load(raw: Path, split: str = "train", only=None):
    """The rows of a split; `only`: a filter applied file by file (memory: the whole split as Python objects is
    several GB, and forked workers copy what they touch)."""
    import pyarrow.parquet as pq

    rows = []
    for f in sorted(raw.glob(f"{split}-*.parquet")):
        part = pq.read_table(f, columns=COLUMNS).to_pylist()
        rows += [r for r in part if only is None or only(r)]
        del part
    return rows


def inspect(raw: Path) -> None:
    from collections import Counter

    import numpy as np

    rows = load(raw)
    print(f"{len(rows):,} rows")
    for k in ("style", "subset", "gpt_difficulty"):
        print(k, Counter(r[k] for r in rows).most_common())
    print("style x difficulty", Counter((r["style"], r["gpt_difficulty"]) for r in rows).most_common())
    sim = np.array([r["benchmark_similarity"] or 0 for r in rows], float)
    print("benchmark_similarity percentiles", {q: round(float(np.percentile(sim, q)), 3) for q in (50, 90, 99, 99.9)})
    easy = [r for r in rows if r["gpt_difficulty"] == "easy" and r["style"] in ("instruct", "complete")]
    for k in ("question", "solution", "test"):
        n = np.array([len(r[k]) for r in easy])
        print(f"easy {k} chars p50 {np.percentile(n, 50):.0f} p90 {np.percentile(n, 90):.0f}")


def keep(r: dict) -> bool:
    return (r["gpt_difficulty"] == "easy" and r["style"] in ("instruct", "complete")
            and r["subset"] not in ("Docs", "Package") and (r["benchmark_similarity"] or 0) < 0.8
            and len(r["question"]) <= 1200 and len(r["solution"]) <= 900 and len(r["test"]) <= 1500
            and "from solution import" in r["test"])


def base_id(qid: str) -> str:
    return qid.rsplit("_", 1)[0]


# -- the sandbox ------------------------------------------------------------------------------------------------------

SANDBOX = {"uid": None, "gid": None, "bin": None, "root": None}


def setup_sandbox(user: str, python: str) -> None:
    import pwd

    pw = pwd.getpwnam(user)
    SANDBOX.update(uid=pw.pw_uid, gid=pw.pw_gid)
    SANDBOX["root"] = tempfile.mkdtemp(prefix="kodcode_sbx_")
    os.chmod(SANDBOX["root"], 0o755)
    SANDBOX["bin"] = os.path.join(SANDBOX["root"], "bin")
    os.mkdir(SANDBOX["bin"])
    os.symlink(python, os.path.join(SANDBOX["bin"], "python"))


def _limits():
    import resource

    os.setsid()
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
    os.setgroups([])
    os.setgid(SANDBOX["gid"])
    os.setuid(SANDBOX["uid"])


def run(files: dict, command: str, timeout: float = 15.0):
    """Runs `command` with bash in a fresh directory holding `files`, as the sandbox user: (output, timed out)."""
    d = tempfile.mkdtemp(dir=SANDBOX["root"])
    try:
        for name, text in files.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        for x in [d] + [os.path.join(d, n) for n in files]:
            os.chown(x, SANDBOX["uid"], SANDBOX["gid"])
        env = {"PATH": f"{SANDBOX['bin']}:/usr/bin:/bin", "HOME": d, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
        p = subprocess.Popen(["bash", "-c", command], cwd=d, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, preexec_fn=_limits)
        try:
            out, _ = p.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
            out, _ = p.communicate()
            timed_out = True
        text = out.decode("utf-8", "replace").replace("\r", "").rstrip("\n")
        lines = text.split("\n")
        return "\n".join(lines[-MAX_LINES:]), timed_out
    finally:
        shutil.rmtree(d, ignore_errors=True)


# -- conversations ----------------------------------------------------------------------------------------------------

PASSED = re.compile(r"^(\d+) passed in [\d.]+s$", re.M)


def call(name: str, lead: str, result: str, **args: str) -> str:
    return (f"<assistant>{lead + chr(10) if lead else ''}<call>{render_call(name, args)}</call></assistant>"
            f"<result>{result}</result>")


def ask(r: dict, rng: random.Random, kind: str) -> str:
    q = r["question"].strip()
    task = (f"Complete this function:\n\n```python\n{q}\n```" if r["style"] == "complete" else q)
    if kind == "solve":
        tail = rng.choice(["Write the solution in solution.py. The tests are in test_solution.py.",
                           "Put your solution in solution.py; test_solution.py has the tests.",
                           "Save it as solution.py and make sure the tests in test_solution.py pass."])
        return f"{task}\n\n{tail}"
    head = rng.choice(["The tests in test_solution.py fail. Please fix solution.py.",
                       "solution.py has a bug: test_solution.py does not pass. Can you fix it?",
                       "Something is wrong in solution.py, the tests fail. Fix it please."])
    return f"{head}\n\nThe task:\n{task}"


def mutants(solution: str, rng: random.Random):
    """(buggy solution, buggy line, original line) candidates: one mutation on one code line (no docstrings, strings,
    comments), compiling, the buggy line unique in the file."""
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return []
    skip = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and \
                isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            skip.update(range(body[0].lineno, body[0].end_lineno + 1))
    lines = solution.split("\n")
    cands = []
    for i, line in enumerate(lines):
        s = line.strip()
        if i + 1 in skip or not s or s.startswith(("#", "def ", "class ", "import ", "from ")) or "'" in s or '"' in s \
                or "#" in s:
            continue
        for pat, rep in MUTATIONS:
            for m in re.finditer(pat, line):
                new = line[:m.start()] + re.sub(pat, rep, line[m.start():m.end()]) + line[m.end():]
                if new == line:
                    continue
                buggy = "\n".join(lines[:i] + [new] + lines[i + 1:])
                if buggy.count(new.strip()) != 1:
                    continue
                try:
                    compile(buggy, "solution.py", "exec")
                except SyntaxError:
                    continue
                cands.append((buggy, new.strip(), line.strip()))
    rng.shuffle(cands)
    return cands


def solved(output: str) -> int:
    """Number of tests if the pytest output is a clean pass, else 0."""
    m = PASSED.search(output)
    if not m or "failed" in output or "error" in output.lower():
        return 0
    return int(m.group(1))


def convert(r: dict, kind: str, seed: int):
    """(text, None) or (None, drop reason)."""
    rng = random.Random(seed)
    sol, test = r["solution"].strip("\n") + "\n", r["test"].strip("\n") + "\n"
    out, to = run({"solution.py": sol, "test_solution.py": test}, TEST_CMD)
    n = solved(out)
    if to or not n:
        return None, "solution does not pass"
    names = [t["function_name"] for t in (r["test_info"] or []) if t.get("function_name")]
    fn = f"`{names[0]}`" if names else "the function"
    parts = [TOOLS, f"<user>{ask(r, rng, kind)}</user>"]
    tests = "test" if n == 1 else "tests"
    if kind == "solve":
        if rng.random() < 0.5:
            parts.append(call("read_file", rng.choice(["Let me look at the tests first.", "First, the tests.", ""]),
                              test.rstrip("\n"), path="test_solution.py"))
        parts.append(call("write_file", rng.choice(["I'll write the solution.", "Writing solution.py.", ""]),
                          f"Wrote {len(sol.encode())} bytes to solution.py", path="solution.py", content=sol))
        parts.append(call("run", rng.choice(["Now the tests.", "Let me run the tests.", ""]), out, command=TEST_CMD))
        parts.append(f"<assistant>{rng.choice([f'All {n} {tests} pass.', f'Done: {n} {tests} passed.'])} "
                     f"solution.py defines {fn}.</assistant>")
        return "".join(parts), None
    for buggy, bad, good in mutants(sol, rng)[:4]:
        bout, bto = run({"solution.py": buggy, "test_solution.py": test}, TEST_CMD)
        if bto or "failed" not in bout or "SyntaxError" in bout:
            continue
        parts.append(call("run", rng.choice(["Let me run the tests to see what fails.", "First, the failing tests.", ""]),
                          bout, command=TEST_CMD))
        parts.append(call("read_file", rng.choice(["Now the code.", "Let me read solution.py.", ""]),
                          buggy.rstrip("\n"), path="solution.py"))
        parts.append(call("edit_file", rng.choice([f"The bug is `{bad}`: it should be `{good}`.",
                                                   f"`{bad}` is wrong, fixing it.", ""]),
                          "Edited solution.py", path="solution.py", old=bad, new=good))
        parts.append(call("run", rng.choice(["Running the tests again.", "Checking.", ""]), out, command=TEST_CMD))
        parts.append(f"<assistant>Fixed: `{bad}` is now `{good}`. All {n} {tests} pass.</assistant>")
        return "".join(parts), None
    return None, "no failing mutation"


def work(job):
    split, r, kind, seed = job
    text, why = convert(r, kind, seed)
    return split, r, kind, text, why


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=str(ROOT / "datasets" / "raw" / "kodcode"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--tokenizer", default="pretrain-mix-bpe32768")
    p.add_argument("--name", default="kodcode-agent-bpe32768")
    p.add_argument("--out", default=None)
    p.add_argument("--problems", type=int, default=40000, help="train problems (sampled)")
    p.add_argument("--fix-share", type=float, default=0.4, help="share of problems made into fix conversations")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--user", default="kodsbx", help="the sandbox user")
    p.add_argument("--python", default="/usr/bin/python3", help="python with pytest, for the sandbox")
    p.add_argument("--samples", type=int, default=0, help="print N conversations of each kind and exit")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    raw = Path(a.raw)
    if a.download:
        download(raw)
    if a.inspect:
        inspect(raw)
    if a.download or a.inspect:
        return
    setup_sandbox(a.user, a.python)
    rows = [{k: r[k] for k in ("style", "subset", "question_id", "question", "solution", "test", "test_info")}
            for r in load(raw, only=keep)]
    bases = sorted({base_id(r["question_id"]) for r in rows})
    rng = random.Random(a.seed)
    rng.shuffle(bases)
    split_of = {b: "validation" for b in bases[:500]} | {b: "test" for b in bases[500:1500]}
    by = {"train": [], "validation": [], "test": []}
    for r in rows:
        by[split_of.get(base_id(r["question_id"]), "train")].append(r)
    rng.shuffle(by["train"])
    by["train"] = by["train"][:a.problems]
    print(f"{len(rows):,} problems kept, {len(bases):,} base problems; " +
          ", ".join(f"{s} {len(v):,}" for s, v in by.items()), flush=True)
    jobs = []
    for sp, rs in by.items():
        for r in rs:
            seed = zlib.crc32(r["question_id"].encode()) ^ a.seed
            jobs.append((sp, r, "fix" if random.Random(seed).random() < a.fix_share else "solve", seed))
    if a.samples:
        for kind in KINDS:
            done = 0
            for j in jobs:
                if j[2] != kind:
                    continue
                text, why = convert(j[1], kind, j[3])
                if text:
                    print(f"===== {kind} {j[1]['question_id']}\n{text}\n")
                    done += 1
                if done >= a.samples:
                    break
        return

    from multiprocessing import Pool

    os.environ["RAYON_NUM_THREADS"] = "2"
    import numpy as np
    from tokenizers import Tokenizer

    t0 = time.time()
    docs = {s: [] for s in by}
    drops = {}
    with Pool(a.workers) as pool:
        for i, (sp, r, kind, text, why) in enumerate(pool.imap_unordered(work, jobs, chunksize=16)):
            if text is None:
                drops.setdefault(sp, {}).setdefault(f"{kind}: {why}", 0)
                drops[sp][f"{kind}: {why}"] += 1
            else:
                docs[sp].append((r, kind, text))
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1:,}/{len(jobs):,} problems, {time.time() - t0:.0f} s", flush=True)
    shutil.rmtree(SANDBOX["root"], ignore_errors=True)

    tok = Tokenizer.from_file(str(ROOT / "tokenizers" / a.tokenizer / "tokenizer.json"))
    bos, eos, a_open, a_close = (tok.token_to_id(t) for t in ("<bos>", "<eos>", "<assistant>", "</assistant>"))
    out = Path(a.out) if a.out else ROOT / "datasets" / a.name
    out.mkdir(parents=True, exist_ok=True)
    cat_ids = {k: i for i, k in enumerate(KINDS)}
    meta = {"name": a.name, "tokenizer": a.tokenizer, "vocabulary_size": tok.get_vocab_size(), "doc_token_id": bos,
            "bos_token_id": bos, "eos_token_id": eos, "dtype": "uint16", "loss_spans": [[a_open, a_close]],
            "category_index": "{split}.categories.npy", "category_ids": cat_ids,
            "source": json.loads((raw / "revision.json").read_text()), "drops": drops}
    cats = {k: {} for k in KINDS}
    span_chars = span_tokens = 0
    for sp, ds in docs.items():
        ds.sort(key=lambda d: d[0]["question_id"])
        random.Random(a.seed).shuffle(ds)
        encs = tok.encode_batch(["<bos>" + d[2] + "<eos>" for d in ds], add_special_tokens=False)
        lens = np.array([len(e.ids) for e in encs], np.int64)
        tokens = np.concatenate([np.asarray(e.ids, np.uint32) for e in encs])
        assert tokens.max() < 65536
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        np.save(out / f"{sp}.npy", tokens.astype(np.uint16))
        np.save(out / f"{sp}.categories.npy", np.stack([starts, lens, [cat_ids[d[1]] for d in ds]], 1))
        with open(out / f"{sp}.docs.jsonl", "w") as fh, open(out / f"{sp}.problems.jsonl", "w") as fp:
            for r, kind, text in ds:
                fh.write(json.dumps({"id": r["question_id"], "kind": kind, "subset": r["subset"], "style": r["style"],
                                     "text": text}, ensure_ascii=False) + "\n")
                fp.write(json.dumps({"id": r["question_id"], "kind": kind, "subset": r["subset"], "style": r["style"],
                                     "question": r["question"], "solution": r["solution"], "test": r["test"],
                                     "functions": [t["function_name"] for t in (r["test_info"] or [])]},
                                    ensure_ascii=False) + "\n")
        for k in KINDS:
            sel = [i for i, d in enumerate(ds) if d[1] == k]
            cats[k][sp] = {"documents": len(sel), "tokens": int(lens[sel].sum()) if sel else 0}
        if sp == "validation":
            for e in encs:
                inside = False
                for t, (o0, o1) in zip(e.ids, e.offsets):
                    if inside:
                        span_tokens += 1
                        span_chars += 0 if t == a_close else o1 - o0
                    inside = (inside or t == a_open) and t != a_close
        assert all(tokens[s0] == bos and tokens[s0 + n - 1] == eos for s0, n in zip(starts, lens))
        bad = sum(tok.decode(e.ids[1:-1], skip_special_tokens=False) != d[2] for e, d in zip(encs, ds))
        assert bad == 0, f"{sp}: {bad} documents do not decode back"
        meta[sp] = {"file": f"{sp}.npy", "tokens": int(len(tokens)), "documents": len(ds),
                    "doc_tokens": {k: int(np.percentile(lens, q)) for k, q in (("p50", 50), ("p90", 90), ("p99", 99))}
                    | {"max": int(lens.max())}}
        print(f"{sp}: {len(ds):,} documents, {len(tokens):,} tokens, p50 / p99 {meta[sp]['doc_tokens']['p50']} / "
              f"{meta[sp]['doc_tokens']['p99']}", flush=True)
    meta["avg_chars_per_token"] = round(span_chars / max(span_tokens, 1), 4)
    meta["categories"] = cats
    meta["notes"] = ("KodCode-V1 easy instruct/complete problems as solve / fix conversations with write_file, "
                     "read_file, edit_file, run; every result executed in a sandbox (datasets/build_kodcode_agent.py).")
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"drops: {json.dumps(drops)}\ndone in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    sys.exit(main())
