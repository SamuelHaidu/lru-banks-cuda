"""Tool-calling test of an lbc (C++) model on held-out kodcode-agent problems: the model gets the tools header and the
user turn of a test document, generates one assistant turn at a time (`lbc generate --stop </call>,</assistant>`),
every call is parsed and really executed in a sandbox (a fresh directory with test_solution.py, plus the buggy
solution.py for `fix`), the result is appended as `<result>`, and the loop goes on until a final answer, a malformed
call, or the turn limit. At the end the tests are run on the final workspace.

Runs on the pod (needs the GPU and root, for the sandbox user):
    python3 lbc_agent_eval.py --lbc /workspace/lru-banks-cuda/build/release/lbc --run RUN --checkpoint PATH \
        --data /root/data/kodcode-agent-bpe32768 --tokenizer /root/data/tokenizer.json --n 20 --out /root/agent_eval

Safety: `run` only executes `python ... .py` / `python -m pytest ...` / `pytest ...` with plain arguments (no shell
operators, no paths outside the workspace); everything executes as an unprivileged user with CPU, memory, file size
and time limits, in a directory under /tmp.
"""

import argparse
import ast
import json
import os
import pwd
import random
import re
import resource
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from tokenizers import Tokenizer

CALL_END, ASSIST_END = 9, 7
TEST_CMD = "python -m pytest -q --tb=short test_solution.py"
MAX_LINES = 40
SAFE_RUN = re.compile(r"^(python3? -m pytest|pytest|python3? [\w.-]+\.py)( [\w.=:-]+)*$")
SAFE_PATH = re.compile(r"^[\w.-]+$")
PASSED = re.compile(r"^(\d+) passed in [\d.]+s$", re.M)
SANDBOX = {}


def parse_call(text):
    """`name(key="str", ...)` as in build_agent_shell.parse_call: (name, args) or ValueError."""
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as e:
        raise ValueError(f"not a call: {e.msg}") from e
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args:
        raise ValueError("not a keyword call")
    args = {}
    for kw in node.keywords:
        if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, str):
            raise ValueError("arguments must be strings")
        args[kw.arg] = kw.value.value
    return node.func.id, args


def _limits():
    os.setsid()
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
    os.setgroups([])
    os.setgid(SANDBOX["gid"])
    os.setuid(SANDBOX["uid"])


def shell(d, command, timeout=15.0):
    env = {"PATH": f"{SANDBOX['bin']}:/usr/bin:/bin", "HOME": d, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
    for x in [d] + [os.path.join(d, n) for n in os.listdir(d)]:
        os.chown(x, SANDBOX["uid"], SANDBOX["gid"])
    p = subprocess.Popen(["bash", "-c", command], cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         preexec_fn=_limits)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        out, _ = p.communicate()
    text = out.decode("utf-8", "replace").replace("\r", "").rstrip("\n")
    return "\n".join(text.split("\n")[-MAX_LINES:])


def execute(d, name, args):
    """The tool result, and an error tag (None when the call was valid)."""
    want = {"write_file": {"path", "content"}, "read_file": {"path"}, "edit_file": {"path", "old", "new"},
            "run": {"command"}}
    if name not in want:
        return f"unknown tool {name}", "unknown tool"
    if set(args) != want[name]:
        return f"{name}: wrong arguments {sorted(args)}", "wrong arguments"
    if "path" in args and not SAFE_PATH.match(args["path"]):
        return f"{name}: invalid path {args['path']}", "bad path"
    p = os.path.join(d, args.get("path", "x"))
    if name == "write_file":
        with open(p, "w") as f:
            f.write(args["content"])
        return f"Wrote {len(args['content'].encode())} bytes to {args['path']}", None
    if name == "read_file":
        if not os.path.exists(p):
            return f"read_file: {args['path']}: No such file or directory", "missing file"
        return open(p).read().rstrip("\n"), None
    if name == "edit_file":
        if not os.path.exists(p):
            return f"edit_file: {args['path']}: No such file or directory", "missing file"
        text = open(p).read()
        n = text.count(args["old"])
        if n != 1:
            return f"edit_file: old text found {n} times in {args['path']}", "edit mismatch"
        with open(p, "w") as f:
            f.write(text.replace(args["old"], args["new"]))
        return f"Edited {args['path']}", None
    cmd = args["command"].strip()
    if not SAFE_RUN.match(cmd):
        return f"bash: command not allowed in this evaluation: {cmd}", "command not allowed"
    return shell(d, cmd), None


def workspace(doc, prob):
    """Initial files: the tests, plus the buggy solution.py of a fix document (its read_file result)."""
    files = {"test_solution.py": prob["test"].strip("\n") + "\n"}
    if doc["kind"] == "fix":
        m = re.search(r'<call>read_file\(path="solution\.py"\)</call></assistant><result>(.*?)</result>', doc["text"],
                      re.S)
        if not m:
            return None
        files["solution.py"] = m.group(1) + "\n"
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lbc", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="latest")
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=20, help="problems per kind (solve, fix)")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=450, help="max tokens per assistant turn")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--user", default="kodsbx")
    ap.add_argument("--bin", default="/opt/agentenv/bin", help="python with pytest for the sandbox")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pw = pwd.getpwnam(a.user)
    SANDBOX.update(uid=pw.pw_uid, gid=pw.pw_gid, bin=a.bin)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)

    probs = {}
    for line in open(Path(a.data) / "test.problems.jsonl"):
        r = json.loads(line)
        probs[(r["id"], r["kind"])] = r
    docs = [json.loads(l) for l in open(Path(a.data) / "test.docs.jsonl")]
    random.Random(a.seed).shuffle(docs)
    convs = []
    for kind in ("solve", "fix"):
        for doc in [d for d in docs if d["kind"] == kind]:
            if sum(c["kind"] == kind for c in convs) == a.n:
                break
            files = workspace(doc, probs[(doc["id"], kind)])
            if files is None:
                continue
            d = tempfile.mkdtemp(prefix="agent_", dir="/tmp")
            os.chmod(d, 0o755)
            for k, v in files.items():
                Path(d, k).write_text(v)
            head = doc["text"][:doc["text"].index("</user>") + len("</user>")]
            convs.append(dict(id=doc["id"], kind=kind, dir=d, text=head + "<assistant>", calls=[], end=None))

    t0 = time.time()
    for turn in range(a.turns):
        live = [c for c in convs if c["end"] is None]
        if not live:
            break
        ids_file, gen_file = out / "prompts.ids", out / "gen.ids"
        with open(ids_file, "w") as f:
            for c in live:
                f.write(" ".join(["0"] + [str(i) for i in tok.encode(c["text"]).ids]) + "\n")
        subprocess.run([a.lbc, "generate", a.run, "--checkpoint", a.checkpoint, "--ids", str(ids_file), "--out",
                        str(gen_file), "--tokens", str(a.tokens), "--temperature", str(a.temperature),
                        "--seed", str(a.seed + turn), "--stop", f"{CALL_END},{ASSIST_END}"], check=True,
                       stdout=subprocess.DEVNULL)
        gens = [[int(x) for x in l.split()] for l in open(gen_file).read().split("\n")[:len(live)]]
        for c, g in zip(live, gens):
            text = tok.decode(g, skip_special_tokens=False)
            c["text"] += text
            if g and g[-1] == ASSIST_END:
                c["end"] = "answer"
                continue
            if not g or g[-1] != CALL_END:
                c["end"] = "no stop (turn too long or document end)"
                continue
            if "<call>" not in text:
                c["end"] = "</call> without <call>"
                continue
            src = text[text.rindex("<call>") + len("<call>"):-len("</call>")]
            try:
                name, args = parse_call(src)
            except ValueError as e:
                c["calls"].append(dict(src=src, error=f"malformed: {e}"))
                c["end"] = "malformed call"
                continue
            result, err = execute(c["dir"], name, args)
            c["calls"].append(dict(name=name, args=args, error=err, result=result[:300]))
            c["text"] += f"</assistant><result>{result}</result><assistant>"
        print(f"turn {turn + 1}: {len(live)} live, {time.time() - t0:.0f}s", flush=True)
    for c in convs:
        c["end"] = c["end"] or "turn limit"
        has = os.path.exists(os.path.join(c["dir"], "solution.py"))
        res = shell(c["dir"], TEST_CMD) if has else ""
        m = PASSED.search(res)
        c["final_tests"] = "pass" if m and "failed" not in res and "error" not in res.lower() else \
            ("no solution.py" if not has else "fail")
        shutil.rmtree(c["dir"], ignore_errors=True)
    with open(out / "conversations.jsonl", "w") as f:
        for c in convs:
            f.write(json.dumps(c) + "\n")

    calls = [x for c in convs for x in c["calls"]]
    print(f"\n{len(convs)} conversations, {len(calls)} calls, {time.time() - t0:.0f}s")
    print(f"calls parsed: {sum('malformed' not in (x['error'] or '') for x in calls)}/{len(calls)}, "
          f"valid (executed without a tool error): {sum(x['error'] is None for x in calls)}/{len(calls)}")
    errs = {}
    for x in calls:
        if x["error"]:
            errs[x["error"].split(":")[0]] = errs.get(x["error"].split(":")[0], 0) + 1
    print("tool errors:", errs)
    for kind in ("solve", "fix"):
        cs = [c for c in convs if c["kind"] == kind]
        ends, fin = {}, {}
        for c in cs:
            ends[c["end"]] = ends.get(c["end"], 0) + 1
            fin[c["final_tests"]] = fin.get(c["final_tests"], 0) + 1
        seq = {}
        for c in cs:
            s = " > ".join(x.get("name", "?") for x in c["calls"])
            seq[s] = seq.get(s, 0) + 1
        print(f"\n{kind}: {len(cs)}; endings {ends}; final tests {fin}")
        for s, k in sorted(seq.items(), key=lambda kv: -kv[1])[:6]:
            print(f"   {k:3d} x  {s or '(no call)'}")


if __name__ == "__main__":
    main()
