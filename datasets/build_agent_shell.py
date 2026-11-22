"""Builds `agent-shell-bytes`: synthetic tool-use conversations, byte-level tokens plus special tags.

Every document is a conversation with no system prompt: the tools available in it, then user and assistant turns,
tool calls and their results:

    <bos><tools>
    run(command: str): run a shell command in the workspace
    read_file(path: str): return the contents of a file
    </tools><user>is mango in fruits.txt?</user><assistant>Let me check.
    <call>run(command="grep -n mango fruits.txt")</call></assistant><result>4:mango</result><assistant>Yes, mango is
    on line 4 of fruits.txt.</assistant><eos>

Tokens: 12 special tags (ids 0-11, below), then the 256 bytes (id 12 + byte), vocabulary 268. Everything that is not
a tag is UTF-8 bytes, so any text can be written and read back exactly.

The tool results are not invented: every call is executed. `run` goes to a simulated shell over a virtual
filesystem (ls, cat, echo, printf, wc, grep, head, tail, sort, uniq, mkdir, touch, rm, mv, cp, pwd, cd, python3,
pipes, `&&`, `;`, `>` / `>>`, globs) whose output follows GNU coreutils / bash for the forms generated here (checked
against a real shell by `--check-shell`); `python` / `python3 file.py` really execute the (generated) code;
`calculator` evaluates arithmetic. Each document gets its own random subset of the tools (plus tools it never needs),
and a task picks one of the ways to solve it that the available tools allow, so the model has to read the tool
list. Some workspaces hold files the assistant has to discover; some tasks need no tool at all.

Task families (each with several phrasings and several solution paths): files (create, append, folders, rename,
copy, delete, sort, recover from a wrong file name), inspection (count lines, show, list, head / tail), lookup
(is X in a file, which file has X, a config value, a CSV row, the same questions about a list pasted in the message,
follow-up questions answered from what is already in the context), math (calculator, word problems, small sums
done directly, Python for loops / primes / factorials), code (write a function to a file and run its tests, fix a
bug, run a script), and tool-less replies when no listed tool can do the job.

Usage:
    .venv/bin/python datasets/build_agent_shell.py [--train-docs 50000] [--validation-docs 1000] [--seed 0]
    .venv/bin/python datasets/build_agent_shell.py --samples 5        # print a few documents, write nothing
    .venv/bin/python datasets/build_agent_shell.py --check-shell      # compare the simulator with real bash
"""

import argparse
import ast
import contextlib
import fnmatch
import io
import json
import operator
import posixpath
import re
import shlex
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

NAME = "agent-shell-bytes"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / NAME
TOKENIZER_DIR = ROOT / "tokenizers" / NAME

SPECIAL = ["<bos>", "<eos>", "<tools>", "</tools>", "<user>", "</user>", "<assistant>", "</assistant>",
           "<call>", "</call>", "<result>", "</result>"]
BOS, EOS = 0, 1
BYTE0 = len(SPECIAL)
VOCAB_SIZE = BYTE0 + 256
HOME = "/home/user"

# ---------------------------------------------------------------------------------------------------------------
# vocabulary for content


FRUITS = "apple banana cherry grape lemon mango melon orange peach pear plum kiwi lime papaya apricot fig guava".split()
VEGETABLES = "carrot potato onion garlic pepper tomato lettuce spinach broccoli cabbage celery cucumber pumpkin".split()
ANIMALS = "cat dog horse rabbit tiger lion zebra panda koala otter wolf fox bear eagle owl shark whale frog".split()
COLORS = "red blue green yellow purple orange pink brown black white gray silver gold teal".split()
CITIES = ("Lisbon Paris Berlin Madrid Rome Oslo Vienna Prague Dublin Athens Tokyo Lima Quito Cairo Nairobi Denver "
          "Austin Boston Toronto Sydney").split()
FIRST = ("Ana Ben Carla David Elena Felix Gina Hugo Iris Jonas Karen Leo Maria Nina Omar Paula Rafael Sara Tomas "
         "Vera Yuri Zoe Bruno Clara Diego Laura Marco Rita").split()
ITEMS = "milk eggs bread butter cheese rice pasta coffee tea sugar flour salt beans soap juice yogurt honey".split()
TASKS = ["buy milk", "call mom", "pay rent", "fix the bike", "water the plants", "book a dentist", "clean the kitchen",
         "send the report", "walk the dog", "renew passport", "read a book", "update the website", "backup photos"]
WORDS = FRUITS + VEGETABLES + ANIMALS + COLORS + ITEMS
SYL = "ra ro xo li ta mi ve ko lu ne si da pa zo be fi gu ny el or".split()


def invented(rng) -> str:
    """A name the model cannot know in advance (it has to copy it)."""
    return "".join(rng.choice(SYL) for _ in range(int(rng.integers(2, 4)))).capitalize()


def code_id(rng) -> str:
    return f"{''.join(rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'), 2))}-{int(rng.integers(1000, 10000))}"


# ---------------------------------------------------------------------------------------------------------------
# tools: the simulated shell, file tools, calculator, python


def run_python(src: str, filename: str) -> str:
    """Executes generated code; stdout, then a traceback in CPython's layout if it raised."""
    buf = io.StringIO()
    try:
        code = compile(src, filename, "exec")
        with contextlib.redirect_stdout(buf):
            exec(code, {"__name__": "__main__"})
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001  (the program's own error is the tool's output)
        lines = src.splitlines()
        tb = ["Traceback (most recent call last):"]
        for f in traceback.extract_tb(e.__traceback__):
            if f.filename == filename:
                tb.append(f'  File "{filename}", line {f.lineno}, in {f.name}')
                if 0 < f.lineno <= len(lines):
                    tb.append("    " + lines[f.lineno - 1].strip())
        last = traceback.format_exception_only(e)[-1].rstrip("\n")
        return buf.getvalue() + "\n".join(tb + [last]) + "\n"


_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def fmt_number(x) -> str:
    if isinstance(x, float):
        if x.is_integer() and abs(x) < 1e15:
            return str(int(x))
        return f"{round(x, 6):.6f}".rstrip("0").rstrip(".")
    return str(x)


def calculator(expression: str) -> str:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(b) > 100:
                raise ValueError("exponent too large")
            return _BINOPS[type(node.op)](a, b)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -ev(node.operand) if isinstance(node.op, ast.USub) else ev(node.operand)
        raise ValueError("unsupported expression")
    try:
        return fmt_number(ev(ast.parse(expression.replace("^", "**"), mode="eval")))
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, SyntaxError):
        return f"Error: cannot evaluate {expression!r}"


class Shell:
    """A bash-like shell over a virtual filesystem: `files` maps a path relative to HOME to its text, `dirs` holds
    the directories ("" is HOME)."""

    def __init__(self, files: Optional[Dict[str, str]] = None):
        self.files: Dict[str, str] = {}
        self.dirs = {""}
        self.cwd = ""
        for p, text in (files or {}).items():
            self.write(p, text)

    # -- filesystem --
    def norm(self, p: str) -> str:
        if p.startswith(HOME):
            p = p[len(HOME):].lstrip("/")
        elif p.startswith("~"):
            p = p[1:].lstrip("/")
        else:
            p = posixpath.join(self.cwd, p)
        p = posixpath.normpath(p) if p else ""
        return "" if p in (".", "") else p

    def write(self, p: str, text: str, append: bool = False) -> None:
        p = self.norm(p)
        parent = posixpath.dirname(p)
        while parent and parent not in self.dirs:  # (write_file creates parents; the shell checks before)
            self.dirs.add(parent)
            parent = posixpath.dirname(parent)
        self.files[p] = (self.files.get(p, "") if append else "") + text

    def is_dir(self, p: str) -> bool:
        return self.norm(p) in self.dirs

    def exists(self, p: str) -> bool:
        q = self.norm(p)
        return q in self.files or q in self.dirs

    def children(self, d: str) -> List[str]:
        d = self.norm(d)
        out = [posixpath.basename(p) for p in list(self.files) + list(self.dirs) if p and posixpath.dirname(p) == d]
        return sorted(set(out))

    def glob(self, pattern: str) -> List[str]:
        d, base = posixpath.split(pattern)
        names = [n for n in self.children(d) if fnmatch.fnmatchcase(n, base) and not n.startswith(".")]
        return [posixpath.join(d, n) if d else n for n in names]

    # -- commands --
    def run(self, command: str) -> str:
        """Runs a command line; stdout and stderr together, as a terminal shows them (no trailing newline)."""
        lex = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lex.whitespace_split = True
        try:
            toks = list(lex)
        except ValueError as e:
            return f"bash: syntax error: {e}"
        out, i, ok = [], 0, True
        seq: List[Tuple[str, List[str]]] = []
        cur: List[str] = []
        for t in toks:  # split into (connector, tokens) by && and ;
            if t in ("&&", ";"):
                seq.append((t, cur))
                cur = []
            else:
                cur.append(t)
        seq.append(("end", cur))
        prev_conn = ";"
        for conn, part in seq:
            if part and (prev_conn == ";" or ok):
                text, ok = self.pipeline(part)
                out.append(text)
            prev_conn = conn
        return "".join(out).rstrip("\n")

    def pipeline(self, toks: List[str]) -> Tuple[str, bool]:
        stages, cur = [], []
        for t in toks:
            if t == "|":
                stages.append(cur)
                cur = []
            else:
                cur.append(t)
        stages.append(cur)
        stdin, err, ok = None, "", True
        for k, st in enumerate(stages):
            redirect, mode = None, None
            if len(st) >= 2 and st[-2] in (">", ">>"):
                redirect, mode, st = st[-1], st[-2], st[:-2]
            o, e, ok = self.simple(st, stdin)
            err += e
            if redirect is not None:
                q = self.norm(redirect)
                if posixpath.dirname(q) not in self.dirs:
                    err += f"bash: {redirect}: No such file or directory\n"
                    ok, o = False, ""
                elif q in self.dirs:
                    err += f"bash: {redirect}: Is a directory\n"
                    ok, o = False, ""
                else:
                    self.write(redirect, o, append=mode == ">>")
                    o = ""
            stdin = o
        return err + (stdin or ""), ok

    def expand(self, args: List[str]) -> List[str]:
        out = []
        for a in args:
            if any(c in a for c in "*?["):
                hits = self.glob(a)
                out.extend(hits if hits else [a])
            else:
                out.append(a)
        return out

    def read(self, name: str, cmd: str) -> Tuple[Optional[str], str]:
        q = self.norm(name)
        if q in self.dirs:
            return None, f"{cmd}: {name}: Is a directory\n"
        if q not in self.files:
            return None, f"{cmd}: {name}: No such file or directory\n"
        return self.files[q], ""

    def simple(self, st: List[str], stdin: Optional[str]) -> Tuple[str, str, bool]:
        if not st:
            return "", "", True
        cmd, args = st[0], self.expand(st[1:])
        fn = getattr(self, "c_" + cmd.replace("python3", "python"), None)
        if fn is None:
            return "", f"bash: {cmd}: command not found\n", False
        return fn(args, stdin)

    @staticmethod
    def flags(args: List[str], allowed: str) -> Tuple[set, List[str]]:
        fl, rest = set(), []
        for a in args:
            if a.startswith("-") and len(a) > 1 and all(c in allowed for c in a[1:]):
                fl.update(a[1:])
            else:
                rest.append(a)
        return fl, rest

    def inputs(self, files: List[str], stdin: Optional[str], cmd: str):
        """(name, text) for each file argument (or stdin), and the errors."""
        if not files:
            return [(None, stdin or "")], ""
        out, err = [], ""
        for f in files:
            text, e = self.read(f, cmd)
            err += e
            if text is not None:
                out.append((f, text))
        return out, err

    def c_ls(self, args, stdin):
        fl, paths = self.flags(args, "a1")
        if not paths:
            names = self.children(self.cwd)
            if "a" in fl:
                names = sorted([".", ".."] + names)
            else:
                names = [n for n in names if not n.startswith(".")]
            return "".join(n + "\n" for n in names), "", True
        files, dirs, err = [], [], ""
        for p in paths:
            q = self.norm(p)
            if q in self.files:
                files.append(p)
            elif q in self.dirs:
                dirs.append(p)
            else:
                err += f"ls: cannot access '{p}': No such file or directory\n"
        out = "".join(f + "\n" for f in sorted(files))
        for k, d in enumerate(sorted(dirs)):
            names = [n for n in self.children(d) if not n.startswith(".") or "a" in fl]
            if len(paths) > 1:
                out += ("\n" if out else "") + f"{d}:\n"
            out += "".join(n + "\n" for n in names)
        return out, err, not err

    def c_cat(self, args, stdin):
        _, files = self.flags(args, "")
        got, err = self.inputs(files, stdin, "cat")
        return "".join(t for _, t in got), err, not err

    def c_echo(self, args, stdin):
        nl = "\n"
        if args and args[0] == "-n":
            nl, args = "", args[1:]
        return " ".join(args) + nl, "", True

    def c_printf(self, args, stdin):
        if not args:
            return "", "printf: usage: printf format [arguments]\n", False
        fmt = args[0].replace("\\n", "\n").replace("\\t", "\t")
        vals = args[1:]
        n = fmt.count("%s")
        if not vals or n == 0:
            return fmt.replace("%s", ""), "", True
        out = ""
        for k in range(0, len(vals), n):
            group = vals[k:k + n] + [""] * (n - len(vals[k:k + n]))
            piece = fmt
            for v in group:
                piece = piece.replace("%s", v, 1)
            out += piece
        return out, "", True

    def c_wc(self, args, stdin):
        fl, files = self.flags(args, "lwc")
        got, err = self.inputs(files, stdin, "wc")
        out = ""
        for name, t in got:
            counts = {"l": t.count("\n"), "w": len(t.split()), "c": len(t.encode())}
            sel = [c for c in "lwc" if c in fl] or ["l", "w", "c"]
            nums = [counts[c] for c in sel]
            if len(nums) == 1 and name is not None:
                out += f"{nums[0]} {name}\n"
            elif len(nums) == 1:
                out += f"{nums[0]}\n"
            else:
                w = max(len(str(v)) for v in nums) if name is not None else 7
                out += " ".join(f"{v:>{w}}" for v in nums) + (f" {name}" if name is not None else "") + "\n"
        return out, err, not err

    @staticmethod
    def bre(pattern: str) -> str:
        """A grep basic regular expression as a Python one (+ ? | ( ) { } are literal in BRE)."""
        out = ""
        for ch in pattern:
            out += "\\" + ch if ch in "+?|(){}" else ch
        return out

    def c_grep(self, args, stdin):
        fl, rest = self.flags(args, "nicvlwF")
        if not rest:
            return "", "Usage: grep [OPTION]... PATTERNS [FILE]...\n", False
        pat, files = rest[0], rest[1:]
        rx = re.escape(pat) if "F" in fl else self.bre(pat)
        if "w" in fl:
            rx = r"(?<!\w)(?:" + rx + r")(?!\w)"
        rx = re.compile(rx, re.I if "i" in fl else 0)
        got, err = self.inputs(files, stdin, "grep")
        multi = len(files) > 1
        out, found = "", False
        for name, t in got:
            lines = t.split("\n")
            if lines and lines[-1] == "":
                lines = lines[:-1]
            hits = [(k + 1, l) for k, l in enumerate(lines) if bool(rx.search(l)) != ("v" in fl)]
            found |= bool(hits)
            if "l" in fl:
                out += f"{name}\n" if hits and name else ""
            elif "c" in fl:
                out += (f"{name}:" if multi else "") + f"{len(hits)}\n"
            else:
                for k, l in hits:
                    out += (f"{name}:" if multi else "") + (f"{k}:" if "n" in fl else "") + l + "\n"
        return out, err, found and not err

    def _count_arg(self, args, default=10):
        n, rest, k = default, [], 0
        while k < len(args):
            a = args[k]
            if a == "-n" and k + 1 < len(args):
                n, k = int(args[k + 1]), k + 2
                continue
            if re.fullmatch(r"-n?\d+", a):
                n = int(a.lstrip("-n"))
            else:
                rest.append(a)
            k += 1
        return n, rest

    def _headtail(self, args, stdin, cmd):
        n, files = self._count_arg(args)
        got, err = self.inputs(files, stdin, cmd)
        out = ""
        for k, (name, t) in enumerate(got):
            lines = t.splitlines(keepends=True)
            part = lines[:n] if cmd == "head" else (lines[-n:] if n else [])
            if len(got) > 1:
                out += ("\n" if k else "") + f"==> {name} <==\n"
            out += "".join(part)
        return out, err, not err

    def c_head(self, args, stdin):
        return self._headtail(args, stdin, "head")

    def c_tail(self, args, stdin):
        return self._headtail(args, stdin, "tail")

    def c_sort(self, args, stdin):
        fl, files = self.flags(args, "rnu")
        got, err = self.inputs(files, stdin, "sort")
        lines = [l for _, t in got for l in t.splitlines()]
        if "n" in fl:
            def key(l):
                m = re.match(r"\s*(-?\d+(?:\.\d+)?)", l)
                return (float(m.group(1)) if m else 0.0, l)
            lines.sort(key=key, reverse="r" in fl)
        else:
            lines.sort(reverse="r" in fl)
        if "u" in fl:
            lines = list(dict.fromkeys(lines))
        return "".join(l + "\n" for l in lines), err, not err

    def c_uniq(self, args, stdin):
        _, files = self.flags(args, "")
        got, err = self.inputs(files[:1], stdin, "uniq")
        out, prev = [], None
        for l in (got[0][1] if got else "").splitlines():
            if l != prev:
                out.append(l)
            prev = l
        return "".join(l + "\n" for l in out), err, not err

    def c_mkdir(self, args, stdin):
        fl, paths = self.flags(args, "p")
        err = ""
        for p in paths:
            q = self.norm(p)
            if q in self.dirs or q in self.files:
                if "p" not in fl or q in self.files:
                    err += f"mkdir: cannot create directory '{p}': File exists\n"
                continue
            parent = posixpath.dirname(q)
            if parent not in self.dirs and "p" not in fl:
                err += f"mkdir: cannot create directory '{p}': No such file or directory\n"
                continue
            while q and q not in self.dirs:
                self.dirs.add(q)
                q = posixpath.dirname(q)
        return "", err, not err

    def c_touch(self, args, stdin):
        err = ""
        for p in args:
            q = self.norm(p)
            if posixpath.dirname(q) not in self.dirs:
                err += f"touch: cannot touch '{p}': No such file or directory\n"
            elif q not in self.files and q not in self.dirs:
                self.files[q] = ""
        return "", err, not err

    def c_rm(self, args, stdin):
        fl, paths = self.flags(args, "rfR")
        err = ""
        for p in paths:
            q = self.norm(p)
            if q in self.files:
                del self.files[q]
            elif q in self.dirs and q:
                if not fl & {"r", "R"}:
                    err += f"rm: cannot remove '{p}': Is a directory\n"
                    continue
                self.dirs = {d for d in self.dirs if d != q and not d.startswith(q + "/")}
                self.files = {f: t for f, t in self.files.items() if not f.startswith(q + "/")}
            elif "f" not in fl:
                err += f"rm: cannot remove '{p}': No such file or directory\n"
        return "", err, not err

    def _move(self, args, cmd):
        fl, paths = self.flags(args, "r")
        if len(paths) < 2:
            return "", f"{cmd}: missing destination file operand after '{paths[0] if paths else ''}'\n", False
        *srcs, dst = paths
        err = ""
        for s in srcs:
            q = self.norm(s)
            if q not in self.files and q not in self.dirs:
                err += f"{cmd}: cannot stat '{s}': No such file or directory\n"
                continue
            target = posixpath.join(dst, posixpath.basename(q)) if self.is_dir(dst) else dst
            t = self.norm(target)
            if posixpath.dirname(t) not in self.dirs:
                verb = "move" if cmd == "mv" else "create regular file"
                err += f"{cmd}: cannot {verb} '{s}' to '{target}': No such file or directory\n"
                continue
            if q in self.files:
                self.files[t] = self.files[q]
                if cmd == "mv":
                    del self.files[q]
            else:
                if cmd == "cp" and "r" not in fl:
                    err += f"cp: -r not specified; omitting directory '{s}'\n"
                    continue
                for d in [d for d in self.dirs if d == q or d.startswith(q + "/")]:
                    self.dirs.add(t + d[len(q):])
                for f in [f for f in self.files if f.startswith(q + "/")]:
                    self.files[t + f[len(q):]] = self.files[f]
                if cmd == "mv":
                    self.dirs = {d for d in self.dirs if not (d == q or d.startswith(q + "/"))}
                    self.files = {f: x for f, x in self.files.items() if not f.startswith(q + "/")}
        return "", err, not err

    def c_mv(self, args, stdin):
        return self._move(args, "mv")

    def c_cp(self, args, stdin):
        return self._move(args, "cp")

    def c_pwd(self, args, stdin):
        return (HOME + ("/" + self.cwd if self.cwd else "")) + "\n", "", True

    def c_cd(self, args, stdin):
        target = args[0] if args else HOME
        q = self.norm(target)
        if q not in self.dirs:
            return "", f"bash: cd: {target}: No such file or directory\n", False
        self.cwd = q
        return "", "", True

    def c_python(self, args, stdin):
        if not args:
            return "", "", True
        q = self.norm(args[0])
        if q not in self.files:
            return "", f"python3: can't open file '{HOME}/{q}': [Errno 2] No such file or directory\n", False
        out = run_python(self.files[q], f"{HOME}/{q}")
        return out, "", "Traceback" not in out


class Tools:
    """The tools a document can list; each call runs against the document's shell."""

    SPECS = {
        "run": (["command"], ["run a shell command in the workspace and return its output",
                              "execute a bash command", "run a terminal command"]),
        "read_file": (["path"], ["return the contents of a file", "read a text file", "show what a file contains"]),
        "write_file": (["path", "content"], ["create or overwrite a file with the given content",
                                              "write text to a file (replaces it)", "save content to a file"]),
        "calculator": (["expression"], ["evaluate an arithmetic expression", "compute a math expression",
                                        "do exact arithmetic"]),
        "python": (["code"], ["run Python code and return what it prints", "execute a Python snippet",
                              "run a Python program"]),
    }
    FAKE = {  # listed, never needed
        "get_weather": (["city"], ["get the current weather for a city"]),
        "send_email": (["to", "subject", "body"], ["send an email"]),
        "search_web": (["query"], ["search the internet"]),
        "set_timer": (["minutes"], ["start a countdown timer"]),
        "translate": (["text", "language"], ["translate text to another language"]),
    }

    def __init__(self, shell: Shell):
        self.sh = shell

    def call(self, name: str, args: Dict[str, str]) -> str:
        if name == "run":
            return self.sh.run(args["command"])
        if name == "read_file":
            text, err = self.sh.read(args["path"], "read_file")
            return err.strip().replace("read_file: ", "Error: ") if text is None else text.rstrip("\n")
        if name == "write_file":
            q = self.sh.norm(args["path"])
            if q in self.sh.dirs:
                return f"Error: {args['path']} is a directory"
            self.sh.write(args["path"], args["content"])
            return f"Wrote {len(args['content'].encode())} bytes to {args['path']}"
        if name == "calculator":
            return calculator(args["expression"])
        if name == "python":
            return run_python(args["code"], "<string>").rstrip("\n")
        raise KeyError(name)


def render_call(name: str, args: Dict[str, str]) -> str:
    return f"{name}(" + ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items()) + ")"


def parse_call(text: str) -> Tuple[str, Dict[str, str]]:
    """The inverse of `render_call` (for evaluating a model's calls): ValueError if it is not a valid call."""
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as e:
        raise ValueError(f"not a call: {e}") from e
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args:
        raise ValueError("not a keyword call")
    args = {}
    for kw in node.keywords:
        if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, str):
            raise ValueError("arguments must be strings")
        args[kw.arg] = kw.value.value
    return node.func.id, args


# ---------------------------------------------------------------------------------------------------------------
# a conversation


class Doc:
    def __init__(self, rng, tools: List[str], fake: List[str], files: Dict[str, str]):
        self.rng = rng
        self.tools = set(tools)
        self.listed = tools + fake
        rng.shuffle(self.listed)
        self.sh = Shell(files)
        self.exec = Tools(self.sh)
        self.parts: List[str] = []
        self.seen: Dict[str, str] = {}  # file -> its full text as last shown in the conversation
        self.snapshot = None  # the workspace before the first call (for --check-shell)

    def in_context(self, path: str) -> bool:
        """The file's current content is already in the conversation (read or written, unchanged since)."""
        q = self.sh.norm(path)
        return q in self.sh.files and self.seen.get(q) == self.sh.files[q]

    def has(self, *names: str) -> bool:
        return all(n in self.tools for n in names)

    def header(self) -> str:
        lines = []
        for n in self.listed:
            params, descs = Tools.SPECS.get(n) or Tools.FAKE[n]
            lines.append(f"{n}({', '.join(p + ': str' for p in params)}): {self.rng.choice(descs)}")
        return "<tools>\n" + "\n".join(lines) + "\n</tools>"

    def user(self, text: str) -> None:
        self.parts.append(f"<user>{text}</user>")

    def say(self, text: str) -> None:
        self.parts.append(f"<assistant>{text}</assistant>")

    def call(self, name: str, pre: str = "", **args: str) -> str:
        assert name in self.tools, name
        if self.snapshot is None:
            self.snapshot = (dict(self.sh.files), set(self.sh.dirs))
        result = self.exec.call(name, args)
        lead = (pre + "\n") if pre else ""
        self.parts.append(f"<assistant>{lead}<call>{render_call(name, args)}</call></assistant><result>{result}</result>")
        shown = None
        if name in ("read_file", "write_file"):
            shown = args["path"]
        elif name == "run" and re.fullmatch(r"cat \S+", args["command"]):
            shown = args["command"].split()[1]
        if shown is not None and self.sh.norm(shown) in self.sh.files:
            self.seen[self.sh.norm(shown)] = self.sh.files[self.sh.norm(shown)]
        return result

    def text(self) -> str:
        return "<bos>" + self.header() + "".join(self.parts) + "<eos>"


# ---------------------------------------------------------------------------------------------------------------
# phrasing helpers


def ch(rng, options: Sequence):
    return options[int(rng.integers(len(options)))]


def polite(rng, text: str) -> str:
    """Random framing of a request: a prefix, a suffix, casing."""
    if text.endswith("?"):  # a question: no "please" / "can you" in front
        pre = ch(rng, ["", "", "", "", "hey, ", "quick question: ", "so, "])
    else:
        pre = ch(rng, ["", "", "", "please ", "can you ", "could you ", "hey, ", "I need you to ", "quick one: "])
    if pre in ("can you ", "could you "):
        text = text[0].lower() + text[1:]
        text = text.rstrip(".") + "?" if not text.endswith("?") else text
    elif pre:
        text = text[0].lower() + text[1:]
    text = pre + text
    text = text[0].upper() + text[1:] if rng.random() < 0.7 else text[0].lower() + text[1:]
    if rng.random() < 0.15:
        text += ch(rng, [" thanks", " thank you!", " thx", " please"])
    return text


def sq(x: str) -> str:
    """A shell word (quoted only when it has to be)."""
    return shlex.quote(x)


def lead(rng, *options: str) -> str:
    """What the assistant says before a call (often nothing)."""
    return ch(rng, ("", "", "") + options)


def nums_text(xs) -> str:
    return ", ".join(str(x) for x in xs)


def join_words(xs: Sequence[str]) -> str:
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def ordinal(n: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(n, f"{n}th")


# ---------------------------------------------------------------------------------------------------------------
# workspaces: files that exist before the conversation (the assistant has to look)


LIST_KINDS = [("fruits.txt", FRUITS), ("animals.txt", ANIMALS), ("colors.txt", COLORS), ("shopping.txt", ITEMS),
              ("vegetables.txt", VEGETABLES), ("guests.txt", FIRST), ("cities.txt", CITIES), ("todo.txt", TASKS)]


def make_list(rng, pool, lo=4, hi=12) -> List[str]:
    k = int(rng.integers(lo, min(hi, len(pool)) + 1))
    return list(rng.choice(pool, k, replace=False))


def workspace(rng) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for fname, pool in [LIST_KINDS[i] for i in rng.choice(len(LIST_KINDS), int(rng.integers(1, 4)), replace=False)]:
        prefix = ch(rng, ["", "", "", "data/", "notes/"])
        files[prefix + fname] = "".join(x + "\n" for x in make_list(rng, pool))
    if rng.random() < 0.5:
        files[ch(rng, ["config.ini", "settings.conf", "app.env", ".env"])] = "".join(
            f"{k}={v}\n" for k, v in config_pairs(rng))
    if rng.random() < 0.4:
        rows = [(n, int(rng.integers(18, 80)), ch(rng, CITIES)) for n in rng.choice(FIRST, int(rng.integers(3, 8)), replace=False)]
        files[ch(rng, ["people.csv", "users.csv", "team.csv"])] = "name,age,city\n" + "".join(f"{n},{a},{c}\n" for n, a, c in rows)
    if rng.random() < 0.3:
        files[ch(rng, ["app.log", "server.log", "build.log"])] = log_text(rng)
    if rng.random() < 0.3:
        spec = FUNCS[int(rng.integers(len(FUNCS)))]
        files[ch(rng, ["main.py", "script.py", "tool.py"])] = spec.program(rng, ch(rng, spec.impls))
    return files


def config_pairs(rng):
    keys = ["port", "host", "user", "timeout", "retries", "mode", "token", "region", "max_items", "version"]
    vals = {"port": lambda: str(int(rng.integers(1024, 65000))), "host": lambda: ch(rng, ["localhost", "10.0.0.5", "db.internal", "example.org"]),
            "user": lambda: invented(rng).lower(), "timeout": lambda: str(int(rng.integers(5, 120))),
            "retries": lambda: str(int(rng.integers(1, 9))), "mode": lambda: ch(rng, ["debug", "production", "test"]),
            "token": lambda: code_id(rng), "region": lambda: ch(rng, ["eu-west", "us-east", "ap-south"]),
            "max_items": lambda: str(int(rng.integers(10, 1000))), "version": lambda: f"{int(rng.integers(1, 5))}.{int(rng.integers(0, 20))}.{int(rng.integers(0, 10))}"}
    ks = list(rng.choice(keys, int(rng.integers(3, 8)), replace=False))
    return [(k, vals[k]()) for k in ks]


def log_text(rng) -> str:
    msgs = ["server started", "connected to database", "user login", "request timeout", "disk almost full",
            "cache cleared", "job finished", "retrying request", "invalid token", "backup done"]
    out = ""
    n = int(rng.integers(4, 10))
    for minute in sorted(int(m) for m in rng.choice(np.arange(8 * 60, 18 * 60), n, replace=False)):
        level = ch(rng, ["INFO", "INFO", "INFO", "WARN", "ERROR"])
        out += f"{minute // 60:02d}:{minute % 60:02d} {level} {ch(rng, msgs)}\n"
    return out


# ---------------------------------------------------------------------------------------------------------------
# code: functions the assistant writes, runs and fixes


class Func:
    def __init__(self, name, descs, impls, args_fn, bugs=()):
        self.name, self.descs, self.impls, self.args_fn, self.bugs = name, descs, impls, args_fn, bugs

    def tests(self, rng, k=2) -> List[str]:
        return [f"{self.name}({self.args_fn(rng)})" for _ in range(k)]

    def program(self, rng, impl: str, tests: Optional[List[str]] = None) -> str:
        return impl + "\n" + "".join(f"print({t})\n" for t in (tests or self.tests(rng)))


def _ints(rng, k, lo=-20, hi=50):
    return "[" + ", ".join(str(int(v)) for v in rng.integers(lo, hi, k)) + "]"


FUNCS = [
    Func("add", ["adds two numbers", "returns the sum of a and b"],
         ["def add(a, b):\n    return a + b\n"], lambda r: f"{int(r.integers(-50, 100))}, {int(r.integers(-50, 100))}",
         bugs=[("def add(a, b):\n    return a - b\n", "wrong")]),
    Func("multiply", ["multiplies two numbers", "returns a times b"],
         ["def multiply(a, b):\n    return a * b\n"], lambda r: f"{int(r.integers(-12, 30))}, {int(r.integers(-12, 30))}",
         bugs=[("def multiply(a, b):\n    return a * c\n", "crash")]),
    Func("square", ["returns the square of a number", "squares n"],
         ["def square(n):\n    return n * n\n", "def square(n):\n    return n ** 2\n"], lambda r: str(int(r.integers(-20, 40)))),
    Func("is_even", ["tells if a number is even", "returns True when n is even"],
         ["def is_even(n):\n    return n % 2 == 0\n", "def is_even(n):\n    if n % 2 == 0:\n        return True\n    return False\n"],
         lambda r: str(int(r.integers(0, 100))), bugs=[("def is_even(n):\n    return n % 2 == 1\n", "wrong")]),
    Func("maximum", ["returns the largest number in a list", "finds the max of a list"],
         ["def maximum(nums):\n    return max(nums)\n",
          "def maximum(nums):\n    best = nums[0]\n    for x in nums:\n        if x > best:\n            best = x\n    return best\n"],
         lambda r: _ints(r, int(r.integers(3, 7))),
         bugs=[("def maximum(nums):\n    best = nums[0]\n    for x in nums:\n        if x < best:\n            best = x\n    return best\n", "wrong")]),
    Func("total", ["sums a list of numbers", "returns the sum of a list"],
         ["def total(nums):\n    return sum(nums)\n",
          "def total(nums):\n    s = 0\n    for x in nums:\n        s += x\n    return s\n"],
         lambda r: _ints(r, int(r.integers(2, 7)), 0, 30),
         bugs=[("def total(nums):\n    s = 0\n    for x in nums:\n        s += x\n    return sum\n", "wrong"),
               ("def total(nums):\n    s = 0\n    for x in nums:\n        s += x\n    return t\n", "crash")]),
    Func("average", ["returns the mean of a list", "computes the average of numbers"],
         ["def average(nums):\n    return sum(nums) / len(nums)\n"], lambda r: _ints(r, int(r.integers(2, 6)), 0, 20)),
    Func("reverse", ["reverses a string", "returns the string backwards"],
         ["def reverse(s):\n    return s[::-1]\n", "def reverse(s):\n    return ''.join(reversed(s))\n"],
         lambda r: json.dumps(ch(r, WORDS + [invented(r).lower()])), bugs=[("def reverse(s):\n    return s[::1]\n", "wrong")]),
    Func("count_vowels", ["counts the vowels in a string", "returns how many vowels a word has"],
         ["def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n",
          "def count_vowels(s):\n    n = 0\n    for c in s:\n        if c in 'aeiou':\n            n += 1\n    return n\n"],
         lambda r: json.dumps(ch(r, WORDS))),
    Func("is_palindrome", ["checks if a word is a palindrome", "returns True if s reads the same backwards"],
         ["def is_palindrome(s):\n    return s == s[::-1]\n"],
         lambda r: json.dumps(ch(r, ["level", "radar", "noon", "kayak", "civic"] + WORDS))),
    Func("factorial", ["computes n factorial", "returns n!"],
         ["def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n",
          "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n"],
         lambda r: str(int(r.integers(0, 11))),
         bugs=[("def factorial(n):\n    result = 1\n    for i in range(2, n):\n        result *= i\n    return result\n", "wrong")]),
    Func("fib", ["returns the nth Fibonacci number", "computes Fibonacci of n"],
         ["def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
          "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"],
         lambda r: str(int(r.integers(0, 16)))),
    Func("to_fahrenheit", ["converts Celsius to Fahrenheit", "turns a temperature in C into F"],
         ["def to_fahrenheit(c):\n    return c * 9 / 5 + 32\n"], lambda r: str(int(r.integers(-10, 40))),
         bugs=[("def to_fahrenheit(c):\n    return c * 5 / 9 + 32\n", "wrong")]),
    Func("is_prime", ["checks if a number is prime", "returns True for prime numbers"],
         ["def is_prime(n):\n    if n < 2:\n        return False\n    for d in range(2, int(n ** 0.5) + 1):\n        if n % d == 0:\n            return False\n    return True\n"],
         lambda r: str(int(r.integers(1, 60)))),
    Func("count_words", ["counts the words in a sentence", "returns the number of words in s"],
         ["def count_words(s):\n    return len(s.split())\n"],
         lambda r: json.dumps(" ".join(r.choice(WORDS, int(r.integers(2, 7)))))),
    Func("evens", ["keeps only the even numbers of a list", "filters a list down to its even numbers"],
         ["def evens(nums):\n    return [x for x in nums if x % 2 == 0]\n",
          "def evens(nums):\n    out = []\n    for x in nums:\n        if x % 2 == 0:\n            out.append(x)\n    return out\n"],
         lambda r: _ints(r, int(r.integers(3, 8)), 0, 30)),
    Func("shout", ["returns the text in upper case with an exclamation mark", "makes a string uppercase and adds !"],
         ["def shout(s):\n    return s.upper() + '!'\n"], lambda r: json.dumps(ch(r, WORDS)),
         bugs=[("def shout(s):\n    return s.uper() + '!'\n", "crash")]),
]


# ---------------------------------------------------------------------------------------------------------------
# tasks: each returns False if the document's tools allow no way to do it (another task is picked)


def task_create_list(d: Doc) -> bool:
    r = d.rng
    fname, pool = ch(r, LIST_KINDS)
    fname = ch(r, [fname, "list.txt", "notes.txt", f"{invented(r).lower()}.txt"])
    if d.sh.exists(fname):
        return False
    items = make_list(r, pool, 2, 6)
    ask = ch(r, [f"Create a file {fname} with {join_words(items)}, one per line.",
                 f"Make {fname} containing these lines: {nums_text(items)}.",
                 f"Write {join_words(items)} into a new file called {fname}, each on its own line.",
                 f"Save this list to {fname}: {nums_text(items)}",
                 f"I need a file named {fname} that lists {join_words(items)}."])
    d.user(polite(r, ask))
    content = "".join(x + "\n" for x in items)
    ways = []
    if d.has("write_file"):
        ways.append(lambda: d.call("write_file", lead(r, "Sure.", f"Creating {fname}."), path=fname, content=content))
    if d.has("run"):
        ways.append(lambda: d.call("run", lead(r, "On it."), command="printf '" + "\\n".join(items) + "\\n' > " + fname))
        ways.append(lambda: d.call("run", lead(r), command=" && ".join(
            [f"echo {sq(items[0])} > {fname}"] + [f"echo {sq(x)} >> {fname}" for x in items[1:]])))
    if not ways:
        return no_tool(d, "create files")
    ch(r, ways)()
    if d.has("run") and r.random() < 0.4:
        d.call("run", lead(r, "Let me check it.", "Checking:"), command=f"cat {fname}")
    d.say(ch(r, [f"Done. {fname} has {len(items)} lines: {nums_text(items)}.", f"Created {fname} with {len(items)} items.",
                 f"{fname} is ready with {join_words(items)}.", "Done."]))
    return True


def pick_file(d: Doc, pred: Callable[[str, str], bool]) -> Optional[str]:
    cands = [p for p, t in d.sh.files.items() if pred(p, t)]
    return ch(d.rng, sorted(cands)) if cands else None


def is_list(p, t):
    return p.endswith(".txt") and t.count("\n") >= 3


def task_append(d: Doc) -> bool:
    r = d.rng
    f = pick_file(d, is_list)
    if f is None:
        return False
    pool = next((pool for n, pool in LIST_KINDS if f.endswith(n)), WORDS)
    lines = d.sh.files[f].splitlines()
    new = ch(r, [x for x in pool if x not in lines] or [invented(r)])
    d.user(polite(r, ch(r, [f"Add {new} to the end of {f}.", f"Append {new} to {f}.", f"Put {new} at the bottom of {f}.",
                             f"{f} is missing {new}, add it as a new line."])))
    ways = []
    if d.has("run"):
        ways.append(lambda: d.call("run", lead(r, "Appending it."), command=f"echo {sq(new)} >> {f}"))
    if d.has("read_file", "write_file"):
        def rw():
            text = d.call("read_file", lead(r, "First I'll read the file."), path=f)
            d.call("write_file", lead(r, "Now writing it back with the new line."), path=f, content=text + "\n" + new + "\n")
        ways.append(rw)
    if not ways:
        return no_tool(d, "edit files")
    ch(r, ways)()
    if d.has("run") and r.random() < 0.4:
        d.call("run", lead(r, "Checking the end of the file."), command=f"tail -n 2 {f}")
    d.say(ch(r, [f"Added {new} to {f}.", f"{new} is now the last line of {f}.", "Done, appended."]))
    return True


def task_folder(d: Doc) -> bool:
    r = d.rng
    folder = ch(r, ["src", "docs", "backup", "notes", "project", "tmp", invented(r).lower()])
    if d.sh.exists(folder) or not d.has("run"):
        return False
    names = [ch(r, ["a", "b", "main", "readme", "todo", "data", "test", "index"]) for _ in range(int(r.integers(1, 4)))]
    ext = ch(r, [".txt", ".md", ".py"])
    names = list(dict.fromkeys(n + ext for n in names))
    d.user(polite(r, ch(r, [f"Make a folder {folder} with empty files {join_words(names)} inside.",
                             f"Create the directory {folder} and add {join_words(names)} to it.",
                             f"Set up {folder}/ containing {nums_text(names)}."])))
    paths = " ".join(f"{folder}/{n}" for n in names)
    if r.random() < 0.6:
        d.call("run", lead(r, "Creating them."), command=f"mkdir {folder} && touch {paths}")
    else:
        d.call("run", lead(r, "First the folder."), command=f"mkdir -p {folder}")
        d.call("run", lead(r, "Now the files."), command=f"touch {paths}")
    out = d.call("run", lead(r, "Let me verify."), command=f"ls {folder}")
    d.say(ch(r, [f"Created {folder} with {join_words(out.split())}.", f"{folder} now contains: {nums_text(out.split())}.", "All set."]))
    return True


def task_count_lines(d: Doc) -> bool:
    r = d.rng
    f = pick_file(d, lambda p, t: t.count("\n") >= 2)
    if f is None:
        return False
    n = d.sh.files[f].count("\n")
    d.user(polite(r, ch(r, [f"How many lines are in {f}?", f"Count the lines of {f}.", f"how long is {f}, in lines?",
                             f"What's the line count of {f}?", f"How many entries does {f} have?"])))
    ways = []
    if d.has("run"):
        ways.append(lambda: d.call("run", lead(r, "Let me count."), command=f"wc -l {f}"))
        ways.append(lambda: d.call("run", lead(r), command=f"cat {f} | wc -l"))
    if d.has("read_file") and n <= 8:
        ways.append(lambda: d.call("read_file", lead(r, "Let me look at it."), path=f))
    if not ways:  # (a long file and only read_file: counting by eye is not offered, pick another task)
        return False if d.has("read_file") else no_tool(d, "read files")
    ch(r, ways)()
    d.say(ch(r, [f"{f} has {n} lines.", f"It has {n} lines.", f"There are {n} lines in {f}.", f"{n}."]))
    return True


def task_show(d: Doc) -> bool:
    r = d.rng
    f = pick_file(d, lambda p, t: 0 < t.count("\n") <= 8)
    if f is None:
        return False
    d.user(polite(r, ch(r, [f"What's in {f}?", f"Show me {f}.", f"Print the contents of {f}.", f"open {f}",
                             f"What does {f} say?"])))
    if d.has("read_file") and (not d.has("run") or r.random() < 0.5):
        text = d.call("read_file", lead(r), path=f)
    elif d.has("run"):
        text = d.call("run", lead(r), command=f"cat {f}")
    else:
        return no_tool(d, "read files")
    d.say(ch(r, [f"{f} contains:\n{text}", f"Here it is:\n{text}", f"It has {text.count(chr(10)) + 1} lines:\n{text}"]))
    return True


def task_list_dir(d: Doc) -> bool:
    r = d.rng
    if not d.has("run"):
        return False
    d.user(polite(r, ch(r, ["What files are here?", "List the files in the workspace.", "What's in this folder?",
                             "ls please", "Which files do I have?"])))
    out = d.call("run", lead(r, "Listing them."), command=ch(r, ["ls", "ls", "ls -1"]))
    names = out.split()
    d.say(ch(r, [f"There are {len(names)} entries: {nums_text(names)}.", f"You have: {nums_text(names)}.",
                 f"The workspace contains {join_words(names)}."]) if names else "The folder is empty.")
    return True


def task_headtail(d: Doc) -> bool:
    r = d.rng
    f = pick_file(d, lambda p, t: t.count("\n") >= 4)
    if f is None or not d.has("run"):
        return False
    n = int(r.integers(1, 4))
    first = r.random() < 0.5
    word = "first" if first else "last"
    d.user(polite(r, ch(r, [f"Show the {word} {n} lines of {f}.", f"What are the {word} {n} lines in {f}?",
                             f"Give me the {word} {n} entries of {f}."])))
    out = d.call("run", lead(r), command=f"{'head' if first else 'tail'} -n {n} {f}")
    d.say(ch(r, [f"The {word} {n} lines are:\n{out}", out]))
    return True


def task_contains(d: Doc) -> bool:
    """Is X in the file: grep (the tool looks) or read (the model looks: an attention test)."""
    r = d.rng
    f = pick_file(d, is_list)
    if f is None:
        return False
    lines = d.sh.files[f].splitlines()
    pool = next((pool for n, pool in LIST_KINDS if f.endswith(n)), WORDS)
    absent = [x for x in pool if x not in lines]
    present = r.random() < 0.5 or not absent
    x = ch(r, lines) if present else ch(r, absent)
    ask = ch(r, [f"Is {x} in {f}?", f"Does {f} contain {x}?", f"Check whether {x} appears in {f}.",
                 f"Can you find {x} in {f}?", f"is {x} on the list in {f}?", f"Look for {x} in {f}."])
    d.user(polite(r, ask))
    ways = []
    if d.in_context(f):
        ways.append("context")
    if d.has("run"):
        ways += ["grep", "grepn"]
    if d.has("read_file"):
        ways.append("read")
    if not ways:
        return no_tool(d, "read files")
    way = "context" if "context" in ways and r.random() < 0.75 else ch(r, ways)
    k = lines.index(x) + 1 if present else 0
    if way in ("grep", "grepn"):  # grep matches substrings: the answer follows its output
        out = d.call("run", lead(r, "Searching.", "Let me search the file."), command=f"grep {'-n ' if way == 'grepn' else ''}{sq(x)} {f}")
        present = bool(out)
        k = int(out.split(":", 1)[0]) if present and way == "grepn" else 0
    elif way == "read":
        d.call("read_file", lead(r, "Let me read it."), path=f)
    if present:
        d.say(ch(r, [f"Yes, {x} is in {f}" + (f" (line {k})." if k else "."),
                     "Yes, it's there" + (f" on line {k}." if k else "."), "Yes."]))
    else:
        d.say(ch(r, [f"No, {x} is not in {f}.", f"No, {f} doesn't contain {x}.", "No, it's not there."]))
    return True


def task_followup(d: Doc) -> bool:
    """A question about a list file the conversation already shows (read, cat or written, unchanged since):
    answered from the context, no tool (the attention test)."""
    r = d.rng
    cands = sorted(p for p in d.seen if d.in_context(p) and d.sh.files[p].count("\n") >= 2
                   and not p.endswith((".py", ".csv", ".log")))
    if not cands:
        return False
    f = ch(r, cands)
    lines = d.sh.files[f].splitlines()
    pool = next((pool for n, pool in LIST_KINDS if f.endswith(n)), WORDS)
    q = int(r.integers(4))
    if q == 0:
        absent = [x for x in pool if x not in lines]
        present = r.random() < 0.5 or not absent
        x = ch(r, lines) if present else ch(r, absent)
        it = len(cands) == 1  # "it" only when one file is shown
        d.user(polite(r, ch(r, [f"And is {x} in it?", f"does it have {x}?"] if it and r.random() < 0.5 else
                                 [f"What about {x}, is it in {f}?", f"Is {x} in {f} too?", f"and {x}? is it in {f}?"])))
        d.say(ch(r, [f"Yes, {x} is on line {lines.index(x) + 1}.", "Yes."]) if present else ch(r, [f"No, {x} isn't in {f}.", "No."]))
    elif q == 1:
        k = int(r.integers(1, len(lines) + 1))
        d.user(polite(r, ch(r, [f"What's on line {k} of {f}?", f"Which entry is {ordinal(k)} in {f}?"]
                                 + ([f"and line {k}?"] if len(cands) == 1 else []))))
        d.say(ch(r, [f"Line {k} is {lines[k - 1]}.", lines[k - 1], f"The {ordinal(k)} entry is {lines[k - 1]}."]))
    elif q == 2:
        d.user(polite(r, ch(r, [f"How many entries were in {f}?", f"count the items in {f}"]
                                 + (["How many lines did it have?"] if len(cands) == 1 else []))))
        d.say(ch(r, [f"{f} has {len(lines)} lines.", f"{len(lines)}."]))
    else:
        d.user(polite(r, ch(r, [f"What was the last entry in {f}?", f"And the first one in {f}?"])))
        last = "last" in d.parts[-1]
        x = lines[-1] if last else lines[0]
        d.say(ch(r, [f"The {'last' if last else 'first'} entry is {x}.", x]))
    return True


def task_which_file(d: Doc) -> bool:
    r = d.rng
    lists = sorted(p for p, t in d.sh.files.items() if is_list(p, t))
    if len(lists) < 2 or not d.has("run"):
        return False
    f = ch(r, lists)
    x = ch(r, d.sh.files[f].splitlines())
    d.user(polite(r, ch(r, [f"Which file has {x}?", f"Where is {x} listed?", f"Find the file that mentions {x}.",
                             f"I wrote {x} somewhere, which file?"])))
    dirs = {posixpath.dirname(p) for p in lists}
    if len(dirs) == 1 and r.random() < 0.6:
        dd = dirs.pop()
        out = d.call("run", lead(r, "Searching all the text files."), command=f"grep -l {sq(x)} {(dd + '/') if dd else ''}*.txt")
    else:
        out = d.call("run", lead(r, "Searching."), command=f"grep -l {sq(x)} " + " ".join(lists))
    owners = out.split()
    d.say(ch(r, [f"{x} is in {join_words(owners)}.", f"It's in {join_words(owners)}.", f"Found it in {join_words(owners)}."]))
    return True


def task_config(d: Doc) -> bool:
    r = d.rng
    f = pick_file(d, lambda p, t: p.endswith((".ini", ".conf", ".env")) and "=" in t)
    if f is None:
        return False
    pairs = [l.split("=", 1) for l in d.sh.files[f].splitlines()]
    k, v = ch(r, pairs)
    d.user(polite(r, ch(r, [f"What is the {k} in {f}?", f"Which {k} is set in {f}?", f"Look up {k} in {f}.",
                             f"what's {k} set to?", f"Tell me the value of {k} from {f}."])))
    ways = []
    if d.in_context(f):
        ways.append("context")
    if d.has("run"):
        ways.append("grep")
    if d.has("read_file"):
        ways.append("read")
    if not ways:
        return no_tool(d, "read files")
    way = "context" if "context" in ways and r.random() < 0.75 else ch(r, ways)
    if way == "grep":
        d.call("run", lead(r, "Looking it up."), command=f"grep {k} {f}")
    elif way == "read":
        d.call("read_file", lead(r, "Reading the config."), path=f)
    d.say(ch(r, [f"{k} is {v}.", f"The {k} in {f} is {v}.", f"It's set to {v}.", v]))
    return True


def task_csv(d: Doc) -> bool:
    r = d.rng
    f = pick_file(d, lambda p, t: p.endswith(".csv"))
    if f is None:
        return False
    rows = [l.split(",") for l in d.sh.files[f].splitlines()[1:]]
    name, age, city = ch(r, rows)
    q = int(r.integers(3))
    ask = [ch(r, [f"How old is {name}?", f"What's {name}'s age in {f}?"]),
           ch(r, [f"Where does {name} live?", f"Which city is {name} in, according to {f}?"]),
           ch(r, [f"Who in {f} lives in {city}?", f"List the people from {city} in {f}."])][q]
    d.user(polite(r, ask))
    ways = []
    if d.in_context(f):
        ways.append("context")
    if d.has("run"):
        ways.append("grep")
    if d.has("read_file"):
        ways.append("read")
    if not ways:
        return no_tool(d, "read files")
    way = "context" if "context" in ways and r.random() < 0.75 else ch(r, ways)
    if way == "grep":
        d.call("run", lead(r, "Searching the CSV."), command=f"grep {sq(city if q == 2 else name)} {f}")
    elif way == "read":
        d.call("read_file", lead(r), path=f)
    if q == 0:
        d.say(ch(r, [f"{name} is {age}.", f"{name} is {age} years old."]))
    elif q == 1:
        d.say(ch(r, [f"{name} lives in {city}.", f"{city}."]))
    else:
        who = [n for n, _, c in rows if c == city]
        d.say(f"{join_words(who)} " + ("lives" if len(who) == 1 else "live") + f" in {city}.")
    return True


def task_inline_lookup(d: Doc) -> bool:
    """The data is in the message: no tool needed (attention over the user turn)."""
    r = d.rng
    if r.random() < 0.5:
        _, pool = ch(r, LIST_KINDS)
        items = make_list(r, pool, 5, 14)
        absent = [x for x in pool if x not in items]
        present = r.random() < 0.5 or not absent
        x = ch(r, items) if present else ch(r, absent)
        d.user(polite(r, ch(r, [f"Here is my list: {nums_text(items)}. Is {x} on it?",
                                 f"{nums_text(items)}\nDoes this list include {x}?",
                                 f"Check if {x} is in: {nums_text(items)}"])))
        d.say(ch(r, [f"Yes, {x} is on the list (item {items.index(x) + 1}).", "Yes."]) if present
              else ch(r, [f"No, {x} is not on the list.", "No."]))
    else:
        people = [invented(r) for _ in range(int(r.integers(3, 9)))]
        people = list(dict.fromkeys(people))
        codes = {p: code_id(r) for p in people}
        p = ch(r, people)
        table = "\n".join(f"{q}: {codes[q]}" for q in people)
        d.user(polite(r, ch(r, [f"Codes:\n{table}\nWhat is {p}'s code?", f"{table}\nWhich code belongs to {p}?",
                                 f"Here are the access codes:\n{table}\nGive me the one for {p}."])))
        d.say(ch(r, [f"{p}'s code is {codes[p]}.", codes[p], f"The code for {p} is {codes[p]}."]))
    return True


def task_rename(d: Doc) -> bool:
    r = d.rng
    f = pick_file(d, lambda p, t: True)
    if f is None or not d.has("run"):
        return False
    base, ext = posixpath.splitext(f)
    new = ch(r, [base + "_old" + ext, base + "_backup" + ext, posixpath.join(posixpath.dirname(f), invented(r).lower() + ext)])
    copy = r.random() < 0.4
    if copy:
        d.user(polite(r, ch(r, [f"Make a copy of {f} called {new}.", f"Copy {f} to {new}.", f"Duplicate {f} as {new}."])))
        d.call("run", lead(r, "Copying."), command=f"cp {f} {new}")
    else:
        d.user(polite(r, ch(r, [f"Rename {f} to {new}.", f"Move {f} to {new}.", f"Change the name of {f} to {new}."])))
        d.call("run", lead(r, "Renaming."), command=f"mv {f} {new}")
    if r.random() < 0.5:
        d.call("run", lead(r, "Checking."), command=f"ls {posixpath.dirname(f)}".strip())
    d.say(ch(r, [f"Done, {'copied' if copy else 'renamed'} {f} to {new}.", "Done."]))
    return True


def task_delete(d: Doc) -> bool:
    r = d.rng
    if not d.has("run"):
        return False
    logs = sorted(p for p in d.sh.files if p.endswith(".log") and "/" not in p)
    if logs and r.random() < 0.5:
        d.user(polite(r, ch(r, ["Delete all the log files.", "Remove every .log file here.", "Clean up the logs."])))
        d.call("run", lead(r, "Removing them."), command="rm *.log")
        out = d.call("run", lead(r, "Checking what's left."), command="ls")
        d.say(f"Deleted {join_words(logs)}." + (f" Remaining: {nums_text(out.split())}." if out else " The folder is now empty."))
        return True
    f = pick_file(d, lambda p, t: True)
    if f is None:
        return False
    d.user(polite(r, ch(r, [f"Delete {f}.", f"Remove {f}, I don't need it.", f"Get rid of {f}."])))
    d.call("run", lead(r, "Deleting it."), command=f"rm {f}")
    d.say(ch(r, [f"Deleted {f}.", f"{f} is gone.", "Done."]))
    return True


def task_wrong_name(d: Doc) -> bool:
    """The user misnames a file: error, look around, retry."""
    r = d.rng
    f = pick_file(d, lambda p, t: "/" not in p and 0 < t.count("\n") <= 8)
    if f is None or not d.has("run"):
        return False
    base, ext = posixpath.splitext(f)
    wrong = ch(r, [base + "s" + ext, base[:-1] + ext if len(base) > 3 else base + "_1" + ext, base + ".md"])
    if d.sh.exists(wrong):
        return False
    d.user(polite(r, ch(r, [f"Show me {wrong}.", f"What's in {wrong}?", f"Print {wrong}."])))
    d.call("run", lead(r), command=f"cat {wrong}")
    d.call("run", lead(r, f"{wrong} doesn't exist. Let me see what's here.", "That file isn't there, checking the folder."), command="ls")
    text = d.call("run", lead(r, f"You probably meant {f}."), command=f"cat {f}")
    d.say(ch(r, [f"There is no {wrong}, but {f} has:\n{text}", f"I couldn't find {wrong}; {f} contains:\n{text}"]))
    return True


def task_sort(d: Doc) -> bool:
    r = d.rng
    f = pick_file(d, is_list)
    if f is None or not d.has("run"):
        return False
    rev = r.random() < 0.3
    dst = ch(r, [f, posixpath.splitext(f)[0] + "_sorted.txt"])
    flag = "-r " if rev else ""
    order = "in reverse order" if rev else "alphabetically"
    if dst == f:
        d.user(polite(r, ch(r, [f"Sort {f} {order}.", f"Put the lines of {f} in order ({order})."])))
        d.call("run", lead(r, "Sorting into a temp file and replacing."), command=f"sort {flag}{f} > tmp.txt && mv tmp.txt {f}")
    else:
        d.user(polite(r, ch(r, [f"Sort {f} {order} and save it as {dst}.", f"Write a sorted copy of {f} to {dst} ({order})."])))
        d.call("run", lead(r), command=f"sort {flag}{f} > {dst}")
    out = d.call("run", lead(r, "Here's the result."), command=f"cat {dst}")
    d.say(ch(r, [f"Sorted. {dst} now reads: {nums_text(out.splitlines())}.", "Done, the file is sorted."]))
    return True


def task_calc(d: Doc) -> bool:
    r = d.rng
    a, b = int(r.integers(12, 999)), int(r.integers(2, 99))
    op, word = ch(r, [("+", "plus"), ("-", "minus"), ("*", "times"), ("/", "divided by")])
    expr = f"{a} {op} {b}"
    ask = ch(r, [f"What is {a} {op} {b}?", f"What's {a} {word} {b}?", f"Calculate {expr}.", f"compute {a}{op}{b}",
                 f"How much is {a} {word} {b}?", f"{expr} = ?"])
    d.user(polite(r, ask))
    if d.has("calculator") and (not d.has("python") or r.random() < 0.7):
        res = d.call("calculator", lead(r), expression=expr)
    elif d.has("python"):
        res = d.call("python", lead(r), code=f"print({expr})")
    else:
        res = calculator(expr)  # no tool: the assistant does it itself
        d.say(ch(r, [f"{expr} = {res}", f"It's {res}."]))
        return True
    d.say(ch(r, [f"{expr} = {res}", f"It's {res}.", f"The answer is {res}.", res]))
    return True


def task_small_math(d: Doc) -> bool:
    r = d.rng
    a, b = int(r.integers(0, 10)), int(r.integers(0, 10))
    op = ch(r, ["+", "-", "*"])
    res = calculator(f"{a} {op} {b}")
    d.user(polite(r, ch(r, [f"What is {a} {op} {b}?", f"{a}{op}{b}?", f"quick: {a} {op} {b}"])))
    d.say(ch(r, [res, f"{a} {op} {b} = {res}", f"It's {res}."]))
    return True


def task_word_problem(d: Doc) -> bool:
    r = d.rng
    k = int(r.integers(6))
    if k == 0:
        n, per, eat = int(r.integers(2, 12)), int(r.integers(3, 25)), int(r.integers(1, 10))
        q, expr, unit = (f"I have {n} boxes with {per} {ch(r, FRUITS)}s each and give away {eat}. How many are left?",
                         f"{n} * {per} - {eat}", "")
    elif k == 1:
        price, qty = round(float(r.integers(100, 2000)) / 100, 2), int(r.integers(2, 15))
        q, expr, unit = (f"{ch(r, ITEMS).capitalize()} costs ${price:.2f}. How much do {qty} cost?", f"{qty} * {price}", "$")
    elif k == 2:
        speed, hours = int(r.integers(30, 120)), int(r.integers(2, 9))
        q, expr, unit = (f"A car drives at {speed} km/h for {hours} hours. How far does it go?", f"{speed} * {hours}", " km")
    elif k == 3:
        w, h = int(r.integers(2, 40)), int(r.integers(2, 40))
        q, expr, unit = (f"What is the area of a {w} by {h} meter room?", f"{w} * {h}", " square meters")
    elif k == 4:
        xs = [int(v) for v in r.integers(40, 100, int(r.integers(3, 6)))]
        q, expr, unit = (f"My test scores are {nums_text(xs)}. What's my average?", f"({' + '.join(map(str, xs))}) / {len(xs)}", "")
    else:
        p, total = int(r.integers(5, 60)), int(r.integers(20, 900))
        q, expr, unit = (f"What is {p}% of {total}?", f"{p} / 100 * {total}", "")
    d.user(polite(r, q))
    if d.has("calculator"):
        res = d.call("calculator", lead(r, "Let me compute that."), expression=expr)
    elif d.has("python"):
        res = d.call("python", lead(r), code=f"print({expr})")
    else:
        return False
    shown = f"${res}" if unit == "$" else res + unit
    d.say(ch(r, [f"{shown}.", f"The answer is {shown}.", f"That's {expr} = {shown}."]))
    return True


def task_python_math(d: Doc) -> bool:
    r = d.rng
    if not (d.has("python") or d.has("run", "write_file")):
        return False
    k = int(r.integers(5))
    n = int(r.integers(5, 60))
    if k == 0:
        q, codes = f"What is the sum of all numbers from 1 to {n}?", [f"print(sum(range(1, {n + 1})))",
                                                                         f"total = 0\nfor i in range(1, {n + 1}):\n    total += i\nprint(total)"]
    elif k == 1:
        n = int(r.integers(5, 21))
        q, codes = f"What is {n} factorial?", [f"import math\nprint(math.factorial({n}))",
                                               f"r = 1\nfor i in range(2, {n + 1}):\n    r *= i\nprint(r)"]
    elif k == 2:
        q, codes = f"List the prime numbers below {n}.", [f"print([p for p in range(2, {n}) if all(p % d for d in range(2, p))])"]
    elif k == 3:
        q, codes = f"How many multiples of 3 or 5 are there below {n * 10}?", [
            f"print(len([i for i in range(1, {n * 10}) if i % 3 == 0 or i % 5 == 0]))"]
    else:
        q, codes = f"What is the sum of the squares from 1 to {n}?", [f"print(sum(i * i for i in range(1, {n + 1})))"]
    d.user(polite(r, ch(r, [q, q.replace("What is", "Compute").replace("?", "."), "Use code to answer: " + q])))
    code = ch(r, codes)
    if d.has("python") and (not d.has("run", "write_file") or r.random() < 0.7):
        out = d.call("python", lead(r, "I'll compute it with Python."), code=code)
    else:
        fname = ch(r, ["calc.py", "solve.py", "math_task.py"])
        d.call("write_file", lead(r, "I'll write a small script."), path=fname, content=code + "\n")
        out = d.call("run", lead(r), command=f"python3 {fname}")
    d.say(ch(r, [f"The answer is {out}.", out, f"It's {out}."]))
    return True


def task_write_function(d: Doc) -> bool:
    r = d.rng
    spec = ch(r, FUNCS)
    impl = ch(r, spec.impls)
    fname = ch(r, [f"{spec.name}.py", "solution.py", "utils.py", f"{invented(r).lower()}.py"])
    if d.sh.exists(fname):
        return False
    desc = ch(r, spec.descs)
    d.user(polite(r, ch(r, [f"Write a Python function {spec.name} that {desc}, save it in {fname} and test it.",
                             f"Create {fname} with a function `{spec.name}` that {desc}. Run it on a couple of inputs.",
                             f"I need a function that {desc}. Call it {spec.name}, put it in {fname} and check it works.",
                             f"In {fname}, implement {spec.name}: it {desc}. Then try it."])))
    tests = spec.tests(r, int(r.integers(1, 4)))
    prog = spec.program(r, impl, tests)
    if d.has("write_file", "run"):
        d.call("write_file", lead(r, "Here is the function with a few test prints.", f"Writing {fname}."), path=fname, content=prog)
        out = d.call("run", lead(r, "Running it.", "Let me test it."), command=f"python3 {fname}")
    elif d.has("write_file", "python"):
        d.call("write_file", lead(r), path=fname, content=prog)
        out = d.call("python", lead(r, "Testing it."), code=prog)
    elif d.has("write_file"):  # saved, but nothing to run it with
        d.call("write_file", lead(r), path=fname, content=prog)
        d.say(ch(r, [f"I wrote {spec.name} to {fname} with test prints, but none of my tools can run Python, so it is untested.",
                     f"{fname} is saved. I can't run code with these tools, so I couldn't test it."]))
        return True
    elif d.has("run"):  # writing a multi-line file through the shell: not generated
        return False
    else:
        return no_tool(d, "write files")
    res = out.splitlines()
    pairs = "; ".join(f"{t} -> {v}" for t, v in zip(tests, res))
    d.say(ch(r, [f"{fname} is written and works: {pairs}.", f"Done. Test output: {pairs}.", f"It works ({pairs})."]))
    return True


def task_fix_bug(d: Doc) -> bool:
    r = d.rng
    if not d.has("run", "read_file", "write_file"):
        return False
    spec = ch(r, [f for f in FUNCS if f.bugs])
    buggy, kind = ch(r, spec.bugs)
    fname = ch(r, [f"{spec.name}.py", "main.py", "app.py"])
    if d.sh.exists(fname):
        return False
    tests = spec.tests(r, 2)
    good = spec.program(r, spec.impls[0], tests)
    d.sh.write(fname, spec.program(r, buggy, tests))
    expect = run_python(good, fname).splitlines()
    if kind == "crash":
        ask = ch(r, [f"{fname} crashes when I run it. Can you fix it?", f"Running {fname} gives an error, please fix it.",
                     f"Something is broken in {fname}, fix it."])
    else:
        ask = ch(r, [f"{tests[0]} in {fname} should give {expect[0]} but doesn't. Fix the bug.",
                     f"{fname} prints wrong results; {tests[0]} must be {expect[0]}. Please fix.",
                     f"The function in {fname} is wrong, it should {ch(r, spec.descs).split(' and ')[0]}. Fix it."])
    d.user(polite(r, ask))
    before = d.call("run", lead(r, "Let me run it first.", "Reproducing it."), command=f"python3 {fname}")
    d.call("read_file", lead(r, "Now the code." if "Traceback" in before else "The output is wrong. Let me read the code."), path=fname)
    d.call("write_file", lead(r, "Fixing the function.", "Here is the corrected version."), path=fname, content=good)
    after = d.call("run", lead(r, "Running it again."), command=f"python3 {fname}")
    d.say(ch(r, [f"Fixed. {fname} now prints:\n{after}", f"The bug is fixed; the output is now {nums_text(after.splitlines())}.",
                 "Fixed, it gives the right results now."]))
    return True


def task_run_script(d: Doc) -> bool:
    r = d.rng
    f = pick_file(d, lambda p, t: p.endswith(".py"))
    if f is None or not d.has("run"):
        return False
    d.user(polite(r, ch(r, [f"Run {f} and tell me what it prints.", f"What does {f} output?", f"execute {f}"])))
    out = d.call("run", lead(r, "Running it."), command=f"python3 {f}")
    d.say(ch(r, [f"It prints:\n{out}", f"The output is:\n{out}", out]))
    return True


def task_chat(d: Doc) -> bool:
    """No tool needed and none called."""
    r = d.rng
    q, a = ch(r, [("Hi!", "Hi! How can I help?"), ("thanks", "You're welcome!"), ("What tools do you have?", None),
                  ("Who are you?", "I'm an assistant that can use the tools listed above."),
                  (f"How do you spell {ch(r, WORDS)} backwards?", None)])
    d.user(q)
    if q.startswith("What tools"):
        a = "I can use: " + ", ".join(n for n in d.listed) + "."
    elif a is None:
        word = q.split()[-2]
        a = f"{word} backwards is {word[::-1]}."
    d.say(a)
    return True


def no_tool(d: Doc, what: str) -> bool:
    """The task needs a tool this document does not list: say so (sometimes), else pick another task."""
    if d.rng.random() < 0.5:
        return False
    d.say(ch(d.rng, [f"I can't {what} with the tools I have.", f"None of my tools can {what}, sorry.",
                     f"I don't have a tool to {what}."]))
    return True


TASKS_W = [(task_create_list, 6), (task_append, 4), (task_folder, 3), (task_count_lines, 5), (task_show, 4),
           (task_list_dir, 3), (task_headtail, 3), (task_contains, 10), (task_followup, 12), (task_which_file, 4), (task_config, 6),
           (task_csv, 5), (task_inline_lookup, 7), (task_rename, 3), (task_delete, 2), (task_wrong_name, 3),
           (task_sort, 3), (task_calc, 6), (task_small_math, 2), (task_word_problem, 5), (task_python_math, 5),
           (task_write_function, 9), (task_fix_bug, 8), (task_run_script, 3), (task_chat, 1)]
_P = np.array([w for _, w in TASKS_W], dtype=np.float64)
_P /= _P.sum()


def make_doc(rng) -> str:
    real = list(Tools.SPECS)
    tools = [t for t in real if rng.random() < ch(rng, [0.45, 0.65, 0.85])] or [ch(rng, real)]
    fake = list(rng.choice(list(Tools.FAKE), int(rng.choice([0, 0, 1, 2])), replace=False))
    d = Doc(rng, tools, fake, workspace(rng))
    turns = 1 + int(rng.random() < 0.45) + int(rng.random() < 0.2)
    done, tries = 0, 0
    while done < turns and tries < 40:
        tries += 1
        fn = TASKS_W[int(rng.choice(len(TASKS_W), p=_P))][0]
        if done and rng.random() < 0.5 and any(d.in_context(p) for p in d.seen):
            fn = task_followup  # a file is on screen: ask about it (answered from the context)
        mark = len(d.parts)
        if fn(d):
            done += 1
        else:
            del d.parts[mark:]  # (tasks only append after deciding; a refusal path never leaves half a turn)
    return d.text()


# ---------------------------------------------------------------------------------------------------------------
# tokens


_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL) + ")")
_SPECIAL_ID = {s: i for i, s in enumerate(SPECIAL)}


def encode(text: str) -> List[int]:
    ids: List[int] = []
    for piece in _SPLIT.split(text):
        if piece in _SPECIAL_ID:
            ids.append(_SPECIAL_ID[piece])
        elif piece:
            ids.extend(BYTE0 + b for b in piece.encode("utf-8"))
    return ids


def decode(ids: Sequence[int]) -> str:
    out, buf = [], bytearray()
    for i in ids:
        if i < BYTE0:
            out.append(buf.decode("utf-8", "replace") + SPECIAL[i])
            buf = bytearray()
        else:
            buf.append(i - BYTE0)
    return "".join(out) + buf.decode("utf-8", "replace")


def build_tokenizer():
    """A `tokenizers` Tokenizer with the same ids (for `model generate` and the analysis scripts): the special tags,
    then the 256 bytes in GPT-2's byte-to-unicode alphabet, no merges."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    b2u = dict(zip(bs, map(chr, cs)))
    vocab = {s: i for i, s in enumerate(SPECIAL)}
    vocab.update({b2u[b]: BYTE0 + b for b in range(256)})
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIAL)
    return tok


# ---------------------------------------------------------------------------------------------------------------


def build_split(seed: int, n_docs: int) -> Tuple[np.ndarray, List[int]]:
    rng = np.random.default_rng(seed)
    chunks, lengths = [], []
    for _ in range(n_docs):
        ids = encode(make_doc(rng))
        chunks.append(np.asarray(ids, dtype=np.int32))
        lengths.append(len(ids))
    return np.concatenate(chunks), lengths


def check_shell(n: int = 300, seed: int = 1) -> None:
    """Replays the `run` calls of generated documents in a real bash (fresh temp dir per document, same starting
    files) and compares every output with the simulator's."""
    import os
    import subprocess
    import tempfile

    rng = np.random.default_rng(seed)
    checked = bad = 0
    for _ in range(n):
        files = workspace(rng)
        d = Doc(rng, list(Tools.SPECS), [], files)
        for fn in rng.permutation([f for f, _ in TASKS_W]):
            if fn(d):
                break
        if d.snapshot is None:
            continue
        start, dirs = d.snapshot
        sim = Shell(start)
        sim.dirs |= dirs
        tools = Tools(sim)
        with tempfile.TemporaryDirectory() as tmp:
            for dd in dirs:
                os.makedirs(os.path.join(tmp, dd), exist_ok=True)
            for p, t in start.items():
                os.makedirs(os.path.join(tmp, posixpath.dirname(p)), exist_ok=True)
                Path(tmp, p).write_text(t)
            for part in d.parts:
                m = re.search(r"<call>(.*?)</call>", part, re.S)
                if not m:
                    continue
                name, args = parse_call(m.group(1))
                got = tools.call(name, args)
                if name == "write_file":
                    os.makedirs(os.path.join(tmp, posixpath.dirname(args["path"])), exist_ok=True)
                    Path(tmp, args["path"]).write_text(args["content"])
                    continue
                if name != "run":
                    continue
                real = subprocess.run(["bash", "-c", args["command"]], cwd=tmp, capture_output=True, text=True,
                                      env={"LC_ALL": "C", "PATH": os.environ["PATH"], "HOME": HOME})
                want = (real.stderr + real.stdout).rstrip("\n").replace(tmp, HOME)
                want = re.sub(r"\n\s+[~^]+\s*(?=\n|$)", "", want)  # CPython's ^^^ markers (not simulated)
                checked += 1
                if want != got:
                    bad += 1
                    if bad <= 10:
                        print(f"MISMATCH  $ {args['command']}\n  bash: {want!r}\n  sim:  {got!r}")
    print(f"shell check: {checked} commands, {bad} mismatches")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-docs", type=int, default=50_000)
    p.add_argument("--validation-docs", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=0, help="print this many documents and exit")
    p.add_argument("--check-shell", action="store_true", help="compare the simulated shell with real bash and exit")
    a = p.parse_args()
    if a.check_shell:
        check_shell()
        return
    if a.samples:
        rng = np.random.default_rng(a.seed)
        for _ in range(a.samples):
            text = make_doc(rng)
            assert decode(encode(text)) == text
            print(text, f"\n[{len(encode(text))} tokens]\n" + "-" * 100)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, tl = build_split(a.seed, a.train_docs)
    val, vl = build_split(a.seed + 1_000_003, a.validation_docs)
    np.save(OUT_DIR / "train.npy", train)
    np.save(OUT_DIR / "validation.npy", val)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer()
    sample = make_doc(np.random.default_rng(12345))
    assert tok.encode(sample).ids == encode(sample), "tokenizer.json disagrees with the builder's encoding"
    tok.save(str(TOKENIZER_DIR / "tokenizer.json"))
    q = lambda xs: {k: int(np.percentile(xs, v)) for k, v in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))}
    meta = {
        "name": NAME,
        "source": "synthetic tool-use conversations (datasets/build_agent_shell.py): tools, user, assistant, "
                  "calls and executed results; shell simulated, Python really executed",
        "tokenizer": NAME,
        "vocabulary_size": VOCAB_SIZE,
        "doc_token_id": BOS,
        "bos_token_id": BOS,
        "eos_token_id": EOS,
        "special_tokens": {s: i for i, s in enumerate(SPECIAL)},
        # the loss covers only what the assistant writes: inside <assistant> ... </assistant>, the close tag included
        # (prompts, tool lists and tool results are context, not predictions; see model.data.loss_target_mask)
        "loss_spans": [[SPECIAL.index("<assistant>"), SPECIAL.index("</assistant>")]],
        "byte_offset": BYTE0,
        "avg_chars_per_token": 1.0,
        "dtype": "int32",
        "seed": a.seed,
        "train": {"file": "train.npy", "tokens": int(train.size), "documents": a.train_docs, "doc_tokens": q(tl)},
        "validation": {"file": "validation.npy", "tokens": int(val.size), "documents": a.validation_docs, "doc_tokens": q(vl)},
        "notes": "Token = special tag (ids 0-11) or one UTF-8 byte (id 12 + byte). Documents start with <bos> and end "
                 "with <eos>. Calls are Python-style keyword calls with JSON string literals: "
                 "name(arg=\"...\"); `parse_call` in the builder parses them.",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {OUT_DIR}/: train {train.size:,} tokens ({a.train_docs} docs), validation {val.size:,} tokens; "
          f"doc length {q(tl)}")


if __name__ == "__main__":
    main()
