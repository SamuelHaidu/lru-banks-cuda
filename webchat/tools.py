"""The tools the page lists and the server executes (see `webchat/server.py`): the simulated shell and files of the
synthetic dataset, a calculator, real Python in a separate process, and a persistent memory."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "datasets") not in sys.path:
    sys.path.insert(0, str(ROOT / "datasets"))

import build_agent_shell as shell_ds  # noqa: E402

STATE = ROOT / ".webchat"
MEMORY_FILE = STATE / "memory.json"
PYTHON_TIMEOUT = 10


def run_python_process(code: str) -> str:
    """Runs code in a fresh `python -I` in a temp directory: stdout + stderr, cut at 4000 characters."""
    with tempfile.TemporaryDirectory(prefix="webchat-py-") as tmp:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp, capture_output=True, text=True,
                               timeout=PYTHON_TIMEOUT)
            out = p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            out = f"Error: timed out after {PYTHON_TIMEOUT} s"
    return out.rstrip("\n")[:4000]


class Shell(shell_ds.Shell):
    """The dataset's simulated shell, with `python file.py` run in a separate process instead of in this one."""

    def c_python(self, args, stdin):
        if not args:
            return "", "", True
        q = self.norm(args[0])
        if q not in self.files:
            return "", f"python3: can't open file '{shell_ds.HOME}/{q}': [Errno 2] No such file or directory\n", False
        out = run_python_process(self.files[q])
        return out + ("\n" if out else ""), "", "Traceback" not in out


MEMORY_SAVE = {"memory_save", "memory_store", "remember", "save_note"}
MEMORY_LOOKUP = {"memory_search", "memory_recall", "recall", "get_note"}


class Workspace:
    """The tool state of the page: a virtual filesystem (reset on demand) and the persistent memory."""

    def __init__(self, seed: int = 0):
        self.lock = threading.Lock()
        self.reset(seed)
        STATE.mkdir(exist_ok=True)
        self.memory: Dict[str, str] = json.loads(MEMORY_FILE.read_text()) if MEMORY_FILE.exists() else {}

    def reset(self, seed: int) -> None:
        self.shell = Shell(shell_ds.workspace(np.random.default_rng(seed)))
        self.tools = shell_ds.Tools(self.shell)

    def _save_memory(self) -> None:
        MEMORY_FILE.write_text(json.dumps(self.memory, indent=2, ensure_ascii=False))

    def call(self, text: str) -> str:
        """Executes one `<call>` body; returns what goes inside `<result>`."""
        try:
            name, args = parse_call(text)
        except ValueError as e:
            return f"Error: {e}"
        with self.lock:
            try:
                if name in MEMORY_SAVE:
                    key, value = str(args.get("key", "")), args.get("value", "")
                    self.memory[key] = value if isinstance(value, str) else json.dumps(value)
                    self._save_memory()
                    return json.dumps({"saved": True, "key": key}, ensure_ascii=False)
                if name in MEMORY_LOOKUP:
                    key = str(args.get("key", args.get("query", "")))
                    if key in self.memory:
                        return json.dumps({"key": key, "value": self.memory[key]}, ensure_ascii=False)
                    hits = {k: v for k, v in self.memory.items() if key.lower() in (k + " " + v).lower()}
                    if hits:
                        k, v = next(iter(hits.items()))
                        return json.dumps({"key": k, "value": v}, ensure_ascii=False)
                    return json.dumps({"error": "not found", "key": key}, ensure_ascii=False)
                if name in ("python", "execute_python"):
                    return run_python_process(str(args.get("code", "")))
                if name == "execute_bash":  # the Atum09 names, on the same shell
                    return self.tools.call("run", {"command": str(args.get("command", ""))})
                if name == "list_directory":
                    return self.tools.call("run", {"command": "ls " + str(args.get("path", "."))})
                if name in self.tools.SPECS:
                    return self.tools.call(name, {k: str(v) for k, v in args.items()})
            except KeyError as e:
                return f"Error: missing argument {e}"
        return json.dumps({"error": "tool not available", "tool": name})

    def files(self) -> Dict[str, str]:
        return dict(sorted(self.shell.files.items()))


def parse_call(text: str) -> Tuple[str, Dict[str, Any]]:
    """`name(key=<literal>, ...)` -> (name, args); literals of any type (the dataset's parser takes strings only)."""
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as e:
        raise ValueError(f"not a call: {e.msg}") from e
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args:
        raise ValueError("not a call with keyword arguments")
    try:
        return node.func.id, {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    except ValueError as e:
        raise ValueError("arguments must be literals") from e
