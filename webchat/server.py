"""Local web chat with a trained run: pick a run and checkpoint, a system prompt and tools from the training data, and
talk to the model; tokens stream to the page, and the model's tool calls run for real.

    uv run python webchat/server.py [--port 8000] [--device xpu|cpu]     # then open http://localhost:8000

Tools the page can list and the server executes (the formats of the training data):
  run(command)                  the simulated bash of the synthetic dataset (`datasets/build_agent_shell.py`: ls, cat,
                                grep, wc, sort, head, mkdir, rm, mv, cp, echo, python file.py, ...) over a virtual
                                filesystem filled like the training workspaces (lists, configs, csv, logs, programs)
  read_file(path), write_file(path, content)    the same virtual filesystem
  calculator(expression)        exact arithmetic
  python(code)                  real Python, in a separate process (`python -I`, temp directory, 10 s timeout)
  memory tools                  every naming of the training data (memory_save / memory_search, memory_store /
                                memory_recall, remember / recall, save_note / get_note); saved to
                                .webchat/memory.json, so they persist across conversations and restarts
  Atum09 tools                  the Atum09 source's own lines (`ATUM_TOOL_LINES`; the models think only with them):
                                execute_bash / list_directory on the simulated bash, execute_python as python,
                                read_file / write_file / memory_search as above
  anything else                listed tools the demo does not implement answer {"error": "tool not available"}

One model on one device per process (the device of the first run loaded, or --device). Generation is the project's
own (`model.generation.generate`, the recurrent state carried token by token for `min-gru` / `rope-state-memory`).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import queue
import random
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datasets"))

import build_agent_shell as shell_ds  # noqa: E402
import build_agent_mix_v2 as mix_v2  # noqa: E402

HERE = Path(__file__).resolve().parent
STATE = ROOT / ".webchat"
MEMORY_FILE = STATE / "memory.json"
PYTHON_TIMEOUT = 10


# -- tools ----------------------------------------------------------------------------------------------------------

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


# -- presets from the training data ---------------------------------------------------------------------------------

TOOL_LINES = {
    "run": "run(command: str): run a shell command in the workspace and return its output",
    "read_file": "read_file(path: str): return the contents of a file",
    "write_file": "write_file(path: str, content: str): create or overwrite a file with the given content",
    "calculator": "calculator(expression: str): evaluate an arithmetic expression",
    "python": "python(code: str): run Python code and return what it prints",
}
for _save, _look, _save_line, _look_line in mix_v2.MEMORY_TOOLS:
    TOOL_LINES[_save], TOOL_LINES[_look] = _save_line, _look_line
DECOY_TOOLS = mix_v2.EXTRA_TOOLS

TOOL_SETS = {
    "none": [],
    "shell + files + code": ["run", "read_file", "write_file", "calculator", "python"],
    "memory (memory_save / memory_search)": ["memory_save", "memory_search"],
    "memory (memory_store / memory_recall)": ["memory_store", "memory_recall"],
    "memory + shell": ["memory_save", "memory_search", "run", "read_file", "write_file"],
    "calculator + python": ["calculator", "python"],
}

# The Atum09 source's tools (the most common of its 56 lines in agent-mix-v2's train split, exact text). Thinking is
# only in Atum09 documents and nearly all of them list these tools: the models think when they see them (P(<think>)
# after <assistant> ~1.0 with them, ~0.0 with the synthetic tools above).
ATUM_TOOL_LINES = {line.split("(")[0]: line for line in [
    "web_search(query: str, num_results: int): Search the web for current information",
    "email_send(to: str, subject: str, body: str): Send an email",
    "email_read(folder: str, limit: int): Read emails",
    "write_file(path: str, content: str): Write content to a file",
    "read_file(path: str): Read a file",
    "list_directory(path: str): List directory contents",
    "execute_python(code: str): Execute Python code",
    "execute_bash(command: str): Execute a bash command",
    "csv_parse(file_path: str, query: str): Parse a CSV file",
    "chart_create(data: dict, chart_type: str, title: str): Create a chart",
    "stock_price(ticker: str, period: str): Get stock price",
    "crypto_price(symbol: str, vs_currency: str): Get crypto price",
    "currency_convert(amount: float, from_currency: str, to_currency: str): Convert currencies",
    "weather_get(location: str, units: str): Get weather for a location",
    "calendar_create_event(title: str, start_time: str, end_time: str): Create a calendar event",
    "calendar_list_events(days_ahead: int): List calendar events",
    "social_schedule(platform: str, content: str, scheduled_time: str): Schedule a social media post",
    "twitter_post(text: str): Post to Twitter/X",
    "linkedin_post(content: str): Post to LinkedIn",
    "system_info(metrics: list): Get system info",
    "process_list(filter: str): List running processes",
    "docker_manage(action: str, container: str): Manage Docker containers",
    "memory_search(query: str): Search memory",
    "arxiv_search(query: str, max_results: int): Search academic papers",
]}
ATUM_IMPLEMENTED = ["write_file", "read_file", "list_directory", "execute_python", "execute_bash", "memory_search"]
for _name, _set in (("email + web + files", ["email_send", "web_search", "write_file"]),
                    ("email", ["email_read", "email_send"]), ("finance", ["stock_price", "crypto_price", "currency_convert"]),
                    ("data + charts + python", ["csv_parse", "chart_create", "execute_python"]),
                    ("files + bash + python", ["read_file", "write_file", "list_directory", "execute_bash", "execute_python"]),
                    ("social", ["social_schedule", "twitter_post", "linkedin_post"])):
    TOOL_SETS[f"atum09 (thinks): {_name}"] = ["atum:" + n for n in _set]

EXAMPLES = {
    "shell / files": ["What files are in my home directory?", "Is there a banana in fruits.txt?",
                      "How many lines does todo.txt have?", "Show me the port in config.ini.",
                      "Create notes.txt with the text: buy milk", "Sort the names in guests.txt alphabetically."],
    "math / code": ["What's 652 plus 48?", "Flour costs $12.69. How much do 9 cost?",
                    "Write a Python function that reverses a string and test it on 'hello'.",
                    "Run python to print the first 10 square numbers."],
    "memory": ["Please remember that my wifi password is maple-4821.", "What's my wifi password?",
               "Remember these: my locker code is QT-8814, the meeting room is Room 4B.",
               "Do you remember my locker code?"],
    "copy / recall": ["Repeat this exactly: The quick brown fox jumps over the lazy dog.",
                      "Here is a list: Farid, river, Pavel, Greta, Dev. What is the third item?",
                      "port: 8080\nhost: db.internal\nuser: ralo\n\nWhat is the host?"],
    "sentiment": ["Is this sentence negative, positive or neutral? I loved the movie, best night in ages!",
                  "What is the sentiment of this text: \"the bus was late again, great.\""],
    "general knowledge": ["What is the largest species of shark?", "Who painted the Mona Lisa?",
                          "Explain what photosynthesis is."],
    "chat": ["hey, what are you up to tonight?", "lol did you see the game yesterday"],
}


def system_presets() -> Dict[str, List[str]]:
    out = {kind: list(v) for kind, v in mix_v2.SYSTEM.items()}
    out["atum09 (use with its tools)"] = [  # its five system prompts (the training text)
        "You are a helpful AI assistant with access to tools. Think step by step before calling any tool. After getting "
        "results, synthesize a clear response. If a tool fails, handle the error and try an alternative.",
        "You are a capable AI assistant. You have tools available. Use them when the task requires external data, "
        "computation, or actions. Answer from your own knowledge when you can \u2014 don't call tools unnecessarily.",
        "You are a smart assistant that can use tools to get information, run code, manage files, and more. Be "
        "efficient \u2014 plan before acting. Give clear final answers.",
        "You are an autonomous AI agent with tools available. For each step: reason about what you need, choose the "
        "right tool, call it with correct arguments, observe the result, decide next action. Only use tools when "
        "necessary.",
        "You are an AI agent for real-world task completion. Use tools when needed, chain them for complex tasks, "
        "handle errors gracefully, and confirm task completion."]
    return out


_DATA_EXAMPLES: Dict[str, Dict[str, List[str]]] = {}


def data_examples(dataset: str, per_category: int = 8) -> Dict[str, List[str]]:
    """First user messages of random validation conversations, by category when the dataset has an index."""
    if dataset in _DATA_EXAMPLES:
        return _DATA_EXAMPLES[dataset]
    from model import config as cfg_mod
    from tokenizers import Tokenizer

    out: Dict[str, List[str]] = {}
    try:
        meta = cfg_mod.load_dataset_metadata(dataset)
        tok = Tokenizer.from_file(str(cfg_mod.tokenizer_path(dataset)))
        d = cfg_mod.DATASETS_DIR / dataset
        tokens = np.load(d / meta["validation"]["file"], mmap_mode="r")
        rng = np.random.default_rng(0)
        if (d / "validation.categories.npy").exists():
            index = np.load(d / "validation.categories.npy")
            names = {v: k for k, v in meta["category_ids"].items()}
            groups = {names[c]: index[index[:, 2] == c] for c in names}
        else:
            starts = np.flatnonzero(np.asarray(tokens[:3_000_000]) == meta["doc_token_id"])
            groups = {"validation": np.stack([starts[:-1], np.diff(starts)], 1)}
        for name, rows in groups.items():
            msgs = []
            for s, n in rows[rng.permutation(len(rows))[:60], :2]:
                text = tok.decode(np.asarray(tokens[s:s + n]).tolist(), skip_special_tokens=False)
                m = re.search(r"<user>(.*?)</user>", text, re.S)
                if m and len(m.group(1)) <= 400 and m.group(1).strip() not in msgs:
                    msgs.append(m.group(1).strip())
                if len(msgs) == per_category:
                    break
            out[name] = msgs
    except Exception as e:  # noqa: BLE001 (examples are optional)
        out = {"(could not read the dataset)": [str(e)]}
    _DATA_EXAMPLES[dataset] = out
    return out


_DOCS: Dict[Tuple[str, str], Dict[str, List[str]]] = {}


_STARTERS: Dict[str, Dict[str, List[str]]] = {}


def completion_starters(dataset: str, per_category: int = 6, tokens: int = 24) -> Dict[str, List[str]]:
    """Openings of random validation documents (their first `tokens` tokens, cut at a word end), by category: prompts
    for completion mode."""
    if dataset in _STARTERS:
        return _STARTERS[dataset]
    from tokenizers import Tokenizer

    from model import config as cfg_mod

    meta = cfg_mod.load_dataset_metadata(dataset)
    tok = Tokenizer.from_file(str(cfg_mod.tokenizer_path(dataset)))
    d = cfg_mod.DATASETS_DIR / dataset
    tokens_all = np.load(d / "validation.npy", mmap_mode="r")
    index = np.load(d / "validation.categories.npy")
    names = {v: k for k, v in meta.get("category_ids", {}).items()}
    rng = np.random.default_rng(0)
    out: Dict[str, List[str]] = {}
    for c in np.unique(index[:, 2]):
        rows = index[index[:, 2] == c]
        for s, n, _ in rows[rng.permutation(len(rows))[:per_category]]:
            text = tok.decode(np.asarray(tokens_all[s + 1:s + min(n, tokens + 1)]).tolist(), skip_special_tokens=False)
            text = text[:text.rfind(" ")] if " " in text else text
            out.setdefault(names.get(int(c), str(c)), []).append(text.strip())
    _STARTERS[dataset] = out
    return out


def dataset_documents(dataset: str, split: str, per_category: int = 12, max_tokens: int = 1024) -> Dict[str, List[str]]:
    """Whole documents of a split (as the model reads them), up to `max_tokens` long, a random sample per category
    (or of the whole split when the dataset has no category index)."""
    if (dataset, split) in _DOCS:
        return _DOCS[(dataset, split)]
    from model import config as cfg_mod
    from tokenizers import Tokenizer

    meta = cfg_mod.load_dataset_metadata(dataset)
    tok = Tokenizer.from_file(str(cfg_mod.tokenizer_path(dataset)))
    d = cfg_mod.DATASETS_DIR / dataset
    tokens = np.load(d / meta[split]["file"], mmap_mode="r")
    if (d / f"{split}.categories.npy").exists():
        index = np.load(d / f"{split}.categories.npy")
        names = {v: k for k, v in meta["category_ids"].items()}
        groups = {names[c]: index[index[:, 2] == c][:, :2] for c in names}
    else:
        starts = np.flatnonzero(np.asarray(tokens[:5_000_000]) == meta["doc_token_id"])
        groups = {split: np.stack([starts[:-1], np.diff(starts)], 1)}
    rng = np.random.default_rng(1)
    out = {}
    for name, rows in groups.items():
        rows = rows[rows[:, 1] <= max_tokens]
        pick = rows[rng.permutation(len(rows))[:per_category]]
        out[name] = [tok.decode(np.asarray(tokens[a:a + n]).tolist(), skip_special_tokens=False) for a, n in pick]
    _DOCS[(dataset, split)] = out
    return out


# -- models ---------------------------------------------------------------------------------------------------------

class Models:
    """The loaded (run, checkpoint): one at a time, one generation at a time."""

    def __init__(self, device: Optional[str]):
        self.device = device
        self.key: Optional[Tuple[str, str]] = None
        self.model = self.tokenizer = None
        self.lock = threading.Lock()

    def runs(self) -> List[Dict[str, Any]]:
        from model import config as cfg_mod

        out = []
        for run in sorted(p for p in (ROOT / "runs").iterdir() if (p / "config.json").exists()):
            ckpts = ["latest"] if (run / "model.keras").exists() else []
            if (run / "checkpoints" / "best.keras").exists():
                ckpts.append("best")
            epochs = sorted((int(m.group(1)) for p in (run / "checkpoints").glob("checkpoint_*.keras")
                             if (m := re.search(r"checkpoint_(\d+)", p.name))), reverse=True)
            ckpts += [str(e) for e in epochs]
            if not ckpts:
                continue
            try:
                cfg = cfg_mod.load(run.name)
            except Exception:  # noqa: BLE001 (a run with an old or broken config: skip it)
                continue
            try:  # chat: the dataset scores assistant spans; completion: plain text (pretraining)
                chat = bool(cfg_mod.load_dataset_metadata(cfg["data"]["dataset"]).get("loss_spans"))
            except Exception:  # noqa: BLE001 (the dataset is not here)
                chat = True
            out.append({"run": run.name, "arch": cfg["arch"], "dataset": cfg["data"]["dataset"], "checkpoints": ckpts,
                        "mode": "chat" if chat else "completion",
                        "inspect": cfg["arch"].startswith("min-gru") or cfg["arch"] == "gated-deltanet"})
        return out

    def load(self, run: str, checkpoint: str):
        from model import config as cfg_mod

        with self.lock:
            if self.key != (run, checkpoint):
                if self.device is None:
                    import torch

                    self.device = cfg_mod.load(run).get("device") or "xpu"
                    if self.device == "cuda" and not torch.cuda.is_available():  # a run trained on a pod: here the XPU
                        self.device = "xpu"
                    from model.device import setup_device

                    setup_device(self.device)
                from model.evaluate import load_checkpoint
                from tokenizers import Tokenizer

                ckpt: Any = int(checkpoint) if checkpoint.isdigit() else checkpoint
                self.model = load_checkpoint(run, ckpt)
                self.tokenizer = Tokenizer.from_file(str(cfg_mod.tokenizer_path(cfg_mod.load(run)["data"]["dataset"])))
                self.key = (run, checkpoint)
        return self.model, self.tokenizer


# -- inspection (min-gru, gated-deltanet) ---------------------------------------------------------------------------

MAX_INSPECT_TOKENS = 1024


def _b64(a: np.ndarray) -> str:
    import base64

    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def inspect_text(models: "Models", body: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the Inspect tab needs for one text, for a `min-gru` run.

    Per layer (float32, base64, row-major): update_gate, candidate, state `(T, H)`; W = norm_state[:, None] *
    out_proj `(H, D)`; write_scale `(D,)`; the channels' half-lives on this text; per channel the tokens its
    direction promotes / suppresses through the final norm and the readout (direct logit effect). Global: the
    tokens, which positions reset the state, the readout kernel `(D, V)` times final_norm (so the page can project
    contributions onto any token), the final stream's rms per position, and the logit lens: top-5 next tokens after
    the embedding and after each layer, at every position."""
    import torch

    from model.archs.min_gru import MinGRULM

    from model.archs.gated_deltanet import GatedDeltaNetLM

    model, tok = models.load(body["run"], str(body.get("checkpoint", "latest")))
    if isinstance(model, GatedDeltaNetLM):
        return inspect_gdn(models, body, model, tok)
    if not isinstance(model, MinGRULM):
        raise ValueError(f"inspection is for min-gru and gated-deltanet runs ({type(model).__name__} loaded)")
    ids = tok.encode(body["text"]).ids
    if not ids or ids[0] != model.reset_token:
        ids = [model.reset_token] + ids
    truncated = len(ids) > MAX_INSPECT_TOKENS
    ids = ids[:MAX_INSPECT_TOKENS]
    raw_ids, meta = ids, None
    with models.lock, torch.no_grad():
        if model.static_memory:  # the layers see the row without the span contents: inspect that sequence
            info, meta = model.inspect_memory(np.asarray([ids], np.int32))
            ids = [ids[i] for i in meta["kept"]]
        else:
            info = model.inspect(np.asarray([ids], np.int32))
        E = model.token_embedding.embeddings.value
        emb = E[torch.as_tensor(ids, device=E.device)].float().cpu().numpy()
        K = model.readout.kernel.value.detach().float().cpu().numpy()  # (D, V)
        g_final = model.final_norm.value.detach().float().cpu().numpy()
        params = [{k: v.value.detach().float().cpu().numpy() for k, v in p.items()} for p in model._params]
    readout = g_final[:, None] * K  # logits = (h / rms(h)) @ readout
    rms = lambda x: np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)
    vocab = [tok.decode([i], skip_special_tokens=False) for i in range(K.shape[1])]

    def top(h: np.ndarray, k: int = 5):
        logits = (h / rms(h)) @ readout
        logits -= logits.max(-1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(-1, keepdims=True)
        idx = np.argsort(-p, -1)[:, :k]
        return [[[int(i), round(float(p[t, i]), 4)] for i in row] for t, row in enumerate(idx)]

    stream = emb.copy()
    lens = [top(stream)]
    layers = []
    not_reset = np.asarray(ids) != model.reset_token
    for n, (lay, p) in enumerate(zip(info, params)):
        z, cand, state, write = (lay[k][0] for k in ("update_gate", "candidate", "state", "layer_write"))
        stream = stream + write + (lay["memory_add"][0] if "memory_add" in lay else 0)
        lens.append(top(stream))
        keep = np.clip(1 - z[not_reset], 1e-12, 1)
        half_life = np.log(0.5) / np.minimum(np.log(keep).mean(0), -1e-9)
        layers.append({
            "update_gate": _b64(z), "candidate": _b64(cand), "state": _b64(state),
            "write_scale": p["write_scale"].tolist(), "half_life": np.round(half_life, 1).tolist(),
        })
        if "out_proj" in p:  # the tanh projection (not the MLP): each channel's direct push on the logits
            W = p["norm_state"][:, None] * p["out_proj"]  # (H, D)
            order = np.argsort((W * p["write_scale"][None]) @ readout, -1)  # (H, V)
            layers[-1].update({"W": _b64(W), "promotes": [[vocab[i] for i in row[::-1][:6]] for row in order],
                               "suppresses": [[vocab[i] for i in row[:6]] for row in order]})
    final = stream
    T, H, D = len(ids), model.hidden_size, model.dim
    memory = None if meta is None else memory_reads(model, info, meta, raw_ids, vocab, layers)
    logits = (final / rms(final)) @ readout
    stats = _prediction_stats(logits, ids, body["run"], model.reset_token)
    return {**stats, "memory": memory, "static_memory": bool(model.static_memory), "raw_tokens": len(raw_ids),
            "raw": [vocab[i] for i in raw_ids],  # the whole text; the layers see the positions in `kept`
            "kept": [int(i) for i in meta["kept"]] if meta is not None else list(range(len(raw_ids))),"tokens": [vocab[i] for i in ids], "ids": ids, "reset": (~not_reset).astype(int).tolist(),
            "T": T, "H": H, "D": D, "V": int(K.shape[1]), "truncated": truncated, "layers": layers,
            "readout": _b64(readout), "final_rms": rms(final)[:, 0].tolist(), "lens": lens, "vocab": vocab}


def _top5(logits: np.ndarray, k: int = 5):
    logits = logits - logits.max(-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(-1, keepdims=True)
    idx = np.argsort(-p, -1)[:, :k]
    return [[[int(i), round(float(p[t, i]), 4)] for i in row] for t, row in enumerate(idx)]


def inspect_gdn(models: "Models", body: Dict[str, Any], model: Any, tok: Any) -> Dict[str, Any]:
    """The Inspect tab for a `gated-deltanet` run: tokens, the final prediction's loss / rank / entropy at every
    position, the logit lens (top-5 after the embedding and after each layer) and per layer and head the memory's
    half-life (median over the text's positions) and mean write strength beta. Every layer's delta-rule inputs (q, k, v
    after the conv, g, beta) and read o are kept (`models.attn`) for `/api/attend`, which unrolls a position's read
    into weights over the past tokens."""
    import uuid

    import torch

    import model.archs.gated_deltanet as gd

    ids = tok.encode(body["text"]).ids
    if not ids or ids[0] != model.reset_token:
        ids = [model.reset_token] + ids
    truncated = len(ids) > MAX_INSPECT_TOKENS
    ids = ids[:MAX_INSPECT_TOKENS]
    rec: List[List[torch.Tensor]] = []
    streams: List[torch.Tensor] = []
    orig_rule = gd.gated_delta_rule

    def rule(q, k, v, g, beta, reset, initial=None, need_state=False):
        o, S = orig_rule(q, k, v, g, beta, reset, initial, need_state)
        rec.append([x[0].detach().double().cpu() for x in (q, k, v, g, beta, o)])
        return o, S

    def layer(*args):
        out = type(model)._layer(model, *args)
        streams.append(out[0][0].detach().float())
        return out

    with models.lock, torch.no_grad():
        gd.gated_delta_rule, model._layer = rule, layer
        try:
            model.dense_hidden(np.asarray([ids], np.int32))
        finally:
            gd.gated_delta_rule = orig_rule
            del model._layer
        E = model.token_embedding.embeddings.value
        emb = E[torch.as_tensor(ids, device=E.device)].float()
        logits = [torch.as_tensor(model.readout(model.readout_input(s))).float().cpu().numpy() for s in [emb] + streams]
    vocab = [tok.decode([i], skip_special_tokens=False) for i in range(model.vocabulary_size)]
    reset = np.asarray(ids) == model.reset_token
    q, k, v, g, beta, o = (torch.stack([r[i] for r in rec]) for i in range(6))  # (layers, T, heads, .)
    models.attn = {"id": uuid.uuid4().hex, "reset": reset, "q": q, "k": k, "v": v, "g": g, "beta": beta, "o": o}
    heads = []
    for n in range(len(rec)):
        gn = g[n][torch.as_tensor(~reset)]
        hl = (np.log(0.5) / gn.clamp_max(-1e-9)).median(0).values
        heads.append({"half_life": [round(float(x), 1) for x in hl], "beta": [round(float(x), 3) for x in beta[n].mean(0)]})
    stats = _prediction_stats(logits[-1], ids, body["run"], model.reset_token)
    T = len(ids)
    return {**stats, "kind": "gdn", "attn_id": models.attn["id"], "heads": heads, "conv_size": int(model.conv_size),
            "tokens": [vocab[i] for i in ids], "raw": [vocab[i] for i in ids], "kept": list(range(T)), "ids": ids,
            "reset": reset.astype(int).tolist(), "T": T, "truncated": truncated, "layers": [], "memory": None,
            "static_memory": False, "raw_tokens": T, "lens": [_top5(x) for x in logits], "vocab": vocab}


def attend(models: "Models", body: Dict[str, Any]) -> Dict[str, Any]:
    """Position t's read in every layer and head of the last `inspect_gdn` text, unrolled: o_t = sum_i c_i v_i over
    the document's positions start..t (`delta_rule_read_weights`). Returns c and each position's share of the read,
    c_i (v_i . o_t) / |o_t|^2 (they sum to 1), both `(layers, heads, t + 1 - start)` float32 base64, and the largest
    difference between the rebuilt read and the model's, relative to the read's size."""
    from model.archs.delta_rule import delta_rule_read_weights

    a = getattr(models, "attn", None)
    if a is None or a["id"] != body.get("attn_id"):
        raise ValueError("inspect the text again (another text or model was inspected since)")
    t = int(body["t"])
    c, start = delta_rule_read_weights(a["q"], a["k"], a["g"], a["beta"], a["reset"], t)
    vals = a["v"][:, start:t + 1]  # (layers, n, heads, V)
    rebuilt = np.einsum("lhi,lihv->lhv", c.numpy(), vals.numpy())
    o = a["o"][:, t].numpy()
    err = float(np.abs(rebuilt - o).max() / max(1e-12, float(np.abs(o).max())))
    share = np.einsum("lhi,lihv,lhv->lhi", c.numpy(), vals.numpy(), rebuilt) \
        / np.maximum((rebuilt ** 2).sum(-1), 1e-30)[..., None]
    return {"t": t, "start": start, "n": t + 1 - start, "c": _b64(c.numpy()), "share": _b64(share), "err": err,
            "o_norm": np.round(np.linalg.norm(o, axis=-1), 4).tolist()}


def _prediction_stats(logits: np.ndarray, ids: List[int], run: str, reset_token: int) -> Dict[str, Any]:
    """The prediction at every position (target: the next token): its entropy, the loss and rank of the actual next
    token (rank 0: the top-1 choice), and which targets the training scores (inside the dataset's loss_spans, the
    closing tag included; every one without loss_spans)."""
    from model import config as cfg_mod

    T = len(ids)
    logits = logits - logits.max(-1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    entropy = -(np.exp(logp) * logp).sum(-1)
    loss = np.zeros(T)
    loss[:-1] = -logp[np.arange(T - 1), ids[1:]]
    rank = np.zeros(T, int)
    rank[:-1] = (logp[:-1] > logp[np.arange(T - 1), ids[1:]][:, None]).sum(-1)
    spans = cfg_mod.load_dataset_metadata(cfg_mod.load(run)["data"]["dataset"]).get("loss_spans") or []
    target = np.zeros(T, bool)  # token j is a scored target (predicted at j - 1)
    for o, c in spans:
        inside = False
        for j, t in enumerate(ids):
            if t == reset_token:
                inside = False
            target[j] |= inside
            inside = True if t == o else (False if t == c else inside)
    scored = np.zeros(T, int)
    scored[:-1] = target[1:] if spans else 1
    return {"entropy": np.round(entropy, 4).tolist(), "loss": np.round(loss, 4).tolist(), "rank": rank.tolist(),
            "scored": scored.tolist()}


def memory_reads(model: Any, info: List[Dict[str, np.ndarray]], meta: Dict[str, Any], raw_ids: List[int],
                 vocab: List[str], layers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The static memory for the Inspect tab. Global: each memory entry's token, memory (0 system, 1 tools) and
    document; the reading positions (among the inspected ones) and their documents. Per layer (added to `layers`):
    `mem_w` `(reads, entries)`: the read weights q . k_j / sum over the entries of the same memory and document, mean
    over the heads; `mem_gate` `(reads, memories)`: the gates, mean over the heads; `mem_add` `(T,)`: the size of the
    memory's addition to the stream at every position (0 where nothing reads)."""
    h, d, nm = model.memory_heads, model.memory_dim, len(model.memory_spans)
    read_pos, read_doc = np.asarray(meta["read_pos"]), np.asarray(meta["read_doc"])
    kind, edoc = np.asarray(meta["entry_kind"]), np.asarray(meta["entry_doc"])
    same = (read_doc[:, None] == edoc[None]).astype(np.float32)  # (reads, entries): a position reads its document
    onehot = np.eye(nm, dtype=np.float32)[kind]  # (entries, memories)
    for n, (lay, L) in enumerate(zip(info, layers)):
        q = lay["memory_query"][0][read_pos].reshape(-1, h, d)
        s = np.einsum("rhd,jhd->rhj", q, meta["entry_keys"][:, n]) * same[:, None]
        tot = np.einsum("rhj,jm->rhm", s, onehot)[..., kind]  # each entry's memory total (reads, heads, entries)
        w = s / np.maximum(tot, 1e-12)
        gate = lay["memory_gate"][0][read_pos].reshape(-1, nm, h).mean(-1)
        L.update(mem_w=_b64(w.mean(1)), mem_gate=_b64(gate),
                 mem_add=np.round(np.linalg.norm(lay["memory_add"][0], axis=-1), 5).tolist())
    names = [{(12, 13): "system", (2, 3): "tools"}.get(tuple(sp), f"memory {i}") for i, sp in enumerate(model.memory_spans)]
    return {"entries": [vocab[raw_ids[p]] for p in meta["entry_pos"]], "entry_pos": [int(p) for p in meta["entry_pos"]],
            "entry_kind": kind.tolist(),
            "entry_doc": edoc.tolist(), "read_pos": read_pos.tolist(), "read_doc": read_doc.tolist(), "names": names}


# -- http -----------------------------------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    models: Models
    workspace: Workspace

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/runs":
            self._json(self.models.runs())
        elif self.path.startswith("/api/presets"):
            dataset = self.path.split("dataset=", 1)[1] if "dataset=" in self.path else ""
            self._json({"system": system_presets(), "tool_lines": TOOL_LINES, "tool_sets": TOOL_SETS,
                        "atum_tool_lines": ATUM_TOOL_LINES, "atum_implemented": ATUM_IMPLEMENTED,
                        "decoys": DECOY_TOOLS, "examples": EXAMPLES,
                        "data_examples": data_examples(dataset) if dataset else {}})
        elif self.path.startswith("/api/starters"):
            dataset = self.path.split("dataset=", 1)[1] if "dataset=" in self.path else ""
            try:
                self._json(completion_starters(dataset))
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"})
        elif self.path.startswith("/api/documents"):
            from urllib.parse import parse_qs, urlparse

            q = parse_qs(urlparse(self.path).query)
            try:
                self._json(dataset_documents(q["dataset"][0], q.get("split", ["validation"])[0]))
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"})
        elif self.path == "/api/workspace":
            self._json({"files": self.workspace.files(), "memory": self.workspace.memory})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            body = self._body()
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        if self.path == "/api/tool":
            self._json({"result": self.workspace.call(body.get("call", ""))})
        elif self.path == "/api/workspace/reset":
            with self.workspace.lock:
                self.workspace.reset(int(body.get("seed", random.randrange(1 << 30))))
            self._json({"files": self.workspace.files()})
        elif self.path == "/api/memory/clear":
            with self.workspace.lock:
                self.workspace.memory.clear()
                self.workspace._save_memory()
            self._json({"memory": {}})
        elif self.path == "/api/generate":
            self._generate(body)
        elif self.path == "/api/inspect":
            try:
                self._json(inspect_text(self.models, body))
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"})
        elif self.path == "/api/attend":
            try:
                self._json(attend(self.models, body))
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"})
        else:
            self._json({"error": "not found"}, 404)

    def _generate(self, body: Dict[str, Any]) -> None:
        """Server-sent events: {"text": delta} as tokens come, then {"done": true, "tokens": n, "tps": ...}."""
        import time

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(obj: Dict[str, Any]) -> bool:
            try:
                self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False

        try:
            model, tok = self.models.load(body["run"], str(body.get("checkpoint", "latest")))
        except Exception as e:  # noqa: BLE001
            send({"error": f"cannot load {body.get('run')} / {body.get('checkpoint')}: {e}"})
            return
        events: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        new_ids: List[int] = []
        shown = [0]
        stamps: List[float] = []

        def on_token(i: int) -> None:
            stamps.append(time.perf_counter())
            new_ids.append(i)
            text = tok.decode(new_ids, skip_special_tokens=False)
            if not text.endswith("�"):  # hold a partial UTF-8 character
                events.put({"text": text[shown[0]:]})
                shown[0] = len(text)

        def work() -> None:
            from model.generation import generate

            try:
                with self.models.lock:
                    generate(model, body["document"], tokenizer=tok,
                             max_new_tokens=int(body.get("max_new_tokens", 256)),
                             temperature=float(body.get("temperature", 0.0)),
                             top_k=int(body["top_k"]) if body.get("top_k") else None,
                             top_p=float(body["top_p"]) if body.get("top_p") else None,
                             repetition_penalty=float(body.get("repetition_penalty", 1.0)),
                             stop_tokens=["<eos>"] if body.get("mode") == "completion" else ["</assistant>", "<eos>"],
                             on_token=on_token)
                rest = tok.decode(new_ids, skip_special_tokens=False)[shown[0]:]
                if rest:
                    events.put({"text": rest})
                tps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
                events.put({"done": True, "tokens": len(new_ids), "tps": round(tps, 1)})
            except Exception as e:  # noqa: BLE001
                events.put({"error": str(e)})
            events.put(None)

        threading.Thread(target=work, daemon=True).start()
        while (ev := events.get()) is not None:
            if not send(ev):
                break


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", default=None, help="xpu / cpu / cuda (default: the first loaded run's device)")
    a = p.parse_args()
    if a.device:
        from model.device import setup_device

        setup_device(a.device)
    Handler.models = Models(a.device)
    Handler.workspace = Workspace()
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"webchat on http://{a.host}:{a.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
