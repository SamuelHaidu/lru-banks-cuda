"""What the page offers to pick from: system prompts, tool lines and sets, examples, and samples of a dataset
(first user messages, completion starters, whole documents)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "datasets") not in sys.path:
    sys.path.insert(0, str(ROOT / "datasets"))

import build_agent_mix_v2 as mix_v2  # noqa: E402


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
