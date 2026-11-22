"""The Calc-X calculator task (`datasets/build_calc_x.py`): questions, the answer check, and the batched agent loop
(`rollout`) shared by the evaluation (`ad_hoc_scripts/calc_x_eval.py`) and the RL trainer (`model/grpo.py`).

A question is asked as in training (the calculator tool listed, `<user>question</user><assistant>`); the model writes,
every `<call>calculator(expression="...")</call></assistant>` is executed and its result fed back
(`<result>...</result><assistant>`), until it closes a turn without a call (or `max_new` tokens). Correct: the last
"The answer is X" equals the reference (numbers within 1e-4 relative; multiple choice: the letter).

Batched on a stateful model (`advance`): rows are left-padded with `<bos>` (every `<bos>` restarts the state, so the
padding changes nothing), the prompts run in one pass, then one token per row per step; a row being fed a tool result
takes its tokens one by one. Finished rows are dropped from the batch (their states removed) once they are a quarter
of it.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CALL = re.compile(r'<call>calculator\(expression=("(?:[^"\\]|\\.)*")\)</call>')
ANSWER = re.compile(r"The answer is (.+?)\.?</assistant>")
CHOICE = re.compile(r"([A-E])\)")
RESULT = re.compile(r"<result>(.*?)</result>", re.S)
QNUM = re.compile(r"\d+(?:\.\d+)?")
DATA = ROOT / "datasets" / "calc-x-bpe32768"
TOKENIZER = ROOT / "tokenizers" / "pretrain-mix-bpe32768" / "tokenizer.json"


def _calculator():
    if str(ROOT / "datasets") not in sys.path:
        sys.path.insert(0, str(ROOT / "datasets"))
    from build_agent_shell import calculator

    return calculator


def calc(expression: str) -> str:
    """The dataset's calculator, guarded against what a sampling model writes (stacked powers, float overflow)."""
    if len(expression) > 200 or expression.count("**") + expression.count("^") > 2:
        return f"Error: cannot evaluate {expression!r}"
    try:
        return _calculator()(expression)
    except Exception:  # OverflowError and the like: the calculator catches only ValueError / SyntaxError
        return f"Error: cannot evaluate {expression!r}"


def number(s: str) -> Optional[float]:
    try:
        return float(_calculator()(s.replace(",", "").replace("$", "").strip()))
    except Exception:
        return None


def correct(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    m = CHOICE.match(gold)
    if m:  # multiple choice
        return pred.strip()[:1] == m.group(1)
    a, b = number(pred), number(gold)
    return a is not None and b is not None and abs(a - b) <= 1e-4 * max(1.0, abs(b))


def step_values(text: str) -> List[float]:
    """The calculator results in a text (tool outputs `<result>...</result>`), as numbers; errors skipped."""
    vals = [number(r) for r in RESULT.findall(text) if not r.startswith("Error")]
    return [v for v in vals if v is not None]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-4 * max(1.0, abs(b))


def step_credit(output: str, calls: int, item: Dict[str, Any]) -> float:
    """Share of the reference solution's calculator steps that the model also computed: one-to-one matches between the
    model's tool results and the reference's (within 1e-4 relative), excluding values written in the question (restating
    a given number is not a step), divided by max(model calls, reference steps), so extra calls lower it. 0 when the
    reference has no steps."""
    gold = [v for v in item.get("steps", []) if not any(_close(v, q) for q in item.get("qnums", []))]
    if not gold:
        return 0.0
    left = list(gold)
    matched = 0
    for v in step_values(output):
        for j, g in enumerate(left):
            if _close(v, g):
                matched += 1
                del left[j]
                break
    return matched / max(calls, len(item.get("steps", [])), 1)


def reward(pred: Optional[str], output: str, calls: int, item: Dict[str, Any], partial: float = 0.0) -> float:
    """1 for a correct final answer; else `partial` x `step_credit` (0: the binary reward)."""
    if correct(pred, item["gold"]):
        return 1.0
    return partial * step_credit(output, calls, item) if partial else 0.0


def answer(output: str) -> Optional[str]:
    found = ANSWER.findall(output)
    return found[-1] if found else None


def load_questions(path: Any, sources: Optional[Sequence[str]] = None, limit: int = 0,
                   exclude_choice: bool = False, with_steps: bool = False) -> List[Dict[str, Any]]:
    """Calc-X documents (`{split}.docs.jsonl`) as questions: source, id, prompt (the text up to `<assistant>`), gold,
    steps (the reference solution's calculator results, numbers), qnums (the numbers written in the question).
    `limit`: the first N per source. `exclude_choice`: drop multiple-choice questions."""
    items, per = [], {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if sources and d["source"] not in sources:
                continue
            if limit and per.get(d["source"], 0) >= limit:
                continue
            t = d["text"]
            gold = ANSWER.findall(t)[-1]
            if exclude_choice and CHOICE.match(gold):
                continue
            per[d["source"]] = per.get(d["source"], 0) + 1
            cut = t.index("</user>") + len("</user>")
            question = t[t.index("<user>") + len("<user>"):cut - len("</user>")]
            items.append({"source": d["source"], "id": d["id"], "gold": gold, "prompt": t[:cut] + "<assistant>",
                          "steps": [float(v) for v in step_values(t[cut:])] if with_steps else [],
                          "qnums": [float(x) for x in QNUM.findall(question.replace(",", ""))]})
    return items


def load_tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER))


def _select(state: Any, idx: Any) -> Any:
    """Rows `idx` of a model state (nested lists / tuples of batch-first tensors, None kept)."""
    if state is None:
        return None
    if isinstance(state, (list, tuple)):
        return type(state)(_select(s, idx) for s in state)
    return state.index_select(0, idx)


def rollout(m: Any, tok: Any, prompts: Sequence[str], max_new: int = 200, temperature: float = 0.0,
            batch: int = 256, generator: Any = None) -> List[Dict[str, Any]]:
    """Runs the agent loop on every prompt (`load_questions`' "prompt"). Greedy at temperature 0, else sampled from
    softmax(logits / temperature). Returns per prompt: `ids` (`<bos>`, the prompt, then everything after it, tool
    results included), `n_prompt` (ids before the first generated token), `sampled` (bool per generated token: chosen
    by the model, not a forced tool result), `output` (the generated text), `pred`, `calls`, `bad_call` (a `<call>`
    that did not parse), `finished`, `entropy` (mean entropy of the sampled tokens' distributions, nats)."""
    import torch

    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    a_close = tok.token_to_id("</assistant>")
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    out = []
    for b0 in range(0, len(prompts), batch):
        chunk = [[bos] + enc(p) for p in prompts[b0:b0 + batch]]
        B = len(chunk)
        L = max(len(x) for x in chunk)
        x = np.full((B, L), bos, np.int32)
        for i, ids in enumerate(chunk):
            x[i, L - len(ids):] = ids
        gen = [[] for _ in range(B)]
        sampled = [[] for _ in range(B)]
        span = [0] * B  # where the current assistant turn starts in gen
        forced = [[] for _ in range(B)]
        done = [False] * B
        calls, bad = [0] * B, [False] * B
        ent_sum, ent_n = np.zeros(B), np.zeros(B)
        rows = list(range(B))  # chunk index of each row still in the state
        with torch.no_grad():
            logits, state = m.advance(torch.as_tensor(x), None)
            for _ in range(max_new):
                lg = logits.float()
                if temperature > 0:
                    logp = torch.log_softmax(lg / temperature, -1)
                    nxt = torch.multinomial(logp.exp(), 1, generator=generator)[:, 0]
                    ent = -(logp.exp() * logp).sum(-1)
                    nxt, ent = nxt.cpu().numpy(), ent.cpu().numpy()
                else:
                    nxt, ent = torch.argmax(lg, -1).cpu().numpy(), None
                feed = np.full((len(rows), 1), eos, np.int32)
                for j, i in enumerate(rows):
                    if done[i]:
                        continue
                    own = not forced[i]
                    t = int(nxt[j]) if own else forced[i].pop(0)
                    gen[i].append(t)
                    sampled[i].append(own)
                    if own and ent is not None:
                        ent_sum[i] += ent[j]
                        ent_n[i] += 1
                    feed[j, 0] = t
                    if t == eos:
                        done[i] = True
                    elif t == a_close and own:
                        turn = tok.decode(gen[i][span[i]:], skip_special_tokens=False)
                        c = CALL.search(turn)
                        if c:
                            try:
                                res = calc(json.loads(c.group(1)))
                            except ValueError:
                                res = "Error: bad call"
                            calls[i] += 1
                            forced[i] = enc(f"<result>{res}</result><assistant>")
                            span[i] = len(gen[i]) + len(forced[i])
                        else:
                            bad[i] = "<call>" in turn
                            done[i] = True
                live = [j for j, i in enumerate(rows) if not done[i]]
                if not live:
                    break
                if len(rows) - len(live) >= max(8, len(rows) // 4):  # drop the finished rows from the state
                    idx = torch.as_tensor(live, device=logits.device)
                    state = _select(state, idx)
                    rows = [rows[j] for j in live]
                    feed = feed[live]
                logits, state = m.advance(torch.as_tensor(feed), state)
        for i in range(B):
            text = tok.decode(gen[i], skip_special_tokens=False)
            out.append({"ids": chunk[i] + gen[i], "n_prompt": len(chunk[i]), "sampled": sampled[i], "output": text,
                        "pred": answer(text), "calls": calls[i], "bad_call": bad[i], "finished": done[i],
                        "entropy": float(ent_sum[i] / ent_n[i]) if ent_n[i] else None})
    return out
