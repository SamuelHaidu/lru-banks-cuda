"""`agent-mix-v2-bpe1024`: `agent-mix-bpe1024`'s sources plus general knowledge, sentiment, real two-person Discord
chats and synthetic memory / recall / copy tasks, every document with a random mix of (coherent) system prompts and
tool lists, encoded with a new byte-level BPE learned on the mixed train split: 1024 ids in total (16 tags + 256
bytes + 752 merges).

Sources (raw files in datasets/raw/, see the curl commands below), one category each in the token report:
  agent-shell   our synthetic tool-use conversations (`agent-shell-bytes`); a system prompt added to half (shell /
                file-assistant prompts); their tool lists are kept (the calls need them)
  atum09        Atum09/agent-training-dataset as in `agent-mix-bpe1024`; its system prompt dropped from 40%
  general-qa    MuskumPillerum/General-Knowledge (rev dae2c56, 37.6k Q/A): question -> <user>, answer -> <assistant>,
                1-3 Q/A per conversation
  sentiment     Sp1786/multiclass-sentiment-analysis-dataset (rev a7ede74, train / val CSVs): a varied question
                about the text's sentiment -> <user>; half have a system prompt asking for one word (positive,
                negative or neutral) and the answer is that word, the rest a short sentence
  chat          mookiezi/Discord-Dialogues (rev a8b2294): 10k English conversations between two people with at
                least 2 messages each (the dataset keeps two-author chains only); <|im_start|>user / assistant ->
                <user> / <assistant>
  memory        synthetic: save a fact to persistent memory (a tool call), recall it later in the conversation
                (a copy from far back in the context, after unrelated turns), recall it from memory in a new
                conversation (a call, then a copy from the result), several facts at once, and copy tasks
                (repeat a text / code exactly, an item of a list by position or by key)

System prompts and tools: in every category, about a quarter of the documents have neither, a quarter a system
prompt only, a quarter tools only and a quarter both (when the conversation calls a tool its list is always there);
system prompts are drawn from prompts written for that kind of task (a general-knowledge question never gets a
shell-assistant prompt), tool lists add tools the conversation does not need. So the model cannot learn that
answering depends on a system prompt or a tool list being present.

Loss: inside <assistant> ... </assistant> (`loss_spans`), as in the other agent datasets.

    R=datasets/raw
    curl -L -o $R/general-knowledge/output.json \\
      https://huggingface.co/datasets/MuskumPillerum/General-Knowledge/resolve/dae2c562dd24a2e36f7fafd4d2ab539bea1f973e/output.json
    curl -L -o $R/sentiment-sp1786/train_df.csv \\
      https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset/resolve/a7ede749d4b3e7ef516603cad80bd6b39b0a1fca/train_df.csv
      (same for val_df.csv)
    curl -L -o $R/discord-dialogues/train.parquet \\
      https://huggingface.co/datasets/mookiezi/Discord-Dialogues/resolve/a8b2294bd5b4acfe4ce537b688e7eee111c50fe2/data/train.parquet
    (atum09: see build_agent_mix_bpe.py; agent-shell-bytes: build_agent_shell.py)

    uv run --with pyarrow --with pandas python datasets/build_agent_mix_v2.py --samples 2   # print documents
    uv run --with pyarrow --with pandas python datasets/build_agent_mix_v2.py               # -> datasets/agent-mix-v2-bpe1024/
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_agent_mix_bpe as mix  # noqa: E402
import build_agent_shell as shell_ds  # noqa: E402
from build_agent_shell_bpe import documents, train_tokenizer  # noqa: E402

ROOT = HERE.parent
RAW = HERE / "raw"
SPECIAL = mix.SPECIAL
BYTE0 = len(SPECIAL)
TAG_RE = re.compile("|".join(re.escape(t) for t in SPECIAL))

# -- system prompts, by kind of task ---------------------------------------------------------------------------------

SYSTEM = {
    "general": [
        "You are a helpful assistant.",
        "You are a helpful assistant. Answer clearly and concisely.",
        "You are a knowledgeable assistant. Answer the user's questions accurately.",
        "You are an assistant that answers general knowledge questions with clear, factual explanations.",
        "Answer the user's questions. Be accurate and keep the answers short.",
        "You are a friendly and knowledgeable assistant. Explain things simply.",
        "You are a helpful assistant. Use a tool only when the question needs one; otherwise answer directly.",
    ],
    "shell": [
        "You are an assistant with access to a Linux workspace. Use the tools to inspect and change files when needed.",
        "You are a command-line assistant. Run commands and read files with the available tools to complete the "
        "user's tasks, then report the result.",
        "You help the user with files, shell commands, calculations and small programs, using the listed tools.",
        "You are a helpful assistant with tools. Call a tool when the task needs it and answer from its result.",
    ],
    "sentiment_word": [
        "Answer with a single word: positive, negative or neutral.",
        "You are a sentiment classifier. Reply with exactly one word: positive, neutral or negative.",
        "Respond only with one of these words: positive, negative, neutral.",
        "Classify the sentiment of the text you are given. Your answer must be a single word, one of: positive, "
        "negative, neutral.",
        "Reply with one word only. The only possible answers are positive, negative and neutral.",
    ],
    "sentiment": [
        "You are a helpful assistant that analyzes the tone of texts.",
        "You classify texts by sentiment: positive, negative or neutral.",
        "You are a helpful assistant.",
    ],
    "chat": [
        "You are chatting casually with someone on Discord. Keep it relaxed and natural.",
        "You are a friendly person having an informal online conversation.",
        "Have a casual conversation with the user, like friends chatting online.",
        "You are a laid-back chat partner. Reply naturally and briefly.",
    ],
    "memory": [
        "You are a personal assistant with persistent memory. Save what the user asks you to remember with the "
        "memory tool and look it up when they ask about it later.",
        "You can store and retrieve facts in long-term memory with the listed tools. Use them when the user asks you "
        "to remember or recall something.",
        "You are a helpful assistant with a memory. When asked to remember something, save it; when asked about "
        "something saved earlier, answer from the conversation or look it up in memory.",
    ],
    "copy": [
        "You are a helpful assistant. Copy text, codes and values exactly when asked.",
        "When the user asks you to repeat or copy something, reproduce it exactly, character for character.",
        "You are a precise assistant. Answer with exactly the requested text or value and nothing else.",
    ],
}

# tools a conversation can list without needing them (besides those of the atum09 and shell conversations)
EXTRA_TOOLS = [
    "search_web(query: str): search the internet",
    "get_weather(location: str): current weather for a location",
    "calculator(expression: str): evaluate an arithmetic expression",
    "send_email(to: str, subject: str, body: str): send an email",
    "read_file(path: str): return the contents of a file",
    "python(code: str): run Python code and return what it prints",
    "translate(text: str, target_language: str): translate text into another language",
    "set_timer(minutes: int): start a countdown timer",
    "create_calendar_event(title: str, date: str, time: str): add an event to the calendar",
    "get_stock_price(ticker: str): latest price of a stock",
    "play_music(query: str): play a song or playlist",
    "convert_currency(amount: float, from_currency: str, to_currency: str): convert between currencies",
]
# memory tools: several naming conventions, so the model reads the tool list instead of memorizing one name
MEMORY_TOOLS = [
    ("memory_save", "memory_search", "memory_save(key: str, value: str): save a fact to persistent memory",
     "memory_search(key: str): look up a fact in persistent memory"),
    ("memory_store", "memory_recall", "memory_store(key: str, value: str): store a value in long-term memory under a key",
     "memory_recall(key: str): return the value stored under a key"),
    ("remember", "recall", "remember(key: str, value: str): remember something for later conversations",
     "recall(key: str): retrieve something remembered earlier"),
    ("save_note", "get_note", "save_note(key: str, value: str): save a note that persists between conversations",
     "get_note(key: str): read a saved note"),
]


def q(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def tools_block(lines: List[str]) -> str:
    return "<tools>\n" + "".join(l + "\n" for l in lines) + "</tools>"


def frame(rng: random.Random, kind: str, needed_tools: List[str], tool_pool: List[str],
          p_system: float = 0.5, p_tools: float = 0.5, system_text: Optional[str] = None) -> str:
    """`<bos>` + an optional system prompt for `kind` + a tool list (always when `needed_tools`, else with p_tools),
    with 0-4 unneeded tools mixed in."""
    out = "<bos>"
    if system_text is not None or rng.random() < p_system:
        out += f"<system>{system_text or rng.choice(SYSTEM[kind])}</system>"
    if needed_tools or rng.random() < p_tools:
        extra = rng.sample(tool_pool, rng.randint(0 if needed_tools else 1, 4))
        lines = list(dict.fromkeys(needed_tools + extra))
        rng.shuffle(lines)
        out += tools_block(lines)
    return out


def clean(s: str) -> str:
    return s.replace("\r\n", "\n").strip()


def has_tags(*texts: str) -> bool:
    return any(TAG_RE.search(t) for t in texts)


# -- general knowledge ----------------------------------------------------------------------------------------------

def load_general_qa() -> List[Tuple[str, str]]:
    rows = json.loads((RAW / "general-knowledge" / "output.json").read_text())
    seen, out = set(), []
    for r in rows:
        qn, an = clean(r.get("Question") or ""), clean((r.get("Answer") or "").replace("\\n", "\n"))
        if not qn or not an or has_tags(qn, an) or (qn, an) in seen:
            continue
        seen.add((qn, an))
        out.append((qn, an))
    return out


def general_docs(pairs: List[Tuple[str, str]], rng: random.Random, tool_pool: List[str]) -> List[str]:
    docs, i = [], 0
    while i < len(pairs):
        n = rng.choice((1, 1, 2, 3))
        body = "".join(f"<user>{qn}</user><assistant>{an}</assistant>" for qn, an in pairs[i:i + n])
        docs.append(frame(rng, "general", [], tool_pool) + body + "<eos>")
        i += n
    return docs


# -- sentiment ------------------------------------------------------------------------------------------------------

SENT_ASK = [
    "Is this sentence negative, positive or neutral?\n\n{t}",
    "Is this sentence negative, positive or neutral sentiment? \"{t}\"",
    "What is the sentiment of this text: \"{t}\"",
    "Classify the sentiment (positive, negative, or neutral): {t}",
    "{t}\n\nIs the sentiment of the text above positive, negative or neutral?",
    "Tell me if this is positive, neutral or negative: {t}",
    "Sentiment of this message? {t}",
    "Read this and tell me whether it sounds positive, negative or neutral:\n{t}",
    "Is the following positive, negative, or neutral?\n{t}",
    "What's the tone here, positive, negative or neutral? \"{t}\"",
    "Label the sentiment of: {t}",
    "Would you say this is a positive, negative or neutral statement? {t}",
    "Text: {t}\nSentiment (positive / negative / neutral)?",
    "can you check the sentiment of this one: {t}",
]
SENT_SAY = {
    "positive": ["Positive.", "The sentiment is positive.", "It's positive.", "This text is positive.",
                 "Positive: the tone is upbeat.", "I'd say positive."],
    "negative": ["Negative.", "The sentiment is negative.", "It's negative.", "This text is negative.",
                 "Negative: the tone is unhappy.", "I'd say negative."],
    "neutral": ["Neutral.", "The sentiment is neutral.", "It's neutral.", "This text is neutral.",
                "Neutral: no clear positive or negative tone.", "I'd say neutral."],
}


def load_sentiment(split: str) -> List[Tuple[str, str]]:
    out = []
    with open(RAW / "sentiment-sp1786" / f"{split}_df.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t, lab = clean(r["text"] or ""), (r["sentiment"] or "").strip().lower()
            if t and lab in SENT_SAY and not has_tags(t):
                out.append((t, lab))
    return out


def sentiment_docs(rows: List[Tuple[str, str]], rng: random.Random, tool_pool: List[str]) -> List[str]:
    docs = []
    for t, lab in rows:
        ask = rng.choice(SENT_ASK).format(t=t)
        u = rng.random()
        if u < 0.5:  # half: a system prompt asking for a single word, answered with the word
            head = frame(rng, "sentiment_word", [], tool_pool, system_text=rng.choice(SYSTEM["sentiment_word"]))
            ans = lab
        else:
            head = frame(rng, "sentiment", [], tool_pool, p_system=0.5)
            ans = rng.choice(SENT_SAY[lab])
        docs.append(head + f"<user>{ask}</user><assistant>{ans}</assistant><eos>")
    return docs


# -- Discord chats --------------------------------------------------------------------------------------------------

_MSG = re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", re.S)
_EN_WORDS = set("""the a an and or but if so to of in on at for with is are was were be been it this that i you he she
we they me my your our not no yes do does did have has had what why how when where who can could would should will
just like im its dont lol yeah ok okay oh really think know get got go going one all about there here""".split())


def english(text: str) -> bool:
    """ASCII text with a share of common English words (the dataset is mostly English, some other languages)."""
    if sum(ord(c) > 127 for c in text) > 0.03 * len(text):
        return False
    words = re.findall(r"[a-z']+", text.lower())
    return len(words) >= 8 and sum(w.replace("'", "") in _EN_WORDS for w in words) / len(words) >= 0.2


def load_chats(n_train: int, n_val: int, seed: int) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
    import pandas as pd

    df = pd.read_parquet(RAW / "discord-dialogues" / "train.parquet", columns=["text", "turns"])
    df = df[df.turns >= 4]  # at least two messages from each person
    order = np.random.default_rng(seed).permutation(len(df))
    texts = df.text.to_numpy()
    out = []
    for i in order:
        msgs = [(r, clean(m)) for r, m in _MSG.findall(texts[i])]
        while msgs and msgs[0][0] != "user":
            msgs = msgs[1:]
        while msgs and msgs[-1][0] != "assistant":
            msgs = msgs[:-1]
        roles = [r for r, _ in msgs]
        if (len(msgs) < 4 or any(a == b for a, b in zip(roles, roles[1:])) or any(not m for _, m in msgs)
                or has_tags(*[m for _, m in msgs]) or not english(" ".join(m for _, m in msgs))):
            continue
        out.append(msgs)
        if len(out) == n_train + n_val:
            break
    return out[:n_train], out[n_train:]


def chat_docs(convs: List[List[Tuple[str, str]]], rng: random.Random, tool_pool: List[str]) -> List[str]:
    return [frame(rng, "chat", [], tool_pool, p_system=0.5, p_tools=0.3)
            + "".join(f"<{r}>{m}</{r}>" for r, m in c) + "<eos>" for c in convs]


# -- memory, recall and copy tasks ----------------------------------------------------------------------------------

FIRST = ["Ana", "Ben", "Carla", "Dev", "Elif", "Farid", "Greta", "Hugo", "Ines", "Jonas", "Kemi", "Lars", "Mira",
         "Nico", "Olga", "Pavel", "Rosa", "Sami", "Tara", "Umar", "Vera", "Wen", "Yara", "Zeno"]
WORDS = ["maple", "river", "falcon", "copper", "lantern", "orchid", "granite", "velvet", "harbor", "comet", "willow",
         "ember", "quartz", "meadow", "cobalt", "tundra", "saffron", "pixel", "summit", "violet"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def _code(rng: random.Random) -> str:
    L = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{rng.choice(L)}{rng.choice(L)}-{rng.randint(1000, 9999)}"


FACTS: List[Tuple[str, List[str], Callable[[random.Random], str]]] = [
    # key, ways the user refers to it, value generator
    ("wifi password", ["my wifi password", "the wifi password"], lambda r: f"{r.choice(WORDS)}-{r.randint(100, 9999)}"),
    ("locker code", ["my locker code", "the code for my locker"], _code),
    ("parking spot", ["where I parked", "my parking spot"],
     lambda r: f"level {r.randint(1, 6)}, spot {r.choice('ABCDEFG')}{r.randint(1, 60)}"),
    ("flight number", ["my flight number", "which flight I'm on"],
     lambda r: f"{r.choice(['LH', 'BA', 'AF', 'TP', 'KL', 'IB'])} {r.randint(100, 9999)}"),
    ("dentist appointment", ["my dentist appointment", "when I see the dentist"],
     lambda r: f"{r.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])} at {r.randint(8, 17)}:"
               f"{r.choice(['00', '15', '30', '45'])}"),
    ("mom's birthday", ["my mom's birthday", "when my mother's birthday is"],
     lambda r: f"{r.choice(MONTHS)} {r.randint(1, 28)}"),
    ("gym membership number", ["my gym membership number", "my gym member ID"],
     lambda r: str(r.randint(10_000_000, 99_999_999))),
    ("license plate", ["my license plate", "my car's plate number"],
     lambda r: f"{''.join(r.choice('ABCDEFGHJKLMNPRSTUVWXYZ') for _ in range(3))} {r.randint(100, 9999)}"),
    ("meeting room", ["the meeting room", "where the meeting is"], lambda r: f"Room {r.randint(1, 9)}{r.choice('ABCD')}"),
    ("project codename", ["the project codename", "what we call the project"],
     lambda r: f"{r.choice(WORDS).title()} {r.choice(WORDS).title()}"),
    ("server IP", ["the server IP", "the server's address"],
     lambda r: f"10.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}"),
    ("order number", ["my order number", "the order number"], lambda r: f"#{r.randint(10000, 99999)}-{r.randint(10, 99)}"),
    ("vet's name", ["my dog's vet", "the vet's name"], lambda r: f"Dr. {r.choice(FIRST)} {r.choice(WORDS).title()}"),
    ("train departure", ["when my train leaves", "my train time"],
     lambda r: f"{r.randint(5, 22):02d}:{r.randint(0, 59):02d} from platform {r.randint(1, 14)}"),
    ("apartment number", ["my apartment number", "which apartment I live in"], lambda r: f"{r.randint(1, 30)}{r.choice('ABCDEF')}"),
    ("emergency contact", ["my emergency contact", "who to call in an emergency"],
     lambda r: f"{r.choice(FIRST)}, {r.randint(200, 999)}-{r.randint(100, 999)}-{r.randint(1000, 9999)}"),
]

ASK_STORE = ["Please remember that {ref} is {v}.", "Remember this: {ref} is {v}.", "Can you save {ref}? It's {v}.",
             "Note for later: {ref} is {v}.", "Store this for me: {ref} = {v}", "Keep in mind that {ref} is {v}.",
             "{ref} is {v}. Please remember it."]
ASK_RECALL = ["What's {ref}?", "Do you remember {ref}?", "Remind me, what is {ref}?", "What did I tell you about {ref}?",
              "I forgot {ref}. What was it?", "Can you tell me {ref}?"]
CONFIRM = ["Saved: {ref} is {v}.", "Got it. I'll remember that {ref} is {v}.", "Done, {ref} ({v}) is saved.",
           "Noted: {v}."]
ANSWER = ["{V}.", "It's {v}.", "{Ref} is {v}.", "You told me {ref} is {v}.", "{v}"]
NOT_FOUND = ["I don't have {ref} saved. Could you tell me?", "I couldn't find {ref} in my memory.",
             "Nothing is saved under {ref} yet."]


def _you(ref: str) -> str:
    """The user's words seen from the assistant: "my X" -> "your X", "I'm" -> "you're"."""
    out = re.sub(r"\bmy\b", "your", ref)
    out = re.sub(r"\bI'm\b", "you're", out)
    out = re.sub(r"\bI\b", "you", out)
    return out


def _fact(rng: random.Random) -> Tuple[str, str, str]:
    key, refs, gen = rng.choice(FACTS)
    return key, rng.choice(refs), gen(rng)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _say(template: str, ref: str, v: str) -> str:
    r = _you(ref)
    return template.format(ref=r, Ref=r[:1].upper() + r[1:], v=v, V=v[:1].upper() + v[1:])


def memory_docs(n: int, rng: random.Random, tool_pool: List[str], fillers: List[Tuple[str, str]]) -> List[str]:
    """Save / recall / copy conversations. `fillers`: (user, assistant) turns put between a save and its recall."""
    docs = []
    for _ in range(n):
        save, search, save_line, search_line = rng.choice(MEMORY_TOOLS)
        task = rng.choices(["save", "save_recall_context", "recall_tool", "multi", "copy"], [2, 3, 2.5, 1, 1.5])[0]
        body = ""
        tools: List[str] = []
        if task in ("save", "save_recall_context", "multi"):
            tools = [save_line, search_line] if rng.random() < 0.7 else [save_line]
            facts = [_fact(rng) for _ in range(rng.randint(2, 4) if task == "multi" else 1)]
            facts = list({k: (k, r, v) for k, r, v in facts}.values())
            if len(facts) == 1:
                k, ref, v = facts[0]
                body += f"<user>{_cap(rng.choice(ASK_STORE).format(ref=ref, v=v))}</user>"
            else:
                body += "<user>Please remember these:\n" + "".join(f"- {ref}: {v}\n" for _, ref, v in facts) + "</user>"
            body += "<assistant>" + "".join(f"<call>{save}(key={q(k)}, value={q(v)})</call>" for k, _, v in facts)
            body += "</assistant>" + "".join(f'<result>{{"saved": true, "key": {q(k)}}}</result>' for k, _, _ in facts)
            body += ("<assistant>" + (_say(rng.choice(CONFIRM), facts[0][1], facts[0][2]) if len(facts) == 1
                                      else "Saved both." if len(facts) == 2 else f"Saved all {len(facts)}.")
                     + "</assistant>")
            if task != "save":
                for qn, an in rng.sample(fillers, rng.randint(0, 4)):
                    body += f"<user>{qn}</user><assistant>{an}</assistant>"
                k, ref, v = rng.choice(facts)
                body += f"<user>{_cap(rng.choice(ASK_RECALL).format(ref=ref))}</user>"
                body += f"<assistant>{_say(rng.choice(ANSWER), ref, v)}</assistant>"
        elif task == "recall_tool":  # a new conversation: the fact is only in memory
            tools = [search_line] + ([save_line] if rng.random() < 0.5 else [])
            for qn, an in rng.sample(fillers, rng.randint(0, 2)):
                body += f"<user>{qn}</user><assistant>{an}</assistant>"
            k, ref, v = _fact(rng)
            body += f"<user>{_cap(rng.choice(ASK_RECALL).format(ref=ref))}</user>"
            body += f"<assistant><call>{search}(key={q(k)})</call></assistant>"
            if rng.random() < 0.85:
                body += f'<result>{{"key": {q(k)}, "value": {q(v)}}}</result>'
                body += f"<assistant>{_say(rng.choice(ANSWER), ref, v)}</assistant>"
            else:
                body += f'<result>{{"error": "not found", "key": {q(k)}}}</result>'
                body += f"<assistant>{_say(rng.choice(NOT_FOUND), ref, '')}</assistant>"
        else:  # copy
            kind = rng.choice(["repeat", "code", "index", "lookup"])
            if kind == "repeat":
                text = rng.choice([an for _, an in rng.sample(fillers, 20) if len(an) > 40] or ["The quick brown fox."])
                text = text.split("\n")[0][:300]
                ask = rng.choice(["Repeat this exactly: {t}", "Copy this text back to me word for word:\n{t}",
                                  "Say exactly this: \"{t}\"", "Write the following back unchanged:\n{t}"])
                body += f"<user>{ask.format(t=text)}</user><assistant>{text}</assistant>"
            elif kind == "code":
                codes = ["".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(rng.randint(6, 14)))
                         for _ in range(rng.randint(1, 3))]
                ask = rng.choice(["Copy these codes exactly, one per line:\n{c}", "Repeat back: {c}",
                                  "Type these back to me: {c}"])
                sep = "\n" if "line" in ask else " "
                body += f"<user>{ask.format(c=sep.join(codes))}</user><assistant>{sep.join(codes)}</assistant>"
            elif kind == "index":
                items = rng.sample(WORDS + FIRST, rng.randint(3, 9))
                i = rng.randrange(len(items))
                nth = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"][i]
                body += (f"<user>Here is a list: {', '.join(items)}. What is the {nth} item?</user>"
                         f"<assistant>{items[i]}</assistant>")
            else:
                facts = [_fact(rng) for _ in range(rng.randint(3, 6))]
                facts = list({k: (k, r, v) for k, r, v in facts}.values())
                k, ref, v = rng.choice(facts)
                body += ("<user>" + "\n".join(f"{kk}: {vv}" for kk, _, vv in facts) + f"\n\nWhat is the {k}?</user>"
                         f"<assistant>{v}</assistant>")
        docs.append(frame(rng, "copy" if task == "copy" else "memory", tools, tool_pool) + body + "<eos>")
    return docs


# -- the existing sources, with system prompts mixed --------------------------------------------------------------

def shell_mix(docs: List[str], rng: random.Random) -> List[str]:
    out = []
    for d in docs:
        assert d.startswith("<bos>")
        if rng.random() < 0.5:
            d = f"<bos><system>{rng.choice(SYSTEM['shell'])}</system>" + d[len("<bos>"):]
        out.append(d)
    return out


def atum_mix(docs: List[str], rng: random.Random) -> List[str]:
    return [re.sub(r"<system>.*?</system>", "", d, count=1, flags=re.S) if rng.random() < 0.4 else d for d in docs]


def tool_lines(docs: List[str]) -> List[str]:
    lines = set()
    for d in docs:
        m = re.search(r"<tools>\n(.*?)</tools>", d, re.S)
        if m:
            lines.update(l for l in m.group(1).split("\n") if l)
    return sorted(lines)


# -- build ----------------------------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=1024, help="ids in total: 16 tags + 256 bytes + merges")
    p.add_argument("--chats", type=int, default=10_000)
    p.add_argument("--memory", type=int, default=40_000, help="synthetic memory / recall / copy conversations")
    p.add_argument("--samples", type=int, default=0, help="print this many documents of each category and exit")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = random.Random(a.seed)
    stats = collections.Counter()

    atum_train = mix.load_atum("train", stats)
    atum_val = mix.load_atum("val", stats, exclude=set(atum_train))
    shell_train = documents(HERE / shell_ds.NAME / "train.npy")
    shell_val = documents(HERE / shell_ds.NAME / "validation.npy")
    pool = sorted(set(EXTRA_TOOLS) | set(tool_lines(atum_train)) | set(tool_lines(shell_train)))

    qa = load_general_qa()
    rng.shuffle(qa)
    n_qa_val = max(1, len(qa) // 30)
    qa_val, qa_train = qa[:n_qa_val], qa[n_qa_val:]
    sent_train, sent_val = load_sentiment("train"), load_sentiment("val")
    chats_train, chats_val = load_chats(a.chats, max(1, a.chats // 20), a.seed)
    fillers_train = qa_train + [(rng.choice(SENT_ASK).format(t=t), rng.choice(SENT_SAY[l])) for t, l in sent_train]
    fillers_val = qa_val + [(rng.choice(SENT_ASK).format(t=t), rng.choice(SENT_SAY[l])) for t, l in sent_val]

    cats = {
        "agent-shell": (shell_mix(shell_train, rng), shell_mix(shell_val, rng)),
        "atum09": (atum_mix(atum_train, rng), atum_mix(atum_val, rng)),
        "general-qa": (general_docs(qa_train, rng, pool), general_docs(qa_val, rng, pool)),
        "sentiment": (sentiment_docs(sent_train, rng, pool), sentiment_docs(sent_val, rng, pool)),
        "chat": (chat_docs(chats_train, rng, pool), chat_docs(chats_val, rng, pool)),
        "memory": (memory_docs(a.memory, rng, pool, fillers_train),
                   memory_docs(max(1, a.memory // 40), random.Random(a.seed + 1), pool, fillers_val)),
    }
    if a.samples:
        for name, (tr, _) in cats.items():
            for d in rng.sample(tr, a.samples):
                print(f"=== {name}\n{d}\n")
        return

    train = [(c, d) for c, (tr, _) in cats.items() for d in tr]
    val = [(c, d) for c, (_, va) in cats.items() for d in va]
    rng.shuffle(train)
    rng.shuffle(val)

    name = f"agent-mix-v2-bpe{a.vocab_size}"
    out_dir, tok_dir = HERE / name, ROOT / "tokenizers" / name
    tok, n_merges = train_tokenizer([d for _, d in train], a.vocab_size, SPECIAL)
    V = tok.get_vocab_size()
    if V != a.vocab_size or n_merges != a.vocab_size - BYTE0 - 256:
        raise SystemExit(f"vocabulary {V} with {n_merges} merges, expected {a.vocab_size} ids")
    for _, s in train[:300] + val[:300]:  # lossless, the tags encode as tags
        ids = tok.encode(s).ids
        assert tok.decode(ids, skip_special_tokens=False) == s
        assert [SPECIAL[i] for i in ids if i < BYTE0] == TAG_RE.findall(s)
    assert tok.encode("\x00").ids == [BYTE0] and tok.encode("é").ids == [BYTE0 + 0xC3, BYTE0 + 0xA9]
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(tok_dir / "tokenizer.json"))

    open_id, close_id = SPECIAL.index("<assistant>"), SPECIAL.index("</assistant>")
    cat_ids = {c: i for i, c in enumerate(cats)}
    qs = lambda xs: {k: int(np.percentile(xs, v)) for k, v in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))}
    report: Dict[str, Dict[str, Dict[str, int]]] = {c: {} for c in cats}
    splits = {}
    span_bytes = span_tokens = 0
    for split, docs in (("train", train), ("validation", val)):
        enc = [e.ids for e in tok.encode_batch([d for _, d in docs])]
        arr = np.concatenate([np.asarray(e, dtype=np.int32) for e in enc])
        np.save(out_dir / f"{split}.npy", arr)
        # per-document index (start token, length, category id), to rebalance the categories without rebuilding
        lengths = np.asarray([len(e) for e in enc], dtype=np.int64)
        index = np.stack([np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths,
                          np.asarray([cat_ids[c] for c, _ in docs], dtype=np.int64)], 1)
        np.save(out_dir / f"{split}.categories.npy", index)
        n_bytes = sum(len(TAG_RE.sub("\x00", d).encode("utf-8")) for _, d in docs)
        splits[split] = {"file": f"{split}.npy", "tokens": int(arr.size), "documents": len(docs),
                         "doc_tokens": qs([len(e) for e in enc]), "bytes_per_token": round(n_bytes / arr.size, 4)}
        per = collections.defaultdict(lambda: collections.Counter())
        for (c, _), e in zip(docs, enc):
            inside, scored = False, 0
            for i in e:
                if inside:
                    scored += 1
                    if split == "validation":
                        span_tokens += 1
                        span_bytes += 1 if i < BYTE0 else len(tok.decode([i]).encode("utf-8"))
                inside = (inside or i == open_id) and i != close_id
            per[c]["documents"] += 1
            per[c]["tokens"] += len(e)
            per[c]["assistant_tokens"] += scored
        for c in cats:
            report[c][split] = dict(per[c])
    meta = {
        "name": name,
        "source": "agent-shell-bytes + Atum09 (as agent-mix-bpe1024) + MuskumPillerum/General-Knowledge (rev dae2c56) "
                  "+ Sp1786/multiclass-sentiment-analysis-dataset (rev a7ede74) + mookiezi/Discord-Dialogues (rev "
                  "a8b2294, 2-person English multi-turn sample) + synthetic memory / recall / copy tasks; system "
                  "prompts and tool lists mixed (datasets/build_agent_mix_v2.py)",
        "categories": report,
        # {split}.categories.npy: one row per document, (start token, length, category id), ids as listed here
        "category_ids": cat_ids,
        "category_index": "{split}.categories.npy",
        "tokenizer": name,
        "vocabulary_size": V,
        "doc_token_id": 0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "special_tokens": {s: i for i, s in enumerate(SPECIAL)},
        "loss_spans": [[open_id, close_id]],
        "merges": n_merges,
        "avg_chars_per_token": round(span_bytes / span_tokens, 4),
        "dtype": "int32",
        "seed": a.seed,
        **splits,
        "notes": f"Token = tag (ids 0-{BYTE0 - 1}), one UTF-8 byte (id {BYTE0} + byte) or a merge "
                 f"(ids {BYTE0 + 256}..{V - 1}).",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {out_dir}/: vocab {V} ({n_merges} merges); train {splits['train']['tokens']:,} tokens "
          f"({len(train):,} docs, {splits['train']['bytes_per_token']} bytes/token), validation "
          f"{splits['validation']['tokens']:,} ({len(val):,} docs); assistant spans {meta['avg_chars_per_token']} "
          f"bytes/token")
    print(f"\n{'category':12s} {'docs':>9s} {'tokens':>12s} {'assistant':>12s} {'share':>7s}   "
          f"{'val docs':>8s} {'val tokens':>11s}")
    tot = splits["train"]["tokens"]
    for c, r in report.items():
        t, v = r["train"], r["validation"]
        print(f"{c:12s} {t['documents']:9,d} {t['tokens']:12,d} {t['assistant_tokens']:12,d} "
              f"{t['tokens'] / tot:7.1%}   {v['documents']:8,d} {v['tokens']:11,d}")
    print(f"{'total':12s} {len(train):9,d} {tot:12,d} "
          f"{sum(r['train']['assistant_tokens'] for r in report.values()):12,d}")


if __name__ == "__main__":
    main()
