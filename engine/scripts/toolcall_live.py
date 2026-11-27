"""Tool use with tools the model has not seen: a few small scenarios (new tool names and schemas, new prompts) in the
agentic format, and a loop that runs the model's calls in Python and feeds the results back, until the model answers
without a call (or a round / call budget ends). Each scenario runs several samples in two modes: free (the model may
reason first) and direct (<think></think> pre-filled at every assistant turn). All rows decode together; a round is
one `lbc generate` call that stops each row at </call>, </assistant> or <user>.

    uv run --no-project --with numpy --with tokenizers python scripts/toolcall_live.py RUN_DIR \
        [--checkpoint latest] [--lbc build/release/lbc] [--samples 4] [--out DIR]

Writes DIR/docs.json for toolcall_chat.py and prints per scenario: solved (the task check on the calls and the final
answer), calls per sample, the share of calls that are valid (declared tool, declared and required arguments) and how
the sample ended.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--checkpoint", default="latest")
ap.add_argument("--lbc", default="build/release/lbc")
ap.add_argument("--tokenizer", default="/work/data/agentic/tokenizer.json")
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--round-tokens", type=int, default=1024)
ap.add_argument("--max-calls", type=int, default=6)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="/tmp/toolcall_live")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tok = Tokenizer.from_file(a.tokenizer)
T = {t: tok.token_to_id(t) for t in ("<bos>", "<assistant>", "</assistant>", "<call>", "</call>", "<think>", "</think>",
                                     "<user>", "<result>", "</result>")}
assert all(v is not None for v in T.values()), T

SYSTEM = ("You are a helpful assistant with access to tools.\n\nIn each turn you can either:\n- Send a message to the "
          "user.\n- Make a tool call.\nYou cannot do both at the same time. Use the tools to answer; do not guess.")


# -- the scenarios: tools as (name, description, {param: (type, description, required)}, function) ------------------
def tool(name, desc, params, fn):
    return {"name": name, "desc": desc, "params": params, "fn": fn}


FORECAST = {"lisbon": {"tomorrow": {"condition": "rain", "chance_of_rain_pct": 80, "high_c": 17, "low_c": 12},
                       "today": {"condition": "cloudy", "chance_of_rain_pct": 20, "high_c": 19, "low_c": 13}}}
FILES = {"notes.txt": "shopping list\nmilk\neggs\nabc123 is the door code\n", "todo.md": "- call mom\n- fix bike\n"}
STOCKS = {"AAPL": 231.54, "MSFT": 418.20, "NVDA": 141.95}
EVENTS = {"2026-12-03": [{"title": "Team sync", "start": "10:00", "end": "11:00"},
                         {"title": "Lunch with Ana", "start": "12:30", "end": "13:30"}]}


def forecast(city, day="today"):
    c = FORECAST.get(city.strip().lower())
    return {"city": city, "day": day, **c[day]} if c and day in c else {"error": f"no forecast for {city!r} {day!r}"}


def read_text_file(path):
    p = path.strip().lstrip("./").split("/")[-1]
    return {"path": path, "content": FILES[p]} if p in FILES else {"error": f"{path}: no such file"}


def calculate(expression):
    if not re.fullmatch(r"[\d\s+\-*/().]+", expression):
        return {"error": "only numbers and + - * / ( ) are allowed"}
    return {"expression": expression, "result": eval(expression)}  # noqa: S307 (checked above: arithmetic only)


def convert_currency(amount, from_currency, to_currency):
    rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "BRL": 5.70}
    f, t = from_currency.upper(), to_currency.upper()
    if f not in rates or t not in rates:
        return {"error": f"unknown currency; known: {sorted(rates)}"}
    return {"amount": amount, "from": f, "to": t, "result": round(float(amount) / rates[f] * rates[t], 2)}


def find_book(title):
    if "dune" in title.lower():
        return {"results": [{"book_id": "B-7731", "title": "Dune", "author": "Frank Herbert", "on_loan_to": "M-2048"}]}
    return {"results": []}


def renew_loan(member_id, book_id):
    if member_id == "M-2048" and book_id == "B-7731":
        return {"status": "renewed", "book_id": book_id, "new_due_date": "2026-12-15"}
    return {"error": "no such loan for this member and book"}


def list_events(date):
    return {"date": date, "events": EVENTS.get(date, [])}


def create_event(title, date, start_time, duration_minutes=60):
    return {"status": "created", "event_id": "E-501", "title": title, "date": date, "start": start_time,
            "duration_minutes": duration_minutes}


def set_thermostat(room, celsius):
    if room not in ("living_room", "bedroom", "kitchen"):
        return {"error": f"unknown room {room!r}"}
    return {"room": room, "target_c": celsius, "status": "ok"}


def stock_price(ticker):
    t = ticker.strip().upper()
    return {"ticker": t, "price_usd": STOCKS[t]} if t in STOCKS else {"error": f"unknown ticker {ticker!r}"}


def send_email(to, subject, body):
    return {"status": "sent"}


EMAIL = tool("send_email", "Send an email.", {"to": ("string", "", True), "subject": ("string", "", True),
                                              "body": ("string", "", True)}, send_email)


def calls_of(s, name):
    return [c["args"] for c in s["calls"] if c["name"] == name]


SCENARIOS = [
    {"name": "weather",
     "tools": [tool("get_forecast", "Weather forecast for a city.",
                    {"city": ("string", "City name.", True), "day": ('"today" | "tomorrow"', "", False)}, forecast),
               EMAIL],
     "user": "Will it rain in Lisbon tomorrow?",
     "expect": 'get_forecast(city="Lisbon", day="tomorrow"), then: yes, 80% chance of rain',
     "check": lambda s: any("lisbon" in str(x.get("city", "")).lower() and x.get("day") == "tomorrow"
                            for x in calls_of(s, "get_forecast")) and re.search(r"\byes\b|\brain", s["final"], re.I)},
    {"name": "file",
     "tools": [tool("read_text_file", "Read a text file from the user's workspace.",
                    {"path": ("string", "File path.", True)}, read_text_file),
               tool("list_directory", "List the files of a directory.", {"path": ("string", "", True)},
                    lambda path: {"files": sorted(FILES)})],
     "user": "Does the file notes.txt contain the text abc?",
     "expect": 'read_text_file(path="notes.txt"), then: yes (line "abc123 is the door code")',
     "check": lambda s: bool(calls_of(s, "read_text_file")) and re.search(r"\byes\b|does contain|contains", s["final"],
                                                                           re.I)},
    {"name": "calc",
     "tools": [tool("calculate", "Evaluate an arithmetic expression.",
                    {"expression": ("string", "For example \"2 * (3 + 4)\".", True)}, calculate)],
     "user": "What is 17 * 23 + 4?",
     "expect": 'calculate(expression="17 * 23 + 4"), then: 395',
     "check": lambda s: bool(calls_of(s, "calculate")) and "395" in s["final"]},
    {"name": "currency",
     "tools": [tool("convert_currency", "Convert an amount between currencies at today's rate.",
                    {"amount": ("number", "", True), "from_currency": ("string", "ISO code, e.g. USD.", True),
                     "to_currency": ("string", "ISO code.", True)}, convert_currency)],
     "user": "How much is 50 US dollars in euros?",
     "expect": 'convert_currency(amount=50, from_currency="USD", to_currency="EUR"), then: 46 euros',
     "check": lambda s: bool(calls_of(s, "convert_currency")) and re.search(r"\b46\b|46\.0", s["final"])},
    {"name": "library",
     "tools": [tool("find_book", "Search the library catalogue by title.", {"title": ("string", "", True)}, find_book),
               tool("renew_loan", "Renew a member's loan of a book for two more weeks.",
                    {"member_id": ("string", "", True), "book_id": ("string", "", True)}, renew_loan)],
     "user": "Hi, my member id is M-2048. Please renew my loan of the book Dune.",
     "expect": 'find_book(title="Dune") -> B-7731, renew_loan(member_id="M-2048", book_id="B-7731"), '
               'then: renewed, due 2026-12-15',
     "check": lambda s: any(x.get("book_id") == "B-7731" and x.get("member_id") == "M-2048"
                            for x in calls_of(s, "renew_loan")) and re.search(r"2026-12-15|December 15|15 December",
                                                                              s["final"])},
    {"name": "calendar",
     "tools": [tool("list_events", "List the user's calendar events on a date.",
                    {"date": ("string", "YYYY-MM-DD.", True)}, list_events),
               tool("create_event", "Add an event to the user's calendar.",
                    {"title": ("string", "", True), "date": ("string", "YYYY-MM-DD.", True),
                     "start_time": ("string", "HH:MM, 24 h.", True), "duration_minutes": ("integer", "", False)},
                    create_event)],
     "user": "Am I free on 2026-12-03 at 15:00? If I am, add a dentist appointment then.",
     "expect": 'list_events(date="2026-12-03") (busy only 10:00-11:00 and 12:30-13:30), '
               'create_event(title="Dentist", date="2026-12-03", start_time="15:00")',
     "check": lambda s: any(x.get("date") == "2026-12-03" for x in calls_of(s, "list_events"))
                        and any(x.get("start_time") == "15:00" and x.get("date") == "2026-12-03"
                                for x in calls_of(s, "create_event"))},
    {"name": "thermostat",
     "tools": [tool("set_thermostat", "Set the target temperature of a room.",
                    {"room": ('"living_room" | "bedroom" | "kitchen"', "", True), "celsius": ("number", "", True)},
                    set_thermostat), EMAIL],
     "user": "Set the bedroom to 19 degrees please.",
     "expect": 'set_thermostat(room="bedroom", celsius=19), then a confirmation',
     "check": lambda s: any(x.get("room") == "bedroom" and x.get("celsius") in (19, 19.0, "19")
                            for x in calls_of(s, "set_thermostat"))},
    {"name": "stocks",
     "tools": [tool("get_stock_price", "Latest price of a stock.", {"ticker": ("string", "e.g. AAPL.", True)},
                    stock_price)],
     "user": "Which costs more right now, one Apple share or one Microsoft share?",
     "expect": 'get_stock_price(ticker="AAPL") 231.54, get_stock_price(ticker="MSFT") 418.20, then: Microsoft',
     "check": lambda s: {"AAPL", "MSFT"} <= {str(x.get("ticker", "")).upper() for x in calls_of(s, "get_stock_price")}
                        and re.search(r"microsoft|msft", s["final"], re.I)},
]


def render_tools(tools):
    lines = []
    for t in tools:
        sig = ", ".join(f"{k}{'' if req else '?'}: {ty}" for k, (ty, _, req) in t["params"].items())
        lines.append(f"{t['name']}({sig}): {t['desc']}")
        lines += [f"  {k}: {d}" for k, (_, d, _) in t["params"].items() if d]
    return "<tools>\n" + "\n".join(lines) + "\n</tools>"


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


def execute(sc, body):
    """(result, record) for a call body: the record holds the name, args and validity."""
    pc = parse_call(body)
    if not pc:
        return {"error": "could not parse the call; use name(key=<JSON value>, ...)"}, {"name": None, "args": {},
                                                                                         "valid": False}
    name, args = pc
    t = next((t for t in sc["tools"] if t["name"] == name), None)
    if not t:
        return {"error": f"unknown tool {name!r}"}, {"name": name, "args": args, "valid": False}
    req = {k for k, (_, _, r) in t["params"].items() if r}
    valid = set(args) <= set(t["params"]) and req <= set(args)
    if not valid:
        return ({"error": f"bad arguments; expected {render_tools([t]).splitlines()[1]}"},
                {"name": name, "args": args, "valid": False})
    try:
        res = t["fn"](**args)
    except Exception as e:  # a wrong value type, for example
        res = {"error": f"{type(e).__name__}: {e}"}
    return res, {"name": name, "args": args, "valid": True}


# -- the loop -------------------------------------------------------------------------------------------------------
def enc(text):
    return tok.encode(text, add_special_tokens=False).ids


rows = []
for sc in SCENARIOS:
    prompt = f"<bos><system>{SYSTEM}</system>{render_tools(sc['tools'])}<user>{sc['user']}</user><assistant>"
    ids = enc(prompt)
    assert ids[0] == T["<bos>"] and ids[-1] == T["<assistant>"], "special tokens did not encode as single ids"
    for mode in ("free", "direct"):
        pre = [T["<think>"], T["</think>"]] if mode == "direct" else []
        for k in range(a.samples):
            rows.append({"sc": sc, "mode": mode, "sample": k, "prompt": prompt, "n_prompt": len(ids), "pre": pre,
                         "ids": ids + pre, "calls": [], "end": None, "tokens": 0, "final": ""})

stop = f"{T['</call>']},{T['</assistant>']},{T['<user>']}"
for rnd in range(a.max_calls + 1):
    live = [r for r in rows if r["end"] is None]
    if not live:
        break
    (out / "round.ids").write_text("".join(" ".join(map(str, r["ids"])) + "\n" for r in live))
    subprocess.run([a.lbc, "generate", a.run, "--checkpoint", a.checkpoint, "--ids", str(out / "round.ids"),
                    "--out", str(out / "round.gen"), "--tokens", str(a.round_tokens), "--temperature", "0.7",
                    "--top-k", "40", "--seed", str(a.seed + rnd), "--stop", stop], check=True)
    gens = [[int(x) for x in line.split()] for line in (out / "round.gen").read_text().splitlines()]
    for r, g in zip(live, gens):
        r["ids"] += g
        r["tokens"] += len(g)
        last = g[-1] if g else None
        if last == T["</call>"]:
            body = tok.decode(g, skip_special_tokens=False).rsplit("<call>", 1)[-1].removesuffix("</call>")
            if rnd == a.max_calls:
                r["end"] = "call budget"
                continue
            res, rec = execute(r["sc"], body)
            r["calls"].append(rec)
            r["ids"] += [T["</assistant>"], T["<result>"]] + enc(json.dumps(res, ensure_ascii=False)) + \
                [T["</result>"], T["<assistant>"]] + r["pre"]
        else:
            turn = tok.decode(g, skip_special_tokens=False)
            r["final"] = re.sub(  # the last turn's answer, without its reasoning
r"<think>.*?</think>", "", turn, flags=re.S).replace("</assistant>", "").strip()
            r["end"] = {T["</assistant>"]: "answered", T["<user>"]: "opened a user turn"}.get(last, "token budget")
for r in rows:
    if r["end"] is None:
        r["end"] = "call budget"
    r["solved"] = bool(r["sc"]["check"](r)) and r["end"] == "answered"

# -- report ---------------------------------------------------------------------------------------------------------
print(f"\n{'scenario':11s} {'mode':7s} {'solved':>7s} {'calls':>6s} {'valid':>6s}  ends")
for mode in ("free", "direct"):
    tot = []
    for sc in SCENARIOS:
        rs = [r for r in rows if r["sc"] is sc and r["mode"] == mode]
        tot += rs
        nc = sum(len(r["calls"]) for r in rs)
        ends = {}
        for r in rs:
            ends[r["end"]] = ends.get(r["end"], 0) + 1
        print(f"{sc['name']:11s} {mode:7s} {sum(r['solved'] for r in rs):3d}/{len(rs):<3d} {nc / len(rs):6.1f} "
              f"{sum(c['valid'] for r in rs for c in r['calls']) / max(1, nc):6.0%}  "
              + ", ".join(f"{k} {v}" for k, v in ends.items()))
    nc = sum(len(r["calls"]) for r in tot)
    print(f"{'all':11s} {mode:7s} {sum(r['solved'] for r in tot):3d}/{len(tot):<3d} {nc / len(tot):6.1f} "
          f"{sum(c['valid'] for r in tot for c in r['calls']) / max(1, nc):6.0%}\n")

docs = []
for sc in SCENARIOS:
    for k in range(a.samples):
        outs = []
        for r in rows:
            if r["sc"] is sc and r["sample"] == k:
                gen = tok.decode(r["ids"][r["n_prompt"] + len(r["pre"]):], skip_special_tokens=False)
                outs.append({"mode": r["mode"], "generated": gen, "tokens": r["tokens"], "executed": True,
                             "ok": r["solved"], "calls": len(r["calls"]), "valid": sum(c["valid"] for c in r["calls"]),
                             "checks": [["solved", r["solved"]], ["answered", r["end"] == "answered"],
                                        ["all calls valid", bool(r["calls"]) and all(c["valid"] for c in r["calls"])]],
                             "end": r["end"]})
                prompt, n_prompt = r["prompt"], r["n_prompt"]
        docs.append({"source": sc["name"], "doc": k, "prompt_tokens": n_prompt, "prompt": prompt,
                     "reference": "Expected: " + sc["expect"], "outputs": outs})
(out / "docs.json").write_text(json.dumps(docs))
print(f"{out / 'docs.json'}: {len(docs)} documents")
