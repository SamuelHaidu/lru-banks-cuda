"""The tool-call eval's documents as a chat viewer (one self-contained HTML page, no network): per document the
conversation up to the evaluated turn (system and tools folded), then the reference assistant turn and the model's two
continuations side by side, with reasoning, calls and results formatted and each call's scores.

    python3 scripts/toolcall_chat.py DOCS_JSON [--out toolcall_chat.html] [--title "..."]

DOCS_JSON: a list of {source, doc, prompt_tokens, prompt, reference, outputs: [{mode, generated, call, parses,
declared, args, same, tokens}]} (the eval's samples with the prompts decoded).
"""

import argparse
import html
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("docs", type=Path)
ap.add_argument("--out", type=Path, default=Path("toolcall_chat.html"))
ap.add_argument("--title", default="Tool-call evaluation")
a = ap.parse_args()
docs = json.loads(a.docs.read_text())

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --ink:#1f2328; --mute:#656d76; --line:#e3e6ea; --bg:#ffffff; --side:#f7f8fa; --user:#eef2f8; --code:#f6f8fa;
          --ok:#1a7f37; --bad:#b42318; --accent:#2f5fa7; }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body { font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg);
         display: grid; grid-template-columns: 260px 1fr; }
  aside { background: var(--side); border-right: 1px solid var(--line); overflow-y: auto; height: 100vh; position: sticky; top: 0; }
  aside header { padding: 16px 16px 8px; }
  aside h1 { font-size: 15px; margin: 0 0 10px; font-weight: 600; }
  select { width: 100%; font: inherit; padding: 5px 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--bg); color: var(--ink); }
  .legend { color: var(--mute); font-size: 12px; margin-top: 8px; }
  nav button { display: flex; width: 100%; align-items: center; justify-content: space-between; gap: 8px; border: 0;
               background: none; padding: 7px 16px; font: inherit; color: var(--ink); cursor: pointer; text-align: left; }
  nav button:hover { background: #eef0f3; }
  nav button[aria-current="true"] { background: #e6ebf3; font-weight: 600; }
  .marks { display: inline-flex; gap: 4px; }
  .mark { width: 9px; height: 9px; border-radius: 50%; background: #d0d7de; }
  .mark.ok { background: var(--ok); }
  main { overflow-y: auto; height: 100vh; }
  .thread { max-width: 1500px; margin: 0 auto; padding: 24px 28px 64px; }
  .meta { color: var(--mute); font-size: 13px; margin-bottom: 16px; }
  .msg { margin: 0 0 14px; }
  .role { font-size: 12px; color: var(--mute); margin-bottom: 4px; }
  .user { display: flex; justify-content: flex-end; }
  .user .bubble { background: var(--user); border-radius: 12px; padding: 10px 14px; max-width: 80%; white-space: pre-wrap; }
  details.fold { border: 1px solid var(--line); border-radius: 8px; margin: 0 0 12px; background: var(--bg); }
  details.fold > summary { cursor: pointer; padding: 8px 12px; color: var(--mute); list-style-position: inside; }
  details.fold > summary b { color: var(--ink); font-weight: 600; }
  details.fold > .body { padding: 4px 14px 12px; white-space: pre-wrap; font-size: 13px; max-height: 420px; overflow: auto; }
  .chip { display: inline-block; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--code);
          border: 1px solid var(--line); border-radius: 4px; padding: 0 5px; margin: 2px 3px 0 0; color: var(--ink); }
  .text { white-space: pre-wrap; margin: 6px 0; }
  details.think { margin: 6px 0; border-left: 3px solid var(--line); padding-left: 10px; }
  details.think > summary { cursor: pointer; color: var(--mute); font-size: 13px; }
  details.think .body { white-space: pre-wrap; color: #4b535c; font-size: 13px; }
  .empty { color: var(--mute); font-size: 13px; font-style: italic; margin: 6px 0; }
  .call { margin: 8px 0; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  .call .head { background: var(--code); padding: 5px 10px; font-size: 12px; color: var(--mute); border-bottom: 1px solid var(--line); }
  .call .head b { color: var(--accent); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .call pre, .result pre { margin: 0; padding: 8px 10px; font: 12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
          white-space: pre-wrap; word-break: break-word; }
  .call .arg { padding: 6px 10px 0; font-size: 12px; color: var(--mute); }
  .result { margin: 8px 0; background: var(--code); border-radius: 8px; }
  .result.sim { border: 1px dashed var(--mute); }
  .result .head { padding: 5px 10px 0; font-size: 12px; color: var(--mute); }
  .cut { color: var(--bad); font-size: 12px; margin-top: 4px; }
  h2 { font-size: 14px; font-weight: 600; margin: 26px 0 10px; padding-top: 14px; border-top: 1px solid var(--line); }
  .cols { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; align-items: start; }
  .col { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; min-width: 0; }
  .col h3 { margin: 0 0 6px; font-size: 13px; font-weight: 600; }
  .flags { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; font-size: 12px; }
  .flag { border-radius: 4px; padding: 0 6px; border: 1px solid var(--line); color: var(--mute); }
  .flag.ok { color: var(--ok); border-color: #b7dfc2; background: #f0faf3; }
  .flag.no { color: var(--bad); border-color: #f1c4bf; background: #fdf3f2; }
  @media (max-width: 1150px) { .cols { grid-template-columns: 1fr; } }
  @media (max-width: 760px) { body { grid-template-columns: 1fr; } aside { position: static; height: auto; } main { height: auto; } }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
</style></head><body>
<aside>
  <header>
    <h1>__TITLE__</h1>
    <label for="src" class="legend">Source</label>
    <select id="src"></select>
    <div class="legend">Dots: valid call with reasoning, then with reasoning skipped. Arrow keys move between documents.</div>
  </header>
  <nav id="list"></nav>
</aside>
<main id="main"><div class="thread" id="thread"></div></main>
<script>
const DOCS = __DATA__;
const TAGS = ["system", "tools", "user", "assistant", "think", "call", "result"];
const SPLIT = new RegExp("(<\\/?(?:" + TAGS.join("|") + ")>|<bos>|<eos>)");
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

// the text as a flat list of {tag, text, closed}: tag is the innermost open structural tag ("" outside any), with
// `start` naming the tag a model continuation is inside of before its first token
function parse(text, start) {
  const out = [], stack = start ? [start] : [];
  for (const piece of text.split(SPLIT)) {
    if (!piece) continue;
    const m = piece.match(/^<(\/?)([a-z]+)>$/);
    if (m && (TAGS.includes(m[2]) || m[2] === "bos" || m[2] === "eos")) {
      if (m[2] === "bos" || m[2] === "eos") continue;
      if (m[1]) {  // close: mark the last open item of that tag closed
        for (let i = out.length - 1; i >= 0; i--) if (out[i].tag === m[2] && !out[i].closed) { out[i].closed = true; break; }
        const k = stack.lastIndexOf(m[2]); if (k >= 0) stack.splice(k, 1);
      } else {
        stack.push(m[2]);
        if (m[2] !== "assistant") out.push({tag: m[2], text: "", closed: false});
      }
      continue;
    }
    const tag = stack.length ? stack[stack.length - 1] : "";
    const last = out[out.length - 1];
    if (last && last.tag === tag && !last.closed && tag !== "assistant" && tag !== "") last.text += piece;
    else out.push({tag, text: piece, closed: tag === "assistant" || tag === ""});
  }
  return out;
}

// `key=<JSON>, ...)` as [[key, value]], or null when it does not parse (then the raw text is shown)
function callArgs(rest) {
  const out = []; let i = 0;
  const skip = () => { while (i < rest.length && /\s/.test(rest[i])) i++; };
  while (true) {
    skip();
    if (rest[i] === ")" ) return out;
    const km = rest.slice(i).match(/^([A-Za-z_][\w\-]*)=/);
    if (!km) return null;
    i += km[0].length;
    let j = i, depth = 0, inStr = false;
    for (; j < rest.length; j++) {
      const ch = rest[j];
      if (inStr) { if (ch === "\\") j++; else if (ch === '"') inStr = false; continue; }
      if (ch === '"') inStr = true;
      else if (ch === "[" || ch === "{") depth++;
      else if (ch === "]" || ch === "}") depth--;
      else if ((ch === "," || ch === ")") && depth === 0) break;
    }
    try { out.push([km[1], JSON.parse(rest.slice(i, j))]); } catch (e) { return null; }
    i = j;
    if (rest[i] === ",") i++;
    if (i >= rest.length) return out;
  }
}

function renderPart(p, box, simulated) {
  const t = p.text;
  if (p.tag === "think") {
    if (!t.trim()) { box.append(el("div", "empty", p.closed ? "Reasoning skipped (empty think)" : "Reasoning: empty")); return; }
    const d = el("details", "think"); d.open = true;
    d.append(el("summary", null, "Reasoning (" + t.trim().split(/\s+/).length + " words)" + (p.closed ? "" : ", cut off")));
    d.append(el("div", "body", t.trim()));
    box.append(d);
  } else if (p.tag === "call") {
    const c = el("div", "call"), h = el("div", "head");
    const name = (t.match(/^\s*([\w\-.]+)\(/) || [])[1];
    h.append("Tool call "); if (name) h.append(el("b", null, name));
    c.append(h);
    const args = name ? callArgs(t.trim().slice(t.trim().indexOf("(") + 1)) : null;
    if (args && args.length) {
      for (const [k, v] of args) {
        c.append(el("div", "arg", k));
        c.append(el("pre", null, typeof v === "string" ? v : JSON.stringify(v, null, 2)));
      }
    } else {
      c.append(el("pre", null, t.trim()));
    }
    if (!p.closed) c.append(el("div", "cut", "  not closed (the token budget ended)"));
    box.append(c);
  } else if (p.tag === "result") {
    const r = el("div", "result" + (simulated ? " sim" : ""));
    r.append(el("div", "head", simulated ? "Tool result, written by the model (not executed)" : "Tool result"), el("pre", null, t.trim()));
    box.append(r);
  } else if (t.trim()) {
    box.append(el("div", "text", t.trim()));
  }
}

function renderPrompt(prompt, thread) {
  for (const p of parse(prompt)) {
    if (p.tag === "system") {
      const d = el("details", "fold"), s = el("summary");
      s.append(el("b", null, "System prompt"), " " + p.text.trim().split("\n")[0].slice(0, 110));
      d.append(s, el("div", "body", p.text.trim())); thread.append(d);
    } else if (p.tag === "tools") {
      const names = [...p.text.matchAll(/^([\w\-.]+)\(/gm)].map(m => m[1]);
      const d = el("details", "fold"), s = el("summary");
      s.append(el("b", null, "Tools (" + names.length + ") "));
      names.forEach(n => s.append(el("span", "chip", n)));
      d.append(s, el("div", "body", p.text.trim())); thread.append(d);
    } else if (p.tag === "user") {
      const m = el("div", "msg user"), b = el("div", "bubble", p.text.trim()); m.append(b); thread.append(m);
    } else if (p.tag === "result") {
      renderPart(p, thread);
    } else if (p.text.trim()) {
      const m = el("div", "msg"); m.append(el("div", "role", "Assistant")); renderPart(p, m); thread.append(m);
    }
  }
}

const FLAGS = [["call", "call"], ["parses", "parses"], ["declared", "declared tool"], ["args", "valid args"], ["same", "same tool as reference"]];
function column(title, text, o, start) {
  const c = el("div", "col"); c.append(el("h3", null, title));
  if (o) {
    const f = el("div", "flags");
    const checks = o.checks || FLAGS.map(([k, n]) => [n, o[k]]);
    for (const [n, v] of checks) f.append(el("span", "flag " + (v ? "ok" : "no"), (v ? "✓ " : "✗ ") + n));
    if (o.end) f.append(el("span", "flag", "ended: " + o.end));
    f.append(el("span", "flag", o.tokens + " tokens"));
    if (o.calls != null) f.append(el("span", "flag", `${o.calls} calls, ${o.valid} valid`));
    c.append(f);
  }
  for (const p of parse(text, start)) renderPart(p, c, !!o && !o.executed);
  return c;
}

function show(i) {
  const d = DOCS[i], th = $("thread");
  th.replaceChildren();
  th.append(el("div", "meta", `${d.source}, validation document ${d.doc}: the prompt is ${d.prompt_tokens} tokens; the model continues the last assistant turn`));
  renderPrompt(d.prompt, th);
  th.append(el("h2", null, "The assistant turn"));
  const cols = el("div", "cols");
  const free = d.outputs.find(o => o.mode === "free"), direct = d.outputs.find(o => o.mode === "direct");
  cols.append(column("Reference (the trace)", d.reference, null, "assistant"));
  if (free) cols.append(column("Model, with reasoning", free.generated, free, "assistant"));
  if (direct) cols.append(column("Model, reasoning skipped", "<think></think>" + direct.generated, direct, "assistant"));
  th.append(cols);
  document.querySelectorAll("nav button").forEach(b => b.setAttribute("aria-current", String(+b.dataset.i === i)));
  $("main").scrollTop = 0;
  current = i;
}

let current = 0;
const sources = [...new Set(DOCS.map(d => d.source))];
$("src").innerHTML = sources.map(s => `<option>${s}</option>`).join("");
function list() {
  const src = $("src").value, nav = $("list");
  nav.replaceChildren();
  DOCS.forEach((d, i) => {
    if (d.source !== src) return;
    const b = el("button"); b.dataset.i = i;
    b.append(el("span", null, `${d.source} #${nav.children.length + 1}`));
    const m = el("span", "marks");
    for (const mode of ["free", "direct"]) {
      const o = d.outputs.find(x => x.mode === mode);
      const good = o && (o.ok ?? o.args);
      const dot = el("span", "mark" + (good ? " ok" : ""));
      dot.title = (mode === "free" ? "with reasoning: " : "reasoning skipped: ") + (good ? "passed" : "failed");
      m.append(dot);
    }
    b.append(m); b.onclick = () => show(i); nav.append(b);
  });
  show(+nav.firstChild.dataset.i);
}
if (sources.includes("oswe")) $("src").value = "oswe";
$("src").onchange = list;
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || !["ArrowDown", "ArrowUp"].includes(e.key)) return;
  const ids = [...document.querySelectorAll("nav button")].map(b => +b.dataset.i);
  const k = ids.indexOf(current) + (e.key === "ArrowDown" ? 1 : -1);
  if (k >= 0 && k < ids.length) { e.preventDefault(); show(ids[k]); }
});
list();
</script></body></html>
"""
page = PAGE.replace("__TITLE__", html.escape(a.title)).replace("__DATA__", json.dumps(docs).replace("</", "<\\/"))
a.out.write_text(page)
print(f"{a.out}: {len(docs)} documents")
