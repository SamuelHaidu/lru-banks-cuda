// The chat: the conversation is one document in the training format (<bos>, system prompt, tools, then <user> /
// <assistant> turns with <think>, <call> and <result>). The model writes the assistant turns; its tool calls run on the
// server and their results are appended, for up to `rounds` rounds per turn.
import { type ReactNode, useEffect, useRef, useState } from "react";
import { api, streamGenerate } from "../api";
import { buildHeader, type Setup } from "../setup";
import type { Presets, Run } from "../types";

interface Props {
  visible: boolean;
  current: Run | null;
  ckpt: string;
  presets: Presets | null;
  setup: Setup;
  examples: Record<string, string[]>;
  draft: string;
  onDraft: (t: string) => void;
  rawMode: boolean;
  docRef: React.MutableRefObject<string>;
  onBusy: (busy: boolean) => void;
  onTemp: (temp: number) => void;
  onWorkspaceChanged: () => void;
}

const TAGS = /(<\/?(?:user|assistant|think|call|result)>|<eos>)/;

export function ChatView(p: Props) {
  const [header, setHeader] = useState<string | null>(null);
  const [, setTick] = useState(0);
  const docRef = useRef("");  // the document; changes trigger a render through `tick`
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [editing, setEditing] = useState<number | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const abort = useRef<AbortController | null>(null);
  const scroll = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const setDoc = (d: string) => { docRef.current = d; p.docRef.current = d; setTick((t) => t + 1); };
  const doc = docRef.current;

  useEffect(() => { const s = scroll.current; if (s) s.scrollTop = s.scrollHeight; });
  useEffect(() => {  // the composer grows with its text
    const t = input.current;
    if (t) { t.style.height = "auto"; t.style.height = Math.min(220, t.scrollHeight) + "px"; }
  }, [p.draft]);

  async function assistantTurns() {
    const g = p.setup.gen, rounds = Math.max(0, g.rounds);
    let thinkOnly = 0;  // the training data puts thinking in its own turn: <assistant><think>..</think></assistant><assistant>answer
    for (let round = 0; round <= rounds; round++) {
      setDoc(docRef.current + "<assistant>");
      let turn = "";
      setStatus("Generating");
      abort.current = new AbortController();
      const info = await streamGenerate({
        run: p.current!.run, checkpoint: p.ckpt, document: docRef.current, temperature: g.temp,
        top_k: g.topk || null, max_new_tokens: g.maxtok, repetition_penalty: g.rep || 1,
      }, (t) => { turn += t; setDoc(docRef.current + t); }, abort.current.signal);
      if (!(turn.endsWith("</assistant>") || turn.endsWith("<eos>"))) setDoc(docRef.current + "</assistant>");
      setStatus(info ? `${info.tokens} tokens at ${info.tps} tokens/s` : "");
      if (turn.endsWith("<eos>")) break;
      const calls = [...turn.matchAll(/<call>([\s\S]*?)<\/call>/g)].map((m) => m[1]);
      const visible = turn.replace(/<think>[\s\S]*?<\/think>/g, "").replace("</assistant>", "").trim();
      if (!calls.length && !visible && turn.includes("</think>") && thinkOnly < 2) { thinkOnly++; round--; continue; }
      if (!calls.length || round === rounds) break;
      for (const c of calls) {
        setStatus("Running " + c.split("(")[0]);
        const { result } = await api.tool(c);
        setDoc(docRef.current + `<result>${result}</result>`);
      }
      p.onWorkspaceChanged();
    }
  }

  async function reply() {
    setBusy(true); p.onBusy(true);
    try { await assistantTurns(); } catch (e) {
      const err = e as Error;
      if (err.name !== "AbortError") setStatus("Generation failed: " + err.message);
      else { setDoc(docRef.current + "</assistant>"); setStatus("Stopped"); }
    }
    setBusy(false); p.onBusy(false);
    p.onWorkspaceChanged();
  }

  async function send() {
    const text = p.draft.trim();
    if (!text || busy || !p.current) return;
    let d = docRef.current;
    if (header === null) { const h = buildHeader(p.setup, p.presets); setHeader(h); d = h; }
    setDoc(d + `<user>${text}</user>`);
    p.onDraft("");
    await reply();
  }

  const userOffset = (n: number) => {  // where the n-th user message starts in the document
    let i = (header ?? "").length - 1;
    for (let k = 0; k <= n; k++) { i = docRef.current.indexOf("<user>", i + 1); if (i < 0) return -1; }
    return i;
  };
  function startEdit(n: number) {
    if (busy) return;
    const at = userOffset(n);
    if (at < 0) return;
    setEditing(n);
    setEditDraft(docRef.current.slice(at + "<user>".length, docRef.current.indexOf("</user>", at)));
  }
  async function saveEdit() {
    const text = editDraft.trim(), at = userOffset(editing!);
    if (!text || busy || at < 0) return;
    setEditing(null);
    setDoc(docRef.current.slice(0, at) + `<user>${text}</user>`);  // the conversation after it goes (tool side effects stay)
    await reply();
  }
  async function regenerate() {
    const end = docRef.current.lastIndexOf("</user>");
    if (busy || !p.current || end < 0) return;
    setDoc(docRef.current.slice(0, end + "</user>".length));  // everything the model wrote after it goes
    await reply();
  }

  return (
    <section className="view" hidden={!p.visible}>
      <div className="scroll" ref={scroll}>
        <div className="col">
          {p.rawMode ? <div className="raw">{doc || buildHeader(p.setup, p.presets)}</div>
            : header === null ? <Empty {...p} />
            : <Transcript header={header} doc={doc} streaming={busy} editing={editing} editDraft={editDraft}
                          onEditDraft={setEditDraft} onStartEdit={startEdit} onSaveEdit={saveEdit}
                          onCancelEdit={() => setEditing(null)} temp={p.setup.gen.temp} onTemp={p.onTemp}
                          onRegenerate={regenerate} />}
        </div>
      </div>
      <div className="composer-wrap">
        <div className="composer">
          <textarea ref={input} rows={1} placeholder="Message the model" value={p.draft}
                    onChange={(e) => p.onDraft(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} />
          {busy ? <button className="stop" onClick={() => abort.current?.abort()}>Stop</button>
                : <button className="send" onClick={send}>Send</button>}
        </div>
        <div className="status">{status}</div>
      </div>
    </section>
  );
}

function Context({ header }: { header: string }) {
  const sys = (header.match(/<system>([\s\S]*?)<\/system>/) || [])[1];
  const tools = (header.match(/<tools>\n([\s\S]*?)<\/tools>/) || [])[1];
  const names = tools ? tools.trim().split("\n").map((l) => l.split("(")[0]) : [];
  return (
    <div className="context">
      {sys ? <details><summary>System prompt</summary><pre>{sys}</pre></details> : "No system prompt"}{" "}
      {names.length ? <details><summary>{names.length} tool{names.length > 1 ? "s" : ""}: {names.join(", ")}</summary>
        <pre>{tools!.trim()}</pre></details> : <span>No tools</span>}
    </div>
  );
}

function Empty(p: Props) {
  const ex = Object.values(p.examples)[0] || [];
  return (
    <div className="empty">
      <h1>{p.current ? p.current.run : "No model"}</h1>
      <p>{p.current ? `${p.current.arch} trained on ${p.current.dataset}. The system prompt and tools from Setup are fixed when you send the first message.` : ""}</p>
      <Context header={buildHeader(p.setup, p.presets)} />
      <div className="chips">
        {ex.slice(0, 4).map((t, i) => <button key={i} className="chip" onClick={() => p.onDraft(t)}>{t}</button>)}
      </div>
    </div>
  );
}

interface TranscriptProps {
  header: string;
  doc: string;
  streaming: boolean;
  editing: number | null;
  editDraft: string;
  onEditDraft: (t: string) => void;
  onStartEdit: (n: number) => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
  temp: number;
  onTemp: (t: number) => void;
  onRegenerate: () => void;
}

/** The document after the header as messages: user bubbles, assistant text, thoughts, tool calls and results. */
function Transcript(t: TranscriptProps) {
  const out: ReactNode[] = [<Context key="ctx" header={t.header} />];
  let where: string | null = null, buf = "", inner: ReactNode[] = [], nUser = 0, k = 0;
  const text = () => { if (buf.trim()) inner.push(<div key={k++}>{buf.replace(/^\n+/, "")}</div>); buf = ""; };
  const flush = () => { if (inner.length) out.push(<div key={k++} className="msg">{inner}</div>); inner = []; };
  for (const part of t.doc.slice(t.header.length).split(TAGS)) {
    if (!part) continue;
    if (part === "<user>") { where = "user"; buf = ""; }
    else if (part === "</user>") {
      const n = nUser++;
      out.push(t.editing === n ? (
        <div key={k++} className="msg user"><div className="edit-box">
          <textarea autoFocus rows={3} aria-label="Edit message" value={t.editDraft}
                    onChange={(e) => t.onEditDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); t.onSaveEdit(); }
                      if (e.key === "Escape") t.onCancelEdit();
                    }} />
          <div className="edit-actions"><button className="stop" onClick={t.onCancelEdit}>Cancel</button>
            <button className="send" onClick={t.onSaveEdit}>Send</button></div>
        </div></div>
      ) : (
        <div key={k++} className="msg user">
          {!t.streaming && <button className="edit ghost" onClick={() => t.onStartEdit(n)}
            title="Edit this message and answer it again (the conversation after it is dropped)">Edit</button>}
          <div className="bubble">{buf}</div>
        </div>
      ));
      buf = ""; where = null;
    }
    else if (part === "<assistant>") { where = "assistant"; inner = []; buf = ""; }
    else if (part === "</assistant>") { text(); flush(); where = null; }
    else if (part === "<think>") { text(); where = "think"; }
    else if (part === "</think>") {
      inner.push(<details key={k++} className="think"><summary>Thought</summary><div>{buf}</div></details>);
      buf = ""; where = "assistant";
    }
    else if (part === "<call>") { text(); where = "call"; }
    else if (part === "</call>") { inner.push(<div key={k++} className="call"><i>Tool call</i>{buf}</div>); buf = ""; where = "assistant"; }
    else if (part === "<result>") { flush(); where = "result"; buf = ""; }
    else if (part === "</result>") {
      out.push(<div key={k++} className="msg"><div className="result"><i>Result</i>{buf}</div></div>); buf = ""; where = null;
    }
    else if (part === "<eos>") out.push(<p key={k++} className="note">End of conversation.</p>);
    else buf += part;
  }
  if (where) {  // the turn being written
    if (where === "think") inner.push(<details key={k++} className="think" open><summary>Thinking</summary><div className="cursor">{buf}</div></details>);
    else if (where === "call") inner.push(<div key={k++} className="call cursor"><i>Tool call</i>{buf}</div>);
    else if (where === "assistant") inner.push(<div key={k++} className={t.streaming ? "cursor" : ""}>{buf.replace(/^\n+/, "")}</div>);
    if (where !== "user" && where !== "result") flush();
  }
  // after the last reply: answer the last message again (sampled at the temperature; at 0 it is greedy: the same reply)
  const d = t.doc;
  const canRegen = !t.streaming && t.editing === null && d.lastIndexOf("</user>") >= 0 && d.lastIndexOf("</user>") > d.lastIndexOf("<user>");
  if (canRegen) out.push(
    <div key={k++} className="turn-actions">
      <button className="ghost" title="Discard the reply after your last message and sample a new one" onClick={t.onRegenerate}>Regenerate</button>
      <label>Temperature <input type="number" step="0.1" min="0" value={t.temp} onChange={(e) => t.onTemp(+e.target.value)} /></label>
      {t.temp > 0 ? null : <span>greedy: the same reply; raise it to sample</span>}
    </div>,
  );
  return <>{out}</>;
}
