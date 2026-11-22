// The Inspect tab: run a document through the model and look at its predictions and internals token by token.
// The computations live in logic.ts; this component holds the controls, the tokens and the side panel.
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../../api";
import type { Run } from "../../types";
import { esc, showTok } from "../../util";
import {
  computeView, controlsShown, decodeAttend, decodeBanks, firstScored, headOptions, type Ins, layerOptions, type Mode,
  modesOf, prepare, renderSide, type Sel,
} from "./logic";

interface Props {
  current: Run | null;
  ckpt: string;
  visible: boolean;
  chatDoc: () => string;  // the chat's document, or its header when nothing was sent
}

const MODES: [Mode, string][] = [["loss", "Loss"], ["stored", "Stored"], ["recall", "Recall"], ["attention", "Attention"], ["gate", "Gate"]];
const docLabel = (d: string) => {
  const m = d.match(/<user>([\s\S]*?)<\/user>/);
  return (m ? m[1] : d.replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").slice(0, 80);
};

export function InspectView({ current, ckpt, visible, chatDoc }: Props) {
  const [text, setText] = useState("");
  const [docs, setDocs] = useState<Record<string, string[]> | null>(null);
  const [docError, setDocError] = useState("");
  const [ins, setIns] = useState<Ins | null>(null);
  const [version, setVersion] = useState(0);  // bumped when unrolled reads arrive (cached on `ins`)
  const [message, setMessage] = useState("Inspect a document to see how the model predicts it, what its states hold, and what it reads from its memories.");
  const [sel, setSel] = useState<Sel>({ t: 0, mode: "loss", layer: "all", head: "all", hop: "0", measure: "share", hide: 8 });
  const pending = useRef<string | null>(null);
  const tokensBox = useRef<HTMLDivElement>(null);

  useEffect(() => { setDocs(null); setDocError(""); }, [current]);
  useEffect(() => {  // the validation documents of the run's dataset, on first show
    if (!visible || !current || docs) return;
    let live = true;
    api.documents(current.dataset).then((d) => {
      if (!live) return;
      if (d.error) { setDocError(d.error); setDocs({}); } else setDocs(d);
    });
    return () => { live = false; };
  }, [visible, current, docs]);

  async function run(source?: string) {
    if (!current) return;
    const doc = (source ?? text).trim() || chatDoc();
    setMessage("Inspecting");
    const out = await api.inspect({ run: current.run, checkpoint: ckpt, text: doc });
    if (out.error) { setMessage("Inspection failed: " + out.error); return; }
    const next = prepare(out), m = modesOf(next);
    let mode = sel.mode;
    if ((mode === "stored" && !m.stored) || (mode === "recall" && (!m.recall || !m.recallEnabled))
        || (mode === "attention" && !m.attention) || (mode === "gate" && !m.gate)) mode = "loss";
    setIns(next);
    setMessage("");
    setSel({ ...sel, mode, t: firstScored(next), layer: layerOptions(next)[0][0], head: "all",
             hop: next.kind === "banks" && next.hops < 2 ? "0" : sel.hop });
  }

  const view = useMemo(() => (ins ? computeView(ins, sel) : null), [ins, sel, version]);

  useEffect(() => {  // fetch an unrolled read the view needs, once
    if (!ins || !view?.need) return;
    const { op, t } = view.need, key = `${op}:${t}`;
    if (pending.current === key) return;
    pending.current = key;
    const asked = ins;
    (op === "attend" ? api.attend(ins.attn_id, t) : api.banks(ins.banks_id, t)).then((r) => {
      pending.current = null;
      if (asked !== ins) return;
      if (r.error) { setMessage("Unrolling failed: " + r.error); return; }
      if (op === "attend") ins.att[t] = decodeAttend(r); else ins.bk[t] = decodeBanks(r);
      setVersion((v) => v + 1);
    });
  }, [ins, view]);

  useEffect(() => { tokensBox.current?.querySelector(".tk.sel")?.scrollIntoView({ block: "nearest" }); }, [sel.t, ins]);

  useEffect(() => {  // arrow keys move the selection
    if (!visible || !ins) return;
    const onKey = (e: KeyboardEvent) => {
      if (["TEXTAREA", "INPUT", "SELECT"].includes((e.target as HTMLElement).tagName)) return;
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        setSel((s) => ({ ...s, t: Math.max(0, Math.min(ins.T - 1, s.t + (e.key === "ArrowLeft" ? -1 : 1))) }));
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [visible, ins]);

  const sideClick = (e: React.MouseEvent) => {  // the side panel's buttons (it is HTML from renderSide)
    const t = e.target as HTMLElement, j = t.closest<HTMLElement>("[data-j]");
    if (j) {
      tokensBox.current?.querySelector(`[data-k="${j.dataset.j}"]`)?.scrollIntoView({ block: "center" });
      return;
    }
    const cell = t.closest<HTMLElement>(".acell"), row = t.closest<HTMLElement>("tr.rd");
    if (cell) setSel((s) => ({ ...s, layer: cell.dataset.l ?? s.layer, hop: cell.dataset.o ?? s.hop, head: cell.dataset.h ?? s.head }));
    else if (row) setSel((s) => ({ ...s, layer: row.dataset.l!, hop: row.dataset.o! }));
  };

  const modes = modesOf(ins), shown = controlsShown(ins, sel.mode);
  const hopsOk = !ins || ins.kind !== "banks" || ins.hops > 1;
  return (
    <section className="view" hidden={!visible}>
      <div className="ins">
        <div className="ins-main">
          <div className="ins-source">
            <textarea rows={1} value={text} onChange={(e) => setText(e.target.value)}
                      placeholder="Paste a document to inspect (tags like <user> are kept), or load one" />
            <select value="" onChange={(e) => {
              const v = e.target.value; if (!v || !docs) return;
              const k = v.lastIndexOf("|"), d = docs[v.slice(0, k)][+v.slice(k + 1)];
              setText(d); run(d);
            }}>
              <option value="">{docs ? docError || "Load a validation document" : "Loading documents"}</option>
              {docs && Object.entries(docs).map(([cat, list]) => (
                <optgroup key={cat} label={cat}>
                  {list.map((d, i) => <option key={i} value={`${cat}|${i}`}>{docLabel(d)}</option>)}
                </optgroup>
              ))}
            </select>
            <button className="small-btn" style={{ padding: "7px 10px" }} onClick={() => setText(chatDoc())}>Use this chat</button>
            <button className="primary" onClick={() => run()}>Inspect</button>
          </div>
          <div className="ins-controls">
            <nav className="seg">
              {MODES.filter(([m]) => m === "loss" || modes[m as "stored" | "recall" | "attention" | "gate"]).map(([m, label]) => (
                <button key={m} className={sel.mode === m ? "on" : ""} disabled={m === "recall" && !modes.recallEnabled}
                        onClick={() => setSel({ ...sel, mode: m })}>{label}</button>
              ))}
            </nav>
            {ins && shown.layer && <label>Layer <select value={sel.layer} onChange={(e) => setSel({ ...sel, layer: e.target.value })}>
              {layerOptions(ins).map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>}
            {ins && shown.head && <label>Head <select value={sel.head} onChange={(e) => setSel({ ...sel, head: e.target.value })}>
              {headOptions(ins).map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>}
            {shown.hop && <label>Read <select value={sel.hop} onChange={(e) => setSel({ ...sel, hop: e.target.value })}>
              <option value="0">hop 1</option>{hopsOk && <option value="1">hop 2</option>}{hopsOk && <option value="all">both</option>}
            </select></label>}
            {shown.measure && <label>Show <select value={sel.measure} onChange={(e) => setSel({ ...sel, measure: e.target.value })}>
              <option value="share">share of the read</option><option value="c">weight on the value</option></select></label>}
            {shown.hide && <label>Ignore the last <input type="number" min="0" value={sel.hide}
              onChange={(e) => setSel({ ...sel, hide: +e.target.value })} /> tokens</label>}
            {view && <span className="legend" dangerouslySetInnerHTML={{ __html: view.legend }} />}
          </div>
          <div className="summary" dangerouslySetInnerHTML={{ __html: message ? esc(message) : view?.summary ?? "" }} />
          <div className="tokens" ref={tokensBox}>
            {ins && view && ins.raw.map((t: string, r: number) => {
              const k = ins.rawToKept[r], tag = /^<\/?[a-z]+>$/.test(t) ? " tag" : "";
              if (k >= 0) {
                const s = view.el[k];
                return <span key={r} data-k={k} className={`tk${tag}${s.cls.size ? " " + [...s.cls].join(" ") : ""}`}
                             style={s.bg ? { background: s.bg } : undefined} title={s.title || undefined}
                             onClick={() => setSel({ ...sel, t: k })}>{showTok(t)}</span>;
              }
              const e = ins.rawToEntry[r], s = e >= 0 ? view.elE[e] : null;
              return <span key={r} className="tk mem" style={s?.bg ? { background: s.bg } : undefined}
                           title={s?.title || undefined}>{showTok(t)}</span>;
            })}
          </div>
        </div>
        <aside className="side" onClick={sideClick}
               dangerouslySetInnerHTML={{ __html: ins && view ? renderSide(ins, sel, view) : '<p class="note">Select a token to see its prediction.</p>' }} />
      </div>
    </section>
  );
}
