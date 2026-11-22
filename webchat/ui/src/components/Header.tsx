// The top bar: the run picker (searchable, keyboard), the checkpoint, the tabs, New chat and Setup.
import { useEffect, useRef, useState } from "react";
import type { Run } from "../types";

export type Tab = "chat" | "complete" | "inspect";

interface Props {
  runs: Run[];
  current: Run | null;
  ckpt: string;
  tab: Tab;
  setupOpen: boolean;
  onSelectRun: (name: string) => void;
  onCkpt: (ckpt: string) => void;
  onTab: (tab: Tab) => void;
  onNewChat: () => void;
  onToggleSetup: () => void;
}

const ckptLabel = (c: string) => (c === "latest" || c === "best" ? c : "epoch " + c);

export function Header(p: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const pickerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const q = query.trim().toLowerCase();
  const list = p.runs.filter((r) => !q || `${r.run} ${r.arch} ${r.dataset}`.toLowerCase().includes(q));

  useEffect(() => {  // a click outside closes the picker
    const close = (e: MouseEvent) => { if (!pickerRef.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, []);
  useEffect(() => { listRef.current?.querySelector(".run.active")?.scrollIntoView({ block: "nearest" }); }, [active]);

  const pick = (name: string) => { setOpen(false); p.onSelectRun(name); };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(0, Math.min(list.length - 1, i + (e.key === "ArrowDown" ? 1 : -1))));
    }
    if (e.key === "Enter" && list[active]) pick(list[active].run);
    if (e.key === "Escape") setOpen(false);
  };
  const completion = p.current?.mode === "completion";

  return (
    <header className="bar">
      <div className="brand">rnn-research</div>
      <div className="picker" ref={pickerRef}>
        <button className="model-btn" aria-haspopup="listbox" aria-expanded={open}
                onClick={() => { setOpen(!open); setQuery(""); setActive(0); }}>
          <b>{p.current ? p.current.run : "Choose a model"}</b>
          <span className="muted">{p.current?.arch ?? ""}</span><span className="caret">&#9662;</span>
        </button>
        {open && (
          <div className="pop">
            <input autoFocus placeholder="Search runs, archs, datasets" autoComplete="off" value={query}
                   onChange={(e) => { setQuery(e.target.value); setActive(0); }} onKeyDown={onKey} />
            <div className="run-list" role="listbox" ref={listRef}>
              {list.length ? list.map((r, i) => (
                <button key={r.run} role="option"
                        className={`run${i === active ? " active" : ""}${p.current?.run === r.run ? " current" : ""}`}
                        onClick={() => pick(r.run)}>
                  <b>{r.run}</b>
                  <span>{r.arch}, {r.dataset}, {r.checkpoints.length} checkpoint{r.checkpoints.length > 1 ? "s" : ""}</span>
                </button>
              )) : <p className="note" style={{ padding: "6px 10px" }}>No run matches "{q}".</p>}
            </div>
          </div>
        )}
      </div>
      <select className="ckpt" title="Checkpoint" value={p.ckpt} onChange={(e) => p.onCkpt(e.target.value)}>
        {p.current?.checkpoints.map((c) => <option key={c} value={c}>{ckptLabel(c)}</option>)}
      </select>
      <nav className="seg">
        {!completion && <button className={p.tab === "chat" ? "on" : ""} onClick={() => p.onTab("chat")}>Chat</button>}
        <button className={p.tab === "complete" ? "on" : ""} onClick={() => p.onTab("complete")}>Complete</button>
        {p.current?.inspect && (
          <button className={p.tab === "inspect" ? "on" : ""} onClick={() => p.onTab("inspect")}>Inspect</button>
        )}
      </nav>
      {p.tab === "chat" && <button className="ghost" onClick={p.onNewChat}>New chat</button>}
      {p.tab === "chat" && (
        <button className={`ghost${p.setupOpen ? " on" : ""}`} aria-pressed={p.setupOpen} onClick={p.onToggleSetup}>
          Setup
        </button>
      )}
    </header>
  );
}
