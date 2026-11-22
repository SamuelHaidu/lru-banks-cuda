// Completion: the model continues a plain text (pretraining runs; any run). Starters are openings of validation texts.
import { useEffect, useRef, useState } from "react";
import { api, streamGenerate } from "../api";
import type { Run } from "../types";

interface Props { visible: boolean; current: Run | null; ckpt: string; }

export function CompleteView({ visible, current, ckpt }: Props) {
  const [prompt, setPrompt] = useState("");
  const [base, setBase] = useState("");
  const [gen, setGen] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [s, setS] = useState({ temp: 0.8, topk: 50, rep: 1.1, max: 256 });
  const [starters, setStarters] = useState<Record<string, string[]>>({});
  const abort = useRef<AbortController | null>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {  // a new run: new starters, no output
    setBase(""); setGen(""); setStarters({});
    if (!current) return;
    let live = true;
    api.starters(current.dataset).then((st) => { if (live && !st.error) setStarters(st); });
    return () => { live = false; };
  }, [current]);
  useEffect(() => { if (visible) box.current?.focus(); }, [visible]);

  async function complete(text: string) {
    if (busy || !current || !text.trim()) return;
    setBase(text); setGen(""); setBusy(true); setStatus("Generating");
    abort.current = new AbortController();
    let out = "";
    try {
      const info = await streamGenerate({
        run: current.run, checkpoint: ckpt, document: "<bos>" + text, mode: "completion", temperature: s.temp,
        top_k: s.topk || null, max_new_tokens: s.max, repetition_penalty: s.rep || 1,
      }, (t) => { out += t; setGen(out); }, abort.current.signal);
      setStatus(info ? `${info.tokens} tokens at ${info.tps} tokens/s` : "");
    } catch (e) {
      const err = e as Error;
      setStatus(err.name === "AbortError" ? "Stopped" : "Generation failed: " + err.message);
    }
    setBusy(false);
  }

  const ended = gen.endsWith("<eos>"), clean = gen.replace(/<eos>$/, "");
  const num = (k: keyof typeof s) => (e: React.ChangeEvent<HTMLInputElement>) => setS({ ...s, [k]: +e.target.value });
  return (
    <section className="view" hidden={!visible}>
      <div className="scroll"><div className="col cmp">
        <textarea ref={box} rows={5} value={prompt} onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Write the beginning of a text. The model continues it (Ctrl+Enter)."
                  onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); complete(prompt); } }} />
        <div className="cbar">
          {busy ? <button className="stop" onClick={() => abort.current?.abort()}>Stop</button>
                : <button className="send" onClick={() => complete(prompt)}>Generate</button>}
          <button className="ghost" disabled={busy || !base} title="Generate more after the text so far"
                  onClick={() => complete(base + clean)}>Continue</button>
          <button className="ghost" disabled={busy || !base} title="Sample a new continuation of the same prompt"
                  onClick={() => complete(base)}>Regenerate</button>
          <button className="ghost" disabled={busy || !base} title="Put the prompt and the continuation in the box to edit"
                  onClick={() => { setPrompt(base + clean); box.current?.focus(); }}>Edit as prompt</button>
          <span className="cset">
            <label>Temperature <input type="number" step="0.1" min="0" value={s.temp} onChange={num("temp")} /></label>
            <label>Top-k <input type="number" min="0" value={s.topk} onChange={num("topk")} /></label>
            <label>Repetition <input type="number" step="0.05" min="1" value={s.rep} onChange={num("rep")} /></label>
            <label>Max tokens <input type="number" min="1" value={s.max} onChange={num("max")} /></label>
          </span>
        </div>
        <div className="cout">
          {(base || gen) && <><span className="p">{base}</span><span className={busy ? "cursor" : ""}>{clean}</span></>}
          {ended && <p className="note">End of the text.</p>}
        </div>
        <div className="status" style={{ margin: "6px 0 0", padding: 0 }}>{status}</div>
        {!base && !gen && (
          <div className="cstart"><p className="muted">Or start from the beginning of a validation text</p>
            {Object.entries(starters).map(([cat, list]) => (
              <div key={cat}><h4>{cat}</h4><div className="chips">
                {list.map((t, i) => <button key={i} className="chip" onClick={() => { setPrompt(t); complete(t); }}>{t}</button>)}
              </div></div>
            ))}
          </div>
        )}
      </div></div>
    </section>
  );
}
