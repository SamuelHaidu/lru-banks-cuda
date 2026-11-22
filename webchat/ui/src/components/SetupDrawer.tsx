// The chat's Setup drawer: system prompt, tools, example requests, generation settings, the workspace (files and
// saved memories) and the raw-document toggle.
import { useState } from "react";
import { api } from "../api";
import { applyToolSet, type Setup } from "../setup";
import type { GenSettings, Presets, WorkspaceState } from "../types";
import { IMPLEMENTED } from "../util";

interface Props {
  presets: Presets | null;
  setup: Setup;
  onSetup: (s: Setup) => void;
  examples: Record<string, string[]>;
  onExample: (text: string) => void;
  workspace: WorkspaceState | null;
  onWorkspaceChanged: () => void;
  rawMode: boolean;
  onRawMode: (raw: boolean) => void;
}

const desc = (line: string) => line.split("): ")[1] || "";

export function SetupDrawer({ presets, setup, onSetup, examples, onExample, workspace, onWorkspaceChanged,
                              rawMode, onRawMode }: Props) {
  const [preset, setPreset] = useState("");
  const [viewed, setViewed] = useState<string | null>(null);
  const atumChecked = [...setup.checked].some((k) => k.startsWith("atum:"));
  const [atumOpen, setAtumOpen] = useState(false);
  const toggle = (key: string) => {
    const checked = new Set(setup.checked);
    if (checked.has(key)) checked.delete(key); else checked.add(key);
    onSetup({ ...setup, checked });
  };
  const gen = (k: keyof GenSettings, v: string) => onSetup({ ...setup, gen: { ...setup.gen, [k]: +v } });
  const tool = (key: string, name: string, line: string, note: string) => (
    <label className="tool" key={key}>
      <input type="checkbox" checked={setup.checked.has(key)} onChange={() => toggle(key)} />
      <span><code>{name}</code><span className="desc">{desc(line)}{note}</span></span>
    </label>
  );
  const atumOk = new Set(presets?.atum_implemented || []);
  const exList = examples[setup.exCat] || [];

  return (
    <aside className="drawer">
      <details open><summary>System prompt</summary><div className="inner">
        <label className="field"><span>Preset</span>
          <select value={preset} onChange={(e) => { setPreset(e.target.value); onSetup({ ...setup, sys: e.target.value }); }}>
            <option value="">No system prompt</option>
            {Object.entries(presets?.system || {}).map(([kind, list]) => (
              <optgroup key={kind} label={kind}>
                {list.map((s) => <option key={s} value={s}>{s.slice(0, 64)}{s.length > 64 ? "..." : ""}</option>)}
              </optgroup>
            ))}
          </select></label>
        <label className="field">
          <textarea rows={3} placeholder="No system prompt" value={setup.sys}
                    onChange={(e) => onSetup({ ...setup, sys: e.target.value })} />
        </label>
      </div></details>

      <details open><summary>Tools</summary><div className="inner">
        <label className="field"><span>Set</span>
          <select value={setup.toolSet} onChange={(e) => {
            if (!presets) return;
            const set = e.target.value;
            onSetup({ ...setup, toolSet: set, checked: applyToolSet(setup, presets, set) });
            if ((presets.tool_sets[set] || []).some((n) => n.startsWith("atum:"))) setAtumOpen(true);
          }}>
            {Object.keys(presets?.tool_sets || {}).map((k) => <option key={k}>{k}</option>)}
          </select></label>
        <div>{Object.entries(presets?.tool_lines || {}).map(([name, line]) =>
          tool(name, name, line, IMPLEMENTED.has(name) ? "" : " (not available here)"))}</div>
        <details style={{ border: 0, padding: "6px 0 0" }} open={atumOpen || atumChecked}
                 onToggle={(e) => setAtumOpen((e.target as HTMLDetailsElement).open)}>
          <summary className="muted" style={{ fontWeight: 400, fontSize: 13.5 }}>Atum09 tools (the model thinks with these)</summary>
          <div style={{ paddingTop: 4 }}>{Object.entries(presets?.atum_tool_lines || {}).map(([name, line]) =>
            tool("atum:" + name, name, line, atumOk.has(name) ? "" : " (not available here)"))}</div>
        </details>
        <details style={{ border: 0, padding: "6px 0 0" }}>
          <summary className="muted" style={{ fontWeight: 400, fontSize: 13.5 }}>Unneeded tools to list</summary>
          <div style={{ paddingTop: 4 }}>{(presets?.decoys || []).map((line) =>
            tool("decoy:" + line, line.split("(")[0], line, ""))}</div>
        </details>
      </div></details>

      <details><summary>Example requests</summary><div className="inner">
        <label className="field">
          <select value={setup.exCat} onChange={(e) => onSetup({ ...setup, exCat: e.target.value })}>
            {Object.keys(examples).map((k) => <option key={k}>{k}</option>)}
          </select></label>
        <div>{exList.map((t, i) => <button key={i} className="ex" onClick={() => onExample(t)}>{t}</button>)}</div>
      </div></details>

      <details><summary>Generation</summary><div className="inner grid2">
        <label className="field"><span>Temperature</span>
          <input type="number" step="0.1" min="0" value={setup.gen.temp} onChange={(e) => gen("temp", e.target.value)} /></label>
        <label className="field"><span>Top-k</span>
          <input type="number" min="0" value={setup.gen.topk} onChange={(e) => gen("topk", e.target.value)} /></label>
        <label className="field"><span>Max new tokens</span>
          <input type="number" min="1" value={setup.gen.maxtok} onChange={(e) => gen("maxtok", e.target.value)} /></label>
        <label className="field"><span>Repetition penalty</span>
          <input type="number" step="0.05" min="1" value={setup.gen.rep} onChange={(e) => gen("rep", e.target.value)} /></label>
        <label className="field"><span>Tool rounds per turn</span>
          <input type="number" min="0" value={setup.gen.rounds} onChange={(e) => gen("rounds", e.target.value)} /></label>
      </div></details>

      <details><summary>Workspace</summary><div className="inner">
        <p className="note" style={{ margin: "0 0 6px" }}>Files the simulated shell tools see.</p>
        <div className="files">
          {workspace && Object.keys(workspace.files).length
            ? Object.keys(workspace.files).map((p) => <div key={p} onClick={() => setViewed(p)}>~/{p}</div>)
            : <p className="note">No files.</p>}
        </div>
        {viewed && workspace?.files[viewed] !== undefined && (
          <pre className="view-box">{`~/${viewed}\n\n${workspace.files[viewed]}`}</pre>
        )}
        <p style={{ margin: "10px 0 4px" }}>
          <button className="small-btn" onClick={async () => { await api.resetWorkspace(); setViewed(null); onWorkspaceChanged(); }}>
            New random workspace</button></p>
        <p className="note" style={{ margin: "14px 0 4px" }}>Saved memories (memory tools)</p>
        <pre className="view-box">{workspace && Object.keys(workspace.memory).length
          ? Object.entries(workspace.memory).map(([k, v]) => `${k}: ${v}`).join("\n") : "Nothing saved."}</pre>
        <p style={{ margin: "8px 0 0" }}>
          <button className="small-btn" onClick={async () => { await api.clearMemory(); onWorkspaceChanged(); }}>
            Clear memories</button></p>
      </div></details>

      <details><summary>Raw document</summary><div className="inner">
        <p className="note" style={{ margin: "0 0 8px" }}>The exact text the model reads, special tags included.</p>
        <button className="small-btn" onClick={() => onRawMode(!rawMode)}>
          {rawMode ? "Show conversation" : "Show raw document"}</button>
      </div></details>
    </aside>
  );
}
