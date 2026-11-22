// Typed calls to the server. Generation is a POST answered with server-sent events: {"text": delta} as tokens come,
// then {"done": true, ...}; {"error": ...} on failure.
import type { GenerateBody, GenerateInfo, Presets, Run, WorkspaceState } from "./types";

async function get<T>(path: string): Promise<T> {
  return (await fetch(path)).json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return r.json();
}

const q = encodeURIComponent;

export const api = {
  runs: () => get<Run[]>("/api/runs"),
  presets: (dataset: string) => get<Presets>(`/api/presets?dataset=${q(dataset)}`),
  starters: (dataset: string) => get<Record<string, string[]> & { error?: string }>(`/api/starters?dataset=${q(dataset)}`),
  documents: (dataset: string, split = "validation") =>
    get<Record<string, string[]> & { error?: string }>(`/api/documents?dataset=${q(dataset)}&split=${q(split)}`),
  workspace: () => get<WorkspaceState>("/api/workspace"),
  resetWorkspace: () => post<{ files: Record<string, string> }>("/api/workspace/reset", {}),
  clearMemory: () => post<{ memory: Record<string, string> }>("/api/memory/clear", {}),
  tool: (call: string) => post<{ result: string }>("/api/tool", { call }),
  inspect: (body: { run: string; checkpoint: string; text: string }) => post<any>("/api/inspect", body),
  attend: (attn_id: string, t: number) => post<any>("/api/attend", { attn_id, t }),
  banks: (banks_id: string, t: number) => post<any>("/api/banks", { banks_id, t }),
};

/** Streams a generation: calls onText with each piece; resolves with the final summary (null if none came). */
export async function streamGenerate(body: GenerateBody, onText: (t: string) => void,
                                     signal: AbortSignal): Promise<GenerateInfo | null> {
  const res = await fetch("/api/generate", {
    method: "POST", signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.body) throw new Error(`HTTP ${res.status}`);
  const reader = res.body.getReader(), dec = new TextDecoder();
  let pending = "", info: GenerateInfo | null = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    pending += dec.decode(value, { stream: true });
    let i;
    while ((i = pending.indexOf("\n\n")) >= 0) {
      const line = pending.slice(0, i);
      pending = pending.slice(i + 2);
      if (!line.startsWith("data: ")) continue;
      const ev = JSON.parse(line.slice(6));
      if (ev.error) throw new Error(ev.error);
      if (ev.text) onText(ev.text);
      if (ev.done) info = ev;
    }
  }
  return info;
}
