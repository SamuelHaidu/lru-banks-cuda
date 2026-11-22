// The Setup drawer's settings and the document header they produce (system prompt + listed tools).
import type { GenSettings, Presets } from "./types";

export interface Setup {
  sys: string;
  toolSet: string;
  /** checked tool keys: a tool's name, "atum:" + an Atum09 tool's name, "decoy:" + a decoy tool's line */
  checked: Set<string>;
  exCat: string;
  gen: GenSettings;
}

export const initialSetup: Setup = {
  sys: "", toolSet: "shell + files + code", checked: new Set(), exCat: "",
  gen: { temp: 0, topk: 0, maxtok: 300, rep: 1, rounds: 4 },
};

/** The tool lines to list, in the drawer's order: the tools, the Atum09 tools, the decoys. */
export function toolLines(setup: Setup, presets: Presets | null): string[] {
  if (!presets) return [];
  return [
    ...Object.entries(presets.tool_lines).filter(([n]) => setup.checked.has(n)).map(([, l]) => l),
    ...Object.entries(presets.atum_tool_lines || {}).filter(([n]) => setup.checked.has("atum:" + n)).map(([, l]) => l),
    ...presets.decoys.filter((l) => setup.checked.has("decoy:" + l)),
  ];
}

/** The start of a chat document: <bos>, the system prompt, the tools. */
export function buildHeader(setup: Setup, presets: Presets | null): string {
  const sys = setup.sys.trim(), lines = toolLines(setup, presets);
  let h = "<bos>";
  if (sys) h += `<system>${sys}</system>`;
  if (lines.length) h += "<tools>\n" + lines.join("\n") + "\n</tools>";
  return h;
}

/** A tool set's tools checked (and only those), the decoys left as they are. */
export function applyToolSet(setup: Setup, presets: Presets, set: string): Set<string> {
  const keep = [...setup.checked].filter((k) => k.startsWith("decoy:"));
  return new Set([...keep, ...(presets.tool_sets[set] || [])]);
}
