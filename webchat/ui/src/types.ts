// The server's JSON (webchat/server.py). Inspect payloads are loosely typed: see components/inspect/logic.ts.

export interface Run {
  run: string;
  arch: string;
  dataset: string;
  checkpoints: string[];
  mode: "chat" | "completion";
  inspect: boolean;
}

export interface Presets {
  system: Record<string, string[]>;
  tool_lines: Record<string, string>;
  tool_sets: Record<string, string[]>;
  atum_tool_lines: Record<string, string>;
  atum_implemented: string[];
  decoys: string[];
  examples: Record<string, string[]>;
  data_examples: Record<string, string[]>;
}

export interface WorkspaceState {
  files: Record<string, string>;
  memory: Record<string, string>;
}

export interface GenerateBody {
  run: string;
  checkpoint: string;
  document: string;
  mode?: "chat" | "completion";
  temperature: number;
  top_k: number | null;
  max_new_tokens: number;
  repetition_penalty: number;
}

export interface GenerateInfo {
  done: true;
  tokens: number;
  tps: number;
  cancelled?: boolean;
  prompt_tokens?: number;
  reused_tokens?: number;
}

/** Generation settings shared by the chat (Setup drawer). */
export interface GenSettings {
  temp: number;
  topk: number;
  maxtok: number;
  rep: number;
  rounds: number;
}
