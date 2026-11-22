export const esc = (s: unknown) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
/** A token as shown: newlines and tabs made visible. */
export const showTok = (s: string) => s.replace(/\n/g, "↵\n").replace(/\t/g, "→");
export const range = (n: number) => [...Array(n).keys()];
/** base64 float32 array (the server's `_b64`) */
export const f32 = (b64: string) => {
  const bin = atob(b64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new Float32Array(u.buffer);
};
/** Tools the server implements (the rest answer "tool not available"). */
export const IMPLEMENTED = new Set(["run", "read_file", "write_file", "calculator", "python", "memory_save",
  "memory_search", "memory_store", "memory_recall", "remember", "recall", "save_note", "get_note"]);
