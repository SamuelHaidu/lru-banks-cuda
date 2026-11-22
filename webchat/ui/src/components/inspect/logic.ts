// The Inspect tab's computations, ported from the original page: per token the colour and title of each view, the
// summary and legend, and the side panel (HTML, its clicks handled by the component through data-* attributes).
//
// Views: Loss (how well each token was predicted), Stored (what the states / banks hold at the selected token),
// Recall (what the selected token reads: static memories for min-gru, the banks for lru-banks), Attention (gated-
// deltanet's exact read unrolled over the past tokens), Gate (lru-banks' LRU input gate). Unrolled reads are fetched
// per position from the server (`need`), then cached on the inspection (`ins.att`, `ins.bk`).
import { esc, f32, range, showTok } from "../../util";
export type Ins = any;  // the server's inspect payload, plus the caches and indexes added by `prepare`
export type Mode = "loss" | "stored" | "recall" | "attention" | "gate";

/** The controls' values (the old page read them from its <select>s). */
export interface Sel {
  t: number;
  mode: Mode;
  layer: string;    // "all" or a layer index
  head: string;     // "all" or a head index
  hop: string;      // "0", "1" or "all"
  measure: string;  // "share" or "c"
  hide: number;
}

export interface TokStyle { bg: string; title: string; cls: Set<string>; }
export interface View {
  el: TokStyle[];   // per kept position
  elE: TokStyle[];  // per static-memory entry
  summary: string;  // HTML
  legend: string;   // HTML
  need: { op: "attend" | "banks"; t: number } | null;
  storedVals: Float32Array | null;
  storedOldest: number;
}

const LOSS_CAP = 5;  // nats: the colour saturates here
export const colour = {
  // Loss view: green where the token was the model's top-1 choice (stronger the more confident), red where it was not
  right: (p: number) => `rgba(30, 150, 80, ${(0.10 + 0.30 * p).toFixed(3)})`,
  wrong: (a: number) => `rgba(214, 48, 49, ${(0.22 + 0.58 * a).toFixed(3)})`,
  stored: (a: number) => `rgba(14, 112, 104, ${0.07 + 0.5 * a})`,
  recall: (a: number) => `rgba(109, 62, 214, ${0.05 + 0.4 * a})`,
  gate: (a: number) => `rgba(200, 120, 20, ${(0.04 + 0.5 * a).toFixed(3)})`,
  // Attention (gated-deltanet): positive weights teal, negative (a later write took more than this token wrote) orange
  attnPos: (a: number) => `rgba(14, 112, 104, ${(0.06 + 0.55 * a).toFixed(3)})`,
  attnNeg: (a: number) => `rgba(214, 110, 20, ${(0.06 + 0.55 * a).toFixed(3)})`,
};

/** The server's payload with its arrays decoded and the position indexes built (the old runInspect). */
export function prepare(out: any): Ins {
  out.layers.forEach((L: any) => {
    L.z = f32(L.update_gate); L.c = f32(L.candidate);
    if (L.mem_w) { L.mw = f32(L.mem_w); L.mg = f32(L.mem_gate); }
  });
  out.att = {}; out.bk = {};
  out.rawToKept = new Int32Array(out.raw.length).fill(-1);
  out.kept.forEach((r: number, k: number) => (out.rawToKept[r] = k));
  out.rawToEntry = new Int32Array(out.raw.length).fill(-1);
  if (out.memory) {
    out.memory.entry_pos.forEach((r: number, j: number) => (out.rawToEntry[r] = j));
    out.readRow = {};
    out.memory.read_pos.forEach((t: number, r: number) => (out.readRow[t] = r));
  }
  return out;
}

export const firstScored = (ins: Ins) => { const t = ins.scored.indexOf(1); return t >= 0 ? t : ins.T - 1; };

/** Which modes and controls a payload offers. */
export function modesOf(ins: Ins | null) {
  const gdn = ins?.kind === "gdn", banks = ins?.kind === "banks";
  return {
    stored: !gdn, recall: !gdn, attention: gdn, gate: banks,
    recallEnabled: !ins || !!ins.memory || banks,
  };
}
export function layerOptions(ins: Ins): [string, string][] {
  if (ins.kind === "gdn") return ins.heads.map((_: unknown, i: number) => [String(i), String(i + 1)]);
  return [["all", "All"], ...range(ins.kind === "banks" ? ins.num_layers : ins.layers.length).map((i) => [String(i), String(i + 1)] as [string, string])];
}
export function headOptions(ins: Ins): [string, string][] {
  const n = ins.kind === "gdn" ? ins.heads[0].half_life.length : ins.kind === "banks" ? ins.num_heads : 0;
  return [["all", "All"], ...range(n).map((h) => [String(h), String(h + 1)] as [string, string])];
}
export function controlsShown(ins: Ins | null, m: Mode) {
  const banks = ins?.kind === "banks";
  return {
    layer: m !== "loss",
    head: (m === "attention" || (banks && m !== "loss")) && m !== "gate",
    hop: banks && m === "recall" && ins.hops > 1,
    measure: m === "attention",
    hide: m === "stored",
  };
}

const layersShown = (ins: Ins, sel: Sel) => (sel.layer === "all" ? ins.layers.map((_: unknown, i: number) => i) : [+sel.layer]);

// exact weight of every past token j in each channel at position t: w[j][c] = gate[j][c] * prod_{s>j} keep[s][c]
function weights(ins: Ins, L: any, t: number) {
  const H = ins.H, w = new Float32Array((t + 1) * H), P = new Float32Array(H).fill(1);
  for (let j = t; j >= 0; j--) {
    const keepOn = ins.reset[j] ? 0 : 1;
    for (let c = 0; c < H; c++) { const z = L.z[j * H + c]; w[j * H + c] = z * P[c]; P[c] *= (1 - z) * keepOn; }
  }
  return w;
}
// each past token's share of the state at t: per channel its part of the channel's content (weight x |candidate|),
// then the mean over the channels
function storedShares(ins: Ins, L: any, t: number) {
  const H = ins.H, w = weights(ins, L, t), share = new Float32Array(t + 1);
  let n = 0;
  for (let c = 0; c < H; c++) {
    let tot = 0; for (let j = 0; j <= t; j++) tot += w[j * H + c] * Math.abs(L.c[j * H + c]);
    if (tot <= 1e-12) continue;
    n++; for (let j = 0; j <= t; j++) share[j] += w[j * H + c] * Math.abs(L.c[j * H + c]) / tot;
  }
  if (n) for (let j = 0; j <= t; j++) share[j] /= n;
  return share;
}
// the selected position's read weights over the memory entries (mean over the shown layers), and the gates
function readsAt(ins: Ins, sel: Sel, t: number) {
  const M = ins.memory, r = ins.readRow ? ins.readRow[t] : undefined;
  if (!M || r === undefined) return null;
  const E = M.entries.length, nm = M.names.length, ls = layersShown(ins, sel), w = new Float32Array(E), g = new Float32Array(nm);
  for (const li of ls) {
    const L = ins.layers[li];
    for (let j = 0; j < E; j++) w[j] += L.mw[r * E + j] / ls.length;
    for (let i = 0; i < nm; i++) g[i] += L.mg[r * nm + i] / ls.length;
  }
  return { w, g, doc: M.read_doc[r] };
}

// Attention (gated-deltanet): o_t = sum_i c_i v_i; "share": c_i (v_i . o_t) / |o_t|^2, "weight": c_i itself.
const nHeads = (ins: Ins) => ins.heads[0].half_life.length;
function attVals(ins: Ins, A: any, l: number, hsel: string, meas: string) {
  const arr = meas === "c" ? A.c : A.share, H = nHeads(ins), n = A.n, out = new Float32Array(n);
  const hs = hsel === "all" ? range(H) : [+hsel];
  for (const h of hs) for (let i = 0; i < n; i++) out[i] += arr[(l * H + h) * n + i] / hs.length;
  return out;
}
function attSplit(ins: Ins, A: any, l: number, h: number) {  // |share| of this token / the conv's window / older
  const H = nHeads(ins), n = A.n, base = (l * H + h) * n, w = ins.conv_size - 1, sp = [0, 0, 0];
  for (let i = 0; i < n; i++) { const d = n - 1 - i; sp[d === 0 ? 0 : d <= w ? 1 : 2] += Math.abs(A.share[base + i]); }
  const tot = sp[0] + sp[1] + sp[2] || 1; return sp.map((x) => x / tot);
}
export const decodeAttend = (r: any) => ({ start: r.start, n: r.n, c: f32(r.c), share: f32(r.share), err: r.err });

// Banks (lru-banks): position t's reads and bank contents unrolled exactly over the document's positions.
function banksSel(ins: Ins, sel: Sel) {
  const pick = (v: string, n: number) => (v === "all" ? range(n) : [+v]);
  return { ls: pick(sel.layer, ins.num_layers), hs: pick(sel.head, ins.num_heads), os: pick(sel.hop, ins.hops) };
}
function bankVals(ins: Ins, B: any, kind: string, ls: number[], hs: number[], os: number[]) {
  const n = B.n, H = ins.num_heads, out = new Float32Array(n);
  if (kind === "read") {
    const pairs = ls.flatMap((l) => os.filter((o) => o < ins.layer_hops[l]).map((o) => [l, o])), k = pairs.length * hs.length;
    if (!k) return out;
    for (const [l, o] of pairs) for (const h of hs) {
      const base = ((l * ins.hops + o) * H + h) * n;
      for (let i = 0; i < n; i++) out[i] += B.read[base + i] / k;
    }
  } else {
    const k = ls.length * hs.length;
    for (const l of ls) for (const h of hs) {
      const base = (l * H + h) * n;
      for (let i = 0; i < n; i++) out[i] += B.stored[base + i] / k;
    }
  }
  return out;
}
export const decodeBanks = (r: any) => ({ ...r, read: f32(r.read), stored: f32(r.stored) });
const gatedLayers = (ins: Ins, ls: number[]) => ls.filter((l) => ins.gate[l]);

function banksView(ins: Ins, sel: Sel, v: View): [string, string] {
  const el = v.el, T = ins.T, insT = sel.t;
  const { ls, hs, os } = banksSel(ins, sel);
  if (sel.mode === "gate") {
    const gl = gatedLayers(ins, ls);
    if (!gl.length) return ["Layer 1 has no input gate: its LRU takes every token in full (g = 1).", ""];
    const g = new Float32Array(T); let mx = 1e-6;
    for (let j = 0; j < T; j++) { for (const l of gl) g[j] += ins.gate[l][j] / gl.length; mx = Math.max(mx, 1 - g[j]); }
    for (let j = 0; j < T; j++) {
      el[j].bg = colour.gate((1 - g[j]) / mx);
      el[j].title = gl.map((l) => `layer ${l + 1}: ${ins.gate[l][j].toFixed(3)}`).join(", ");
    }
    const means = gl.map((l) => ins.gate[l].reduce((a: number, b: number) => a + b, 0) / T);
    let lo = 0; for (let j = 1; j < T; j++) if (g[j] < g[lo]) lo = j;
    return [`LRU input gate g (1: the state takes the token in full, 0: skips it; its decay is lambda^g). Mean over the text:
      ${gl.map((l, i) => `layer ${l + 1} <b>${means[i].toFixed(3)}</b>`).join(", ")}. Lowest ${gl.length > 1 ? "mean " : ""}gate
      <b>${g[lo].toFixed(3)}</b> at "${esc(showTok(ins.tokens[lo]))}" (position ${lo}).`,
      `takes the token <i style="background:linear-gradient(90deg, ${colour.gate(0)}, ${colour.gate(1)})"></i> skips it most (${(1 - mx).toFixed(3)})`];
  }
  for (let j = insT + 1; j < T; j++) el[j].cls.add("after");
  const B = ins.bk[insT];
  if (!B) { v.need = { op: "banks", t: insT }; return ["Unrolling the banks at this token", ""]; }
  for (let j = 0; j < B.start; j++) el[j].cls.add("off");
  const what = ls.length === 1 && ls[0] === 0 ? "tokens" : "positions";
  const mixed = ls.some((l) => l > 0) ? ` Above layer 1 a position's value is its stream (the token plus what earlier layers
    read and the FFNs added), so this shows where the read lands; "What the read retrieves" shows what it carries.` : "";
  const who = `${ls.length === 1 ? "layer " + (ls[0] + 1) : "all layers"}, ${hs.length === 1 ? "head " + (hs[0] + 1) : "mean of the heads"}`;
  if (sel.mode === "recall") {
    if (!ls.some((l) => os.some((o) => o < ins.layer_hops[l]))) return [`Layer ${ls[0] + 1} reads its banks once: choose read 1.`, ""];
    const vals = bankVals(ins, B, "read", ls, hs, os);
    let mx = 1e-12; for (let i = 0; i < B.n; i++) mx = Math.max(mx, vals[i]);
    for (let i = 0; i < B.n; i++) {
      const j = B.start + i;
      el[j].bg = colour.recall(Math.sqrt(vals[i] / mx)); el[j].title = `${(100 * vals[i]).toFixed(2)}% of the read`;
    }
    let self = 0, near = 0, far = 0;
    for (let i = 0; i < B.n; i++) { const d = B.n - 1 - i; if (d === 0) self += vals[i]; else if (d <= 8) near += vals[i]; else far += vals[i]; }
    const hop = os.length > 1 ? "both reads" : `read ${os[0] + 1}`;
    return [`<b>${who}, ${hop}</b>: <b>${(100 * self).toFixed(0)}%</b> from the selected position itself,
      <b>${(100 * near).toFixed(0)}%</b> from the 8 ${what} before, <b>${(100 * far).toFixed(0)}%</b> from older ones.
      Exact weights on the values (they rebuild the read to ${B.err.toExponential(1)}).${mixed}`,
      `less <i style="background:linear-gradient(90deg, ${colour.recall(0)}, ${colour.recall(1)})"></i> more`];
  }
  const hide = Math.max(0, sel.hide || 0), oldest = insT - hide, vv = bankVals(ins, B, "stored", ls, hs, os);
  const vals = new Float32Array(insT + 1); let mx = 1e-12, old = 0;
  for (let i = 0; i < B.n; i++) { const j = B.start + i; vals[j] = vv[i]; if (j <= oldest) { mx = Math.max(mx, vv[i]); old += vv[i]; } }
  for (let i = 0; i < B.n; i++) {
    const j = B.start + i;
    if (j > oldest) { el[j].cls.add("off"); continue; }
    el[j].bg = colour.stored(Math.sqrt(vals[j] / mx)); el[j].title = `${(100 * vals[j]).toFixed(2)}% of the banks`;
  }
  v.storedVals = vals; v.storedOldest = oldest;
  const filled = ls.map((l) => hs.reduce((a, h) => a + B.filled[l][h], 0) / hs.length);
  return [oldest < B.start ? `Nothing older than the last ${hide} ${what} in this document yet.`
    : `<b>${who}</b>: ${what} older than the last ${hide} hold <b>${(100 * old).toFixed(1)}%</b> of the banks (each slot's
      average counted equally). Slots in use: ${filled.map((f) => f.toFixed(0)).join(", ")} of ${ins.slots}.${mixed}`,
    `less <i style="background:linear-gradient(90deg, ${colour.stored(0)}, ${colour.stored(1)})"></i> more`];
}

/** Every token's colour and title for the selection, and the summary and legend (the old insSelect). */
export function computeView(ins: Ins, sel: Sel): View {
  const T = ins.T, insT = sel.t;
  const mk = (): TokStyle => ({ bg: "", title: "", cls: new Set() });
  const v: View = { el: range(T).map(mk), elE: (ins.memory?.entries || []).map(mk), summary: "", legend: "",
                    need: null, storedVals: null, storedOldest: -1 };
  const el = v.el;
  let summary = "", legend = "";
  if (ins.kind === "banks" && sel.mode !== "loss") {
    [summary, legend] = banksView(ins, sel, v);
  } else if (sel.mode === "loss") {
    let n = 0, sum = 0, sumH = 0, top1 = 0;
    for (let j = 1; j < T; j++) {
      if (!ins.scored[j - 1]) { el[j].cls.add("off"); continue; }
      const loss = ins.loss[j - 1], rank = ins.rank[j - 1], p = Math.exp(-loss);
      n++; sum += loss; sumH += ins.entropy[j - 1]; top1 += rank === 0 ? 1 : 0;
      el[j].bg = rank === 0 ? colour.right(p) : colour.wrong(Math.min(1, loss / LOSS_CAP));
      el[j].title = `${rank === 0 ? "top-1" : "rank " + (rank + 1)}, p = ${p.toFixed(3)}, loss ${loss.toFixed(3)} nats`;
    }
    el[0].cls.add("off");
    summary = n ? `<b>${n}</b> ${n === T - 1 ? "tokens" : "assistant tokens"} scored: <b>${(100 * top1 / n).toFixed(1)}%</b> top-1 (green), ${n - top1} not (red). Mean loss <b>${(sum / n).toFixed(3)}</b> nats (${(sum / n / Math.LN2).toFixed(3)} bits per token), mean entropy <b>${(sumH / n / Math.LN2).toFixed(2)}</b> bits.`
      : "This text has no assistant tokens: the training scores only what the assistant writes.";
    legend = `top-1 <i style="width:44px; background:linear-gradient(90deg, ${colour.right(0)}, ${colour.right(1)})"></i>
      &nbsp; missed, loss 0 <i style="background:linear-gradient(90deg, ${colour.wrong(0)}, ${colour.wrong(1)})"></i> ${LOSS_CAP}+ nats`;
  } else if (sel.mode === "stored") {
    const hide = Math.max(0, sel.hide || 0), oldest = insT - hide, ls = layersShown(ins, sel);
    const vals = new Float32Array(insT + 1), per: [number, number][] = [];
    for (const li of ls) {
      const s = storedShares(ins, ins.layers[li], insT); let old = 0;
      for (let j = 0; j <= insT; j++) { vals[j] += s[j] / ls.length; if (j <= oldest) old += s[j]; }
      per.push([li, old]);
    }
    let mx = 1e-12; for (let j = 0; j <= oldest; j++) mx = Math.max(mx, vals[j]);
    for (let j = 0; j < T; j++) {
      if (j > insT) { el[j].cls.add("after"); continue; }
      if (j > oldest) { el[j].cls.add("off"); continue; }
      el[j].bg = colour.stored(Math.sqrt(vals[j] / mx));
      el[j].title = `${(100 * vals[j]).toFixed(2)}% of the state`;
    }
    v.storedVals = vals; v.storedOldest = oldest;
    const olds = per.map((x) => 100 * x[1]), mean = olds.reduce((a, b) => a + b, 0) / olds.length;
    summary = oldest < 0 ? `Nothing older than the last ${hide} tokens yet.` : per.length === 1
      ? `Tokens older than the last ${hide} hold <b>${olds[0].toFixed(1)}%</b> of layer ${per[0][0] + 1}'s state at the selected token.`
      : `Tokens older than the last ${hide} hold <b>${mean.toFixed(1)}%</b> of the state at the selected token, from ${Math.min(...olds).toFixed(1)}% to ${Math.max(...olds).toFixed(1)}% across the layers.`;
    legend = `less <i style="background:linear-gradient(90deg, ${colour.stored(0)}, ${colour.stored(1)})"></i> more`;
  } else if (sel.mode === "attention") {
    const A = ins.att[insT];
    for (let j = insT + 1; j < T; j++) el[j].cls.add("after");
    if (!A) { v.need = { op: "attend", t: insT }; summary = "Unrolling the read at this token"; }
    else {
      const l = +sel.layer, hs = sel.head, meas = sel.measure, vals = attVals(ins, A, l, hs, meas);
      let mx = 1e-12; for (let i = 0; i < A.n; i++) mx = Math.max(mx, Math.abs(vals[i]));
      for (let j = 0; j < A.start; j++) el[j].cls.add("off");
      for (let i = 0; i < A.n; i++) {
        const j = A.start + i, a = Math.sqrt(Math.abs(vals[i]) / mx);
        el[j].bg = vals[i] >= 0 ? colour.attnPos(a) : colour.attnNeg(a);
        el[j].title = meas === "c" ? `weight ${vals[i].toFixed(4)} on this token's value` : `${(100 * vals[i]).toFixed(1)}% of the read`;
      }
      const sh = attVals(ins, A, l, hs, "share"), w = ins.conv_size - 1; let self = 0, near = 0, far = 0;
      for (let i = 0; i < A.n; i++) { const d = A.n - 1 - i; if (d === 0) self += sh[i]; else if (d <= w) near += sh[i]; else far += sh[i]; }
      const who = hs === "all" ? `layer ${l + 1}, mean of the heads`
        : `layer ${l + 1}, head ${+hs + 1} (half-life ${ins.heads[l].half_life[+hs]} tokens, write strength ${ins.heads[l].beta[+hs].toFixed(2)})`;
      summary = `<b>${who}</b>: the read at the selected token is <b>${(100 * self).toFixed(0)}%</b> its own value,
        <b>${(100 * near).toFixed(0)}%</b> the ${w} tokens before, <b>${(100 * far).toFixed(0)}%</b> older ones.
        Exact unroll of the memory (rebuilds the read to ${A.err.toExponential(1)}); each value already mixes its token with the ${w} before it (the short conv).`;
      legend = `negative <i style="width:60px; background:linear-gradient(90deg, ${colour.attnNeg(1)}, ${colour.attnNeg(0)})"></i>
        <i style="width:60px; background:linear-gradient(90deg, ${colour.attnPos(0)}, ${colour.attnPos(1)})"></i> positive`;
    }
  } else {  // recall: the static memories (min-gru)
    const M = ins.memory, ls = layersShown(ins, sel);
    let mx = 1e-12; const add = new Float32Array(T);
    for (const li of ls) { const a = ins.layers[li].mem_add; for (let j = 0; j < T; j++) add[j] += a[j] / ls.length; }
    for (let j = 0; j < T; j++) mx = Math.max(mx, add[j]);
    for (let j = 0; j < T; j++) {
      if (add[j] > 0) { el[j].bg = colour.recall(add[j] / mx); el[j].title = `the memories add ${add[j].toFixed(3)} to the stream here`; }
      else el[j].cls.add("off");
    }
    const rd = readsAt(ins, sel, insT);
    if (rd) {
      let wm = 1e-12; M.entries.forEach((_: unknown, j: number) => { if (M.entry_doc[j] === rd.doc) wm = Math.max(wm, rd.w[j]); });
      M.entries.forEach((_: unknown, j: number) => {
        if (M.entry_doc[j] !== rd.doc) return;
        v.elE[j].bg = colour.recall(rd.w[j] / wm);
        v.elE[j].title = `${(100 * rd.w[j]).toFixed(2)}% of the ${M.names[M.entry_kind[j]]} read`;
      });
    }
    const counts = M.names.map((_: unknown, i: number) => M.entry_kind.filter((k: number) => k === i).length);
    summary = `${M.names.map((n: string, i: number) => `${n} <b>${counts[i]}</b>`).join(", ")} tokens in the memories, read by <b>${M.read_pos.length}</b> positions.
      Assistant tokens are coloured by how much the memories add there; the dimmed system and tool tokens by how much the selected token reads them.`;
    legend = `less <i style="background:linear-gradient(90deg, ${colour.recall(0)}, ${colour.recall(1)})"></i> more`;
  }
  el[insT].cls.add("sel");
  v.summary = summary; v.legend = legend;
  return v;
}

const posPill = (ins: Ins, j: number, val: number) =>
  `<button class="pill" data-j="${j}" title="position ${j}">${esc(showTok(ins.tokens[j]))} ${(100 * val).toFixed(0)}%</button>`;
const topPos = (B: any, val: Float32Array, k: number) => range(B.n).sort((a, b) => val[b] - val[a]).slice(0, k).filter((i) => val[i] > 0.005);

function banksSide(ins: Ins, sel: Sel, t: number) {
  let h = `<h3>LRU input gate at this token</h3><table class="lens">` + range(ins.num_layers).map((l) => {
    const g = ins.gate[l];
    return `<tr><td>Layer ${l + 1}</td><td>${g ? `<div class="cand"><span class="bar"><i style="width:${(100 * g[t]).toFixed(0)}%"></i></span>
      <span class="p" style="width:46px">${g[t].toFixed(3)}</span></div>` : `<span class="note">no gate: takes every token</span>`}</td></tr>`;
  }).join("") + `</table>`;
  if (sel.mode !== "recall") return h;
  const B = ins.bk[t]; if (!B) return h;
  const { ls, hs, os } = banksSel(ins, sel), one = ls.length === 1, H = ins.num_heads;
  h += `<h3>What each read retrieved</h3><p class="note" style="margin:-4px 0 6px">Per layer and read (${hs.length === 1 ? "head " + (hs[0] + 1) : "all heads"}):
    the positions the query took most, then the tokens the read's addition to the stream raises most (o W_o through the readout).
    Click a row to show it.</p><table class="lens">`;
  for (const l of range(ins.num_layers)) for (const o of range(ins.layer_hops[l])) {
    const val = bankVals(ins, B, "read", [l], hs, [o]), says = hs.length === 1 ? B.says[l][o][hs[0]] : B.says_all[l][o];
    const on = one && ls[0] === l && os.length === 1 && os[0] === o;
    h += `<tr class="rd${on ? " on" : ""}" data-l="${l}" data-o="${o}"><td>L${l + 1}${ins.layer_hops[l] > 1 ? " read " + (o + 1) : ""}</td>
      <td>${topPos(B, val, 3).map((i) => posPill(ins, B.start + i, val[i])).join("")}<div class="says">${says.length ? "raises " + says.map((s: string) => `<span class="pill">${esc(showTok(s))}</span>`).join("") : "adds nothing to the stream (output projection 0)"}</div></td></tr>`;
  }
  h += `</table>`;
  if (!one) return h + `<p class="note">Select a layer (a row above) to see its heads and slots.</p>`;
  const l = ls[0], lo = range(ins.layer_hops[l]);
  h += `<h3>Layer ${l + 1}: every head</h3><p class="note" style="margin:-4px 0 8px">The position each head's read takes most and its share. Click to select the head.</p>`
    + `<div class="agrid" style="grid-template-columns:28px repeat(${lo.length}, minmax(0, 1fr))"><span></span>`
    + lo.map((o) => `<span class="hd">read ${o + 1}</span>`).join("")
    + range(H).map((k) => `<span class="ln">H${k + 1}</span>` + lo.map((o) => {
        const val = bankVals(ins, B, "read", [l], [k], [o]), i = topPos(B, val, 1)[0] ?? B.n - 1;
        const on = hs.length === 1 && hs[0] === k && os.includes(o);
        return `<button class="acell${on ? " on" : ""}" data-h="${k}" data-o="${o}" title="${esc(B.says[l][o][k].map(showTok).join(" "))}">
          <span class="tt">${esc(showTok(ins.tokens[B.start + i]))}</span><span class="pc">${(100 * val[i]).toFixed(0)}% &middot; ${B.says[l][o][k].length ? "raises " + esc(showTok(B.says[l][o][k][0])) : "adds nothing"}</span></button>`;
      }).join("")).join("")
    + `</div>`;
  if (hs.length !== 1) return h;
  const k = hs[0];
  for (const o of os.filter((o) => o < ins.layer_hops[l])) {
    h += `<h3>Layer ${l + 1}, head ${k + 1}${ins.layer_hops[l] > 1 ? ", read " + (o + 1) : ""}: slots read</h3>`;
    const sl = B.slots[l][o][k];
    h += sl.length ? sl.map(([m, p, items]: [number, number, [number, number][]]) => `<div class="facts" style="margin:6px 0 2px"><b>Slot ${m + 1}</b> ${(100 * p).toFixed(0)}% of the read,
        half-life ${ins.slot_half_life[m]} tokens; holds:</div>` + items.map(([j, s]) => posPill(ins, j, s)).join("")).join("")
      : `<p class="note">The banks are empty here.</p>`;
  }
  return h;
}

function cands(ins: Ins, rows: [number, number][], actual: number | null) {
  const mx = rows.length ? rows[0][1] : 1;
  return rows.map(([i, p]) => `<div class="cand"><span class="pill${i === actual ? " hit" : ""}">${esc(showTok(ins.vocab[i]))}</span>
    <span class="bar"><i style="width:${(100 * p / Math.max(mx, 1e-9)).toFixed(0)}%"></i></span><span class="p">${(100 * p).toFixed(0)}%</span></div>`).join("");
}

/** The side panel's HTML for the selected token: its prediction, the view's details, the logit lens. Clicks:
 *  [data-j] scrolls to a position; .acell[data-l/h/o] and tr.rd[data-l/o] select a layer / head / read. */
export function renderSide(ins: Ins, sel: Sel, v: View): string {
  const t = sel.t, tok = ins.tokens[t], next = t + 1 < ins.T ? ins.ids[t + 1] : null, L = ins.lens.length - 1;
  let h = `<h3>Selected token</h3><span class="big-tok">${esc(showTok(tok))}</span><span class="note" style="margin-left:8px">position ${t}</span>`;
  const scored = t > 0 && ins.scored[t - 1];
  if (scored) {
    const loss = ins.loss[t - 1], rank = ins.rank[t - 1];
    h += `<div class="facts">${rank === 0 ? `<b style="color:#1E9650">Top-1</b>` : `<b style="color:#D63031">Rank ${rank + 1}</b>`}: predicted with <b>${(100 * Math.exp(-loss)).toFixed(1)}%</b>, loss <b>${loss.toFixed(3)}</b> nats, entropy ${(ins.entropy[t - 1] / Math.LN2).toFixed(2)} bits</div>`;
  } else if (t > 0) h += `<div class="facts">Not scored: the training only scores the assistant's tokens.</div>`;
  if (ins.kind === "banks") h += banksSide(ins, sel, t);
  if (sel.mode === "stored" && v.storedVals) {
    const top: [number, number][] = []; for (let j = 0; j <= v.storedOldest; j++) top.push([v.storedVals[j], j]);
    top.sort((a, b) => b[0] - a[0]);
    h += `<h3>Most stored</h3>` + (top.length ? top.slice(0, 14).filter((x) => x[0] > 0).map(([val, j]) =>
      `<button class="pill" data-j="${j}" title="position ${j}">${esc(showTok(ins.tokens[j]))} ${(100 * val).toFixed(1)}%</button>`).join("") : `<p class="note">Nothing old enough yet.</p>`);
  }
  if (sel.mode === "recall" && ins.memory) {
    const M = ins.memory, rd = readsAt(ins, sel, t);
    if (!rd) h += `<h3>Memory read</h3><p class="note">This token does not read the memories. Only the assistant's turns read them: <code>&lt;assistant&gt;</code> and what follows.</p>`;
    else M.names.forEach((name: string, i: number) => {
      const js = M.entries.map((_: unknown, j: number) => j).filter((j: number) => M.entry_doc[j] === rd.doc && M.entry_kind[j] === i);
      h += `<h3>Reads from the ${name}</h3>`;
      if (!js.length) { h += `<p class="note">Empty in this document.</p>`; return; }
      js.sort((a: number, b: number) => rd.w[b] - rd.w[a]);
      h += `<div class="facts" style="margin:-4px 0 6px">Gate ${rd.g[i].toFixed(3)}: how much of this read reaches the stream</div>`
        + js.slice(0, 6).map((j: number) => `<div class="cand"><span class="pill${next !== null && M.entries[j] === ins.vocab[next] ? " hit" : ""}">${esc(showTok(M.entries[j]))}</span>
          <span class="bar"><i style="width:${(100 * rd.w[j] / rd.w[js[0]]).toFixed(0)}%"></i></span><span class="p">${(100 * rd.w[j]).toFixed(0)}%</span></div>`).join("");
    });
  }
  if (sel.mode === "attention" && ins.att && ins.att[t]) {
    const A = ins.att[t], H = nHeads(ins), l = +sel.layer, hs = sel.head, meas = sel.measure;
    h += `<h3>Every head at this token</h3><p class="note" style="margin:-4px 0 8px">The past token each head reads most, its share of the read, and the split: this token / the ${ins.conv_size - 1} before / older.</p>`
      + `<div class="agrid" style="grid-template-columns:28px repeat(${H}, minmax(0, 1fr))"><span></span>`
      + range(H).map((k) => `<span class="hd">head ${k + 1}</span>`).join("")
      + ins.heads.map((hd: any, li: number) => `<span class="ln">L${li + 1}</span>` + range(H).map((k) => {
          const base = (li * H + k) * A.n; let j = 0;
          for (let i = 1; i < A.n; i++) if (Math.abs(A.share[base + i]) > Math.abs(A.share[base + j])) j = i;
          const sp = attSplit(ins, A, li, k), on = li === l && (hs === "all" || +hs === k);
          return `<button class="acell${on ? " on" : ""}" data-l="${li}" data-h="${k}" title="half-life ${hd.half_life[k]} tokens, write strength ${hd.beta[k].toFixed(2)}">
            <span class="tt">${esc(showTok(ins.tokens[A.start + j]))}</span><span class="pc">${(100 * A.share[base + j]).toFixed(0)}%</span>
            <span class="asplit">${sp.map((x) => `<i style="width:${(100 * x).toFixed(0)}%"></i>`).join("")}</span></button>`;
        }).join("")).join("")
      + `</div>`;
    const val = attVals(ins, A, l, hs, meas), top = [...val.keys()].sort((a, b) => Math.abs(val[b]) - Math.abs(val[a])).slice(0, 12);
    h += `<h3>Most read: layer ${l + 1}${hs === "all" ? ", all heads" : ", head " + (+hs + 1)}</h3>`
      + top.map((i) => `<button class="pill" data-j="${A.start + i}" title="position ${A.start + i}" style="background:${val[i] >= 0 ? colour.attnPos(0.5) : colour.attnNeg(0.5)}">${esc(showTok(ins.tokens[A.start + i]))} ${meas === "c" ? val[i].toFixed(3) : (100 * val[i]).toFixed(1) + "%"}</button>`).join("");
  }
  if (scored) h += `<h3>What the model expected here</h3>${cands(ins, ins.lens[L][t - 1], ins.ids[t])}`;
  h += `<h3>Next token, layer by layer</h3>`
    + (next !== null ? `<div class="facts" style="margin:-4px 0 6px">Actual next: <span class="pill hit">${esc(showTok(ins.vocab[next]))}</span></div>` : "")
    + `<table class="lens">` + ins.lens.map((rows: any, n: number) => `<tr class="${n === L ? "final" : ""}"><td>${n === 0 ? "Embedding" : "Layer " + n}</td>
       <td>${cands(ins, rows[t].slice(0, n === L ? 5 : 3), next)}</td></tr>`).join("") + `</table>`;
  return h;
}
