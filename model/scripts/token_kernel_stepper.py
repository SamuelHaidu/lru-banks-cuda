"""`model scripts token-kernel-stepper`: step through a story with a `rope-token-kernel` run.

At every step (predicting the next token) the context vector is a normalized weighted sum
of the last `window` tokens. The page shows, for the current step:

- the story with each in-window token shaded by its *energy* (its share |w_j| / sum |w|),
- energy per context position against distance, with each token's prior s, forgetting
  lambda, mention number and saturation factor delta, and kernel value w(k^lambda),
- the *state composition*: energy summed per token type (all mentions of a word together),
- energy of the main token types across the whole story (how a word's presence evolves),
- the model's top predictions vs the actual next token, the effective number of tokens in
  the state (1 / sum energy^2) and the share held by document padding.

By default the first validation story with at least `--min-tokens` tokens.
"""

import argparse
import html as _html
import json
from pathlib import Path
from typing import List

import numpy as np

from model.scripts._model_io import load_full_model, load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="model scripts token-kernel-stepper",
                                description="Interactive per-step state composition of a rope-token-kernel run.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--text", default=None, help="Text to step through (default: a validation story).")
    p.add_argument("--min-tokens", type=int, default=220, help="Pick the first validation story at least this long.")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--doc-skip", type=int, default=0, help="Skip this many qualifying stories.")
    return p


def _story(config, tokenizer, args) -> List[int]:
    from model import config as cfg_mod
    from model.data import _document_bounds, load_tokens

    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
    doc_id = int(meta["doc_token_id"])
    if args.text is not None:
        return [doc_id] + tokenizer.encode(args.text).ids[: args.max_tokens - 1]
    path = str(cfg_mod.dataset_split_path(config["data"]["dataset"], "validation"))
    tokens = load_tokens(path)
    starts, ends = _document_bounds(path, tokens, doc_id, True)
    skip = args.doc_skip
    for s, e in zip(starts, ends):
        if e - s >= args.min_tokens:
            if skip == 0:
                return [int(t) for t in tokens[s: min(e, s + args.max_tokens)]]
            skip -= 1
    raise SystemExit(f"No validation story with >= {args.min_tokens} tokens.")


def main(argv: List[str]) -> None:
    args = build_parser().parse_args(argv)
    import torch
    from keras import ops

    model, config, ckpt = load_full_model(args.run_name, args.checkpoint)
    if config["arch"] != "rope-token-kernel":
        raise SystemExit(f"Run {args.run_name!r} is {config['arch']!r}, not rope-token-kernel.")
    tok = load_tokenizer(config)
    ids = _story(config, tok, args)
    N, T = len(ids), model.window_size
    label = lambda i: (tok.id_to_token(int(i)) or f"<{i}>").replace("Ġ", " ").replace("Ċ", "\\n")
    words = [label(i) for i in ids]

    # Windows predicting ids[1..N-1]: left-zero-padded like the data pipeline.
    steps = N - 1
    x = np.zeros((steps, T), np.int32)
    for i in range(steps):
        ctx = ids[max(0, i + 1 - T): i + 1]
        x[i, T - len(ctx):] = ctx
    dev = model.token_embedding.embeddings.value.device
    xt = torch.as_tensor(x, device=dev)
    to_np = lambda t: np.asarray(ops.convert_to_numpy(t), dtype=np.float64)
    with torch.no_grad():
        logits = to_np(model(xt))
        idsx = ops.cast(xt, "int32")
        num = model.token_numbers(idsx)
        log_k = ops.convert_to_tensor(model._log_distance)[None]
        mention = None
        if model.saturation:
            same = ops.cast(ops.equal(idsx[:, :, None], idsx[:, None, :]), "float32")
            mention = ops.sum(same * ops.convert_to_tensor(np.tril(np.ones((T, T), "float32")))[None], axis=-1)
        w = to_np(model._weights(idsx, log_k, mention))  # (steps, T)
        lam = num["lambda"]
        kern = to_np(model.shared_kernel(ops.exp(log_k * lam) if lam is not None else ops.exp(log_k)))
        s = to_np(num["s"]) if num["s"] is not None else np.ones_like(w)
        lam_np = to_np(lam) if lam is not None else np.ones_like(w)
        delta = to_np(model._delta(mention, num["kappa"])) if model.saturation else np.ones_like(w)
        ment = to_np(mention) if mention is not None else np.ones_like(w)
    energy = np.abs(w) / (np.abs(w).sum(-1, keepdims=True) + 1e-12)

    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    top = np.argsort(-probs, axis=-1)[:, :5]

    data = []
    for i in range(steps):
        n_real = min(i + 1, T)
        first = i + 1 - n_real  # story index of the oldest in-window token
        cols = range(T - n_real, T)
        pad_share = float(energy[i, : T - n_real].sum())
        data.append({
            "pos": [first + (c - (T - n_real)) for c in cols],  # story positions in the window
            "e": [round(float(energy[i, c]), 5) for c in cols],
            "s": [round(float(s[i, c]), 3) for c in cols],
            "l": [round(float(lam_np[i, c]), 3) for c in cols],
            "n": [int(ment[i, c]) for c in cols],
            "d": [round(float(delta[i, c]), 3) for c in cols],
            "f": [round(float(kern[i, c]), 4) for c in cols],
            "pad": round(pad_share, 4),
            "neff": round(float(1.0 / (energy[i] ** 2).sum()), 1),
            "top": [[label(t), round(float(probs[i, t]), 4)] for t in top[i]],
            "tgt": [label(ids[i + 1]), round(float(probs[i, ids[i + 1]]), 4)],
        })

    # Token types with the most total energy over the story, for the evolution chart.
    tot = {}
    for i in range(steps):
        for p_, e_ in zip(data[i]["pos"], data[i]["e"]):
            tot[ids[p_]] = tot.get(ids[p_], 0.0) + e_
    main_types = [t for t, _ in sorted(tot.items(), key=lambda kv: -kv[1])[:10]]
    evo = {label(t): [round(sum(e for p_, e in zip(d["pos"], d["e"]) if ids[p_] == t), 4) for d in data] for t in main_types}

    payload = json.dumps({"words": words, "steps": data, "evo": evo, "T": T})
    out = ckpt.parent.parent / "reports" / "token_kernel_stepper.html" if ckpt.parent.name == "checkpoints" \
        else ckpt.parent / "reports" / "token_kernel_stepper.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__TITLE__", _html.escape(f"{args.run_name}: state composition, {N} tokens"))
                   .replace("__DATA__", payload))
    print(f"Wrote {out}  ({N} tokens, window {T})")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>token kernel stepper</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{font-family:system-ui,sans-serif;margin:18px;max-width:1300px;color:#222}
 .panel{border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin:10px 0}
 #story span{padding:1px 1px;border-radius:3px;cursor:pointer;line-height:1.9;font-size:15px}
 #story .cur{outline:2px solid #000}
 #story .nxt{text-decoration:underline;text-decoration-thickness:3px;color:#1a9850}
 #story .out{color:#bbb}
 .row{display:flex;gap:14px;flex-wrap:wrap}
 .stat{font-size:14px} .stat b{font-size:16px}
 table{border-collapse:collapse;font-size:13px} td{padding:2px 8px}
 button{font-size:14px;padding:4px 12px}
</style></head><body>
<h2>__TITLE__</h2>
<div class="panel">
 <button onclick="go(cur-1)">&larr;</button> <button onclick="go(cur+1)">&rarr;</button>
 <button id="play" onclick="toggle()">play</button>
 <input type="range" id="slider" min="0" value="0" style="width:60%" oninput="go(+this.value)">
 <span id="stepl"></span>
 <div class="note" style="font-size:12px;color:#777">Click a word to jump there. Shading = energy (share of the state) at the current step;
 grey = outside the window; black box = last token read; green underline = the token to predict.</div>
</div>
<div class="panel" id="story"></div>
<div class="row">
 <div class="panel" style="flex:1;min-width:300px">
  <div class="stat" id="stats"></div>
  <table id="pred"></table>
 </div>
 <div class="panel" style="flex:2;min-width:500px"><div id="comp" style="height:330px"></div></div>
</div>
<div class="panel"><div id="pos" style="height:380px"></div></div>
<div class="panel"><div id="evo" style="height:360px"></div></div>
<script>
const D = __DATA__;
const W = D.words, S = D.steps;
let cur = 0, timer = null;
document.getElementById("slider").max = S.length - 1;
const story = document.getElementById("story");
W.forEach((w, i) => { const sp = document.createElement("span"); sp.textContent = w; sp.onclick = () => go(Math.max(0, i - 1)); story.appendChild(sp); });
const esc = s => s.replace(/</g, "&lt;");
function toggle() {
  if (timer) { clearInterval(timer); timer = null; document.getElementById("play").textContent = "play"; return; }
  timer = setInterval(() => { if (cur >= S.length - 1) toggle(); else go(cur + 1); }, 350);
  document.getElementById("play").textContent = "pause";
}
document.addEventListener("keydown", e => { if (e.key === "ArrowRight") go(cur + 1); if (e.key === "ArrowLeft") go(cur - 1); });
function go(i) {
  cur = Math.min(Math.max(i, 0), S.length - 1);
  const st = S[cur];
  document.getElementById("slider").value = cur;
  document.getElementById("stepl").textContent = `step ${cur + 1}/${S.length}`;
  const emax = Math.max(...st.e);
  const epos = {}; st.pos.forEach((p, j) => epos[p] = j);
  [...story.children].forEach((sp, p) => {
    sp.className = ""; sp.style.background = ""; sp.title = "";
    if (p in epos) {
      const j = epos[p], a = st.e[j] / emax;
      sp.style.background = `rgba(214,96,77,${a.toFixed(3)})`;
      sp.title = `energy ${(st.e[j] * 100).toFixed(2)}%  k=${cur + 1 - p}  s=${st.s[j]}  lambda=${st.l[j]}  mention #${st.n[j]} delta=${st.d[j]}  w(k^lambda)=${st.f[j]}`;
    } else if (p <= cur) sp.className = "out";
    if (p === cur) sp.classList.add("cur");
    if (p === cur + 1) sp.classList.add("nxt");
  });
  // stats + predictions
  document.getElementById("stats").innerHTML =
    `next token: <b>${esc(st.tgt[0])}</b> (p = ${st.tgt[1].toFixed(3)})<br>effective tokens in state: <b>${st.neff}</b>` +
    ` of ${st.pos.length}${st.pad > 0 ? `<br>padding energy: ${(st.pad * 100).toFixed(1)}%` : ""}<br><br>top predictions:`;
  document.getElementById("pred").innerHTML = st.top.map(([t, p]) =>
    `<tr><td>${esc(t)}</td><td>${(p * 100).toFixed(1)}%</td><td><div style="background:#4393c3;height:10px;width:${Math.round(p * 200)}px"></div></td></tr>`).join("");
  // composition by token type
  const byType = {};
  st.pos.forEach((p, j) => { const w = W[p]; byType[w] = (byType[w] || { e: 0, c: 0 }); byType[w].e += st.e[j]; byType[w].c += 1; });
  const ord = Object.entries(byType).sort((a, b) => b[1].e - a[1].e).slice(0, 18).reverse();
  Plotly.react("comp", [{ type: "bar", orientation: "h", x: ord.map(o => o[1].e), y: ord.map(o => `${o[0]} x${o[1].c}`),
    marker: { color: "#d6604d" }, hovertemplate: "%{y}<br>%{x:.1%}<extra></extra>" }],
    { title: "State composition: energy per token type (all its mentions)", xaxis: { tickformat: ".0%" }, margin: { l: 120, t: 40 } });
  // energy per position vs distance
  const ks = st.pos.map(p => cur + 1 - p);
  Plotly.react("pos", [
    { type: "bar", x: ks, y: st.e, marker: { color: st.l.map(l => `rgba(${Math.round(60 + 160 * l)},${Math.round(120 - 60 * l)},${Math.round(200 - 150 * l)},0.85)`) },
      text: st.pos.map((p, j) => `${esc(W[p])}<br>s=${st.s[j]} lambda=${st.l[j]}<br>mention #${st.n[j]} delta=${st.d[j]}<br>w(k^lambda)=${st.f[j]}`),
      hovertemplate: "k=%{x}<br>%{text}<br>energy %{y:.2%}<extra></extra>" }],
    { title: "Energy per context position (color: blue = kept, lambda low; red = forgets fast, lambda high)",
      xaxis: { title: "distance k", autorange: "reversed", type: "log" }, yaxis: { title: "energy", tickformat: ".1%" }, margin: { t: 40 } });
  Plotly.relayout("evo", { shapes: [{ type: "line", x0: cur + 1, x1: cur + 1, yref: "paper", y0: 0, y1: 1, line: { color: "#000", dash: "dot" } }] });
}
Plotly.newPlot("evo", Object.entries(D.evo).map(([t, ys]) => ({ x: ys.map((_, i) => i + 1), y: ys, mode: "lines", name: t })),
  { title: "Energy of the main token types across the story (all mentions summed)", xaxis: { title: "step" }, yaxis: { tickformat: ".0%" }, margin: { t: 40 } });
go(0);
</script></body></html>"""
