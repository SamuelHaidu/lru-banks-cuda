"""The training curves of a run as one self-contained HTML page (no network): the per-step values parsed from
train.log's progress lines (one redraw every ~0.5 s, the latest step's loss, lr, grad_norm, tok/s), the per-epoch
train / validation points of metrics.csv, the stages marked, and an exponential moving average (EMA) to smooth.

    python3 scripts/plot_loss.py RUN_DIR [--out RUN_DIR/loss.html]

RUN_DIR needs train.log; metrics.csv and config.json (data.epoch_tokens, for the token axis) are used when present.
"""

import argparse
import csv
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path)
a = ap.parse_args()
out = a.out or a.run / "loss.html"

epoch_tokens = 125_000_000
cfg = a.run / "config.json"
if cfg.exists():
    epoch_tokens = json.loads(cfg.read_text()).get("data", {}).get("epoch_tokens") or epoch_tokens

ansi = re.compile(r"\x1b\[[0-9;]*m")
step_re = re.compile(r"^\s*(\d+)/(\d+)\s")
num = r"([-+0-9.eE]+|nan|inf)"
fields = {k: re.compile(rf"(?:^| - ){re.escape(k)}: {num}") for k in ("loss", "bpc", "lr", "grad_norm")}
toks_re = re.compile(r"tok/s: ([0-9.]+)k")
stage_re = re.compile(r"^== stage (\S+): epochs (\d+)\.\.(\d+), context (\d+)")

points = {}  # (epoch, step) -> values; a resumed epoch overwrites the earlier attempt
stages = {}
epoch = None
text = (a.run / "train.log").read_text(errors="replace")
for raw in re.split(r"[\r\n]", text):
    line = ansi.sub("", raw)
    if m := stage_re.match(line):
        stages[m[1]] = {"name": m[1], "first": int(m[2]), "last": int(m[3]), "context": int(m[4])}
        continue
    if m := re.match(r"^Epoch (\d+)/\d+", line):
        epoch = int(m[1])
        continue
    if epoch is None or "loss:" not in line or not (m := step_re.match(line)):
        continue
    done, total = int(m[1]), int(m[2])
    if done >= total:  # the epoch's final line: averages, not one step
        continue
    p = {"x": (epoch - 1 + done / total) * epoch_tokens / 1e9, "epoch": epoch, "step": done}
    for k, r in fields.items():
        if mm := r.search(line):
            try:
                p[k] = float(mm[1])
            except ValueError:
                pass
    if mm := toks_re.search(line):
        p["toks"] = float(mm[1])
    points[(epoch, done)] = p
series = [points[k] for k in sorted(points)]

epochs = []
mpath = a.run / "metrics.csv"
if mpath.exists():
    for r in csv.DictReader(mpath.open()):
        e = int(r["epoch"]) + 1
        epochs.append({"x": e * epoch_tokens / 1e9, "epoch": e,
                       **{k: float(r[k]) for k in ("loss", "val_loss", "val_mix_loss", "bpc", "lr", "grad_norm")
                          if r.get(k) not in (None, "")}})

data = {"run": a.run.resolve().name, "epoch_tokens": epoch_tokens, "series": series, "epochs": epochs,
        "stages": [{**s, "x0": (s["first"] - 1) * epoch_tokens / 1e9, "x1": s["last"] * epoch_tokens / 1e9}
                   for s in stages.values()]}

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__RUN__ training curves</title>
<style>
  :root { --ink:#1f2328; --mute:#6e7781; --line:#d0d7de; --bg:#ffffff; --panel:#f6f8fa; --raw:#9db4d6; --ema:#0b5cad;
          --val:#b3541e; --mix:#7a4fb3; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
  header { display: flex; flex-wrap: wrap; gap: 18px; align-items: center; padding: 14px 20px; border-bottom: 1px solid var(--line); }
  h1 { font-size: 16px; font-weight: 600; margin: 0 12px 0 0; }
  label { display: inline-flex; gap: 6px; align-items: center; color: var(--mute); }
  select, input[type=range] { accent-color: var(--ema); }
  select { font: inherit; padding: 3px 6px; border: 1px solid var(--line); border-radius: 4px; background: var(--bg); color: var(--ink); }
  #ema-v { color: var(--ink); min-width: 3.2em; font-variant-numeric: tabular-nums; }
  main { padding: 12px 20px 20px; }
  #wrap { position: relative; height: calc(100vh - 140px); min-height: 360px; }
  canvas { width: 100%; height: 100%; display: block; }
  #tip { position: absolute; pointer-events: none; background: var(--bg); border: 1px solid var(--line); border-radius: 4px;
         padding: 6px 8px; font-size: 12px; font-variant-numeric: tabular-nums; display: none; white-space: nowrap; }
  #legend { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 8px; color: var(--mute); font-size: 13px; }
  #legend i { display: inline-block; width: 18px; height: 3px; vertical-align: middle; margin-right: 6px; }
  #legend .dot { width: 8px; height: 8px; border-radius: 50%; }
  :focus-visible { outline: 2px solid var(--ema); outline-offset: 2px; }
</style></head><body>
<header>
  <h1>__RUN__</h1>
  <label>Metric <select id="metric">
    <option value="loss">loss</option><option value="bpc">bits per char</option><option value="lr">learning rate</option>
    <option value="grad_norm">grad norm</option><option value="toks">tokens/s (k)</option></select></label>
  <label>EMA <input id="ema" type="range" min="0" max="0.999" step="0.001" value="0.98"><span id="ema-v"></span></label>
  <label><input id="restart" type="checkbox" checked> restart EMA at each stage</label>
  <label><input id="raw" type="checkbox" checked> raw</label>
  <label><input id="logy" type="checkbox"> log y</label>
  <label>From <select id="from"></select></label>
</header>
<main>
  <div id="wrap"><canvas id="c"></canvas><div id="tip"></div></div>
  <div id="legend"></div>
</main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const cv = $("c"), ctx = cv.getContext("2d"), tip = $("tip");
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fromSel = $("from");
fromSel.innerHTML = '<option value="0">start</option>' +
  D.stages.map(s => `<option value="${s.x0}">${s.name}</option>`).join("");

function stageOf(x) { let k = -1; D.stages.forEach((s, i) => { if (x >= s.x0 - 1e-9) k = i; }); return k; }

function emaSeries(pts, key, beta, restart) {
  // debiased EMA (as TensorBoard): ema_t = (beta m + (1-beta) v) / (1 - beta^n), restarted per stage if asked
  const out = []; let m = 0, n = 0, st = null;
  for (const p of pts) {
    const v = p[key]; if (v === undefined || !isFinite(v)) continue;
    const s = stageOf(p.x);
    if (restart && s !== st) { m = 0; n = 0; st = s; }
    m = beta * m + (1 - beta) * v; n++;
    out.push({x: p.x, y: beta > 0 ? m / (1 - Math.pow(beta, n)) : v, p});
  }
  return out;
}

let view = null;
function draw() {
  const key = $("metric").value, beta = +$("ema").value, logy = $("logy").checked, x0 = +fromSel.value;
  $("ema-v").textContent = beta.toFixed(3);
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const pts = D.series.filter(p => p.x >= x0 && p[key] !== undefined && isFinite(p[key]) && (!logy || p[key] > 0));
  const sm = emaSeries(pts, key, beta, $("restart").checked);
  const ep = D.epochs.filter(e => e.x >= x0);
  const epKeys = key === "loss" ? [["loss", css("--ema"), "train (epoch mean)"], ["val_loss", css("--val"), "val_loss"],
                                   ["val_mix_loss", css("--mix"), "val_mix_loss"]]
               : (key in (ep[0] || {}) ? [[key, css("--ema"), key + " (epoch mean)"]] : []);
  if (!pts.length) { ctx.fillStyle = css("--mute"); ctx.fillText("no data for this metric", 20, 30); return; }

  // y range: robust (the raw series' 1..97 percentiles and the epoch values), so the first steps' loss of ~10 does
  // not flatten the rest; lines are clipped to the plot
  const ys = pts.map(p => p[key]).sort((a, b) => a - b);
  let lo = ys[Math.floor(ys.length * 0.01)], hi = ys[Math.ceil(ys.length * 0.97) - 1];
  for (const [k] of epKeys) for (const e of ep) if (e[k] !== undefined && (!logy || e[k] > 0)) { lo = Math.min(lo, e[k]); hi = Math.max(hi, e[k]); }
  const f = v => logy ? Math.log10(v) : v, finv = v => logy ? Math.pow(10, v) : v;
  let fl = f(lo), fh = f(hi); const pad = (fh - fl) * 0.05 || 0.1; fl -= pad; fh += pad;
  const xmin = Math.min(x0, pts[0].x), xmax = Math.max(pts[pts.length - 1].x, ...ep.map(e => e.x));
  const L = 64, R = 16, T = 22, B = 36, pw = W - L - R, ph = H - T - B;
  const X = x => L + (x - xmin) / (xmax - xmin || 1) * pw, Y = v => T + (fh - f(v)) / (fh - fl) * ph;
  view = {X, Y, xmin, xmax, L, pw, sm, key, ep, epKeys};

  // stage bands and labels
  D.stages.forEach((s, i) => {
    const a = Math.max(X(s.x0), L), b = Math.min(X(s.x1), L + pw); if (b <= a) return;
    if (i % 2) { ctx.fillStyle = css("--panel"); ctx.fillRect(a, T, b - a, ph); }
    ctx.fillStyle = css("--mute"); ctx.font = "12px system-ui"; ctx.textAlign = "left";
    ctx.fillText(s.name, a + 4, T - 7);
  });
  // grid and axes
  ctx.strokeStyle = css("--line"); ctx.fillStyle = css("--mute"); ctx.lineWidth = 1; ctx.font = "12px system-ui";
  const yt = 6;
  for (let i = 0; i <= yt; i++) {
    const v = fl + (fh - fl) * i / yt, y = T + ph - ph * i / yt;
    ctx.beginPath(); ctx.moveTo(L, y + .5); ctx.lineTo(L + pw, y + .5); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(fmt(finv(v)), L - 8, y + 4);
  }
  const step = niceStep((xmax - xmin) / 8);
  ctx.textAlign = "center";
  for (let x = Math.ceil(xmin / step) * step; x <= xmax + 1e-9; x += step) {
    ctx.fillText(x.toFixed(step < 1 ? 1 : 0) + "B", X(x), T + ph + 18);
  }
  ctx.fillText("tokens seen", L + pw / 2, T + ph + 33);

  // raw, smoothed, epoch points
  ctx.save(); ctx.beginPath(); ctx.rect(L, T, pw, ph); ctx.clip();
  if ($("raw").checked) line(pts.map(p => [X(p.x), Y(p[key])]), css("--raw"), 1);
  line(sm.map(s => [X(s.x), Y(s.y)]), css("--ema"), 2);
  for (const [k, col] of epKeys) {
    if (k === key && key !== "loss") continue;
    ctx.fillStyle = col;
    for (const e of ep) if (e[k] !== undefined && (!logy || e[k] > 0)) {
      ctx.beginPath(); ctx.arc(X(e.x), Y(e[k]), 3.5, 0, 7); ctx.fill();
    }
  }
  ctx.restore();
  $("legend").innerHTML =
    ($("raw").checked ? `<span><i style="background:${css("--raw")}"></i>per step (latest step at each redraw)</span>` : "") +
    `<span><i style="background:${css("--ema")}"></i>EMA ${beta.toFixed(3)}</span>` +
    epKeys.filter(([k]) => !(k === key && key !== "loss"))
          .map(([k, c, n]) => `<span><i class="dot" style="background:${c}"></i>${n}</span>`).join("") +
    (key === "loss" ? `<span>val_loss starts rows at document starts (boilerplate), so it reads low</span>` : "");
}

function line(xy, color, w) {
  if (!xy.length) return;
  ctx.strokeStyle = color; ctx.lineWidth = w; ctx.lineJoin = "round"; ctx.beginPath();
  xy.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke();
}
function niceStep(r) { const p = Math.pow(10, Math.floor(Math.log10(r))); for (const m of [1, 2, 5, 10]) if (m * p >= r) return m * p; return 10 * p; }
function fmt(v) { const a = Math.abs(v); return a !== 0 && (a < 0.01 || a >= 1e4) ? v.toExponential(1) : (+v.toPrecision(3)).toString(); }

cv.addEventListener("mousemove", ev => {
  if (!view || !view.sm.length) return;
  const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left;
  const x = view.xmin + (mx - view.L) / view.pw * (view.xmax - view.xmin);
  let lo = 0, hi = view.sm.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (view.sm[m].x < x) lo = m + 1; else hi = m; }
  const s = view.sm[lo], p = s.p, st = D.stages[stageOf(p.x)];
  tip.innerHTML = `${p.x.toFixed(3)}B tokens · epoch ${p.epoch}, step ${p.step}${st ? " · " + st.name : ""}<br>` +
    `${view.key}: ${fmt(p[view.key])} · EMA ${fmt(s.y)}` + (p.lr !== undefined ? ` · lr ${p.lr.toExponential(2)}` : "");
  tip.style.display = "block";
  tip.style.left = Math.min(mx + 14, r.width - tip.offsetWidth - 4) + "px";
  tip.style.top = (ev.clientY - r.top + 14) + "px";
});
cv.addEventListener("mouseleave", () => { tip.style.display = "none"; });
for (const id of ["metric", "ema", "restart", "raw", "logy", "from"]) $(id).addEventListener("input", draw);
window.addEventListener("resize", draw);
draw();
</script></body></html>
"""
out.write_text(HTML.replace("__RUN__", data["run"]).replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print(f"{out}: {len(series)} step points, {len(epochs)} epochs, {len(stages)} stages")
