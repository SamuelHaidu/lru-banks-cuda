"""Builds the HTML report on the lbc model's mechanisms from `lbc_study.py`'s results.json (self-contained: inline SVG,
no external assets).

    uv run python "ad hoc scripts/lbc_report.py" /tmp/study/results.json studies/lbc-mechanisms.html
"""

import html
import json
import math
import sys

R = json.load(open(sys.argv[1]))
OUT = sys.argv[2]

INK, MUTED, RULE, BG = "#1d2321", "#5e6863", "#d6dbd6", "#f7f8f6"
MEM, KNOW, FAIL, NEUTRAL = "#2e6f8e", "#8a6d1f", "#b0413e", "#9aa39e"

# measured with scripts/analyze_reads.py on 2 rows of the same snapshot (attention mass by slot half-life band)
READS = {
    "L0 hop 1": ([0.27, 0.28, 0.18, 0.12, 0.15], 2.62, 46), "L0 hop 2": ([0.30, 0.34, 0.16, 0.11, 0.09], 2.77, 33),
    "L1 hop 1": ([0.35, 0.19, 0.19, 0.14, 0.13], 2.66, 32), "L1 hop 2": ([0.31, 0.20, 0.23, 0.14, 0.12], 2.51, 36),
    "L2 hop 1": ([0.35, 0.22, 0.21, 0.10, 0.11], 2.43, 27), "L2 hop 2": ([0.29, 0.25, 0.24, 0.10, 0.12], 2.43, 33),
    "L3 hop 1": ([0.19, 0.26, 0.22, 0.15, 0.18], 2.44, 53), "L3 hop 2": ([0.15, 0.29, 0.22, 0.16, 0.18], 2.89, 55),
}
BANDS = ["1–4", "4–16", "16–64", "64–256", "256–1024"]


def esc(s):
    return html.escape(s, quote=True)


def tokshow(s):
    return esc(s.replace("\n", "↵").replace("\t", "⇥").replace("\r", ""))


def hbars(rows, width=640, label_w=190, unit="", vmax=None, series=None, legend=None, signed=True):
    """Horizontal bars: rows = [(label, [values...])], one bar per series value."""
    series = series or [MEM]
    n = len(series)
    bh, gap = 12, 10
    vmax = vmax or max(max(v) for _, v in rows) * 1.08
    h = len(rows) * (n * bh + gap) + 28
    plot = width - label_w - 60
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" class="chart">']
    y = 6
    for label, vals in rows:
        out.append(f'<text x="{label_w - 10}" y="{y + n * bh / 2 + 4}" class="lab" text-anchor="end">{esc(label)}</text>')
        for k, v in enumerate(vals):
            w = max(1.0, plot * v / vmax)
            out.append(f'<rect x="{label_w}" y="{y + k * bh}" width="{w:.1f}" height="{bh - 2}" fill="{series[k]}"/>')
            out.append(f'<text x="{label_w + w + 6}" y="{y + k * bh + bh - 3}" class="val">'
                       f'{(f"{v:+.2f}" if signed else f"{v:.2f}")}{unit}</text>' if v >= 0 else "")
        y += n * bh + gap
    if legend:
        x = label_w
        for color, name in legend:
            out.append(f'<rect x="{x}" y="{h - 14}" width="10" height="10" fill="{color}"/>'
                       f'<text x="{x + 15}" y="{h - 5}" class="lab">{esc(name)}</text>')
            x += 16 + 7.2 * len(name) + 20
    out.append("</svg>")
    return "".join(out)


def lines(xlabels, series, width=640, height=260, ymin=None, ymax=None, ylabel="", hline=None):
    """series = [(name, color, [y...], dashed)]"""
    left, right, top, bottom = 52, 120, 14, 36
    vals = [v for _, _, ys, _ in series for v in ys if v is not None] + ([hline[1]] if hline else [])
    ymin = min(vals) if ymin is None else ymin
    ymax = max(vals) * 1.05 if ymax is None else ymax
    pw, ph = width - left - right, height - top - bottom
    X = lambda i: left + pw * (i / max(1, len(xlabels) - 1))
    Y = lambda v: top + ph * (1 - (v - ymin) / (ymax - ymin))
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    for k in range(5):
        v = ymin + (ymax - ymin) * k / 4
        out.append(f'<line x1="{left}" x2="{left + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{left - 8}" y="{Y(v) + 4:.1f}" class="val" text-anchor="end">{v:.1f}</text>')
    step = max(1, len(xlabels) // 10)
    for i, xl in enumerate(xlabels):
        if i % step == 0 or i == len(xlabels) - 1:
            out.append(f'<text x="{X(i):.1f}" y="{top + ph + 18}" class="val" text-anchor="middle">{esc(str(xl))}</text>')
    if ylabel:
        out.append(f'<text x="14" y="{top + ph / 2}" class="lab" transform="rotate(-90 14 {top + ph / 2})" '
                   f'text-anchor="middle">{esc(ylabel)}</text>')
    if hline:
        out.append(f'<line x1="{left}" x2="{left + pw}" y1="{Y(hline[1]):.1f}" y2="{Y(hline[1]):.1f}" '
                   f'stroke="{NEUTRAL}" stroke-dasharray="4 4"/>'
                   f'<text x="{left + pw + 6}" y="{Y(hline[1]) + 4:.1f}" class="lab">{esc(hline[0])}</text>')
    ends = []
    for name, color, ys, dashed in series:
        pts = [(X(i), Y(v)) for i, v in enumerate(ys) if v is not None]
        ends.append(pts[-1][1])
    order = sorted(range(len(series)), key=lambda k: ends[k])
    label_y = {}
    last = -1e9
    for k in order:  # labels at the line ends, pushed apart by at least 15 px
        yy = max(ends[k], last + 15)
        label_y[k] = yy
        last = yy
    for si, (name, color, ys, dashed) in enumerate(series):
        pts = [(X(i), Y(v)) for i, v in enumerate(ys) if v is not None]
        d = " ".join(f"{'M' if j == 0 else 'L'}{x:.1f},{y:.1f}" for j, (x, y) in enumerate(pts))
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')
        for x, y in pts if len(pts) <= 12 else []:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        lx = pts[-1][0]
        out.append(f'<text x="{lx + 8:.1f}" y="{label_y[si] + 4:.1f}" class="lab" fill="{color}">{esc(name)}</text>')
    out.append("</svg>")
    return "".join(out)


def stacked(rows, width=640):
    label_w, bh, gap = 90, 16, 7
    shades = ["#d4e4ec", "#a3c4d4", "#6f9fb8", "#3f7896", "#1f4e66"]
    h = len(rows) * (bh + gap) + 34
    pw = width - label_w - 150
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" class="chart">']
    y = 4
    for label, (mass, ent, age) in rows.items():
        out.append(f'<text x="{label_w - 10}" y="{y + 12}" class="lab" text-anchor="end">{esc(label)}</text>')
        x = label_w
        for m, c in zip(mass, shades):
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{pw * m:.1f}" height="{bh}" fill="{c}"/>')
            x += pw * m
        out.append(f'<text x="{label_w + pw + 10}" y="{y + 12}" class="val">≈{math.exp(ent):.0f} slots · {age} tok back</text>')
        y += bh + gap
    x = label_w
    for band, c in zip(BANDS, shades):
        out.append(f'<rect x="{x}" y="{h - 16}" width="10" height="10" fill="{c}"/>'
                   f'<text x="{x + 14}" y="{h - 7}" class="lab">{band}</text>')
        x += 92
    out.append("</svg>")
    return "".join(out)


def grid(rowlabels, collabels, vals, lo=0.3, hi=0.95):
    out = ['<table class="grid"><tr><th></th>' + "".join(f"<th>{esc(c)}</th>" for c in collabels) + "</tr>"]
    for rl, row in zip(rowlabels, vals):
        cells = []
        for v in row:
            t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
            bg = f"rgba(46,111,142,{0.08 + 0.72 * t:.2f})"
            fg = "#fff" if t > 0.6 else INK
            cells.append(f'<td style="background:{bg};color:{fg}">{v:.2f}</td>')
        out.append(f"<tr><th>{esc(rl)}</th>" + "".join(cells) + "</tr>")
    return "".join(out) + "</table>"


def network_svg():
    a = R["ablations"]
    cost = lambda k: a[k]["all"]
    w, h = 860, 330
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" class="hero" aria-label="The network with the cost of silencing each part">']
    out.append(f'<text x="20" y="168" class="lab">token</text>'
               f'<rect x="64" y="146" width="74" height="34" rx="2" class="box"/>'
               f'<text x="101" y="167" class="lab" text-anchor="middle">embedding</text>')
    x0 = 160
    for n in range(4):
        x = x0 + n * 160
        out.append(f'<rect x="{x}" y="40" width="138" height="250" rx="3" class="layer"/>'
                   f'<text x="{x + 69}" y="30" class="lab" text-anchor="middle">layer {n}</text>')
        parts = [("LRU state", None, NEUTRAL, 62), ("write", None, MEM, 100),
                 ("hop-1 read", f"{n}:1", MEM, 138), ("FFN", f"{n}:ffn", KNOW, 176),
                 ("hop-2 read", f"{n}:2" if f"{n}:2" in a else None, MEM, 214)]
        for name, key, color, y in parts:
            out.append(f'<rect x="{x + 10}" y="{y}" width="118" height="28" rx="2" fill="#fff" stroke="{color}"/>'
                       f'<text x="{x + 18}" y="{y + 18}" class="lab">{name}</text>')
            if key and key in a:
                c = cost(key)
                out.append(f'<text x="{x + 122}" y="{y + 18}" class="val" text-anchor="end" fill="{color}">+{c:.2f}</text>')
        out.append(f'<text x="{x + 69}" y="268" class="val" text-anchor="middle">whole layer '
                   f'{"+%.2f" % a[f"{n}:all"]["all"] if f"{n}:all" in a else "(essential)"}</text>')
        if n < 3:
            out.append(f'<path d="M{x + 140},165 L{x + 158},165" stroke="{INK}" marker-end="url(#arr)"/>')
    out.append(f'<path d="M140,163 L{x0 - 2},163" stroke="{INK}" marker-end="url(#arr)"/>')
    xr = x0 + 4 * 160 - 18
    out.append(f'<path d="M{xr},165 L{xr + 16},165" stroke="{INK}" marker-end="url(#arr)"/>'
               f'<text x="{xr + 20}" y="160" class="lab">next</text><text x="{xr + 20}" y="174" class="lab">token</text>')
    out.append('<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" '
               f'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{INK}"/></marker></defs>')
    out.append(f'<text x="20" y="318" class="val">numbers: validation loss increase (nats/token) when the part is '
               f'silenced; intact loss {R["intact"]["loss"]:.2f}. The four hop-2 reads together: +'
               f'{a["0:2,1:2,2:2,3:2"]["all"]:.2f}</text></svg>')
    return "".join(out)


def sample_text():
    spans = []
    for s, r in R["writes"]["sample"]:
        t = max(0.0, min(1.0, (r - 0.2) / 0.8))
        bg = f"rgba(46,111,142,{0.06 + 0.5 * t:.2f})"
        dec = "line-through" if r < 0.35 else "none"
        show = esc(s).replace("\n", "<br>")
        spans.append(f'<span class="tk" style="background:{bg};text-decoration:{dec}" title="layer 0 writes: '
                     f'{r:.0%} of heads">{show}</span>')
    return "".join(spans)


a = R["ablations"]
intact = R["intact"]
dist = R["distance"]
ind = R["induction"]
writes = R["writes"]
CATS = ["word start", "continuation", "function", "punct", "newline", "digits"]

abl_rows = [(lab, [a[k]["repeat"], a[k]["new"]]) for lab, k in [
    ("L0 hop-1 read", "0:1"), ("L1 hop-1 read", "1:1"), ("L2 hop-1 read", "2:1"), ("L3 hop-1 read", "3:1"),
    ("all hop-2 reads", "0:2,1:2,2:2,3:2"), ("L1 FFN", "1:ffn"), ("L2 FFN", "2:ffn"), ("L3 FFN", "3:ffn")]]
ablation_chart = hbars(abl_rows, series=[MEM, KNOW], legend=[(MEM, "repeated tokens"), (KNOW, "new tokens")])

depth_chart = lines(["embedding + L0", "+ L1", "+ L2", "+ L3 (all)"],
                    [("loss", INK, [a["1:all,2:all,3:all"]["loss"], a["2:all,3:all"]["loss"], a["3:all"]["loss"],
                                    intact["loss"]], False)], ylabel="validation loss", ymin=2.0, height=230)

dist_labels = [f"{d['range'][0]}–{d['range'][1]}" for d in dist]
dist_chart = lines(dist_labels, [
    ("intact", MEM, [d["intact"] for d in dist], False),
    ("no hop-2 reads", KNOW, [d["0:2,1:2,2:2,3:2"] for d in dist], True),
    ("no L0 hop-1 read", FAIL, [d["0:1"] for d in dist], True)],
    ylabel="loss on the repeated token", ymin=0, hline=("new token", R["distance_new"]["intact"]))

ind_chart = lines(list(range(0, 100)), [
    ("first time", NEUTRAL, ind["first"][:99] + [None], False),
    ("repeated", MEM, ind["second"][:100], False)],
    ylabel="loss", ymin=0, ymax=16, hline=("uniform guess", ind["uniform"]))

freq_rows = [(f"{f['range'][0]:.0e}–{f['range'][1]:.0e}" if f['range'][0] else f"< {f['range'][1]:.0e}",
              [f["repeat"] or 0, f["new"] or 0]) for f in R["by_freq"]]
freq_chart = hbars(freq_rows, series=[MEM, FAIL], legend=[(MEM, "seen earlier in the document"), (FAIL, "new to it")],
                   unit="", signed=False)
pos_chart = lines([f"{p['range'][0]}–{p['range'][1]}" for p in R["by_position"]],
                  [("loss", INK, [p["loss"] for p in R["by_position"]], False)], ylabel="loss", ymin=2.0, height=220)

writes_grid = grid([c for c in CATS] + ["first occurrence", "repeat"], ["L0", "L1", "L2", "L3"],
                   [writes["by_cat"][c] for c in CATS] + [writes["first"], writes["repeat"]])

hard = "".join(f"<li><code>{tokshow(t)}</code> <span class='num'>{m:.1f}</span></li>" for t, m, c in R["hard_tokens"][:14])
easy = "".join(f"<li><code>{tokshow(t)}</code> <span class='num'>{m:.2f}</span></li>" for t, m, c in R["easy_tokens"][:10])
worst = "".join(
    f"<tr><td class='ctx'>…{esc(e['context'][-70:]).replace(chr(10), '↵')}</td><td><code>{tokshow(e['target'])}</code></td>"
    f"<td class='num'>{e['loss']:.1f}</td></tr>" for e in R["worst_examples"][:8])
missed = "".join(
    f"<tr><td class='ctx'>…{esc(e['context'][-64:]).replace(chr(10), '↵')}</td><td><code>{tokshow(e['target'])}</code></td>"
    f"<td class='num'>{e['repeat_distance']}</td><td class='num'>{e['loss']:.1f}</td></tr>" for e in R["missed_repeats"][:6])
hop2_hurt = ", ".join(f"<code>{tokshow(t)}</code>" for t, m, c in a["0:2,1:2,2:2,3:2"]["hurt"][:8])
l1_hurt = ", ".join(f"<code>{tokshow(t)}</code>" for t, m, c in a["1:1"]["hurt"][:6])
ffn2_hurt = ", ".join(f"<code>{tokshow(t)}</code>" for t, m, c in a["2:ffn"]["hurt"][:6])
ffn3_hurt = ", ".join(f"<code>{tokshow(t)}</code>" for t, m, c in a["3:ffn"]["hurt"][:6])

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>How lru-banks reads and writes: a mechanism study</title>
<style>
:root {{ --ink:{INK}; --muted:{MUTED}; --rule:{RULE}; --bg:{BG}; --mem:{MEM}; --know:{KNOW}; --fail:{FAIL}; }}
* {{ box-sizing:border-box; }}
html {{ background:var(--bg); color:var(--ink); }}
body {{ margin:0; font:17px/1.62 "Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif; }}
main {{ max-width:1000px; margin:0 auto; padding:56px 28px 96px; }}
.prose {{ max-width:68ch; }}
h1 {{ font-size:2.35rem; line-height:1.15; font-weight:600; letter-spacing:-0.01em; margin:0 0 14px; max-width:22ch; }}
h2 {{ font-size:1.45rem; font-weight:600; margin:64px 0 10px; padding-top:20px; border-top:1px solid var(--rule); }}
h2 .step {{ font:600 0.8rem/1 system-ui,sans-serif; color:var(--muted); display:block; margin-bottom:8px; }}
h3 {{ font-size:1.08rem; margin:28px 0 6px; }}
p {{ margin:0 0 14px; }}
.lede {{ font-size:1.18rem; color:#2f3734; max-width:62ch; }}
.meta {{ font:0.82rem/1.5 system-ui,sans-serif; color:var(--muted); margin:18px 0 34px; }}
figure {{ margin:22px 0 26px; }}
figcaption {{ font:0.84rem/1.5 system-ui,sans-serif; color:var(--muted); max-width:72ch; margin-top:6px; }}
svg.chart, svg.hero {{ width:100%; height:auto; display:block; font-family:system-ui,sans-serif; }}
svg .lab {{ font-size:12px; fill:var(--ink); }}
svg .val {{ font-size:11px; fill:var(--muted); }}
svg .grid {{ stroke:var(--rule); }}
svg .layer {{ fill:#fff; stroke:var(--rule); }}
svg .box {{ fill:#fff; stroke:var(--ink); }}
.key {{ display:inline-block; width:0.8em; height:0.8em; vertical-align:-0.05em; margin-right:4px; }}
.mem {{ color:var(--mem); }} .know {{ color:var(--know); }} .fail {{ color:var(--fail); }}
.finding {{ border-left:3px solid var(--mem); padding:2px 0 2px 16px; margin:18px 0; max-width:66ch; }}
.finding.k {{ border-color:var(--know); }} .finding.f {{ border-color:var(--fail); }}
.finding b {{ font-weight:600; }}
.lit {{ font:0.88rem/1.55 system-ui,sans-serif; color:#3b4440; background:#eef1ee; padding:10px 14px; margin:14px 0 18px;
       max-width:72ch; }}
.lit a {{ color:var(--mem); }}
code {{ font:0.86em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; background:#eceeeb; padding:0 3px; border-radius:2px; }}
table {{ border-collapse:collapse; font:0.86rem/1.4 system-ui,sans-serif; }}
table.grid th, table.grid td {{ padding:5px 10px; text-align:right; border-bottom:1px solid #fff; }}
table.grid th {{ font-weight:500; color:var(--muted); text-align:left; }}
table.ex {{ width:100%; margin:10px 0; }}
table.ex td, table.ex th {{ padding:6px 8px; border-bottom:1px solid var(--rule); vertical-align:top; text-align:left; }}
table.ex th {{ font-weight:500; color:var(--muted); }}
td.ctx {{ font:0.82rem ui-monospace,Menlo,monospace; color:#3b4440; word-break:break-word; }}
.num {{ font:0.84rem system-ui,sans-serif; color:var(--muted); font-variant-numeric:tabular-nums; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:28px; }}
.cols ul {{ list-style:none; padding:0; margin:6px 0; font:0.9rem/1.9 system-ui,sans-serif; }}
.sample {{ background:#fff; border:1px solid var(--rule); padding:16px 18px; line-height:2.0; font-size:0.95rem; max-width:76ch; }}
.tk {{ border-radius:2px; padding:1px 0; }}
.refs {{ font:0.84rem/1.6 system-ui,sans-serif; color:#3b4440; padding-left:18px; }}
.refs li {{ margin-bottom:6px; }}
a {{ color:var(--mem); }}
a:focus-visible {{ outline:2px solid var(--mem); outline-offset:2px; }}
@media (max-width:720px) {{ main {{ padding:32px 18px 64px; }} h1 {{ font-size:1.8rem; }} .cols {{ grid-template-columns:1fr; }} }}
</style></head><body><main>

<h1>How a slot-memory recurrent model reads, writes and fails</h1>
<p class="lede">A 32.8M-parameter lru-banks model trained on 0.5B tokens of text and code, taken apart: what it chooses to
store, what each read retrieves, what the second hop adds, where its knowledge sits, and the tokens it still gets wrong.</p>
<p class="meta">Model: 4 layers × 2 hops, width 512, tied 32,768 × 512 embedding, 8 bank heads × 64, 128 slots per layer,
trained write gates, reads held open. Snapshot at epoch {R['epoch']} (≈0.26 of a pass, lr 1e-3).
Measurements on {R['targets']:,} validation targets of pretrain-mix-bpe32768 unless noted; every “silenced” number is the
validation loss increase in nats per token when that part's output is removed from the stream.
<span class="key" style="background:{MEM}"></span>memory &nbsp;
<span class="key" style="background:{KNOW}"></span>weights (FFN) &nbsp;
<span class="key" style="background:{FAIL}"></span>failure</p>

<figure>{network_svg()}
<figcaption>The path of one token. Each layer runs an LRU over the stream; the state before the token forms the key it
writes into the banks, the state after it forms the hop-1 query. An FFN follows; a second query, formed after the FFN,
reads the same banks again. Numbers are the cost of silencing each part.</figcaption></figure>

<h2><span class="step">1 · Entering the network</span>The first FFN finishes the embedding</h2>
<div class="prose">
<p>Silencing layer 0's FFN raises the loss by <b>+{a['0:ffn']['all']:.2f}</b> nats, more than removing every memory read
of the network ({intact['loss']:.2f} → {a['0:ffn']['loss']:.2f}). Nothing downstream works without it. With an
unfactorized table each token enters as a raw 512-vector; the first FFN turns it into the representation the rest of the
network expects, and the hardest-hit tokens are word fragments (<code>inary</code>, <code>erman</code>) whose meaning
depends on the token itself rather than on context.</p>
</div>
<div class="lit">Transformers show the same division of labor: the earliest layers “detokenize”, building word-level
features out of sub-word tokens, and models are far less robust to deleting early (and final) layers than middle ones —
<a href="https://arxiv.org/abs/2406.19384">Lad, Lee, Gurnee &amp; Tegmark, <i>The Remarkable Robustness of LLMs: Stages of
Inference?</i></a></div>

<h2><span class="step">2 · Writing</span>The first layer stores what is new, and skips what is predictable</h2>
<div class="prose">
<p>Each token can write into the banks of every layer, per head, through a binary gate the model learned. Layer 0 writes
{writes['per_layer'][0]:.0%} of (token, head) pairs; layers 1–3 write {writes['per_layer'][1]:.0%},
{writes['per_layer'][2]:.0%} and {writes['per_layer'][3]:.0%}. What layer 0 keeps is not random:</p>
</div>
<figure>{writes_grid}
<figcaption>Fraction of heads that write, by the type of the input token and by whether it already appeared in the
document.</figcaption></figure>
<div class="finding"><b>Layer 0 indexes content.</b> Content-word starts are written by {writes['by_cat']['word start'][0]:.0%} of
heads, function words by {writes['by_cat']['function'][0]:.0%}, punctuation by {writes['by_cat']['punct'][0]:.0%}. A token
seen before in the document is written by {writes['repeat'][0]:.0%} of heads against {writes['first'][0]:.0%} the first
time: the memory already holds it. The tails of a rare word, once its first piece is in, are skipped too.</div>
<div class="finding"><b>The upper layers keep structure.</b> Layers 1–3 write newlines more than anything
({writes['by_cat']['newline'][1]:.0%}, {writes['by_cat']['newline'][2]:.0%}, {writes['by_cat']['newline'][3]:.0%}); in code a
newline marks a statement boundary, a position landmark for what follows.</div>
<figure><div class="sample">{sample_text()}</div>
<figcaption>A validation document; shading is the share of layer-0 heads that write each token (struck through: under
35%). Hover a token for its value.</figcaption></figure>
<div class="lit">Surprise as the trigger for memorizing is the design principle of
<a href="https://arxiv.org/abs/2501.00663">Titans (Behrouz, Zhong &amp; Mirrokni)</a>, whose neural memory updates in
proportion to how unexpected an input is. Here the same policy emerged from a binary gate trained only on next-token
loss.</div>

<h2><span class="step">3 · Reading</span>Hop 1 is the memory; it fades with distance</h2>
<div class="prose">
<p>The hop-1 read of layer 0 is the single most important memory path: silencing it costs
<b>+{a['0:1']['all']:.2f}</b>, on repeated and new tokens alike. Its cost on a token that already appeared grows with the
distance back to that appearance — and so does the intact model's loss: memory fades.</p>
</div>
<figure>{dist_chart}
<figcaption>Loss on a token that appeared earlier in the document, by how many tokens back. Dashed grey: a token new to
the document. At 257–512 tokens back, the model gains only {R['distance_new']['intact'] - dist[-1]['intact']:.1f} nats from
having seen it, against {R['distance_new']['intact'] - dist[0]['intact']:.1f} when it was within four
tokens.</figcaption></figure>
<div class="prose">
<p>Reads are broad. Each one spreads its attention over about a dozen slots (entropy ≈2.5 nats) and more than half of the
mass lands on slots with half-lives under 16 tokens: the banks mostly serve recent context, with a thinner channel to
the slow slots. Layer 3 reads the oldest content.</p>
</div>
<figure>{stacked(READS)}
<figcaption>Attention mass of each read over the slots' fixed half-lives (tokens), the effective number of slots attended
(e<sup>entropy</sup>) and the mean age of the content read. Float64 replay of two validation rows (matches the C++ loss to
3·10<sup>-4</sup>).</figcaption></figure>

<h2><span class="step">4 · The second hop</span>A different question, mostly about entities in context</h2>
<div class="prose">
<p>After the FFN, each layer forms a second query and reads the same banks again. The second query differs from the first
(cosine ≈0.3 per head), so the FFN does turn the stream into a new question. Together the four hop-2 reads are worth
<b>+{a['0:2,1:2,2:2,3:2']['all']:.2f}</b> nats; any single one is nearly redundant (+0.02 to +0.05), so they back each other
up. The cost is larger on repeated tokens ({a['0:2,1:2,2:2,3:2']['repeat']:+.2f}) than new ones
({a['0:2,1:2,2:2,3:2']['new']:+.2f}), and the tokens that suffer most belong to the document's own subject — its names, terms and
identifiers: {hop2_hurt}.</p>
</div>
<div class="finding"><b>What it is not, yet:</b> silencing shows hop 2 is needed for these tokens, not that it chains two
lookups (read A, which points to B). That would need the 2-hop recall task on this model, or tracing the slots hop 2
reads against what hop 1 returned.</div>

<h2><span class="step">5 · Deeper layers</span>Knowledge in the weights, syntax in the memory</h2>
<figure>{ablation_chart}
<figcaption>Loss increase when each part is silenced, on tokens repeated from earlier in the document (recoverable from
memory) and on tokens new to it (only from knowledge or grammar).</figcaption></figure>
<div class="finding k"><b>FFNs of layers 1–2 hold knowledge.</b> They help new tokens 2.6–4× more than repeated ones
(layer 2: {a['2:ffn']['new']:+.2f} vs {a['2:ffn']['repeat']:+.2f}). Their hardest-hit tokens are content that cannot be copied
from context: {ffn2_hurt}.</div>
<div class="finding"><b>Middle-layer reads track syntax.</b> Layer 1's hop-1 read matters most for closing brackets and
quotes — {l1_hurt} — tokens whose prediction requires remembering what was opened.</div>
<div class="finding k"><b>Layer 3 shapes the output.</b> Its FFN costs the same on repeated and new tokens and most on
function words: {ffn3_hurt}. Its hop-1 read is nearly unused (+{a['3:1']['all']:.2f}).</div>
<figure>{depth_chart}
<figcaption>Validation loss when the network stops after each layer (every later layer silenced). Each layer still pays
for itself.</figcaption></figure>
<div class="lit">FFN layers act as key–value memories whose keys match input patterns and whose values push the output
distribution — <a href="https://arxiv.org/abs/2012.14913">Geva, Schuster, Berant &amp; Levy, EMNLP 2021</a>. The
progression from detokenization through feature building to output sharpening is the depth picture of
<a href="https://arxiv.org/abs/2406.19384">Lad et al.</a></div>

<h2><span class="step">6 · Failures</span>Where the model struggles</h2>
<div class="prose">
<p><b>It does not yet copy arbitrary sequences.</b> A random 100-token sequence shown twice is predicted only
{ind['first_mean'] - ind['second_mean']:.1f} nats better the second time — {ind['second_mean']:.1f} nats, still close to a
uniform guess ({ind['uniform']:.1f}) — and the second copy does not improve as it goes. The memory retrieves a broad
mixture of recent context, not the exact successor of a remembered token. The architecture can copy (a model of this
family trained on copying reaches 99.99% at length 64), so this is a capacity the language model has not learned yet.</p>
</div>
<figure>{ind_chart}
<figcaption>Loss along a random 100-token sequence seen for the first time (grey) and repeated (blue); induction would
drive the blue line toward 0.</figcaption></figure>
<div class="lit">In transformers, induction heads — attention that finds the previous occurrence of the current token
and copies its successor — form in a sudden phase change early in training, and account for most in-context learning
— <a href="https://arxiv.org/abs/2209.11895">Olsson et al., <i>In-context Learning and Induction Heads</i></a>. Recurrent
and state-space models are known to lag behind attention on exact recall and copying:
<a href="https://arxiv.org/abs/2312.04927">Arora et al., <i>Zoology</i></a>;
<a href="https://arxiv.org/abs/2402.01032">Jelassi et al., <i>Repeat After Me</i></a>.</div>

<div class="prose">
<p><b>Rare tokens and new names cost the most.</b> Loss falls steeply with a token's frequency, and memory rescues
rare tokens only when they already appeared in the document:</p>
</div>
<figure>{freq_chart}
<figcaption>Loss by the target's frequency in the validation split, split by whether it appeared earlier in the
document.</figcaption></figure>
<div class="cols"><div><h3 class="fail">Hardest tokens</h3><ul>{hard}</ul></div>
<div><h3 class="mem">Easiest frequent tokens</h3><ul>{easy}</ul></div></div>

<h3>Worst single predictions</h3>
<table class="ex"><tr><th>context</th><th>target</th><th>loss</th></tr>{worst}</table>
<p class="prose">Most are rare continuations only knowledge or memorized phrases could supply: an English word after a
Chinese prompt, <code>Multi The</code>→<code>aft</code> (the game <i>Multi Theft Auto</i>), <code>push up</code>→<code>slope</code>
(“upslope”, split across tokens), <code>standing</code>→<code>toe</code> (“toe-to-toe”).</p>

<h3>Repeats it still misses</h3>
<div class="prose"><p>{R['missed_repeat_share']:.1%} of tokens that appeared within the previous 64 tokens are still predicted
badly (loss above 5). The token was available a few dozen positions back, so these are most likely retrieval failures
— it was not written, or the read did not find it — rather than missing knowledge.</p></div>
<table class="ex"><tr><th>context</th><th>target</th><th>seen back</th><th>loss</th></tr>{missed}</table>

<div class="prose">
<p><b>Context helps, then fades.</b> Loss drops from {R['by_position'][0]['loss']:.2f} in a document's first 8 tokens to
{R['by_position'][2]['loss']:.2f} at 32–128, then climbs back to {R['by_position'][4]['loss']:.2f} after 256 tokens —
where a transformer's loss keeps falling. Long documents differ from short ones, so part of this is the text, but it
matches the memory fading with distance above.</p>
</div>
<figure>{pos_chart}<figcaption>Loss by position in the document.</figcaption></figure>

<h2><span class="step">7 · Summary</span>What the model does, in one paragraph</h2>
<div class="prose">
<p>The first layer's FFN completes the embedding; its gates write new content words into the banks and skip predictable
tokens and repeats. One read — layer 0, hop 1 — carries most of the memory, retrieving a broad mixture of recent
context whose value fades with distance. Second hops ask a different question after the FFN and resolve references
back into the document, backing each other up. The middle layers' FFNs hold knowledge for tokens the context cannot
supply, the middle reads track open brackets and quotes, and the last layer shapes the output distribution. It has not
yet learned exact copying of unseen sequences, struggles with rare tokens and unannounced switches, and its use of
context peaks around a hundred tokens back.</p>
<p><b>Next measurements:</b> the same study at the end of training (does induction appear?); the 2-hop recall task on
this model; balancing the router and matching slot lifetimes to the distances the reads need.</p>
</div>

<h2><span class="step">Methods</span>How these numbers were measured</h2>
<div class="prose" style="font-size:0.95rem">
<p>Silencing: <code>lbc eval/dump --silence layer:1|2|ffn|all</code> removes a part's output from the residual stream at
inference; the trained model is unchanged. Zeroing pushes later layers off their training distribution, so costs are
upper bounds on a part's contribution. Per-token losses from <code>lbc dump</code> (64 rows × 512 tokens from document
starts); “repeated” means the target token appeared earlier in the same document. Reads: a float64 replay of the C++
model on two rows that records every read's attention over slots (<code>scripts/analyze_reads.py</code>). Induction:
documents made of a random sequence of 100 tokens drawn from the vocabulary, written twice. Scripts:
<code>ad hoc scripts/lbc_study.py</code>, <code>lbc_report.py</code>.</p>
</div>

<h2><span class="step">References</span>Literature</h2>
<ol class="refs">
<li>Lad, Lee, Gurnee, Tegmark. The Remarkable Robustness of LLMs: Stages of Inference? NeurIPS 2025.
<a href="https://arxiv.org/abs/2406.19384">arXiv:2406.19384</a></li>
<li>Geva, Schuster, Berant, Levy. Transformer Feed-Forward Layers Are Key-Value Memories. EMNLP 2021.
<a href="https://arxiv.org/abs/2012.14913">arXiv:2012.14913</a></li>
<li>Behrouz, Zhong, Mirrokni. Titans: Learning to Memorize at Test Time. 2024.
<a href="https://arxiv.org/abs/2501.00663">arXiv:2501.00663</a></li>
<li>Olsson et al. In-context Learning and Induction Heads. Transformer Circuits Thread, 2022.
<a href="https://arxiv.org/abs/2209.11895">arXiv:2209.11895</a></li>
<li>Arora et al. Zoology: Measuring and Improving Recall in Efficient Language Models. 2023.
<a href="https://arxiv.org/abs/2312.04927">arXiv:2312.04927</a></li>
<li>Jelassi, Brandfonbrener, Kakade, Malach. Repeat After Me: Transformers are Better than State Space Models at Copying.
ICML 2024. <a href="https://arxiv.org/abs/2402.01032">arXiv:2402.01032</a></li>
</ol>
</main></body></html>
"""
open(OUT, "w").write(page)
print("wrote", OUT, len(page) // 1024, "KB")
