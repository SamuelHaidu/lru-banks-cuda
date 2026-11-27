"""Per-stage statistics of the per-step training loss parsed from train.log (the progress lines), to tell batch
sampling noise from optimization instability:

  - trend: a linear fit of the loss over the stage's steady (peak-lr) updates, per 1000 updates
  - noise: the std of the residual around a centered rolling mean, and that std x sqrt(rows per update): if the
    bouncing is only the batch's sampling noise, this product is about constant across stages
  - lag-1 / lag-5 autocorrelation of the residual: ~0 for independent batch noise; negative for an oscillation
    (overshooting back and forth); positive for slow excursions (a spike that takes many updates to recover)
  - spikes: residuals beyond 4 sigma, and the largest one
  - grad_norm: mean, coefficient of variation, and its correlation with the loss residual

    python3 scripts/loss_stats.py RUN_DIR
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
cfg = json.loads((run / "config.json").read_text())
stage_cfg = {s["name"]: s for s in cfg["training"].get("stages", [])}

ansi = re.compile(r"\x1b\[[0-9;]*m")
stage_re = re.compile(r"^== stage (\S+): epochs (\d+)\.\.(\d+)")
step_re = re.compile(r"^\s*(\d+)/(\d+)\s")
loss_re = re.compile(r" - loss: ([-+0-9.eE]+)")
gn_re = re.compile(r" - grad_norm: ([-+0-9.eE]+)")
lr_re = re.compile(r" - lr: ([-+0-9.eE]+)")

stages, pts, epoch = [], {}, None
for raw in re.split(r"[\r\n]", (run / "train.log").read_text(errors="replace")):
    line = ansi.sub("", raw)
    if m := stage_re.match(line):
        stages.append((m[1], int(m[2]), int(m[3])))
        continue
    if m := re.match(r"^Epoch (\d+)/", line):
        epoch = int(m[1])
        continue
    if epoch is None or not (m := step_re.match(line)) or int(m[1]) >= int(m[2]):
        continue
    if (lm := loss_re.search(line)) and (gm := gn_re.search(line)) and (rm := lr_re.search(line)):
        pts[(epoch, int(m[1]))] = (int(m[2]), float(lm[1]), float(gm[1]), float(rm[1]))

seen = {}
for name, first, last in stages:  # a resumed stage prints its line again: keep the last
    seen[name] = (first, last)


def acf(x, k):
    return float(np.corrcoef(x[:-k], x[k:])[0, 1])


hdr = (f"{'stage':8s} {'rows':>5s} {'n':>6s} {'mean':>7s} {'trend/1k':>9s} {'noise':>7s} {'x√rows':>7s} "
       f"{'lag1':>6s} {'lag5':>6s} {'>4σ':>5s} {'max σ':>6s} {'gn':>6s} {'gn cv':>6s} {'r(gn,res)':>9s}")
print("steady part only: lr at the stage's peak (after warmup, before decay)\n" + hdr)
for name, (first, last) in seen.items():
    s = stage_cfg.get(name, {}).get("set", {})
    rows, peak = s.get("data.batch_size"), s.get("training.learning_rate")
    sel = sorted((e, k, v) for (e, k), v in pts.items() if first <= e <= last)
    if not sel:
        continue
    spe = sel[0][2][0]
    upd = np.array([(e - first) * spe + k for e, k, _ in sel], float)
    loss = np.array([v[1] for *_, v in sel])
    gn = np.array([v[2] for *_, v in sel])
    lr = np.array([v[3] for *_, v in sel])
    steady = np.abs(lr - peak) / peak < 0.01 if peak else np.ones_like(lr, bool)
    upd, loss, gn = upd[steady], loss[steady], gn[steady]
    if len(loss) < 300:
        print(f"{name:8s} {rows:5d} {len(loss):6d}  (too few steady points)")
        continue
    w = 101
    trend_roll = np.convolve(loss, np.ones(w) / w, mode="valid")
    res = loss[w // 2: -(w // 2)] - trend_roll
    gn_c = gn[w // 2: -(w // 2)]
    slope = np.polyfit(upd, loss, 1)[0] * 1000
    sd = res.std()
    z = res / sd
    print(f"{name:8s} {rows:5d} {len(loss):6d} {loss.mean():7.4f} {slope:+9.4f} {sd:7.4f} {sd * np.sqrt(rows):7.3f} "
          f"{acf(res, 1):+6.2f} {acf(res, 5):+6.2f} {(np.abs(z) > 4).mean():5.1%} {np.abs(z).max():6.1f} "
          f"{gn.mean():6.3f} {gn.std() / gn.mean():6.2f} {np.corrcoef(gn_c, res)[0, 1]:+9.2f}")
