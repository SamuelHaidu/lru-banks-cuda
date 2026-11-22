"""`model report compare-runs`: validation curves and rankings across all runs.

Reads every `runs/*/metrics.csv` (plus each run's `config.json` for arch,
dataset, steps per epoch and batch size) and writes one interactive HTML
report:

- validation curves of every run, grouped by dataset;
- a ranking table at a chosen epoch N (or each run's last epoch), by the
  value at that epoch or by the best value up to it.

Distillation runs (those with a `distill_config.json`) log the distillation
loss as `val_loss`, not cross-entropy, so they are flagged and hidden by
default.

No model is loaded, so it is cheap to rerun while training is in progress.
"""

ARCHS = ('no-run',)  # takes no run

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from model import config as cfg_mod
from model.reports._report import render_report

VAL_METRICS = ("val_loss", "val_sparse_categorical_accuracy")


def _read_metrics(path: Path) -> Dict[str, List[Optional[float]]]:
    """Returns `{column: [value per epoch]}`, 1 entry per completed epoch.

    If an epoch appears twice (e.g. a resume that rewrote a row), the last
    row wins.
    """
    rows: Dict[int, Dict[str, str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows[int(row["epoch"])] = row
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        return {}
    epochs = sorted(rows)
    out: Dict[str, List[Optional[float]]] = {}
    for metric in VAL_METRICS:
        values: List[Optional[float]] = []
        for e in epochs:
            try:
                v = float(rows[e].get(metric, ""))
                values.append(v if math.isfinite(v) else None)
            except (TypeError, ValueError):
                values.append(None)
        out[metric] = values
    return out


def collect_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    runs = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.csv")):
        run_dir = metrics_path.parent
        metrics = _read_metrics(metrics_path)
        if not metrics or not metrics["val_loss"]:
            continue
        try:
            config = json.loads((run_dir / "config.json").read_text())
        except (OSError, ValueError):
            config = {}
        data = config.get("data", {})
        runs.append(
            {
                "name": run_dir.name,
                "arch": config.get("arch", "?"),
                "dataset": data.get("dataset", "?"),
                "steps_per_epoch": data.get("steps_per_epoch"),
                "batch_size": data.get("batch_size"),
                "epochs": len(metrics["val_loss"]),
                "distill": (run_dir / "distill_config.json").exists(),
                "metrics": metrics,
            }
        )
    return runs


def run(runs_dir: Path, out_path: Path) -> Path:
    runs = collect_runs(runs_dir)
    if not runs:
        raise SystemExit(f"No runs with metrics.csv under {runs_dir}")
    datasets = sorted({r["dataset"] for r in runs})
    default_dataset = "tinystories-10k" if "tinystories-10k" in datasets else datasets[0]
    context = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "num_runs": len(runs),
        "runs_dir": str(runs_dir),
        "datasets": datasets,
        "default_dataset": default_dataset,
        "data": {"runs": runs, "metrics": list(VAL_METRICS)},
    }
    render_report("compare_runs.html", context, out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model report compare-runs",
        description="Interactive validation-metric comparison and ranking across all runs.",
    )
    parser.add_argument("--runs-dir", default=str(cfg_mod.RUNS_DIR), help="Directory holding the runs.")
    parser.add_argument(
        "--out",
        default=str(cfg_mod.RUNS_DIR / "compare_runs.html"),
        help="Output HTML path (default: runs/compare_runs.html).",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_path = run(Path(args.runs_dir), Path(args.out))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
