"""`model report name-swap-trace`: does the prediction track a swapped name, or ignore it?

`order-sensitivity-loss` showed the model depends heavily on sequence order,
which could mean either (a) content-sensitive copying/routing -- the model
finds the *specific* earlier name and routes its identity forward -- or
(b) pure structural inference -- the model uses the sequence's shape/pattern
(a name occupies this slot, this narrative structure) while being largely
indifferent to *which* name is actually there. Swapping only the earlier
name's identity, holding the prompt template, position, and length fixed,
distinguishes them directly: if the top prediction tracks the swap (predicts
the new name), that is copying/routing; if it stays fixed regardless of
which name was substituted, that is template-level structural inference.

Reuses `query_recall_trace.DEFAULT_PROMPTS`' template family (each of the
form `... named <NAME>. ... said/shouted/yelled, "`), swapping in every
other name from the same pool at the same slot.
"""

ARCHS = ('any',)  # works on every architecture

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from model.device import setup_backend

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.reports._model_io import load_full_model, load_tokenizer  # noqa: E402
from model.reports._report import render_report, token_labels  # noqa: E402
from model.reports.query_recall_trace import DEFAULT_PROMPTS  # noqa: E402

NAME_POOL: List[str] = sorted({name for name, _ in DEFAULT_PROMPTS})


def _encode(tokenizer: Any, prompt: str, window_size: int) -> np.ndarray:
    ids = tokenizer.encode(prompt).ids
    x = np.zeros((1, window_size), dtype=np.int32)
    n = min(len(ids), window_size)
    x[0, -n:] = ids[-n:]
    return x


def _predict_top1(model: Any, tokenizer: Any, prompt: str) -> str:
    import torch

    x = _encode(tokenizer, prompt, model.window_size)
    with torch.no_grad():
        logits = model(ops.convert_to_tensor(x, dtype="int32"), training=False)[0]
    predicted_id = int(ops.argmax(logits))
    return token_labels(tokenizer, [predicted_id])[0].strip()


def run(run_name: str, checkpoint: str = "latest") -> Path:
    model, config, ckpt_path = load_full_model(run_name, checkpoint)
    tokenizer = load_tokenizer(config)
    if tokenizer is None:
        raise ValueError(f"Run {run_name!r} has no tokenizer.")

    rows = []
    for original_name, template_prompt in DEFAULT_PROMPTS:
        if original_name not in template_prompt:
            continue
        variants = []
        for swap_name in NAME_POOL:
            prompt = template_prompt.replace(original_name, swap_name)
            predicted = _predict_top1(model, tokenizer, prompt)
            variants.append(
                {
                    "swap_name": swap_name,
                    "prompt": prompt,
                    "predicted": predicted,
                    "tracks_swap": predicted == swap_name,
                    "is_original": swap_name == original_name,
                }
            )
        rows.append({"original_name": original_name, "variants": variants})

    total = sum(len(r["variants"]) for r in rows)
    tracked = sum(1 for r in rows for v in r["variants"] if v["tracks_swap"])
    unchanged_from_original = sum(
        1
        for r in rows
        for v in r["variants"]
        if not v["is_original"] and v["predicted"] == r["original_name"]
    )

    context: Dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "checkpoint_path": str(ckpt_path),
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "num_templates": len(rows),
        "num_names": len(NAME_POOL),
        "total_variants": total,
        "tracked": tracked,
        "tracked_pct": tracked / total * 100.0 if total else 0.0,
        "unchanged_from_original": unchanged_from_original,
        "rows": rows,
    }

    out_path = cfg_mod.run_dir(run_name) / "reports" / "name_swap_trace.html"
    render_report("name_swap_trace.html", context, out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model report name-swap-trace",
        description="Swap the earlier name's identity and check whether the prediction tracks it (copying) or stays fixed (structural inference).",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="latest", help="latest | best | <epoch number>")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_path = run(args.run_name, checkpoint=args.checkpoint)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
