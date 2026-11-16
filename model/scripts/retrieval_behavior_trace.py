"""`model scripts retrieval-behavior-trace`: what does retrieval do on real data?

`query-recall-trace` used hand-crafted name-recall prompts and found the
softmax retrieval step had collapsed to near-uniform for every one of them.
This script checks whether that finding is specific to names/hand-crafted
prompts or general: it samples real `(context, target)` pairs directly from
the training corpus (`VariablePrefixDataset`'s own sampling scheme, so
lengths and content are exactly what the model was trained on), across three
context-length buckets (short/medium/long), and splits them into cases the
model gets exactly right (target is the top-1 prediction) and cases it gets
badly wrong (target is outside its own top-10 predictions) -- so both
"working" and "failing" retrieval can be inspected side by side, on
naturally occurring text rather than only names.

For each sampled case this reports the same diagnostics as
`query-recall-trace`: Block 1's attention from the final context position,
the retrieval softmax's entropy ratio and top candidates, and an ablation
comparing the target's rank with the real vs. a uniform-substituted
retrieved vector.
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from model.device import setup_backend

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.data import _document_bounds, _fill_windows, load_tokens  # noqa: E402
from model.scripts import _spherical_ops as so  # noqa: E402
from model.scripts._alibi_retriever_trace import (  # noqa: E402
    finish_forward,
    retrieval_weights,
    retrieve,
    trace_block1,
    vocab_mean_retrieved_tokens,
)
from model.scripts._model_io import load_full_model, load_tokenizer  # noqa: E402
from model.scripts._report import plotly_div, render_report, token_labels  # noqa: E402

LENGTH_BUCKETS: List[Tuple[str, int, int]] = [
    ("short", 1, 8),
    ("medium", 9, 32),
    ("long", 33, 128),
]


def _sample_targets(
    doc_starts: np.ndarray,
    doc_ends: np.ndarray,
    min_len: int,
    max_len: int,
    n: int,
    rng: np.random.Generator,
    max_tries: int = 20_000,
) -> np.ndarray:
    """Random target positions with context length in `[min_len, max_len]`.

    Mirrors `VariablePrefixDataset`'s own sampling (random document, random
    cut point) restricted to a length bucket, so short/medium/long prompts
    are still drawn from the model's actual training distribution rather
    than hand-built.
    """
    results: List[int] = []
    tries = 0
    while len(results) < n and tries < max_tries:
        tries += 1
        d = rng.integers(0, len(doc_starts))
        s, e = int(doc_starts[d]), int(doc_ends[d])
        if e - s < min_len + 1:
            continue
        lo = s + min_len
        hi = min(e, s + max_len + 1)
        if hi <= lo:
            continue
        results.append(int(rng.integers(lo, hi)))
    return np.array(results, dtype=np.int64)


def _attention_heatmap(attn_weights: Any, prompt_tokens: List[str]) -> Any:
    import plotly.graph_objects as go

    matrix = ops.convert_to_numpy(attn_weights[0])
    n = len(prompt_tokens)
    matrix = matrix[-n:, -n:]
    fig = go.Figure(
        data=go.Heatmap(z=matrix, x=prompt_tokens, y=prompt_tokens, colorscale="Blues", colorbar=dict(title="attn"))
    )
    fig.update_layout(
        title="Block 1 attention, head 0",
        margin=dict(l=100, r=20, t=40, b=100),
        height=420,
        plot_bgcolor="white",
        xaxis=dict(tickangle=-60, tickfont=dict(size=9)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
    )
    return fig


def _last_row_bar(attn_weights: Any, prompt_tokens: List[str]) -> Any:
    import plotly.graph_objects as go

    n = len(prompt_tokens)
    num_heads = attn_weights.shape[0]
    colors = ["#2f5fa3", "#2f7d4f", "#b23b3b", "#8a97a3"]
    fig = go.Figure()
    for head in range(num_heads):
        row = ops.convert_to_numpy(attn_weights[head, -1, -n:])
        fig.add_trace(go.Bar(x=prompt_tokens, y=row, name=f"head {head}", marker_color=colors[head % len(colors)], opacity=0.8))
    fig.update_layout(
        title="Attention from the final context position",
        margin=dict(l=50, r=20, t=40, b=100),
        height=320,
        plot_bgcolor="white",
        barmode="group",
        xaxis=dict(tickangle=-60, tickfont=dict(size=9)),
        yaxis_title="attention weight",
    )
    return fig


def _weight_bar(top_tokens: List[str], top_weights: List[float], uniform_weight: float, target_token: str) -> Any:
    import plotly.graph_objects as go

    colors = ["#b23b3b" if t.strip() == target_token.strip() else "#2f5fa3" for t in top_tokens]
    fig = go.Figure(data=go.Bar(x=top_tokens, y=top_weights, marker_color=colors))
    fig.add_hline(y=uniform_weight, line_dash="dash", line_color="#8a97a3", annotation_text="uniform")
    fig.update_layout(
        title="Retrieval softmax: top candidates by weight",
        margin=dict(l=50, r=20, t=40, b=80),
        height=320,
        plot_bgcolor="white",
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis_title="softmax weight",
    )
    return fig


def _trace_case(model: Any, tokenizer: Any, x_ids: np.ndarray, target_id: int, top_n: int) -> Dict[str, Any]:
    import torch

    token_ids = ops.convert_to_tensor(x_ids.reshape(1, -1), dtype="int32")
    nonzero = np.flatnonzero(x_ids)
    prompt_tokens = token_labels(tokenizer, [int(i) for i in x_ids[nonzero[0] :]]) if nonzero.size else []

    with torch.no_grad():
        trace = trace_block1(model, token_ids)
        attn_weights = trace["attn_weights"][0]
        query = trace["query"][0]
        x = ops.reshape(trace["x"], (1, model.window_size, model.dim_in))

        actual_logits = model(token_ids, training=False)[0]
        order_real = ops.convert_to_numpy(ops.argsort(-actual_logits))
        target_rank = int(list(order_real).index(target_id)) + 1
        predicted_token = token_labels(tokenizer, [int(order_real[0])])[0]
        top10_tokens = token_labels(tokenizer, [int(i) for i in order_real[:top_n]])

        weights, entropy_ratio = retrieval_weights(model, query)
        entropy_ratio = float(entropy_ratio)
        _, top_weight_idx = so.top_k(weights[0], k=top_n)
        top_weight_idx_list = [int(i) for i in top_weight_idx]
        top_weight_tokens = token_labels(tokenizer, top_weight_idx_list)
        top_weights = [float(weights[0, i]) for i in top_weight_idx_list]
        uniform_weight = 1.0 / model.vocabulary_size

        real_retrieved = retrieve(model, query)
        uniform_retrieved = vocab_mean_retrieved_tokens(model)
        retrieval_invariance = float(
            ops.sum(ops.reshape(real_retrieved, (-1,)) * ops.reshape(uniform_retrieved, (-1,)))
            / (
                ops.sqrt(ops.sum(ops.reshape(real_retrieved, (-1,)) ** 2))
                * ops.sqrt(ops.sum(ops.reshape(uniform_retrieved, (-1,)) ** 2))
            )
        )

        logits_uniform = finish_forward(model, x, uniform_retrieved)[0]
        order_uniform = ops.convert_to_numpy(ops.argsort(-logits_uniform))
        rank_with_uniform = int(list(order_uniform).index(target_id)) + 1
        predicted_with_uniform = token_labels(tokenizer, [int(order_uniform[0])])[0]

    target_token = token_labels(tokenizer, [target_id])[0]

    return {
        "prompt_tokens": prompt_tokens,
        "context_length": int(len(prompt_tokens)),
        "target_token": target_token,
        "predicted_token": predicted_token,
        "top10_tokens": top10_tokens,
        "target_rank": target_rank,
        "entropy_ratio": entropy_ratio,
        "retrieval_invariance": retrieval_invariance,
        "top_weight_tokens": list(zip(top_weight_tokens, top_weights)),
        "uniform_weight": uniform_weight,
        "rank_with_uniform_retrieval": rank_with_uniform,
        "predicted_with_uniform_retrieval": predicted_with_uniform,
        "attn_weights": attn_weights,
    }


def run(
    run_name: str,
    checkpoint: str = "latest",
    cases_per_bucket: int = 4,
    top_n: int = 10,
    seed: int = 0,
) -> Path:
    model, config, ckpt_path = load_full_model(run_name, checkpoint)
    tokenizer = load_tokenizer(config)
    if tokenizer is None:
        raise ValueError(f"Run {run_name!r} has no tokenizer.")
    if not hasattr(model, "attn1"):
        raise ValueError(f"Architecture {config['arch']!r} is not supported by this trace.")

    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
    path = str(cfg_mod.dataset_split_path(config["data"]["dataset"], "train"))
    tokens = load_tokens(path)
    doc_starts, doc_ends = _document_bounds(path, tokens, meta["doc_token_id"], True)

    rng = np.random.default_rng(seed)
    window_size = model.window_size

    hits: List[Dict[str, Any]] = []
    misses: List[Dict[str, Any]] = []

    for bucket_name, min_len, max_len in LENGTH_BUCKETS:
        found_hit = found_miss = 0
        needed = cases_per_bucket
        tries = 0
        while (found_hit < needed or found_miss < needed) and tries < needed * 400:
            tries += 1
            targets = _sample_targets(doc_starts, doc_ends, min_len, max_len, 1, rng)
            if targets.size == 0:
                continue
            t = targets[0]
            d = np.searchsorted(doc_starts, t, side="right") - 1
            s = doc_starts[d]
            x_ids, _ = _fill_windows(tokens, np.array([t]), np.array([s]), window_size)
            target_id = int(tokens[t])

            case = _trace_case(model, tokenizer, x_ids[0], target_id, top_n)
            case["bucket"] = bucket_name
            case["min_len"], case["max_len"] = min_len, max_len

            is_hit = case["predicted_token"].strip() == case["target_token"].strip()
            is_top10_miss = case["target_rank"] > top_n

            if is_hit and found_hit < needed:
                hits.append(case)
                found_hit += 1
            elif is_top10_miss and found_miss < needed:
                misses.append(case)
                found_miss += 1

    def _finalize(cases: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
        out = []
        for i, c in enumerate(cases):
            num_heads = c["attn_weights"].shape[0]
            heatmap = plotly_div(
                _attention_heatmap(c["attn_weights"], c["prompt_tokens"]),
                f"{prefix}-heat-{i}",
                include_js=(prefix == "hit" and i == 0),
            )
            last_row = plotly_div(_last_row_bar(c["attn_weights"], c["prompt_tokens"]), f"{prefix}-lastrow-{i}")
            weight_bar = plotly_div(
                _weight_bar(
                    [t for t, _ in c["top_weight_tokens"]],
                    [w for _, w in c["top_weight_tokens"]],
                    c["uniform_weight"],
                    c["target_token"],
                ),
                f"{prefix}-weights-{i}",
            )
            c = dict(c)
            c.pop("attn_weights")
            c["plots"] = {"heatmap": heatmap, "last_row": last_row, "weights": weight_bar}
            c["prompt_text"] = "".join(c["prompt_tokens"])
            out.append(c)
        return out

    hits = _finalize(hits, "hit")
    misses = _finalize(misses, "miss")
    all_cases = hits + misses

    mean_entropy_ratio = sum(c["entropy_ratio"] for c in all_cases) / len(all_cases) if all_cases else float("nan")
    mean_retrieval_invariance = (
        sum(c["retrieval_invariance"] for c in all_cases) / len(all_cases) if all_cases else float("nan")
    )

    context: Dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "checkpoint_path": str(ckpt_path),
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "num_hits": len(hits),
        "num_misses": len(misses),
        "cases_per_bucket": cases_per_bucket,
        "buckets": [b[0] for b in LENGTH_BUCKETS],
        "mean_entropy_ratio": mean_entropy_ratio,
        "mean_retrieval_invariance": mean_retrieval_invariance,
        "hits": hits,
        "misses": misses,
    }

    out_path = cfg_mod.run_dir(run_name) / "reports" / "retrieval_behavior_trace.html"
    render_report("retrieval_behavior_trace.html", context, out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model scripts retrieval-behavior-trace",
        description="Trace retrieval behavior on real training-data prompts, hits vs. top-10 misses, across context lengths.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="latest", help="latest | best | <epoch number>")
    parser.add_argument("--cases-per-bucket", type=int, default=4, help="Hit cases and miss cases per length bucket.")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N candidates shown; also the miss threshold.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_path = run(
        args.run_name,
        checkpoint=args.checkpoint,
        cases_per_bucket=args.cases_per_bucket,
        top_n=args.top_n,
        seed=args.seed,
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
