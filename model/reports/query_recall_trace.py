"""`model report query-recall-trace`: how does Block 1's attention do name recall?

The earlier studies established *what* the small and big tables encode. This
script looks at *how* the model gets there for a concrete behavior: recalling
a name that already appeared in context. For each prompt, a name is
introduced once, and the model's actual next-token prediction is that same
name reappearing (verified against the trained checkpoint, not assumed).

For each prompt this traces:
  1. Block 1's causal self-attention matrix (`AlibiCausalSelfAttention`, one
     head per row) -- specifically the attention distribution at the final
     context position, which is the position whose output feeds the
     retrieval query. High attention back to the name's occurrence is the
     mechanism's "looking up" the name in-context.
  2. The 128-D query vector extracted after Block 1 (the same query used for
     softmax retrieval against the big table), scored by cosine similarity
     against the big table -- shown as the target name's rank/similarity and
     the top-10 candidates the retrieval step actually favors.

The forward pass is replicated using the model's own sub-layers (`attn1`,
`trans1_mlp1/2`) exactly as `AlibiRetrieverSphereLM.call` does, so the traced
attention/query are byte-for-byte what the trained model computes -- this is
inspection, not a separate approximate model. Only `alibi-retriever-sphere`
(or any architecture with `attn1` + the same query-extraction shape) is
supported.
"""

ARCHS = ('alibi-retriever-sphere',)

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from model.device import setup_backend

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.reports import _spherical_ops as so  # noqa: E402
from model.reports._alibi_retriever_trace import (  # noqa: E402
    finish_forward as _finish_forward,
    retrieval_weights as _retrieval_weights,
    retrieve as _retrieve,
    trace_block1 as _trace_block1,
    vocab_mean_retrieved_tokens as _vocab_mean_retrieved_tokens,
)
from model.reports._model_io import load_full_model, load_tokenizer  # noqa: E402
from model.reports._report import plotly_div, render_report, token_labels  # noqa: E402

DEFAULT_PROMPTS: List[Tuple[str, str]] = [
    ("Emma", 'Once there was a kind girl named Emma. Her teacher said, "'),
    ("Tom", 'There was a little boy named Tom. His sister shouted, "'),
    ("Max", 'There was a girl named Max who loved to play. Her friend yelled, "'),
    ("Amy", 'Once upon a time, there was a girl named Amy. Her mom called, "'),
    ("Molly", 'Once upon a time, there was a girl named Molly. Her mom called, "'),
    ("Timmy", 'There was a little boy named Timmy. His sister shouted, "'),
    ("Lily", 'There was a little girl named Lily. Her brother shouted, "'),
    ("Sarah", 'There was a little girl named Sarah. Her brother shouted, "'),
    ("Mia", 'There was a girl named Mia who loved to play. Her friend yelled, "'),
    ("Jane", 'There was a little girl named Jane. Her brother shouted, "'),
    ("Sue", 'Once upon a time, there was a girl named Sue. Her mom called, "'),
    ("Daisy", 'Once upon a time, there was a girl named Daisy. Her mom called, "'),
    ("Mary", 'Once upon a time, there was a girl named Mary. Her mom called, "'),
    ("Lucy", 'Once upon a time, there was a girl named Lucy. Her mom called, "'),
    ("Jill", 'There was a little girl named Jill. Her brother shouted, "'),
    ("Anna", 'Once upon a time, there was a girl named Anna. Her mom called, "'),
    ("Emily", 'Once upon a time, there was a girl named Emily. Her mom called, "'),
    ("Jenny", 'Once upon a time, there was a girl named Jenny. Her mom called, "'),
    ("Maria", 'Once upon a time, there was a girl named Maria. Her mom called, "'),
    ("Kate", 'There was a girl named Kate who loved to play. Her friend yelled, "'),
]


def _attention_heatmap(attn_weights: Any, prompt_tokens: List[str], name_positions: List[int], head: int) -> Any:
    import plotly.graph_objects as go

    matrix = ops.convert_to_numpy(attn_weights[head])
    n = len(prompt_tokens)
    matrix = matrix[-n:, -n:]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=prompt_tokens,
            y=prompt_tokens,
            colorscale="Blues",
            colorbar=dict(title="attn"),
        )
    )
    for pos in name_positions:
        fig.add_shape(
            type="rect",
            x0=pos - 0.5,
            x1=pos + 0.5,
            y0=-0.5,
            y1=n - 0.5,
            line=dict(color="#b23b3b", width=1.5),
            fillcolor="rgba(0,0,0,0)",
        )
    fig.update_layout(
        title=f"Block 1 attention, head {head}",
        margin=dict(l=100, r=20, t=40, b=100),
        height=460,
        plot_bgcolor="white",
        xaxis=dict(tickangle=-60, tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
    )
    return fig


def _last_row_bar(attn_weights: Any, prompt_tokens: List[str], name_positions: List[int]) -> Any:
    import plotly.graph_objects as go

    n = len(prompt_tokens)
    num_heads = attn_weights.shape[0]
    fig = go.Figure()
    colors = ["#2f5fa3", "#2f7d4f", "#b23b3b", "#8a97a3"]
    for head in range(num_heads):
        row = ops.convert_to_numpy(attn_weights[head, -1, -n:])
        fig.add_trace(
            go.Bar(
                x=prompt_tokens,
                y=row,
                name=f"head {head}",
                marker_color=colors[head % len(colors)],
                opacity=0.8,
            )
        )
    for pos in name_positions:
        fig.add_vrect(x0=pos - 0.5, x1=pos + 0.5, fillcolor="#fbeaea", opacity=0.6, line_width=0)
    fig.update_layout(
        title="Attention from the final context position (feeds the retrieval query)",
        margin=dict(l=50, r=20, t=40, b=100),
        height=340,
        plot_bgcolor="white",
        barmode="group",
        xaxis=dict(tickangle=-60, tickfont=dict(size=10)),
        yaxis_title="attention weight",
    )
    return fig


def _query_similarity_bar(top_tokens: List[str], top_sims: List[float], target_name: str, target_sim: float, target_rank: int) -> Any:
    import plotly.graph_objects as go

    colors = ["#b23b3b" if t.strip() == target_name else "#2f5fa3" for t in top_tokens]
    fig = go.Figure(data=go.Bar(x=top_tokens, y=top_sims, marker_color=colors))
    fig.update_layout(
        title=f"Query vs. big table: top-10 by cosine similarity (target '{target_name}' at rank {target_rank}, sim {target_sim:.3f})",
        margin=dict(l=50, r=20, t=40, b=80),
        height=340,
        plot_bgcolor="white",
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis_title="cosine similarity",
    )
    return fig


def _encode_prompt(tokenizer: Any, prompt: str, window_size: int):
    import numpy as np

    ids = tokenizer.encode(prompt).ids
    x = np.zeros((1, window_size), dtype=np.int32)
    n = min(len(ids), window_size)
    x[0, -n:] = ids[-n:]
    prompt_tokens = [token_labels(tokenizer, [i])[0] for i in ids[-n:]]
    return ops.convert_to_tensor(x, dtype="int32"), prompt_tokens, ids[-n:]


def run(
    run_name: str,
    checkpoint: str = "latest",
    prompts: Optional[List[Tuple[str, str]]] = None,
    top_n: int = 10,
) -> Path:
    import torch

    model, config, ckpt_path = load_full_model(run_name, checkpoint)
    tokenizer = load_tokenizer(config)
    if tokenizer is None:
        raise ValueError(f"Run {run_name!r} has no tokenizer; cannot build/label prompts.")
    if not hasattr(model, "attn1"):
        raise ValueError(f"Architecture {config['arch']!r} has no `attn1` block; this trace only supports alibi-retriever-sphere.")

    prompts = prompts or DEFAULT_PROMPTS
    big_table = ops.normalize(model.target_embeddings, axis=-1, order=2)

    cases = []
    for name, prompt in prompts:
        with torch.no_grad():
            token_ids, prompt_tokens, ids = _encode_prompt(tokenizer, prompt, model.window_size)
            trace = _trace_block1(model, token_ids)
            attn_weights = trace["attn_weights"][0]
            query = trace["query"][0]

            sims = ops.matmul(ops.reshape(query, (1, -1)), ops.transpose(big_table))[0]
            _, top_idx = so.top_k(sims, k=top_n)
            top_idx_list = [int(i) for i in top_idx]
            top_tokens = token_labels(tokenizer, top_idx_list)
            top_sims = [float(sims[i]) for i in top_idx_list]

            name_ids = set(tokenizer.encode(" " + name).ids) | set(tokenizer.encode(name).ids)
            name_positions = [i for i, tid in enumerate(ids) if tid in name_ids]

            # The correct target id is whatever token the tokenizer actually
            # produces right after the prompt's own text -- e.g. after a
            # trailing `"` with no space, "Lily" tokenizes as the no-space
            # variant, not `tokenizer.encode(" " + name)`'s `Ġ`-prefixed one.
            prompt_ids = tokenizer.encode(prompt).ids
            continued_ids = tokenizer.encode(prompt + name).ids
            target_id = continued_ids[len(prompt_ids)] if len(continued_ids) > len(prompt_ids) else None
            target_sim = float(sims[target_id]) if target_id is not None else float("nan")
            full_order = ops.convert_to_numpy(ops.argsort(-sims))
            target_rank = int(list(full_order).index(target_id)) + 1 if target_id is not None else -1

            actual_logits = model(token_ids, training=False)[0]
            predicted_id = int(ops.argmax(actual_logits))
            predicted_token = token_labels(tokenizer, [predicted_id])[0]

            # Is the softmax retrieval step content-dependent, or has it
            # collapsed into a near-constant "average token" regardless of
            # the query? Compare the real retrieved vector against the one a
            # perfectly uniform softmax would produce, and re-run the rest of
            # the forward pass (Block 2 onward, exact trained weights) with
            # each, to see whether the final recall survives without it.
            x = ops.reshape(trace["x"], (1, model.window_size, model.dim_in))
            retrieval_weights, entropy_ratio = _retrieval_weights(model, query)
            entropy_ratio = float(entropy_ratio)
            _, top_weight_idx = so.top_k(retrieval_weights[0], k=top_n)
            top_weight_idx_list = [int(i) for i in top_weight_idx]
            top_weight_tokens = token_labels(tokenizer, top_weight_idx_list)
            top_weights = [float(retrieval_weights[0, i]) for i in top_weight_idx_list]
            uniform_weight = 1.0 / model.vocabulary_size

            real_retrieved = _retrieve(model, query)
            uniform_retrieved = _vocab_mean_retrieved_tokens(model)
            retrieval_invariance = float(
                ops.sum(ops.reshape(real_retrieved, (-1,)) * ops.reshape(uniform_retrieved, (-1,)))
                / (
                    ops.sqrt(ops.sum(ops.reshape(real_retrieved, (-1,)) ** 2))
                    * ops.sqrt(ops.sum(ops.reshape(uniform_retrieved, (-1,)) ** 2))
                )
            )

            logits_with_uniform_retrieval = _finish_forward(model, x, uniform_retrieved)[0]
            rank_with_real = target_rank
            if target_id is not None:
                order_real = ops.convert_to_numpy(ops.argsort(-actual_logits))
                order_uniform = ops.convert_to_numpy(ops.argsort(-logits_with_uniform_retrieval))
                rank_with_real = int(list(order_real).index(target_id)) + 1
                rank_with_uniform = int(list(order_uniform).index(target_id)) + 1
            else:
                rank_with_uniform = -1
            predicted_with_uniform = token_labels(
                tokenizer, [int(ops.argmax(logits_with_uniform_retrieval))]
            )[0]

        num_heads = attn_weights.shape[0]
        head_plots = [
            plotly_div(
                _attention_heatmap(attn_weights, prompt_tokens, name_positions, head),
                f"attn-heatmap-{len(cases)}-{head}",
                include_js=(len(cases) == 0 and head == 0),
            )
            for head in range(num_heads)
        ]
        last_row_plot = plotly_div(
            _last_row_bar(attn_weights, prompt_tokens, name_positions),
            f"attn-lastrow-{len(cases)}",
        )
        sim_plot = plotly_div(
            _query_similarity_bar(top_tokens, top_sims, name, target_sim, target_rank),
            f"query-sim-{len(cases)}",
        )

        cases.append(
            {
                "name": name,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "name_positions": name_positions,
                "predicted_token": predicted_token,
                "is_top1": predicted_token.strip() == name,
                "target_rank": target_rank,
                "target_sim": target_sim,
                "top_tokens": list(zip(top_tokens, top_sims)),
                "retrieval_invariance": retrieval_invariance,
                "entropy_ratio": entropy_ratio,
                "top_weight_tokens": list(zip(top_weight_tokens, top_weights)),
                "uniform_weight": uniform_weight,
                "rank_with_real_retrieval": rank_with_real,
                "rank_with_uniform_retrieval": rank_with_uniform,
                "predicted_with_uniform_retrieval": predicted_with_uniform,
                "plots": {"heads": head_plots, "last_row": last_row_plot, "similarity": sim_plot},
            }
        )

    hits = sum(1 for c in cases if c["is_top1"])
    mean_retrieval_invariance = sum(c["retrieval_invariance"] for c in cases) / len(cases)
    mean_entropy_ratio = sum(c["entropy_ratio"] for c in cases) / len(cases)
    unaffected_by_ablation = sum(
        1 for c in cases if c["predicted_with_uniform_retrieval"].strip() == c["name"]
    )

    # Least-uniform (most content-dependent) retrieval distributions first --
    # these are the prompts most likely to show a genuine, query-dependent
    # retrieval rather than the near-constant average-token vector.
    cases_by_uniformity = sorted(cases, key=lambda c: c["entropy_ratio"])

    context: Dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "checkpoint_path": str(ckpt_path),
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "num_cases": len(cases),
        "num_hits": hits,
        "num_heads": model.attn1.num_heads,
        "mean_retrieval_invariance": mean_retrieval_invariance,
        "mean_entropy_ratio": mean_entropy_ratio,
        "unaffected_by_ablation": unaffected_by_ablation,
        "cases": cases_by_uniformity,
    }

    out_path = cfg_mod.run_dir(run_name) / "reports" / "query_recall_trace.html"
    render_report("query_recall_trace.html", context, out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model report query-recall-trace",
        description="Trace Block 1's attention and the retrieval query for in-context name recall.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="latest", help="latest | best | <epoch number>")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N big-table candidates to show per prompt.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_path = run(args.run_name, checkpoint=args.checkpoint, top_n=args.top_n)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
