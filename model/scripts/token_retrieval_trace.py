"""`model scripts token-retrieval-trace`: what does token-anchored retrieval actually pull?

`query-recall-trace` (`model/scripts/query_recall_trace.py`) answers this
question for `alibi-retriever-sphere`'s free-query mechanism, where the
query is compared against the entire vocabulary. That trace does not apply
to `alibi-token-retriever-sphere`: there, the query is compared only against
the tokens already present in the context window, and the resulting
per-position softmax weights are scattered into a vocabulary-sized bucket by
token id before pooling the big table. "What is it retrieving" for this
architecture means: which in-context positions does the query favor, and
which token ids (of the ones actually present) end up dominating the pooled
vector.

For each prompt this traces:
  1. Block 1's causal self-attention matrix, same as `query-recall-trace`.
  2. The in-context similarity scores: the dim_in-space query vs. each of the
     `window_size` transformed tokens in `x` (not the vocab table) -- the
     softmax weight this architecture actually assigns to each context
     position.
  3. The token-id scatter: those per-position weights rolled up into buckets
     by token id, i.e. the real "what got retrieved" distribution -- since
     only ids present in context can receive nonzero weight, this table is
     at most `window_size` rows wide, unlike the full-vocabulary ranking in
     `query-recall-trace`.

The forward pass is replicated using the model's own trained sub-layers
exactly as `AlibiTokenRetrieverSphereLM.call` does (`_alibi_token_retriever_trace.py`),
so the traced attention/weights are byte-for-byte what the trained model
computes. Only `alibi-token-retriever-sphere` (or any architecture exposing
`query_proj`) is supported.
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from model.device import setup_backend

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.scripts import _spherical_ops as so  # noqa: E402
from model.scripts._alibi_token_retriever_trace import (  # noqa: E402
    finish_forward as _finish_forward,
    retrieve_from_vocab_weights as _retrieve_from_vocab_weights,
    trace_block1 as _trace_block1,
    vocab_mean_retrieved_tokens as _vocab_mean_retrieved_tokens,
)
from model.scripts._model_io import load_full_model, load_tokenizer  # noqa: E402
from model.scripts._report import plotly_div, render_report, token_labels  # noqa: E402

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

# Non-name recall: a common noun (object/animal/place) introduced once, then
# referenced again later in the same window -- tests whether token-anchored
# retrieval generalizes beyond proper names to ordinary in-context nouns.
NOUN_PROMPTS: List[Tuple[str, str]] = [
    ("ball", "Tom found a red ball in the yard. He picked up the"),
    ("dog", "There was a small dog in the park. Everyone loved the"),
    ("cat", "She saw a fluffy cat on the fence. She wanted to pet the"),
    ("cake", "Mom baked a big cake for the party. Everyone wanted a slice of the"),
    ("kite", "He flew a colorful kite in the sky. The wind carried the"),
    ("box", "They found an old box in the attic. Inside the"),
    ("garden", "Grandma had a beautiful garden full of flowers. She loved working in the"),
    ("boat", "The fisherman had a small boat by the lake. He rowed the"),
    ("tree", "There was a tall tree in the backyard. Birds nested in the"),
    ("book", "She picked up a book from the shelf. She opened the"),
    ("car", "Dad parked the car in the driveway. He washed the"),
    ("bird", "A little bird landed on the window. She watched the"),
    ("bike", "He got a new bike for his birthday. He rode the"),
    ("hat", "She wore a bright yellow hat to school. Everyone noticed the"),
    ("apple", "He picked a shiny red apple from the tree. He bit into the"),
    ("river", "They walked along the river near the village. Fish swam in the"),
    ("castle", "In the story, there was an old castle on the hill. Knights guarded the"),
    ("balloon", "She held a big blue balloon at the fair. The wind pulled the"),
    ("cookie", "Grandpa gave her a warm cookie from the oven. She took a bite of the"),
    ("umbrella", "It started to rain, so she grabbed her umbrella. She opened the"),
]

# Long-context recall: the target noun is introduced once near the start,
# then buried under ~120 tokens of distractor sentences full of other
# concrete nouns before being asked for again. Left-padding to window_size
# only kicks in for short prompts (see _encode_prompt) -- these are written
# long enough to fill the full 128-token window with real content, so
# retrieval cannot trivially win just because the target is the only salient
# token near the padding boundary. It has to actually pick the right id out
# of many plausible candidates spread across the whole window.
LONG_CONTEXT_PROMPTS: List[Tuple[str, str]] = [
    (
        "kite",
        "Once upon a time, a boy named Sam found a red kite in his backyard. "
        "It was a sunny day, and the sky was full of white clouds. Sam's "
        "sister brought a ball, a jump rope, and a bucket of chalk to the "
        "yard. Their dog ran around chasing a butterfly near the fence. "
        "Their mom was planting flowers in the garden while their dad "
        "fixed an old bicycle by the shed. A cat watched from the porch, "
        "and birds sang in the big oak tree above the house. Sam's friend "
        "Ben came over carrying a basketball and a skateboard. They all "
        "played in the yard until it got windy. Sam looked up and smiled, "
        "then ran to grab his",
    ),
    (
        "necklace",
        "Grandma kept an old wooden box on her dresser filled with special "
        "things. Inside were a silver spoon, a small mirror, a pair of "
        "gloves, and a shiny necklace her mother once gave her. There was "
        "also a photograph of a farm, a dried flower, and a tiny music "
        "box that played a soft tune. Grandma loved to sit by the window "
        "with her cat and tell stories about her childhood on the farm "
        "with chickens, a horse, and a big red barn. One afternoon, her "
        "granddaughter Lily visited and asked to see the box again. "
        "Grandma smiled and carefully lifted out the",
    ),
    (
        "lantern",
        "The campers set up their tent near a quiet lake surrounded by "
        "tall pine trees. They packed a tent, sleeping bags, a cooler, "
        "fishing rods, and an old lantern for the night. During the day "
        "they saw a deer, a rabbit, and several birds near the water. "
        "One of the campers, Jake, cooked soup over a small fire while "
        "his friends played cards at a wooden table. As the sun went "
        "down, the sky turned orange and the crickets started singing. "
        "When it got dark, Jake reached into the bag and pulled out the",
    ),
    (
        "trophy",
        "The school held a big competition with painting, singing, and "
        "running races for every class. Students brought brushes, drums, "
        "sneakers, and colorful posters to show their projects. Emma's "
        "class built a small volcano, a birdhouse, and a paper rocket for "
        "the science fair. After the judges walked around with clipboards "
        "and pencils, they announced the winners near the big stage. "
        "Emma's team had worked for weeks on their rocket design. When "
        "her name was called, she walked up and proudly held the",
    ),
    (
        "letter",
        "Every summer, Mia's family visited her uncle's house near the "
        "coast. Her uncle kept fishing nets, an old radio, seashells, and "
        "a stack of maps in his study. Mia would explore the beach with "
        "a bucket, sunscreen, a towel, and her favorite storybook. One "
        "rainy afternoon, her uncle handed her an envelope with a "
        "handwritten letter inside from a friend overseas. Mia sat by "
        "the window with her tea and a blanket. She carefully opened the "
        "envelope and began reading the",
    ),
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
        title="Attention from the final context position (feeds Block 1's output at every position)",
        margin=dict(l=50, r=20, t=40, b=100),
        height=340,
        plot_bgcolor="white",
        barmode="group",
        xaxis=dict(tickangle=-60, tickfont=dict(size=10)),
        yaxis_title="attention weight",
    )
    return fig


def _context_weight_bar(token_weights: Any, prompt_tokens: List[str], name_positions: List[int]) -> Any:
    """Per-context-position softmax weight the query assigns (in-context similarity)."""
    import plotly.graph_objects as go

    n = len(prompt_tokens)
    weights = ops.convert_to_numpy(token_weights[-n:])
    fig = go.Figure(data=go.Bar(x=prompt_tokens, y=weights, marker_color="#2f5fa3"))
    for pos in name_positions:
        fig.add_vrect(x0=pos - 0.5, x1=pos + 0.5, fillcolor="#fbeaea", opacity=0.6, line_width=0)
    fig.update_layout(
        title="Query vs. in-context tokens: per-position softmax weight",
        margin=dict(l=50, r=20, t=40, b=100),
        height=340,
        plot_bgcolor="white",
        xaxis=dict(tickangle=-60, tickfont=dict(size=10)),
        yaxis_title="softmax weight",
    )
    return fig


def _bucket_bar(top_tokens: List[str], top_weights: List[float], target_name: str) -> Any:
    import plotly.graph_objects as go

    colors = ["#b23b3b" if t.strip() == target_name else "#2f5fa3" for t in top_tokens]
    fig = go.Figure(data=go.Bar(x=top_tokens, y=top_weights, marker_color=colors))
    fig.update_layout(
        title="Token-id scatter: retrieval weight by vocabulary id (only ids present in context are nonzero)",
        margin=dict(l=50, r=20, t=40, b=80),
        height=340,
        plot_bgcolor="white",
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis_title="scattered weight",
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
    prompt_set_name: str = "names",
) -> Path:
    import torch

    model, config, ckpt_path = load_full_model(run_name, checkpoint)
    tokenizer = load_tokenizer(config)
    if tokenizer is None:
        raise ValueError(f"Run {run_name!r} has no tokenizer; cannot build/label prompts.")
    if not hasattr(model, "query_proj"):
        raise ValueError(
            f"Architecture {config['arch']!r} has no `query_proj`; this trace only supports "
            "alibi-token-retriever-sphere."
        )

    prompts = prompts or DEFAULT_PROMPTS

    cases = []
    for name, prompt in prompts:
        with torch.no_grad():
            token_ids, prompt_tokens, ids = _encode_prompt(tokenizer, prompt, model.window_size)
            n = len(ids)

            trace = _trace_block1(model, token_ids)
            token_weights = trace["token_weights"][0]
            vocab_weights = trace["vocab_weights"]
            x = trace["x"]

            name_ids = set(tokenizer.encode(" " + name).ids) | set(tokenizer.encode(name).ids)
            name_positions = [i for i, tid in enumerate(ids) if tid in name_ids]

            # Attention still needs to be recomputed separately since
            # `trace_block1` only returns `x`/query/weights, not the raw
            # attention matrix -- mirror `attn1` exactly to get it.
            h = ops.normalize(ops.reshape(
                ops.take(ops.normalize(model.input_embeddings, axis=-1, order=2), ops.reshape(token_ids, (-1,)), axis=0),
                (1, model.window_size, model.dim_in),
            ), axis=-1, order=2)
            seq_len = ops.shape(h)[1]
            attn1 = model.attn1
            q = attn1.q_proj(h)
            k = attn1.k_proj(h)
            q = ops.transpose(ops.reshape(q, (1, seq_len, attn1.num_heads, attn1.head_dim)), (0, 2, 1, 3))
            k = ops.transpose(ops.reshape(k, (1, seq_len, attn1.num_heads, attn1.head_dim)), (0, 2, 1, 3))
            scale = 1.0 / (float(attn1.head_dim) ** 0.5)
            scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale
            alibi, mask = attn1._get_alibi_and_mask(seq_len, scores.dtype)
            scores = scores + alibi + mask
            attn_weights = ops.softmax(scores, axis=-1)[0]

            # The correct target id is whatever token the tokenizer actually
            # produces right after the prompt's own text.
            prompt_ids = tokenizer.encode(prompt).ids
            continued_ids = tokenizer.encode(prompt + name).ids
            target_id = continued_ids[len(prompt_ids)] if len(continued_ids) > len(prompt_ids) else None

            bucket = vocab_weights[0]
            target_weight = float(bucket[target_id]) if target_id is not None else float("nan")
            full_order = ops.convert_to_numpy(ops.argsort(-bucket))
            target_rank = int(list(full_order).index(target_id)) + 1 if target_id is not None else -1
            target_in_context = target_id in set(ids) if target_id is not None else False

            _, top_idx = so.top_k(bucket, k=top_n)
            top_idx_list = [int(i) for i in top_idx]
            top_tokens = token_labels(tokenizer, top_idx_list)
            top_weights_vals = [float(bucket[i]) for i in top_idx_list]
            uniform_weight = 1.0 / model.vocabulary_size

            actual_logits = model(token_ids, training=False)[0]
            predicted_id = int(ops.argmax(actual_logits))
            predicted_token = token_labels(tokenizer, [predicted_id])[0]

            real_retrieved = _retrieve_from_vocab_weights(model, vocab_weights)
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
        context_weight_plot = plotly_div(
            _context_weight_bar(token_weights, prompt_tokens, name_positions),
            f"context-weight-{len(cases)}",
        )
        bucket_plot = plotly_div(
            _bucket_bar(top_tokens, top_weights_vals, name),
            f"bucket-{len(cases)}",
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
                "target_weight": target_weight,
                "target_in_context": target_in_context,
                "top_tokens": list(zip(top_tokens, top_weights_vals)),
                "uniform_weight": uniform_weight,
                "retrieval_invariance": retrieval_invariance,
                "rank_with_real_retrieval": rank_with_real,
                "rank_with_uniform_retrieval": rank_with_uniform,
                "predicted_with_uniform_retrieval": predicted_with_uniform,
                "plots": {
                    "heads": head_plots,
                    "last_row": last_row_plot,
                    "context_weight": context_weight_plot,
                    "bucket": bucket_plot,
                },
            }
        )

    hits = sum(1 for c in cases if c["is_top1"])
    mean_retrieval_invariance = sum(c["retrieval_invariance"] for c in cases) / len(cases)
    in_context_hits = sum(1 for c in cases if c["target_in_context"])
    unaffected_by_ablation = sum(
        1 for c in cases if c["predicted_with_uniform_retrieval"].strip() == c["name"]
    )

    # Least-invariant (most content-dependent) retrieval first.
    cases_by_invariance = sorted(cases, key=lambda c: c["retrieval_invariance"])

    context: Dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "checkpoint_path": str(ckpt_path),
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "num_cases": len(cases),
        "num_hits": hits,
        "num_heads": model.attn1.num_heads,
        "mean_retrieval_invariance": mean_retrieval_invariance,
        "in_context_hits": in_context_hits,
        "unaffected_by_ablation": unaffected_by_ablation,
        "cases": cases_by_invariance,
    }

    out_path = cfg_mod.run_dir(run_name) / f"reports/token_retrieval_trace_{prompt_set_name}.html"
    render_report("token_retrieval_trace.html", context, out_path)
    return out_path


PROMPT_SETS: Dict[str, List[Tuple[str, str]]] = {
    "names": DEFAULT_PROMPTS,
    "nouns": NOUN_PROMPTS,
    "long-context": LONG_CONTEXT_PROMPTS,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model scripts token-retrieval-trace",
        description="Trace Block 1's attention and token-anchored retrieval for in-context recall.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="latest", help="latest | best | <epoch number>")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N scattered-bucket candidates to show per prompt.")
    parser.add_argument(
        "--prompt-set",
        default="names",
        choices=sorted(PROMPT_SETS),
        help=(
            "'names' (default, proper-name recall), 'nouns' (common-noun recall: "
            "objects/animals/places) or 'long-context' (target buried under ~120 "
            "tokens of distractor content, filling the full window)."
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_path = run(
        args.run_name,
        checkpoint=args.checkpoint,
        prompts=PROMPT_SETS[args.prompt_set],
        top_n=args.top_n,
        prompt_set_name=args.prompt_set,
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
