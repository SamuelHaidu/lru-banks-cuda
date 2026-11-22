"""`model report log-gaussian-kernel-stepper`: step through a text, see the kernel per word.

For `rope-log-gaussian-sum`. Runs the model at every position of one text
(a validation story by default, or `--text`) and writes an interactive HTML
page: step token by token (slider, buttons or arrow keys), the context is
shaded by the weight the model gives each earlier token, and below it the
distance kernel generated at that step is plotted, with the step's top
predictions and the true next token.

For `rope-kernel-attention` it also plots each attention head's weights over
the context at that step, and the text shading can switch to any head. For
`rope-kernel-hops` every further hop gets its own section: the kernel curve
("waves") that hop generated at that step and its attention heads.
`--generate N` steps through a text the model samples itself from `<bos>`
instead of a validation story.
"""

ARCHS = ('rope-log-gaussian-sum',)

import argparse
import datetime as dt
from typing import Any, Dict, List, Optional

import numpy as np

from model.device import setup_backend

setup_backend()

from model import config as cfg_mod  # noqa: E402
from model.archs.rope_log_gaussian_sum import RopeLogGaussianSumLM  # noqa: E402
from model.data import _document_bounds, load_tokens  # noqa: E402
from model.reports._model_io import load_full_model, load_tokenizer  # noqa: E402
from model.reports._report import render_report, token_label  # noqa: E402


def text_ids(config: Dict[str, Any], tokenizer: Any, text: Optional[str], doc_index: int, max_tokens: int) -> List[int]:
    """Token ids starting with the document marker, as the training windows see them."""
    name = config["data"]["dataset"]
    meta = cfg_mod.load_dataset_metadata(name)
    doc_id = int(meta["doc_token_id"])
    if text is not None:
        return [doc_id] + tokenizer.encode(text).ids[: max_tokens - 1]
    path = str(cfg_mod.dataset_split_path(name, "validation"))
    tokens = load_tokens(path)
    starts, ends = _document_bounds(path, tokens, doc_id, True)
    d = doc_index % len(starts)
    return [int(t) for t in tokens[starts[d] : min(ends[d], starts[d] + max_tokens)]]


def generate_ids(model: Any, doc_id: int, eos_id: Optional[int], n: int, temperature: float, top_k: int, seed: int) -> List[int]:
    """Samples up to `n` tokens from the model, starting from just the document marker."""
    import torch

    rng = np.random.default_rng(seed)
    T = model.window_size
    ids = [doc_id]
    device = model.distance_weights.value.device
    with torch.no_grad():
        for _ in range(n):
            x = np.zeros((1, T), dtype=np.int32)
            ctx = ids[-T:]
            x[0, T - len(ctx) :] = ctx
            logits = model(torch.as_tensor(x, device=device))[0].cpu().double().numpy()
            if temperature <= 0:
                nxt = int(np.argmax(logits))
            else:
                logits = logits / temperature
                if top_k:
                    cut = np.partition(logits, -top_k)[-top_k]
                    logits[logits < cut] = -np.inf
                p = np.exp(logits - logits.max())
                nxt = int(rng.choice(p.size, p=p / p.sum()))
            ids.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
    return ids


def run(run_name: str, checkpoint: str, text: Optional[str], doc_index: int, max_tokens: int, top_k: int,
        generate: int = 0, temperature: float = 0.8, sample_top_k: int = 50, seed: int = 0):
    import torch

    model, config, ckpt_path = load_full_model(run_name, checkpoint)
    if not isinstance(model, RopeLogGaussianSumLM):
        raise SystemExit(f"Run {run_name!r} is {config['arch']!r}; this script needs 'rope-log-gaussian-sum'.")
    tokenizer = load_tokenizer(config)
    T = model.window_size
    if generate > 0:
        meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
        ids = generate_ids(model, int(meta["doc_token_id"]), meta.get("eos_token_id"), generate, temperature, sample_top_k, seed)
    else:
        ids = text_ids(config, tokenizer, text, doc_index, max_tokens)
    if len(ids) < 2:
        raise SystemExit("Need at least 2 tokens.")

    # One window per step: context = everything up to and including position i,
    # left-padded with the doc marker (0) exactly like the training sampler.
    n_steps = len(ids) - 1
    x = np.zeros((n_steps, T), dtype=np.int32)
    for i in range(n_steps):
        ctx = ids[max(0, i + 1 - T) : i + 1]
        x[i, T - len(ctx) :] = ctx

    device = model.distance_weights.value.device
    with torch.no_grad():
        xb = torch.as_tensor(x, device=device)
        _, _, alpha, w = model.kernels(xb)
        logits = model(xb)  # the model's own forward, whatever stages it has
        has_attention = hasattr(model, "attention")
        hop_trace = []
        if hasattr(model, "hops"):
            _, trace = model.hops(xb)
            attn = trace[0]["probs"].cpu().numpy()[:, :, ::-1]  # (steps, H, T), distance order
            hop_trace = [{"kernel": t["kernel"].cpu().numpy()[:, ::-1], "probs": t["probs"].cpu().numpy()[:, :, ::-1]}
                         for t in trace[1:]]
        elif has_attention:
            attn = model.attention(xb)[1].cpu().numpy()[:, :, ::-1]  # (steps, H, T), distance order
        # softmax/topk on CPU: XPU topk exceeds the Iris Xe work-group limit.
        probs = torch.softmax(logits.cpu(), dim=-1)
        top_p, top_i = probs.topk(top_k, dim=-1)
        y = torch.as_tensor(ids[1:])
        p_true = probs.gather(1, y[:, None])[:, 0]
        static = model.distance_weights.value.detach()

    static_k = static.cpu().numpy()[::-1]  # distance order: index 0 = k=1
    adaptive_k = w.cpu().numpy()[:, ::-1]
    combined_k = static_k[None, :] + adaptive_k
    alpha_np = alpha.cpu().numpy()

    label = lambda t: token_label(tokenizer, int(t)) if tokenizer else str(int(t))
    steps = []
    for i in range(n_steps):
        n_real = min(i + 1, T)
        comb = combined_k[i]
        mass = comb / comb.sum()
        steps.append({
            "combined": [round(float(v), 5) for v in comb],
            "adaptive": [round(float(v), 5) for v in adaptive_k[i]],
            "mass": [round(float(v), 5) for v in mass[:n_real]],  # per real context token, k = 1..n_real
            "alpha": [round(float(v), 4) for v in alpha_np[i]],
            "top": [[label(t), round(float(p), 4)] for t, p in zip(top_i[i].tolist(), top_p[i].tolist())],
            "p_true": round(float(p_true[i]), 4),
            "long_share": round(float(mass[16:].sum()), 3),
            "k50": int(1 + (np.cumsum(mass) < 0.5).sum()),
        })
        if has_attention:
            steps[-1]["attn"] = [[round(float(v), 4) for v in head[:n_real]] for head in attn[i]]
        if hop_trace:
            steps[-1]["hops"] = [{
                "kernel": [round(float(v), 4) for v in hop["kernel"][i]],
                "attn": [[round(float(v), 4) for v in head[:n_real]] for head in hop["probs"][i]],
            } for hop in hop_trace]

    context = {
        "run_name": run_name,
        "checkpoint": ckpt_path.name,
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "kernel_basis": model.kernel_basis,
        "source": (f"self-generated from <bos> (T={temperature}, top-k {sample_top_k}, seed {seed})" if generate > 0
                   else "custom text" if text is not None else f"validation story #{doc_index}"),
        "generated_text": tokenizer.decode(ids[1:]) if generate > 0 and tokenizer else None,
        "has_attention": has_attention,
        "num_extra_hops": len(hop_trace),
        "hop_info": ([{"hop": i + 2, "attention_gate": round(float(np.asarray(model.hop_attention_gate[i].numpy())), 3),
                       "kernel_gate": round(float(np.asarray(model.hop_kernel_gate[i].numpy())), 3),
                       "basis": model.hop_kernel_basis} for i in range(len(hop_trace))] if hop_trace else []),
        "output_mode": getattr(model, "output_mode", None),
        "attention_gate": float(model.attention_gate.numpy()) if getattr(model, "attention_residual", False) else None,
        "temperatures": [round(float(t), 2) for t in np.exp(model.log_temperature.numpy())] if has_attention else None,
        "num_tokens": len(ids),
        "data": {
            "tokens": [label(t) for t in ids],
            "steps": steps,
            "static": [round(float(v), 5) for v in static_k],
            "window": T,
            "centers": model.basis_labels(),
            "signed": model.kernel_basis != "log-gaussian",
        },
    }
    name = "log_gaussian_kernel_stepper_generated.html" if generate > 0 else "log_gaussian_kernel_stepper.html"
    out_path = cfg_mod.run_dir(run_name) / "reports" / name
    render_report("log_gaussian_kernel_stepper.html", context, out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model report log-gaussian-kernel-stepper",
        description="Interactive page: step through a text and see the distance kernel at each token.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="latest", help="latest | best | <epoch number>")
    parser.add_argument("--text", default=None, help="Text to step through (default: a validation story).")
    parser.add_argument("--doc-index", type=int, default=0, help="Which validation story, if no --text.")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=5, help="Predictions shown per step.")
    parser.add_argument("--generate", type=int, default=0, help="Sample N tokens from <bos> and step through them.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for --generate (0 = greedy).")
    parser.add_argument("--sample-top-k", type=int, default=50, help="Top-k filter for --generate.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_path = run(args.run_name, args.checkpoint, args.text, args.doc_index, args.max_tokens, args.top_k,
                   args.generate, args.temperature, args.sample_top_k, args.seed)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
