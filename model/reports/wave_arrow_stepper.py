"""`model report wave-arrow-stepper`: step through a story with a `wave-arrow` run.

Exports the model's exact state S_{l,k}(i) of every layer after every token (its own `scan_layers`, the whole text as context) of 10
random training documents and 2 texts the model writes itself (cut at the first <eos>), and the learned decay and frequency of every
channel. The page shows, for the chosen text:

* loss per word (red-green) and, at the chosen step, what each layer adds to the logits (the readout is linear, so the
  logits are the sum of the layers' MLP outputs times the readout) and the prediction;
* clocks: one per channel, one row per state space of every layer, all layers on the page; the hand is S (angle = phase, length = amplitude, scaled by
  the largest amplitude of that state space over the story);
* the clicked channel: amplitude and phase over the story, and its learned decay e^{-alpha d} (dashed: the init);
* what the current token did to every channel: amplitude and phase of S(i) against lambda S(i-1), the state the
  clock would have shown without the token.

Offline page, Tailwind CSS inlined (`templates/report_tailwind.css`, built by `tools/report-css`).
"""

ARCHS = ('wave-arrow',)

import argparse
import html as _html
import json
from pathlib import Path
from typing import List

import numpy as np

from model.reports._model_io import load_full_model, load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="model report wave-arrow-stepper",
                                description="Channel clocks, timelines, decays and token effects of a wave-arrow run.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--text", default=None, help="An extra text to step through.")
    p.add_argument("--documents", type=int, default=10, help="Training documents (random, seeded).")
    p.add_argument("--generated", type=int, default=2, help="Texts the model writes from <bos> (temperature 0.8, top-k 50).")
    p.add_argument("--min-tokens", type=int, default=150)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    return p


def _training_documents(config, n: int, min_tokens: int, max_tokens: int, seed: int) -> List[List[int]]:
    from model import config as cfg_mod
    from model.data import _document_bounds, load_tokens

    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
    path = str(cfg_mod.dataset_split_path(config["data"]["dataset"], "train"))
    tokens = load_tokens(path)
    starts, ends = _document_bounds(path, tokens, int(meta["doc_token_id"]), True)
    long_enough = np.flatnonzero(ends - starts >= min_tokens)
    pick = np.random.default_rng(seed).choice(long_enough, size=n, replace=False)
    return [[int(t) for t in tokens[starts[d]: min(ends[d], starts[d] + max_tokens)]] for d in pick]


def main(argv: List[str]) -> None:
    args = build_parser().parse_args(argv)
    import torch
    from keras import ops

    from model import config as cfg_mod

    model, config, ckpt = load_full_model(args.run_name, args.checkpoint)
    if config["arch"] != "wave-arrow":
        raise SystemExit(f"Run {args.run_name!r} is {config['arch']!r}, not wave-arrow.")
    tok = load_tokenizer(config)
    K, L = model.channels, model.state_spaces
    label = lambda i: (tok.id_to_token(int(i)) or f"<{i}>").replace("Ġ", " ").replace("Ċ", "\\n")

    eos = int(cfg_mod.load_dataset_metadata(config["data"]["dataset"])["eos_token_id"])
    texts = [(f"training document {n + 1}", ids) for n, ids in
             enumerate(_training_documents(config, args.documents, args.min_tokens, args.max_tokens, args.seed))]
    for n in range(args.generated):
        np.random.seed(args.seed + n)  # `generate` samples with np.random
        out = model.generate([model.reset_token], max_new_tokens=args.max_tokens - 1, temperature=0.8, top_k=50)
        ids = [int(t) for t in out.split()]
        if eos in ids[1:]:  # the text ends at the first <eos>
            ids = ids[: ids.index(eos, 1) + 1]
        texts.append((f"generated text {n + 1}", ids))
    if args.text is not None:
        texts.append(("your text", [model.reset_token] + tok.encode(args.text).ids[: args.max_tokens - 1]))

    np32 = lambda t: ops.convert_to_numpy(t).astype("float64")
    r = lambda a: [float(f"{v:.3g}") for v in np.asarray(a).reshape(-1)]
    r4 = lambda a: [float(f"{v:.4g}") for v in np.asarray(a).reshape(-1)]
    NL, TOP = model.num_layers, 5
    kernel = np32(model.readout.kernel)  # (dim, V), no bias: the logits are the sum of the layers' y_n @ kernel
    docs = []
    for name, ids in texts:
        N = len(ids)
        with torch.no_grad():
            x = ops.convert_to_tensor(np.asarray([ids], dtype=np.int32))
            states = model.scan_layers(x)  # per layer (1, N, L K)
            ys = [np32(model._mlp(n, sr, si))[0] for n, (sr, si) in enumerate(states)]  # per layer (N, dim)
        parts = [y @ kernel for y in ys]  # per layer (N, V): what the layer adds to every logit
        logits = sum(parts)
        logp = logits - logits.max(-1, keepdims=True)
        logp -= np.log(np.exp(logp).sum(-1, keepdims=True))
        nxt = np.asarray(ids[1:] + [ids[-1]])  # the token each position predicts (the last one has none)
        valid = np.asarray([i + 1 < N and ids[i + 1] != model.reset_token for i in range(N)])
        loss = np.where(valid, -logp[np.arange(N), nxt], 0.0)
        pred = logits.argmax(-1)
        order = np.argsort(-logits, axis=-1)[:, :TOP]
        rows = np.arange(N)
        layer_info = []
        for part in parts:
            o = np.argsort(-part, axis=-1)[:, :TOP]
            layer_info.append({"top": [[label(t) for t in row] for row in o], "val": r(np.take_along_axis(part, o, -1)),
                               "act": r(part[rows, nxt]), "pred": r(part[rows, pred])})
        docs.append({"name": name, "words": [label(i) for i in ids], "N": N,
                     "reset": [int(i == model.reset_token) for i in ids],  # the state restarts at these tokens
                     "valid": [int(v) for v in valid], "loss": r(loss),
                     "pred": [label(i) for i in pred], "top": [[label(t) for t in row] for row in order],
                     "prob": r(np.exp(np.take_along_axis(logp, order, -1))), "pact": r(np.exp(logp[rows, nxt])),
                     "lact": r(logits[rows, nxt]), "lpred": r(logits[rows, pred]), "parts": layer_info,
                     # all layers' states side by side (N, NL * L, K); 4 digits: the page recovers writes from differences
                     "sr": r4(np.concatenate([np32(a[0]) for a, _ in states], -1)),
                     "si": r4(np.concatenate([np32(b[0]) for _, b in states], -1))})
    alpha0 = np.log(2.0) / np.geomspace(1.0, model.window_size / 2, L)  # init decay per space
    alpha = np.concatenate([np.log1p(np.exp(np32(model._w("decay_raw", n)))).reshape(-1) for n in range(NL)])  # (NL L K)
    omega = np.concatenate([np.tile(np32(model._w("channel_omega", n)), L) for n in range(NL)])  # per row of K channels
    payload = json.dumps({"docs": docs, "K": K, "L": L, "NL": NL, "alpha": r(alpha), "alpha0": r(np.tile(alpha0, NL)),
                          "omega": [float(f"{v:.6g}") for v in omega]}, separators=(",", ":"))
    out = (ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent) / "reports" / "wave_arrow_stepper.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    templates = Path(__file__).resolve().parent / "templates"
    page = (templates / "wave_arrow_stepper.html").read_text()
    out.write_text(page.replace("/*__CSS__*/", (templates / "report_tailwind.css").read_text())
                   .replace("__TITLE__", _html.escape(f"{args.run_name} / {ckpt.name} - {model.num_layers} layer(s) x {L} state spaces x {K} channels"))
                   .replace("/*__DATA__*/null", payload))
    print(f"Wrote {out}  ({len(docs)} texts, {len(payload) / 1e6:.1f} MB data)")
