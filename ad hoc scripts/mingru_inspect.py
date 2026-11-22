"""What a `min-gru` run stores: per layer, the channels' timescales, the update gate per token, and for the slowest
channels which earlier tokens make up their value at the end of the text. CPU, writes nothing.

    uv run python "ad hoc scripts/mingru_inspect.py" --run-name X [--checkpoint latest] [--text "..."] [--doc 0]
        [--max-tokens 300] [--channels 4] [--show-tokens 60]

Without --text: validation document --doc of the run's dataset (first --max-tokens tokens).

Exact contributions: the state at the end T of a channel is sum_j weight_j * candidate_j with
weight_j = update_gate_j * keep_{j+1} * ... * keep_T (weights >= 0, summing to <= 1), so the tokens with the
largest weights are what the channel holds.
"""

import argparse
import json

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()
setup_device("cpu")

from model import config as cfg_mod  # noqa: E402
import model.archs  # noqa: E402,F401
from model.evaluate import load_checkpoint  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--text", default=None)
    p.add_argument("--doc", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--channels", type=int, default=4, help="slowest channels per layer to break down")
    p.add_argument("--show-tokens", type=int, default=60, help="tokens in the per-token table")
    a = p.parse_args()
    config = cfg_mod.load(a.run_name)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(cfg_mod.tokenizer_path(config["data"]["dataset"])))
    m = load_checkpoint(a.run_name, a.checkpoint)
    setup_device("cpu")
    if a.text is not None:
        ids = tok.encode(a.text).ids
        if ids[0] != m.reset_token:
            ids = [m.reset_token] + ids
    else:
        ds = config["data"]["dataset"]
        meta = json.loads((cfg_mod.DATASETS_DIR / ds / "metadata.json").read_text())
        tokens = np.load(cfg_mod.DATASETS_DIR / ds / meta["validation"]["file"], mmap_mode="r")
        starts = np.flatnonzero(np.asarray(tokens[:5_000_000]) == m.reset_token)
        ids = np.asarray(tokens[starts[a.doc]:starts[a.doc + 1]]).tolist()
    ids = ids[:a.max_tokens]
    T = len(ids)
    pieces = [tok.id_to_token(i) or str(i) for i in ids]
    show = lambda s: repr(s.replace("Ġ", " ").replace("Ċ", "\n"))[1:-1][:14]
    info = m.inspect(np.asarray([ids], np.int32))
    print(f"run {a.run_name}  checkpoint {a.checkpoint}  layers {m.num_layers}  hidden {m.hidden_size}  tokens {T}")

    for n, (layer, bias_hl) in enumerate(zip(info, m.timescales())):
        z, keep, cand, state = (layer[k][0] for k in ("update_gate", "keep_factor", "candidate", "state"))
        write = layer["layer_write"][0]
        # effective half-life per channel over this text: from the mean log keep (document markers excluded)
        body = keep[np.asarray(ids) != m.reset_token]
        eff_hl = np.log(0.5) / np.minimum(np.log(np.clip(body, 1e-12, 1)).mean(0), -1e-9)
        q = lambda x: " ".join(f"{v:8.1f}" for v in np.percentile(x, [0, 10, 50, 90, 100]))
        print(f"\n== layer {n + 1}")
        print(f"   half-life (tokens) min/10/50/90/max   bias only: {q(bias_hl)}")
        print(f"                                         on text:   {q(eff_hl)}")
        print(f"   update gate: mean {z.mean():.3f}   |state| mean {np.abs(state).mean():.3f}   "
              f"|write to stream| mean {np.linalg.norm(write, axis=-1).mean():.3f}")
        # what the slowest channels hold at the end: the exact weight of every earlier token
        log_keep_after = np.concatenate([np.cumsum(np.log(np.clip(keep[::-1], 1e-30, 1)), 0)[::-1][1:],
                                         np.zeros((1, keep.shape[1]))], 0)  # sum_{s>j} log keep_s
        weights = z * np.exp(log_keep_after)  # (T, H)
        for c in np.argsort(-eff_hl)[:a.channels]:
            top = np.argsort(-weights[:, c])[:6]
            parts = ", ".join(f"{show(pieces[j])}@{j}:{weights[j, c]:.2g}({cand[j, c]:+.2f})" for j in sorted(top))
            print(f"   channel {c:4d}  half-life {eff_hl[c]:8.1f}  state {state[-1, c]:+.3f}  "
                  f"held weight {weights[:, c].sum():.2f}  top tokens: {parts}")

    print("\n== per token: mean update gate per layer | size of each layer's write to the stream")
    head = " ".join(f"   g{n + 1}" for n in range(m.num_layers)) + " | " + " ".join(f"  w{n + 1}" for n in range(m.num_layers))
    print(f"{'pos':>4} {'token':14s} {head}")
    for t in range(min(T, a.show_tokens)):
        g = " ".join(f"{info[n]['update_gate'][0, t].mean():5.2f}" for n in range(m.num_layers))
        w = " ".join(f"{np.linalg.norm(info[n]['layer_write'][0, t]):4.2f}" for n in range(m.num_layers))
        print(f"{t:4d} {show(pieces[t]):14s} {g} | {w}")


if __name__ == "__main__":
    main()
