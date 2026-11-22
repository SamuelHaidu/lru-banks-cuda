"""`rope-state-memory`: does the model use its memory on long context? CPU only.

    uv run python "ad hoc scripts/rsm_memory_probe.py" --run-name rsm-8L-768-tbptt [--checkpoint latest]

1. Real text: validation rows of `--length` tokens from a document start, loss on the assistant-span targets, with the
   memory read on, off (gate 0) and forced open (gate 1), by distance from the row start.
2. Planted string: <bos> + a random token string S + filler (real validation text) + S again. The loss on the second
   S vs a control row whose first S differs, for several filler lengths: a gap means the string was stored and read.
   Also with the memory read off.
"""

import argparse
import json

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()
setup_device("cpu")

import torch  # noqa: E402

from model import config as cfg_mod  # noqa: E402
import model.archs  # noqa: E402,F401
from model.evaluate import load_checkpoint  # noqa: E402


def nll_of(m, ids):
    """Per-position next-token NLL (B, T-1) of int64 rows."""
    with torch.no_grad():
        x = torch.as_tensor(ids[:, :-1], dtype=torch.int32)
        h = m.dense_hidden(x)
        lp = torch.log_softmax(torch.as_tensor(m.readout(m.readout_input(h))), -1)
        return -lp.gather(-1, torch.as_tensor(ids[:, 1:])[..., None])[..., 0].numpy()


def with_gate(m, value, fn):
    orig = m._gate
    if value is not None:
        m._gate = lambda n, q: torch.full(q.shape[:-1] + (1,), value)
    try:
        return fn()
    finally:
        m._gate = orig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--length", type=int, default=2048)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--strings", type=int, default=6)
    a = p.parse_args()
    config = cfg_mod.load(a.run_name)
    m = load_checkpoint(a.run_name, a.checkpoint)
    setup_device("cpu")
    m.use_kernels = False
    ds = config["data"]["dataset"]
    meta = json.loads((cfg_mod.DATASETS_DIR / ds / "metadata.json").read_text())
    tokens = np.load(cfg_mod.DATASETS_DIR / ds / meta["validation"]["file"], mmap_mode="r")
    bos = int(meta["doc_token_id"])
    starts = np.flatnonzero(np.asarray(tokens[:5_000_000]) == bos)
    rng = np.random.default_rng(0)
    print(f"run {a.run_name}  checkpoint {a.checkpoint}  layers {m.num_layers}  window {m.window_size}")

    # 1. real text
    L = a.length
    pick = rng.choice(starts[starts + L + 1 < len(tokens)], a.rows, replace=False)
    seq = np.stack([np.asarray(tokens[q:q + L + 1]) for q in pick]).astype(np.int64)
    asst, end = 6, 7  # <assistant> </assistant>
    inside = np.zeros(seq.shape, bool)
    for r in range(len(seq)):
        on = False
        for t in range(L + 1):
            if seq[r, t] == asst:
                on = True
            inside[r, t] = on
            if seq[r, t] == end:
                on = False
    target_in = inside[:, 1:]
    print(f"\n== real text: {a.rows} rows x {L}, {int(target_in.sum())} assistant targets")
    edges = [0, 128, 512, 1024, L]
    res = {}
    for tag, val in (("memory on", None), ("memory off (g=0)", 0.0), ("gate open (g=1)", 1.0)):
        res[tag] = with_gate(m, val, lambda: nll_of(m, seq))
    for tag, nll in res.items():
        parts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = target_in[:, lo:hi]
            parts.append(f"{lo}-{hi}: {nll[:, lo:hi][sel].mean():.3f}")
        print(f"   {tag:18s} all {nll[target_in].mean():.3f}   " + "  ".join(parts))

    # 2. planted string
    V = m.vocabulary_size
    S = 12
    print(f"\n== planted string: S = {S} random tokens, then filler, then S again (loss on the repeat, mean nats)")
    print("   filler   same S   different S   same S, memory off   chance")
    for filler in (0, 32, 128, 512, 1500):
        same, ctrl, off = [], [], []
        for _ in range(a.strings):
            s1 = rng.integers(16, V, S)
            s2 = rng.integers(16, V, S)
            q = rng.choice(starts[starts + filler + 1 < len(tokens)])
            fill = np.asarray(tokens[q + 1:q + 1 + filler]).astype(np.int64)
            fill = np.where(fill == bos, 16, fill)
            row = lambda first: np.concatenate([[bos], first, fill, s1])[None]
            sl = slice(len(row(s1)[0]) - S - 1, len(row(s1)[0]) - 1)  # targets: the S tokens of the repeat
            same.append(nll_of(m, row(s1))[0, sl].mean())
            ctrl.append(nll_of(m, row(s2))[0, sl].mean())
            off.append(with_gate(m, 0.0, lambda: nll_of(m, row(s1)))[0, sl].mean())
        print(f"   {filler:6d}   {np.mean(same):6.3f}   {np.mean(ctrl):9.3f}   {np.mean(off):17.3f}   {np.log(V - 16):6.2f}")


if __name__ == "__main__":
    main()
