"""`rope-state-memory` banks: how the softmax writes behave, how the bank decays moved, and whether banks get
overwritten.

    uv run python "ad hoc scripts/rsm_bank_writes.py" --run-name rsm-1L-banks [--checkpoint latest] [--rows 64]

CPU only. Over `rows` validation rows (packed documents, as in training), layer 1:
  - write sharpness: effective banks per write (exp entropy), top-1 weight;
  - decays: every bank's half-life at init vs. now, and whether the long-lived banks are the busy ones;
  - bank use: write mass per bank; for the busiest banks, the address context (the token before) and the content
    (the token written), so what each bank specialized in;
  - overwriting: within one document, strong writes (top-1 weight >= 0.5) of a *different* token into a bank whose
    previous strong write is still within the bank's half-life.
"""

import argparse
import json
from collections import Counter

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()
setup_device("cpu")

import torch  # noqa: E402

from model import config as cfg_mod  # noqa: E402
import model.archs  # noqa: E402,F401
from model.evaluate import load_checkpoint  # noqa: E402
from model.scripts._model_io import load_tokenizer  # noqa: E402


def rows(config, n, length, seed=0):
    ds = config["data"]["dataset"]
    meta = json.loads((cfg_mod.DATASETS_DIR / ds / "metadata.json").read_text())
    tokens = np.load(cfg_mod.DATASETS_DIR / ds / meta["validation"]["file"], mmap_mode="r")
    bos = int(meta["doc_token_id"])
    head = np.asarray(tokens[: min(len(tokens), 5_000_000)])
    starts = np.flatnonzero(head == bos)
    pick = np.random.default_rng(seed).choice(starts[starts + length < len(head)], n, replace=False)
    return np.stack([head[p:p + length] for p in pick]).astype(np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--rows", type=int, default=64)
    a = p.parse_args()
    config = cfg_mod.load(a.run_name)
    m = load_checkpoint(a.run_name, a.checkpoint)
    setup_device("cpu")
    m.use_kernels = False
    assert m.memory == "banks", "this analysis is for memory='banks'"
    tok = load_tokenizer(config)
    name = lambda t: repr(tok.id_to_token(int(t))) if tok else str(int(t))
    x = rows(config, a.rows, m.window_size)
    S, n = m.num_banks, 0

    with torch.no_grad():
        ids = torch.as_tensor(x, dtype=torch.int32)
        E = m.token_embedding.embeddings.value
        reset = ids == m.reset_token
        u = torch.nn.functional.embedding(ids.long(), E)
        u = m._norm(m._w("in_norm", n), u) if m.pre_norm else u
        s, _ = m._state_torch(n, u, reset, E.new_zeros(len(x), m.dim))
        prev = torch.cat([E.new_zeros(len(x), 1, m.dim), s[:, :-1]], 1)
        read = s @ m._w("address_q", n).value
        write = torch.where(reset[..., None], 0.0, prev) @ m._w("address_k", n).value
        w = torch.softmax(write, -1).numpy()  # (B, N, S) write weights (before the (1 - mu) factor)
        r = torch.softmax(read, -1).numpy()
        mu = m.decays(n)[1].numpy().astype("float64")
    hl = np.log(0.5) / np.log(mu)
    init = np.geomspace(m.memory_half_life_min, m.memory_half_life_max, S)
    B, N, _ = w.shape
    valid = ~(x == m.reset_token)  # writes at the markers are the fresh-memory writes; leave them out

    ent = lambda p: np.exp(-(p * np.log(p + 1e-12)).sum(-1))
    top1 = w.max(-1)
    print(f"run {a.run_name}  checkpoint {a.checkpoint}  {B} rows x {N} tokens  {S} banks")
    print("\n== writes")
    print(f"   effective banks per write: median {np.median(ent(w)[valid]):.1f}   (per read: {np.median(ent(r)[valid]):.1f})")
    print("   top-1 write weight percentiles 10/50/90: " + " ".join(f"{v:.2f}" for v in np.percentile(top1[valid], [10, 50, 90])))
    print(f"   writes with top-1 >= 0.5: {np.mean(top1[valid] >= 0.5):.0%}   >= 0.9: {np.mean(top1[valid] >= 0.9):.0%}")

    print("\n== decays (half-life in tokens), init -> now, banks sorted by their init")
    order = np.argsort(init)
    for lo, hi in ((0, 16), (16, 48), (48, 80), (80, 112), (112, 128)):
        sel = order[lo:hi]
        print(f"   banks {lo:>3}-{hi - 1:<3}: init {init[sel].min():7.1f} .. {init[sel].max():7.1f}   now "
              f"{hl[sel].min():7.1f} .. {hl[sel].max():7.1f}   median ratio now/init {np.median(hl[sel] / init[sel]):.2f}")
    mass = (w * valid[..., None]).sum((0, 1))
    mass = mass / mass.sum()
    busy = np.argsort(-mass)
    print(f"   corr(log half-life, write share) {np.corrcoef(np.log(hl), mass)[0, 1]:+.2f}")

    print("\n== bank use")
    cum = np.cumsum(mass[busy])
    print(f"   banks holding 50% / 90% of the write mass: {np.searchsorted(cum, 0.5) + 1} / {np.searchsorted(cum, 0.9) + 1} of {S};"
          f"  busiest bank {mass[busy[0]]:.1%} (uniform: {1 / S:.1%});  banks with < 0.1%: {int((mass < 0.001).sum())}")
    hard = np.where(valid, w.argmax(-1), -1)  # (B, N) the bank each token mainly writes to
    prev_tok = np.concatenate([np.full((B, 1), m.reset_token), x[:, :-1]], 1)
    print("   busiest banks: share, half-life, address (token before) -> content (token written):")
    for b in busy[:12]:
        sel = (hard == b) & (top1 >= 0.3)
        ctx = Counter(prev_tok[sel].tolist()).most_common(3)
        cnt = Counter(x[sel].tolist()).most_common(4)
        distinct = len(set(x[sel].tolist()))
        print(f"   bank {b:>3}: {mass[b]:5.1%}  hl {hl[b]:6.1f}  | after {', '.join(f'{name(t)}' for t, _ in ctx):<32} "
              f"| writes {', '.join(f'{name(t)} {c}' for t, c in cnt)}  ({distinct} distinct tokens)")

    print("\n== overwriting (strong writes, top-1 >= 0.5, within one document)")
    events = same = 0
    gaps = []
    for i in range(B):
        last = {}  # bank -> (position, token) of its previous strong write in this document
        for t in range(N):
            if x[i, t] == m.reset_token:
                last = {}
                continue
            if top1[i, t] < 0.5:
                continue
            b = int(hard[i, t])
            if b in last:
                pt, ptok = last[b]
                if t - pt <= hl[b]:
                    events += 1
                    gaps.append(t - pt)
                    same += int(ptok == x[i, t])
            last[b] = (t, x[i, t])
    strong = int((top1[valid] >= 0.5).sum())
    print(f"   strong writes {strong}; into a bank still holding an earlier strong write (within its half-life): "
          f"{events} ({events / max(strong, 1):.0%})")
    if events:
        print(f"   of those, the same token again: {same / events:.0%}  (a different token: overwrite {1 - same / events:.0%});"
              f"  median gap {np.median(gaps):.0f} tokens")


if __name__ == "__main__":
    main()
