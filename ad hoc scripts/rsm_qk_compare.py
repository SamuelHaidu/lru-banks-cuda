"""Did `rope-state-memory` learn the same matrix for the read (W_q) and write (W_k) bank addresses?

    uv run python "ad hoc scripts/rsm_qk_compare.py" --run-name rsm-1L-banks

CPU only. Compares the matrices (cosine, per-bank cosine, relative difference), their addresses on the same states
(top-1 bank agreement, overlap of the two softmaxes), and the validation loss when one matrix replaces the other.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "ad hoc scripts")
from rsm_bank_writes import rows  # noqa: E402  (also sets the CPU up)
import torch  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402
from model.device import setup_device  # noqa: E402


def loss(m, x):
    with torch.no_grad():
        ids = torch.as_tensor(x[:, :-1], dtype=torch.int32)
        logits = m.readout(m.readout_input(m.dense_hidden(ids)))
        lp = torch.log_softmax(torch.as_tensor(logits), -1)
        y = torch.as_tensor(x[:, 1:])
        nll = -lp.gather(-1, y[..., None])[..., 0]
        return float(nll[y != m.reset_token].mean())


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
    for n in range(m.num_layers):
        Wq = m._w("address_q", n).value.detach().clone()
        Wk = m._w("address_k", n).value.detach().clone()
        cos = float((Wq * Wk).sum() / (Wq.norm() * Wk.norm()))
        col = torch.nn.functional.cosine_similarity(Wq, Wk, dim=0).numpy()  # per bank (column)
        print(f"== layer {n + 1}: |W_q| {float(Wq.norm()):.2f}  |W_k| {float(Wk.norm()):.2f}")
        print(f"   cosine(W_q, W_k) {cos:+.3f}   |W_q - W_k| / |W_k| {float((Wq - Wk).norm() / Wk.norm()):.3f}")
        print("   per-bank cosine percentiles 10/50/90: " + " ".join(f"{v:+.2f}" for v in np.percentile(col, [10, 50, 90])))
        # addresses on the same states
        x = rows(config, a.rows, m.window_size + 1)
        with torch.no_grad():
            ids = torch.as_tensor(x[:, :-1], dtype=torch.int32)
            E = m.token_embedding.embeddings.value
            u = torch.nn.functional.embedding(ids.long(), E)
            u = m._norm(m._w("in_norm", n), u) if m.pre_norm else u
            s, _ = m._state_torch(n, u, ids == m.reset_token, E.new_zeros(len(x), m.dim))
            pq, pk = torch.softmax(s @ Wq, -1), torch.softmax(s @ Wk, -1)
        agree = float((pq.argmax(-1) == pk.argmax(-1)).float().mean())
        overlap = float((pq * pk).sum(-1).mean() / (pq * pq).sum(-1).mean())
        bc = float(torch.sqrt(pq * pk).sum(-1).mean())  # Bhattacharyya coefficient: 1 = identical distributions
        print(f"   same state s: top-1 bank agrees {agree:.0%}; Bhattacharyya overlap of the two softmaxes {bc:.3f}"
              f"  (1 = identical, uniform-vs-peaked ~ lower)")
        # function: share one matrix
        base = loss(m, x)
        res = {}
        for tag, q, k in (("W_q := W_k (one matrix: the write one)", Wk, Wk), ("W_k := W_q (one matrix: the read one)", Wq, Wq),
                          ("both := (W_q + W_k) / 2", (Wq + Wk) / 2, (Wq + Wk) / 2)):
            m._w("address_q", n).assign(q.numpy())
            m._w("address_k", n).assign(k.numpy())
            res[tag] = loss(m, x)
            m._w("address_q", n).assign(Wq.numpy())
            m._w("address_k", n).assign(Wk.numpy())
        print(f"   validation loss ({a.rows} rows): model {base:.4f}")
        for tag, v in res.items():
            print(f"   {tag:<42} {v:.4f}  ({v - base:+.4f})")


if __name__ == "__main__":
    main()
