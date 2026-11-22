"""What each layer of an `lru-banks` run contributes, on validation rows (window-long, from a zero state, every
position scored as in dense validation):
- the size of its output projections (W_o, W_o2, the FFN's W2: a `silent_from` layer starts with them at 0),
- the size of its addition to the stream (rms of layer output - layer input) relative to the stream it gets,
- its LRU input gate (mean),
- the loss when it is silenced (its output projections zeroed: the layer adds nothing) minus the full model's.

    uv run python "ad hoc scripts/lru_banks_layer_use.py" --run-name X [--checkpoint latest] [--rows 16] [--seed 0]
"""

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="xpu")
    a = p.parse_args()
    from model.device import setup_device

    setup_device(a.device)
    import torch

    import model.archs.lru_banks as lb
    from model import config as cfg_mod
    from model.evaluate import load_checkpoint

    m = load_checkpoint(a.run_name, int(a.checkpoint) if a.checkpoint.isdigit() else a.checkpoint)
    cfg = cfg_mod.load(a.run_name)
    path = cfg_mod.dataset_split_path(cfg["data"]["dataset"], "validation")
    tokens = np.load(path, mmap_mode="r")
    T = m.window_size + 1
    rng = np.random.default_rng(a.seed)
    starts = rng.integers(0, len(tokens) - T, a.rows)
    rows = np.stack([np.asarray(tokens[s:s + T], dtype=np.int64) for s in starts])
    x, y = rows[:, :-1].astype(np.int32), torch.as_tensor(rows[:, 1:])

    streams = []

    def layer(*args):
        out = type(m)._layer(m, *args)
        streams.append((args[1].detach(), out[0].detach()))
        return out

    def loss(record=False):
        streams.clear()
        if record:
            m._layer = layer
        try:
            with torch.no_grad():
                h = m.dense_hidden(x)
                tot = 0.0
                for b in range(h.shape[0]):  # one row at a time: (512, 32768) logits
                    lg = torch.as_tensor(m.readout(m.readout_input(h[b:b + 1]))).float()[0].cpu()
                    tot += float(torch.nn.functional.cross_entropy(lg, y[b], reduction="sum"))
        finally:
            if record:
                del m._layer
        return tot / y.numel()

    lb.FUSED = False
    base = loss(record=True)
    print(f"{a.run_name} ({a.checkpoint}), {a.rows} validation rows x {m.window_size}: loss {base:.4f} nats")
    sizes = []
    for n, (h_in, h_out) in enumerate(streams):
        rms = lambda t: float(t.float().pow(2).mean().sqrt())
        sizes.append((rms(h_out - h_in), rms(h_in)))
    gates = []
    orig_lru = lb.lru_states

    def lru(proj, nu, theta, gate, *args, **kw):
        gates.append(None if gate is None else float(gate.float().mean()))
        return orig_lru(proj, nu, theta, gate, *args, **kw)

    lb.lru_states = lru
    try:
        with torch.no_grad():
            m.dense_hidden(x[:2])
    finally:
        lb.lru_states = orig_lru
    print(f"{'layer':>5} {'hops':>4} {'|W_o|':>7} {'|W_o2|':>7} {'|W2|':>7} {'add/stream':>10} {'gate':>6} "
          f"{'silenced: d loss':>16}")
    for n, prm in enumerate(m._params):
        outs = [k for k in ("w_o", "w_o2", "w2", "b2_bias") if k in prm]
        saved = {k: prm[k].value.detach().clone() for k in outs}
        norm = lambda k: f"{float(saved[k].float().norm()):7.3f}" if k in saved else f"{'-':>7}"
        for k in outs:
            prm[k].assign(np.zeros(tuple(saved[k].shape), "float32"))
        try:
            d = loss() - base
        finally:
            for k in outs:
                prm[k].assign(saved[k])
        add, stream = sizes[n]
        g = "-" if gates[n] is None else f"{gates[n]:.3f}"
        print(f"{n + 1:>5} {m.layer_hops[n]:>4} {norm('w_o')} {norm('w_o2')} {norm('w2')} {add / stream:10.3f} "
              f"{g:>6} {d:+16.4f}")


if __name__ == "__main__":
    main()
