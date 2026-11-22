"""Zero-shot: the trained `memory="banks"` model with its memory written by the delta rule instead (no training).

    uv run python ad_hoc_scripts/rsm_delta_zero_shot.py --run-name rsm-1L-banks

Same W_k, W_q, bank decays and read gate; the banks are filled by Z <- D Z + beta k (v - k^T D Z)^T with a fixed
beta, and read o = Z^T q, for unit-length (k, q = softmax / |softmax|, as the delta mode) or sum-1 (the softmaxes
themselves) addresses. Reports validation loss and the Roxo / Roxy / Rilo recall slot. CPU only.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ad_hoc_scripts)
from rsm_bank_writes import rows  # noqa: E402  (CPU setup)
from rsm_name_recall import encode, piece_probs, NAMES  # noqa: E402
import torch  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402
from model.device import setup_device  # noqa: E402
from model.scripts._model_io import load_tokenizer  # noqa: E402


def delta_layer(m, beta_value, unit):
    """A replacement for m._layer: the trained model's state and addresses, the delta rule for the memory."""
    def layer(n, x, reset, carry, collect=False, fresh=False):
        s0, m0 = carry
        s, s_end = m._state_torch(n, x, reset, s0)
        prev = torch.cat([s0[:, None], s[:, :-1]], 1)
        read = s @ m._w("address_q", n).value
        write = torch.where(reset[..., None], 0.0, prev) @ m._w("address_k", n).value
        r, w = torch.softmax(read, -1), torch.softmax(write, -1)
        if unit:
            r, w = torch.nn.functional.normalize(r, dim=-1), torch.nn.functional.normalize(w, dim=-1)
        v = x @ m._w("value", n).kernel.value
        a = torch.nn.functional.softplus(m._w("decay_banks", n).value).clamp_max(80.0 / m.delta_chunk)
        beta = torch.full(v.shape[:2], float(beta_value))
        o, m_end = m._delta_memory(r, w, v, beta, a, reset, m0[..., :m.dim])
        o = o * m._gate(n, m.read_gate_in(s, read))
        return s + o, (s_end, m_end), ((s, o) if collect else None)
    return layer


def val_loss(m, x):
    with torch.no_grad():
        logits = m.readout(m.readout_input(m.dense_hidden(torch.as_tensor(x[:, :-1], dtype=torch.int32))))
        lp = torch.log_softmax(torch.as_tensor(logits), -1)
        y = torch.as_tensor(x[:, 1:])
        nll = -lp.gather(-1, y[..., None])[..., 0]
        return float(nll[y != m.reset_token].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--rows", type=int, default=32)
    a = p.parse_args()
    config = cfg_mod.load(a.run_name)
    m = load_checkpoint(a.run_name, a.checkpoint)
    setup_device("cpu")
    m.use_kernels = False
    assert m.memory == "banks"
    tok = load_tokenizer(config)
    x = rows(config, a.rows, m.window_size + 1)
    orig = m._layer
    variants = [("trained banks (running average)", None)]
    for unit in (True, False):
        for beta in (0.1, 0.3, 0.6, 1.0):
            variants.append((f"delta, {'unit' if unit else 'sum-1'} addresses, beta {beta}", (beta, unit)))
    print(f"run {a.run_name}  checkpoint {a.checkpoint}  {a.rows} validation rows")
    print(f"{'memory':<44} {'val loss':>8} | {'Roxo whole':>10} {'Roxo o rank':>11} | {'Roxy whole':>10} {'Rilo R rank':>11}")
    for tag, cfgv in variants:
        m._layer = orig if cfgv is None else delta_layer(m, *cfgv)
        loss = val_loss(m, x)
        res = {}
        for name in NAMES:
            ids, pos, _ = encode(tok, m.reset_token, name, name)
            res[name] = piece_probs(m, ids, pos)
        whole = lambda nm: float(np.prod([r[0] for r in res[nm]]))
        print(f"{tag:<44} {loss:8.4f} | {whole(' Roxo'):10.2e} {res[' Roxo'][2][1]:11d} | {whole(' Roxy'):10.2e} "
              f"{res[' Rilo'][0][1]:11d}")
    m._layer = orig


if __name__ == "__main__":
    main()
