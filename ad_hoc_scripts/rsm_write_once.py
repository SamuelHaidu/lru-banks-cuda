"""Training-free edit of a trained `memory="banks"` model: attenuate redundant writes by cosine similarity.

    uv run python ad_hoc_scripts/rsm_write_once.py --run-name rsm-1L-banks

Before writing v_t with weights w (the trained softmax), the write is scaled by how new v_t is:
    per-bank     red_b = max(0, cos(v_t, bank_b / z_b))           (empty banks: 0)
    per-address  red   = max(0, cos(v_t, sum_b w_b bank_b / sum_b w_b z_b))   what the memory returns at the address
    w'_b = w_b (1 - red)^gamma;  bank_b <- mu_b bank_b + (1 - mu_b) w'_b v_t,  z_b <- mu_b z_b + (1 - mu_b) w'_b
gamma = 0 is the trained model (checked). Same weights, no training; validation loss and the name-recall slot.
Runs the memory token by token in torch (CPU).
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


def novelty_layer(m, gamma, mode):
    def layer(n, x, reset, carry, collect=False, fresh=False):
        s0, m0 = carry
        s, s_end = m._state_torch(n, x, reset, s0)
        prev = torch.cat([s0[:, None], s[:, :-1]], 1)
        read = s @ m._w("address_q", n).value
        write = torch.where(reset[..., None], 0.0, prev) @ m._w("address_k", n).value
        r, w = torch.softmax(read, -1), torch.softmax(write, -1)
        v = x @ m._w("value", n).kernel.value
        mu = m.decays(n)[1]  # (S,)
        D = m.dim
        bank, z = m0[..., :D].clone(), m0[..., D].clone()  # (B, S, D), (B, S)
        keep = (~reset).to(v.dtype)
        outs = []
        for t in range(v.shape[1]):
            kt = keep[:, t, None]
            bank = bank * (mu * kt)[..., None]
            z = z * mu * kt
            wt, vt = w[:, t], v[:, t]
            if gamma > 0:
                if mode == "bank":
                    mean = bank / z.clamp_min(1e-6)[..., None]
                    red = torch.nn.functional.cosine_similarity(mean, vt[:, None], dim=-1).clamp_min(0)
                    red = torch.where(z > 1e-4, red, torch.zeros_like(red))
                else:
                    num = (wt[..., None] * bank).sum(1)
                    den = (wt * z).sum(1, keepdim=True)
                    red = torch.nn.functional.cosine_similarity(num / den.clamp_min(1e-6), vt, dim=-1).clamp_min(0)
                    red = torch.where(den[:, 0] > 1e-4, red, torch.zeros_like(red))[:, None]
                wt = wt * (1 - red) ** gamma
            add = (1 - mu) * wt  # (B, S)
            bank = bank + add[..., None] * vt[:, None]
            z = z + add
            rt = r[:, t]
            outs.append((rt[..., None] * bank).sum(1) / (rt * z).sum(1, keepdim=True).clamp_min(1e-12))
        o = torch.stack(outs, 1) * m._gate(n, m.read_gate_in(s, read))
        m_end = torch.cat([bank, z[..., None], bank.new_zeros(bank.shape[:2] + (m.memory_cols - D - 1,))], -1)
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
    variants = [("trained model (reference)", None), ("edit, gamma 0 (must equal the reference)", (0.0, "bank"))]
    for mode in ("bank", "address"):
        for g in (0.5, 1.0, 2.0, 4.0):
            variants.append((f"per-{mode} redundancy, gamma {g}", (g, mode)))
    print(f"run {a.run_name}  checkpoint {a.checkpoint}  {a.rows} validation rows, no training")
    print(f"{'write':<42} {'val loss':>8} | {'Roxo whole':>10} {'Ro rank':>7} {'x rank':>6} {'o rank':>6} | "
          f"{'Roxy whole':>10} {'xy rank':>7} | {'Rilo R':>6}")
    for tag, v in variants:
        m._layer = orig if v is None else novelty_layer(m, *v)
        loss = val_loss(m, x)
        res = {nm: piece_probs(m, *encode(tok, m.reset_token, nm, nm)[:2]) for nm in NAMES}
        whole = lambda nm: float(np.prod([r[0] for r in res[nm]]))
        ro, ry = res[" Roxo"], res[" Roxy"]
        print(f"{tag:<42} {loss:8.4f} | {whole(' Roxo'):10.2e} {ro[0][1]:7d} {ro[1][1]:6d} {ro[2][1]:6d} | "
              f"{whole(' Roxy'):10.2e} {ry[1][1]:7d} | {res[' Rilo'][0][1]:6d}")
    m._layer = orig


if __name__ == "__main__":
    main()
