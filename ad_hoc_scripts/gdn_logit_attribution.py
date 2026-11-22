"""Why did a `gated-deltanet` run choose token B over token A at one position? Direct logit attribution, exact:

    logit(A) - logit(B) = x_final . d,   d = final_norm * (W[:, A] - W[:, B]) / rms(x_final)
    x_final = embedding + sum_n (mixer_n + moe_n)                 each layer's vote, in logits
    mixer_n = sum_h (rms_norm(o_h) * o_norm * gate_h) W_o[h rows]  each head's vote
    o_h     = sum_i c_i v_i  (delta_rule_read_weights)             split by the past position i whose value it is

With the position's norms fixed (rms of the final stream, of each head's read), every split is linear, so the votes
add up to the logit difference (checked, printed). The MoE vote is per layer (with the chosen expert). A value v_i
already mixes token i with the 3 before it (the short conv): the past tokens are shown with that window.

    uv run python ad_hoc_scripts/gdn_logit_attribution.py --run-name X --text-file F
        (--pos N | --at TOKEN [--occurrence -1]) [--a TOKEN] [--b TOKEN] [--heads 8] [--tokens 6]

--at: the position of the last (or --occurrence k-th) token equal to TOKEN (the model predicts what follows it).
--a: default the actual next token; --b: default the model's top choice if it is not A, else its second.
"""

import argparse
import re

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

ap = argparse.ArgumentParser()
ap.add_argument("--run-name", required=True)
ap.add_argument("--checkpoint", default="latest")
ap.add_argument("--text-file", required=True)
ap.add_argument("--pos", type=int, default=None)
ap.add_argument("--at", default=None)
ap.add_argument("--occurrence", type=int, default=-1)
ap.add_argument("--a", default=None)
ap.add_argument("--b", default=None)
ap.add_argument("--heads", type=int, default=8, help="heads to break down by past position")
ap.add_argument("--tokens", type=int, default=6, help="past positions per head")
ap.add_argument("--device", default="xpu")
a = ap.parse_args()
setup_device(a.device)

import torch  # noqa: E402

from model import config as cfg_mod  # noqa: E402
import model.archs  # noqa: E402,F401
import model.archs.gated_deltanet as gd  # noqa: E402
from model.archs.delta_rule import delta_rule_read_weights  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402


def rms(x, eps=1e-6):
    return torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps)


def main():
    config = cfg_mod.load(a.run_name)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(cfg_mod.tokenizer_path(config["data"]["dataset"])))
    m = load_checkpoint(a.run_name, a.checkpoint)
    text = open(a.text_file).read()
    ids = tok.encode(text).ids
    if not ids or ids[0] != m.reset_token:
        ids = [m.reset_token] + ids
    piece = lambda i: tok.decode([i], skip_special_tokens=False)
    pieces = [piece(i) for i in ids]
    if a.pos is not None:
        t = a.pos
    else:
        hits = [j for j, p in enumerate(pieces) if p == a.at]
        if not hits:
            raise SystemExit(f"no token {a.at!r}; tokens near the end: {pieces[-30:]}")
        t = hits[a.occurrence]
    tok_id = lambda s: next((i for i in range(m.vocabulary_size) if piece(i) == s), None)

    rec, layer_io = [], []
    orig_rule = gd.gated_delta_rule

    def rule(q, k, v, g, beta, reset, initial=None, need_state=False):
        o, S = orig_rule(q, k, v, g, beta, reset, initial, need_state)
        rec.append([x[0].detach().double().cpu() for x in (q, k, v, g, beta, o)])
        return o, S

    def layer(n, x, *args):
        out = type(m)._layer(m, n, x, *args)
        layer_io.append((x[0].detach().double().cpu(), out[0][0].detach().double().cpu()))
        return out

    with torch.no_grad():
        gd.gated_delta_rule, m._layer = rule, layer
        try:
            m.dense_hidden(np.asarray([ids], np.int32))
        finally:
            gd.gated_delta_rule = orig_rule
            del m._layer
        P = [{k: v.value.detach().double().cpu() for k, v in p.items()} for p in m._params]
        W = m.readout.kernel.value.detach().double().cpu()
        g_final = m.final_norm.value.detach().double().cpu()
    x_final = layer_io[-1][1][t]
    logits = (x_final / rms(x_final)) * g_final @ W
    order = torch.argsort(logits, descending=True).tolist()
    A = tok_id(a.a) if a.a else (ids[t + 1] if t + 1 < len(ids) else order[0])
    B = tok_id(a.b) if a.b else (order[0] if order[0] != A else order[1])
    p = torch.softmax(logits, -1)
    d = g_final * (W[:, A] - W[:, B]) / rms(x_final)
    ctx = "".join(pieces[max(0, t - 40):t + 1])
    print(f"position {t}: ...{ctx!r}\nA = {piece(A)!r} (p {p[A]:.3f}, rank {order.index(A) + 1})   "
          f"B = {piece(B)!r} (p {p[B]:.3f}, rank {order.index(B) + 1})   logit(A) - logit(B) = {float(x_final @ d):+.3f}")

    H, Dh = m.num_heads, m.head_dim
    emb = layer_io[0][0][t]
    total = float(emb @ d)
    rows, head_votes = [("embedding", total, None, None)], []
    reset = torch.as_tensor(np.asarray(ids) == m.reset_token)
    for n, (x_in, x_out) in enumerate(layer_io):
        p_ = P[n]
        u = (x_in[t] / rms(x_in[t])) * p_["attn_norm"]
        gate = torch.nn.functional.silu(u @ p_["w_gate"])  # (H * Dh,)
        q, k, v, g, beta, o = rec[n]
        c, start = delta_rule_read_weights(q[None], k[None], g[None], beta[None], reset, t)  # (1, H, n)
        mixer = torch.zeros_like(emb)
        for h in range(H):
            sl = slice(h * Dh, (h + 1) * Dh)
            r = rms(o[t, h])
            scale = p_["o_norm"] * gate[sl] / r  # what multiplies the read before W_o
            proj = (scale[None] * v[start:t + 1, h]) @ p_["w_o"][sl] @ d  # (n,) a unit of each value, in logits
            per_tok = c[0, h] * proj
            mixer += (o[t, h] / r * p_["o_norm"] * gate[sl]) @ p_["w_o"][sl]
            head_votes.append((float(per_tok.sum()), n, h, start, per_tok, float((c[0, h].sum()))))
        moe = x_out[t] - x_in[t] - mixer
        h_moe = (x_in[t] + mixer)
        hm = (h_moe / rms(h_moe)) * p_["ffn_norm"]
        expert = int(torch.argmax(hm @ p_["router"] + p_["router_bias"]))
        gate_p = float(torch.softmax(hm @ p_["router"], -1)[expert])
        rows.append((f"layer {n}", float(mixer @ d), float(moe @ d), (expert, gate_p)))
        total += float(mixer @ d) + float(moe @ d)
    print(f"sum of the votes {total:+.3f} (must equal the logit difference)\n")
    print("layer       mixer    MoE (expert, gate)    running total   lens: p(A)  p(B)")
    run = rows[0][1]
    print(f"embedding                                    {run:+7.2f}")
    for n, (name, mx, mo, (e, gp)) in enumerate(rows[1:]):
        run += mx + mo
        xs = layer_io[n][1][t]
        lp = torch.softmax((xs / rms(xs)) * g_final @ W, -1)
        print(f"{name:9s} {mx:+7.2f}  {mo:+7.2f} (e{e}, {gp:.2f})           {run:+7.2f}        {lp[A]:.3f} {lp[B]:.3f}")
    print(f"\nheads with the largest votes (positive: for {piece(A)!r}, negative: for {piece(B)!r}); per past position "
          f"(its token, with the 3 before it in brackets): the vote, and the weight c_i")
    for vote, n, h, start, per_tok, _ in sorted(head_votes, key=lambda x: -abs(x[0]))[:a.heads]:
        top = torch.argsort(per_tok.abs(), descending=True)[:a.tokens].tolist()
        c_row = None
        print(f"  layer {n:2d} head {h}: {vote:+.2f}")
        for i in top:
            j = start + i
            win = "".join(pieces[max(0, j - 3):j])
            print(f"      pos {j:4d} [{win!r}] {pieces[j]!r}: {float(per_tok[i]):+.3f}")


if __name__ == "__main__":
    main()
