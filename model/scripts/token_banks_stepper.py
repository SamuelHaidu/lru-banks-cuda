"""`model scripts token-banks-stepper`: step through a story with a `rope-token-banks` run.

Same story and controls as `token-kernel-stepper`. For each of the 5 states (the general
plain-kernel state and the 4 banks) the page shows, at the current step, the window text
shaded by that state's energy (share |w_j| / sum |w|) and a bar chart of the energy per
context token. Each bank panel's title gives the router's weight for that bank.

The top "combined" panel shows each token's share of the *final* vector: every state is a
normalized weighted average times a learned gain, so token j's coefficient in
`h = [g] + sum_b pi_b m_b` is `[gain_0 e_0j] + sum_b pi_b gain_b e_bj` (e = its energy in that
state); the stacked bars show which state each token's share comes from.

"Votes": the readout is linear, so every logit splits exactly into the votes of the general
state, each routed bank and the MLP block. For the top predictions and the true next token
the page shows that split, and which context tokens voted for (or against) a chosen word.
"Copy": from the weights alone, whether a token at distance k votes for itself.

The page (`templates/token_banks_stepper.html`) is offline: Tailwind CSS from
`templates/report_tailwind.css` (built by `tools/report-css`, `npm run build`) is inlined.
"""

import argparse
import html as _html
import json
from pathlib import Path
from typing import List

import numpy as np

from model.scripts._model_io import load_full_model, load_tokenizer
from model.scripts.token_kernel_stepper import _story


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="model scripts token-banks-stepper",
                                description="Per-step energy of the general state and each bank of a rope-token-banks run.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--text", default=None, help="Text to step through (default: a validation story).")
    p.add_argument("--min-tokens", type=int, default=220)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--doc-skip", type=int, default=0)
    return p


def main(argv: List[str]) -> None:
    args = build_parser().parse_args(argv)
    import torch
    from keras import ops

    model, config, ckpt = load_full_model(args.run_name, args.checkpoint)
    if config["arch"] != "rope-token-banks":
        raise SystemExit(f"Run {args.run_name!r} is {config['arch']!r}, not rope-token-banks.")
    tok = load_tokenizer(config)
    ids = _story(config, tok, args)
    N, T, C, nb = len(ids), model.window_size, model.context_length, model.num_banks
    NM = int(getattr(model, "memory_heads", 0) or 0)
    label = lambda i: (tok.id_to_token(int(i)) or f"<{i}>").replace("Ġ", " ").replace("Ċ", "\\n")
    words = [label(i) for i in ids]

    steps = N - 1
    x = np.zeros((steps, C), np.int32)  # full context (255 with memories); banks use the last T
    for i in range(steps):
        ctx = ids[max(0, i + 1 - C): i + 1]
        x[i, C - len(ctx):] = ctx
    dev = model.token_embedding.embeddings.value.device
    xt = torch.as_tensor(x, device=dev)
    xT = xt[:, -T:]
    to_np = lambda t: np.asarray(ops.convert_to_numpy(t), dtype=np.float64)
    chunk = 64
    with torch.no_grad():
        z = torch.cat([model.hidden(xt[a:a + chunk]) for a in range(0, steps, chunk)])  # [h ; reads]
        logits = to_np(model.readout(model.readout_input(z)))
        pi = to_np(model.bank_mix(xT))  # (steps, banks)
        idsx = ops.cast(xT, "int32")
        w0 = np.abs(to_np(model.general_kernel(xT.float())))  # (T,) or (D, T) with one kernel per RoPE pair
        if w0.ndim == 2:  # energy of a position: its normalized |weight| averaged over channels
            w0 = (w0 / w0.sum(-1, keepdims=True)).mean(0)
        gain = np.exp(to_np(model.log_gain))
        s, lam = model.bank_numbers(idsx)
        log_k = ops.convert_to_tensor(model._log_distance)
        wb = to_np(s * model.bank_kernel(ops.exp(lam * log_k)))  # (banks, steps, T)
        lam_tok = to_np(ops.sigmoid(ops.take(model.bank_lambda, torch.as_tensor(ids, device=dev), axis=1)))  # (banks, N)
    weights = np.concatenate([np.broadcast_to(np.abs(w0), (1, steps, T)), np.abs(wb)], axis=0)  # (5, steps, T)
    energy = weights / (weights.sum(-1, keepdims=True) + 1e-12)
    # Share of each (state, position) in the final vector h = [g] + sum_b pi_b m_b.
    scale = np.concatenate([np.full((1, steps), gain[0] if getattr(model, "general_in_output", False) else 0.0),
                            (pi * gain[1:][None]).T], axis=0)  # (5, steps)
    contrib = energy * scale[..., None]
    contrib = contrib / (contrib.sum(axis=(0, 2), keepdims=True) + 1e-12)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    top = np.argsort(-probs, axis=-1)[:, :5]
    cand = np.concatenate([top, np.asarray(ids[1:])[:, None]], axis=1)  # (steps, 6): top-5 + target
    comp_votes, tok_votes = _votes(model, xT, torch.as_tensor(cand, device=dev), z)
    copy_rows = _copy_table(model, tok)
    mem = _memory(model, xt, ids, label, z[:, model.dim:].mean(0)) if NM else None

    q4 = lambda a: [int(round(float(v) * 1e4)) for v in a]  # shares as integers / 10^4: a much smaller page
    q2 = lambda a: [round(float(v), 2) for v in a]
    data = []
    for i in range(steps):
        n_real = min(i + 1, T)
        cols = slice(T - n_real, T)
        st = {
            "first": i + 1 - n_real,
            "e": [q4(energy[b, i, cols]) for b in range(nb + 1)],
            "pad": [round(float(energy[b, i, : T - n_real].sum()), 4) for b in range(nb + 1)],
            "c": [q4(contrib[b, i, cols]) for b in range(nb + 1)],
            "cpad": round(float(contrib[:, i, : T - n_real].sum()), 4),
            "pi": [None] + [round(float(v), 4) for v in pi[i]],  # general: router input, no weight
            "top": [[label(t), round(float(probs[i, t]), 4)] for t in top[i]],
            "tgt": [label(ids[i + 1]), round(float(probs[i, ids[i + 1]]), 4)],
            "cand": [label(t) for t in cand[i]],
            "cv": [q2(comp_votes[i, :, c]) for c in range(cand.shape[1])],  # [cand][component]
            "tv": [q2(tok_votes[i, cols, c]) for c in range(cand.shape[1])],  # [cand][token]
        }
        if mem is not None:
            st["m"] = mem[i]
        data.append(st)
    gen_kind = "64 per-pair kernels" if getattr(model, "general_pair_kernels", False) else "plain kernel"
    gen_role = "router input + added to output" if getattr(model, "general_in_output", False) else "router input"
    names = [f"general ({gen_kind}, {gen_role})"] + [f"bank {b + 1} (lambda init {model.bank_lambdas[b]})" for b in range(nb)]
    comp_names = ["general"] + [f"bank {b + 1}" for b in range(nb)] + (["MLP block" + (" (+ memory reads)" if NM else "")] if getattr(model, "memory_hidden", None) else [])
    payload = json.dumps({"words": words, "steps": data, "names": names, "comp": comp_names, "copy": copy_rows,
                          "lam": [q2(lam_tok[b]) for b in range(nb)], "nm": NM,
                          "mwin": int(getattr(model, "memory_window", 0) or 0), "mrange": int(getattr(model, "memory_value_range", 0) or 0)},
                         separators=(",", ":"))
    out = ckpt.parent.parent / "reports" / "token_banks_stepper.html" if ckpt.parent.name == "checkpoints" \
        else ckpt.parent / "reports" / "token_banks_stepper.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    templates = Path(__file__).resolve().parent / "templates"
    page = (templates / "token_banks_stepper.html").read_text()
    out.write_text(page.replace("/*__CSS__*/", (templates / "report_tailwind.css").read_text())
                   .replace("__TITLE__", _html.escape(f"{args.run_name} / {ckpt.name} - banks{' + memories' if NM else ''}, {N} tokens"))
                   .replace("/*__DATA__*/null", payload))
    print(f"Wrote {out}  ({N} tokens, window {T}, context {C}, {nb} banks, {NM} memories, {len(payload) / 1e6:.1f} MB data)")


def _memory(model, xt, ids, label, read_mean, n_top: int = 3, n_dec: int = 5):
    """Per step and memory: what the read retrieved and from which writes.

    The read is `r = sum_a w_a e^{i(phi_a - psi)} v_a` with `w_a = g_a / sum g` (gate-weighted
    average over the window), so it splits exactly into one term per write; a write's *share*
    is that term's projection on r, `<c_a, r> / |r|^2` (shares sum to 1). Key match is
    `mean cos(phi_a - psi)`: 1 = the read key equals that write key. The read is also decoded
    against the stored form `R_k P_n e_v` of every token in the window (best lag k; both centered on the vocabulary
    mean, which otherwise dominates the cosine), and each memory's effect on the target is
    measured by replacing its read with its mean over the story (mean ablation).
    """
    import torch

    T, C, NM, D = model.window_size, model.context_length, model.memory_heads, model.dim
    half, R = D // 2, model.memory_value_range
    steps = xt.shape[0]
    tgt = torch.as_tensor(ids[1:], device=xt.device)
    E = model.token_embedding.embeddings.value
    inv = 1.0 / (model.rope_base ** (torch.arange(half, dtype=torch.float64) / half))
    rot = lambda v, k: torch.cat([v[..., :half] * torch.cos(k * inv).float().to(v.device) - v[..., half:] * torch.sin(k * inv).float().to(v.device),
                                  v[..., :half] * torch.sin(k * inv).float().to(v.device) + v[..., half:] * torch.cos(k * inv).float().to(v.device)], -1)
    with torch.no_grad():
        # Every token's stored form at each lag, unit-normalized: (NM, R, V, D)
        PE = torch.einsum("vd,nde->nve", E, model.mem_value.value)
        dic = torch.stack([torch.stack([rot(PE[n], k) for k in range(1, R + 1)]) for n in range(NM)])
        dic_mean = dic.mean(2)  # (NM, R, D)
        dic = dic - dic_mean[:, :, None]
        dic = dic / (dic.norm(dim=-1, keepdim=True) + 1e-9)
    out = []
    for a0 in range(0, steps, 64):
        xb = xt[a0:a0 + 64]
        B = xb.shape[0]
        with torch.no_grad():
            p = model.memory_trace(xb)
            g = p["gate"]  # (B, Qb, NM)
            w = g / g.sum(1, keepdim=True)
            d = p["phi"] - p["psi"]  # (B, Qb, NM, half)
            vr, vi = p["value"][..., :half], p["value"][..., half:]
            cr = w[..., None] * (torch.cos(d) * vr - torch.sin(d) * vi)
            ci = w[..., None] * (torch.sin(d) * vr + torch.cos(d) * vi)
            r = p["read"][:, 0]  # (B, NM, D)
            share = (torch.einsum("bqnh,bnh->bqn", cr, r[..., :half]) + torch.einsum("bqnh,bnh->bqn", ci, r[..., half:])) \
                / (r.square().sum(-1)[:, None] + 1e-12)
            match = torch.cos(d).mean(-1)  # (B, Qb, NM)
            kern = p["kern"]  # (B, Qb, NM, R)
            kern = kern / (kern.abs().sum(-1, keepdim=True) + 1e-9)
            rc = r[:, :, None] - dic_mean[None]  # (B, NM, R, D): centered per lag
            rc = rc / (rc.norm(dim=-1, keepdim=True) + 1e-9)
            sims = torch.einsum("bnkd,nkvd->bnkv", rc, dic)  # (B, NM, R, V)
            best, lag = (t.cpu() for t in sims.max(2))  # (B, NM, V); topk on CPU (XPU work-group limit)
            # Only tokens in the window can have been written: rank those (rare, barely trained
            # embeddings otherwise crowd the top of the list).
            seen = torch.zeros(best.shape[0], best.shape[-1], dtype=torch.bool)
            seen.scatter_(1, xb.cpu().long(), True)
            seen[:, 0] = False  # padding
            best = best.masked_fill(~seen[:, None], -2.0)
            dv, di = best.topk(n_dec, -1)
            dlag = torch.gather(lag, 2, di)
            # effect of each read on log p(target): full model vs that read (or all reads) removed
            z = p["z"]
            lp = lambda zz: torch.log_softmax(model.readout(model.readout_input(zz)), -1)
            tb = tgt[a0:a0 + B]
            full = lp(z)
            rows = torch.arange(B, device=z.device)
            base = full[rows, tb]
            z0 = z.clone(); z0[:, D:] = read_mean
            no_all = lp(z0)
            effects = [base - no_all[rows, tb]]
            for n in range(NM):
                zn = z.clone(); zn[:, D * (1 + n): D * (2 + n)] = read_mean[D * n: D * (n + 1)]
                effects.append(base - lp(zn)[rows, tb])
            eff = torch.stack(effects, 1)  # (B, 1 + NM)
            top_no = no_all.argmax(-1)
        g, share, match, kern = (t.cpu().double().numpy() for t in (g, share, match, kern))
        pu, pb = p["pi_unbind"][:, 0].cpu().numpy(), p["pi_bind"].cpu().numpy()
        dv, di, dlag, eff, top_no = dv.cpu().numpy(), di.cpu().numpy(), dlag.cpu().numpy(), eff.cpu().numpy(), top_no.cpu().numpy()
        Qb = g.shape[1]
        for bi in range(B):
            i = a0 + bi
            nw = min(Qb, i + 1)  # writes on real tokens: write a is at story position i - (Qb-1) + a
            cols = slice(Qb - nw, Qb)
            first = i + 1 - nw
            mems = []
            for n in range(NM):
                sh = share[bi, cols, n]
                order = np.argsort(-sh)[:n_top]
                mems.append({
                    "g": [int(round(v * 1000)) for v in g[bi, cols, n]],
                    "s": [int(round(v * 1000)) for v in sh],
                    "k": [int(round(v * 100)) for v in match[bi, cols, n]],
                    "tw": [[int(first + j), [int(round(v * 100)) for v in kern[bi, Qb - nw + j, n]],
                            [round(float(v), 2) for v in pb[bi, Qb - nw + j]]] for j in order],
                    "dec": [[label(t), round(float(c), 2), int(l) + 1] for t, c, l in zip(di[bi, n], dv[bi, n], dlag[bi, n]) if c > -1.5],
                    "pu": [round(float(v), 3) for v in pu[bi]],
                    "eff": round(float(eff[bi, 1 + n]), 3),
                })
            out.append({"first": first, "mem": mems, "eff": round(float(eff[bi, 0]), 3), "no": label(top_no[bi])})
    return out


def _rotated(model, xt):
    """Context tokens rotated by distance, as in `RopeTokenBanksLM.states`: `(general (S, T, D),
    banks (banks or 1, S, T, D))` -- each state's own view with `bank_projections`, else shared."""
    from keras import ops

    ids = ops.cast(xt, "int32")
    x0, xb = model._state_tokens(model.token_embedding(ids), ids)
    half = model.dim // 2
    cos = ops.convert_to_tensor(model._cos, dtype=x0.dtype)
    sin = ops.convert_to_tensor(model._sin, dtype=x0.dtype)
    rot = lambda v: ops.concatenate([v[..., :half] * cos - v[..., half:] * sin, v[..., :half] * sin + v[..., half:] * cos], axis=-1)
    return rot(x0), rot(xb)


def _votes(model, xt, cand, z=None):
    """Exact split of the candidates' logits.

    Returns `comp (steps, components, C)`: votes of the general state (if it is in the output),
    each routed bank and the MLP block, summing to the logit; and `tok (steps, T, C)`: each
    context token's vote through the linear part (general + banks).
    """
    import torch
    from keras import ops

    with torch.no_grad():
        r0, rb = _rotated(model, xt)  # (S, T, D), (banks or 1, S, T, D)
        rb = rb.expand(model.num_banks, -1, -1, -1)
        gain = torch.exp(model.log_gain.value)
        w0 = model.general_kernel(r0)  # (T,) or (D, T)
        coef0 = (w0 / w0.abs().sum(-1, keepdim=True)) * gain[0]
        coef0 = coef0 if coef0.dim() == 2 else coef0[None].expand(model.dim, -1)  # (D, T)
        if not getattr(model, "general_in_output", False):
            coef0 = torch.zeros_like(coef0)
        s, lam = model.bank_numbers(ops.cast(xt, "int32"))
        wb = s * model.bank_kernel(ops.exp(lam * ops.convert_to_tensor(model._log_distance)))  # (nb, S, T)
        general, banks = model.states(xt)
        pi = torch.softmax(model.router(general), -1)  # (S, nb)
        coefb = wb / wb.abs().sum(-1, keepdim=True) * gain[1:, None, None] * pi.T[:, :, None]  # (nb, S, T)
        K = model.readout.kernel
        Wsel = getattr(K, "value", K)[:, cand].permute(1, 0, 2)  # (S, D, C); a tensor when the readout is tied
        g_vec = torch.einsum("dt,std->sd", coef0, r0)
        b_vec = torch.einsum("nst,nstd->nsd", coefb, rb)  # (nb, S, D)
        comps = [g_vec] + list(b_vec)
        h = g_vec + b_vec.sum(0)
        if getattr(model, "memory_hidden", None):  # the block's vote: readout input minus its residual part
            zz = z if z is not None else model.hidden(xt)  # [h ; (general) ; (reads)] in the model's own layout
            comps.append(model.readout_input(zz) - model.residual_base(zz))
        comp = torch.stack([torch.einsum("sd,sdc->sc", v, Wsel) for v in comps], dim=1)  # (S, comp, C)
        vec_tok = coef0.T[None] * r0 + torch.einsum("nst,nstd->std", coefb, rb)  # (S, T, D): each token's part of h
        tok = torch.einsum("std,sdc->stc", vec_tok, Wsel)
    return comp.cpu().double().numpy(), tok.cpu().double().numpy()


def _copy_table(model, tok, ks=(1, 2, 4, 8, 16, 32, 64, 128)):
    """From the weights: does token v at distance k vote for itself? Rank of v's own logit
    among all 10k logits that `R_k e_v` produces through the readout, by word group."""
    import torch

    from model.scripts.token_kernel_report import FUNCTION_WORDS

    V, D = model.vocabulary_size, model.dim
    raw = [(tok.id_to_token(i) or "") for i in range(V)]
    text = [t.replace("Ġ", " ") for t in raw]
    def group(i):
        t = text[i].strip()
        if not t or not t.isalpha():
            return None
        if t.lower() in FUNCTION_WORDS:
            return "function words"
        if text[i].startswith(" ") and t[0].isupper() and t[1:].islower() and len(t) > 2:
            return "capitalized (names...)"
        return "other words" if text[i].startswith(" ") and len(t) > 2 else None
    groups = {}
    for i in range(V):
        g = group(i)
        if g:
            groups.setdefault(g, []).append(i)
    E = model.token_embedding.embeddings.value
    Wout = getattr(model.readout.kernel, "value", model.readout.kernel)
    half = D // 2
    inv = 1.0 / (model.rope_base ** (torch.arange(half, dtype=torch.float64) / half))
    rows = []
    with torch.no_grad():
        for k in ks:
            c, s_ = torch.cos(k * inv).float().to(E.device), torch.sin(k * inv).float().to(E.device)
            r = torch.cat([E[:, :half] * c - E[:, half:] * s_, E[:, :half] * s_ + E[:, half:] * c], -1)  # (V, D)
            ranks = torch.empty(V, device=E.device)
            for a in range(0, V, 2000):
                lg = r[a:a + 2000] @ Wout  # (chunk, V)
                own = lg[torch.arange(lg.shape[0]), torch.arange(a, a + lg.shape[0])]
                ranks[a:a + 2000] = (lg > own[:, None]).sum(-1).float()  # 0 = own token is the top vote
            rk = ranks.cpu().numpy()
            row = {"k": k}
            for g, idx in groups.items():
                row[g] = [round(float(np.median(rk[idx])), 0), round(float((rk[idx] < 10).mean() * 100), 1)]
            rows.append(row)
    return {"groups": list(groups), "rows": rows}
