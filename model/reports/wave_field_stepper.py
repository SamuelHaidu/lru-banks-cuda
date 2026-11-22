"""`model report wave-field-stepper`: step through a story with a `wave-field` run.

At step i the model reads token i and predicts token i+1 from the state of the window before it. Every context
token j writes into channel k of state space l the wave

    u_{j,l,k} = g_{i,l,k} w_{j,l,k} e^{(-alpha_{l,k} + i omega_k)(i - j)} / Z_{l,k}

as the reader hears it (w = the token's write, p_l (s P_l); g_{i,l} = the filter the reader's cascade built for
state space l), and the reading is r = sum_j u_j. Everything on the page is exact:

* Share of token j in what is heard: sum_{l,k} Re(conj(r) u_j) / sum |r|^2 (shares sum to 1; a negative share
  is destructive interference). The same with g = 1 gives the share before the filter.
* Votes: the readings are [Re r, Im r, |r|] and |r| = sum_j Re(conj(r) u_j) / |r|, so the readings split exactly
  into one part per token. After RMSNorm (a per-step scalar) the linear base path splits the logit into token
  votes and state-space votes; the bias and the GELU MLP are their own components. The parts sum to the logit.

The wave quantities (spectra, phasors, wavelets, spectrogram) are recomputed in the page from the exported writes
of the story's tokens and the global parameters (channel frequencies and decays); the filters come per step, and
the per-step heard energy is exported as a check. Offline page, Tailwind CSS inlined
(`templates/report_tailwind.css`, built by `tools/report-css`).
"""

ARCHS = ('wave-field',)

import argparse
import html as _html
import json
import math
from pathlib import Path
from typing import List

from model.reports._model_io import load_full_model, load_tokenizer
from model.reports.token_kernel_stepper import _story


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="model report wave-field-stepper",
                                description="Waves, cascade filters, shares and votes per step of a wave-field run.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--text", default=None, help="Text to step through (default: a validation story).")
    p.add_argument("--min-tokens", type=int, default=220)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--doc-skip", type=int, default=0)
    return p


def _val(v):
    return getattr(v, "value", v)


def main(argv: List[str]) -> None:
    args = build_parser().parse_args(argv)
    import torch

    model, config, ckpt = load_full_model(args.run_name, args.checkpoint)
    if config["arch"] != "wave-field":
        raise SystemExit(f"Run {args.run_name!r} is {config['arch']!r}, not wave-field.")
    tok = load_tokenizer(config)
    ids = _story(config, tok, args)
    N, T, K, L = len(ids), model.window_size, model.channels, model.state_spaces
    label = lambda i: (tok.id_to_token(int(i)) or f"<{i}>").replace("Ġ", " ").replace("Ċ", "\\n")
    words = [label(i) for i in ids]
    dev = _val(model.signature.embeddings).device
    steps = N - 1
    story = torch.as_tensor(ids, device=dev)
    pos = torch.arange(steps, device=dev)[:, None] - (T - 1) + torch.arange(T, device=dev)[None]  # story position of each slot
    x = torch.where(pos >= 0, story[pos.clamp(min=0)], torch.zeros_like(pos)).int()  # (steps, T), left padding = id 0

    with torch.no_grad():
        E = _val(model.signature.embeddings).float()  # (V, 2K) signatures
        Ep = torch.einsum("vd,lde->vle", E, _val(model.space_proj).float())  # (V, L, 2K) what each state space receives
        mix = torch.softmax(_val(model.space_mix.embeddings).float(), -1)  # (V, L)
        writes = mix[..., None] * torch.complex(Ep[..., :K].contiguous(), Ep[..., K:].contiguous())  # (V, L, K)
        sig = torch.complex(E[:, :K], E[:, K:])  # (V, K)
        alpha = torch.nn.functional.softplus(_val(model.decay_raw).float())  # (L, K)
        omega = _val(model.channel_omega).float()
        dist = torch.arange(T, device=dev, dtype=torch.float32)
        decay = torch.exp(-alpha[..., None] * dist)
        decay = decay / decay.sum(-1, keepdim=True)  # (L, K, T)
        kern_slot = torch.flip(decay * torch.exp(1j * omega[:, None] * dist)[None], [-1])  # slot t -> distance T-1-t
        gamma = _val(model.feature_norm.scale).float()
        eps = model.feature_norm.epsilon
        G = gamma[:, None] * _val(model.base.kernel).float()  # (F, dim)
        bb = _val(model.base.bias).float()
        Wout = _val(model.readout.kernel).float()  # (dim, V)
        En = E / (E.norm(dim=-1, keepdim=True) + 1e-9)

        top_l, cand_l, cv_l, tv_l, en_l, p_l, g_l, q_l = [], [], [], [], [], [], [], []
        max_err = 0.0
        for a in range(0, steps, 32):
            xb = x[a:a + 32].long()
            B = xb.shape[0]
            h_model = model.hidden(xb).float()
            f = writes[xb] * kern_slot.permute(2, 0, 1)[None]  # (B, T, L, K) unfiltered arrivals
            S = f.sum(1)
            trace = []  # the model's own cascade over the state: (query, filter re, filter im) per state space
            model._read(xb[:, -1], S.real.contiguous(), S.imag.contiguous(), trace)
            g = torch.stack([torch.complex(gr.float(), gi.float()) for _, gr, gi in trace], 1)  # (B, L, K)
            q = torch.stack([qq.float() for qq, _, _ in trace], 1)  # (B, L, 2K)
            u = f * g[:, None]
            U = u.sum(1)  # (B, L, K)
            mag = torch.sqrt(U.real ** 2 + U.imag ** 2 + 1e-8)
            proj = (U.conj()[:, None] * u).real  # (B, T, L, K): Re(conj U u_j)
            hj = torch.stack([u.real, u.imag, proj / mag[:, None]], 3).reshape(B, T, -1)  # per-token readings, (L, 3, K)
            max_err = max(max_err, float((hj.sum(1) - h_model).abs().max() / h_model.abs().max()))
            s = torch.rsqrt(h_model.square().mean(-1, keepdim=True) + eps)  # (B, 1)
            z = model.feature_norm(h_model)
            logits = model.readout(model.readout_input(h_model)).float()
            probs = torch.softmax(logits, -1)
            top = probs.cpu().topk(5, -1).indices.to(dev)  # topk on CPU (XPU work-group limit)
            tgt = story[a + 1:a + 1 + B]
            cand = torch.cat([top, tgt[:, None]], 1)  # (B, 6)
            Wc = Wout[:, cand].permute(1, 0, 2)  # (B, dim, 6)
            tv = s[:, :, None] * torch.einsum("btd,bdc->btc", hj @ G, Wc)  # (B, T, 6) per-token linear votes
            hc = h_model.reshape(B, L, 3 * K)
            spaces = []
            for l in range(L):
                m = torch.zeros_like(hc)
                m[:, l] = hc[:, l]
                spaces.append(torch.einsum("bd,bdc->bc", s * (m.reshape(B, -1) @ G), Wc))
            bias = torch.einsum("d,bdc->bc", bb, Wc)
            mlp = torch.einsum("bd,bdc->bc", model.mlp_out(model.mlp_in(z)).float(), Wc)
            comp = torch.stack(spaces + [bias, mlp], 1)  # (B, L+2, 6)
            max_err = max(max_err, float((comp.sum(1) - torch.gather(logits, 1, cand)).abs().max() / logits.abs().max()))
            qs = (q / (q.norm(dim=-1, keepdim=True) + 1e-9)) @ En.T  # (B, L, V): what each stage's query resembles
            qs[..., 0] = -2
            top_l.append(top.cpu()); cand_l.append(cand.cpu()); cv_l.append(comp.cpu()); tv_l.append(tv.cpu())
            en_l.append((U.abs() ** 2).sum((1, 2)).cpu()); p_l.append(torch.gather(probs, 1, cand).cpu()); g_l.append(g.cpu())
            q_l.append(torch.stack(qs.cpu().topk(5, -1), -1))  # (B, L, 5, 2): [cos, id]
        print(f"reconstruction: max relative error {max_err:.2e}")
        top, cand, cv, tv, en, pc, gall, qall = (torch.cat(v) for v in (top_l, cand_l, cv_l, tv_l, en_l, p_l, g_l, q_l))

        r5 = lambda t: [float(f"{v:.5g}") for v in t.reshape(-1).tolist()]
        vocab = {str(v): {"w": label(v), "cr": r5(Ep[v, :, :K]), "ci": r5(Ep[v, :, K:]), "p": r5(mix[v]),  # cr/ci: (L, K)
                          "sr": r5(sig[v].real), "si": r5(sig[v].imag)} for v in sorted(set(ids) | {0})}
        params = {"omega": r5(omega), "alpha": r5(alpha), "T": T, "K": K, "L": L}

    r2 = lambda t: [round(float(v), 2) for v in t.tolist()]
    r4 = lambda t: [float(f"{v:.4g}") for v in t.reshape(-1).tolist()]
    data = []
    for i in range(steps):
        n_real = min(i + 1, T)
        data.append({
            "top": [[label(t), round(float(p), 4)] for t, p in zip(top[i].tolist(), pc[i, :5].tolist())],
            "tgt": [label(ids[i + 1]), round(float(pc[i, 5]), 4)],
            "cand": [label(t) for t in cand[i].tolist()],
            "cv": [r2(cv[i, :, c]) for c in range(6)],  # [cand][component]
            "tv": [r2(tv[i, T - n_real:, c]) for c in range(6)],  # [cand][real token]
            "tvpad": [round(float(tv[i, :T - n_real, c].sum()), 2) for c in range(6)],
            "en": float(f"{float(en[i]):.5g}"),
            "g": [r4(gall[i].real), r4(gall[i].imag)],  # the cascade's filter per state space, (L, K) flattened
            "q": [[[label(int(qall[i, l, n, 1])), round(float(qall[i, l, n, 0]), 2)] for n in range(5)] for l in range(L)],
        })
    half = [round(math.log(2) / (sum(alpha[l].tolist()) / K), 2) for l in range(L)]  # state space half-life (mean rate)
    payload = json.dumps({"words": words, "ids": ids, "steps": data, "vocab": vocab, "params": params, "half": half,
                          "comp": [f"state space {l + 1} (half-life {half[l]})" for l in range(L)] + ["bias", "GELU MLP"]},
                         separators=(",", ":"))
    out = (ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent) / "reports" / "wave_field_stepper.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    templates = Path(__file__).resolve().parent / "templates"
    page = (templates / "wave_field_stepper.html").read_text()
    out.write_text(page.replace("/*__CSS__*/", (templates / "report_tailwind.css").read_text())
                   .replace("__TITLE__", _html.escape(f"{args.run_name} / {ckpt.name} - {L} state spaces x {K} channels, {N} tokens"))
                   .replace("/*__DATA__*/null", payload))
    print(f"Wrote {out}  ({N} tokens, window {T}, {len(payload) / 1e6:.1f} MB data)")
