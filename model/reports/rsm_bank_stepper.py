"""`model report rsm-bank-stepper`: step through a story with a `rope-state-memory` run (`memory="banks"`).

For every text (random training documents, texts the model writes itself, the Roxy name-recall story, `--text`) it
exports, per layer and per token, the write softmax w_t over the banks, the read softmax r_t, the read gate g_t, and
the state's token shares, the write gate beta_t (if the run has one); plus every bank's learned decay mu_b. The page
rebuilds the memory from those exactly:

  bank_b(t) = sum_j c_jb(t) v_j,   z_b(t) = sum_j c_jb(t),   c_jb(t) = (1 - mu_b) mu_b^(t - j) beta_j w_jb

(j from the last <bos> to t), so the read o_t = sum_b r_tb bank_b / sum_b r_tb z_b is attention over past tokens with
weights a_tj = sum_b r_tb c_jb(t) / sum_b r_tb z_b (they sum to 1). The script checks this against the model's own
read before writing the page. Shown for the chosen token and layer:

* the read as attention over past tokens, the gate, and which banks it read from (r_tb z_b share);
* every bank's write and read weight at this token (click one), and a banks x tokens heatmap of the whole text;
* the selected bank: what it stores (each past token's share c_jb / z_b), its writes, mass and reads over the text;
* the state s_t: each token's share of it (its rotated, decayed write projected on s_t).

CPU only (does not compete with a training run). Offline page, Tailwind CSS inlined.
"""

ARCHS = ('rope-state-memory',)

import argparse
import html as _html
import json
from pathlib import Path
from typing import List

import numpy as np

from model.reports._model_io import load_tokenizer

# agent datasets (`datasets/build_agent_shell.py`, bytes or BPE): conversations in the dataset's own format, each
# ending in a recall (a code from a list in the message; a file line from an earlier tool result)
AGENT_TEXTS = [
    ("agent recall: code from a list in the message",
     "<bos><tools>\nrun(command: str): run a shell command in the workspace and return its output\n"
     "read_file(path: str): return the contents of a file\n</tools><user>Codes:\nRoxo: XK-4821\nTamive: PB-9930\n"
     "Kolu: RM-1207\nNesida: DA-5512\nWhat is Kolu's code?</user><assistant>Kolu's code is RM-1207.</assistant><eos>"),
    ("agent recall: file line from an earlier tool result",
     "<bos><tools>\nread_file(path: str): return the contents of a file\n"
     "calculator(expression: str): evaluate an arithmetic expression\n</tools><user>Is mango in fruits.txt?</user>"
     "<assistant>Let me read it.\n<call>read_file(path=\"fruits.txt\")</call></assistant><result>apple\nkiwi\nmango\n"
     "pear\nlime</result><assistant>Yes, mango is in fruits.txt (line 3).</assistant><user>What's on line 5 of "
     "fruits.txt?</user><assistant>Line 5 is lime.</assistant><eos>"),
]
STATE_WINDOW = 64  # the state's token shares are kept for the last 64 tokens (its half-life is a few tokens)

STORY = ("Once upon a time, there was a little dog named{A}.{A} liked to run and play in the park every day. "
         "One day,{A} found a big red ball under a tree. The ball was shiny and new.{A} was very happy. "
         "He took the ball home and showed it to his mom. His mom smiled and said, \"What a nice ball,{A}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="model report rsm-bank-stepper",
                                description="Bank memory of a rope-state-memory run: writes, reads, attention per token.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--text", default=None, help="An extra text to step through.")
    p.add_argument("--name", default=" Roxy", help="Name used in the built-in recall story.")
    p.add_argument("--documents", type=int, default=None, help="Training documents (random, seeded; 8, or 4 for bytes).")
    p.add_argument("--generated", type=int, default=2,
                   help="Texts the model writes: from <bos> (temperature 0.8, top-k 50); with loss spans (agent data), "
                        "replies to validation conversations prefilled up to their first <assistant> (greedy).")
    p.add_argument("--min-tokens", type=int, default=None, help="150, or 300 for a byte-level dataset.")
    p.add_argument("--max-tokens", type=int, default=None, help="400, or 1024 for a byte-level dataset.")
    p.add_argument("--seed", type=int, default=0)
    return p


def _prefilled_replies(m, config, tok, spans, n, seed):
    """`n` validation conversations, each cut right after its first span opening (`<assistant>`): the model
    continues greedily until the span closes (or 400 tokens). Returns (name, ids, {"gen": first generated index,
    "note": the reference reply}) entries."""
    import torch

    from model import config as cfg_mod
    from model.data import _document_bounds, load_tokens

    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
    path = str(cfg_mod.dataset_split_path(config["data"]["dataset"], "validation"))
    tokens = load_tokens(path)
    starts, ends = _document_bounds(path, tokens, int(meta["doc_token_id"]), True)
    open_id, close_id = int(spans[0][0]), int(spans[0][1])
    out = []
    for d in np.random.default_rng(seed + 1).permutation(len(starts)):
        doc = [int(t) for t in tokens[starts[d]:ends[d]]]
        if open_id not in doc or close_id not in doc:
            continue
        cut = doc.index(open_id) + 1
        ref = doc[cut: doc.index(close_id, cut) + 1]
        with torch.no_grad():
            gen = [int(t) for t in m.generate(doc[:cut], max_new_tokens=400, temperature=0.0).split()][cut:]
        if close_id in gen:
            gen = gen[: gen.index(close_id) + 1]
        out.append((f"prefilled validation conversation {len(out) + 1}: the model's reply", doc[:cut] + gen,
                    {"gen": cut, "note": "reference reply: " + tok.decode(ref, skip_special_tokens=False)}))
        if len(out) == n:
            break
    return out


def main(argv: List[str]) -> None:
    args = build_parser().parse_args(argv)
    from model import config as cfg_mod
    from model.device import setup_device
    from model.evaluate import load_checkpoint, resolve_checkpoint
    from model.reports.wave_arrow_stepper import _training_documents

    config = cfg_mod.load(args.run_name)
    if config["arch"] != "rope-state-memory":
        raise SystemExit(f"Run {args.run_name!r} is {config['arch']!r}, not rope-state-memory.")
    setup_device("cpu")
    import torch

    m = load_checkpoint(args.run_name, args.checkpoint)
    ckpt = resolve_checkpoint(args.run_name, args.checkpoint)
    setup_device("cpu")
    m.use_kernels = False
    if m.memory != "banks":
        raise SystemExit(f"Run {args.run_name!r} has memory={m.memory!r}; this page is for memory='banks'.")
    tok = load_tokenizer(config)
    def label(i):  # the token's text (a BPE token may hold several characters); a lone UTF-8 byte: its byte symbol
        t = tok.decode([int(i)], skip_special_tokens=False)
        if not t or "\ufffd" in t:
            t = (tok.id_to_token(int(i)) or f"<{i}>").replace("Ġ", " ").replace("Ċ", "\n")
        return t.replace("\n", "\\n")
    bos, S, NL, D = m.reset_token, m.num_banks, m.num_layers, m.dim

    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
    eos = int(meta["eos_token_id"])
    byte_level = "byte_offset" in meta  # tags + bytes: conversations, ~4x more tokens per text
    agent = "<assistant>" in meta.get("special_tokens", {})  # the agent conversations (bytes or BPE)
    for k, small, big in (("documents", 8, 4), ("min_tokens", 150, 300), ("max_tokens", 400, 1024)):
        if getattr(args, k) is None:
            setattr(args, k, big if byte_level else small)
    spans = meta.get("loss_spans")  # the model is trained to predict only inside these (e.g. the assistant's turns)
    if agent:
        texts = [(name, tok.encode(text).ids, {}) for name, text in AGENT_TEXTS]
    else:
        texts = [("name recall story " + repr(args.name), [bos] + tok.encode(STORY.replace("{A}", args.name)).ids, {})]
    if args.text is not None:
        ids = tok.encode(args.text).ids
        texts.append(("your text", (ids if ids and ids[0] == bos else [bos] + ids)[: args.max_tokens], {}))
    texts += [(f"training document {n + 1}", ids, {}) for n, ids in
              enumerate(_training_documents(config, args.documents, args.min_tokens, args.max_tokens, args.seed))]
    if spans:  # prefill: a validation conversation up to its first span opening, the model writes the reply
        texts += _prefilled_replies(m, config, tok, spans, args.generated, args.seed)
    else:
        for n in range(args.generated):
            np.random.seed(args.seed + n)  # `generate` samples with np.random
            with torch.no_grad():
                out = m.generate([bos], max_new_tokens=args.max_tokens - 1, temperature=0.8, top_k=50)
            ids = [int(t) for t in out.split()]
            if eos in ids[1:]:
                ids = ids[: ids.index(eos, 1) + 1]
            texts.append((f"generated text {n + 1}", ids, {"gen": 1}))

    r3 = lambda a: [float(f"{v:.3g}") for v in np.asarray(a).reshape(-1)]
    # softmax weights as ln(w) in units of 1e-3 (0.1% relative error at any size: sharp softmaxes leave banks
    # at 1e-9 that the read can still select, which a fixed step would round to 0)
    q5 = lambda a: np.rint(np.log(np.clip(np.asarray(a), 1e-300, None)) * 1e3).astype(np.int64).reshape(-1).tolist()
    TOP = 5
    E = m.token_embedding.embeddings.value
    half = D // 2
    theta = torch.as_tensor(m.rope_base ** (-np.arange(half, dtype=np.float64) / half))
    mus = [m.decays(n)[1].detach().double() for n in range(NL)]
    lams = [float(m.decays(n)[0].detach()) for n in range(NL)]
    vocab_values = []  # what each token would write as a value, per layer (normalized, for decoding the reads)
    docs = []
    from model.data import loss_target_mask

    for name, ids, info in texts:
        N = len(ids)
        # the tokens the model is trained to predict (all of them without loss spans)
        target = loss_target_mask("", np.asarray(ids), spans, bos, use_cache=False) if spans else np.ones(N, bool)
        with torch.no_grad():
            x_ids = torch.as_tensor([ids], dtype=torch.int32)
            reset = x_ids == bos
            x = torch.nn.functional.embedding(x_ids.long(), E)
            seg = np.cumsum(np.asarray(ids) == bos)  # document index of every token (the memory restarts at <bos>)
            same = torch.as_tensor(seg[:, None] == seg[None, :]) & torch.ones(N, N, dtype=torch.bool).tril()
            layers = []
            for n in range(NL):
                u = m._norm(m._w("in_norm", n), x) if m.pre_norm else x
                zero = (E.new_zeros(1, D), E.new_zeros(1, m.memory_rows, m.memory_cols))
                h, _, (s, o) = m._layer(n, u, reset, zero, collect=True, fresh=True)
                prev = torch.cat([zero[0][:, None], s[:, :-1]], 1)
                r = torch.softmax(s @ m._w("address_q", n).value, -1)[0].double()  # (N, S)
                prev0 = torch.where(reset[..., None], 0.0, prev)
                a_w = prev0 @ m._w("address_k", n).value
                g_in = prev0 if getattr(m, "write_gate_input", "logits") == "state" else a_w
                w = torch.softmax(a_w, -1)[0].double()
                # write gate beta_t (1 without one): every bank write of token t is beta_t w_tb
                beta = m._write_gate(n, g_in)[0, :, 0].double() if m.write_gate else torch.ones(N, dtype=torch.float64)
                g = m._gate(n, m.read_gate_in(s, s @ m._w("address_q", n).value))[0, :, 0].double()
                v = (u @ m._w("value", n).kernel.value)[0].double()  # (N, D)
                # exact attention: a_tj = sum_b r_tb (1 - mu_b) mu_b^(t-j) w_jb / sum_b r_tb z_b(t)
                mu = mus[n]
                d = (torch.arange(N)[:, None] - torch.arange(N)[None, :]).clamp_min(0).double()
                A = torch.zeros(N, N, dtype=torch.float64)
                for t in range(N):
                    c = (1 - mu) * mu[None] ** d[t, :, None] * w * beta[:, None] * same[t, :, None]  # (N, S) c_jb(t)
                    num = c @ r[t]
                    A[t] = num / num.sum().clamp_min(1e-12)  # the model clamps the mass at 1e-12 (memory nearly empty: the read shrinks)
                err = float(((A @ v) * g[:, None] - o[0].double()).abs().max() / o.abs().max().clamp_min(1e-12))
                if err > 1e-3:
                    raise SystemExit(f"attention decomposition does not match the model's read (rel. error {err:.2e})")
                # state shares: s_t = sum_j gamma lam^(t-j) R^(t-j) u_j, share_j = <s_t, c_j> / |s_t|^2
                lam, gam = lams[n], (1 - lams[n] ** 2) ** 0.5
                u64, s64 = u[0].double(), s[0].double()
                state = []
                for t in range(N):
                    j0 = int(np.flatnonzero(seg[: t + 1] == seg[t])[0])
                    dd = torch.arange(t - j0, -1, -1, dtype=torch.float64)  # distance t - j for j = j0..t
                    ang = dd[:, None] * theta[None]
                    uj = u64[j0: t + 1]
                    rot = torch.cat([uj[:, :half] * ang.cos() - uj[:, half:] * ang.sin(),
                                     uj[:, :half] * ang.sin() + uj[:, half:] * ang.cos()], -1)
                    cj = gam * lam ** dd[:, None] * rot
                    st = s64[t]
                    if float((cj.sum(0) - st).abs().max()) > 1e-3 * float(st.abs().max()) + 1e-5:
                        raise SystemExit("state decomposition does not match the model's state")
                    sh = (cj @ st) / (st @ st).clamp_min(1e-30)
                    state.append(np.rint(sh.numpy()[-STATE_WINDOW:] * 1000).astype(int).tolist())  # per mille, last tokens
                # the read decoded: tokens whose own value is nearest (cosine) to the ungated read
                if len(vocab_values) <= n:
                    ev = (m._norm(m._w("in_norm", n), E) if m.pre_norm else E) @ m._w("value", n).kernel.value
                    vocab_values.append(torch.nn.functional.normalize(ev.double(), dim=-1))
                read = torch.nn.functional.normalize(A @ v, dim=-1) @ vocab_values[n].T  # (N, V)
                cv, ci = read.topk(TOP, -1)
                extra = {}  # the keep / add gates of the "gates" mixes, per token
                if getattr(m, "read_mix", "add") == "gates":
                    extra["ka"] = r3(m._keep_state(n, m.read_gate_in(s, s @ m._w("address_q", n).value))[0, :, 0].double().numpy())
                if getattr(m, "mlp_residual", "input") == "gates":
                    w1, w2 = m._w("mlp_in", n), m._w("mlp_out", n)
                    zz = m._norm(m._w("mlp_norm", n), h)
                    mo = torch.nn.functional.gelu(zz @ w1.kernel.value + w1.bias.value, approximate="tanh") @ w2.kernel.value + w2.bias.value
                    a2, g2 = m._out_gates(n, mo)
                    extra["a2"], extra["g2"] = r3(a2[0, :, 0].double().numpy()), r3(g2[0, :, 0].double().numpy())
                # the page rebuilds the masses and read shares from the encoded values: check they still match
                wq, rq = (np.exp(np.asarray(q5(a.numpy()), dtype=np.float64) * 1e-3).reshape(N, S) for a in (w, r))
                bq = np.asarray([float(f"{v:.4g}") for v in beta.numpy()])
                mu_np, rs_np = mu.numpy(), reset[0].numpy()

                def read_shares(ws_, rs_, bt_):  # r_tb z_b / sum_b r_tb z_b, z the written mass (as the page does)
                    z, out = np.zeros(S), np.zeros((N, S))
                    for t in range(N):
                        z = (0.0 if rs_np[t] else mu_np * z) + (1 - mu_np) * ws_[t] * bt_[t]
                        out[t] = rs_[t] * z / max(float(rs_[t] @ z), 1e-300)
                    return out
                share_err = np.abs(read_shares(w.numpy(), r.numpy(), beta.numpy()) - read_shares(wq, rq, bq)).max()
                if share_err > 1e-2:
                    raise SystemExit(f"layer {n + 1}: the page's read shares are off by {share_err:.3f} after encoding")
                layers.append({**extra, "w": q5(w.numpy()), "r": q5(r.numpy()), "g": r3(g.numpy()), "state": state,
                               "wg": bq.tolist(),
                               "dec": [[label(t) for t in row] for row in ci.numpy()], "decv": r3(cv.numpy())})
                x = m._mlp(n, h, h if m.mlp_residual == "state" else x)
            logits = torch.as_tensor(m.readout(m.readout_input(x)))[0].double().numpy()
        logp = logits - logits.max(-1, keepdims=True)
        logp -= np.log(np.exp(logp).sum(-1, keepdims=True))
        nxt = np.asarray(ids[1:] + [ids[-1]])
        valid = np.asarray([i + 1 < N and ids[i + 1] != bos for i in range(N)])
        rows = np.arange(N)
        loss = np.where(valid, -logp[rows, nxt], 0.0)
        order = np.argsort(-logits, -1)[:, :TOP]
        docs.append({**info, "name": name, "words": [label(i) for i in ids], "N": N, "tgt": [int(v) for v in target],
                     "reset": [int(i == bos) for i in ids], "valid": [int(t) for t in valid], "loss": r3(loss),
                     "top": [[label(t) for t in row] for row in order],
                     "prob": r3(np.exp(np.take_along_axis(logp, order, -1))), "pact": r3(np.exp(logp[rows, nxt])),
                     "layers": layers})
        pred = valid & np.r_[target[1:], False]  # positions whose next token is a prediction target
        print(f"  {name}: {N} tokens, mean loss {loss[pred].mean():.3f} over the {int(pred.sum())} predicted tokens")

    hl = lambda mu: (np.log(0.5) / np.log(mu.numpy())).tolist()
    init = np.geomspace(m.memory_half_life_min, m.memory_half_life_max, S)
    payload = json.dumps({"docs": docs, "S": S, "NL": NL, "mu": [[float(f"{v:.7g}") for v in mu.numpy()] for mu in mus],
                          "hl": [r3(hl(mu)) for mu in mus], "hl0": r3(init),
                          "lam_hl": [float(np.log(0.5) / np.log(l)) for l in lams], "write_gate": bool(m.write_gate)}, separators=(",", ":"))
    out = (ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent) / "reports" / "rsm_bank_stepper.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    templates = Path(__file__).resolve().parent / "templates"
    page = (templates / "rsm_bank_stepper.html").read_text()
    title = f"{args.run_name} / {ckpt.name} - {NL} layer(s) x {S} banks"
    out.write_text(page.replace("/*__CSS__*/", (templates / "report_tailwind.css").read_text())
                   .replace("__TITLE__", _html.escape(title)).replace("/*__DATA__*/null", payload))
    print(f"Wrote {out}  ({len(docs)} texts, {len(payload) / 1e6:.1f} MB data)")
