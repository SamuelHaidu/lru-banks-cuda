"""How a `gated-deltanet` run assembles words out of BPE pieces ("photosynthesis" is 10 tokens: `Ġp h ot os y n t he
si s`). Prose validation rows (the cosmopedia categories); every token gets a word depth: 0 for a word's first piece
(`Ġ` + letters, or letters after a non-word token), d for its d-th continuation piece (letters right after letters),
-1 outside words (spaces, punctuation, tags). A prediction at t is "inside" when token t + 1 continues the word (its
depth >= 1: the word so far is the d pieces t - d + 1 .. t), else "start" (a new word or punctuation).

1. logit lens: each layer's stream through the final norm and readout, top-1 accuracy on inside targets by depth
   and on start targets;
2. ablations, loss change by target kind: each layer's MoE output zeroed, each layer's token mixer zeroed (w_o = 0),
   and the short causal conv cut to the current token in every layer (its taps on the 3 previous tokens zeroed: a
   conv-only word reader then fails past depth 1);
3. linear probes (ridge to one-hot, half the rows to fit, half to test): at continuation pieces, the word's first
   piece read from each layer's stream, by how far back it is (1..3: inside one conv's window, 4+: beyond it);
4. routing: per layer, I(expert; depth bucket) / H(expert) and the experts' shares of first vs continuation pieces.
5. one text (--text) piece by piece: the first layer whose logit-lens top-1 is the actual next piece.

    uv run python "ad hoc scripts/gdn_subword_trace.py" --run-name X [--dataset D] [--docs 2] [--device xpu]
        [--no-ablate] [--text "..."]
"""

import argparse
import json
import re
import time

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

ap = argparse.ArgumentParser()
ap.add_argument("--run-name", required=True)
ap.add_argument("--checkpoint", default="latest")
ap.add_argument("--dataset", default=None)
ap.add_argument("--docs", type=int, default=2, help="validation documents per prose category")
ap.add_argument("--window", type=int, default=512)
ap.add_argument("--device", default="xpu")
ap.add_argument("--no-ablate", action="store_true")
ap.add_argument("--text", default="The photosynthesis of chloroplasts is unbelievable, and the mitochondria "
                                  "transform glucose into energy.")
a = ap.parse_args()
setup_device(a.device)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model import config as cfg_mod  # noqa: E402
import model.archs  # noqa: E402,F401
from model.evaluate import load_checkpoint  # noqa: E402

PROSE = ["auto_math_text", "khanacademy", "openstax", "stanford", "wikihow", "stories", "web_samples_v1",
         "web_samples_v2"]
BUCKETS = ["d1", "d2", "d3", "d4", "d5+"]  # depth of the predicted piece: the word so far has d pieces


def word_depths(ids, pieces):
    depth = np.full(len(ids), -1)
    for t, i in enumerate(ids):
        s = pieces[i]
        if i < 16 or not s:
            continue
        if s[0] == "Ġ":
            depth[t] = 0 if len(s) > 1 and s[1].isascii() and s[1].isalnum() else -1
        elif s[0].isascii() and s[0].isalnum():
            prev = pieces[ids[t - 1]] if t else ""
            depth[t] = depth[t - 1] + 1 if t and depth[t - 1] >= 0 and prev[-1:].isascii() and prev[-1:].isalnum() else 0
    return depth


def bucket(d):
    return BUCKETS[min(d, 5) - 1]


def main():
    config = cfg_mod.load(a.run_name)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(cfg_mod.tokenizer_path(config["data"]["dataset"])))
    pieces = [tok.id_to_token(i) or "" for i in range(tok.get_vocab_size())]
    m = load_checkpoint(a.run_name, a.checkpoint)
    L, E = m.num_layers, m.num_experts
    ds = a.dataset or config["data"]["dataset"]
    meta = json.loads((cfg_mod.DATASETS_DIR / ds / "metadata.json").read_text())
    tokens = np.load(cfg_mod.DATASETS_DIR / ds / meta["validation"]["file"], mmap_mode="r")
    index = np.load(cfg_mod.DATASETS_DIR / ds / "validation.categories.npy")
    rng = np.random.default_rng(0)
    rows = []
    for name in PROSE:
        if name not in meta["category_ids"]:
            continue
        docs = index[(index[:, 2] == meta["category_ids"][name]) & (index[:, 1] > a.window)]
        for s, _, _ in docs[rng.choice(len(docs), min(a.docs, len(docs)), replace=False)]:
            rows.append(np.asarray(tokens[s:s + a.window + 1]).astype(np.int64))
    rows = np.stack(rows)
    x_ids, y_ids = rows[:, :-1], rows[:, 1:]
    depth = np.stack([word_depths(r.tolist(), pieces) for r in rows])  # (R, T + 1)
    d_x, d_y = depth[:, :-1].reshape(-1), depth[:, 1:].reshape(-1)
    kind = np.where(d_y >= 1, np.array([bucket(max(d, 1)) for d in d_y]), "start")
    print(f"run {a.run_name} checkpoint {a.checkpoint}, {ds}: {len(rows)} prose rows x {a.window}; predictions: "
          + ", ".join(f"{k} {np.mean(kind == k):.1%}" for k in BUCKETS + ["start"]))

    mode = {"record": False, "moe_off": None}
    rec = {"stream": {}, "choice": {}}
    orig_layer, orig_moe = m._layer, m._moe

    def layer(n, x, reset, state, rows_=None, need_state=False):
        out = orig_layer(n, x, reset, state, rows_, need_state)
        if mode["record"]:
            rec["stream"].setdefault(n, []).append(out[0].reshape(-1, out[0].shape[-1]).float().cpu())
        return out

    def moe(p, h, bias_var):
        n = int(re.search(r"l(\d+)_router_bias", bias_var.path).group(1))
        if mode["record"]:
            logits = h.float() @ p["router"].float() + p["router_bias"].float()
            rec["choice"].setdefault(n, []).append(logits.argmax(-1).cpu())
        if mode["moe_off"] == n:
            return torch.zeros_like(h)
        return orig_moe(p, h, bias_var)

    m._layer, m._moe = layer, moe
    y_t = torch.as_tensor(y_ids.reshape(-1))

    def losses():
        out = []
        with torch.no_grad():
            for b in range(0, len(x_ids), 8):
                h = m.dense_hidden(x_ids[b:b + 8].astype(np.int32))
                out.append(torch.as_tensor(m.readout(m.readout_input(h.reshape(-1, h.shape[-1])))).float().cpu())
        return F.cross_entropy(torch.cat(out), y_t, reduction="none").numpy()

    def by_kind(v):
        return {k: float(v[kind == k].mean()) for k in BUCKETS + ["start"]}

    t0 = time.time()
    mode["record"] = True
    with torch.no_grad():
        E0 = m.token_embedding.embeddings.value
        rec["stream"][-1] = [E0.index_select(0, torch.as_tensor(x_ids.reshape(-1), device=E0.device)).float().cpu()]
    base = losses()
    mode["record"] = False
    stream = {n: torch.cat(v) for n, v in rec["stream"].items()}
    choice = {n: torch.cat(v).numpy() for n, v in rec["choice"].items()}
    print(f"loss {base.mean():.4f} (one pass {time.time() - t0:.0f} s); by target: "
          + ", ".join(f"{k} {v:.3f}" for k, v in by_kind(base).items()))

    # 1. logit lens
    print("\n1. logit lens top-1 accuracy by layer (stream after the layer; -1: the embedding)")
    print("layer | " + " ".join(f"{k:>6}" for k in BUCKETS + ["start"]))
    with torch.no_grad():
        for n in range(-1, L):
            top = torch.as_tensor(m.readout(m.readout_input(stream[n].to(E0.device)))).argmax(-1).cpu().numpy()
            hit = top == y_ids.reshape(-1)
            print(f"{n:5d} | " + " ".join(f"{hit[kind == k].mean():6.1%}" for k in BUCKETS + ["start"]))

    # 2. ablations
    if not a.no_ablate:
        print("\n2. loss change when a part is removed (nats per prediction)")
        print("part       | " + " ".join(f"{k:>6}" for k in BUCKETS + ["start"]))

        def show(label, v):
            d = by_kind(v - base)
            print(f"{label:10s} | " + " ".join(f"{d[k]:+6.3f}" for k in BUCKETS + ["start"]), flush=True)

        convs = [m._params[n]["conv"] for n in range(L)]
        saved = [c.value.detach().clone() for c in convs]
        with torch.no_grad():
            for c in convs:
                c.value[:-1].zero_()  # keep tap S-1 (the current token)
        show("conv past", losses())
        with torch.no_grad():
            for c, s in zip(convs, saved):
                c.value.copy_(s)
        for n in range(L):
            mode["moe_off"] = n
            show(f"moe {n}", losses())
        mode["moe_off"] = None
        for n in range(L):
            w = m._params[n]["w_o"]
            s = w.value.detach().clone()
            with torch.no_grad():
                w.value.zero_()
            show(f"mixer {n}", losses())
            with torch.no_grad():
                w.value.copy_(s)

    # 3. probes: the word's first piece from the stream at continuation pieces
    flat_x = x_ids.reshape(-1)
    R, T = x_ids.shape
    pos = np.arange(R * T)
    back = np.where(d_x >= 1, d_x, 0)  # pieces back to the word's first piece
    sel = (d_x >= 1) & ((pos % T) >= back)
    first = flat_x[np.clip(pos - back, 0, None)]
    rows_of = pos // T
    fit_m, test_m = sel & (rows_of % 2 == 0), sel & (rows_of % 2 == 1)
    groups = {"back 1": back == 1, "back 2-3": (back >= 2) & (back <= 3), "back 4+": back >= 4}
    print("\n3. probe: the word's first piece from the stream at a continuation piece (test accuracy; "
          f"most common first piece: {np.mean(first[test_m] == np.bincount(first[fit_m]).argmax()):.1%})")
    print("layer | " + " ".join(f"{g:>9}" for g in groups) + "   (test positions: "
          + ", ".join(f"{g} {int((test_m & s).sum())}" for g, s in groups.items()) + ")")
    Y = F.one_hot(torch.as_tensor(first[fit_m]), m.vocabulary_size).double()
    for n in range(-1, L):
        S = stream[n].double()
        mu, sd = S[torch.as_tensor(fit_m)].mean(0), S[torch.as_tensor(fit_m)].std(0) + 1e-6
        H = torch.cat([(S - mu) / sd, torch.ones(len(flat_x), 1, dtype=torch.float64)], 1)
        Hf = H[torch.as_tensor(fit_m)]
        W = torch.linalg.solve(Hf.T @ Hf + 1.0 * torch.eye(H.shape[1], dtype=torch.float64), Hf.T @ Y)
        pred = (H @ W).argmax(-1).numpy()
        print(f"{n:5d} | " + " ".join(f"{np.mean(pred[test_m & s] == first[test_m & s]):9.1%}" for s in groups.values()))

    # 4. routing by word position
    pos_b = np.where(d_x < 0, "other", np.where(d_x == 0, "first", np.where(d_x <= 3, "cont 1-3", "cont 4+")))
    print("\n4. routing: I(expert; piece position) / H(expert), and the expert most used by continuation pieces")

    def ent(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    labels = {s: i for i, s in enumerate(["other", "first", "cont 1-3", "cont 4+"])}
    pb = np.array([labels[s] for s in pos_b])
    for n in range(L):
        c = choice[n]
        joint = np.zeros((E, 4))
        np.add.at(joint, (c, pb), 1)
        joint /= joint.sum()
        mi = ent(joint.sum(1)) + ent(joint.sum(0)) - ent(joint.ravel())
        share = lambda lab: np.bincount(c[pb == labels[lab]], minlength=E) / max(1, (pb == labels[lab]).sum())
        cont, fst = share("cont 1-3"), share("first")
        e = int(np.argmax(cont - fst))
        print(f"  layer {n:2d}: I/H {mi / ent(joint.sum(1)):.3f}   expert {e}: {cont[e]:.0%} of continuation pieces, "
              f"{fst[e]:.0%} of first pieces")

    # 5. one text
    ids = tok.encode(a.text).ids
    ids = ([m.reset_token] + ids) if ids[0] != m.reset_token else ids
    rec["stream"].clear()
    mode["record"] = True
    with torch.no_grad():
        m.dense_hidden(np.asarray([ids], np.int32))
    mode["record"] = False
    with torch.no_grad():
        tops = [torch.as_tensor(m.readout(m.readout_input(rec["stream"][n][0].to(E0.device)))).argmax(-1).cpu().numpy()
                for n in range(L)]
    dep = word_depths(ids, pieces)
    print(f"\n5. {a.text!r}: for each next piece, the first layer from which the logit lens' top-1 is right "
          "(and stays right); '-': never")
    out = []
    for t in range(len(ids) - 1):
        ok = [tops[n][t] == ids[t + 1] for n in range(L)]
        first_ok = next((n for n in range(L) if all(ok[n:])), None)
        nxt = pieces[ids[t + 1]].replace("Ġ", "_")
        tag = "in" if dep[t + 1] >= 1 else "new"
        out.append(f"{nxt}[{tag}]:{'-' if first_ok is None else first_ok}")
    print("  " + "  ".join(out))


if __name__ == "__main__":
    main()
