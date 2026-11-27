"""The Inspect tab's computations (run in the worker): per-token predictions, and per architecture the internals
(min-gru states and static memory, gated-deltanet reads, lru-banks bank reads and contents)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

MAX_INSPECT_TOKENS = 1024


def _b64(a: np.ndarray) -> str:
    import base64

    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def inspect_text(models: "Models", body: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the Inspect tab needs for one text, for a `min-gru` run.

    Per layer (float32, base64, row-major): update_gate, candidate, state `(T, H)`; W = norm_state[:, None] *
    out_proj `(H, D)`; write_scale `(D,)`; the channels' half-lives on this text; per channel the tokens its
    direction promotes / suppresses through the final norm and the readout (direct logit effect). Global: the
    tokens, which positions reset the state, the readout kernel `(D, V)` times final_norm (so the page can project
    contributions onto any token), the final stream's rms per position, and the logit lens: top-5 next tokens after
    the embedding and after each layer, at every position."""
    import torch

    model, tok = models.load(body["run"], str(body.get("checkpoint", "latest")))
    if type(model).__name__ == "LbcModel":
        return inspect_lbc(models, body, model, tok)

    from model.archs.min_gru import MinGRULM

    from model.archs.gated_deltanet import GatedDeltaNetLM

    from model.archs.lru_banks import LRUBanksLM

    if isinstance(model, GatedDeltaNetLM):
        return inspect_gdn(models, body, model, tok)
    if isinstance(model, LRUBanksLM):
        return inspect_banks(models, body, model, tok)
    if not isinstance(model, MinGRULM):
        raise ValueError(f"inspection is for min-gru and gated-deltanet runs ({type(model).__name__} loaded)")
    ids = tok.encode(body["text"]).ids
    if not ids or ids[0] != model.reset_token:
        ids = [model.reset_token] + ids
    truncated = len(ids) > MAX_INSPECT_TOKENS
    ids = ids[:MAX_INSPECT_TOKENS]
    raw_ids, meta = ids, None
    with models.lock, torch.no_grad():
        if model.static_memory:  # the layers see the row without the span contents: inspect that sequence
            info, meta = model.inspect_memory(np.asarray([ids], np.int32))
            ids = [ids[i] for i in meta["kept"]]
        else:
            info = model.inspect(np.asarray([ids], np.int32))
        E = model.token_embedding.embeddings.value
        emb = E[torch.as_tensor(ids, device=E.device)].float().cpu().numpy()
        K = model.readout.kernel.value.detach().float().cpu().numpy()  # (D, V)
        g_final = model.final_norm.value.detach().float().cpu().numpy()
        params = [{k: v.value.detach().float().cpu().numpy() for k, v in p.items()} for p in model._params]
    readout = g_final[:, None] * K  # logits = (h / rms(h)) @ readout
    rms = lambda x: np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)
    vocab = [tok.decode([i], skip_special_tokens=False) for i in range(K.shape[1])]

    def top(h: np.ndarray, k: int = 5):
        logits = (h / rms(h)) @ readout
        logits -= logits.max(-1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(-1, keepdims=True)
        idx = np.argsort(-p, -1)[:, :k]
        return [[[int(i), round(float(p[t, i]), 4)] for i in row] for t, row in enumerate(idx)]

    stream = emb.copy()
    lens = [top(stream)]
    layers = []
    not_reset = np.asarray(ids) != model.reset_token
    for n, (lay, p) in enumerate(zip(info, params)):
        z, cand, state, write = (lay[k][0] for k in ("update_gate", "candidate", "state", "layer_write"))
        stream = stream + write + (lay["memory_add"][0] if "memory_add" in lay else 0)
        lens.append(top(stream))
        keep = np.clip(1 - z[not_reset], 1e-12, 1)
        half_life = np.log(0.5) / np.minimum(np.log(keep).mean(0), -1e-9)
        layers.append({
            "update_gate": _b64(z), "candidate": _b64(cand), "state": _b64(state),
            "write_scale": p["write_scale"].tolist(), "half_life": np.round(half_life, 1).tolist(),
        })
        if "out_proj" in p:  # the tanh projection (not the MLP): each channel's direct push on the logits
            W = p["norm_state"][:, None] * p["out_proj"]  # (H, D)
            order = np.argsort((W * p["write_scale"][None]) @ readout, -1)  # (H, V)
            layers[-1].update({"W": _b64(W), "promotes": [[vocab[i] for i in row[::-1][:6]] for row in order],
                               "suppresses": [[vocab[i] for i in row[:6]] for row in order]})
    final = stream
    T, H, D = len(ids), model.hidden_size, model.dim
    memory = None if meta is None else memory_reads(model, info, meta, raw_ids, vocab, layers)
    logits = (final / rms(final)) @ readout
    stats = _prediction_stats(logits, ids, body["run"], model.reset_token)
    return {**stats, "memory": memory, "static_memory": bool(model.static_memory), "raw_tokens": len(raw_ids),
            "raw": [vocab[i] for i in raw_ids],  # the whole text; the layers see the positions in `kept`
            "kept": [int(i) for i in meta["kept"]] if meta is not None else list(range(len(raw_ids))),"tokens": [vocab[i] for i in ids], "ids": ids, "reset": (~not_reset).astype(int).tolist(),
            "T": T, "H": H, "D": D, "V": int(K.shape[1]), "truncated": truncated, "layers": layers,
            "readout": _b64(readout), "final_rms": rms(final)[:, 0].tolist(), "lens": lens, "vocab": vocab}


def _top5(logits: np.ndarray, k: int = 5):
    logits = logits - logits.max(-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(-1, keepdims=True)
    idx = np.argsort(-p, -1)[:, :k]
    return [[[int(i), round(float(p[t, i]), 4)] for i in row] for t, row in enumerate(idx)]


def inspect_gdn(models: "Models", body: Dict[str, Any], model: Any, tok: Any) -> Dict[str, Any]:
    """The Inspect tab for a `gated-deltanet` run: tokens, the final prediction's loss / rank / entropy at every
    position, the logit lens (top-5 after the embedding and after each layer) and per layer and head the memory's
    half-life (median over the text's positions) and mean write strength beta. Every layer's delta-rule inputs (q, k, v
    after the conv, g, beta) and read o are kept (`models.attn`) for `/api/attend`, which unrolls a position's read
    into weights over the past tokens."""
    import uuid

    import torch

    import model.archs.gated_deltanet as gd

    ids = tok.encode(body["text"]).ids
    if not ids or ids[0] != model.reset_token:
        ids = [model.reset_token] + ids
    truncated = len(ids) > MAX_INSPECT_TOKENS
    ids = ids[:MAX_INSPECT_TOKENS]
    rec: List[List[torch.Tensor]] = []
    streams: List[torch.Tensor] = []
    orig_rule = gd.gated_delta_rule

    def rule(q, k, v, g, beta, reset, initial=None, need_state=False):
        o, S = orig_rule(q, k, v, g, beta, reset, initial, need_state)
        rec.append([x[0].detach().double().cpu() for x in (q, k, v, g, beta, o)])
        return o, S

    def layer(*args):
        out = type(model)._layer(model, *args)
        streams.append(out[0][0].detach().float())
        return out

    with models.lock, torch.no_grad():
        gd.gated_delta_rule, model._layer = rule, layer
        try:
            model.dense_hidden(np.asarray([ids], np.int32))
        finally:
            gd.gated_delta_rule = orig_rule
            del model._layer
        E = model.token_embedding.embeddings.value
        emb = E[torch.as_tensor(ids, device=E.device)].float()
        logits = [torch.as_tensor(model.readout(model.readout_input(s))).float().cpu().numpy() for s in [emb] + streams]
    vocab = [tok.decode([i], skip_special_tokens=False) for i in range(model.vocabulary_size)]
    reset = np.asarray(ids) == model.reset_token
    q, k, v, g, beta, o = (torch.stack([r[i] for r in rec]) for i in range(6))  # (layers, T, heads, .)
    models.attn = {"id": uuid.uuid4().hex, "reset": reset, "q": q, "k": k, "v": v, "g": g, "beta": beta, "o": o}
    heads = []
    for n in range(len(rec)):
        gn = g[n][torch.as_tensor(~reset)]
        hl = (np.log(0.5) / gn.clamp_max(-1e-9)).median(0).values
        heads.append({"half_life": [round(float(x), 1) for x in hl], "beta": [round(float(x), 3) for x in beta[n].mean(0)]})
    stats = _prediction_stats(logits[-1], ids, body["run"], model.reset_token)
    T = len(ids)
    return {**stats, "kind": "gdn", "attn_id": models.attn["id"], "heads": heads, "conv_size": int(model.conv_size),
            "tokens": [vocab[i] for i in ids], "raw": [vocab[i] for i in ids], "kept": list(range(T)), "ids": ids,
            "reset": reset.astype(int).tolist(), "T": T, "truncated": truncated, "layers": [], "memory": None,
            "static_memory": False, "raw_tokens": T, "lens": [_top5(x) for x in logits], "vocab": vocab}


def attend(models: "Models", body: Dict[str, Any]) -> Dict[str, Any]:
    """Position t's read in every layer and head of the last `inspect_gdn` text, unrolled: o_t = sum_i c_i v_i over
    the document's positions start..t (`delta_rule_read_weights`). Returns c and each position's share of the read,
    c_i (v_i . o_t) / |o_t|^2 (they sum to 1), both `(layers, heads, t + 1 - start)` float32 base64, and the largest
    difference between the rebuilt read and the model's, relative to the read's size."""
    from model.archs.delta_rule import delta_rule_read_weights

    a = getattr(models, "attn", None)
    if a is None or a["id"] != body.get("attn_id"):
        raise ValueError("inspect the text again (another text or model was inspected since)")
    t = int(body["t"])
    c, start = delta_rule_read_weights(a["q"], a["k"], a["g"], a["beta"], a["reset"], t)
    vals = a["v"][:, start:t + 1]  # (layers, n, heads, V)
    rebuilt = np.einsum("lhi,lihv->lhv", c.numpy(), vals.numpy())
    o = a["o"][:, t].numpy()
    err = float(np.abs(rebuilt - o).max() / max(1e-12, float(np.abs(o).max())))
    share = np.einsum("lhi,lihv,lhv->lhi", c.numpy(), vals.numpy(), rebuilt) \
        / np.maximum((rebuilt ** 2).sum(-1), 1e-30)[..., None]
    return {"t": t, "start": start, "n": t + 1 - start, "c": _b64(c.numpy()), "share": _b64(share), "err": err,
            "o_norm": np.round(np.linalg.norm(o, axis=-1), 4).tolist()}


def inspect_banks(models: "Models", body: Dict[str, Any], model: Any, tok: Any) -> Dict[str, Any]:
    """The Inspect tab for an `lru-banks` run: tokens, the final prediction's loss / rank / entropy, the logit lens
    (after the embedding and each layer) and each layer's LRU input gate at every position. The torch path runs with
    every layer's routing weights r, values v, read probabilities p over the slots (per hop) and reads o recorded
    (`models.banks`) for `/api/banks`, which unrolls one position's reads and bank contents over the past tokens."""
    import uuid

    import torch

    import model.archs.lru_banks as lb

    ids = tok.encode(body["text"]).ids
    if not ids or ids[0] != model.reset_token:
        ids = [model.reset_token] + ids
    truncated = len(ids) > MAX_INSPECT_TOKENS
    ids = ids[:MAX_INSPECT_TOKENS]
    writes: List[List[np.ndarray]] = []  # per layer: r, v
    reads: List[List[np.ndarray]] = []  # per read (layer-major, then hop): p, o
    gates: List[Optional[np.ndarray]] = []
    streams: List[torch.Tensor] = []
    np_ = lambda x: x[0].detach().float().cpu().numpy()
    orig = (lb.bank_write_chunk, lb.bank_read, lb.lru_states, lb.FUSED)

    def write(k, v, r, *args, **kw):
        writes.append([np_(r), np_(v)])
        return orig[0](k, v, r, *args, **kw)

    def read(mem, q, tau, record=None):
        rec: List[torch.Tensor] = []
        o = orig[1](mem, q, tau, rec)
        reads.append([np_(rec[0]), np_(o)])
        return o

    def lru(proj, nu, theta, gate, *args, **kw):
        gates.append(None if gate is None else np_(gate)[:, 0])
        return orig[2](proj, nu, theta, gate, *args, **kw)

    def layer(*args):
        out = type(model)._layer(model, *args)
        streams.append(out[0][0].detach().float())
        return out

    with models.lock, torch.no_grad():
        lb.bank_write_chunk, lb.bank_read, lb.lru_states, lb.FUSED = write, read, lru, False
        model._layer = layer
        try:
            model.dense_hidden(np.asarray([ids], np.int32))
        finally:
            lb.bank_write_chunk, lb.bank_read, lb.lru_states, lb.FUSED = orig
            del model._layer
        E = model.token_embedding.embeddings.value
        emb = (E[torch.as_tensor(ids, device=E.device)] @ model.w_in.value).float()
        logits = [torch.as_tensor(model.readout(model.readout_input(s))).float().cpu().numpy() for s in [emb] + streams]
        loga = [p["loga"].value.detach().double().cpu().numpy() for p in model._params]
        cpu = lambda w: w.value.detach().float().cpu().numpy()
        lh = model.layer_hops
        hops = max(lh)  # a layer with fewer reads: its missing ones are zeros (p, o, W_o), shown as absent
        w_o = [np.stack([cpu(p["w_o"])] + ([cpu(p["w_o2"]) if "w_o2" in p else np.zeros_like(cpu(p["w_o"]))]
                                            if hops > 1 else [])) for p in model._params]
        lens_w = (cpu(model.final_norm)[:, None] * cpu(model.w_out), cpu(model.token_embedding.embeddings))
    nl = model.num_layers
    by_layer, i = [], 0
    for h in lh:  # the reads were recorded layer after layer, then hop after hop
        rs = reads[i:i + h]
        i += h
        by_layer.append(rs + [[np.zeros_like(x) for x in rs[0]]] * (hops - h))
    reads = [r for rs in by_layer for r in rs]
    reset = np.asarray(ids) == model.reset_token
    models.banks = {"id": uuid.uuid4().hex, "reset": reset, "loga": np.stack(loga),  # (L, H, M)
                    "r": np.stack([w[0] for w in writes]), "v": np.stack([w[1] for w in writes]),  # (L, T, H, .)
                    "p": np.stack([r[0] for r in reads]).reshape(nl, hops, *reads[0][0].shape),  # (L, hops, T, H, M)
                    "o": np.stack([r[1] for r in reads]).reshape(nl, hops, *reads[0][1].shape), "ids": ids,
                    "w_o": np.stack(w_o), "lens_w": lens_w, "tok": tok, "layer_hops": lh}  # w_o (L, hops, H * D, dim)
    stats = _prediction_stats(logits[-1], ids, body["run"], model.reset_token)
    lens = [_top5(x) for x in logits]
    used = set(ids) | {i for rows in lens for row in rows for i, _ in row}
    vocab = {i: tok.decode([i], skip_special_tokens=False) for i in sorted(used)}
    slot_hl = np.log(0.5) / models.banks["loga"][0, 0]
    T = len(ids)
    return {**stats, "kind": "banks", "banks_id": models.banks["id"], "num_layers": nl, "hops": hops, "layer_hops": lh,
            "num_heads": model.num_heads, "slots": model.slots, "slot_half_life": np.round(slot_hl, 1).tolist(),
            "gate": [None if g is None else np.round(g, 3).tolist() for g in gates],
            "tokens": [vocab[i] for i in ids], "raw": [vocab[i] for i in ids], "kept": list(range(T)), "ids": ids,
            "reset": reset.astype(int).tolist(), "T": T, "truncated": truncated, "layers": [], "memory": None,
            "static_memory": False, "raw_tokens": T, "lens": lens, "vocab": vocab}


def _topk_rows(p: Any, k: int):
    """torch.topk over the last dim by k argmax passes: on the Iris Xe, torch-xpu's topk over a 32k vocabulary asks
    for a work-group larger than the device's 512."""
    import torch

    p = p.clone()
    vals, idx = [], []
    for _ in range(k):
        i = p.argmax(-1, keepdim=True)
        vals.append(p.gather(-1, i))
        idx.append(i)
        p.scatter_(-1, i, -1.0)
    return torch.cat(vals, -1), torch.cat(idx, -1)


def inspect_lbc(models: "Models", body: Dict[str, Any], model: Any, tok: Any) -> Dict[str, Any]:
    """The Inspect tab for an lbc checkpoint (`model/lbc_torch.py`): the `inspect_banks` payload, from one recording
    pass of the decoder (the generation path, token by token). Its gates are binary per head: `gate` per layer is the
    fraction of the heads whose write gate is open at each token, `gates` per layer [write, read 1, read 2] the same
    for each gate (read 2: 0 on 1-hop layers)."""
    import uuid

    import torch

    from model.lbc_torch import trace

    ids = tok.encode(body["text"]).ids
    if not ids or ids[0] != model.reset_token:
        ids = [model.reset_token] + ids
    truncated = len(ids) > MAX_INSPECT_TOKENS
    ids = ids[:MAX_INSPECT_TOKENS]
    with models.lock, torch.no_grad():
        r = trace(model, ids, MAX_INSPECT_TOKENS)
        lens = []
        hs = torch.from_numpy(r["h"]).to(model.device)  # (T, L+1, D)
        for n in range(hs.shape[1]):  # the logit lens on the device: top-5 only
            p = torch.softmax(torch.cat([model.logits(hs[i:i + 256, n]) for i in range(0, len(ids), 256)]), -1)
            v, i = _topk_rows(p, 5)
            lens.append([[[int(a), round(float(b), 4)] for a, b in zip(ri, rv)]
                         for ri, rv in zip(i.cpu().tolist(), v.cpu().tolist())])
        lens_w = model.lens_weights()
    L, lh = model.num_layers, model.layer_hops
    w_o = []
    for n, p in enumerate(model.layers):
        wo = p["Wo"].float().cpu().numpy()
        w_o.append(np.stack([wo, p["Wo2"].float().cpu().numpy() if "Wo2" in p else np.zeros_like(wo)]))
    reset = np.asarray(ids) == model.reset_token
    models.banks = {"id": uuid.uuid4().hex, "reset": reset, "loga": model.loga,  # (L, H, M)
                    "r": r["w"].transpose(1, 0, 2, 3), "v": r["v"].transpose(1, 0, 2, 3),  # (L, T, H, .)
                    "p": r["p"].transpose(1, 2, 0, 3, 4), "o": r["o"].transpose(1, 2, 0, 3, 4),  # (L, 2, T, H, .)
                    "ids": ids, "w_o": np.stack(w_o), "lens_w": lens_w, "tok": tok, "layer_hops": lh}
    g = r["g"].mean(-1)  # (T, L, 3): open fraction of the heads
    stats = _prediction_stats(r["logits"], ids, body["run"], model.reset_token, spans=[])
    used = set(ids) | {i for rows in lens for row in rows for i, _ in row}
    vocab = {i: tok.decode([i], skip_special_tokens=False) for i in sorted(used)}
    T = len(ids)
    return {**stats, "kind": "banks", "gate_kind": "write", "banks_id": models.banks["id"], "num_layers": L, "hops": 2,
            "layer_hops": lh, "num_heads": model.num_heads, "slots": model.slots,
            "slot_half_life": model.slot_half_life,  # [layer][slot]
            "gate": [np.round(g[:, n, 0], 3).tolist() for n in range(L)],
            "gates": [[np.round(g[:, n, k], 3).tolist() for k in range(3)] for n in range(L)],
            "tokens": [vocab[i] for i in ids], "raw": [vocab[i] for i in ids], "kept": list(range(T)), "ids": ids,
            "reset": reset.astype(int).tolist(), "T": T, "truncated": truncated, "layers": [], "memory": None,
            "static_memory": False, "raw_tokens": T, "lens": lens, "vocab": vocab}


def banks_at(models: "Models", body: Dict[str, Any]) -> Dict[str, Any]:
    """Position t of the last `inspect_banks` text, unrolled exactly over its document's tokens start..t. Slot m of a
    head holds the average of the values written to it, token j weighted by r_jm a_m^(t - j) / z_tm (z the written
    mass), and a read takes sum_m p_tm of those averages, so:
    - `read` `(layers, hops, heads, n)`: token j's weight in the read, c_j = sum_m p_tm r_jm a_m^(t - j) / z_tm
      (c >= 0, summing to 1; the read is sum_j c_j v_j: `err` is the largest difference with the model's read,
      relative to its size);
    - `stored` `(layers, heads, n)`: token j's share of the banks, the mean over the non-empty slots of its share of
      each slot's average;
    - `slots` [layer][hop][head]: the 3 slots the read takes most, as [slot, p, [[position, share of the slot], ...]];
    - `filled` [layer][head]: non-empty slots;
    - `says` [layer][hop][head] and `says_all` [layer][hop]: the value lens, the 5 tokens a read's addition to the
      stream (o W_o, per head and all heads) raises most through the final norm's gain and the readout (its direct
      logit effect: what was retrieved, in token space, also where the read position's value is a mixed summary)."""
    from model.archs.bank_memory import EPS

    b = getattr(models, "banks", None)
    if b is None or b["id"] != body.get("banks_id"):
        raise ValueError("inspect the text again (another text or model was inspected since)")
    t = int(body["t"])
    resets = np.flatnonzero(b["reset"][:t + 1])
    start = int(resets[-1]) if len(resets) else 0
    n = t + 1 - start
    age = (t - np.arange(start, t + 1)).astype(np.float64)
    W = b["r"][:, start:t + 1].astype(np.float64) * np.exp(b["loga"][:, None] * age[None, :, None, None])  # (L, n, H, M)
    z = W.sum(1)  # (L, H, M)
    valid = z > EPS
    share = np.where(valid[:, None], W / np.maximum(z, EPS)[:, None], 0.0)  # (L, n, H, M): j's share of slot m
    p = b["p"][:, :, t].astype(np.float64) * valid[:, None]  # (L, hops, H, M)
    c = np.einsum("lohm,ljhm->lohj", p, share)
    rebuilt = np.einsum("lohj,ljhe->lohe", c, b["v"][:, start:t + 1].astype(np.float64))
    o = b["o"][:, :, t]
    err = float(np.abs(rebuilt - o).max() / max(1e-12, float(np.abs(o).max())))
    stored = share.sum(-1).transpose(0, 2, 1) / np.maximum(valid.sum(-1), 1)[..., None]  # (L, H, n)
    slots = []
    for li in range(p.shape[0]):
        per_hop = []
        for hop in range(p.shape[1]):
            per_head = []
            for h in range(p.shape[2]):
                top = np.argsort(-p[li, hop, h])[:3]
                per_head.append([[int(m), round(float(p[li, hop, h, m]), 4),
                                  [[start + int(j), round(float(share[li, j, h, m]), 4)]
                                   for j in np.argsort(-share[li, :, h, m])[:5] if share[li, j, h, m] > 0.005]]
                                 for m in top if p[li, hop, h, m] > 0.005])
            per_hop.append(per_head)
        slots.append(per_hop)
    L, hops, H, D = o.shape
    add = np.einsum("lohd,lohdk->lohk", o, b["w_o"].reshape(L, hops, H, D, -1))  # (L, hops, H, dim) per head
    g_out, E = b["lens_w"]
    logit = np.concatenate([add, add.sum(2, keepdims=True)], 2) @ g_out @ E.T  # (L, hops, H + 1, V)
    top = np.argsort(-logit, -1)[..., :5]
    dec = lambda i: b["tok"].decode([int(i)], skip_special_tokens=False)
    lh = b["layer_hops"]
    full = np.concatenate([add, add.sum(2, keepdims=True)], 2)
    adds = np.abs(full).max(-1) > 1e-12  # a silent layer (W_o = 0, `silent_from`) adds nothing: no tokens
    says = [[[[dec(i) for i in top[li, hop, h]] if hop < lh[li] and adds[li, hop, h] else [] for h in range(H + 1)]
             for hop in range(hops)] for li in range(L)]
    # every readable slot of every (layer, head): [slot, mass z, tokens since its last write, [[position, share]...]]
    wrote = b["r"][:, start:t + 1] > 0  # (L, n, H, M)
    age_last = np.argmax(wrote[:, ::-1], axis=1)  # (L, H, M)
    k = min(4, n)
    top_j = np.argpartition(-share, k - 1, axis=1)[:, :k] if n > k else np.broadcast_to(
        np.arange(n)[None, :, None, None], share.shape)  # (L, k, H, M)
    top_s = np.take_along_axis(share, top_j, 1)
    contents = []
    for li in range(share.shape[0]):
        per_head = []
        for h in range(share.shape[2]):
            rows = []
            for m in np.flatnonzero(valid[li, h]):
                o = np.argsort(-top_s[li, :, h, m])
                rows.append([int(m), float(f"{z[li, h, m]:.3g}"), int(age_last[li, h, m]),
                             [[start + int(top_j[li, i, h, m]), round(float(top_s[li, i, h, m]), 3)]
                              for i in o if top_s[li, i, h, m] > 0.005]])
            per_head.append(rows)
        contents.append(per_head)
    return {"t": t, "start": start, "n": n, "read": _b64(c), "stored": _b64(stored), "err": err, "slots": slots,
            "filled": valid.sum(-1).tolist(), "says": [[s[:H] for s in sl] for sl in says],
            "says_all": [[s[H] for s in sl] for sl in says], "contents": contents}


def _prediction_stats(logits: np.ndarray, ids: List[int], run: str, reset_token: int,
                      spans: Optional[List[Any]] = None) -> Dict[str, Any]:
    """The prediction at every position (target: the next token): its entropy, the loss and rank of the actual next
    token (rank 0: the top-1 choice), and which targets the training scores (inside the dataset's loss_spans, the
    closing tag included; every one without loss_spans). `spans`: the loss spans when the run's dataset is not
    here (lbc runs)."""
    from model import config as cfg_mod

    T = len(ids)
    logits = logits - logits.max(-1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    entropy = -(np.exp(logp) * logp).sum(-1)
    loss = np.zeros(T)
    loss[:-1] = -logp[np.arange(T - 1), ids[1:]]
    rank = np.zeros(T, int)
    rank[:-1] = (logp[:-1] > logp[np.arange(T - 1), ids[1:]][:, None]).sum(-1)
    if spans is None:
        spans = cfg_mod.load_dataset_metadata(cfg_mod.load(run)["data"]["dataset"]).get("loss_spans") or []
    target = np.zeros(T, bool)  # token j is a scored target (predicted at j - 1)
    for o, c in spans:
        inside = False
        for j, t in enumerate(ids):
            if t == reset_token:
                inside = False
            target[j] |= inside
            inside = True if t == o else (False if t == c else inside)
    scored = np.zeros(T, int)
    scored[:-1] = target[1:] if spans else 1
    return {"entropy": np.round(entropy, 4).tolist(), "loss": np.round(loss, 4).tolist(), "rank": rank.tolist(),
            "scored": scored.tolist()}


def memory_reads(model: Any, info: List[Dict[str, np.ndarray]], meta: Dict[str, Any], raw_ids: List[int],
                 vocab: List[str], layers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The static memory for the Inspect tab. Global: each memory entry's token, memory (0 system, 1 tools) and
    document; the reading positions (among the inspected ones) and their documents. Per layer (added to `layers`):
    `mem_w` `(reads, entries)`: the read weights q . k_j / sum over the entries of the same memory and document, mean
    over the heads; `mem_gate` `(reads, memories)`: the gates, mean over the heads; `mem_add` `(T,)`: the size of the
    memory's addition to the stream at every position (0 where nothing reads)."""
    h, d, nm = model.memory_heads, model.memory_dim, len(model.memory_spans)
    read_pos, read_doc = np.asarray(meta["read_pos"]), np.asarray(meta["read_doc"])
    kind, edoc = np.asarray(meta["entry_kind"]), np.asarray(meta["entry_doc"])
    same = (read_doc[:, None] == edoc[None]).astype(np.float32)  # (reads, entries): a position reads its document
    onehot = np.eye(nm, dtype=np.float32)[kind]  # (entries, memories)
    for n, (lay, L) in enumerate(zip(info, layers)):
        q = lay["memory_query"][0][read_pos].reshape(-1, h, d)
        s = np.einsum("rhd,jhd->rhj", q, meta["entry_keys"][:, n]) * same[:, None]
        tot = np.einsum("rhj,jm->rhm", s, onehot)[..., kind]  # each entry's memory total (reads, heads, entries)
        w = s / np.maximum(tot, 1e-12)
        gate = lay["memory_gate"][0][read_pos].reshape(-1, nm, h).mean(-1)
        L.update(mem_w=_b64(w.mean(1)), mem_gate=_b64(gate),
                 mem_add=np.round(np.linalg.norm(lay["memory_add"][0], axis=-1), 5).tolist())
    names = [{(12, 13): "system", (2, 3): "tools"}.get(tuple(sp), f"memory {i}") for i, sp in enumerate(model.memory_spans)]
    return {"entries": [vocab[raw_ids[p]] for p in meta["entry_pos"]], "entry_pos": [int(p) for p in meta["entry_pos"]],
            "entry_kind": kind.tolist(),
            "entry_doc": edoc.tolist(), "read_pos": read_pos.tolist(), "read_doc": read_doc.tolist(), "names": names}
