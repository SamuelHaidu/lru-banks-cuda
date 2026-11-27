"""Inference for lbc (C++ engine, `engine/`) checkpoints in torch: generation and the web chat's Inspect, on the XPU
(Iris Xe), CPU or CUDA, in fp32.

The model is the engine's `lru-banks` with binary read / write gates (`engine/src/model.hpp`), replayed token by token
as `engine/src/decode.cu` does: per layer

    x = rms(h);  [P | v] = x W_a                      W_a with the norm1 gain folded in
    s_t = lambda s_{t-1} + gamma P                    complex diagonal LRU, s = [Re | Im], zeroed at <bos>
    [k_pre | q_pre | a_r | a_w] = s_t W_s
    k = normalize(k_pre(t-1) + b_k)                   the previous token's k_pre (0 at <bos>)
    q = normalize(q_pre);  g_r = [a_r + b_rg > 0];  g_w = [a_w + b_wg > 0] (never at <bos>)
    route: top-2 of k R_h over the slots, weights p1 = sigmoid(l1 - l2), 1 - p1
    banks: per slot the running averages K, V of its writes and the mass Z, decayed by a_m per token:
           Z' = a Z + r,  K += (r / Z') (k - K),  V likewise (the write), then the read
    read(q) = g_r * sum_m softmax_m(tau q.K_m over the slots with Z > 1e-6) V_m
    h += read(q) W_o;  h += FFN(h);  2-hop layers: q2 from rms(h) W_q2, h += read(q2) W_o2 (+ the second FFN)
    logits = (rms(h) g) E^T

Each token's step is one fixed sequence of small kernels on static buffers, so on the XPU (and CUDA) it is captured
once as a device graph and replayed per token (~2.5x fewer launch costs than eager on the Iris Xe). A `Decoder` holds
one document's state; with `record=T` it also stores, per position, what Inspect needs (write weights, values, read
probabilities and outputs, the gates, the stream after every layer).

    uv run python -m model.lbc_torch runs/agentic-47m/model.lbc --prompt "<user>hi</user><assistant>" [--bench]
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from model.device import setup_backend

setup_backend()  # the SYCL kernel cache (env only): before the XPU runtime starts

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MAGIC = b"LBCKPT01"
ALIGN = 256
SLOTS = 128
RMS_EPS = 1e-6
BANK_EPS = 1e-6
NORM_EPS = 1e-12
BOS = 0


def read_header(path: str | Path) -> tuple[Dict[str, Any], int]:
    """The checkpoint's JSON header (format, epoch, step, tensors, config) and the offset of its parameters."""
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError(f"{path} is not an lbc checkpoint")
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    return h, (16 + n + ALIGN - 1) // ALIGN * ALIGN


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)


class LbcModel:
    """An lbc checkpoint's weights on `device` (fp32), with the norm gains folded into the next matrix."""

    stateful_generation = True
    reset_token = BOS

    def __init__(self, path: str | Path, device: str = "xpu"):
        self.path = str(path)
        h, off = read_header(path)
        self.header = h
        a = h["config"]["arch_params"]
        self.config = h["config"]
        n = int(h["num_params"])
        flat = np.fromfile(path, dtype=np.float32, count=n, offset=off)
        raw = {name: flat[o:o + r * c].reshape(r, c) for name, o, r, c in h["tensors"]}
        self.device = torch.device(device)
        dev = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(self.device)  # noqa: E731

        self.V, self.D, self.L = int(a["vocabulary_size"]), int(a["dim"]), int(a["num_layers"])
        self.N, self.H, self.dh = int(a["lru_dim"]), int(a["num_heads"]), int(a.get("head_dim", 32))
        self.HD = self.H * self.dh
        self.hops = [int(x) for x in (a.get("hops") or [2] * self.L)]
        self.factorized = bool(a.get("factorized", True))
        self.tied = bool(a.get("tied_embeddings", True))
        ns = [int(x) for x in (a.get("bank_slots") or [SLOTS] * self.L)]
        hl_min, hl_max = (float(x) for x in a.get("slot_half_life", [1, 1024]))
        hl_max_l = [float(x) for x in (a.get("slot_half_life_max") or [hl_max] * self.L)]
        self.num_heads, self.slots, self.num_layers, self.layer_hops = self.H, SLOTS, self.L, self.hops
        self.epoch, self.step_count = h.get("epoch"), h.get("step")

        # slot decays as Model::init: half-lives log-spaced over [hl_min, hl_max(layer)] across the layer's bank size
        self.loga = np.zeros((self.L, self.H, SLOTS))
        for li in range(self.L):
            for m in range(SLOTS):
                mm = min(m, ns[li] - 1)
                hl = math.exp(math.log(hl_min) + (math.log(hl_max_l[li]) - math.log(hl_min)) * mm / max(ns[li] - 1, 1))
                self.loga[li, :, m] = np.float32(math.log(0.5) / hl)
        self.slot_half_life = (np.log(0.5) / self.loga[:, 0]).round(1).tolist()  # [layer][slot]

        self.E = dev(raw["E"])  # V x E0
        self.E_out = dev(raw["E_out"]) if not self.tied else self.E
        g = raw["final_norm"][0]
        if self.factorized:
            self.W_in = dev(raw["W_in"])
            self.W_out = dev(g[:, None] * raw["W_out"])  # D x E0
        else:
            self.final_gain = dev(g)
        self.final_norm_np = g
        N, H, dh, HD = self.N, self.H, self.dh, self.HD
        self.layers: List[Dict[str, torch.Tensor]] = []
        for li in range(self.L):
            p = lambda k: raw[f"l{li}_{k}"]  # noqa: E731
            nu, theta = p("nu")[0].astype(np.float64), p("theta")[0].astype(np.float64)
            rho = np.exp(nu)
            mag = np.exp(-rho)
            lr, lim = mag * np.cos(theta), mag * np.sin(theta)
            gam = np.sqrt(np.maximum(-np.expm1(-2 * rho), 1e-12))
            mask = np.zeros((H, SLOTS), np.float32)
            mask[:, ns[li]:] = -np.inf
            w = {
                "Wa": dev(p("norm1")[0][:, None] * p("Wa")),
                "LR": dev(np.stack([lr, lr])), "LI": dev(np.stack([-lim, lim])), "GAM": dev(np.stack([gam, gam])),
                "Ws": dev(p("Ws")), "bk": dev(p("bk")[0].reshape(H, dh)),
                "brg": dev(p("brg")[0]), "bwg": dev(p("bwg")[0]),
                "R": dev(p("R").reshape(H, dh, SLOTS)), "rmask": dev(mask),
                "tau": dev(np.exp(p("log_tau")[0])[:, None]),
                "A": dev(np.exp(self.loga[li])),
                "Wo": dev(p("Wo")), "W1": dev(p("ffn_norm")[0][:, None] * p("W1")), "b1": dev(p("b1")[0]),
                "W2": dev(p("W2")), "b2": dev(p("b2")[0]),
            }
            if self.hops[li] == 2:
                w.update({"Wq2": dev(p("q2_norm")[0][:, None] * p("Wq2")), "brg2": dev(p("brg2")[0]),
                          "tau2": dev(np.exp(p("log_tau2")[0])[:, None]), "Wo2": dev(p("Wo2"))})
                if f"l{li}_W1b" in raw:
                    w.update({"W1b": dev(p("ffn2_norm")[0][:, None] * p("W1b")), "b1b": dev(p("b1b")[0]),
                              "W2b": dev(p("W2b")), "b2b": dev(p("b2b")[0])})
            self.layers.append(w)
        self.num_params = n
        self._decoder: Optional[Decoder] = None

    # -- the pieces shared by Decoder and the analysis -----------------------------------------------------------------

    def embed(self, tok: torch.Tensor) -> torch.Tensor:
        e = self.E.index_select(0, tok)
        return e @ self.W_in if self.factorized else e

    def readout_input(self, h: torch.Tensor) -> torch.Tensor:
        return _rms(h) @ self.W_out if self.factorized else _rms(h) * self.final_gain

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.readout_input(h) @ self.E_out.T

    def lens_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """(G, E): the logit effect of a stream addition x is (x G) E^T, through the final norm's gain (factorized:
        and W_out)."""
        E = self.E_out.float().cpu().numpy()
        if self.factorized:
            return self.W_out.float().cpu().numpy(), E
        return np.diag(self.final_norm_np).astype(np.float32), E

    def decoder(self) -> "Decoder":
        """The generation decoder (one document state, its device graph captured on first use)."""
        if self._decoder is None:
            self._decoder = Decoder(self)
        return self._decoder


class Decoder:
    """One document's recurrent state on the device and the per-token step over it (graph-captured when the device
    supports it). `record`: also store per position (up to `record` tokens) what Inspect needs."""

    def __init__(self, model: LbcModel, record: int = 0, graph: Optional[bool] = None):
        self.m = m = model
        dev, L, H, dh, N = m.device, m.L, m.H, m.dh, m.N
        z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
        self.s, self.gprev = z(L, 2, N), z(L, H, dh)
        self.Kb, self.Vb, self.Z = z(L, H, SLOTS, dh), z(L, H, SLOTS, dh), z(L, H, SLOTS)
        self.tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.keep = z(1)  # 0 at <bos>: the state restarts and nothing is written
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.out = z(m.V)
        self.record = int(record)
        if self.record:
            T = self.record
            self.rec = {"h": z(T, L + 1, m.D), "w": z(T, L, H, SLOTS), "v": z(T, L, H, dh),
                        "p": z(T, L, 2, H, SLOTS), "o": z(T, L, 2, H, dh), "g": z(T, L, 3, H)}
        self.n = 0  # tokens fed since the last reset
        if graph is None:
            graph = dev.type in ("xpu", "cuda")
        self.graph: Any = None
        self._want_graph = graph

    def reset(self) -> None:
        for t in (self.s, self.gprev, self.Kb, self.Vb, self.Z):
            t.zero_()
        self.n = 0

    # -- the step ----------------------------------------------------------------------------------------------------

    def _put(self, key: str, x: torch.Tensor) -> None:
        self.rec[key].index_copy_(0, self.pos, x.unsqueeze(0))

    def _read(self, li: int, q: torch.Tensor, tau: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        Z = self.Z[li]
        valid = Z > BANK_EPS
        sc = torch.where(valid, tau * (self.Kb[li] @ q.unsqueeze(-1)).squeeze(-1), -1e30)
        p = torch.softmax(sc, -1) * valid * gate.unsqueeze(-1)  # all empty: uniform x 0 = 0
        return p, (p.unsqueeze(1) @ self.Vb[li]).squeeze(1)  # (H, M), (H, dh)

    def _ffn(self, h: torch.Tensor, W1, b1, W2, b2) -> torch.Tensor:
        return h + torch.addmm(b2, F.gelu(torch.addmm(b1, _rms(h), W1), approximate="tanh"), W2)

    def _step(self) -> None:
        m, keep, rec = self.m, self.keep, self.record
        H, dh, HD, N = m.H, m.dh, m.HD, m.N
        h = m.embed(self.tok)  # (1, D)
        hs = [h] if rec else None
        for li, w in enumerate(m.layers):
            pv = _rms(h) @ w["Wa"]
            P, v = pv[0, :2 * N].view(2, N), pv[0, 2 * N:].view(H, dh)
            s = self.s[li]
            s.copy_(keep * (s * w["LR"] + s.flip(0) * w["LI"]) + w["GAM"] * P)
            G = (s.view(1, 2 * N) @ w["Ws"])[0]
            k = F.normalize(self.gprev[li] * keep + w["bk"], dim=-1, eps=NORM_EPS)
            self.gprev[li].copy_(G[:HD].view(H, dh))
            q = F.normalize(G[HD:2 * HD].view(H, dh), dim=-1, eps=NORM_EPS)
            gr = (G[2 * HD:2 * HD + H] + w["brg"] > 0).float()
            gw = (G[2 * HD + H:2 * HD + 2 * H] + w["bwg"] > 0).float() * keep
            lg = (k.unsqueeze(1) @ w["R"]).squeeze(1) + w["rmask"]  # (H, M)
            top, idx = lg.topk(2, dim=-1)
            p1 = torch.sigmoid(top[:, :1] - top[:, 1:])
            r = torch.zeros_like(lg).scatter_(1, idx, torch.cat([p1, 1 - p1], 1)) * gw.unsqueeze(1)  # (H, M)
            Zn = self.Z[li] * w["A"] * keep + r
            c = (r / Zn.clamp_min(1e-30)).unsqueeze(-1)
            self.Kb[li].add_(c * (k.unsqueeze(1) - self.Kb[li]))
            self.Vb[li].add_(c * (v.unsqueeze(1) - self.Vb[li]))
            self.Z[li].copy_(Zn)
            p, o = self._read(li, q, w["tau"], gr)
            h = h + o.reshape(1, HD) @ w["Wo"]
            h = self._ffn(h, w["W1"], w["b1"], w["W2"], w["b2"])
            if rec:
                gates = [gw, gr]
                ps, os_ = [p], [o]
            if m.hops[li] == 2:
                G2 = (_rms(h) @ w["Wq2"])[0]
                q2 = F.normalize(G2[:HD].view(H, dh), dim=-1, eps=NORM_EPS)
                gr2 = (G2[HD:HD + H] + w["brg2"] > 0).float()
                p2, o2 = self._read(li, q2, w["tau2"], gr2)
                h = h + o2.reshape(1, HD) @ w["Wo2"]
                if "W1b" in w:
                    h = self._ffn(h, w["W1b"], w["b1b"], w["W2b"], w["b2b"])
                if rec:
                    gates.append(gr2)
                    ps.append(p2)
                    os_.append(o2)
            if rec:
                if len(ps) == 1:
                    gates.append(torch.zeros_like(gr))
                    ps.append(torch.zeros_like(p))
                    os_.append(torch.zeros_like(o))
                hs.append(h)
                self.rec["v"][:, li].index_copy_(0, self.pos, v.unsqueeze(0))
                self.rec["w"][:, li].index_copy_(0, self.pos, r.unsqueeze(0))
                self.rec["p"][:, li].index_copy_(0, self.pos, torch.stack(ps).unsqueeze(0))
                self.rec["o"][:, li].index_copy_(0, self.pos, torch.stack(os_).unsqueeze(0))
                self.rec["g"][:, li].index_copy_(0, self.pos, torch.stack(gates).unsqueeze(0))
        if rec:
            self._put("h", torch.cat(hs, 0))
        else:
            self.out.copy_(m.logits(h)[0])

    def _capture(self) -> None:
        """Captures the step as a device graph (after warm-up runs on a scratch copy of the state)."""
        dev = self.m.device
        saved = [t.clone() for t in (self.s, self.gprev, self.Kb, self.Vb, self.Z)]
        mod = torch.xpu if dev.type == "xpu" else torch.cuda
        side = mod.Stream()
        side.wait_stream(mod.current_stream())
        with mod.stream(side):
            for _ in range(2):
                self._step()
        mod.current_stream().wait_stream(side)
        g = mod.XPUGraph() if dev.type == "xpu" else mod.CUDAGraph()
        with (torch.xpu.graph(g) if dev.type == "xpu" else torch.cuda.graph(g)):
            self._step()
        for t, x in zip((self.s, self.gprev, self.Kb, self.Vb, self.Z), saved):
            t.copy_(x)
        self.graph = g

    @torch.no_grad()
    def feed(self, token: int) -> torch.Tensor:
        """Feeds one token (a <bos> restarts the state) and returns the next-token logits (a device buffer, valid
        until the next call; empty in record mode)."""
        if self.record and self.n >= self.record:
            raise ValueError(f"the recording holds {self.record} tokens")
        self.tok.fill_(int(token))
        self.keep.fill_(0.0 if token == BOS or self.n == 0 else 1.0)
        self.pos.fill_(self.n)
        if self._want_graph and self.graph is None:
            try:
                self._capture()
            except Exception as e:  # noqa: BLE001 (eager instead)
                print(f"lbc_torch: no device graph ({type(e).__name__}: {e}); running eager", flush=True)
                self._want_graph = False
        if self.graph is not None:
            self.graph.replay()
        else:
            self._step()
        self.n += 1
        return self.out


# -- generation ---------------------------------------------------------------------------------------------------------

def sample(logits: torch.Tensor, temperature: float = 0.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
           repetition_penalty: float = 1.0, seen: Sequence[int] = (), rng: Optional[np.random.Generator] = None) -> int:
    """The next token: greedy at temperature 0, else sampled (top-k / top-p) on the host."""
    if temperature <= 0 and repetition_penalty == 1.0:
        return int(torch.argmax(logits).item())
    from model.generation import _apply_repetition_penalty, _filter_top_k, _filter_top_p

    x = logits.float().cpu().numpy().astype(np.float64)
    _apply_repetition_penalty(x, seen, repetition_penalty)
    if temperature <= 0:
        return int(np.argmax(x))
    x /= temperature
    if top_k:
        _filter_top_k(x, int(top_k))
    if top_p:
        _filter_top_p(x, float(top_p))
    p = np.exp(x - x.max())
    p /= p.sum()
    return int((rng or np.random.default_rng()).choice(len(p), p=p))


def stream(model: LbcModel, ids: Sequence[int], max_new_tokens: int = 256, temperature: float = 0.0,
           top_k: Optional[int] = None, top_p: Optional[float] = None, repetition_penalty: float = 1.0,
           stop_ids: Sequence[int] = (), session: Any = None, cancel: Any = None) -> Iterator[int]:
    """`model.generation.stream` for lbc models: yields each new token id (the stop token included). `session`
    (`model.generation.Session`): continue from the decoder's state when `ids` extend the ids it covers."""
    dec = model.decoder()
    feed = list(ids)
    if not feed or feed[0] != BOS:
        feed = [BOS] + feed
    if session is not None and session.ids and 0 < len(session.ids) < len(feed) and feed[:len(session.ids)] == session.ids \
            and session.state is dec:
        fed, feed = list(session.ids), feed[len(session.ids):]
    else:
        dec.reset()
        fed = []
    logits = None
    for t in feed:
        if cancel is not None and cancel.is_set():
            return
        logits = dec.feed(t)
    fed += feed
    stop = set(stop_ids)
    pending = None  # yielded, not fed yet
    for _ in range(max_new_tokens):
        if cancel is not None and cancel.is_set():
            break
        if pending is not None:
            logits = dec.feed(pending)
            fed.append(pending)
            pending = None
        nxt = sample(logits, temperature, top_k, top_p, repetition_penalty, fed)
        pending = nxt
        yield nxt
        if nxt in stop:
            break
    if pending is not None:  # the last token too, so the state covers everything that was yielded
        dec.feed(pending)
        fed.append(pending)
    if session is not None:
        session.ids, session.state = fed, dec


# -- Inspect -------------------------------------------------------------------------------------------------------------

_recorders: Dict[int, Decoder] = {}


@torch.no_grad()
def trace(model: LbcModel, ids: Sequence[int], max_tokens: int = 1024) -> Dict[str, np.ndarray]:
    """Runs `ids` (starting with <bos>) through a recording decoder: per position the stream after the embedding and
    each layer `h (T, L+1, D)`, the write weights `w (T, L, H, M)` (routing x write gate), values `v (T, L, H, dh)`,
    read probabilities `p (T, L, 2, H, M)` and reads `o (T, L, 2, H, dh)` (x the read gate; zeros for a missing hop 2),
    the gates `g (T, L, 3, H)` (write, read 1, read 2) and the final logits `(T, V)`."""
    dec = _recorders.get(id(model))
    if dec is None:
        dec = _recorders[id(model)] = Decoder(model, record=max_tokens)
    ids = list(ids)[:max_tokens]
    dec.reset()
    for t in ids:
        dec.feed(t)
    T = len(ids)
    out = {k: v[:T].float().cpu().numpy() for k, v in dec.rec.items()}
    hl = dec.rec["h"][:T, -1]
    out["logits"] = torch.cat([model.logits(hl[i:i + 256]) for i in range(0, T, 256)]).float().cpu().numpy()
    return out


# -- command line --------------------------------------------------------------------------------------------------------

def main() -> None:
    import argparse
    import time

    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser(description="Generate from an lbc checkpoint (torch, fp32).")
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", default="tokenizers/agentic-bpe32768/tokenizer.json")
    ap.add_argument("--prompt", default="<user>What is 17 * 23?</user><assistant>")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--eager", action="store_true", help="no device graph")
    ap.add_argument("--bench", action="store_true", help="time the decode of max-new-tokens tokens, no stop")
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    t0 = time.perf_counter()
    model = LbcModel(a.checkpoint, a.device)
    if a.eager:
        model._decoder = Decoder(model, graph=False)
    print(f"loaded {model.num_params / 1e6:.1f}M params, epoch {model.epoch}, in {time.perf_counter() - t0:.1f} s",
          flush=True)
    ids = tok.encode(a.prompt).ids
    stop = [] if a.bench else [tok.token_to_id("</assistant>"), tok.token_to_id("<eos>")]
    stamps, new = [], []
    t0 = time.perf_counter()
    for i in stream(model, ids, a.max_new_tokens, a.temperature, a.top_k or None, stop_ids=stop):
        stamps.append(time.perf_counter())
        new.append(i)
        if not a.bench:
            print(tok.decode([i], skip_special_tokens=False), end="", flush=True)
    print()
    if len(stamps) > 1:
        print(f"prompt {len(ids)} tokens + first token: {stamps[0] - t0:.2f} s; decode "
              f"{(len(stamps) - 1) / (stamps[-1] - stamps[0]):.1f} tokens/s over {len(stamps) - 1} tokens", flush=True)


if __name__ == "__main__":
    main()
