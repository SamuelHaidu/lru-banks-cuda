"""`min-gru`: a stack of minGRU layers (Feng et al. 2024, "Were RNNs All We Needed?") that write into a residual
stream through tanh. Linear in the context length for compute and memory, plain PyTorch (runs on XPU, CUDA, CPU).

One layer (n), at every position t:

    layer_input        = rms_norm(residual_stream)                              * norm_in
    gate_logit, raw    = split(layer_input @ in_proj + in_bias)                   dim -> 2 x hidden_size
    update_gate        = sigmoid(gate_logit)                                      0..1: how much to replace
    keep_factor        = 1 - update_gate      (0 at the document marker: the state restarts there)
    candidate_state    = tanh(raw)                                                -1..1
    state_after_update = keep_factor * state_before_update + update_gate * candidate_state
    layer_write        = write_scale * tanh(rms_norm(state_after_update) * norm_state @ out_proj)
                         (`mlp_mult` k: write_scale * mlp_out(gelu(rms_norm(state) * norm_state @ mlp_in + mlp_bias)),
                          hidden k x dim, instead of the tanh projection)
    residual_stream    = residual_stream + layer_write

Then logits = rms_norm(residual_stream) * final_norm @ readout.

Why it is stable:
  - each channel of the state is a running weighted average of tanh candidates (the weights sum to 1), so it stays
    in -1..1 over any length;
  - every layer adds at most `write_scale` (learned per channel, init 0.1) per channel to the stream, so the stream
    grows at most linearly with depth, and the identity path carries gradients to every layer;
  - rms_norm before every read of the stream and of the state: each layer sees a fixed scale at any depth.

Why it is fast: the gate and the candidate depend on the current token only (not on the previous state), so a
whole row is a few matmuls plus one linear scan `state[t] = keep[t] * state[t-1] + write[t]` (`linear_scan`).
The scan is chunked: inside chunks of ~sqrt(T) positions a loop over the chunk's positions runs every chunk at
once, then a loop over the chunks carries the state, then one pass adds the carried state. O(T) work, O(sqrt(T))
kernel launches, in linear space (keep factors are in 0..1, so products only underflow to 0, harmlessly). Its backward
is the same scan run backwards; it stores only the keep factors and the states. Every other activation is per token,
so training memory is linear in the row length (`recompute` trades compute for it: each layer recomputes its
activations in the backward pass).

Interpretable by construction: each layer's state is `hidden_size` numbers in -1..1, each channel with its own
timescale (`update_gate` near 0: keeps; near 1: overwrites). `inspect(ids)` returns every gate, candidate, state and
write per position; `ad_hoc_scripts/mingru_inspect.py` prints them for a text.

Agent data: the dataset's `loss_spans` make every non-assistant target ignored (-1). The dense train step then
reads out only the positions with a target, and the last layer's write runs at those positions only
(`dense_hidden_valid`): no compute or capacity goes into predicting prompts or tool results; the state still reads
them.

Infinite context like `wave-arrow`: packed document rows, the state restarts at `reset_token`, generation carries the
state token by token (`advance`). `window_size` is only the training row length.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model.archs import _context  # noqa: E402
from model.archs._dense import DenseSupervisionMixin  # noqa: E402
from model.interfaces import BaseLM  # noqa: E402
from model.memory_rows import READ_CHUNK  # noqa: E402


# -- the scan -------------------------------------------------------------------------------------------------------

def _chunk_length(T: int) -> int:
    """A divisor of T near sqrt(T) (chunks tile the row exactly: no padding, no copies), 0 if T has none between
    sqrt(T) / 2 and 2 sqrt(T). The chunk length barely matters in that range (measured at T = 1024 and 4096)."""
    root, best = math.sqrt(T), 0
    for L in range(max(1, int(root / 2)), int(2 * root) + 1):
        if T % L == 0 and (not best or abs(math.log(L / root)) < abs(math.log(best / root))):
            best = L
    return best


def _heads(keep: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """keep[..., g] * x[..., g * hd:(g + 1) * hd]: a keep factor per head (G of them) on x's D = G hd channels."""
    G = keep.shape[-1]
    return (keep[..., None] * x.reshape(*x.shape[:-1], G, x.shape[-1] // G)).reshape(x.shape)


def _scan(keep: torch.Tensor, write: torch.Tensor, initial: Optional[torch.Tensor], reverse: bool = False,
          out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """state[t] = keep[t] * state[t-1] + write[t] along dim 1 of `(B, T, D)`, state[-1] = `initial` `(B, D)` (zeros
    if None), written into `out` (new if None). `keep` is `(B, T, G)`: one factor per channel (G = D) or per head of
    D / G channels (`gate_heads`), broadcast over the head's channels. No autograd (see `LinearScan`).

    `reverse` (the backward pass): g[t] = write[t] + keep[t+1] * g[t+1] from the end, g[T-1] = write[T-1] +
    `initial` (the term from past the end, zeros if None): the same `keep` tensor, read one position later, so the
    backward needs no shifted or flipped copies.

    Chunks of L ~ sqrt(T) positions (L divides T):
      1. inside every chunk at once, from a zero state: in_chunk[t] = keep[t] * in_chunk[t-1] + write[t], and the
         keep product since the chunk start, kept[t] = keep[chunk start] * ... * keep[t] (a loop over L positions);
      2. the state entering each chunk: entering[c+1] = in_chunk[c, last] + kept[c, last] * entering[c] (a loop over
         the chunks, (B, D) each);
      3. state = in_chunk + kept * entering (one pass).
    If T has no divisor near sqrt(T): the longest prefix that is a multiple of sqrt(T), then the rest from where the
    prefix ends (two calls into the same `out`)."""
    B, T, D = write.shape
    G = keep.shape[-1]
    if out is None:
        out = torch.empty((B, T, D), device=write.device, dtype=write.dtype)
    L = _chunk_length(T)
    if not L:
        main = (T // math.ceil(math.sqrt(T))) * math.ceil(math.sqrt(T))
        if not reverse:
            _scan(keep[:, :main], write[:, :main], initial, out=out[:, :main])
            _scan(keep[:, main:], write[:, main:], out[:, main - 1], out=out[:, main:])
        else:
            _scan(keep[:, main:], write[:, main:], initial, reverse=True, out=out[:, main:])
            _scan(keep[:, :main], write[:, :main], _heads(keep[:, main], out[:, main]), reverse=True, out=out[:, :main])
        return out
    C, hd = T // L, D // G
    keep = keep.reshape(B, C, L, G, 1)  # broadcast over the hd channels of each head
    in_chunk = out.view(B, C, L, G, hd)
    in_chunk.copy_(write.reshape(B, C, L, G, hd))
    state = (initial if initial is not None else out.new_zeros(B, D)).reshape(B, G, hd)
    entering = in_chunk.new_empty(B, C, G, hd)
    if not reverse:
        kept = keep.clone()
        for t in range(1, L):
            in_chunk[:, :, t].addcmul_(keep[:, :, t], in_chunk[:, :, t - 1])
            kept[:, :, t].mul_(kept[:, :, t - 1])
        for c in range(C):
            entering[:, c] = state
            state = torch.addcmul(in_chunk[:, c, -1], kept[:, c, -1], state)
    else:
        # in_chunk[t] = write[t] + keep[t+1] in_chunk[t+1] up to the chunk's end; kept[t] = keep[t+1] ... keep[L-1]
        kept = torch.ones_like(keep)
        for t in range(L - 2, -1, -1):
            in_chunk[:, :, t].addcmul_(keep[:, :, t + 1], in_chunk[:, :, t + 1])
            torch.mul(keep[:, :, t + 1], kept[:, :, t + 1], out=kept[:, :, t])
        # entering[c] = keep[next chunk's first] * g[next chunk's first]; after the last chunk: `initial`
        for c in range(C - 1, -1, -1):
            entering[:, c] = state
            state = keep[:, c, 0] * torch.addcmul(in_chunk[:, c, 0], kept[:, c, 0], state)
    in_chunk.addcmul_(kept, entering[:, :, None])
    return out


class LinearScan(torch.autograd.Function):
    """`linear_scan` with its gradient: the backward of state[t] = keep[t] state[t-1] + write[t] is the reverse scan
    g[t] = d_state[t] + keep[t+1] g[t+1]; then d_write = g, d_keep[t] = g[t] state[t-1] (summed over a head's
    channels with `gate_heads`), d_initial = keep[0] g[0]. Saves only `keep` and the states."""

    @staticmethod
    def forward(ctx: Any, keep: torch.Tensor, write: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            states = _scan(keep, write, initial)
        ctx.save_for_backward(keep, states, initial)
        return states

    @staticmethod
    def backward(ctx: Any, d_states: torch.Tensor):  # type: ignore[override]
        keep, states, initial = ctx.saved_tensors
        g = _scan(keep, d_states, None, reverse=True)
        d_keep = torch.empty_like(g)
        torch.mul(g[:, 1:], states[:, :-1], out=d_keep[:, 1:])
        torch.mul(g[:, 0], initial, out=d_keep[:, 0])
        if keep.shape[-1] != g.shape[-1]:  # per head: the sum over its channels
            d_keep = d_keep.reshape(*g.shape[:2], keep.shape[-1], -1).sum(-1)
        return d_keep, g, _heads(keep[:, 0], g[:, 0])


def linear_scan(keep: torch.Tensor, write: torch.Tensor, initial: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Every state of state[t] = keep[t] * state[t-1] + write[t]: `(B, T, D)` from write `(B, T, D)`, keep
    `(B, T, G)` (G = D: per channel; G heads of D / G channels: per head) and the state before the row `(B, D)`
    (zeros if None). Exact for any T; O(T) work and memory."""
    if initial is None:
        initial = write.new_zeros(write.shape[0], write.shape[2])
    return LinearScan.apply(keep, write, initial)


class RMSNorm(torch.autograd.Function):
    """y = x / rms(x) over the last dim, with a hand-written backward: dx = (dy - y mean(dy y)) / rms(x). Autograd
    would keep and walk x^2, the mean and the rsqrt separately: this saves y and 1 / rms only and takes ~30% less time
    forward + backward (measured on the Iris Xe)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, eps: float) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        y = x * inv_rms
        ctx.save_for_backward(y, inv_rms)
        return y

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor):  # type: ignore[override]
        y, inv_rms = ctx.saved_tensors
        return inv_rms * (dy - y * (dy * y).mean(-1, keepdim=True)), None


class GatherRows(torch.autograd.Function):
    """x[idx] for distinct row indices: the backward copies the gradient rows back (`index_copy`) instead of
    autograd's scatter-add (~6x slower on the Iris Xe, where every add is atomic)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(idx)
        ctx.rows = x.shape[0]
        return x.index_select(0, idx)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor):  # type: ignore[override]
        (idx,) = ctx.saved_tensors
        return g.new_zeros(ctx.rows, *g.shape[1:]).index_copy(0, idx, g), None


def gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return GatherRows.apply(x, idx)


class RepeatSorted(torch.autograd.Function):
    """x[idx] for sorted indices (rows repeated in runs): the backward sums each run of gradient rows with a cumulative
    sum and differences at the run ends, instead of autograd's scatter-add (atomic: ~16 ms per layer for the
    chunks' memories on the Iris Xe)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(idx)
        ctx.rows = x.shape[0]
        return x.index_select(0, idx)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor):  # type: ignore[override]
        (idx,) = ctx.saved_tensors
        ends = torch.cumsum(torch.bincount(idx, minlength=ctx.rows), 0)  # run of row r: [ends[r-1], ends[r])
        sums = torch.cat([g.new_zeros(1, *g.shape[1:]), torch.cumsum(g, 0)])
        return sums.index_select(0, ends) - sums.index_select(0, torch.cat([ends.new_zeros(1), ends[:-1]])), None


def place_rows(src: torch.Tensor, idx: torch.Tensor, rows: int) -> torch.Tensor:
    """Rows `rows` x ... of zeros with src's rows at the distinct indices idx (backward: a gather)."""
    return src.new_zeros(rows, *src.shape[1:]).index_copy(0, idx, src)


def rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    x = RMSNorm.apply(x, eps)
    return x if weight is None else x * weight


def _const(v: Any):
    return lambda shape, dtype=None: keras.ops.convert_to_tensor(np.asarray(v, "float32"), dtype=dtype)


# -- the model ------------------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="model")
class MinGRULM(DenseSupervisionMixin, BaseLM):
    """Args:
        vocabulary_size: Vocabulary size.
        dim: Width of the residual stream (and of the token embedding).
        hidden_size: State channels per layer.
        num_layers: Stacked layers.
        window_size: Training row length (packed documents) and validation window. The state has no window.
        reset_token: Document marker id; every layer's state restarts at it (the dataset's `doc_token_id`).
        half_life_min, half_life_max: Initial timescales of the state channels, log-spaced over the channels (in
            tokens): the gate bias starts where keep_factor = 0.5 ** (1 / half_life), so some channels start fast,
            some slow. Learned from there (per token, the gate also depends on the input).
        write_scale_init: Initial `write_scale` (learned per channel of the stream).
        embedding_std: Token embedding init std.
        recompute: Recompute each layer's activations in the backward pass (2.6 -> 1.0 GiB at 4 x 16384 tokens,
            1.4x the time).
        gate_heads: None (default): one update gate per state channel. G: one gate per head of hidden_size / G
            channels (a sigmoid each, independent), shared by the head's channels: the gate projection is G wide
            instead of hidden_size and the gate math runs on G numbers per token. The heads' initial half-lives are
            log-spaced from half_life_min to half_life_max.
        static_memory: Two compressed key-value memories per layer without decay (each heads x dim x dim, constant
            size), one holding the `<system>` span and one the `<tools>` span of the current document. The span
            contents never enter the layers (the rows hold the rest; their tags stay): a memory encoder (one minGRU
            layer, `mem_enc_*`) runs over each document's span tags and contents only, and token j stores, per layer,
            key = elu(rms_norm(encoder state before j) @ mem_k) + 1 (induction-style: the context that led to it)
            and value = rms_norm(its embedding) @ mem_v (`_memories`: one `(heads, dim, 2 dim + 2)` matrix per
            document and layer). The reading positions (`memory_read_span`) query with elu(state after the update @
            mem_q) + 1, so "my context ends like the context before token j" reads back token j; the state after the
            update also gates each read (a sigmoid per memory and head). The gated reads go through their own
            projection (`mem_out`, init 0) and add to the stream next to the state's write. Batches are dicts
            (`model.memory_rows`; the data loader builds them, raw rows are converted).
        memory_heads, memory_dim: Heads per memory and their key / value size (each memory: heads x dim x dim).
        memory_spans: Token ids [[open, close], ...] of the system and tool spans (the agent datasets' tags).
        memory_read_span: Token ids [open, close] of the spans that read the memories: the positions where the model
            generates (the assistant turns: the opening tag and the tokens inside; the agent datasets' tags). No read
            anywhere else (prompts, results: nothing is predicted there).
        mlp_mult: None (default): each layer writes `write_scale * tanh(rms_norm(state) @ out_proj)`. k: a GELU MLP
            instead, `write_scale * (gelu(rms_norm(state) @ mlp_in + mlp_bias) @ mlp_out)`, hidden k x dim (tanh
            approximation of GELU; mlp_out init std 0.02 like out_proj). The write is no longer bounded by tanh.
        silent_from: Layers from this index on (0-based) start with `write_scale` 0: they write nothing, so a model
            grown from a trained one (`init_weights_from` the trained layers) starts with exactly its outputs;
            `write_scale` learns from the first step (its gradient is tanh(out), not 0), the rest follows. None:
            every layer starts at `write_scale_init`.
    """

    no_weight_decay_names = ("norm", "bias", "write_scale")
    config_defaults = {
        "training": {"learning_rate": 0.003, "warmup_steps": 200, "dense_supervision": True, "dense_validation": True},
        # 32 x 512 rows, one pass per epoch in shuffled windows (rows start anywhere in a document, zero state)
        "data": {"batch_size": 32, "random_windows": True},
    }
    packed_documents = True
    stateful_generation = True

    def __init__(self, vocabulary_size: int = 10000, dim: int = 128, hidden_size: int = 256, num_layers: int = 4,
                 window_size: int = 512, reset_token: int = 0, half_life_min: float = 1.0,
                 half_life_max: float = 1024.0, write_scale_init: float = 0.1, embedding_std: float = 0.02,
                 recompute: bool = False, silent_from: Optional[int] = None, gate_heads: Optional[int] = None,
                 static_memory: bool = False, memory_heads: int = 2, memory_dim: int = 64,
                 memory_spans: Optional[List[List[int]]] = None, memory_read_span: Optional[List[int]] = None,
                 mlp_mult: Optional[int] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._vocabulary_size = int(vocabulary_size)
        self.dim, self.hidden_size, self.num_layers = int(dim), int(hidden_size), int(num_layers)
        self.window_size, self.reset_token = int(window_size), int(reset_token)
        self.half_life_min, self.half_life_max = float(half_life_min), float(half_life_max)
        self.write_scale_init = float(write_scale_init)
        self.embedding_std, self.recompute = float(embedding_std), bool(recompute)
        self.silent_from = None if silent_from is None else int(silent_from)
        self.gate_heads = None if gate_heads is None else int(gate_heads)
        if self.gate_heads is not None and (self.gate_heads < 1 or self.hidden_size % self.gate_heads):
            raise ValueError(f"gate_heads must divide hidden_size {self.hidden_size}, got {gate_heads}")
        self.gates = self.gate_heads or self.hidden_size  # update gates per token and layer
        self.static_memory = bool(static_memory)
        self.mlp_mult = None if mlp_mult is None else int(mlp_mult)
        self.memory_heads, self.memory_dim = int(memory_heads), int(memory_dim)
        self.memory_spans = [list(map(int, s)) for s in (memory_spans or [[12, 13], [2, 3]])]  # <system>, <tools>
        self.memory_read_span = list(map(int, memory_read_span or [6, 7]))  # <assistant>
        self.token_embedding = layers.Embedding(
            self._vocabulary_size, self.dim, name="token_embedding",
            embeddings_initializer=keras.initializers.RandomNormal(stddev=self.embedding_std))
        self.readout = layers.Dense(self._vocabulary_size, use_bias=False, name="readout")
        self.build()

    @property
    def context_length(self) -> int:
        return self.window_size

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    def build(self, input_shape: Optional[Any] = None) -> None:
        if self.built:
            return
        D, H = self.dim, self.hidden_size
        self.token_embedding.build((None,))
        self.readout.build((None, D))
        self.final_norm = self.add_weight(name="final_norm", shape=(D,), initializer="ones")
        G = self.gates
        half_lives = np.geomspace(self.half_life_min, self.half_life_max, G)
        keep0 = 0.5 ** (1.0 / half_lives)
        gate_bias = np.log((1.0 - keep0) / keep0)  # sigmoid(gate_bias) = update_gate = 1 - keep0
        self._params: List[Dict[str, Any]] = []
        for n in range(self.num_layers):
            w = lambda name, shape, init: self.add_weight(name=f"l{n}_{name}", shape=shape, initializer=init)
            self._params.append({
                "norm_in": w("norm_in", (D,), "ones"),
                "in_proj": w("in_proj", (D, G + H), "glorot_uniform"),
                "in_bias": w("in_bias", (G + H,), _const(np.concatenate([gate_bias, np.zeros(H)]))),
                "norm_state": w("norm_state", (H,), "ones"),
                **({"out_proj": w("out_proj", (H, D), keras.initializers.RandomNormal(stddev=0.02))}
                   if self.mlp_mult is None else {
                    "mlp_in": w("mlp_in", (H, self.mlp_mult * D), "glorot_uniform"),
                    "mlp_bias": w("mlp_bias", (self.mlp_mult * D,), "zeros"),
                    "mlp_out": w("mlp_out", (self.mlp_mult * D, D), keras.initializers.RandomNormal(stddev=0.02))}),
                "write_scale": w("write_scale", (D,), _const(np.full(
                    D, 0.0 if self.silent_from is not None and n >= self.silent_from else self.write_scale_init))),
            })
            if self.static_memory:
                h, d, M = self.memory_heads, self.memory_dim, len(self.memory_spans) * self.memory_heads
                self._params[-1].update({
                    "mem_q": w("mem_q", (H, h * d), "glorot_uniform"),
                    "mem_k": w("mem_k", (H, h * d), "glorot_uniform"),
                    "mem_v": w("mem_v", (D, h * d), "glorot_uniform"),
                    "mem_gate": w("mem_gate", (H, M), "zeros"),
                    "mem_gate_bias": w("mem_gate_bias", (M,), "zeros"),
                    "mem_out": w("mem_out", (M * d, D), "zeros"),
                })
        if self.static_memory:  # the memory encoder: one minGRU layer over the span tokens (a gate per channel)
            keep0 = 0.5 ** (1.0 / np.geomspace(self.half_life_min, self.half_life_max, H))
            self._mem_enc = {
                "norm_in": self.add_weight(name="mem_enc_norm_in", shape=(D,), initializer="ones"),
                "in_proj": self.add_weight(name="mem_enc_in_proj", shape=(D, 2 * H), initializer="glorot_uniform"),
                "in_bias": self.add_weight(name="mem_enc_in_bias", shape=(2 * H,), initializer=_const(
                    np.concatenate([np.log((1.0 - keep0) / keep0), np.zeros(H)]))),
            }
        self.built = True

    # -- one layer --------------------------------------------------------------------------------------------------

    def scan(self, keep: torch.Tensor, write: torch.Tensor, initial: Optional[torch.Tensor]) -> torch.Tensor:
        """Every state of state[t] = keep[t] * state[t-1] + write[t] (`linear_scan`; `min-gru-cuda` overrides it)."""
        return linear_scan(keep, write, initial)

    def _layer(self, n: int, residual: torch.Tensor, not_reset: torch.Tensor, initial: Optional[torch.Tensor],
               rows: Optional[torch.Tensor] = None, collect: bool = False, memory: Optional[Dict[str, Any]] = None,
               ends_at: Optional[torch.Tensor] = None):
        """Layer n on the stream `(B, T, dim)`. `rows` (flat indices into B * T): return the stream at those positions
        only `(R, dim)` (the write runs there only; the scan still runs everywhere). `memory` (`static_memory`): the
        documents' memories and the reading positions (`_memory_reads`). `ends_at` `(B,)`: the position whose state
        is the row's end state (-1: `initial`), else the last. Returns (stream, end state `(B, hidden)`, the
        internals if `collect`)."""
        p = {k: v.value for k, v in self._params[n].items()}
        H = self.hidden_size
        # rms_norm(x) * w @ W == rms_norm(x) @ (w[:, None] * W): the norm weight is folded into the (small) matrix,
        # saving a full pass over the activations forward and backward (same for norm_state)
        layer_input = rms_norm(residual)
        gate_logit, raw = (layer_input @ (p["norm_in"][:, None] * p["in_proj"]) + p["in_bias"]).split(
            [self.gates, H], -1)
        update_gate = torch.sigmoid(gate_logit)  # (B, T, gates): per channel, or per head with gate_heads
        keep_factor = (1 - update_gate) * not_reset[..., None]  # 0 at a document marker
        candidate = torch.tanh(raw)
        states = self.scan(keep_factor, _heads(update_gate, candidate), initial)
        if ends_at is None:
            end_state = states[:, -1]
        else:
            B = states.shape[0]
            end_state = states[torch.arange(B, device=states.device), ends_at.clamp_min(0)]
            if initial is not None or bool((ends_at < 0).any()):
                end_state = torch.where((ends_at < 0)[:, None], initial if initial is not None else 0.0, end_state)
        state_n = rms_norm(states)  # the layer's output reads it; with static_memory also the memory queries and gates
        memory_add = memory_gate = None
        if memory is not None:  # the system / tool memories, read at the reading positions only
            memory_add, memory_gate, memory_query = self._memory_reads(p, state_n, memory["C"][n], memory)
            residual = residual + place_rows(memory_add, memory["read_pos"], residual.shape[0] * residual.shape[1]
                                             ).reshape(residual.shape)
        if rows is not None:
            state_n = state_n.reshape(-1, H).index_select(0, rows)
            residual = residual.reshape(-1, self.dim).index_select(0, rows)
        if self.mlp_mult is None:
            layer_write = p["write_scale"] * torch.tanh(state_n @ (p["norm_state"][:, None] * p["out_proj"]))
        else:
            hid = F.gelu(state_n @ (p["norm_state"][:, None] * p["mlp_in"]) + p["mlp_bias"], approximate="tanh")
            layer_write = p["write_scale"] * (hid @ p["mlp_out"])
        residual = residual + layer_write
        info = None
        if collect:
            if self.gates != H:  # per channel for inspection (each channel shows its head's gate)
                update_gate, keep_factor = (x.repeat_interleave(H // self.gates, -1) for x in (update_gate, keep_factor))
            info = {"update_gate": update_gate, "keep_factor": keep_factor, "candidate": candidate, "state": states,
                    "layer_write": layer_write}
            if memory is not None:  # per position (zeros where nothing reads): the read's add, gates and query
                B, T = states.shape[:2]
                info.update({k: place_rows(v, memory["read_pos"], B * T).reshape(B, T, -1) for k, v in
                             (("memory_add", memory_add), ("memory_gate", memory_gate), ("memory_query", memory_query))})
        return residual, end_state, info

    # -- the static memory ------------------------------------------------------------------------------------------

    def _memory_reads(self, p: Dict[str, torch.Tensor], state_n: torch.Tensor, C: torch.Tensor,
                      memory: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The memory's addition to the stream at the reading positions `(R, dim)`, the gates and the queries. Query q =
        elu(state_n @ mem_q) + 1 at a reading position of document g reads each memory i: [S; z] = C_gi^T q,
        read = S / z per head (a weighted average of the values stored, weights q . k_j); gate = sigmoid(state_n @
        mem_gate + bias) per memory and head. The queries go to chunks of READ_CHUNK positions reading one document
        each: one batched product with the chunks' memories."""
        h, d, nm = self.memory_heads, self.memory_dim, len(self.memory_spans)
        chunk_doc, slot = memory["chunk_doc"], memory["read_slot"]
        Nc, Lc = chunk_doc.shape[0], READ_CHUNK
        sn = gather_rows(state_n.reshape(-1, self.hidden_size), memory["read_pos"])  # (R, hidden)
        q_raw, g_raw = (sn @ torch.cat([p["mem_q"], p["mem_gate"]], 1)).split([h * d, nm * h], -1)
        q_r = F.elu(q_raw) + 1
        q = place_rows(q_r, slot, Nc * Lc).reshape(Nc, 1, Lc, h, d).transpose(2, 3)  # (Nc, 1, h, Lc, d)
        out = q @ RepeatSorted.apply(C, chunk_doc)  # (Nc, nm, h, Lc, d + 1)
        out = gather_rows(out.permute(0, 3, 1, 2, 4).reshape(Nc * Lc, nm, h, d + 1), slot)  # (R, nm, h, d + 1)
        gate = torch.sigmoid(g_raw + p["mem_gate_bias"])  # (R, memories x heads)
        # z = 0: an empty memory (no such span in the document; its read is 0). Dividing by 1 there, not by 1e-6,
        # keeps its (unused) gradient bounded: 1e7-sized rows wrecked the precision of RepeatSorted's cumsum
        z = out[..., d]
        z = torch.where(z > 0, z.clamp_min(1e-6), torch.ones_like(z))
        rd = out[..., :d] * (gate.reshape(-1, nm, h) / z)[..., None]
        return rd.reshape(-1, nm * h * d) @ p["mem_out"], gate, q_r

    def _memories(self, mb: Dict[str, torch.Tensor], enc_initial: Optional[torch.Tensor] = None,
                  entry_keys: bool = False):
        """Every layer's memories of each document `(layers, G, memories, heads, d, d + 1)` and the encoder states
        `(E * Le, hidden)` (flat). The encoder (one minGRU layer, `mem_enc_*`) runs over the documents' span tags and
        contents; entry j (a content token of memory i) stores key k_j = elu(rms_norm(encoder state before j) @ mem_k)
        + 1 and value v_j = rms_norm(embedding_j) @ mem_v, per layer: C_i = sum_j k_j [v_j, 1] (the last column sums
        the keys: the read's normalizer). Entries come in chunks of one (document, memory): K^T [V, 1] per chunk, then
        one 0/1 matrix product sums each document's chunks. `enc_initial` `(B, hidden)`: carried encoder states (rows
        `enc_init`). `entry_keys`: also every entry's keys `(entries, layers, heads, d)` (inspection)."""
        E_ = self.token_embedding.embeddings.value
        H, D, L = self.hidden_size, self.dim, self.num_layers
        h, d, nm = self.memory_heads, self.memory_dim, len(self.memory_spans)
        enc, seg = mb["enc"], mb["mem_seg"]
        G, Nck, ch = mb["mem_count"].shape[0], seg.shape[0], READ_CHUNK
        Er, Le = enc.shape
        if Er:
            tok = mb["enc_tok"]  # the documents' tokens (the filling stays zero: it is never read)
            x = place_rows(rms_norm(E_.index_select(0, enc.reshape(-1).index_select(0, tok))), tok, Er * Le)
            w = self._mem_enc
            gate_logit, raw = (x @ (w["norm_in"].value[:, None] * w["in_proj"].value) + w["in_bias"].value).split([H, H], -1)
            update_gate = torch.sigmoid(gate_logit).reshape(Er, Le, H)
            keep = (1 - update_gate) * (enc != self.reset_token).to(x.dtype)[..., None]
            init = None
            if enc_initial is not None:
                rows = mb["enc_init"]
                init = torch.where((rows >= 0)[:, None], enc_initial.index_select(0, rows.clamp_min(0)), 0.0)
            states = self.scan(keep, update_gate * torch.tanh(raw).reshape(Er, Le, H), init)
            before = torch.cat([(init if init is not None else states.new_zeros(Er, H))[:, None], states[:, :-1]], 1)
            states = states.reshape(-1, H)
            before = gather_rows(before.reshape(-1, H), mb["mem_pos"])  # (entries, hidden)
            x = gather_rows(x, mb["mem_pos"])
        else:  # no encoder tokens (generation: carried memories only)
            states = before = E_.new_zeros(0, H)
            x = E_.new_zeros(0, D)
        slot = mb["mem_slot"]
        keys = F.elu(rms_norm(before) @ torch.cat([p["mem_k"].value for p in self._params], 1)) + 1
        entries = keys.reshape(-1, L, h, d) if entry_keys else None
        values = (x @ torch.cat([p["mem_v"].value for p in self._params], 1)).reshape(-1, L, h, d)
        values = torch.cat([values, values.new_ones(values.shape[0], L, h, 1)], -1)
        keys = place_rows(keys, slot, Nck * ch).reshape(Nck, ch, L, h, d).permute(0, 2, 3, 4, 1)  # (Nck, L, h, d, ch)
        values = place_rows(values, slot, Nck * ch).reshape(Nck, ch, L, h, d + 1).permute(0, 2, 3, 1, 4)
        parts = (keys @ values).reshape(Nck, L * h * d * (d + 1))  # each chunk's K^T [V, 1]
        S = (seg[None] == torch.arange(G * nm, device=seg.device)[:, None]).to(parts.dtype)  # (G memories, chunks)
        C = (S @ parts).reshape(G, nm, L, h, d, d + 1).permute(2, 0, 1, 3, 4, 5).contiguous()
        return (C, states, entries) if entry_keys else (C, states)

    def _memory_batch(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        dev = self.token_embedding.embeddings.value.device
        return {k: torch.as_tensor(np.asarray(v) if not torch.is_tensor(v) else v).to(dev)
                for k, v in inputs.items() if not k.startswith("_")}

    # -- the stack --------------------------------------------------------------------------------------------------

    def _run(self, ids: Any, initial: Optional[List[torch.Tensor]] = None, valid: Optional[torch.Tensor] = None,
             collect: bool = False, memory: Optional[Dict[str, Any]] = None, ends_at: Optional[torch.Tensor] = None):
        """`(B, T)` ids through every layer from `initial` (one `(B, hidden)` state per layer, zeros if None).
        `valid` `(B, T)` bool: the stream at those positions only `(R, dim)` (row-major), else `(B, T, dim)`.
        `memory`, `ends_at`: see `_layer`. Returns (stream, end states, per-layer internals if `collect`)."""
        E = self.token_embedding.embeddings.value
        ids = torch.as_tensor(keras.ops.cast(ids, "int32")).to(E.device).long()
        B, T = ids.shape
        residual = E.index_select(0, ids.reshape(-1)).reshape(B, T, self.dim)  # index_select: see AGENTS (Iris Xe)
        not_reset = (ids != self.reset_token).to(E.dtype)
        rows = None if valid is None else torch.as_tensor(valid, device=E.device).reshape(-1).nonzero().squeeze(1)
        ends, infos = [], []
        for n in range(self.num_layers):
            last_rows = rows if n == self.num_layers - 1 else None
            init = None if initial is None else initial[n]
            if self.recompute and torch.is_grad_enabled() and not collect:
                residual, end, info = torch.utils.checkpoint.checkpoint(
                    self._layer, n, residual, not_reset, init, last_rows, False, memory, ends_at, use_reentrant=False)
            else:
                residual, end, info = self._layer(n, residual, not_reset, init, last_rows, collect, memory, ends_at)
            ends.append(end)
            infos.append(info)
        return residual, ends, infos

    def _run_memory(self, mb: Dict[str, torch.Tensor], valid: Optional[torch.Tensor] = None):
        """A batch dict (`model.memory_rows`, no carry): the memories, then the rows reading them."""
        C, _ = self._memories(mb)
        memory = {"C": C, **{k: mb[k] for k in ("read_pos", "read_slot", "chunk_doc")}} if mb["read_pos"].shape[0] else None
        return self._run(mb["ids"], valid=valid, memory=memory)[0]

    def _from_raw(self, ids: Any, valid: Any = None) -> Dict[str, torch.Tensor]:
        """Raw token rows (span contents included) as a batch dict on the device (each row starts fresh)."""
        from model.memory_rows import from_raw

        ids = np.asarray(keras.ops.convert_to_numpy(ids) if not torch.is_tensor(ids) else ids.cpu().numpy())
        valid = None if valid is None else (valid.cpu().numpy() if torch.is_tensor(valid) else np.asarray(valid))
        batch, kept_at, _ = from_raw(ids, self.memory_spans, self.memory_read_span, self.reset_token, valid=valid)
        if valid is not None and int(batch["valid"].sum()) != int(valid.sum()):
            raise ValueError("static_memory: targets inside system / tool spans (the layers never see those tokens)")
        return self._memory_batch(batch)

    def hidden(self, inputs: Any) -> torch.Tensor:
        """The stream after the last token of each `(B, T)` input: `(B, dim)` (before the final norm)."""
        if self.static_memory:  # after the last token the layers see (span contents are not)
            mb = self._from_raw(inputs)
            B, T = mb["ids"].shape
            last = torch.arange(T, device=mb["ids"].device)[None] == (mb["lengths"] - 1).clamp_min(0)[:, None]
            return self._run_memory(mb, valid=last)
        B, T = inputs.shape[0], inputs.shape[1]
        last = torch.zeros(B, T, dtype=torch.bool)
        last[:, -1] = True
        return self._run(inputs, valid=last)[0]

    @property
    def static_memory_spec(self) -> Optional[Dict[str, Any]]:
        """For the data loader (`PackedDocumentDataset(static_memory=...)`): rows without the span contents and each
        document's span tokens for the encoder. None without `static_memory`."""
        return {"spans": self.memory_spans, "read": self.memory_read_span} if self.static_memory else None

    def dense_hidden(self, inputs: Any) -> torch.Tensor:
        """The stream at every position `(B, T, dim)` (target of position t = token t + 1). With `static_memory`:
        at the positions of the rows without span contents (a loader dict, or raw rows compacted)."""
        if self.static_memory:
            mb = self._memory_batch(inputs) if isinstance(inputs, dict) else self._from_raw(inputs)
            return self._run_memory(mb)
        return self._run(inputs)[0]

    def dense_hidden_valid(self, inputs: Any, valid: Any, carry: bool = False) -> torch.Tensor:
        """The stream at the positions with a target only `(R, dim)`, row-major. `carry` (truncated BPTT,
        `training.stateful`): row b continues row b of the previous call from its detached end states."""
        if self.static_memory:
            if carry:
                raise ValueError("static_memory does not support training.stateful")
            mb = self._memory_batch(inputs) if isinstance(inputs, dict) else self._from_raw(inputs, valid)
            return self._run_memory(mb, valid if isinstance(inputs, dict) else mb["valid"])
        if not carry:
            return self._run(inputs, valid=valid)[0]
        prev = self.__dict__.get("_train_carry")
        if prev is not None and prev[-1].shape[0] != inputs.shape[0]:
            prev = None
        out, ends, _ = self._run(inputs, prev, valid=valid)
        self._train_carry = [e.detach() for e in ends]
        return out

    def readout_input(self, h: torch.Tensor) -> torch.Tensor:
        return rms_norm(h, self.final_norm.value)

    def call(self, inputs: Any, training: bool = False) -> Any:
        return self.readout(self.readout_input(self.hidden(inputs)))

    # -- generation -------------------------------------------------------------------------------------------------

    def advance(self, ids: Any, state: Optional[List[Any]] = None):
        """Feeds `(B, T)` ids after `state` (one `(B, hidden)` per layer): (logits of the last position, the states
        after it). With `static_memory` the state also holds (encoder state `(B, hidden)`, the open document's
        memories `(layers, B, ...)`, the last stream `(B, dim)`, the spans open (numpy)): span contents go to the
        encoder and the memory, the layers see the rest."""
        if not self.static_memory:
            B, T = ids.shape[0], ids.shape[1]
            last = torch.zeros(B, T, dtype=torch.bool)
            last[:, -1] = True
            out, ends, _ = self._run(ids, state, valid=last)
            return self.readout(self.readout_input(out)), ends
        from model.memory_rows import from_raw

        ids = np.asarray(keras.ops.convert_to_numpy(ids) if not torch.is_tensor(ids) else ids.cpu().numpy())
        B = ids.shape[0]
        layers, (enc0, C0, hidden0, open0) = (None, (None, None, None, None)) if state is None else (state[:-1], state[-1])
        batch, _, opens = from_raw(ids, self.memory_spans, self.memory_read_span, self.reset_token, open0=open0,
                                   carried=np.full(B, state is not None))
        mb = self._memory_batch(batch)
        dev = mb["ids"].device
        C, enc_states = self._memories(mb, enc0)
        if C0 is not None:  # the continuing document's memory: carried + this call's entries
            rows = torch.nonzero(mb["carry_doc"] >= 0).squeeze(1)
            C = C.index_add(1, mb["carry_doc"].index_select(0, rows), C0.index_select(1, rows))
        memory = {"C": C, **{k: mb[k] for k in ("read_pos", "read_slot", "chunk_doc")}} if mb["read_pos"].shape[0] else None
        lengths = mb["lengths"].long()
        T = mb["ids"].shape[1]
        last = torch.arange(T, device=dev)[None] == (lengths - 1)[:, None]  # rows without a token: none
        out, ends, _ = self._run(mb["ids"], layers, valid=last, memory=memory, ends_at=lengths - 1)
        hidden = out.new_zeros(B, self.dim) if hidden0 is None else hidden0.clone()
        hidden[lengths > 0] = out
        enc_end = mb["enc_end"]  # -1: the last document got no encoder token in this call (it continues)
        enc = enc0 if not enc_states.shape[0] else enc_states.index_select(0, enc_end.clamp_min(0))
        if enc0 is not None and enc_states.shape[0]:
            enc = torch.where((enc_end >= 0)[:, None], enc, enc0)
        return self.readout(self.readout_input(hidden)), ends + [(enc, C.index_select(1, mb["last_doc"]), hidden, opens)]

    def with_context_length(self, context_length: int) -> "MinGRULM":
        """Finite context: a copy that generates and evaluates from the last `context_length` tokens only."""
        new = _context.with_context_length(self, context_length)
        new.stateful_generation = False
        return new

    # -- inspection -------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def inspect(self, ids: Any) -> List[Dict[str, np.ndarray]]:
        """Every layer's internals at every position of `(B, T)` ids, as numpy: `update_gate`, `keep_factor`,
        `candidate`, `state` `(B, T, hidden)`; `layer_write` `(B, T, dim)` (what the layer added to the stream)."""
        if self.static_memory:
            raise NotImplementedError("inspect: with static_memory use inspect_memory (the layers skip span contents)")
        infos = self._run(ids, collect=True)[2]
        return [{k: v.float().cpu().numpy() for k, v in info.items()} for info in infos]

    @torch.no_grad()
    def inspect_memory(self, ids: Any) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
        """`static_memory`: `inspect` for one raw row `(1, T)` (span contents included). The layers' internals are at
        the positions they see (`kept`: the row's positions outside the span contents), plus per layer
        `memory_add` `(1, T', dim)`, `memory_gate` `(1, T', memories x heads)` and `memory_query` `(1, T', heads x d)`
        (zeros where nothing reads). Also: `entry_pos` (each memory entry's position in the row), `entry_kind`
        (0: system, 1: tools, ...), `entry_doc`, `entry_keys` `(entries, layers, heads, d)`, `read_pos` (the reading
        positions, among the kept ones) and `read_doc`: a reading position reads the entries of its document."""
        from model.memory_rows import READ_CHUNK as chunk, from_raw, span_masks

        ids = np.asarray(keras.ops.convert_to_numpy(ids) if not torch.is_tensor(ids) else ids.cpu().numpy())[:1]
        batch, kept_at, _ = from_raw(ids, self.memory_spans, self.memory_read_span, self.reset_token)
        mb = self._memory_batch(batch)
        C, _, keys = self._memories(mb, entry_keys=True)
        memory = {"C": C, **{k: mb[k] for k in ("read_pos", "read_slot", "chunk_doc")}} if mb["read_pos"].shape[0] else None
        infos = self._run(mb["ids"], collect=True, memory=memory)[2]
        kind, _, _, _, doc = span_masks(ids[0].astype(np.int64), self.memory_spans, self.memory_read_span,
                                        self.reset_token)
        nm = len(self.memory_spans)
        order = sorted(batch["_g_of"].items(), key=lambda kv: kv[1])  # document key (number in the row, B = 1) -> g
        entry_pos = [np.flatnonzero((doc == key) & (kind == i + 1)) for key, _ in order for i in range(nm)]
        seg = batch["mem_seg"][batch["mem_slot"] // chunk]
        meta = {"kept": kept_at[0], "entry_pos": np.concatenate(entry_pos + [np.zeros(0, np.int64)]),
                "entry_kind": seg % nm, "entry_doc": seg // nm, "entry_keys": keys.float().cpu().numpy(),
                "read_pos": batch["read_pos"], "read_doc": batch["chunk_doc"][batch["read_slot"] // chunk]}
        return [{k: v.float().cpu().numpy() for k, v in info.items()} for info in infos], meta

    def timescales(self) -> List[np.ndarray]:
        """Per layer, each state channel's half-life in tokens from its gate bias alone (the token-independent part
        of the gate): log(0.5) / log(keep_factor) at gate_logit = bias."""
        out = []
        for p in self._params:
            bias = p["in_bias"].value[: self.gates].detach().float().cpu().numpy()
            bias = np.repeat(bias, self.hidden_size // self.gates)  # per channel
            keep = 1.0 / (1.0 + np.exp(bias))  # sigmoid(-bias)
            out.append(np.log(0.5) / np.log(np.clip(keep, 1e-12, 1 - 1e-12)))
        return out

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "vocabulary_size": self._vocabulary_size, "dim": self.dim,
                "hidden_size": self.hidden_size, "num_layers": self.num_layers, "window_size": self.window_size,
                "reset_token": self.reset_token, "half_life_min": self.half_life_min,
                "half_life_max": self.half_life_max, "write_scale_init": self.write_scale_init,
                "embedding_std": self.embedding_std, "recompute": self.recompute, "silent_from": self.silent_from,
                "gate_heads": self.gate_heads, "static_memory": self.static_memory, "memory_heads": self.memory_heads,
                "memory_dim": self.memory_dim, "memory_spans": self.memory_spans,
                "memory_read_span": self.memory_read_span, "mlp_mult": self.mlp_mult}
