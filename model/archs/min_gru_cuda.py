"""`min-gru-cuda`: `min-gru` (same math, same weights: a run's weights load in either) with CUDA speedups. `min-gru`
stays the reference and the arch for local (Iris Xe) training.

On a CUDA GPU the chunked scan (`min_gru.linear_scan`: ~2 sqrt(T) small sequential steps, each a few kernels) leaves
the GPU idle between launches: 6.4 ms forward + backward at 32 x 512 x 256 on an RTX A5000, ~13 ms per layer in the
train step. Here the scan is one parallel associative scan: the pairs (keep, write) combine associatively,

    (a1, b1) then (a2, b2)  =  (a1 a2, a2 b1 + b2),

so torch's `associative_scan` compiles it (`torch.compile`, Triton) into one kernel scanning every channel's row in
log(T) parallel steps (0.14 ms forward vs 1.7 ms). Its backward is the same scan run from the end on
(keep[t + 1], d_state[t]): g[t] = d_state[t] + keep[t + 1] g[t + 1], then d_keep[t] = g[t] state[t - 1],
d_write = g, d_initial = keep[0] g[0] (`ParallelScan`; autograd through `associative_scan` itself was 39 ms).

Rows shorter than `PARALLEL_MIN_LENGTH` (generation, a token at a time) and non-CUDA devices use the chunked scan.
The kernels compile for fixed shapes (`BATCH_BUCKET`, `LENGTH_BUCKET`: padding with keep 1, write 0 changes nothing).
"""

import os
from typing import Any, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model.archs.min_gru import MinGRULM, linear_scan  # noqa: E402
from model.memory_rows import READ_CHUNK  # noqa: E402

PARALLEL_MIN_LENGTH = 64
FUSED = os.environ.get("MINGRU_CUDA_FUSED", "1") != "0"  # 0: only the parallel scan (to compare)
# Each new shape compiles a new kernel (a few seconds; dynamic shapes do not compile with `associative_scan`): rows
# and lengths are padded up to multiples of these, so a run sees a few shapes (encoder rows and compacted lengths vary).
BATCH_BUCKET, LENGTH_BUCKET = 16, 128
# torch allows 8 compiled shapes per function by default: on cosmo-agent-mix the memory encoder's rows (documents with
# a system / tool span per batch) took a 9th bucket at the first validation and `fullgraph=True` then raises
import torch._dynamo  # noqa: E402

torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
torch._dynamo.config.accumulated_recompile_limit = max(torch._dynamo.config.accumulated_recompile_limit, 1024)


def _bucket(n: int, step: int) -> int:
    return -(-n // step) * step


def _combine(x: Any, y: Any) -> Any:
    """(keep, write) pairs: x then y."""
    return x[0] * y[0], y[0] * x[1] + y[1]


@torch.compile(fullgraph=True, dynamic=False)
def _scan_forward(keep: torch.Tensor, write: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    from torch._higher_order_ops.associative_scan import associative_scan

    write = torch.cat([write[:, :1] + keep[:, :1] * initial[:, None], write[:, 1:]], 1)  # the state before the row
    return associative_scan(_combine, (keep, write), dim=1, combine_mode="pointwise")[1]


@torch.compile(fullgraph=True, dynamic=False)
def _scan_backward(keep: torch.Tensor, states: torch.Tensor, initial: torch.Tensor, d_states: torch.Tensor):
    from torch._higher_order_ops.associative_scan import associative_scan

    keep_next = torch.cat([keep[:, 1:], torch.zeros_like(keep[:, :1])], 1)
    g = associative_scan(_combine, (keep_next, d_states), dim=1, reverse=True, combine_mode="pointwise")[1]
    d_keep = g * torch.cat([initial[:, None], states[:, :-1]], 1)
    return g, d_keep, keep[:, 0] * g[:, 0]


class ParallelScan(torch.autograd.Function):
    """state[t] = keep[t] state[t-1] + write[t] as one associative scan each way (see the module doc). keep `(B, T, G)`:
    per channel (G = D) or per head of D / G channels (expanded here; its gradient summed back)."""

    @staticmethod
    def forward(ctx: Any, keep: torch.Tensor, write: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        B, T, D = write.shape
        G = keep.shape[-1]
        full = keep if G == D else keep.repeat_interleave(D // G, -1)
        Bp, Tp = _bucket(B, BATCH_BUCKET), _bucket(T, LENGTH_BUCKET)
        if (Bp, Tp) != (B, T):  # padding: rows of zeros, positions after the row (keep 1, write 0: nothing changes)
            full = torch.nn.functional.pad(full, (0, 0, 0, Tp - T, 0, Bp - B), value=1.0)
            write = torch.nn.functional.pad(write, (0, 0, 0, Tp - T, 0, Bp - B))
            initial = torch.nn.functional.pad(initial, (0, 0, 0, Bp - B))
        states = _scan_forward(full.contiguous(), write.contiguous(), initial.contiguous())
        ctx.save_for_backward(full, states, initial)
        ctx.shape, ctx.heads = (B, T), G
        return states[:B, :T]

    @staticmethod
    def backward(ctx: Any, d_states: torch.Tensor):  # type: ignore[override]
        full, states, initial = ctx.saved_tensors
        B, T = ctx.shape
        Bp, Tp = full.shape[:2]
        d_states = torch.nn.functional.pad(d_states, (0, 0, 0, Tp - T, 0, Bp - B)) if (Bp, Tp) != (B, T) else d_states
        g, d_keep, d_initial = _scan_backward(full, states, initial, d_states.contiguous())
        g, d_keep, d_initial = g[:B, :T], d_keep[:B, :T], d_initial[:B]
        if ctx.heads != full.shape[-1]:
            d_keep = d_keep.reshape(B, T, ctx.heads, -1).sum(-1)
        return d_keep, g, d_initial


# -- fused layer parts (torch.compile: Inductor generates one kernel per pointwise chain, forward and backward) ---------

def _rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


@torch.compile(fullgraph=True, dynamic=False)
def _layer_in(residual: torch.Tensor, norm_in: torch.Tensor, in_proj: torch.Tensor, in_bias: torch.Tensor,
              not_reset: torch.Tensor, gates: int):
    """The scan's inputs: keep = (1 - update_gate) (0 at a document marker), write = update_gate * tanh(raw)."""
    H = in_proj.shape[1] - gates
    gate_logit, raw = (_rms(residual) @ (norm_in[:, None] * in_proj) + in_bias).split([gates, H], -1)
    update_gate = torch.sigmoid(gate_logit)
    candidate = torch.tanh(raw)
    write = (update_gate[..., None] * candidate.reshape(*raw.shape[:-1], gates, H // gates)).reshape(raw.shape)
    return (1 - update_gate) * not_reset[..., None], write


@torch.compile(fullgraph=True, dynamic=False)
def _state_norm(states: torch.Tensor) -> torch.Tensor:
    return _rms(states)


@torch.compile(fullgraph=True)
def _layer_out(state_n: torch.Tensor, residual: torch.Tensor, norm_state: torch.Tensor, out_proj: torch.Tensor,
               write_scale: torch.Tensor) -> torch.Tensor:
    return residual + write_scale * torch.tanh(state_n @ (norm_state[:, None] * out_proj))


@torch.compile(fullgraph=True)
def _layer_out_mlp(state_n: torch.Tensor, residual: torch.Tensor, norm_state: torch.Tensor, mlp_in: torch.Tensor,
                   mlp_bias: torch.Tensor, mlp_out: torch.Tensor, write_scale: torch.Tensor) -> torch.Tensor:
    hid = F.gelu(state_n @ (norm_state[:, None] * mlp_in) + mlp_bias, approximate="tanh")
    return residual + write_scale * (hid @ mlp_out)


@torch.compile(fullgraph=True, dynamic=True)
def _memory_add(residual: torch.Tensor, state_n: torch.Tensor, C: torch.Tensor, read_pos: torch.Tensor,
                read_slot: torch.Tensor, chunk_doc: torch.Tensor, mem_q: torch.Tensor, mem_gate: torch.Tensor,
                mem_gate_bias: torch.Tensor, mem_out: torch.Tensor, chunk: int):
    """`MinGRULM._memory_reads` plus the add into the stream `(B T, dim)`, with plain index ops (CUDA's atomic adds
    are cheap, unlike the Iris Xe's)."""
    nm, h, d = C.shape[1], C.shape[2], C.shape[3]
    Nc = chunk_doc.shape[0]
    sn = state_n.index_select(0, read_pos)
    q_raw, g_raw = (sn @ torch.cat([mem_q, mem_gate], 1)).split([h * d, nm * h], -1)
    q = torch.zeros(Nc * chunk, h * d, device=sn.device, dtype=sn.dtype).index_copy(0, read_slot, F.elu(q_raw) + 1)
    q = q.reshape(Nc, 1, chunk, h, d).transpose(2, 3)  # (Nc, 1, h, chunk, d)
    out = q @ C.index_select(0, chunk_doc)  # (Nc, nm, h, chunk, d + 1)
    out = out.permute(0, 3, 1, 2, 4).reshape(Nc * chunk, nm, h, d + 1).index_select(0, read_slot)
    gate = torch.sigmoid(g_raw + mem_gate_bias)
    # z = 0: an empty memory (no such span in the document; its read is 0). Dividing by 1 there, not by 1e-6,
    # keeps its (unused) gradient bounded: 1e7-sized rows wrecked the precision of RepeatSorted's cumsum
    z = out[..., d]
    z = torch.where(z > 0, z.clamp_min(1e-6), torch.ones_like(z))
    rd = out[..., :d] * (gate.reshape(-1, nm, h) / z)[..., None]
    return residual.index_add(0, read_pos, rd.reshape(-1, nm * h * d) @ mem_out)


@keras.saving.register_keras_serializable(package="model")
class MinGRUCudaLM(MinGRULM):
    """`min-gru` for CUDA GPUs: the same model and arguments (see `MinGRULM`), the scan as one parallel associative
    scan (`ParallelScan`) and the rest of each layer as fused kernels (`_layer_in`, `_state_norm`, `_memory_add`,
    `_layer_out` / `_layer_out_mlp`) on rows of `PARALLEL_MIN_LENGTH` tokens or more; inspection and generation take `MinGRULM`'s path."""

    def _layer(self, n: int, residual: torch.Tensor, not_reset: torch.Tensor, initial: Optional[torch.Tensor],
               rows: Optional[torch.Tensor] = None, collect: bool = False, memory: Optional[Any] = None,
               ends_at: Optional[torch.Tensor] = None):
        if (collect or ends_at is not None or not FUSED or residual.device.type != "cuda"
                or residual.shape[1] < PARALLEL_MIN_LENGTH):
            return super()._layer(n, residual, not_reset, initial, rows, collect, memory, ends_at)
        p = {k: v.value for k, v in self._params[n].items()}
        B, T, D = residual.shape
        H = self.hidden_size
        keep, write = _layer_in(residual, p["norm_in"], p["in_proj"], p["in_bias"], not_reset, self.gates)
        states = self.scan(keep, write, initial)
        state_n = _state_norm(states).reshape(-1, H)
        residual = residual.reshape(-1, D)
        if memory is not None:
            residual = _memory_add(residual, state_n, memory["C"][n], memory["read_pos"], memory["read_slot"],
                                   memory["chunk_doc"], p["mem_q"], p["mem_gate"], p["mem_gate_bias"], p["mem_out"],
                                   READ_CHUNK)
        if rows is not None:
            state_n, residual = state_n.index_select(0, rows), residual.index_select(0, rows)
        if self.mlp_mult is None:
            out = _layer_out(state_n, residual, p["norm_state"], p["out_proj"], p["write_scale"])
        else:
            out = _layer_out_mlp(state_n, residual, p["norm_state"], p["mlp_in"], p["mlp_bias"], p["mlp_out"],
                                 p["write_scale"])
        return (out if rows is not None else out.reshape(B, T, D)), states[:, -1], None

    def scan(self, keep: torch.Tensor, write: torch.Tensor, initial: Optional[torch.Tensor]) -> torch.Tensor:
        if keep.device.type != "cuda" or keep.shape[1] < PARALLEL_MIN_LENGTH:
            return linear_scan(keep, write, initial)
        if initial is None:
            initial = write.new_zeros(write.shape[0], write.shape[2])
        return ParallelScan.apply(keep, write, initial)
