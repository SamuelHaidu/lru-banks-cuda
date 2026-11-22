"""Fused linear readout + cross-entropy in constant logits memory.

    loss = mean_i ( logsumexp(h_i @ K) - (h_i @ K)[y_i] )

for hidden states `h` `(N, D)`, a bias-free readout kernel `K` `(D, V)` (the
layout of `keras.layers.Dense.kernel`) and targets `y` `(N,)`.

Dense supervision (one target per window position) makes `N = batch * window`,
so the `(N, V)` logits are the dominant cost: at N = 512 * 128 and V = 10k
that is 2.6 GB for the logits alone, plus as much again for their gradient.

`torch.autograd.Function` here processes rows in chunks of `chunk_size`. For
each chunk it computes the logits, the loss, and -- since the loss is a mean,
its gradient w.r.t. the logits is known in the forward pass -- both
gradients directly:

    p     = softmax(logits)                          (chunk, V), one fused kernel, discarded after use
    dh_c  = (p @ K^T - K[:, y]^T) / N                (chunk, D)
    dK^T += (p^T @ h_c) / N, dK^T[y] -= h_c / N     (V, D), addmm + index_add

The one-hot term of `softmax - onehot` is applied on the small side (a
column gather from `K`, an `index_add` into `dK`) instead of on the
`(chunk, V)` block, so each chunk makes a single elementwise pass over it.

Only `dh` `(N, D)` and `dK` `(D, V)` are kept for backward, which just scales
them by the incoming scalar gradient. Peak extra memory is one `(chunk, V)`
block, and each chunk costs 3 matmuls (no recomputation pass), the same
scheme as Liger's fused linear cross-entropy.

When no input requires grad (evaluation), the gradient work is skipped.
"""

from typing import Any, Optional, Tuple

import torch


class _FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, h: torch.Tensor, kernel: torch.Tensor, y: torch.Tensor, chunk_size: int,  # type: ignore[override]
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = h.shape[0]
        need_dh, need_dk = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        acc = torch.promote_types(h.dtype, torch.float32)  # softmax/accumulation precision: >= fp32
        dh = torch.empty_like(h) if need_dh else None
        # dK accumulated transposed, (V, D): `p^T h` into it is ~20% faster than `h^T p` into (D, V) on the Iris Xe
        dk = torch.zeros(kernel.shape[::-1], device=kernel.device, dtype=acc) if need_dk else None
        total = torch.zeros((), device=h.device, dtype=acc)
        y = y.to(torch.int64)
        inv_n = 1.0 / n
        tiny = torch.finfo(acc).tiny
        for start in range(0, n, chunk_size):
            stop = start + chunk_size
            h_c = h[start:stop]
            y_c = y[start:stop]
            # One fused softmax kernel over the (c, V) block; the one-hot part of
            # the gradient never touches it (applied on the small side below).
            probs = torch.softmax((h_c @ kernel).to(acc), dim=1)
            # -log p_y; p_y below the float's smallest normal (loss > ~87 nats)
            # is clamped, which only caps the reported loss, never the gradient.
            logp = probs.gather(1, y_c[:, None]).clamp_min_(tiny).log_().squeeze(1)
            if weights is None:
                total -= logp.sum()
                h_y = h_c.to(acc)  # what the one-hot part adds to dK
            else:  # every row's loss and both gradients scaled by its weight
                w_c = weights[start:stop].to(acc)
                total -= (logp * w_c).sum()
                probs.mul_(w_c[:, None])
                h_y = h_c.to(acc) * w_c[:, None]
            if need_dh:
                # (softmax - onehot) @ K^T / n = (probs @ K^T - K[:, y]^T) / n (rows times their weights)
                dh_c = probs.to(h.dtype) @ kernel.T
                k_y = kernel.index_select(1, y_c).T
                dh_c -= k_y if weights is None else k_y * w_c[:, None].to(k_y.dtype)
                dh[start:stop] = dh_c.mul_(inv_n)
            if need_dk:
                # h^T (softmax - onehot) / n: dense part via addmm, one-hot part via index_add
                dk.addmm_(probs.T, h_c.to(acc), alpha=inv_n)
                dk.index_add_(0, y_c, h_y, alpha=-inv_n)
            del probs
        ctx.save_for_backward(*(t for t in (dh, dk) if t is not None))
        ctx.has = (need_dh, need_dk)
        ctx.kernel_dtype = kernel.dtype
        return (total / n).to(h.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore[override]
        saved = list(ctx.saved_tensors)
        need_dh, need_dk = ctx.has
        dh = saved.pop(0) * grad_output if need_dh else None
        dk = (saved.pop(0).T * grad_output).to(ctx.kernel_dtype).contiguous() if need_dk else None
        return dh, dk, None, None, None


def fused_linear_cross_entropy(h: Any, kernel: Any, y: Any, chunk_size: int = 2048,
                               weights: Any = None) -> torch.Tensor:
    """Mean cross-entropy of `h @ kernel` against `y` without materializing all logits.

    Args:
        h: `(N, D)` hidden states (any leading shape is flattened to rows).
        kernel: `(D, V)` readout kernel, e.g. `model.readout.kernel`.
        y: `(N,)` integer targets (same leading shape as `h`).
        chunk_size: Rows per chunk (2048 was 7% faster than 1024 on the Iris Xe at 65k rows; 2048 to 8192 the same);
            peak extra memory is about `2 * chunk_size * V` floats.
        weights: Optional `(N,)` per-row weights (constants, no gradient): returns `sum_i w_i nll_i / N`, e.g. a
            policy-gradient loss with w = the advantage (`model/grpo.py`). None: every weight 1.
    """
    h = torch.as_tensor(h)
    kernel = kernel.value if hasattr(kernel, "value") else torch.as_tensor(kernel)
    y = torch.as_tensor(y, device=h.device)
    h = h.reshape(-1, h.shape[-1])
    y = y.reshape(-1)
    if h.shape[0] != y.shape[0]:
        raise ValueError(f"h has {h.shape[0]} rows but y has {y.shape[0]} targets")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if weights is not None:
        weights = torch.as_tensor(weights, device=h.device).reshape(-1).detach()
        if weights.shape[0] != y.shape[0]:
            raise ValueError(f"{weights.shape[0]} weights for {y.shape[0]} targets")
    return _FusedLinearCrossEntropy.apply(h, kernel, y, int(chunk_size), weights)
