"""Dense supervision for the RoPE archs: every window position in one pass.

Single-target training reads one window of `T` tokens and predicts the next
one. Dense training reads `S = 2T - 1` tokens and predicts the next token at
each of the last `T` positions, every one of them seeing a full `T`-token
window -- the same `T` predictions as `T` single-target windows, in one pass.

The identity that makes it cheap is RoPE's relative rotation: a token at
absolute position `j`, seen from a query at position `i` (distance
`k = i - j + 1`), is rotated by `k θ = (i + 1) θ - j θ`, so

    R_k e_j = R_{i+1} (R_{-j} e_j) = R_{i+1} u_j

Rotating every token once by `-j` gives `u`; any kernel sum
`sum_k w_k R_k e_{i+1-k}` becomes `R_{i+1} (W @ u)` with a banded matrix
`W[m, m + r] = w_m[r]` (`r` = context-axis index, distance `T - r`), and
attention scores `q . R_k e_j` become `(R_{-(i+1)} q) . u_j`.

`skew_band` builds those banded matrices from the per-query, context-axis
aligned weight vectors the single-target code already uses, so the dense
path reuses every weight with no new parameters.

Query `m` (0..T-1) sits at absolute position `i = T - 1 + m` of the input.
"""

from typing import Any, Tuple

import numpy as np
from keras import ops


def skew_band(v: Any) -> Any:
    """`(..., T, T)` -> `(..., T, 2T-1)` with `out[..., m, m + r] = v[..., m, r]`, 0 elsewhere.

    Shift-by-row via reshape: rows of length `2T` (values + zeros) laid out
    flat and re-read as rows of length `2T - 1` shift row `m` right by `m`.
    """
    T = v.shape[-1]
    lead = tuple(v.shape[:-2])
    padded = ops.concatenate([v, ops.zeros_like(v)], axis=-1)  # (..., T, 2T)
    flat = ops.reshape(padded, lead + (T * 2 * T,))
    flat = flat[..., : T * (2 * T - 1)]
    return ops.reshape(flat, lead + (T, 2 * T - 1))


def band_mask(T: int) -> np.ndarray:
    """`(T, 2T-1)` bool: query `m` may see input positions `m .. m + T - 1`."""
    m = np.arange(T)[:, None]
    j = np.arange(2 * T - 1)[None, :]
    return (j >= m) & (j < m + T)


def rope_angles(dim: int, base: float, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """`(cos, sin)` of shape `(len(positions), dim/2)` for absolute positions."""
    half = dim // 2
    inv_freq = 1.0 / (base ** (np.arange(half, dtype=np.float64) / half))
    angles = positions.astype(np.float64)[:, None] * inv_freq[None, :]
    return np.cos(angles).astype("float32"), np.sin(angles).astype("float32")


def rotate(x: Any, cos: Any, sin: Any) -> Any:
    """Half-split RoPE rotation (pairs `i`, `i + dim/2`) of `x` `(..., L, dim)` by per-row angles."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return ops.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class DenseRope:
    """Per-model cache of the dense-path constants (device tensors, built once)."""

    def __init__(self, model: Any, like: Any):
        T, dim, base = model.window_size, model.dim, model.rope_base
        S = 2 * T - 1
        key_cos, key_sin = rope_angles(dim, base, np.arange(S))
        query_cos, query_sin = rope_angles(dim, base, np.arange(T, 2 * T))  # i + 1 = T + m
        as_t = lambda a: ops.convert_to_tensor(a, dtype=like.dtype)
        self.key_cos, self.key_sin_neg = as_t(key_cos), as_t(-key_sin)  # rotate by -j
        self.query_cos, self.query_sin = as_t(query_cos), as_t(query_sin)  # rotate by +(i+1)
        self.query_sin_neg = as_t(-query_sin)  # rotate by -(i+1)
        self.mask = ops.convert_to_tensor(band_mask(T))

    @staticmethod
    def get(model: Any, like: Any) -> "DenseRope":
        cache = model.__dict__.setdefault("_dense_cache", {})
        key = (str(getattr(like, "device", "")), str(like.dtype))
        if key not in cache:
            cache[key] = DenseRope(model, like)
        return cache[key]

    def unrotate_keys(self, e: Any) -> Any:
        """`u_j = R_{-j} e_j` for the `S` input positions."""
        return rotate(e, self.key_cos, self.key_sin_neg)

    def to_query_frame(self, s: Any) -> Any:
        """`R_{i+1} s_m` for the `T` query positions."""
        return rotate(s, self.query_cos, self.query_sin)

    def from_query_frame(self, q: Any) -> Any:
        """`R_{-(i+1)} q_m` for the `T` query positions."""
        return rotate(q, self.query_cos, self.query_sin_neg)


class DenseSupervisionMixin:
    """Adds a dense train step to a RoPE arch that implements `dense_hidden`.

    Off unless `dense_supervision_enabled` is set (by `model.train` when
    `training.dense_supervision` is true); evaluation and validation always
    use the normal single-target `test_step`, so losses stay comparable
    across runs.

    Data: `x` `(B, 2T-1)` int ids, `y` `(B, T)` int targets with `-1` for
    positions past the document end (see `model.data.DenseWindowDataset`).
    """

    dense_supervision_enabled = False
    stateful_training = False  # truncated BPTT: rows continue the previous batch's rows (see `dense_hidden_valid`)

    def dense_hidden(self, inputs: Any) -> Any:  # pragma: no cover - implemented by archs
        raise NotImplementedError

    def readout_input(self, h: Any) -> Any:
        """Hook for archs with a projection between the context vector and the readout."""
        return h

    def train_step(self, data: Any) -> Any:
        if not self.dense_supervision_enabled:
            return super().train_step(data)

        import torch
        from keras.src.trainers.data_adapters import data_adapter_utils

        from model.fused_linear_cross_entropy import fused_linear_cross_entropy

        x, y, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        # gradient accumulation: `grad_accumulation` micro-batches per optimizer update (set by model.train from
        # data.batch_size / data.micro_batch_size); each adds the gradient of its summed loss, and the update divides
        # by the targets of all of them: exactly the gradient of the mean loss over the whole batch
        accum = int(getattr(self, "grad_accumulation", 1))
        k = self.__dict__.get("_accum_k", 0)
        if k == 0:
            self.zero_grad()
            self._accum_n = 0
        dev = self.token_embedding.embeddings.value.device if hasattr(self, "token_embedding") else None
        y = torch.as_tensor(y, device=dev).to(torch.int64)
        valid = y >= 0
        n_targets = int(valid.sum())
        if hasattr(self, "dense_hidden_valid"):
            # only the positions with a target reach the readout side of the last layer (its output mix and the
            # readout feed nothing else): (n_targets, dim), row-major over (B, T)
            hv = (self.dense_hidden_valid(x, valid, carry=True) if self.stateful_training
                  else self.dense_hidden_valid(x, valid))
            hidden = None
        else:
            hidden = self.dense_hidden(x)  # (B, T, dim)
            hv = hidden[valid]
        if accum == 1:
            self.zero_grad()

        # Pointwise readout-side layers run on the valid rows only. A batch without targets (rows inside a long tool
        # output) adds nothing to the gradient, but still counts as one of the update's micro-batches.
        if n_targets:
            loss = fused_linear_cross_entropy(self.readout_input(hv), self.readout.kernel, y[valid])
            self._loss_tracker.update_state(loss, sample_weight=n_targets)
            objective = loss * n_targets if accum > 1 else loss  # summed over the targets while accumulating
            if self.optimizer is not None:
                objective = self.optimizer.scale_loss(objective)
        else:
            loss = objective = None
        if self.trainable_weights:
            if objective is not None:
                objective.backward()
            self._accum_n += n_targets
            k += 1
            if k == accum:
                trainable_weights = self.trainable_weights[:]
                gradients = [v.value.grad for v in trainable_weights]
                with torch.no_grad():
                    if accum > 1:
                        for g in gradients:
                            if g is not None:
                                g.div_(max(self._accum_n, 1))
                    self.optimizer.apply(gradients, trainable_weights)
                k = 0
            self._accum_k = k

        if loss is None:
            return {**self.get_metrics_result(), "loss": torch.zeros((), device=y.device)}
        # Accuracy on each sample's last valid position only: one (B, V) readout,
        # comparable to the single-target accuracy the other runs log.
        rows = torch.arange(y.shape[0], device=y.device)
        # last valid index (packed rows have ignored targets in the middle, not only at the end)
        cols = torch.arange(y.shape[1], device=y.device)
        last = torch.where(valid, cols, -1).max(dim=1).values.clamp_min(0)
        with torch.no_grad():
            if hidden is None:  # hv holds each row's valid positions in order: its last one closes the row's block
                has = valid.any(1)
                at = (torch.cumsum(valid.sum(1), 0) - 1)[has]
                logits_last = self.readout(self.readout_input(hv.detach()[at]))
                y_last = y[rows, last][has]
            else:
                logits_last = self.readout(self.readout_input(hidden.detach()[rows, last]))
                y_last = y[rows, last]
        logs = self.compute_metrics(x, y_last, logits_last, sample_weight=None)
        # Show this batch's loss, not Keras's running mean over the epoch. The keys are
        # unchanged, so Keras still builds the *epoch* logs (metrics.csv, checkpoints,
        # History) from the metric trackers, i.e. the epoch average.
        return {**logs, "loss": loss.detach()}

    dense_validation_enabled = False

    def test_step(self, data: Any) -> Any:
        """With `dense_validation_enabled` and packed rows `(B, T)` with targets `(B, T)` (-1 ignored): the loss and
        accuracy over every target of the rows, from one pass each. Single-target batches take the normal step."""
        import torch
        from keras.src.trainers.data_adapters import data_adapter_utils

        from model.fused_linear_cross_entropy import fused_linear_cross_entropy

        x, y, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        if not (self.dense_validation_enabled and len(getattr(y, "shape", ())) == 2):
            return super().test_step(data)
        with torch.no_grad():
            dev = self.token_embedding.embeddings.value.device
            y = torch.as_tensor(y, device=dev).to(torch.int64)
            valid = y >= 0
            n = int(valid.sum())
            hv = self.dense_hidden_valid(x, valid) if hasattr(self, "dense_hidden_valid") else self.dense_hidden(x)[valid]
            h = self.readout_input(hv)
            loss = fused_linear_cross_entropy(h, self.readout.kernel, y[valid])
            self._loss_tracker.update_state(loss, sample_weight=n)
            # accuracy over every target (the logits of the valid rows only: (n, vocab))
            logs = self.compute_metrics(x, y[valid], h @ self.readout.kernel, sample_weight=None)
        return {**logs, "loss": loss.detach()}


def dense_kernel_sum(weights: Any, u: Any) -> Any:
    """`sum_r W[b, m, m + r] u[b, m + r]` for context-aligned per-query weights.

    `weights`: `(T,)` (static, shared by every query), `(T, T)` or `(B, T, T)`
    with `weights[..., m, r]` the weight of context index `r` (distance `T - r`)
    for query `m`; `u`: `(B, 2T-1, dim)`. Returns `(B, T, dim)` in the
    unrotated frame (apply `to_query_frame` after).
    """
    T = u.shape[1] // 2 + 1
    if len(weights.shape) == 1:
        band = skew_band(ops.broadcast_to(weights[None, :], (T, T)))  # (T, S)
        return ops.einsum("ms,bsd->bmd", band, u)
    return ops.matmul(skew_band(weights), u)  # (B, T, S) @ (B, S, dim)


__all__ = [
    "DenseRope",
    "DenseSupervisionMixin",
    "band_mask",
    "dense_kernel_sum",
    "rope_angles",
    "rotate",
    "skew_band",
]
