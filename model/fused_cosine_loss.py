"""Memory-efficient cosine-readout cross-entropy: a custom `torch.autograd.Function`
that never keeps the full `(batch, vocabulary_size)` logits tensor alive
across the forward/backward boundary.

Motivation -- dense supervision (one loss per window POSITION, not one loss
per example the way `VariablePrefixDataset` currently trains) would need
`(window_size, vocabulary_size)` logits per document instead of
`(vocabulary_size,)`. For `dual-sphere`-family architectures at
`vocabulary_size=10000`, `window_size=128`, that's 1.28M logits per document
just for one batch item, and the framework's default autograd behavior keeps
that whole tensor alive from forward through backward so it can compute
`d(loss)/d(logits)`. This module removes that specific memory cost (NOT the
128x extra forward-pass compute dense supervision would also require --
that's an unavoidable consequence of this architecture family collapsing the
whole window into one query before it ever sees a target, and no gradient
trick removes it; see `studies/` for the fuller discussion).

How: `torch.autograd.Function.forward` computes the scalar-per-example loss
but explicitly does NOT save the `(N, V)` logits/log-softmax tensor for
backward -- only `q`, `target_embeddings`, `tau`, and `y` (all much smaller)
are saved. `backward` recomputes `logits` and `softmax(logits)` from those
saved tensors on demand, exactly when the gradient is needed, then never
retains them afterward. This trades a second forward matmul for the
memory savings -- classic recompute-instead-of-store, the same idea as
gradient checkpointing, applied specifically to the softmax-cross-entropy
step (the same pattern production kernels like Liger's fused linear
cross-entropy use for large-vocabulary language models).

Correctness: forward value and both gradients (`d loss/d q`,
`d loss/d target_embeddings`) are checked against the existing
`log_softmax` + `take_along_axis` formulation in `model/losses.py` in
`tests/test_fused_cosine_loss.py` -- this is a hand-derived backward pass,
not something autograd verifies for you, so it needs its own tests.
"""

from typing import Any, Tuple

import torch


class _FusedCosineCrossEntropy(torch.autograd.Function):
    """`loss[i] = -log_softmax(tau * q[i] @ target_embeddings.T)[y[i]]`,
    computed without retaining the `(N, vocab)` intermediate for backward.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        target_embeddings: torch.Tensor,
        tau: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            logits = tau * (q @ target_embeddings.T)  # (N, V) -- transient, not saved
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -log_probs.gather(1, y.view(-1, 1).to(torch.int64)).squeeze(1)

        # Only small tensors survive to backward: q (N, D), target_embeddings
        # (V, D), tau (scalar), y (N,) -- never the (N, V) logits/log_probs.
        ctx.save_for_backward(q, target_embeddings, tau, y)
        return loss

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:  # type: ignore[override]
        q, target_embeddings, tau, y = ctx.saved_tensors

        with torch.no_grad():
            # Recomputed here, not reused from forward -- this recomputation
            # (trading compute for memory) is the entire point of this class.
            logits = tau * (q @ target_embeddings.T)  # (N, V)
            probs = torch.softmax(logits, dim=-1)
            probs.scatter_add_(
                1, y.view(-1, 1).to(torch.int64), -torch.ones_like(probs[:, :1])
            )  # probs is now (softmax(logits) - one_hot(y))

            grad_logits = probs * grad_output.view(-1, 1)  # (N, V), d(loss)/d(logits)

            grad_q = tau * (grad_logits @ target_embeddings)  # (N, D)
            grad_target_embeddings = tau * (grad_logits.T @ q)  # (V, D)
            grad_tau = (grad_logits * (q @ target_embeddings.T)).sum()

        return grad_q, grad_target_embeddings, grad_tau, None


def fused_cosine_cross_entropy(
    q: torch.Tensor, target_embeddings: torch.Tensor, tau: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """`(batch,)` per-example loss from a pre-readout query, avoiding ever
    keeping the full `(batch, vocabulary_size)` logits alive for backward.

    Args:
        q: `(batch, dim_out)` L2-normalized query (pre-readout, i.e. what an
            architecture's `call_query()` returns -- NOT full logits).
        target_embeddings: `(vocabulary_size, dim_out)` L2-normalized target
            table.
        tau: Scalar cosine-readout temperature. Every architecture in this
            project stores `tau` as a `keras.Variable`
            (`self.add_weight(...)`), not a raw torch tensor -- multiplying
            a torch tensor by a bare `keras.Variable` inside
            `torch.autograd.Function.forward` was observed to route through
            Keras's backend dispatch and land on a stray TensorFlow tensor
            (`AttributeError: EagerTensor object has no attribute 'T'`) even
            though `KERAS_BACKEND=torch` throughout. Converting explicitly
            with `torch.as_tensor` here, once, at the public entry point,
            avoids every caller needing to remember this.
        y: `(batch,)` integer target token ids.
    """
    from keras import ops as _keras_ops

    tau_t = tau if isinstance(tau, torch.Tensor) else _keras_ops.convert_to_tensor(tau)
    return _FusedCosineCrossEntropy.apply(q, target_embeddings, tau_t, y)
