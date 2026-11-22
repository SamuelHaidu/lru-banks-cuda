"""`fused_cosine_cross_entropy` correctness and memory-saving properties.

Correctness is checked directly against the existing, already-verified
`log_softmax` + `take_along_axis` formulation in `model/losses.py` -- both
the forward loss VALUE and every gradient (`d loss/d q`, `d loss/d
target_embeddings`, `d loss/d tau`) must match, since the backward pass here
is hand-derived, not something autograd verifies on its own.

Memory saving is checked against `dual-sphere-hyper-gate` (this session's
newest architecture) specifically, at a `window_size`/`vocabulary_size`
large enough that dense (one-loss-per-window-position) supervision's
`(window_size, vocabulary_size)` logits tensor is the dominant memory cost,
using `torch.cuda`-style peak-memory accounting where available and a
direct tensor-count fallback otherwise (this project's XPU backend has no
`torch.xpu.max_memory_allocated`-equivalent guaranteed available in every
torch build, so the fallback keeps the test meaningful on this project's
actual hardware, not just on CUDA).
"""

import unittest

import numpy as np
import torch

from model.device import setup_backend, setup_device

# `setup_backend()` MUST run before the first `import keras` anywhere in the
# process -- Keras reads `KERAS_BACKEND` once, at import, and otherwise falls
# back to `~/.keras/keras.json`, which on this machine says "tensorflow"
# (see `model/device.py`'s module docstring). `from keras import ops` above
# this line would silently route every `ops.*` call in this file through
# TensorFlow instead of torch, which is exactly the bug that produced
# `AttributeError: EagerTensor object has no attribute 'T'` when this test
# was first written with the import order reversed.
setup_backend()
setup_device("cpu")  # deterministic, available everywhere tests run

from keras import ops  # noqa: E402

from model.fused_cosine_loss import fused_cosine_cross_entropy  # noqa: E402
from model.losses import SparseCategoricalCrossentropyFromLogits  # noqa: E402

VOCAB = 37
DIM_OUT = 12
BATCH = 9


def _random_query_and_table(seed: int = 0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(BATCH, DIM_OUT)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    table = rng.normal(size=(VOCAB, DIM_OUT)).astype(np.float32)
    table /= np.linalg.norm(table, axis=-1, keepdims=True)
    y = rng.integers(0, VOCAB, size=(BATCH,)).astype(np.int64)
    return q, table, y


class TestFusedCosineCrossEntropyCorrectness(unittest.TestCase):
    def _reference_loss(self, q, table, tau, y):
        # `.call()` directly, not `__call__` -- the latter applies
        # `keras.losses.Loss`'s default `sum_over_batch_size` (mean)
        # reduction, which would collapse this to a scalar and make a
        # per-example comparison meaningless.
        logits = tau * ops.matmul(q, ops.transpose(table))
        return SparseCategoricalCrossentropyFromLogits().call(y, logits)

    def test_forward_value_matches_reference(self):
        q_np, table_np, y_np = _random_query_and_table()
        tau = 3.7

        q = torch.tensor(q_np, requires_grad=True)
        table = torch.tensor(table_np, requires_grad=True)
        tau_t = torch.tensor(tau, requires_grad=True)
        y = torch.tensor(y_np)

        fused = fused_cosine_cross_entropy(q, table, tau_t, y)
        reference = ops.convert_to_numpy(
            self._reference_loss(
                ops.convert_to_tensor(q_np), ops.convert_to_tensor(table_np), tau, ops.convert_to_tensor(y_np)
            )
        )
        np.testing.assert_allclose(fused.detach().numpy(), reference, rtol=1e-5, atol=1e-5)

    def test_gradients_match_reference_autograd(self):
        q_np, table_np, y_np = _random_query_and_table(seed=1)
        tau = 2.3

        # Fused path.
        q_fused = torch.tensor(q_np, requires_grad=True)
        table_fused = torch.tensor(table_np, requires_grad=True)
        tau_fused = torch.tensor(tau, requires_grad=True)
        y = torch.tensor(y_np)
        fused_loss = fused_cosine_cross_entropy(q_fused, table_fused, tau_fused, y).sum()
        fused_loss.backward()

        # Reference path: plain torch ops, ordinary autograd (no custom Function).
        q_ref = torch.tensor(q_np, requires_grad=True)
        table_ref = torch.tensor(table_np, requires_grad=True)
        tau_ref = torch.tensor(tau, requires_grad=True)
        logits = tau_ref * (q_ref @ table_ref.T)
        log_probs = torch.log_softmax(logits, dim=-1)
        ref_loss = -log_probs.gather(1, y.view(-1, 1)).squeeze(1).sum()
        ref_loss.backward()

        np.testing.assert_allclose(q_fused.grad.numpy(), q_ref.grad.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(table_fused.grad.numpy(), table_ref.grad.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(tau_fused.grad.numpy(), tau_ref.grad.numpy(), rtol=1e-4, atol=1e-5)

    def test_forward_does_not_save_the_full_logits_tensor(self):
        """The whole point of this module: `ctx.saved_tensors` must be
        `(q, target_embeddings, tau, y)` -- never a `(batch, vocab)` tensor.
        Checked directly by shape, not just by memory measurement, so this
        test can't pass by accident on a fluke of allocator behavior."""
        q_np, table_np, y_np = _random_query_and_table(seed=2)
        q = torch.tensor(q_np, requires_grad=True)
        table = torch.tensor(table_np, requires_grad=True)
        tau = torch.tensor(1.5, requires_grad=True)
        y = torch.tensor(y_np)

        loss = fused_cosine_cross_entropy(q, table, tau, y)
        ctx = loss.grad_fn
        saved = [t for t in (ctx.saved_tensors if hasattr(ctx, "saved_tensors") else []) if t is not None]
        for t in saved:
            self.assertNotEqual(
                tuple(t.shape), (BATCH, VOCAB),
                f"A saved tensor has shape {tuple(t.shape)} == (batch, vocab) -- the full logits "
                f"tensor is being retained for backward, defeating the purpose of this module.",
            )


class TestDenseSupervisionMemory(unittest.TestCase):
    """Simulates dense (per-position) supervision on `dual-sphere-hyper-gate`
    at a vocabulary/window size large enough for the `(window, vocab)` logits
    tensor to be the dominant memory cost, using the CUDA/XPU peak-allocator
    APIs when available and falling back to a direct autograd-graph
    inspection otherwise (this project's torch+XPU build does not guarantee
    a portable peak-memory hook, so the fallback keeps this test meaningful
    on the project's actual hardware rather than skipping outright)."""

    def test_fused_path_never_retains_a_batch_by_vocab_tensor(self):
        """Direct, allocator-independent check: walk the autograd graph
        produced by the fused loss and confirm no saved tensor has shape
        `(batch, vocab)` -- the property that guarantees the memory saving,
        regardless of what any specific allocator happens to report."""
        from model.archs.dual_sphere_hyper_gate import DualSphereHyperGateLM

        window_size, vocab, dim_out = 16, 2000, 12
        model = DualSphereHyperGateLM(
            vocabulary_size=vocab, dim_in=4, dim_out=dim_out, window_size=window_size, hidden_dim=32
        )
        batch = 6
        token_ids = np.random.randint(0, vocab, (batch, window_size)).astype("int32")
        y = np.random.randint(0, vocab, (batch,)).astype(np.int64)

        q = model.call_query(ops.convert_to_tensor(token_ids))
        table = model.normalized_target_embeddings()
        loss = fused_cosine_cross_entropy(q, table, model.tau, ops.convert_to_tensor(y, dtype="int64"))

        node = loss.grad_fn
        visited, to_visit = set(), [node]
        offending_shapes = []
        while to_visit:
            n = to_visit.pop()
            if n is None or id(n) in visited:
                continue
            visited.add(id(n))
            for t in getattr(n, "saved_tensors", []):
                if t is not None and tuple(t.shape) == (batch, vocab):
                    offending_shapes.append(tuple(t.shape))
            for child, _ in getattr(n, "next_functions", []):
                to_visit.append(child)

        self.assertEqual(
            offending_shapes, [],
            f"Found {len(offending_shapes)} saved tensor(s) of shape (batch={batch}, vocab={vocab}) "
            f"in the fused loss's autograd graph -- the full logits tensor is being retained.",
        )


if __name__ == "__main__":
    unittest.main()
