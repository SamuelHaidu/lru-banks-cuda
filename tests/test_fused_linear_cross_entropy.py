"""`fused_linear_cross_entropy` must equal plain autograd cross-entropy.

Its backward is hand-derived (gradients are computed inside the forward
pass), so both gradients are checked against autograd on the unfused
formulation, across chunk sizes that do and do not divide the row count.
"""

import unittest

import tests.helpers  # noqa: F401  (pins the device before keras/torch use)

import torch

from model.fused_linear_cross_entropy import fused_linear_cross_entropy


def _reference(h, k, y):
    logits = h @ k
    return (torch.logsumexp(logits, -1) - logits.gather(1, y[:, None])[:, 0]).mean()


class TestFusedLinearCrossEntropy(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.n, self.d, self.v = 37, 8, 29
        self.h = torch.randn(self.n, self.d, generator=g, dtype=torch.float64)
        self.k = torch.randn(self.d, self.v, generator=g, dtype=torch.float64)
        self.y = torch.randint(0, self.v, (self.n,), generator=g)

    def _grads(self, fn, **kw):
        h = self.h.clone().requires_grad_(True)
        k = self.k.clone().requires_grad_(True)
        loss = fn(h, k, self.y, **kw)
        loss.backward()
        return loss.detach(), h.grad, k.grad

    def test_matches_autograd_for_any_chunking(self):
        ref = self._grads(_reference)
        for chunk in (1, 5, 16, 37, 1000):
            with self.subTest(chunk=chunk):
                got = self._grads(fused_linear_cross_entropy, chunk_size=chunk)
                for a, b in zip(got, ref):
                    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)

    def test_upstream_gradient_scales(self):
        h = self.h.clone().requires_grad_(True)
        k = self.k.clone().requires_grad_(True)
        (3.0 * fused_linear_cross_entropy(h, k, self.y, chunk_size=8)).backward()
        h2 = self.h.clone().requires_grad_(True)
        k2 = self.k.clone().requires_grad_(True)
        (3.0 * _reference(h2, k2, self.y)).backward()
        torch.testing.assert_close(h.grad, h2.grad)
        torch.testing.assert_close(k.grad, k2.grad)

    def test_only_h_requires_grad(self):
        h = self.h.clone().requires_grad_(True)
        fused_linear_cross_entropy(h, self.k, self.y, chunk_size=8).backward()
        h2 = self.h.clone().requires_grad_(True)
        _reference(h2, self.k, self.y).backward()
        torch.testing.assert_close(h.grad, h2.grad)

    def test_no_grad_forward_value(self):
        with torch.no_grad():
            got = fused_linear_cross_entropy(self.h, self.k, self.y, chunk_size=6)
        torch.testing.assert_close(got, _reference(self.h, self.k, self.y))

    def test_leading_dims_are_flattened(self):
        h3 = self.h[:36].reshape(4, 9, self.d)
        y3 = self.y[:36].reshape(4, 9)
        torch.testing.assert_close(
            fused_linear_cross_entropy(h3, self.k, y3, chunk_size=7), _reference(self.h[:36], self.k, self.y[:36])
        )

    def test_rejects_mismatched_rows(self):
        with self.assertRaises(ValueError):
            fused_linear_cross_entropy(self.h, self.k, self.y[:-1])


if __name__ == "__main__":
    unittest.main()
