"""The assembly fp16 matmul (`model.kernels.hgemm`) against float64 on the CPU, and the fused MLP with it."""

import unittest

import torch

from model.device import setup_device

setup_device("xpu")

from model.kernels import hgemm, rsm_scan  # noqa: E402

HAVE = torch.xpu.is_available() and hgemm.load() is not None


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float((y.cpu().double() - ref).abs().max() / ref.abs().max())


@unittest.skipUnless(HAVE, f"fp16 matmul unavailable: {hgemm.error()}")
class TestHGemm(unittest.TestCase):
    def check(self, a: torch.Tensor, b: torch.Tensor, tol: float = 2e-3, **kw) -> None:
        ref = a.cpu().double() @ b.cpu().double()
        self.assertLess(_rel(hgemm.mm(a, b, **kw), ref), tol)

    def test_shapes_and_views(self):
        """Rows not a multiple of 16, k not of 16, transposed operands (strides), n = 32 and 3072."""
        g = torch.Generator().manual_seed(0)
        for m, k, n in ((1000, 200, 768), (37, 1000, 32), (2048, 768, 3072), (3, 3, 64)):
            a = torch.randn(m, k, generator=g).to("xpu")
            b = (torch.randn(k, n, generator=g) / k ** 0.5).to("xpu")
            self.check(a, b)
            self.check(a.T.contiguous().T, b.T.contiguous().T)  # column-major operands

    def test_scaled_gradients(self):
        """Gradient-sized values (1e-7: below fp16's normal range) are scaled in and out: the error stays relative.
        Unscaled they underflow."""
        g = torch.Generator().manual_seed(1)
        dy = (torch.randn(1000, 3072, generator=g) * 1e-7).to("xpu")
        w = (torch.randn(768, 3072, generator=g) * 0.03).to("xpu")
        h = torch.randn(1000, 3072, generator=g).to("xpu")
        self.check(dy, w.T, scale_a=True)  # dX = dY W^T
        self.check(h.T, dy, scale_b=True)  # dW = H^T dY
        ref = dy.cpu().double() @ w.T.cpu().double()
        self.assertGreater(_rel(hgemm.mm(dy, w.T), ref), 0.5)

    def test_out_and_alpha(self):
        a, b = torch.randn(64, 96, device="xpu"), torch.randn(96, 64, device="xpu")
        out = torch.full((64, 64), 7.0, device="xpu")
        hgemm.mm(a, b, alpha=0.25, out=out)
        ref = 0.25 * (a.cpu().double() @ b.cpu().double())
        self.assertLess(_rel(out, ref), 2e-3)

    def test_fused_mlp_half_matches_fp32(self):
        """`norm_mlp(..., half=True)` vs the fp32 path: output and every gradient within fp16-product error."""
        g = torch.Generator().manual_seed(2)
        R, D, H = 600, 128, 512

        def make():
            torch.manual_seed(0)
            x = torch.randn(R, D, generator=g.manual_seed(3)).to("xpu").requires_grad_()
            ws = [torch.randn(D, generator=g).add(1).to("xpu"), torch.randn(D, H, generator=g).mul(D ** -0.5).to("xpu"),
                  torch.randn(H, generator=g).mul(0.1).to("xpu"), torch.randn(H, D, generator=g).mul(H ** -0.5).to("xpu"),
                  torch.randn(D, generator=g).mul(0.1).to("xpu")]
            return x, [w.requires_grad_() for w in ws]

        outs = {}
        for half in (False, True):
            x, (scale, w1, b1, w2, b2) = make()
            dy = torch.randn(R, D, generator=g.manual_seed(4)).mul(1e-6).to("xpu")
            y = rsm_scan.norm_mlp(x, scale, w1, b1, w2, b2, half=half)
            y.backward(dy)
            outs[half] = [y.detach()] + [t.grad for t in (x, scale, w1, b1, w2, b2)]
        for a, b in zip(outs[False], outs[True]):
            self.assertLess(_rel(b, a.cpu().double()), 5e-3)


if __name__ == "__main__":
    unittest.main()
