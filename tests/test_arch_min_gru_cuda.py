"""`min-gru-cuda` against `min-gru`: the parallel scan's values and gradients, and the whole model (static memory
included) with the same weights. CUDA only (skipped elsewhere)."""

import unittest

import numpy as np

import tests.test_arch_min_gru  # noqa: F401  (imported first: it calls setup_device("cpu") on import; setUp then sets CUDA)
from model.device import setup_backend

setup_backend()

import torch  # noqa: E402

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "needs a CUDA GPU")
class TestMinGRUCuda(unittest.TestCase):
    def setUp(self):
        from model.device import setup_device

        setup_device("cuda")

    def test_scan_values_and_gradients(self):
        from model.archs.min_gru import linear_scan
        from model.archs.min_gru_cuda import ParallelScan

        g = torch.Generator(device="cuda").manual_seed(0)
        for B, T, D, G in ((3, 512, 64, 64), (2, 300, 32, 4), (4, 1000, 16, 16)):
            keep = torch.rand(B, T, G, device="cuda", generator=g, dtype=torch.float64)
            keep[:, ::97] = 0  # document starts
            write = torch.randn(B, T, D, device="cuda", generator=g, dtype=torch.float64)
            init = torch.randn(B, D, device="cuda", generator=g, dtype=torch.float64)
            d_out = torch.randn(B, T, D, device="cuda", generator=g, dtype=torch.float64)
            grads = []
            for f in (linear_scan, ParallelScan.apply):
                xs = [t.clone().requires_grad_() for t in (keep, write, init)]
                out = f(*xs)
                (out * d_out).sum().backward()
                grads.append([out] + [x.grad for x in xs])
            for a, b in zip(*grads):
                self.assertLess(float((a - b).abs().max()), 1e-9, (B, T, D, G))

    def test_model_matches_min_gru(self):
        for mlp_mult in (None, 4):
            with self.subTest(mlp_mult=mlp_mult):
                self._model_matches_min_gru(mlp_mult)

    def _model_matches_min_gru(self, mlp_mult):
        from model.archs.min_gru import MinGRULM
        from model.archs.min_gru_cuda import MinGRUCudaLM
        from tests.test_arch_min_gru import agent_rows

        kw = dict(vocabulary_size=50, dim=16, hidden_size=12, num_layers=2, window_size=128, static_memory=True,
                  memory_heads=2, memory_dim=4, mlp_mult=mlp_mult)
        torch.manual_seed(0)
        ref = MinGRULM(**kw)
        for p in ref._params:
            p["mem_out"].value.data.normal_(0, 0.5)
        fast = MinGRUCudaLM(**kw)
        for a, b in zip(fast.weights, ref.weights):
            a.assign(b.numpy())  # a copy: assigning the variable itself shares its tensor (and its gradient)
        x = agent_rows(B=3, T=128)
        outs = []
        for m in (ref, fast):
            out = m.dense_hidden(x)
            mb = m._from_raw(x)
            g = torch.Generator(device=mb["valid"].device).manual_seed(1)  # the same target positions for both
            valid = mb["valid"] & (torch.rand(mb["valid"].shape, device=mb["valid"].device, generator=g) < 0.5)
            part = m._run_memory(mb, valid)  # the train step's path: the last layer at target positions only
            (out.square().sum() + part.square().sum()).backward()
            outs.append((torch.cat([out.reshape(-1), part.reshape(-1)]), [w.value.grad for w in m.trainable_weights]))
        self.assertLess(float((outs[0][0] - outs[1][0]).abs().max()), 1e-5)
        for a, b in zip(outs[0][1], outs[1][1]):
            if a is None or b is None:
                self.assertIs(a, b)
                continue
            self.assertLess(float((a - b).abs().max() / (a.abs().max() + 1e-12)), 1e-4)


if __name__ == "__main__":
    unittest.main()
