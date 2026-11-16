"""`gated-deltanet`: the chunked delta rule against the token loop (values and gradients, document resets, carried
state), the model's full rows against token-by-token generation, the conv carry across calls, the MoE balancing; on
CUDA the fla kernel against the torch chunks. CPU (CUDA parts skipped elsewhere)."""

import unittest

import numpy as np

from tests.helpers import VOCAB  # noqa: F401  (pins the device to CPU)

import torch  # noqa: E402

from model.archs.delta_rule import delta_rule_chunk, delta_rule_loop, delta_rule_read_weights  # noqa: E402
from model.archs.gated_deltanet import GatedDeltaNetLM  # noqa: E402


def inputs(B=2, T=37, H=2, K=4, V=3, seed=0, dtype=torch.float64, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    q, k, v = r(B, T, H, K), r(B, T, H, K), r(B, T, H, V)
    gl = -torch.rand(B, T, H, generator=g, dtype=dtype) * 2  # log decay in (-2, 0)
    beta = torch.rand(B, T, H, generator=g, dtype=dtype)
    reset = torch.rand(B, T, generator=g) < 0.08
    reset[:, 0] = True
    init = r(B, H, K, V)
    return [x.to(device) for x in (q, k, v, gl, beta, reset, init)]


def tiny(**kw):
    torch.manual_seed(0)
    return GatedDeltaNetLM(**{"vocabulary_size": 50, "dim": 16, "num_layers": 2, "num_heads": 2, "head_dim": 8,
                              "num_experts": 3, "expert_hidden": 16, "window_size": 64, **kw})


class TestDeltaRule(unittest.TestCase):
    def test_chunks_match_the_loop(self):
        for chunk in (8, 16, 64):
            for with_init in (False, True):
                q, k, v, g, beta, reset, init = inputs()
                if with_init:
                    reset[:, 0] = False  # the carried state reaches the first document
                outs = []
                for f in (delta_rule_loop, lambda *a: delta_rule_chunk(*a, chunk=chunk)):
                    xs = [x.clone().requires_grad_() for x in (q, k, v, g, beta)]
                    i0 = init.clone().requires_grad_() if with_init else None
                    o, S = f(*xs, reset, i0)
                    w = torch.Generator().manual_seed(1)  # by shape (randn_like follows the strides)
                    wo = torch.randn(o.shape, generator=w, dtype=o.dtype)
                    wS = torch.randn(S.shape, generator=w, dtype=S.dtype)
                    ((o * wo).sum() + (S * wS).sum()).backward()
                    outs.append([o, S] + [x.grad for x in xs] + ([i0.grad] if with_init else []))
                for a, b in zip(*outs):
                    self.assertLess(float((a - b).abs().max()), 1e-9, (chunk, with_init))

    def test_read_weights_rebuild_the_read(self):
        q, k, v, g, beta, reset, _ = inputs(B=2, T=40)
        reset[1] = reset[0]  # read weights take one reset row for all rows
        o, _ = delta_rule_loop(q, k, v, g, beta, reset)
        for t in (0, 5, 17, 39):
            c, start = delta_rule_read_weights(q, k, g, beta, reset[0], t)
            rebuilt = torch.einsum("bhi,bihv->bhv", c, v[:, start:t + 1])
            self.assertLess(float((rebuilt - o[:, t]).abs().max()), 1e-10, t)
            self.assertTrue(start == 0 or bool(reset[0, start]))

    def test_reset_drops_the_state_before(self):
        q, k, v, g, beta, reset, init = inputs(B=1, T=20)
        reset[:] = False
        reset[0, [0, 9]] = True
        o, _ = delta_rule_chunk(q, k, v, g, beta, reset, chunk=8)
        o2, _ = delta_rule_chunk(q[:, 9:], k[:, 9:], v[:, 9:], g[:, 9:], beta[:, 9:], reset[:, 9:], chunk=8)
        self.assertLess(float((o[:, 9:] - o2).abs().max()), 1e-12)


class TestModel(unittest.TestCase):
    def rows(self, B=2, T=50, seed=3):
        rng = np.random.default_rng(seed)
        x = rng.integers(1, 50, (B, T)).astype(np.int32)
        x[:, 0] = 0
        x[0, 23] = 0  # a document start mid-row
        return x

    def test_generation_matches_the_full_row(self):
        """Token by token and in random chunks: the delta-rule state and the conv taps carried across calls."""
        m = tiny()
        x = self.rows()
        with torch.no_grad():
            full = m.readout(m.readout_input(m.dense_hidden(x)))
            rng = np.random.default_rng(0)
            for sizes in ([1] * 50, list(rng.integers(1, 9, 20))):
                state, t = None, 0
                for size in sizes:
                    if t >= 50:
                        break
                    out, state = m.advance(x[:, t:t + size], state)
                    t = min(50, t + size)
                    self.assertLess(float((out - full[:, t - 1]).abs().max()), 1e-4, t)

    def test_valid_positions_only(self):
        m = tiny()
        x = self.rows()
        valid = np.random.default_rng(1).random(x.shape) < 0.3
        with torch.no_grad():
            full = m.dense_hidden(x).reshape(-1, m.dim)[torch.as_tensor(valid.reshape(-1))]
            part = m.dense_hidden_valid(x, torch.as_tensor(valid))
        self.assertLess(float((full - part).abs().max()), 1e-5)

    def test_router_bias_moves_only_when_training(self):
        m = tiny()
        x = self.rows()
        bias = m._params[0]["router_bias"]
        before = bias.numpy().copy()
        with torch.no_grad():
            m.dense_hidden(x)
        np.testing.assert_array_equal(bias.numpy(), before)
        m.dense_hidden(x).sum().backward()
        np.testing.assert_array_equal(bias.numpy(), before)  # queued: applied by the next forward pass
        with torch.no_grad():
            m.dense_hidden(x)
        self.assertGreater(float(np.abs(bias.numpy() - before).max()), 0)


class TestRecompute(unittest.TestCase):
    def test_recomputed_layers_give_the_same_gradients_and_move_the_bias_once(self):
        x = TestModel().rows()
        res = []
        m = tiny()
        for recompute in (False, True):
            m.recompute = recompute
            for w in m.trainable_weights:
                w.value.grad = None
            for lay in m._params:  # the queued balancing steps moved every layer's bias: the same routing both times
                lay["router_bias"].assign(np.zeros(m.num_experts, "float32"))
            before = m._params[0]["router_bias"].numpy().copy()
            m.dense_hidden(x).square().sum().backward()
            m._apply_balance()  # queued for the next forward pass
            step = np.abs(m._params[0]["router_bias"].numpy() - before)
            self.assertLessEqual(float(step.max()), m.router_bias_rate * 1.0001)  # one update, not two
            res.append([w.value.grad for w in m.trainable_weights])
        for a, b in zip(*res):
            if a is None or b is None:
                self.assertIs(a, b)
                continue
            self.assertLess(float((a - b).abs().max() / (a.abs().max() + 1e-12)), 1e-4)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class TestCuda(unittest.TestCase):
    def test_fla_kernel_matches_the_torch_chunks(self):
        from model.archs.delta_rule import _fla_chunk, delta_rule_fla

        if _fla_chunk is None:
            self.skipTest("flash-linear-attention not installed")
        q, k, v, g, beta, reset, _ = inputs(B=3, T=300, H=4, K=64, V=64, dtype=torch.float32, device="cuda")
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # the reference in true fp32 (the model turns TF32 on)
        try:
            want, _ = delta_rule_chunk(q, k, v, g, beta, reset)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        for dt, tol in ((torch.float32, 2e-3), (torch.bfloat16, 3e-2)):
            xs = [x.to(dt) for x in (q, k, v)]
            got, _ = delta_rule_fla(*xs, g, beta, reset)
            err = float((got.float() - want).abs().max() / want.abs().max())
            self.assertLess(err, tol, dt)

    def test_fla_gradients_match(self):
        from model.archs.delta_rule import _fla_chunk, delta_rule_fla

        if _fla_chunk is None:
            self.skipTest("flash-linear-attention not installed")
        q, k, v, g, beta, reset, _ = inputs(B=2, T=200, H=2, K=64, V=64, dtype=torch.float32, device="cuda")
        grads = []
        for f in (delta_rule_chunk, delta_rule_fla):
            xs = [x.clone().requires_grad_() for x in (q, k, v, g, beta)]
            o, _ = f(*xs, reset)
            w = torch.randn(o.shape, generator=torch.Generator().manual_seed(1)).to(o.device)
            (o.float() * w).sum().backward()
            grads.append([x.grad for x in xs])
        for a, b in zip(*grads):
            self.assertLess(float((a - b).abs().max() / a.abs().max()), 5e-3)

    def test_triton_conv_matches_the_reference(self):
        """gdn_kernels.conv_silu (Triton) against causal_conv + silu: values and gradients (x, taps), document starts
        mid-row, T and C not multiples of the tiles."""
        import torch.nn.functional as F

        from model.archs.gated_deltanet import causal_conv
        from model.archs.gdn_kernels import conv_silu

        g = torch.Generator(device="cuda").manual_seed(0)
        B, T, C = 3, 200, 300
        x = torch.randn(B, T, C, device="cuda", generator=g).to(torch.bfloat16)
        w = torch.randn(4, C, device="cuda", generator=g) * 0.5
        reset = torch.rand(B, T, device="cuda", generator=g) < 0.05
        reset[:, 0] = True
        d = torch.randn(B, T, C, device="cuda", generator=g)
        outs = []
        for triton_path in (True, False):
            xx, ww = x.clone().float().requires_grad_(), w.clone().requires_grad_()
            if triton_path:
                y = conv_silu(xx.to(torch.bfloat16), ww, reset.long().cumsum(1)).float()
            else:
                y = F.silu(causal_conv(xx, ww, reset, None)[0])
            (y * d).sum().backward()
            outs.append([y.detach(), xx.grad, ww.grad])
        for a, b in zip(*outs):
            self.assertLess(float((a - b).abs().max() / (b.abs().max() + 1e-9)), 2e-2)

    def test_moe_experts(self):
        """MoEExperts (all experts at once, gradients written in place, bf16 matmuls) against per-expert fp32 math,
        with an expert that gets no rows."""
        import torch.nn.functional as F

        from model.archs.gated_deltanet import MoEExperts

        g = torch.Generator(device="cuda").manual_seed(0)
        E, D, Fh, N = 3, 16, 32, 300
        r = lambda *s: torch.randn(*s, device="cuda", generator=g) * 0.3
        xs, w1, b1, w2, b2, d = r(N, D), r(E, D, Fh), r(E, Fh), r(E, Fh, D), r(E, D), r(N, D)
        counts = [120, 0, 180]
        outs = []
        for fused in (True, False):
            ps = [t.clone().requires_grad_() for t in (w1, b1, w2, b2)]
            x = xs.clone().requires_grad_()
            if fused:
                y = MoEExperts.apply(x.to(torch.bfloat16), *ps, counts)
            else:
                ys, s = [], 0
                for e, n in enumerate(counts):
                    ys.append(F.gelu(x[s:s + n] @ ps[0][e] + ps[1][e], approximate="tanh") @ ps[2][e] + ps[3][e])
                    s += n
                y = torch.cat(ys)
            (y * d).sum().backward()
            outs.append([y.detach(), x.grad] + [p.grad if p.grad is not None else torch.zeros_like(p) for p in ps])
        for a, b in zip(*outs):
            self.assertLess(float((a - b).abs().max() / (b.abs().max() + 1e-9)), 2e-2)

    def test_fused_path_matches_eager(self):
        """The compiled CUDA path (fused norms / conv / gates / experts) against the eager one, values and gradients."""
        import model.archs.gated_deltanet as gd
        from model.device import setup_device

        setup_device("cuda")
        self.addCleanup(setup_device, "cpu")
        m = tiny(router_bias_rate=0.0)
        x = TestModel().rows(B=4, T=128)
        results = []
        for fused in (False, True):
            gd.FUSED = fused
            for w in m.trainable_weights:
                w.value.grad = None
            h = m.dense_hidden(x)
            (h.float() * torch.linspace(-1, 1, h.numel(), device=h.device).reshape(h.shape)).sum().backward()
            results.append((h.float(), [None if w.value.grad is None else w.value.grad.float()
                                        for w in m.trainable_weights]))
        gd.FUSED = True
        err = float((results[0][0] - results[1][0]).abs().max() / results[0][0].abs().max())
        self.assertLess(err, 3e-2)
        for (a, b), w in zip(zip(results[0][1], results[1][1]), m.trainable_weights):
            if a is None or b is None:  # the readout (dense_hidden stops before it)
                self.assertIs(a, b, w.path)
                continue
            # bf16 noise, and tokens near a routing tie may pick another expert in the other path: gradients agree
            # in direction (the experts and the conv are checked exactly in their own tests)
            cos = float(torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)))
            self.assertGreater(cos, 0.9 if any(n in w.path for n in ("ffn_norm", "router")) else 0.98, w.path)

    def test_model_on_cuda_matches_cpu(self):
        from model.device import setup_device

        setup_device("cuda")
        self.addCleanup(setup_device, "cpu")  # the other tests run on the CPU (fp32)
        m = tiny()
        x = TestModel().rows()
        with torch.no_grad():
            cuda = m.dense_hidden(x).float().cpu()
            state, outs = None, []
            for t in range(x.shape[1]):  # the torch chunk path (carried state)
                out, state = m.advance(x[:, t:t + 1], state)
                outs.append(out.float().cpu())
        logits = m.readout(m.readout_input(torch.as_tensor(cuda, device="cuda"))).float().cpu()
        err = float((logits[:, -1] - outs[-1]).abs().max() / logits.abs().max())
        self.assertLess(err, 5e-2)  # bf16 matmuls


if __name__ == "__main__":
    unittest.main()
