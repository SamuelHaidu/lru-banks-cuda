"""`lru-banks`: a whole row equals position-by-position generation (the carried LRU and bank states, both hops, the
input gate, document resets); the CUDA kernels (`model/archs/bank_kernels.py`) equal the torch reference, values and
gradients (CUDA only)."""

import unittest

from tests.helpers import VOCAB  # noqa: F401  (pins the device to CPU)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model.archs import build_model  # noqa: E402

SMALL = {"vocabulary_size": 50, "embed_dim": 16, "dim": 32, "num_layers": 3, "lru_dim": 16, "num_heads": 2,
         "head_dim": 16, "slots": 16, "topk": 2, "hops": 2, "gate_from": 1, "window_size": 24, "chunk": 8}
# (sizes in multiples of 16: on CUDA the rows run the fused kernels, see `bank_kernels.supported`)


# relative; both sides run the torch path in fp32 (the fused kernels: `TestKernels`). CUDA's fp32 ops round
# differently: ~1e-3 of the (near-zero at init) logits there
TOL = 2e-3 if torch.cuda.is_available() else 1e-4


def small_model(**kw):
    m = build_model({"arch": "lru-banks", "arch_params": SMALL | kw})
    for v in m.trainable_variables:  # non-trivial gates / gate weights (they start at 0)
        if "w_gate" in v.path:
            v.assign(np.random.default_rng(1).normal(0, 0.5, v.shape).astype("float32"))
    return m


def ids_with_resets(B=2, T=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(1, 50, (B, T), generator=g)
    x[:, 0] = 0
    x[0, 9] = 0
    x[1, 17] = 0
    return x


class TestRowsVsGeneration(unittest.TestCase):
    def setUp(self):  # the torch path on any device (on CUDA the rows would run the bf16 kernels)
        from model.archs import lru_banks

        self.saved, lru_banks.FUSED = lru_banks.FUSED, False
        self.tf32, torch.backends.cuda.matmul.allow_tf32 = torch.backends.cuda.matmul.allow_tf32, False  # fp32 exact

    def tearDown(self):
        from model.archs import lru_banks

        lru_banks.FUSED = self.saved
        torch.backends.cuda.matmul.allow_tf32 = self.tf32

    def test_dense_row_equals_position_by_position(self):
        for hops in (1, 2):
            m = small_model(hops=hops)
            x = ids_with_resets()
            with torch.no_grad():
                dense = m.readout(m.readout_input(m.dense_hidden(x)))  # (B, T, V)
                state, steps = None, []
                for t in range(x.shape[1]):
                    logits, state = m.advance(x[:, t:t + 1], state)
                    steps.append(logits)
            err = float((torch.stack(steps, 1) - dense).abs().max())
            self.assertLess(err, TOL * float(dense.abs().max()), hops)

    def test_prefix_then_steps(self):
        """A prefix fed at once (chunks, carried out) then single positions == the whole row."""
        m = small_model()
        x = ids_with_resets(seed=3)
        with torch.no_grad():
            dense = m.readout(m.readout_input(m.dense_hidden(x)))
            _, state = m.advance(x[:, :13])
            logits, _ = m.advance(x[:, 13:14], state)
        self.assertLess(float((logits - dense[:, 13]).abs().max()), TOL * float(dense.abs().max()))

    def test_reset_isolates_documents(self):
        """Changing a token before a <bos> does not change anything after it."""
        m = small_model()
        x = ids_with_resets()
        x2 = x.clone()
        x2[0, 5] = (x2[0, 5] % 49) + 1
        with torch.no_grad():
            a, b = m.dense_hidden(x), m.dense_hidden(x2)
        self.assertGreater(float((a[0, 5:9] - b[0, 5:9]).abs().max()), 1e-6)
        self.assertLess(float((a[0, 9:] - b[0, 9:]).abs().max()), 1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA only")
class TestKernels(unittest.TestCase):
    def setUp(self):
        from model.archs import bank_kernels

        self.bk = bank_kernels

    def test_lru_scan_matches_torch(self):
        from model.archs.lru_banks import lru_states

        g = torch.Generator(device="cuda").manual_seed(0)
        B, T, N = 3, 300, 64
        for gated in (False, True):
            proj = torch.randn(B, T, 2 * N, device="cuda", generator=g, requires_grad=True)
            nu = (torch.rand(N, device="cuda", generator=g) * 3 - 2).requires_grad_()
            theta = (torch.rand(N, device="cuda", generator=g) * 6).requires_grad_()
            gate = torch.rand(B, T, 1, device="cuda", generator=g).requires_grad_() if gated else None
            reset = torch.rand(B, T, device="cuda", generator=g) < 0.02
            reset[:, 0] = True
            w = torch.randn(B, T, 2 * N, device="cuda", generator=g)
            ins = [proj, nu, theta] + ([gate] if gated else [])
            ref, _ = lru_states(proj.double(), nu.double(), theta.double(),
                                None if gate is None else gate.double(), reset, chunk=1)  # float64 loop
            gr = torch.autograd.grad((ref.float() * w).sum(), ins)
            out = self.bk.lru_scan(proj, nu, theta, gate, reset)
            go = torch.autograd.grad((out * w).sum(), ins)
            self.assertLess(float((out - ref).abs().max()), 1e-4, gated)
            for a, b in zip(go, gr):
                self.assertLess(float((a - b).abs().max() / (b.abs().max() + 1e-6)), 1e-3, gated)

    def test_bank_read_matches_torch(self):
        import torch.nn.functional as F

        from model.archs.bank_memory import bank_read, bank_write_chunk, route

        g = torch.Generator(device="cuda").manual_seed(1)
        B, T, H, D, M = 2, 200, 4, 32, 64
        q = F.normalize(torch.randn(B, T, H, D, device="cuda", generator=g), dim=-1).requires_grad_()
        k = F.normalize(torch.randn(B, T, H, D, device="cuda", generator=g), dim=-1).requires_grad_()
        v = torch.randn(B, T, H, D, device="cuda", generator=g).requires_grad_()
        router = torch.randn(H, D, M, device="cuda", generator=g).requires_grad_()
        shl = torch.exp(torch.linspace(0, np.log(1024), M, device="cuda"))
        loga = (np.log(0.5) / shl).repeat(H, 1)
        tau = torch.full((H,), 8.0, device="cuda", requires_grad=True)
        reset = torch.rand(B, T, device="cuda", generator=g) < 0.02
        reset[:, 0] = True
        w = torch.randn(B, T, H, D, device="cuda", generator=g)
        ins = [q, k, v, router, tau]
        mem, _ = bank_write_chunk(k.double(), v.double(), route(k.double(), router.double(), 2), loga.double(),
                                  reset, None, 16, "mass")
        ref = bank_read(mem, q.double(), tau.double())
        gr = torch.autograd.grad((ref.float() * w).sum(), ins)
        # fp32 operands (tf32 dots): tight; bf16 (training): loose
        for dtype, tol in ((torch.float32, 1e-2), (torch.bfloat16, 8e-2)):
            out = self.bk.BankWrite(k, v, route(k, router, 2), loga, reset, dtype=dtype).read(q, tau)
            go = torch.autograd.grad((out.float() * w).sum(), ins)
            self.assertLess(float((out.float() - ref).detach().abs().max() / ref.abs().max()), tol / 2, dtype)
            for name, a, b in zip("q k v router tau".split(), go, gr):
                self.assertLess(float((a - b).abs().max() / (b.abs().max() + 1e-6)), tol, (dtype, name))

    def test_model_fused_equals_torch_path(self):
        from model.archs import lru_banks

        m = small_model()
        m.to("cuda") if hasattr(m, "to") else None
        x = ids_with_resets().cuda()
        with torch.no_grad():
            lru_banks.FUSED = True
            a = m.dense_hidden(x)
            lru_banks.FUSED = False
            b = m.dense_hidden(x)
            lru_banks.FUSED = True
        self.assertLess(float((a - b).abs().max() / b.abs().max()), 5e-2)


if __name__ == "__main__":
    unittest.main()
