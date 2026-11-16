"""`model.archs.bank_memory`: the chunked (parallel) forms against the token loops (the definitions), float64, values
and gradients, with document resets, for every chunk size and write normalization; the read never feeds back into the
banks; the layer row by row (chunks, loop) and position by position (generation) agree."""

import unittest

from tests.helpers import VOCAB  # noqa: F401  (pins the device to CPU)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model.archs.bank_memory import (LRUBankLayer, bank_chunk, bank_loop, bank_read, bank_write_chunk,  # noqa: E402
                                     bank_write_loop, lru_chunk, lru_loop, route,
                                     StreamBankLayer)

DT = torch.float64


def bank_inputs(B=2, T=29, H=2, D=5, E=3, M=7, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    r_ = lambda *s: torch.randn(*s, generator=g, dtype=DT)
    q, k = F.normalize(r_(B, T, H, D), dim=-1), F.normalize(r_(B, T, H, D), dim=-1)
    v = r_(B, T, H, E)
    router = r_(H, D, M)
    loga = -torch.rand(H, M, generator=g, dtype=DT) * 1.5 - 0.01
    tau = torch.rand(H, generator=g, dtype=DT) * 5 + 1
    reset = torch.rand(B, T, generator=g) < 0.1
    reset[:, 0] = True
    return q, k, v, router, loga, tau, reset


def run(f, q, k, v, router, loga, tau, reset, K=2, **kw):
    xs = [x.clone().requires_grad_() for x in (q, k, v, router, tau)]
    r = route(xs[1], xs[3], K)
    o, (Kb, Vb, z) = f(xs[0], xs[1], xs[2], r, loga, xs[4], reset, **kw)
    w = torch.Generator().manual_seed(7)
    loss = (o * torch.randn(o.shape, generator=w, dtype=DT)).sum() + (Kb * torch.randn(Kb.shape, generator=w, dtype=DT)).sum() \
        + (Vb * torch.randn(Vb.shape, generator=w, dtype=DT)).sum() + (z * torch.randn(z.shape, generator=w, dtype=DT)).sum()
    loss.backward()
    return [o, Kb, Vb, z] + [x.grad for x in xs]


class TestLRU(unittest.TestCase):
    def test_chunks_match_the_loop(self):
        g = torch.Generator().manual_seed(0)
        B, T, N = 2, 37, 6
        b = torch.complex(torch.randn(B, T, N, generator=g, dtype=DT), torch.randn(B, T, N, generator=g, dtype=DT))
        log_lam = torch.complex(-torch.rand(N, generator=g, dtype=DT) - 0.01, torch.rand(N, generator=g, dtype=DT) * 6)
        reset = torch.rand(B, T, generator=g) < 0.1
        reset[:, 0] = True
        h0 = torch.complex(torch.randn(B, N, generator=g, dtype=DT), torch.randn(B, N, generator=g, dtype=DT))
        for init in (None, h0):
            ref, end = lru_loop(b, log_lam, reset if init is None else reset & False, init)
            for chunk in (1, 4, 8, 16, 64):
                h, e = lru_chunk(b, log_lam, reset if init is None else reset & False, init, chunk=chunk)
                self.assertLess(float((h - ref).abs().max()), 1e-12, (chunk, init is None))
                self.assertLess(float((e - end).abs().max()), 1e-12)


class TestInputGate(unittest.TestCase):
    def test_gated_lru_chunks_match_the_loop(self):
        """A decay per token (log lambda_t = g_t log lambda): chunks == loop, values and gradients, with resets."""
        g_ = torch.Generator().manual_seed(5)
        B, T, N = 2, 37, 6
        b = torch.complex(torch.randn(B, T, N, generator=g_, dtype=DT), torch.randn(B, T, N, generator=g_, dtype=DT))
        base = torch.complex(-torch.rand(N, generator=g_, dtype=DT) * 2 - 0.01, torch.rand(N, generator=g_, dtype=DT) * 6)
        gate = torch.rand(B, T, 1, generator=g_, dtype=DT).requires_grad_()
        reset = torch.rand(B, T, generator=g_) < 0.1
        reset[:, 0] = True
        wt = torch.randn(B, T, N, generator=g_, dtype=DT)
        ref, _ = lru_loop(b, gate * base, reset)
        gref, = torch.autograd.grad((torch.view_as_real(ref).sum(-1) * wt).sum(), gate)
        for chunk in (1, 4, 16, 64):
            h, _ = lru_chunk(b, gate * base, reset, chunk=chunk)
            self.assertLess(float((h - ref).detach().abs().max()), 1e-10, chunk)
            gg, = torch.autograd.grad((torch.view_as_real(h).sum(-1) * wt).sum(), gate)
            self.assertLess(float((gg - gref).abs().max()), 1e-9, chunk)

    def test_closed_gate_skips_the_token(self):
        """g = 0 at a position: the state passes it unchanged (no decay, no input)."""
        torch.manual_seed(0)
        m = LRUBankLayer(dim=12, lru_dim=8, heads=2, key_dim=6, value_dim=5, slots=9, input_gate=True).double()
        x = torch.randn(1, 6, 12, dtype=DT)
        reset = torch.zeros(1, 6, dtype=torch.bool)
        reset[:, 0] = True
        g = torch.ones(1, 6, 1, dtype=DT)
        g[0, 3] = 0
        h, _ = lru_loop(m.lru_input(x, g).detach(), (g * m.log_lam()).detach(), reset)
        self.assertLess(float((h[0, 3] - h[0, 2]).abs().max()), 1e-5)

    def test_layer_with_gate_chunks_loop_and_generation_agree(self):
        torch.manual_seed(3)
        m = LRUBankLayer(dim=12, lru_dim=8, heads=2, key_dim=6, value_dim=5, slots=9, topk=2, chunk=8, input_gate=True,
                         input_gate_bias=0.0).double()
        torch.nn.init.normal_(m.wg.weight)
        x = torch.randn(2, 27, 12, dtype=DT)
        reset = torch.zeros(2, 27, dtype=torch.bool)
        reset[:, 0] = True
        reset[1, 9] = True
        yc, yl = m(x, reset, "chunk"), m(x, reset, "loop")
        self.assertLess(float((yc - yl).detach().abs().max()), 1e-10)
        st, ys = None, []
        for t in range(27):
            y, st = m.step(x[:, t], reset[:, t], st)
            ys.append(y)
        self.assertLess(float((torch.stack(ys, 1) - yc).detach().abs().max()), 1e-10)


class TestLRUOrder(unittest.TestCase):
    def test_the_state_encodes_order_not_a_bag(self):
        """h_t = sum_j lambda^j b_{t-j}: swapping the tokens at offsets i, j changes the state by exactly
        (lambda^i - lambda^j)(b_x - b_y), so every order of a window gives a different state; lambda = 1 (a bag of
        words) gives the same state for all of them."""
        import itertools
        torch.manual_seed(0)
        m = LRUBankLayer(dim=16, lru_dim=8, lru_half_life=(0.5, 4.0)).double()
        emb = torch.randn(10, 16, dtype=DT)
        lam = m.log_lam().detach()
        perms = torch.tensor(list(itertools.permutations(range(5))))
        w = torch.tensor([3, 7, 1, 9, 4])[perms]  # all 120 orders of 5 distinct tokens
        reset = torch.zeros(w.shape, dtype=torch.bool)
        reset[:, 0] = True
        b = m.lru_input(emb[w]).detach()
        h = lru_loop(b, lam, reset)[0][:, -1]
        bag = lru_loop(b, torch.zeros_like(lam), reset)[0][:, -1]
        d = torch.cdist(torch.view_as_real(h).flatten(1), torch.view_as_real(h).flatten(1))
        d.fill_diagonal_(float("inf"))
        self.assertGreater(float(d.min() / h.abs().norm(dim=-1).mean()), 1e-2)  # 120 distinct states
        self.assertLess(float((bag - bag[0]).abs().max()), 1e-12)  # the bag: one state
        # the swap identity, offsets counted from the newest token (position 4)
        for i, j in [(0, 1), (0, 4), (2, 3)]:
            ws = w[0].clone()
            ws[4 - i], ws[4 - j] = w[0, 4 - j], w[0, 4 - i]
            hs = lru_loop(m.lru_input(emb[ws][None]).detach(), lam, reset[:1])[0][0, -1]
            bx, by = m.lru_input(emb[w[0, 4 - i]]).detach(), m.lru_input(emb[w[0, 4 - j]]).detach()
            expect = (lam * i).exp() * (by - bx) + (lam * j).exp() * (bx - by)
            self.assertLess(float((hs - h[0] - expect).abs().max()), 1e-12, (i, j))


class TestBanks(unittest.TestCase):
    def test_chunks_match_the_loop_values_and_gradients(self):
        for norm in ("mass", "ema", "lru", "none"):
            args = bank_inputs()
            ref = run(bank_loop, *args, norm=norm)
            for chunk in (1, 3, 8, 16, 29, 64):
                got = run(lambda *a, **k: bank_chunk(*a, chunk=chunk, **k), *args, norm=norm)
                for name, a, b in zip(["o", "Kb", "Vb", "z", "dq", "dk", "dv", "drouter", "dtau"], ref, got):
                    self.assertLess(float((a - b).detach().abs().max()), 1e-10, (norm, chunk, name))

    def test_carried_state_and_split_rows(self):
        """Two halves with the state carried == the whole row (the chunk boundary of the proof, at any position)."""
        q, k, v, router, loga, tau, reset = bank_inputs(T=30)
        reset[:, 15] = False
        r = route(k, router, 2)
        whole, _ = bank_chunk(q, k, v, r, loga, tau, reset, chunk=8)
        o1, st = bank_chunk(q[:, :13], k[:, :13], v[:, :13], r[:, :13], loga, tau, reset[:, :13], chunk=4)
        o2, _ = bank_chunk(q[:, 13:], k[:, 13:], v[:, 13:], r[:, 13:], loga, tau, reset[:, 13:], st, chunk=8)
        self.assertLess(float((torch.cat([o1, o2], 1) - whole).abs().max()), 1e-12)

    def test_the_read_does_not_feed_back(self):
        """Changing the query at one position changes only that position's output (the banks never read)."""
        q, k, v, router, loga, tau, reset = bank_inputs()
        r = route(k, router, 2)
        o, _ = bank_chunk(q, k, v, r, loga, tau, reset, chunk=8)
        q2 = q.clone()
        q2[:, 11] = F.normalize(torch.randn_like(q2[:, 11]), dim=-1)
        o2, _ = bank_chunk(q2, k, v, r, loga, tau, reset, chunk=8)
        diff = (o2 - o).abs().amax((0, 2, 3))
        self.assertGreater(float(diff[11]), 1e-6)
        self.assertLess(float(torch.cat([diff[:11], diff[12:]]).max()), 1e-14)

    def test_routing_sums_to_one_on_k_slots(self):
        q, k, v, router, loga, tau, reset = bank_inputs()
        for K in (1, 2, 3):
            r = route(k, router, K)
            self.assertTrue(torch.allclose(r.sum(-1), torch.ones_like(r[..., 0])))
            self.assertTrue(bool(((r > 0).sum(-1) == K).all()))

    def test_mass_mode_reads_averages(self):
        """One write into an empty bank, read with any query: the value comes back whole (average of one item)."""
        B, H, D, E, M = 1, 1, 4, 3, 5
        q = F.normalize(torch.randn(B, 1, H, D, dtype=DT), dim=-1)
        k = F.normalize(torch.randn(B, 1, H, D, dtype=DT), dim=-1)
        v = torch.randn(B, 1, H, E, dtype=DT)
        r = torch.zeros(B, 1, H, M, dtype=DT)
        r[..., 2] = 1
        o, _ = bank_loop(q, k, v, r, torch.full((H, M), -0.3, dtype=DT), torch.ones(H, dtype=DT),
                         torch.ones(B, 1, dtype=torch.bool))
        self.assertLess(float((o - v).abs().max()), 1e-12)


class TestLayer(unittest.TestCase):
    def test_chunks_loop_and_generation_agree(self):
        torch.manual_seed(0)
        for norm in ("mass", "ema"):
            m = LRUBankLayer(dim=12, lru_dim=8, heads=2, key_dim=6, value_dim=5, slots=9, topk=2, chunk=8,
                             norm=norm).double()
            x = torch.randn(2, 27, 12, dtype=DT)
            reset = torch.zeros(2, 27, dtype=torch.bool)
            reset[:, 0] = True
            reset[0, 13] = True
            yc, yl = m(x, reset, "chunk"), m(x, reset, "loop")
            self.assertLess(float((yc - yl).abs().max()), 1e-10, norm)
            st, ys = None, []
            for t in range(27):
                y, st = m.step(x[:, t], reset[:, t], st)
                ys.append(y)
            self.assertLess(float((torch.stack(ys, 1) - yc).abs().max()), 1e-10, norm)


class TestErase(unittest.TestCase):
    def test_erase_chunks_match_the_loop(self):
        """Per-position decays (log factors <= 0, some near the cap): chunks == loop, values and gradients."""
        q, k, v, router, loga, tau, reset = bank_inputs()
        g = torch.Generator().manual_seed(3)
        er = torch.log1p(-0.95 * torch.rand(*k.shape[:3], router.shape[-1], generator=g, dtype=DT) ** 3)
        for norm in ("mass", "ema"):
            ref = run(bank_loop, q, k, v, router, loga, tau, reset, norm=norm)
            for chunk in (1, 5, 16, 64):
                def f(q_, k_, v_, r_, la_, t_, rs_, chunk=chunk, **kw):
                    mem, st = bank_write_chunk(k_, v_, r_, la_, rs_, chunk=chunk, erase=er, **kw)
                    return bank_read(mem, q_, t_), st

                def fl(q_, k_, v_, r_, la_, t_, rs_, **kw):
                    mem, st = bank_write_loop(k_, v_, r_, la_, rs_, erase=er, **kw)
                    return bank_read(mem, q_, t_), st
                a_, b_ = run(fl, q, k, v, router, loga, tau, reset, norm=norm), run(f, q, k, v, router, loga, tau, reset, norm=norm)
                for name, x, y in zip(["o", "Kb", "Vb", "z", "dq", "dk", "dv", "drouter", "dtau"], a_, b_):
                    self.assertLess(float((x - y).detach().abs().max()), 1e-10, (norm, chunk, name))
                self.assertGreater(float((a_[0] - ref[0]).detach().abs().max()), 1e-3)  # the erase changes the reads

    def test_layer_with_erase_chunks_loop_and_generation_agree(self):
        torch.manual_seed(1)
        m = LRUBankLayer(dim=12, lru_dim=8, heads=2, key_dim=6, value_dim=5, slots=9, topk=2, chunk=8, erase=True,
                         erase_bias=1.0).double()
        torch.nn.init.normal_(m.we.weight)
        x = torch.randn(2, 27, 12, dtype=DT)
        reset = torch.zeros(2, 27, dtype=torch.bool)
        reset[:, 0] = True
        reset[1, 9] = True
        yc, yl = m(x, reset, "chunk"), m(x, reset, "loop")
        self.assertLess(float((yc - yl).detach().abs().max()), 1e-10)
        st, ys = None, []
        for t in range(27):
            y, st = m.step(x[:, t], reset[:, t], st)
            ys.append(y)
        self.assertLess(float((torch.stack(ys, 1) - yc).detach().abs().max()), 1e-10)


class TestReadErase(unittest.TestCase):
    def test_split_banks_chunks_loop_and_generation_agree(self):
        """erase "read": the value bank is erased where the key bank's read landed (a function of the key bank only):
        chunks == loop == generation, values and gradients, with resets."""
        torch.manual_seed(2)
        m = LRUBankLayer(dim=12, lru_dim=8, heads=2, key_dim=6, value_dim=5, slots=9, topk=2, chunk=8, erase="read",
                         erase_bias=1.0).double()
        torch.nn.init.normal_(m.we.weight)
        x = torch.randn(2, 27, 12, dtype=DT)
        reset = torch.zeros(2, 27, dtype=torch.bool)
        reset[:, 0] = True
        reset[1, 9] = True
        wt = torch.randn(2, 27, 12, dtype=DT)
        params = list(m.parameters())
        yc, yl = m(x, reset, "chunk"), m(x, reset, "loop")
        st, ys = None, []
        for t in range(27):
            y, st = m.step(x[:, t], reset[:, t], st)
            ys.append(y)
        ys = torch.stack(ys, 1)
        gref = torch.autograd.grad((yc * wt).sum(), params)
        for name, y in (("loop", yl), ("step", ys)):
            self.assertLess(float((y - yc).detach().abs().max()), 1e-10, name)
            for gc, gy in zip(gref, torch.autograd.grad((y * wt).sum(), params)):
                self.assertLess(float((gc - gy).abs().max()), 1e-10, name)
        m.we.bias.data.fill_(-30.0)  # no erase: the same reads as the plain layer
        m.we.weight.data.zero_()
        plain = LRUBankLayer(dim=12, lru_dim=8, heads=2, key_dim=6, value_dim=5, slots=9, topk=2, chunk=8).double()
        plain.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("we.")})
        self.assertLess(float((m(x, reset) - plain(x, reset)).detach().abs().max()), 1e-10)


class TestStreamBank(unittest.TestCase):
    def test_stream_keys_chunks_loop_and_generation_agree(self):
        """Keys / queries from one projection of the input (no LRU), with a second hop: chunks == loop == generation."""
        torch.manual_seed(4)
        m = StreamBankLayer(dim=12, heads=2, key_dim=6, value_dim=5, slots=9, topk=2, chunk=8, hops=2).double()
        x = torch.randn(2, 27, 12, dtype=DT)
        reset = torch.zeros(2, 27, dtype=torch.bool)
        reset[:, 0] = True
        reset[0, 11] = True

        def run_(x, reset, mode, state=None):
            mem, q, state = m.write(x, reset, mode, state)
            h = x + m.read(mem, q, 0)
            return h + m.read(mem, m.query(h, 1), 1), state

        yc, _ = run_(x, reset, "chunk")
        yl, _ = run_(x, reset, "loop")
        st, ys = None, []
        for t in range(27):
            y, st = run_(x[:, t:t + 1], reset[:, t:t + 1], "loop", st)
            ys.append(y)
        self.assertLess(float((yl - yc).detach().abs().max()), 1e-10)
        self.assertLess(float((torch.cat(ys, 1) - yc).detach().abs().max()), 1e-10)


class TestHops(unittest.TestCase):
    def test_second_hop_parallel_equals_loop_and_generation(self):
        """Hop 2's query is made from hop 1's read at the same position (through an MLP) and reads the same banks: the
        chunked rows, the token loop and position-by-position generation agree, values and gradients."""
        torch.manual_seed(0)
        m = LRUBankLayer(dim=12, lru_dim=8, heads=2, key_dim=6, value_dim=5, slots=9, topk=2, chunk=8, hops=2).double()
        mlp = torch.nn.Sequential(torch.nn.Linear(12, 24), torch.nn.GELU(), torch.nn.Linear(24, 12)).double()
        x = torch.randn(2, 27, 12, dtype=DT)
        reset = torch.zeros(2, 27, dtype=torch.bool)
        reset[:, 0] = True
        reset[0, 13] = True
        wt = torch.randn(2, 27, 12, dtype=DT)
        params = list(m.parameters()) + list(mlp.parameters())

        def hops(x, reset, mode, state=None):
            mem, q, state = m.write(x, reset, mode, state)
            h = x + m.read(mem, q, 0)
            h = h + mlp(h)
            return h + m.read(mem, m.query(h, 1), 1), state

        def grads(y):
            return torch.autograd.grad((y * wt).sum(), params)

        yc, _ = hops(x, reset, "chunk")
        yl, _ = hops(x, reset, "loop")
        st, ys = None, []
        for t in range(27):
            y, st = hops(x[:, t:t + 1], reset[:, t:t + 1], "loop", st)
            ys.append(y)
        ys = torch.cat(ys, 1)
        gref = grads(yc)
        for name, y in (("loop", yl), ("step", ys)):
            self.assertLess(float((y - yc).detach().abs().max()), 1e-10, name)
            for gc, gy in zip(gref, grads(y)):
                self.assertLess(float((gc - gy).abs().max()), 1e-10, name)
        # hop 2 changes the output (it is not a no-op)
        mem, q, _ = m.write(x, reset, "chunk")
        self.assertGreater(float(m.read(mem, m.query(x, 1), 1).abs().max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
