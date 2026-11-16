"""Changing a trained model's context window (extrapolation and fine-tuning).

`with_context_length` must keep every trained weight, zero only the new
per-distance entries, keep the kernel basis built for the training window,
and refuse architectures whose weights are tied to their window.
"""

import unittest

from tests.helpers import VOCAB, temp_workspace, tiny_arch_params, tiny_config

import numpy as np
from keras import ops

from model import config as cfg_mod
from model.archs import build_model
from model.train import build_or_load

EXTENDABLE = ("rope-logit-sum", "rope-log-gaussian-sum", "rope-kernel-attention", "rope-kernel-hops")


def _build(arch, **overrides):
    params = {**tiny_arch_params(arch), "dim": 16, **overrides}
    if "mlp_hidden" in params:
        params.update(mlp_hidden=8)
    if "query_hidden" in params:
        params.update(query_hidden=8, num_heads=2)
    return build_model({"arch": arch, "arch_params": params})


def _ids(T, seed=0):
    return np.random.default_rng(seed).integers(1, VOCAB, (3, T)).astype("int32")


class TestWithContextLength(unittest.TestCase):
    def test_same_length_is_identical(self):
        for arch in EXTENDABLE:
            with self.subTest(arch=arch):
                m = _build(arch)
                x = _ids(m.context_length)
                np.testing.assert_array_equal(ops.convert_to_numpy(m(x)), ops.convert_to_numpy(m.with_context_length(m.context_length)(x)))

    def test_extend_then_shrink_round_trips(self):
        for arch in EXTENDABLE:
            with self.subTest(arch=arch):
                m = _build(arch)
                back = m.with_context_length(2 * m.context_length).with_context_length(m.context_length)
                x = _ids(m.context_length)
                np.testing.assert_allclose(ops.convert_to_numpy(m(x)), ops.convert_to_numpy(back(x)), rtol=1e-6, atol=1e-6)

    def test_new_distances_start_at_zero(self):
        for arch in EXTENDABLE:
            with self.subTest(arch=arch):
                m = _build(arch)
                T = m.context_length
                long = m.with_context_length(2 * T)
                w = ops.convert_to_numpy(long.distance_weights)
                np.testing.assert_array_equal(w[:T], 0.0)  # distances T+1..2T
                np.testing.assert_array_equal(w[T:], ops.convert_to_numpy(m.distance_weights))

    def test_per_dim_kernels_resize_along_distance(self):
        m = _build("rope-logit-sum", per_dim_kernels=True)
        T = m.context_length
        long = m.with_context_length(2 * T)
        w = ops.convert_to_numpy(long.distance_weights)
        self.assertEqual(w.shape, (2 * T, 16))
        np.testing.assert_array_equal(w[:T], 0.0)
        np.testing.assert_array_equal(w[T:], ops.convert_to_numpy(m.distance_weights))

    def test_basis_keeps_training_window(self):
        for arch in ("rope-log-gaussian-sum", "rope-kernel-attention"):
            for basis in ("log-gaussian", "damped-oscillator"):
                with self.subTest(arch=arch, basis=basis):
                    m = _build(arch, kernel_basis=basis)
                    T = m.context_length
                    long = m.with_context_length(3 * T)
                    self.assertEqual(long.basis_window, T)
                    np.testing.assert_allclose(long._basis[-T:], m._basis, rtol=1e-6, atol=1e-7)

    def test_window_tied_arch_refuses(self):
        m = build_model({"arch": "dual-sphere", "arch_params": tiny_arch_params("dual-sphere")})
        with self.assertRaises(NotImplementedError):
            m.with_context_length(2 * m.context_length)


class TestInitFrom(unittest.TestCase):
    def test_finetune_run_starts_from_source_weights_at_new_window(self):
        with temp_workspace():
            src = tiny_config(run_name="src", arch="rope-kernel-attention")
            src["arch_params"].update(dim=16, mlp_hidden=8, query_hidden=8, num_heads=2)
            cfg_mod.save("src", src)
            source, _ = build_or_load("src", src, verbose=0)
            source.save(cfg_mod.run_dir("src") / "model.keras")

            ft = {**src, "run_name": "ft", "init_from": {"run": "src", "checkpoint": "latest"}}
            ft["arch_params"] = {**src["arch_params"], "window_size": 2 * src["arch_params"]["window_size"],
                                 "basis_window": src["arch_params"]["window_size"]}
            cfg_mod.save("ft", ft)
            model, initial_epoch = build_or_load("ft", ft, verbose=0)
            self.assertEqual(initial_epoch, 0)
            self.assertEqual(model.context_length, 2 * source.context_length)
            np.testing.assert_array_equal(ops.convert_to_numpy(model.readout.kernel), ops.convert_to_numpy(source.readout.kernel))

    def test_architecture_mismatch_is_rejected(self):
        with temp_workspace():
            src = tiny_config(run_name="src", arch="rope-log-gaussian-sum")
            src["arch_params"].update(dim=16, mlp_hidden=8)
            cfg_mod.save("src", src)
            source, _ = build_or_load("src", src, verbose=0)
            source.save(cfg_mod.run_dir("src") / "model.keras")

            ft = {**src, "run_name": "ft", "init_from": {"run": "src"}}
            ft["arch_params"] = {**src["arch_params"], "kernel_basis": "damped-oscillator"}
            cfg_mod.save("ft", ft)
            with self.assertRaises(ValueError):
                build_or_load("ft", ft, verbose=0)


if __name__ == "__main__":
    unittest.main()
