"""Properties specific to `DualSphereLM`.

The generic contract lives in `test_interface.py`; this file covers what is
peculiar to this architecture -- the spherical invariants, the `tau` bound,
the absence of bias parameters, and the padding behaviour that previously
produced NaN gradients when it was implemented as a mask.
"""

import unittest

from tests.helpers import VOCAB, tiny_arch_params

import numpy as np
from keras import ops

from model.archs.dual_sphere import DualSphereLM


def _model(**overrides) -> DualSphereLM:
    params = tiny_arch_params("dual-sphere")
    params.update(overrides)
    return DualSphereLM(**params)


class TestSphericalInvariants(unittest.TestCase):
    def test_both_tables_are_exactly_unit_norm_after_projection(self):
        m = _model()
        for table in (m.normalized_input_embeddings(), m.normalized_target_embeddings()):
            norms = np.linalg.norm(ops.convert_to_numpy(table), axis=-1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_logits_are_bounded_by_tau(self):
        """Logits are `cos(theta) * tau`, so `|logit| <= tau` always."""
        tau = 4.0
        m = _model(initial_tau=tau)
        x = np.random.randint(0, VOCAB, (8, m.context_length)).astype("int32")
        logits = ops.convert_to_numpy(m(x))
        self.assertLessEqual(float(np.abs(logits).max()), tau + 1e-4)

    def test_query_is_on_the_unit_sphere(self):
        """A logit equals cos(angle) * tau, so max logit / tau <= 1 exactly."""
        m = _model(initial_tau=1.0)
        x = np.random.randint(0, VOCAB, (8, m.context_length)).astype("int32")
        logits = ops.convert_to_numpy(m(x))
        self.assertLessEqual(float(logits.max()), 1.0 + 1e-5)
        self.assertGreaterEqual(float(logits.min()), -1.0 - 1e-5)


class TestParameterization(unittest.TestCase):
    def test_no_bias_parameters_anywhere(self):
        m = _model()
        for var in m.weights:
            self.assertNotIn("bias", var.name.lower(), f"unexpected bias parameter: {var.name}")

    def test_parameter_count_matches_the_analytic_formula(self):
        v, di, do, w, h = 200, 8, 16, 32, 64
        m = DualSphereLM(vocabulary_size=v, dim_in=di, dim_out=do, window_size=w, hidden_dim=h)
        expected = (
            v * di          # input table
            + v * do        # target table
            + 1             # tau
            + (w * di) * h  # dense1
            + h * (h // 2)  # dense2
            + (h // 2) * do  # dense3
        )
        self.assertEqual(m.parameter_breakdown()["total"], expected)
        self.assertEqual(m.parameter_breakdown()["embeddings"], v * di + v * do)

    def test_learnable_tau_toggles_trainability(self):
        for learnable in (True, False):
            with self.subTest(learnable=learnable):
                m = _model(learnable_tau=learnable)
                names = [v.name for v in m.trainable_variables]
                self.assertEqual("tau" in names, learnable)

    def test_context_length_equals_window_size(self):
        m = _model(window_size=11)
        self.assertEqual(m.context_length, 11)
    def test_activation_selection(self):
        for act in ("gelu", "tanh"):
            with self.subTest(activation=act):
                m = _model(activation=act)
                self.assertEqual(m.activation, act)
                self.assertEqual(m.get_config()["activation"], act)

    def test_invalid_arguments_are_rejected(self):
        for bad in (
            {"vocabulary_size": 0},
            {"dim_in": 0},
            {"dim_out": -1},
            {"window_size": 0},
            {"hidden_dim": 1},
            {"activation": "relu"},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _model(**bad)

class TestPaddingBehaviour(unittest.TestCase):
    def test_all_padding_window_produces_finite_logits_and_gradients(self):
        """An all-zero window must not produce NaNs.

        Padding ids are embedded like any other token rather than masked to
        zero. Masking would send a bias-free FFN's output to exactly zero and
        make `ops.normalize` differentiate 0/0, yielding real NaN gradients.
        """
        import torch

        m = _model()
        x = ops.convert_to_tensor(np.zeros((2, m.context_length), dtype="int32"))
        logits = m(x, training=True)
        self.assertTrue(bool(torch.isfinite(logits).all()))

        loss = ops.mean(logits**2)
        grads = torch.autograd.grad(loss, [v.value for v in m.trainable_variables], allow_unused=True)
        for var, grad in zip(m.trainable_variables, grads):
            if grad is not None:
                self.assertTrue(bool(torch.isfinite(grad).all()), f"non-finite grad for {var.name}")

    def test_padding_changes_the_prediction(self):
        """Padding is a learned sentinel, not a no-op."""
        m = _model()
        rng = np.random.default_rng(0)
        full = rng.integers(1, VOCAB, (1, m.context_length)).astype("int32")
        padded = full.copy()
        padded[0, : m.context_length // 2] = 0
        self.assertFalse(
            np.allclose(ops.convert_to_numpy(m(full)), ops.convert_to_numpy(m(padded))),
            "left padding must affect the output",
        )


if __name__ == "__main__":
    unittest.main()
