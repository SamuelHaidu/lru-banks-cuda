"""The architecture contract, applied to every entry in `ARCHS`.

This is the file that replaces per-architecture test duplication. Register a
new architecture and it is covered here immediately -- there is no way to add
one that silently fails to satisfy what training, evaluation and generation
assume about it.
"""

import os
import tempfile
import unittest

from tests.helpers import VOCAB, tiny_arch_params

import keras
import numpy as np
from keras import ops

from model.archs import ARCHS, arch_defaults, build_model
from model.interfaces import BaseLM

REQUIRED_ABSTRACT = ("context_length", "vocabulary_size", "call", "get_config")


def _build(arch: str) -> BaseLM:
    return build_model({"arch": arch, "arch_params": tiny_arch_params(arch)})


class TestArchRegistry(unittest.TestCase):
    def test_registry_is_not_empty(self):
        self.assertGreater(len(ARCHS), 0)

    def test_every_arch_subclasses_baselm(self):
        for name, cls in ARCHS.items():
            with self.subTest(arch=name):
                self.assertTrue(issubclass(cls, BaseLM), f"{name} must subclass BaseLM")

    def test_every_arch_overrides_the_whole_contract(self):
        for name, cls in ARCHS.items():
            for member in REQUIRED_ABSTRACT:
                with self.subTest(arch=name, member=member):
                    self.assertIsNot(
                        getattr(cls, member),
                        getattr(BaseLM, member),
                        f"{name} does not override BaseLM.{member}",
                    )

    def test_arch_defaults_are_constructible(self):
        for name in ARCHS:
            with self.subTest(arch=name):
                defaults = arch_defaults(name)
                self.assertIn("vocabulary_size", defaults)

    def test_build_model_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            build_model({"arch": "does-not-exist", "arch_params": {}})


class TestArchContract(unittest.TestCase):
    """Behavioural contract that train/evaluate/generate rely on."""

    def test_forward_shape_and_finiteness(self):
        for name in ARCHS:
            with self.subTest(arch=name):
                m = _build(name)
                x = np.random.randint(0, m.vocabulary_size, (3, m.context_length)).astype("int32")
                y = ops.convert_to_numpy(m(x))
                self.assertEqual(y.shape, (3, m.vocabulary_size))
                self.assertTrue(np.all(np.isfinite(y)), "logits must be finite")

    def test_context_length_and_vocabulary_size_are_positive_ints(self):
        for name in ARCHS:
            with self.subTest(arch=name):
                m = _build(name)
                self.assertIsInstance(m.context_length, int)
                self.assertIsInstance(m.vocabulary_size, int)
                self.assertGreater(m.context_length, 0)
                self.assertEqual(m.vocabulary_size, VOCAB)

    def test_save_load_round_trip_is_bit_exact(self):
        for name in ARCHS:
            with self.subTest(arch=name):
                m = _build(name)
                x = np.random.randint(0, m.vocabulary_size, (3, m.context_length)).astype("int32")
                before = ops.convert_to_numpy(m(x))
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "m.keras")
                    m.save(path)
                    reloaded = keras.saving.load_model(path)
                # the weights come back bit for bit
                for a, b in zip(m.weights, reloaded.weights):
                    np.testing.assert_array_equal(ops.convert_to_numpy(a.value), ops.convert_to_numpy(b.value),
                                                  err_msg=a.path)
                # outputs to within a few ulps: this CPU torch build's sgemm is not bitwise reproducible across
                # calls (the result's last bit depends on the buffers' memory alignment; the same model called
                # twice differs by ~1e-7), so bit-exact outputs would test the allocator, not the save/load
                after = ops.convert_to_numpy(reloaded(x))
                np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)

    def test_get_config_round_trips_through_constructor(self):
        for name in ARCHS:
            with self.subTest(arch=name):
                m = _build(name)
                config = m.get_config()
                self.assertIsInstance(config, dict)
                clone = type(m).from_config(config)
                self.assertEqual(clone.context_length, m.context_length)
                self.assertEqual(clone.vocabulary_size, m.vocabulary_size)

    def test_parameter_breakdown_is_consistent(self):
        for name in ARCHS:
            with self.subTest(arch=name):
                m = _build(name)
                p = m.parameter_breakdown()
                self.assertEqual(p["total"], p["embeddings"] + p["mechanism"])
                self.assertEqual(p["total"], int(m.count_params()))
                self.assertGreater(p["embeddings"], 0)
                self.assertGreater(p["mechanism"], 0)

    def test_generate_is_available_and_lengths_are_exact(self):
        for name in ARCHS:
            with self.subTest(arch=name):
                m = _build(name)
                out = m.generate([1, 2, 3], max_new_tokens=5)
                self.assertEqual(len(out.split()), 3 + 5)

    def test_gradients_reach_every_trainable_variable(self):
        import torch

        for name in ARCHS:
            with self.subTest(arch=name):
                m = _build(name)
                x = ops.convert_to_tensor(
                    np.random.randint(0, m.vocabulary_size, (4, m.context_length)).astype("int32")
                )
                loss = ops.mean(m(x, training=True) ** 2)
                grads = torch.autograd.grad(loss, [v.value for v in m.trainable_variables], allow_unused=True)
                for var, grad in zip(m.trainable_variables, grads):
                    self.assertIsNotNone(grad, f"{name}: no gradient for {var.name}")
                    self.assertTrue(bool(torch.isfinite(grad).all()), f"{name}: non-finite grad for {var.name}")


if __name__ == "__main__":
    unittest.main()
