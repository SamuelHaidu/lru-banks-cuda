"""Config invariants.

The whole point of this module is that a value has exactly one home. These
tests pin down the two properties that enforce it: unknown keys are rejected
(so a typo cannot silently create a second, ignored home), and dataset facts
resolve from `metadata.json` rather than the run config.
"""

import json
import unittest

from tests.helpers import VOCAB, temp_workspace, tiny_config

from model import config as cfg_mod


class TestSaveLoad(unittest.TestCase):
    def test_round_trip(self):
        with temp_workspace():
            cfg = tiny_config("r1")
            path = cfg_mod.save("r1", cfg)
            self.assertTrue(path.exists())
            self.assertEqual(cfg_mod.load("r1"), cfg)

    def test_load_missing_run_raises_with_guidance(self):
        with temp_workspace():
            with self.assertRaises(cfg_mod.ConfigError) as ctx:
                cfg_mod.load("nope")
            self.assertIn("model create", str(ctx.exception))

    def test_default_config_is_a_valid_config(self):
        cfg = json.loads(json.dumps(cfg_mod.DEFAULT_CONFIG))
        cfg_mod.validate(cfg)  # must not raise


class TestValidation(unittest.TestCase):
    def test_unknown_top_level_key_is_rejected(self):
        cfg = tiny_config()
        cfg["bogus"] = 1
        with self.assertRaises(cfg_mod.ConfigError):
            cfg_mod.validate(cfg)

    def test_unknown_section_key_is_rejected(self):
        for section, key in (("training", "lr"), ("data", "corrupt")):
            with self.subTest(section=section):
                cfg = tiny_config()
                cfg[section][key] = 1
                with self.assertRaises(cfg_mod.ConfigError):
                    cfg_mod.validate(cfg)

    def test_missing_required_key_is_rejected(self):
        cfg = tiny_config()
        del cfg["arch"]
        with self.assertRaises(cfg_mod.ConfigError):
            cfg_mod.validate(cfg)

    def test_saving_an_invalid_config_raises(self):
        with temp_workspace():
            cfg = tiny_config("r1")
            cfg["training"]["lr"] = 1
            with self.assertRaises(cfg_mod.ConfigError):
                cfg_mod.save("r1", cfg)


class TestDottedAccess(unittest.TestCase):
    def test_get_nested(self):
        cfg = tiny_config()
        self.assertEqual(cfg_mod.get(cfg, "training.epochs"), cfg["training"]["epochs"])
        self.assertEqual(cfg_mod.get(cfg, "arch"), "dual-sphere")

    def test_get_unknown_raises(self):
        with self.assertRaises(cfg_mod.ConfigError):
            cfg_mod.get(tiny_config(), "training.nope")

    def test_jit_compile_defaults_off(self):
        """Off by default: the compiler backend doesn't work on this
        project's Intel Iris Xe XPU, so it must be an explicit opt-in."""
        self.assertEqual(cfg_mod.get(tiny_config(), "training.jit_compile"), False)

    def test_set_parses_json_types(self):
        cases = [
            ("training.epochs", "7", 7),
            ("training.learning_rate", "0.0005", 0.0005),
            ("training.jit_compile", "true", True),
            ("arch_params.learnable_tau", "false", False),
            ("data.corrupt_prob", "0.25", 0.25),
            ("arch", "dual-sphere", "dual-sphere"),  # bare string, not valid JSON
        ]
        for key, raw, expected in cases:
            with self.subTest(key=key):
                cfg = tiny_config()
                self.assertEqual(cfg_mod.set_value(cfg, key, raw), expected)
                self.assertEqual(cfg_mod.get(cfg, key), expected)

    def test_set_rejects_unknown_key(self):
        """A typo must fail loudly rather than be silently ignored."""
        for key in ("training.lr", "data.corrupt", "nope"):
            with self.subTest(key=key):
                with self.assertRaises(cfg_mod.ConfigError):
                    cfg_mod.set_value(tiny_config(), key, "1")

    def test_set_allows_new_arch_params(self):
        """`arch_params` is arch-defined, so new keys there are legitimate."""
        cfg = tiny_config()
        cfg_mod.set_value(cfg, "arch_params.brand_new", "3")
        self.assertEqual(cfg["arch_params"]["brand_new"], 3)


class TestDatasetMetadata(unittest.TestCase):
    def test_dataset_facts_resolve_from_metadata(self):
        with temp_workspace(avg_chars_per_token=4.127):
            meta = cfg_mod.load_dataset_metadata("tiny")
            self.assertEqual(meta["vocabulary_size"], VOCAB)
            self.assertEqual(meta["avg_chars_per_token"], 4.127)

    def test_split_paths_resolve_through_metadata(self):
        with temp_workspace():
            self.assertTrue(cfg_mod.dataset_split_path("tiny", "train").exists())
            self.assertTrue(cfg_mod.dataset_split_path("tiny", "validation").exists())

    def test_unknown_split_raises(self):
        with temp_workspace():
            with self.assertRaises(cfg_mod.ConfigError):
                cfg_mod.dataset_split_path("tiny", "test")

    def test_tokenizer_path_follows_the_dataset(self):
        with temp_workspace():
            path = cfg_mod.tokenizer_path("tiny")
            self.assertEqual(path.name, "tokenizer.json")
            self.assertEqual(path.parent.name, "tiny")


if __name__ == "__main__":
    unittest.main()
