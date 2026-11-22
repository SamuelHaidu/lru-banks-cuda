"""`model/reports`: every report declares valid architectures and the CLI shape, and `model report` lists them."""

import ast
import contextlib
import io
import unittest

import tests.helpers  # noqa: F401  (pins the device before keras/torch use)

from model import reports as rep
from model.archs import ARCHS


class TestReportMetadata(unittest.TestCase):
    def test_archs_name_real_architectures(self):
        for name, r in rep.reports().items():
            with self.subTest(report=name):
                for a in r.archs:
                    self.assertIn(a, set(ARCHS) | set(rep.FAMILIES) | {rep.ANY, rep.NO_RUN}, f"{name}: unknown {a!r}")

    def test_families_name_real_architectures(self):
        for fam, members in rep.FAMILIES.items():
            for a in members:
                self.assertIn(a, ARCHS, f"family {fam}: unknown {a!r}")

    def test_every_report_has_the_cli_shape(self):
        for name, r in rep.reports().items():
            with self.subTest(report=name):
                tree = ast.parse((rep.PACKAGE / f"{r.module.rsplit('.', 1)[1]}.py").read_text())
                funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
                self.assertLessEqual({"build_parser", "main"}, funcs)
                self.assertTrue(r.summary, f"{name}: no docstring summary")

    def test_reports_attach_to_their_architectures(self):
        retriever = {r.name for r in rep.for_arch("alibi-retriever-sphere")}
        self.assertIn("query-recall-trace", retriever)  # its own
        self.assertIn("pga-dual-embeddings", retriever)  # through the dual-tables family
        self.assertIn("order-sensitivity-loss", retriever)  # any architecture
        self.assertNotIn("wave-field-stepper", retriever)
        self.assertNotIn("compare-runs", retriever)  # takes no run
        banks = {r.name for r in rep.for_arch("lru-banks")}
        self.assertEqual(banks, {r.name for r in rep.reports().values() if rep.ANY in r.archs})

    def test_run_name_is_found_in_forwarded_arguments(self):
        self.assertEqual(rep.run_name_of(["--n", "3", "--run-name", "x"]), "x")
        self.assertEqual(rep.run_name_of(["--run-name=y"]), "y")
        self.assertIsNone(rep.run_name_of(["--n", "3"]))


class TestReportCLI(unittest.TestCase):
    def test_lists_every_report(self):
        from model.cli import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["report"]), 0)
        for name in rep.reports():
            self.assertIn(name, out.getvalue())

    def test_unknown_report_is_refused(self):
        from model.cli import main

        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["report", "no-such-report"]), 1)


if __name__ == "__main__":
    unittest.main()
