"""Analysis reports: `model report <name> --run-name X` writes an HTML report under `runs/X/reports/`.

A report is a module of this package whose name does not start with `_`. It defines:

    ARCHS = ("arch-name", ...)   the architectures it works on (names from `model.archs.ARCHS`, or a family below);
                                 ("any",): every architecture; ("no-run",): it takes no run (cross-run, dataset)
    build_parser() -> argparse.ArgumentParser
    main(argv) -> int

Its CLI name is the module name with dashes (`query_recall_trace.py` -> `query-recall-trace`). There is nothing to
register: the package is scanned, and `ARCHS` and the docstring are read without importing the module (listing is
instant; only the report that runs is imported).

    model report                         every report and the architectures it belongs to
    model report --run-name X            the reports for X's architecture
    model report <name> --run-name X     run one (refused if X's architecture is not in its ARCHS)

Modules starting with `_` are shared helpers (`_report.py`: HTML rendering; `_model_io.py`: loading).
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ANY = "any"
NO_RUN = "no-run"

# architecture families a report can name instead of listing every member
FAMILIES: Dict[str, Tuple[str, ...]] = {
    # an input table (`input_embeddings`, the context side) and a separate readout table (`target_embeddings`)
    "dual-tables": (
        "dual-sphere", "dual-sphere-dual-readout", "dual-sphere-mid-gate", "dual-sphere-hyper-token-transform",
        "dual-sphere-hyper-gate", "dual-sphere-householder", "dual-sphere-householder-per-token",
        "dual-sphere-householder-rank8", "dual-sphere-linear-recurrence", "sine-sphere", "alibi-sphere",
        "alibi-sphere-two-block", "alibi-retriever-sphere", "alibi-token-retriever-sphere",
        "alibi-token-retriever-sphere-bidirectional", "alibi-token-retriever-sphere-bidirectional-raw",
    ),
}

PACKAGE = Path(__file__).resolve().parent


class Report:
    """One report's metadata, read from its source without importing it."""

    def __init__(self, path: Path):
        tree = ast.parse(path.read_text())
        self.module = f"model.reports.{path.stem}"
        self.name = path.stem.replace("_", "-")
        doc = ast.get_docstring(tree) or ""
        summary = doc.split("\n", 1)[0]
        self.summary = summary.split(":", 1)[1].strip() if summary.startswith("`model report") else summary
        archs = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ARCHS" for t in node.targets):
                archs = tuple(ast.literal_eval(node.value))
        if not archs:
            raise ValueError(f"{path}: a report must define ARCHS as a literal tuple")
        self.archs = archs

    @property
    def needs_run(self) -> bool:
        return NO_RUN not in self.archs

    def supports(self, arch: str) -> bool:
        return ANY in self.archs or arch in self.arch_names()

    def arch_names(self) -> List[str]:
        """ARCHS with the families expanded."""
        out: List[str] = []
        for a in self.archs:
            out += FAMILIES.get(a, (a,))
        return out

    def load(self):
        import importlib

        return importlib.import_module(self.module)


def reports() -> Dict[str, Report]:
    return {r.name: r for r in (Report(p) for p in sorted(PACKAGE.glob("[!_]*.py")))}


def for_arch(arch: str) -> List[Report]:
    return [r for r in reports().values() if r.needs_run and r.supports(arch)]


def describe(r: Report, width: int = 30) -> str:
    where = "any architecture" if ANY in r.archs else "no run" if NO_RUN in r.archs else ", ".join(r.archs)
    return f"  {r.name:<{width}} {r.summary}\n  {'':<{width}} [{where}]"


def run_name_of(argv: List[str]) -> Optional[str]:
    """The `--run-name` value in a report's arguments, if any."""
    for i, a in enumerate(argv):
        if a == "--run-name" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--run-name="):
            return a.split("=", 1)[1]
    return None
