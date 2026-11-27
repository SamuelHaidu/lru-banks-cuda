"""The runs the page can load, and the loaded (run, checkpoint): one at a time (used by the worker; the server only
lists runs)."""

from __future__ import annotations

import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "datasets") not in sys.path:
    sys.path.insert(0, str(ROOT / "datasets"))


def lbc_checkpoints(run: Path) -> List[str]:
    """An lbc (C++ engine) run's checkpoints: latest (model.lbc), best, then the numbered ones, newest first."""
    ckpts = ["latest"] if (run / "model.lbc").exists() else []
    if (run / "checkpoints" / "best.lbc").exists():
        ckpts.append("best")
    epochs = sorted((int(m.group(1)) for p in (run / "checkpoints").glob("checkpoint_*.lbc")
                     if (m := re.search(r"checkpoint_(\d+)", p.name))), reverse=True)
    return ckpts + [str(e) for e in epochs]


def lbc_path(run: Path, checkpoint: str) -> Path:
    if checkpoint == "latest":
        return run / "model.lbc"
    if checkpoint == "best":
        return run / "checkpoints" / "best.lbc"
    return run / "checkpoints" / f"checkpoint_{int(checkpoint):05d}.lbc"


def lbc_sidecar(run: Path) -> Dict[str, Any]:
    """runs/RUN/webchat.json: what the lbc config does not say (it names the pod's dataset paths): {"tokenizer":
    directory under tokenizers/, optional "dataset" (a local dataset for the page's catalogs), "mode": chat |
    completion}."""
    import json

    p = run / "webchat.json"
    return json.loads(p.read_text()) if p.exists() else {}


class Models:
    """The loaded (run, checkpoint): one at a time, one generation at a time."""

    def __init__(self, device: Optional[str]):
        self.device = device
        self.key: Optional[Tuple[str, str]] = None
        self.model = self.tokenizer = None
        self.lock = threading.Lock()

    def runs(self) -> List[Dict[str, Any]]:
        from model import config as cfg_mod

        out: List[Dict[str, Any]] = []
        if not (ROOT / "runs").is_dir():  # a fresh clone: no runs yet
            return out
        for run in sorted(p for p in (ROOT / "runs").iterdir() if (p / "config.json").exists()):
            if lbc_checkpoints(run):
                side = lbc_sidecar(run)
                out.append({"run": run.name, "arch": "lru-banks (lbc)", "dataset": side.get("dataset", ""),
                            "checkpoints": lbc_checkpoints(run), "mode": side.get("mode", "chat"), "inspect": True})
                continue
            ckpts = ["latest"] if (run / "model.keras").exists() else []
            if (run / "checkpoints" / "best.keras").exists():
                ckpts.append("best")
            epochs = sorted((int(m.group(1)) for p in (run / "checkpoints").glob("checkpoint_*.keras")
                             if (m := re.search(r"checkpoint_(\d+)", p.name))), reverse=True)
            ckpts += [str(e) for e in epochs]
            if not ckpts:
                continue
            try:
                cfg = cfg_mod.load(run.name)
            except Exception:  # noqa: BLE001 (a run with an old or broken config: skip it)
                continue
            try:  # chat: the dataset scores assistant spans or says it is chat-format ("chat": the Calc-X mix, which
                # scores every token); completion: plain text (pretraining)
                meta = cfg_mod.load_dataset_metadata(cfg["data"]["dataset"])
                chat = bool(meta.get("loss_spans") or meta.get("chat"))
            except Exception:  # noqa: BLE001 (the dataset is not here)
                chat = True
            out.append({"run": run.name, "arch": cfg["arch"], "dataset": cfg["data"]["dataset"], "checkpoints": ckpts,
                        "mode": "chat" if chat else "completion",
                        "inspect": cfg["arch"].startswith("min-gru") or cfg["arch"] in ("gated-deltanet", "lru-banks")})
        return out

    def load(self, run: str, checkpoint: str):
        from model import config as cfg_mod

        with self.lock:
            if self.key != (run, checkpoint) and lbc_checkpoints(ROOT / "runs" / run):
                from tokenizers import Tokenizer

                from model.lbc_torch import LbcModel

                tok = lbc_sidecar(ROOT / "runs" / run).get("tokenizer")
                if not tok:
                    raise ValueError(f"runs/{run}/webchat.json needs a \"tokenizer\" (a directory under tokenizers/)")
                self.model = None  # free the previous weights first
                self.model = LbcModel(lbc_path(ROOT / "runs" / run, checkpoint), self.device or "xpu")
                self.tokenizer = Tokenizer.from_file(str(ROOT / "tokenizers" / tok / "tokenizer.json"))
                self.key = (run, checkpoint)
            if self.key != (run, checkpoint):
                if self.device is None:
                    import torch

                    self.device = cfg_mod.load(run).get("device") or "xpu"
                    if self.device == "cuda" and not torch.cuda.is_available():  # a run trained on a pod: here the XPU
                        self.device = "xpu"
                    from model.device import setup_device

                    setup_device(self.device)
                from model.evaluate import load_checkpoint
                from tokenizers import Tokenizer

                ckpt: Any = int(checkpoint) if checkpoint.isdigit() else checkpoint
                self.model = load_checkpoint(run, ckpt)
                self.tokenizer = Tokenizer.from_file(str(cfg_mod.tokenizer_path(cfg_mod.load(run)["data"]["dataset"])))
                self.key = (run, checkpoint)
        return self.model, self.tokenizer
