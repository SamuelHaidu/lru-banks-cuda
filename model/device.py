"""Backend and device selection.

There is exactly one backend (PyTorch) and one default device (Intel XPU).
The two are separated because they have different timing constraints:

  * **Backend** must be fixed *before* `import keras` -- Keras reads
    `KERAS_BACKEND` once, at import, and never re-reads it. If it is unset,
    Keras falls back to `~/.keras/keras.json`, which on this machine says
    "tensorflow", and the import fails outright. `setup_backend()` therefore
    runs from `model/__init__.py`, so merely importing anything under `model`
    is enough. It touches only `os.environ` -- no keras, no torch, no cost.

  * **Device** can be chosen at any time, because switching between
    `xpu`/`cuda`/`cpu` is just a torch placement attribute, not a different
    backend. That is what lets the CLI read a run's `config.json` and pin its
    device *after* the package is already imported.

Historically this project ran XPU workloads through six exported variables
(`PYTHONPATH`, `LD_LIBRARY_PATH`, `OverrideDefaultFP64Settings`,
`IGC_EnableDPEmulation`, `SYCL_CACHE_*`, `KERAS_BACKEND`). Those existed only
to bridge two split virtualenvs; with a single environment holding
`torch==2.12.1+xpu`, the XPU is available with nothing exported at all.
"""

import os
import warnings
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# Persistent SPIR-V kernel cache. Never cleared: a cold cache costs a full
# Level-Zero recompile (tens of seconds) on the first training step.
SYCL_CACHE_DIR = ROOT / ".sycl_cache"

BACKEND = "torch"

_resolved_device: Optional[str] = None


def quiet_tensorflow_import() -> None:
    """Imports TensorFlow (pulled in by keras_hub) with the C-level stderr muted.

    TF 2.21 prints absl "log messages before absl::InitializeLog()" and cudart "Could not
    find cuda drivers" lines from C++ at import, before any env var (TF_CPP_MIN_LOG_LEVEL)
    is consulted, so the only way to hide them is to mute file descriptor 2 for the import.
    If the import fails, stderr is restored first so the traceback is visible.
    """
    import sys

    if "tensorflow" in sys.modules:
        return
    try:
        import importlib.util

        if importlib.util.find_spec("tensorflow") is None:
            return
    except (ImportError, ValueError):
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        import tensorflow  # noqa: F401
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def setup_backend() -> None:
    """Pins the Keras backend and Intel runtime settings. Import-time cheap.

    Must run before `import keras`. Called from `model/__init__.py`, so every
    module in this package inherits it just by being imported.
    """
    os.environ["KERAS_BACKEND"] = BACKEND
    SYCL_CACHE_DIR.mkdir(exist_ok=True)
    os.environ.setdefault("OverrideDefaultFP64Settings", "1")
    os.environ.setdefault("IGC_EnableDPEmulation", "1")
    os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")
    os.environ.setdefault("SYCL_CACHE_DIR", str(SYCL_CACHE_DIR))

    # Gen12LP (Iris Xe) sits below PyTorch's officially supported XPU floor and
    # warns on every `torch.xpu` call. Silenced by message, not globally, so
    # genuine warnings still surface.
    warnings.filterwarnings("ignore", message=".*not officially supported by PyTorch XPU.*")


def setup_device(device: Optional[str] = None) -> str:
    """Places computation on `xpu` (default), `cuda` or `cpu`.

    Safe to call repeatedly and safe to call with a different device than
    before -- switching is just a torch placement change, so the run's config
    can override the default after import.

    Args:
        device: Target device. When `None`, reads `KERAS_DEVICE` from the
            environment and falls back to `"xpu"`.

    Returns:
        The device actually in use. Falls back to `"cpu"` with a printed
        warning if the requested accelerator is unavailable, so the test suite
        runs on any machine.
    """
    global _resolved_device

    setup_backend()
    requested = (device or os.environ.get("KERAS_DEVICE") or "xpu").lower()

    import torch

    resolved = "cpu"
    if requested == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            resolved = "xpu"
        elif _resolved_device is None:
            print("Warning: XPU requested but torch.xpu is unavailable -- falling back to CPU.")
    elif requested == "cuda":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif _resolved_device is None:
            print("Warning: CUDA requested but unavailable -- falling back to CPU.")

    from keras.src.backend.torch import core

    core.global_state.set_global_attribute("torch_device", resolved)
    _resolved_device = resolved
    return resolved


def describe_device() -> str:
    """Human-readable one-liner for the resolved device (for logs).

    Resolves the device first if nothing has, so it can never report "cpu"
    merely because it was called before any computation happened.
    """
    dev = _resolved_device or setup_device()
    try:
        import torch

        if dev == "xpu":
            return f"xpu:0 ({torch.xpu.get_device_name(0)})"
        if dev == "cuda":
            return f"cuda:0 ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    return dev
