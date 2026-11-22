"""Spherical language model research package.

Layout (one responsibility per file):

    device.py      backend/device selection (torch + Intel XPU by default)
    config.py      run config + dataset metadata -- one home per value
    interfaces.py  the `BaseLM` contract every architecture implements
    archs/         one file per architecture, plus the `ARCHS` registry
    data.py        the dataset (variable-prefix, document-based)
    losses.py      the cross-entropy loss (XPU-safe)
    train.py       training and resume
    evaluate.py    full-validation scoring
    generation.py  autoregressive sampling
    cli.py         the command-line interface

Importing anything under `model` pins `KERAS_BACKEND=torch` first, so a
downstream `import keras` can never fall back to the machine-global default
in `~/.keras/keras.json`. Device placement (xpu/cuda/cpu) is separate and
happens later, once a run's config has been read -- see `model.device`.
"""

from model.device import describe_device, setup_backend, setup_device

setup_backend()

__all__ = ["setup_backend", "setup_device", "describe_device"]
