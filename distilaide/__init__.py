"""Distilaide: image restoration across training backends, driven by one config.

**The core is backend-free.** ``pip install distilaide`` gives you the CLI and the config surface;
a training framework arrives with the backend you select::

    pip install 'distilaide[lightning]'      # PyTorch Lightning — the reference backend

Which is why every backend name below resolves LAZILY (PEP 562): importing ``distilaide`` must not
require a framework the operator did not choose. Touching a name whose backend is missing raises an
:class:`ImportError` naming the exact extra, instead of a traceback from three libraries down.
"""

import importlib
from typing import Any

__version__ = "0.2.0"

#: Backend-owned name -> (defining module, the extra that installs it). Resolved on first attribute
#: access. Keep in step with the extras in ``pyproject.toml``: a name here whose extra does not
#: exist would advertise an install command that fails.
#
# `DeepInvModel` is the ZOO WRAPPER — one class over deepinv's whole restoration set, learned and
# classical alike. Its extra is the SMALLEST one that supplies what it needs — plain `torch`, not
# `lightning` — so the ImportError advises the smaller install; the same reasoning makes
# `TorchRestorer` a `torch` name while only `LightningRestorer` asks for `lightning`.
_BACKEND_NAMES = {
    "LightningRestorer": ("distilaide.lightning_restorer", "lightning"),
    "TorchRestorer": ("distilaide.torch_restorer", "torch"),
    "FastaiRestorer": ("distilaide.fastai_restorer", "fastai"),
    "KerasRestorer": ("distilaide.keras_restorer", "keras"),
    "TensorFlowRestorer": ("distilaide.tensorflow_restorer", "tensorflow"),
    "JaxRestorer": ("distilaide.jax_restorer", "jax"),
    "MlxRestorer": ("distilaide.mlx_restorer", "mlx"),
    "HuggingFaceRestorer": ("distilaide.huggingface_restorer", "huggingface"),
    "DeepInvModel": ("distilaide.deepinv_model", "torch"),
    "HuggingFaceRestorationModel": ("distilaide.huggingface_model", "huggingface"),
    # The shared spec's implementations. `TorchDnCNN` serves EVERY torch backend (lightning,
    # torch, fastai — they differ in the loop, not in what a model is), which is why there is no
    # name per backend. Each names the SMALLEST extra that supplies it.
    "TorchDnCNN": ("distilaide.dncnn.torch_dncnn", "torch"),
    "KerasDnCNN": ("distilaide.dncnn.keras_dncnn", "keras"),
    "TensorFlowDnCNN": ("distilaide.dncnn.tensorflow_dncnn", "tensorflow"),
    "FlaxDnCNN": ("distilaide.dncnn.flax_dncnn", "jax"),
    "MlxDnCNN": ("distilaide.dncnn.mlx_dncnn", "mlx"),
}

__all__ = [*_BACKEND_NAMES]


def __getattr__(name: str) -> Any:
    """Resolve a backend-owned name on first use, or explain which extra is missing."""
    entry = _BACKEND_NAMES.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, extra = entry
    try:
        return getattr(importlib.import_module(module_path), name)
    except ImportError as exc:
        # The failure is a MISSING BACKEND, not a broken distilaide — say so, and say what to run.
        # Chained (`from exc`) so the underlying "No module named 'torch'" stays visible.
        raise ImportError(
            f"distilaide.{name} needs the {extra!r} backend, which is not installed.\n"
            f"  pip install 'distilaide[{extra}]'\n"
            f"(underlying import error: {exc})"
        ) from exc


def __dir__() -> Any:
    """Advertise the lazy names to ``dir()`` / tab-completion."""
    return sorted({*globals(), *_BACKEND_NAMES})
