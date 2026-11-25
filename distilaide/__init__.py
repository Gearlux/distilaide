"""Distilaide: image restoration across training backends, driven by one config.

**The core is backend-free.** ``pip install distilaide`` gives you the CLI and the config surface;
a training framework arrives with the backend you select::

    pip install 'distilaide[lightning]'      # PyTorch Lightning — the reference backend

Which is why every backend name below resolves LAZILY (PEP 562): importing ``distilaide`` must not
require a framework the operator did not choose. Touching a name whose backend is missing raises an
:class:`ImportError` naming the exact extra, instead of a traceback from three libraries down.
"""

from importlib.metadata import PackageNotFoundError, version

from matrainer._exports import lazy_exports

try:
    # Single source of truth: the installed distribution's metadata, i.e. pyproject.toml's
    # `version` (the distribution is `distilaide`). Never type the number in here as well.
    __version__ = version("distilaide")
except PackageNotFoundError:  # pragma: no cover - uninstalled source checkout
    __version__ = "0.0.0.dev0"


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
    # The waveform denoiser: a 1D U-Net over [N, 2, L] IQ and its sample + STFT loss.
    "IqUNet": ("distilaide.iq_denoiser", "torch"),
    "IqDenoisingLoss": ("distilaide.iq_denoiser", "torch"),
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

# The resolver is matrainer's `lazy_exports` (the machinery was identical in every training
# project's root); the `_BACKEND_NAMES` table above is the project DATA it runs on.
__getattr__, __dir__ = lazy_exports(__name__, _BACKEND_NAMES)
