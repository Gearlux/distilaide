"""Distilaide — adapter over deepinv, asteroid, BM3D, and noisereduce.

A thin glue layer that exposes a single :func:`distilaide.denoise` dispatcher
and a Confluid-:class:`distilaide.Denoiser` configurable, wrapping four
upstream denoising libraries:

- `bm3d <https://pypi.org/project/bm3d/>`__ — classical block-matching 3D
  Gaussian image denoiser.
- `noisereduce <https://pypi.org/project/noisereduce/>`__ — classical
  spectral-gating audio denoiser.
- `deepinv <https://deepinv.github.io/>`__ — pretrained CNN/Transformer image
  denoising backbones (DnCNN, DRUNet, SwinIR, SCUNet, Restormer).
- `asteroid <https://asteroid-team.github.io/>`__ — pretrained speech
  enhancement / source separation models (DCCRN, DCUNet, ConvTasNet, Demucs).

Backends are imported lazily, so any subset of the optional extras
(``[deepinv]``, ``[asteroid]``, ``[classical]``) is enough for the package
to import.
"""

__version__ = "0.1.0"

from confluid import register

from distilaide.api import Denoiser, available_methods, denoise, list_methods

register(Denoiser)

__all__ = [
    "Denoiser",
    "denoise",
    "list_methods",
    "available_methods",
]
