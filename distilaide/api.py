"""Unified denoising API — one ``denoise()`` function and one ``Denoiser`` configurable.

The dispatcher maps a short ``method`` name to the right backend module.
Backends are imported lazily so the package works with any subset of the
optional ``[deepinv]`` / ``[asteroid]`` / ``[classical]`` extras installed.

Method registry
---------------

==============  ================  =================================================
Method          Backend           Domain
==============  ================  =================================================
``bm3d``        bm3d              Classical image denoising (Gaussian).
``noisereduce`` noisereduce       Classical spectral-gating audio denoising.
``spectral-gate`` noisereduce     Alias of ``noisereduce``.
``dncnn``       deepinv           CNN image denoiser (Zhang 2016).
``drunet``      deepinv           Residual U-Net (plug-and-play workhorse).
``swinir``      deepinv           Swin transformer restoration.
``scunet``      deepinv           Swin-Conv-UNet (real-image denoising).
``restormer``   deepinv           Transformer restoration.
``dccrn``       asteroid          Deep complex CRN, speech enhancement.
``dcunet``      asteroid          Deep complex U-Net, speech enhancement.
``convtasnet``  asteroid          ConvTasNet, source separation.
``demucs``      asteroid          Demucs, speech enhancement.
==============  ================  =================================================

Add new backends by:

1. Dropping a ``<name>_backend.py`` module exposing ``METHODS``,
   ``is_available()`` and ``denoise(signal, method, **kwargs)``.
2. Appending it to :data:`_BACKEND_MODULES` below.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Optional

import numpy as np
from confluid import configurable
from logflow import get_logger

logger = get_logger(__name__)

#: Backend module names, in registration order. Keep purely classical first so
#: ``list_methods()`` lists lightweight options before heavy ML deps.
_BACKEND_MODULES: tuple[str, ...] = (
    "distilaide.bm3d_backend",
    "distilaide.noisereduce_backend",
    "distilaide.deepinv_backend",
    "distilaide.asteroid_backend",
)


def _load_backend(module_path: str) -> ModuleType:
    return import_module(module_path)


def _build_registry() -> dict[str, str]:
    """Map method name → backend module path.

    We import every backend module at registry-build time (cheap; backends
    only import their *upstream* lib lazily inside ``denoise()``).
    """
    registry: dict[str, str] = {}
    for module_path in _BACKEND_MODULES:
        backend = _load_backend(module_path)
        for method in backend.METHODS:
            if method in registry:
                raise RuntimeError(
                    f"Method {method!r} is registered by both " f"{registry[method]!r} and {module_path!r}"
                )
            registry[method] = module_path
    return registry


_REGISTRY: dict[str, str] = _build_registry()


def list_methods(only_available: bool = False) -> list[dict[str, Any]]:
    """Return a list of registered methods with their backend and availability.

    Args:
        only_available: If True, omit methods whose upstream package is not
            importable in the current environment.

    Returns:
        A list of ``{"method", "backend", "available"}`` dicts.
    """
    out = []
    for method, module_path in _REGISTRY.items():
        backend = _load_backend(module_path)
        available = backend.is_available()
        if only_available and not available:
            continue
        out.append(
            {
                "method": method,
                "backend": module_path.rsplit(".", 1)[-1].removesuffix("_backend"),
                "available": available,
            }
        )
    return out


def denoise(signal: np.ndarray, method: str = "bm3d", **kwargs: Any) -> np.ndarray:
    """Dispatch ``signal`` to the backend serving ``method``.

    Args:
        signal: Input array — shape constraints depend on the backend
            (1D/2D for audio backends, 2D/3D/4D for image backends).
        method: One of the keys in the registry (see module docstring).
        **kwargs: Forwarded to the backend's ``denoise()``.

    Returns:
        Denoised array with the same dtype as ``signal``.
    """
    if method not in _REGISTRY:
        raise ValueError(f"Unknown denoising method {method!r}. Known methods: " f"{sorted(_REGISTRY.keys())}")
    module_path = _REGISTRY[method]
    backend = _load_backend(module_path)
    if not backend.is_available():
        raise RuntimeError(
            f"Backend {module_path!r} for method {method!r} is not available. "
            f"Install the corresponding extra (e.g. `pip install distilaide[deepinv]`)."
        )
    logger.debug(f"Dispatching method={method} to {module_path}")
    result = backend.denoise(signal, method=method, **kwargs)
    return np.asarray(result)


@configurable
class Denoiser:
    """Confluid-configurable wrapper around :func:`denoise`.

    Use from YAML as::

        denoiser: !class:distilaide.Denoiser
          method: bm3d
          sigma: 0.05

    Then call ``denoiser.apply(signal)`` from your pipeline. Construction is
    cheap (just stores the config); the heavy lifting happens on each call.
    """

    def __init__(self, method: str = "bm3d", **kwargs: Any) -> None:
        if method not in _REGISTRY:
            raise ValueError(f"Unknown denoising method {method!r}. Known methods: " f"{sorted(_REGISTRY.keys())}")
        self.method = method
        self.kwargs = kwargs

    @property
    def backend(self) -> str:
        """Short backend name (``bm3d``, ``noisereduce``, ``deepinv``, ``asteroid``)."""
        return _REGISTRY[self.method].rsplit(".", 1)[-1].removesuffix("_backend")

    def apply(self, signal: np.ndarray, **overrides: Any) -> np.ndarray:
        """Denoise ``signal`` using the configured method.

        Call-time ``overrides`` win over construction-time ``kwargs``.
        """
        merged: dict[str, Any] = {**self.kwargs, **overrides}
        return denoise(signal, method=self.method, **merged)

    def __call__(self, signal: np.ndarray, **overrides: Any) -> np.ndarray:
        return self.apply(signal, **overrides)


def available_methods(method: Optional[str] = None) -> bool:
    """Return whether ``method`` (or any method) has its backend installed."""
    if method is None:
        return any(entry["available"] for entry in list_methods())
    if method not in _REGISTRY:
        return False
    return bool(_load_backend(_REGISTRY[method]).is_available())
