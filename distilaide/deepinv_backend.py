"""deepinv backend — pretrained CNN/Transformer denoisers via ``deepinv.models``.

`deepinv <https://deepinv.github.io/>`__ ships pretrained weights for the
canonical image-denoising backbones:

==============  ==========================================================
Method          Backbone
==============  ==========================================================
``dncnn``       Residual CNN (Zhang 2016) — Gaussian denoising baseline.
``drunet``      Residual U-Net (Zhang 2021) — plug-and-play prior workhorse.
``swinir``      Swin transformer (Liang 2021) — strong, slower.
``scunet``      Swin-Conv-UNet (Zhang 2022) — real-image denoising.
``restormer``   Restormer (Zamir 2022) — high-res restoration.
==============  ==========================================================

Inputs are 2D (HxW), 3D (HxWxC), or 4D (NxCxHxW) ``float32`` arrays in
``[0, 1]``. Internally we reshape to NCHW, run the model, and reshape back.

Available via ``pip install distilaide[deepinv]``.
"""

from typing import Any, Optional

import numpy as np
from loggair import get_logger

logger = get_logger(__name__)

#: Methods this backend serves. Mirrors the model families exposed by
#: ``deepinv.models``; the dispatcher picks the matching constructor.
METHODS = ("dncnn", "drunet", "swinir", "scunet", "restormer")


def is_available() -> bool:
    """Return True iff the upstream ``deepinv`` and ``torch`` packages can be imported."""
    try:
        import deepinv  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _build_model(method: str, **model_kwargs: Any) -> Any:
    """Instantiate the deepinv model class corresponding to ``method``.

    Each branch defers to the pretrained-weights default of the upstream
    constructor — overrides flow through ``model_kwargs``. Method validation
    happens *before* the upstream import so callers get a useful error even
    when the ``[deepinv]`` extra is not installed.
    """
    if method not in METHODS:
        raise ValueError(f"deepinv backend does not serve method {method!r}")

    import deepinv as dinv

    builders = {
        "dncnn": dinv.models.DnCNN,
        "drunet": dinv.models.DRUNet,
        "swinir": dinv.models.SwinIR,
        "scunet": dinv.models.SCUNet,
        "restormer": dinv.models.Restormer,
    }
    return builders[method](pretrained="download", **model_kwargs)


def _to_nchw(signal: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape a 2D/3D/4D array to NCHW; return (tensor, original_shape)."""
    original_shape = signal.shape
    if signal.ndim == 2:
        nchw = signal[None, None, :, :]
    elif signal.ndim == 3:
        # HxWxC → 1xCxHxW
        nchw = np.transpose(signal, (2, 0, 1))[None, ...]
    elif signal.ndim == 4:
        nchw = signal
    else:
        raise ValueError(f"deepinv expects 2D/3D/4D input, got ndim={signal.ndim}")
    return nchw, original_shape


def _from_nchw(out: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of :func:`_to_nchw`."""
    if len(original_shape) == 2:
        return np.asarray(out[0, 0])
    if len(original_shape) == 3:
        return np.asarray(np.transpose(out[0], (1, 2, 0)))
    return np.asarray(out)


def denoise(
    signal: np.ndarray,
    method: str = "dncnn",
    sigma: Optional[float] = None,
    device: str = "cpu",
    model_kwargs: Optional[dict[str, Any]] = None,
    **_: Any,
) -> np.ndarray:
    """Denoise an image (or batch) with a pretrained deepinv backbone.

    Args:
        signal: 2D, 3D (HxWxC), or 4D (NxCxHxW) ``float32`` array in ``[0, 1]``.
        method: One of :data:`METHODS`.
        sigma: Optional noise level (in ``[0, 1]`` units) passed to models that
            accept it (``drunet``, ``scunet``); ignored for blind models.
        device: ``"cpu"`` or ``"cuda"``; the model is moved before inference.
        model_kwargs: Extra kwargs forwarded to the model constructor.

    Returns:
        Denoised array with the same shape and dtype as ``signal``.
    """
    import torch

    model = _build_model(method, **(model_kwargs or {}))
    model = model.to(device).eval()

    nchw, original_shape = _to_nchw(signal)
    tensor = torch.from_numpy(nchw.astype(np.float32)).to(device)
    logger.info(f"deepinv {method}: shape={tuple(tensor.shape)} sigma={sigma} device={device}")

    with torch.no_grad():
        if sigma is not None and method in {"drunet", "scunet"}:
            out_t = model(tensor, sigma=sigma)
        else:
            out_t = model(tensor)

    out = out_t.detach().cpu().numpy()
    return _from_nchw(out, original_shape).astype(signal.dtype)
