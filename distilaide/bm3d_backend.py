"""BM3D backend — classical block-matching 3D denoiser via the ``bm3d`` PyPI package.

BM3D is the gold-standard non-AI baseline for Gaussian image denoising and is
still competitive with shallow CNNs at low noise levels. The upstream package
exposes a single ``bm3d.bm3d`` callable that operates on float64 arrays
normalized to roughly ``[0, 1]`` — we pass through unchanged.

Available via ``pip install distilaide[classical]``.
"""

from typing import Any

import numpy as np

from logflow import get_logger

logger = get_logger(__name__)

#: Methods this backend serves. The dispatcher in :mod:`distilaide.api` maps
#: each method name to the ``denoise()`` callable below.
METHODS = ("bm3d",)


def is_available() -> bool:
    """Return True iff the upstream ``bm3d`` package can be imported."""
    try:
        import bm3d  # noqa: F401
    except ImportError:
        return False
    return True


def denoise(
    signal: np.ndarray, sigma: float = 0.1, method: str = "bm3d", **_: Any
) -> np.ndarray:
    """Denoise ``signal`` with BM3D.

    Args:
        signal: 2D (grayscale) or 3D (HxWxC) float array, expected in roughly
            ``[0, 1]``. ``uint8`` inputs are auto-scaled.
        sigma: Estimated noise standard deviation in the same units as
            ``signal``. Required by BM3D's hard-thresholding stage; defaults
            to ``0.1`` which is a sensible mid-range value for ``[0, 1]`` data.
        method: Accepted but ignored — the backend serves only ``"bm3d"``.

    Returns:
        Denoised array, same shape and dtype as ``signal``.
    """
    if method != "bm3d":
        raise ValueError(f"bm3d backend does not serve method {method!r}")

    import bm3d as _bm3d  # local import — see module docstring

    original_dtype = signal.dtype
    if np.issubdtype(original_dtype, np.integer):
        scale = float(np.iinfo(original_dtype).max)
        work = signal.astype(np.float64) / scale
    else:
        scale = 1.0
        work = signal.astype(np.float64)

    logger.info(f"BM3D denoise: shape={work.shape} sigma={sigma}")
    out = _bm3d.bm3d(work, sigma_psd=sigma)

    if np.issubdtype(original_dtype, np.integer):
        out = np.clip(out * scale, 0, scale).astype(original_dtype)
    else:
        out = out.astype(original_dtype)
    return np.asarray(out)
