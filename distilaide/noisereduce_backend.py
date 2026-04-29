"""noisereduce backend — classical spectral-gating denoiser.

`noisereduce <https://pypi.org/project/noisereduce/>`__ is a tiny, pure-Python
spectral-gating denoiser that excels on stationary noise (HVAC hum, broadband
hiss). It accepts a noise sample for stationary mode, or estimates the noise
profile itself when none is provided.

Available via ``pip install distilaide[classical]``.
"""

from typing import Any, Optional

import numpy as np
from logflow import get_logger

logger = get_logger(__name__)

#: Methods this backend serves.
METHODS = ("noisereduce", "spectral-gate")


def is_available() -> bool:
    """Return True iff the upstream ``noisereduce`` package can be imported."""
    try:
        import noisereduce  # noqa: F401
    except ImportError:
        return False
    return True


def denoise(
    signal: np.ndarray,
    sample_rate: int = 16000,
    noise_clip: Optional[np.ndarray] = None,
    stationary: bool = True,
    method: str = "noisereduce",
    **kwargs: Any,
) -> np.ndarray:
    """Spectral-gate denoise of a 1D (mono) or 2D (channels-first) signal.

    Args:
        signal: 1D (samples,) or 2D (channels, samples) float array.
        sample_rate: Sampling rate in Hz. Required by ``noisereduce``'s STFT.
        noise_clip: Optional noise-only segment (same shape rules as
            ``signal``); when provided, used to estimate the noise profile
            instead of self-estimation.
        stationary: ``True`` for stationary noise (HVAC, hiss); ``False`` for
            non-stationary (footsteps, music).
        method: Accepted for dispatcher symmetry; any of ``METHODS`` is fine.
        **kwargs: Forwarded to ``noisereduce.reduce_noise`` (e.g.
            ``prop_decrease``, ``time_constant_s``, ``n_std_thresh``).

    Returns:
        Denoised array, same shape and dtype as ``signal``.
    """
    if method not in METHODS:
        raise ValueError(f"noisereduce backend does not serve method {method!r}")

    import noisereduce as nr  # local import — see module docstring

    original_dtype = signal.dtype
    work = signal.astype(np.float32)
    logger.info(
        f"noisereduce: shape={work.shape} sample_rate={sample_rate} stationary={stationary} "
        f"noise_clip={'yes' if noise_clip is not None else 'estimated'}"
    )
    out = nr.reduce_noise(
        y=work,
        sr=sample_rate,
        y_noise=noise_clip.astype(np.float32) if noise_clip is not None else None,
        stationary=stationary,
        **kwargs,
    )
    return np.asarray(out, dtype=original_dtype)
