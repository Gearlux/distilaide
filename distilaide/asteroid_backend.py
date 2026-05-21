"""asteroid backend — pretrained speech-enhancement / source-separation models.

`asteroid <https://asteroid-team.github.io/>`__ exposes a unified
``BaseModel.from_pretrained(<hub_id>)`` interface over a wide catalog of
pretrained checkpoints on Hugging Face. We surface the most commonly-used
denoising and separation models by short alias and let users override the
checkpoint via ``checkpoint=`` for anything else.

==================  =====================================================================
Method alias        Hugging Face checkpoint
==================  =====================================================================
``dccrn``           ``JorisCos/DCCRNet_Libri1Mix_enhsingle_16k`` (speech enhancement)
``dcunet``          ``JorisCos/DCUNet_Libri1Mix_enhsingle_16k`` (speech enhancement)
``convtasnet``      ``JorisCos/ConvTasNet_Libri2Mix_sepclean_16k`` (source separation)
``demucs``          ``mhubert/Demucs_speech_enhancement_16k`` (speech enhancement)
==================  =====================================================================

Inputs are 1D (samples,) or 2D (channels, samples) ``float32`` arrays at the
model's native sample rate (typically 16 kHz). Resampling is the caller's
responsibility — keep the package thin.

Available via ``pip install distilaide[asteroid]``.
"""

from typing import Any, Optional

import numpy as np

from logflow import get_logger

logger = get_logger(__name__)

#: Aliases this backend serves, mapped to default Hugging Face checkpoints.
DEFAULT_CHECKPOINTS = {
    "dccrn": "JorisCos/DCCRNet_Libri1Mix_enhsingle_16k",
    "dcunet": "JorisCos/DCUNet_Libri1Mix_enhsingle_16k",
    "convtasnet": "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k",
    "demucs": "mhubert/Demucs_speech_enhancement_16k",
}
METHODS = tuple(DEFAULT_CHECKPOINTS.keys())


def is_available() -> bool:
    """Return True iff ``asteroid`` and ``torch`` can be imported."""
    try:
        import asteroid  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_checkpoint(method: str, checkpoint: Optional[str]) -> str:
    """Resolve an explicit checkpoint or fall back to the alias's default."""
    if checkpoint:
        return checkpoint
    if method not in DEFAULT_CHECKPOINTS:
        raise ValueError(
            f"asteroid backend does not serve method {method!r}; "
            f"pass `checkpoint=...` explicitly to use a custom Hugging Face hub id."
        )
    return DEFAULT_CHECKPOINTS[method]


def denoise(
    signal: np.ndarray,
    method: str = "dccrn",
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    **_: Any,
) -> np.ndarray:
    """Run an asteroid pretrained model on ``signal``.

    Args:
        signal: 1D (samples,) or 2D (channels, samples) ``float32`` array at
            the model's native sample rate.
        method: One of :data:`METHODS`, or an arbitrary string when
            ``checkpoint`` is supplied.
        checkpoint: Optional Hugging Face hub id overriding the alias default.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        Denoised waveform, same shape and dtype as ``signal``. For separation
        models (``convtasnet``) the first source is returned.
    """
    import torch
    from asteroid.models import BaseModel

    hub_id = _resolve_checkpoint(method, checkpoint)
    logger.info(
        f"asteroid {method}: shape={signal.shape} checkpoint={hub_id} device={device}"
    )

    model = BaseModel.from_pretrained(hub_id).to(device).eval()
    work = signal.astype(np.float32)
    if work.ndim == 1:
        tensor = torch.from_numpy(work)[None, None, :]
    elif work.ndim == 2:
        tensor = torch.from_numpy(work)[None, :, :]
    else:
        raise ValueError(f"asteroid expects 1D/2D input, got ndim={work.ndim}")

    tensor = tensor.to(device)
    with torch.no_grad():
        out_t = model(tensor)

    out = out_t.detach().cpu().numpy()
    # Separation models return (batch, n_src, samples); pick the first source.
    if out.ndim == 3 and out.shape[1] > 1 and signal.ndim == 1:
        out = out[:, 0:1, :]

    if signal.ndim == 1:
        out = out.reshape(-1)
    else:
        out = out.reshape(out.shape[-2], out.shape[-1])
    return np.asarray(out, dtype=signal.dtype)
