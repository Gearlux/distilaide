"""The **Keras** restoration losses and metrics — registered, and ONE shim.

One module per ENGINE, because a loss is engine-bound: everything below is ``keras.losses`` /
``keras.metrics`` and carries ``framework="keras"``, so a picker asked for "a restoration loss" for
a torch trainer is not offered one. It serves BOTH non-torch Keras-family backends — in TF >= 2.16
``tf.keras`` IS Keras 3, so a second ``tensorflow_restoration.py`` would register the same classes
under a second tag and make every bare ``!class:MeanAbsoluteError`` ambiguous for no gain.

Loss: the default is :class:`keras.losses.MeanAbsoluteError` — L1, the same choice the torch
backends make, so a config moving between engines does not silently change its objective.

**Metrics are where this engine is genuinely narrower, and the shim below is the exception that
proves it.** ``keras.metrics`` ships NO image-quality metric at all: no PSNR, no SSIM (verified
against Keras 3.15 — the regression family is Mean{Squared,Absolute}Error and friends). What Keras
DOES have is ``keras.ops.psnr``, a backend-neutral FUNCTION — which is exactly the "generic shim"
case the workspace's no-wrappers rule allows: the upstream primitive is a function, so confluid's
``register`` (which only tags classes) cannot reach it, and a minimal ``Metric`` that stores knobs
and delegates is a transport shim rather than a reimplementation.

**SSIM is deliberately absent.** ``keras.ops`` has no SSIM, and the only ones available are
engine-SPECIFIC (``tf.image.ssim``) — wiring that would make the ``tensorflow`` backend compute a
metric the ``keras`` backend cannot, from the same config. A narrower vocabulary is the honest
answer; the torch backends keep SSIM, and PSNR — the field's primary score — is common to all.
"""

from typing import Any, Optional

from confluid import configurable, register
from loggair import get_logger

from distilaide._keras import keras

logger = get_logger(__name__)

MeanAbsoluteError = keras.losses.MeanAbsoluteError
MeanSquaredError = keras.losses.MeanSquaredError
Huber = keras.losses.Huber
LogCosh = keras.losses.LogCosh

#: The metric namesakes. Keras publishes ``MeanSquaredError`` / ``MeanAbsoluteError`` under BOTH
#: ``keras.losses`` and ``keras.metrics``; confluid admits a shared registry name when a tag tells
#: the entries apart, and ``role`` does — a config picks one with ``!class:MeanSquaredError@role=metric``.
#: The PYTHON binding still has to be unique, hence the ``*Metric`` suffix on the alias.
MeanSquaredErrorMetric = keras.metrics.MeanSquaredError
MeanAbsoluteErrorMetric = keras.metrics.MeanAbsoluteError
RootMeanSquaredError = keras.metrics.RootMeanSquaredError


@configurable(task="restoration", role="metric", framework="keras")
class PeakSignalNoiseRatio(keras.metrics.Metric):
    """PSNR as a Keras metric — a transport shim over the mean squared error, not new arithmetic.

    **Why this class exists at all:** ``keras.metrics`` has no image-quality metric, and
    ``keras.ops.psnr`` is a FUNCTION, so there is nothing for confluid's registry to tag and
    nothing a ``compile(metrics=[...])`` list can hold. This is the workspace's sanctioned
    "generic shim": it stores a knob and delegates.

    **What it must NOT do — and this is the subtle part — is call ``keras.ops.psnr`` per batch and
    average.** PSNR is a logarithm of a mean squared error, so the mean of per-batch PSNRs is not
    the PSNR of the split, and the gap grows with how much the batches differ. So this holds an
    upstream :class:`keras.metrics.MeanSquaredError` — which accumulates correctly — and converts
    ONCE in :meth:`result`. The conversion is PSNR's definition, one line; the error arithmetic is
    Keras's.

    That also makes it agree with the torch backends' ``torchmetrics.PeakSignalNoiseRatio``, which
    accumulates the same way — verified to 6 decimal places on the same arrays, which is the whole
    point of having one column called ``psnr``.

    Args:
        data_range: The span of the target's values — ``1.0`` for ``[0, 1]`` floats, ``255.0`` for
            raw ``uint8``. The restorer injects its own, exactly as it does for the torch metrics.
        name: Metric name; ``psnr`` so the column matches every other backend's.
    """

    def __init__(self, data_range: float = 1.0, name: str = "psnr", dtype: Optional[Any] = None) -> None:
        super().__init__(name=name, dtype=dtype)
        self.data_range = float(data_range)
        # The accumulator is UPSTREAM's. Nothing here counts, sums or divides.
        self._mse = keras.metrics.MeanSquaredError(name=f"{name}_mse", dtype=dtype)

    def update_state(self, y_true: Any, y_pred: Any, sample_weight: Any = None) -> Any:
        return self._mse.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self) -> Any:
        """``10 * log10(data_range^2 / mse)`` — PSNR's definition, over the accumulated error."""
        mse = self._mse.result()
        # A perfect reconstruction gives mse == 0 and an infinite PSNR; the epsilon keeps the
        # metric finite rather than poisoning a whole run's history with `inf`.
        mse = keras.ops.maximum(mse, keras.backend.epsilon())
        return 10.0 * keras.ops.log10(keras.ops.square(self.data_range) / mse)

    def reset_state(self) -> None:
        self._mse.reset_state()

    def get_config(self) -> Any:
        """Keras clones a metric through ``get_config`` / ``from_config``; ``data_range`` must ride along."""
        return {**super().get_config(), "data_range": self.data_range}


#: Off-the-shelf pixel losses a Keras image-to-image run can wire. All take the same
#: ``(y_true, y_pred)`` pair of ``[N, H, W, C]`` float images.
_RESTORATION_LOSSES = (
    MeanAbsoluteError,
    MeanSquaredError,
    Huber,
    LogCosh,
)

#: The metrics. Everything except the PSNR shim is upstream's, registered rather than wrapped.
_RESTORATION_METRICS = (
    MeanSquaredErrorMetric,
    MeanAbsoluteErrorMetric,
    RootMeanSquaredError,
)

for _loss in _RESTORATION_LOSSES:
    register(_loss, task="restoration", role="loss", framework="keras")

for _metric in _RESTORATION_METRICS:
    register(_metric, task="restoration", role="metric", framework="keras")


# Every registered class MUST be named here: `confluid.discovery.scan_module` filters members by
# `__module__` (which is `keras.src.*` for these), so the `__all__`-gated pass is what carries them
# into StreamStudio's palette. `PeakSignalNoiseRatio` is OURS and would be picked up either way;
# it is listed for uniformity. `tests/test_categories.py` pins this list against the registrations.
__all__ = [
    # losses — role="loss"
    "MeanAbsoluteError",
    "MeanSquaredError",
    "Huber",
    "LogCosh",
    # metrics — role="metric"
    "MeanSquaredErrorMetric",
    "MeanAbsoluteErrorMetric",
    "RootMeanSquaredError",
    "PeakSignalNoiseRatio",
]
