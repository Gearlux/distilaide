"""``RestorationKerasHooks`` — the restoration task's KERAS-family hook implementations.

The keras twin of :mod:`distilaide.restoration_torch`: the task hooks written once for BOTH
Keras-family machinery bases — the compile/fit member (:class:`matrainer.keras.KerasRunnable`)
and the native-TensorFlow loop (:class:`matrainer.tensorflow.TensorFlowRunnable`) — so the two
concretes cannot drift apart on what restoration means in the keras vocabulary. A concrete
backend composes ``(RestorationKerasHooks, RestorationBase, <Machinery>Runnable)``.

What the task says, in this vocabulary: the default loss is L1 (``MeanAbsoluteError`` — the
same objective the torch backends default to, because a config moving between engines must not
silently change it); the default metrics are PSNR (through this project's ONE shim,
:class:`distilaide.keras_restoration.PeakSignalNoiseRatio`, since ``keras.metrics`` ships no
image metric) sized by the task's ``data_range``, plus the mean squared error; and a
prediction is the one-key :class:`~recordstream.RestorationOutput` (the model emitted the
answer — there is nothing to derive). SSIM is deliberately absent: ``keras.ops`` has none and
an engine-specific one would make the two Keras-family backends disagree from one config. The
family's default ``_to_arrays`` (float32 in, float32 target) IS this task's batch shape, so
these hooks override nothing there.
"""

from typing import Any, List, Optional

import numpy as np
from confluid import Partial
from loggair import get_logger
from matrainer.keras.runnable import KerasLoss, KerasMetric
from recordstream import RestorationOutput
from recordstream.keras import keras

from distilaide.keras_restoration import PeakSignalNoiseRatio
from distilaide.restoration_base import RestorationBase

logger = get_logger(__name__)

__all__ = ["RestorationKerasHooks"]


class RestorationKerasHooks(RestorationBase):
    """The keras-typed restoration hooks, shared by the Keras and TensorFlow concretes."""

    def _default_keras_loss(self) -> Optional[Partial[KerasLoss]]:
        """L1, the same choice the torch backends make — a config moving between engines must
        not silently change its objective."""
        return keras.losses.MeanAbsoluteError()

    def _default_keras_metrics(self) -> Optional[List[Partial[KerasMetric]]]:
        """PSNR (through this project's one shim, since ``keras.metrics`` has none) plus the
        mean squared error — both sized/named so the sweep's shared columns fill."""
        return [
            PeakSignalNoiseRatio(data_range=self.data_range, name="psnr"),
            keras.metrics.MeanSquaredError(name="mse"),
        ]

    def _prediction_output(self, preds: Any) -> Any:
        """The batched :class:`~recordstream.RestorationOutput` — the contract's ONE key.

        The contract is recordstream's and framework-neutral — generic in its array type — so
        the SAME sink that consumes the torch backends' output consumes this numpy one with no
        adapter; the machinery slices the batch per record.
        """
        out: RestorationOutput[np.ndarray] = {"image": np.asarray(preds)}
        return out
