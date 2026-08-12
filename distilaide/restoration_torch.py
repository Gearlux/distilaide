"""``RestorationTorchHooks`` — the torch-family implementations of the task hooks.

The restoration answers to the trainer API's typed hooks
(:class:`matrainer.torch.runnable.TorchTrainerRunnable`): the target-image read, the L1
default loss, the PSNR/SSIM default templates, and the ``data_range`` sizing keyword. No
``_report_nonscalars`` override — every full-reference image metric is a SCALAR, so the
neutral no-op is exactly right (the asymmetry with the classification/per-pixel tasks).

This module imports torch, so the keras/tensorflow/jax/mlx backends must never import it —
the same rule as :mod:`distilaide.torch_restoration` (the registered loss/metric set, a
DIFFERENT module: that one registers upstream classes, this one implements hooks).
"""

from typing import Dict, Union

import torch.nn as nn
from confluid import PartialClass
from loggair import get_logger
from matrainer.torch.metrics import MetricsSpec
from matrainer.torch.runnable import TorchTrainerRunnable
from recordstream import batch_tensor, restoration_output
from recordstream.items import Record
from recordstream.outputs import RestorationOutput
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from distilaide.restoration_base import RestorationBase

logger = get_logger(__name__)

__all__ = ["RestorationTorchHooks"]


class RestorationTorchHooks(RestorationBase, TorchTrainerRunnable):
    """The torch half of the restoration task, mixed into every torch-family backend."""

    def _targets(self, batch: Record) -> Tensor:
        """The clean target IMAGE the loss compares against — ``[N, C, H, W]`` float.

        The one line that says image-to-image out loud. Every other task in this workspace
        reads a LABEL here (a class id, a per-pixel id map, a box set); a restoration target
        is another image, in the same layout, dtype and value range as the input — which is
        why the batch reader is the SAME ``batch_tensor`` call the input uses, with no dtype
        argument: the pipeline that produced the input produced this the same way, so naming
        a dtype here could only introduce a disagreement between the two halves of one pair.
        """
        return batch_tensor(batch, self.target_key, self.device)

    def _default_loss(self) -> nn.Module:
        # L1 rather than MSE: DnCNN's original recipe was MSE, but every modern restoration
        # network reports L1 doing better on the same data — MSE's quadratic penalty is
        # dominated by a few large errors and produces visibly blurrier output at a comparable
        # PSNR. `loss_fn: !class:MSELoss@framework=torch` is the one line back.
        return nn.L1Loss()

    def _default_metrics(self) -> MetricsSpec:
        # The restoration pair: PSNR (the field's primary score, in dB) and SSIM (the
        # perceptual one, in [0, 1]). Both are DEFERRED templates rather than instances,
        # because `data_range` is injected into them at build time — torchmetrics 1.9 makes
        # it a REQUIRED argument of PeakSignalNoiseRatio, so a template that never received
        # one raises rather than quietly scoring against an inferred range; a template that
        # PINS its own keeps it (the injection is a floor).
        return {
            "psnr": PartialClass[Metric](PeakSignalNoiseRatio),
            "ssim": PartialClass[Metric](StructuralSimilarityIndexMeasure),
        }

    def _metric_flow_dims(self) -> Dict[str, Union[int, float]]:
        return {"data_range": self.data_range}

    def _prediction_output(self, preds: Tensor) -> RestorationOutput[Tensor]:
        return restoration_output(preds)
