"""The **torch** restoration losses and metrics — registered, never wrapped.

One module per ENGINE, because a loss is engine-bound: everything below is ``torch.nn`` /
``torchmetrics.image`` and carries ``framework="torch"``, so a picker asked for "a restoration loss"
for a non-torch trainer is not offered one. Each engine's set is its own
``confluid.configurables`` entry point, so on an install without torch this module simply fails to
import and a discovery scan skips it — which is the correct answer, not a degraded one.

**Nothing here is a distilaide class.** These are off-the-shelf upstream classes tagged for
discovery, which is the workspace model: tag and register what exists; navigaitor scans and
generates. A restoration loss that *reimplements* L1 or SSIM does not belong here.

Loss: the default is :class:`torch.nn.L1Loss` over ``[N, C, H, W]`` predicted and target images.
**L1 rather than MSE, and the choice is not arbitrary** — DnCNN's 2016 paper trained on MSE, but
every modern restoration network (Restormer, NAFNet, MPRNet) reports L1 doing better on the same
data, because MSE's quadratic penalty is dominated by a few large errors and yields visibly blurrier
output at a comparable PSNR. MSE is registered right beside it, so a run reproducing the original
recipe writes one line. ``SmoothL1Loss`` / ``HuberLoss`` are the two-regime compromises between
them.

Metrics: :class:`torchmetrics.image.PeakSignalNoiseRatio` (PSNR, the field's primary score) and
:class:`~torchmetrics.image.StructuralSimilarityIndexMeasure` (SSIM, the perceptual one) are the
restorer's defaults, with the rest of the full-reference family registered alongside.

**Every one of them needs ``data_range``, and getting it wrong is silent** — see the restorer's
``data_range`` knob, which is injected into each template exactly as a segmentation task injects
``num_classes``. Measured on one batch of ``[0, 1]`` images: the correct ``data_range=1.0`` gives
20.46 dB where ``data_range=255.0`` gives **68.59 dB** on the identical tensors. torchmetrics 1.9
makes ``data_range`` a REQUIRED argument of ``PeakSignalNoiseRatio`` for this reason, so a
``!lazy:`` template that never received one does not silently mis-score — it fails to build.

Typical wiring (YAML)::

    loss_fn: !class:MSELoss@framework=torch      # the original DnCNN recipe

    val_metrics:
      ms_ssim: !lazy:torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure
"""

from confluid import register
from torch.nn import HuberLoss, L1Loss, MSELoss, SmoothL1Loss
from torchmetrics.image import (
    ErrorRelativeGlobalDimensionlessSynthesis,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    RootMeanSquaredErrorUsingSlidingWindow,
    SpectralAngleMapper,
    StructuralSimilarityIndexMeasure,
    UniversalImageQualityIndex,
)

#: Off-the-shelf pixel losses an image-to-image run can wire. All four take the same
#: ``(prediction, target)`` pair of ``[N, C, H, W]`` float images and return a scalar.
_RESTORATION_LOSSES = (
    L1Loss,
    MSELoss,
    SmoothL1Loss,
    HuberLoss,
)

#: The torchmetrics FULL-REFERENCE image family — every one of these scores a restored image
#: against its ground truth, which is exactly the pair this task produces. Deliberately NOT
#: registered: the no-reference metrics (`QualityWithNoReference`), the distribution metrics
#: (`MemorizationInformedFrechetInceptionDistance`) and `LearnedPerceptualImagePatchSimilarity`,
#: which downloads a VGG/AlexNet backbone on first use — a metric that silently pulls weights over
#: the network does not belong in a picker's default list. A config can still name any of them by
#: dotted path.
_RESTORATION_METRICS = (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
    ErrorRelativeGlobalDimensionlessSynthesis,
    RootMeanSquaredErrorUsingSlidingWindow,
    SpectralAngleMapper,
    UniversalImageQualityIndex,
)

for _loss in _RESTORATION_LOSSES:
    register(_loss, task="restoration", role="loss", framework="torch")

for _metric in _RESTORATION_METRICS:
    register(_metric, task="restoration", role="metric", framework="torch")


# Every registered class MUST be named here: `confluid.discovery.scan_module` filters members by
# `__module__` (which is `torch` / `torchmetrics` for these), so the `__all__`-gated pass is what
# carries them into StreamStudio's palette. `tests/test_categories.py` pins this list against the
# registrations above.
__all__ = [
    # losses — role="loss"
    "L1Loss",
    "MSELoss",
    "SmoothL1Loss",
    "HuberLoss",
    # metrics — role="metric"
    "PeakSignalNoiseRatio",
    "StructuralSimilarityIndexMeasure",
    "MultiScaleStructuralSimilarityIndexMeasure",
    "ErrorRelativeGlobalDimensionlessSynthesis",
    "RootMeanSquaredErrorUsingSlidingWindow",
    "SpectralAngleMapper",
    "UniversalImageQualityIndex",
]
