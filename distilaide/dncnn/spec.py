"""The ONE DnCNN architecture, written once so several frameworks can implement it.

:class:`~distilaide.deepinv_model.DeepInvModel` wraps somebody's zoo. This is the opposite: a small
network specified here and implemented natively per framework. It exists because **the zoo does not
line up** — deepinv is torch-ONLY, so a cross-framework comparison built on it could never include
Keras, TensorFlow, Flax or MLX, and five zoo models would compare five networks instead of five
frameworks. Implementing ONE architecture several times makes the comparison mean something.

For real work prefer the zoo wrapper — a small net trained from scratch loses to a published
checkpoint every time. This is the control, not the recommendation.

**The architecture** — Zhang et al.'s DnCNN (2016), the canonical learned denoiser, deliberately
shallow::

    input (H, W, C)
      head:   conv3x3(width) -> ReLU                       # no normalization on the first layer
      body:   [conv3x3(width) -> BN -> ReLU] x (depth - 2)
      tail:   conv3x3(C)                                   # predicts the NOISE, not the image
    output = input - tail(...)                             # the residual connection

**The residual connection is the architecture's whole idea, and it is parameter-free.** DnCNN
learns to predict the noise and subtracts it, rather than learning to output a clean image. Two
consequences this project depends on:

* **Approximate identity is the easy local optimum, so a run's FIRST EPOCH lands on the do-nothing
  baseline** — predicting zero noise is the cheapest thing the network can learn, and it gets there
  within one pass. That makes the first epoch a free correctness check: a value far from the
  baseline means the degraded/clean pair is wrong, not that the model is good.
  **Note it is NOT an initialization property**, which is the tempting and wrong explanation:
  measured at init on ``[0, 1]`` images, this implementation deviates from the identity by
  ``mean 0.35 / max 1.86`` and deepinv's own DnCNN by ``mean 0.24 / max 1.08``, because both
  initialize the final convolution the usual way rather than at zero. The floor is reached by
  TRAINING, in the first epoch.
* It costs no parameters, so it cannot make the five implementations disagree.

**Three choices that exist to keep the implementations comparable, not because they are best:**

* **Normalization is on the BODY only** — not the first layer, not the last. That is the original
  paper's shape, and pinning it here stops one framework's "sensible default" (Keras layers happily
  normalize everything) from quietly changing the parameter count.
* **Every convolution carries a bias**, including the ones followed by batch norm where it is
  formally redundant. Frameworks disagree on whether to drop it (`use_bias=False` is the common
  optimization) and the disagreement is worth 32 parameters per layer — enough to break the
  five-way match for no benefit at this size.
* **Layout is per-implementation, and only the tensor differs.** Keras, TensorFlow, Flax and MLX
  default to NHWC; torch is NCHW. The ``input_shape`` KNOB stays ``(H, W, C)`` in every
  implementation so one config line drives any of them, and only ``C`` reaches the parameter count.

Unlike a U-Net there is **no size divisor**: DnCNN is fully convolutional with ``padding="same"``
and never changes the spatial size, so any ``H`` and ``W`` work. (The example config's 192 px raster
comes from a METRIC — MS-SSIM's pyramid — not from the model.)

**The constants below are the single source of truth**, imported by each implementation for its
constructor defaults, so "same architecture" is true by construction for the knobs rather than by
several copies agreeing. The LAYER ORDER is prose — different APIs cannot share code — and is held
together instead by :func:`trainable_parameter_count`, which DERIVES the count from the
architecture. Each implementation is checked against that CLAIM rather than against whichever one
was written first.
"""

from typing import Tuple

__all__ = [
    "DEPTH",
    "INPUT_SHAPE",
    "KERNEL_SIZE",
    "WIDTH",
    "trainable_parameter_count",
]

#: Convolution layers, end to end: one head + ``DEPTH - 2`` body layers + one tail. The paper uses
#: 17-20; eight is deep enough that the body dominates the parameter count (as it does at any
#: depth) and shallow enough to train in seconds on CPU, which is what a five-way comparison needs.
DEPTH: int = 8

#: Feature channels in every hidden layer. The paper uses 64.
WIDTH: int = 32

#: Square conv kernel, ``padding="same"`` so the spatial size never changes.
KERNEL_SIZE: int = 3

#: Default input, matching the model wrappers' ``(H, W, C)`` — channels LAST.
INPUT_SHAPE: Tuple[int, int, int] = (192, 192, 3)


def trainable_parameter_count(
    in_channels: int = 3,
    depth: int = DEPTH,
    width: int = WIDTH,
    kernel_size: int = KERNEL_SIZE,
) -> int:
    """The parameter count the spec implies — what every implementation must report.

    Derived from the architecture rather than measured from one framework, so it is a CLAIM the
    implementations are each checked against instead of one of them being the reference.

    Per convolution: ``k*k*cin*cout`` weights plus ``cout`` biases. Per batch norm: ``2*width``
    (scale and shift) — the running mean and variance are STATE, not parameters, and the
    frameworks disagree about whether to count them, so they are excluded here and each
    implementation excludes them the way its own framework requires.

    With the defaults and 3 input channels this is 57,635.
    """
    if depth < 2:
        raise ValueError(f"DnCNN needs at least a head and a tail layer (depth >= 2), got {depth}.")
    k = kernel_size * kernel_size

    # Head: conv -> ReLU. No normalization, per the paper.
    total = k * in_channels * width + width

    # Body: conv -> BN -> ReLU, repeated. This is where a DnCNN's parameters live.
    total += (depth - 2) * (k * width * width + width + 2 * width)

    # Tail: conv back to the input's channel count. It predicts the NOISE; the residual
    # subtraction that turns that into an image costs nothing.
    return total + k * width * in_channels + in_channels
