"""``MlxDnCNN`` — the shared DnCNN spec in Apple's MLX.

The fifth and last implementation of :mod:`distilaide.dncnn.spec`, and like the Flax one it wraps
nothing because there is nothing to wrap: MLX has no restoration zoo.

**Channels-LAST**, like the Keras / TensorFlow / Flax implementations.

Two MLX facts shape this file, and both are the reason the obvious spelling is wrong:

* **``nn.Module.valid_parameter_filter`` ends in ``and not key.startswith("_")``**, so the built
  network is held in the PUBLIC attribute ``net``. Named ``_model`` — which is what its Keras, TF
  and Flax twins call it — the whole network is filtered out of ``parameters()``, an optimizer sees
  an empty tree, and the loop applies gradients to nothing FOREVER without raising.
* **Batch norm's running statistics live in the SAME parameter tree as the weights**, unlike every
  other framework here, so excluding them from the count is a NAME filter rather than a type or
  flag filter. Without it the five-way match breaks.
"""

from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from confluid import configurable
from loggair import get_logger

from distilaide.dncnn.spec import DEPTH, INPUT_SHAPE, KERNEL_SIZE, WIDTH, trainable_parameter_count

logger = get_logger(__name__)

__all__ = ["MlxDnCNN"]

#: Parameter-tree keys that are batch-norm STATE rather than parameters. MLX keeps them beside the
#: weights, so the spec's exclusion has to be done by name here.
_RUNNING_STATS = ("running_mean", "running_var")


class _DnCNN(nn.Module):
    """The spec's layer graph in ``mlx.nn``: conv+ReLU head, conv+BN+ReLU body, conv tail."""

    def __init__(self, in_channels: int, depth: int, width: int, kernel_size: int) -> None:
        super().__init__()
        self.depth = depth
        convs = [nn.Conv2d(in_channels, width, kernel_size, padding=kernel_size // 2)]
        norms = []
        for _ in range(depth - 2):
            convs.append(nn.Conv2d(width, width, kernel_size, padding=kernel_size // 2))
            norms.append(nn.BatchNorm(width))
        convs.append(nn.Conv2d(width, in_channels, kernel_size, padding=kernel_size // 2))
        self.convs = convs
        self.norms = norms

    def __call__(self, x: Any) -> Any:
        inputs = x
        for index in range(self.depth):
            x = self.convs[index](x)
            if index == self.depth - 1:
                break  # the tail is a bare convolution
            if index > 0:
                # MLX carries train/eval as module STATE (`.train(bool)`), not a per-call argument
                # — which is why the restorer sets the mode once per phase rather than per forward.
                x = self.norms[index - 1](x)
            x = nn.relu(x)
        # THE residual connection: the stack predicted the NOISE.
        return inputs - x


@configurable(task="restoration", role="model", lazy=True, framework="mlx")
class MlxDnCNN(nn.Module):
    """The shared DnCNN spec (:mod:`distilaide.dncnn.spec`), implemented in ``mlx.nn``.

    Args:
        input_shape: ``(H, W, C)`` — channels LAST, which is also this engine's tensor layout.
        depth: Convolution layers end to end (head + body + tail).
        width: Feature channels in every hidden layer.
        kernel_size: Square conv kernel; ``padding`` keeps the spatial size.
        seed: Weight-init seed, for a reproducible comparison.

    Attributes:
        net: The built network. **PUBLIC on purpose** — see the module docstring: MLX filters
            underscore-prefixed attributes out of ``parameters()``, so a private name would hand
            the optimizer an empty tree and train nothing, silently.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        depth: int = DEPTH,
        width: int = WIDTH,
        kernel_size: int = KERNEL_SIZE,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.depth = int(depth)
        self.width = int(width)
        self.kernel_size = int(kernel_size)
        self.seed = int(seed)
        self.net: Optional[_DnCNN] = None

    @property
    def in_channels(self) -> int:
        """The channel count, read from the LAST element of the ``(H, W, C)`` knob."""
        return int(self.input_shape[-1])

    def solidify(self) -> "_DnCNN":
        """Build (once) and return the live module — confluid's ``flow()``-time hook."""
        if self.net is None:
            mx.random.seed(self.seed)
            self.net = _DnCNN(self.in_channels, self.depth, self.width, self.kernel_size)
            logger.debug(f"Built the shared DnCNN spec in mlx.nn (depth={self.depth}, width={self.width}).")
        return self.net

    @property
    def model(self) -> "_DnCNN":
        """The live module, built on first access."""
        return self.solidify()

    def __call__(self, x: Any) -> Any:
        """A restored image from an ``[N, H, W, C]`` batch — channels LAST, same shape out."""
        return self.model(x)

    @property
    def trainable_parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`distilaide.dncnn.spec` predicts.

        MLX's exclusion mechanism is the fifth and the odd one out: batch norm's ``running_mean``
        and ``running_var`` sit in the SAME tree as the weights, so they are filtered BY NAME.
        Every other framework distinguishes them structurally (a torch buffer, a Keras
        non-trainable weight, a ``trainable=False`` TF variable, an ``nnx.BatchStat``).
        """
        self.solidify()

        def _count(tree: Any, key: str = "") -> int:
            if isinstance(tree, dict):
                return sum(_count(v, k) for k, v in tree.items())
            if isinstance(tree, (list, tuple)):
                return sum(_count(v, key) for v in tree)
            if key in _RUNNING_STATS:
                return 0
            return int(tree.size)

        return _count(self.parameters())

    def spec_parameter_count(self) -> int:
        """The count the SPEC derives for this configuration."""
        return trainable_parameter_count(self.in_channels, self.depth, self.width, self.kernel_size)
