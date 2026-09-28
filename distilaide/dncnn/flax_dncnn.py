"""``FlaxDnCNN`` — the shared DnCNN spec in Flax NNX.

The JAX entry in the comparison. Like every other implementation of
:mod:`distilaide.dncnn.spec` it wraps nothing — and here that is not even a choice: **there is no
Flax restoration zoo at all.** ``transformers`` dropped Flax in v5, and nothing else publishes JAX
denoising weights, so a JAX backend without this would have no model.

It IS an ``nnx.Module``, the same decision as every sibling, so the loop hands the slot value
straight to ``nnx.value_and_grad`` / ``nnx.Optimizer`` with nothing in between.

**Channels-LAST**, like the Keras and TensorFlow implementations.

Randomness is EXPLICIT in NNX — the module carries its own ``nnx.Rngs`` — which is what makes the
``seed`` knob mean something here rather than depending on a global.
"""

from typing import Any, Optional, Tuple

from confluid import configurable
from flax import nnx
from loggair import get_logger

from distilaide.dncnn.spec import DEPTH, INPUT_SHAPE, KERNEL_SIZE, WIDTH, trainable_parameter_count

logger = get_logger(__name__)

__all__ = ["FlaxDnCNN"]


class _DnCNN(nnx.Module):
    """The spec's layer graph in ``flax.nnx``: conv+ReLU head, conv+BN+ReLU body, conv tail."""

    def __init__(self, in_channels: int, depth: int, width: int, kernel_size: int, rngs: nnx.Rngs) -> None:
        self.depth = depth
        kernel = (kernel_size, kernel_size)
        # `nnx.List` rather than a plain list: NNX treats a plain Python list of Modules as static
        # structure, so the parameters inside would never reach `nnx.Optimizer`.
        convs = [nnx.Conv(in_channels, width, kernel, padding="SAME", rngs=rngs)]
        norms = []
        for _ in range(depth - 2):
            convs.append(nnx.Conv(width, width, kernel, padding="SAME", rngs=rngs))
            norms.append(nnx.BatchNorm(width, rngs=rngs))
        convs.append(nnx.Conv(width, in_channels, kernel, padding="SAME", rngs=rngs))
        self.convs = nnx.List(convs)
        self.norms = nnx.List(norms)

    def __call__(self, x: Any, training: bool = True) -> Any:
        inputs = x
        for index in range(self.depth):
            x = self.convs[index](x)
            if index == self.depth - 1:
                break  # the tail is a bare convolution
            if index > 0:
                # NNX carries batch-norm's train/eval choice as a per-CALL argument, unlike
                # torch's and MLX's module state.
                x = self.norms[index - 1](x, use_running_average=not training)
            x = nnx.relu(x)
        # THE residual connection: the stack predicted the NOISE.
        return inputs - x


@configurable(task="restoration", role="model", lazy=True, framework="jax")
class FlaxDnCNN(nnx.Module):
    """The shared DnCNN spec (:mod:`distilaide.dncnn.spec`), implemented in ``flax.nnx``.

    Args:
        input_shape: ``(H, W, C)`` — channels LAST, which is also this engine's tensor layout.
        depth: Convolution layers end to end (head + body + tail).
        width: Feature channels in every hidden layer.
        kernel_size: Square conv kernel; ``padding="SAME"`` keeps the spatial size.
        seed: Seed for the module's own ``nnx.Rngs`` — explicit randomness, so this reproduces.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        depth: int = DEPTH,
        width: int = WIDTH,
        kernel_size: int = KERNEL_SIZE,
        seed: int = 0,
    ) -> None:
        self.input_shape = tuple(input_shape)
        self.depth = int(depth)
        self.width = int(width)
        self.kernel_size = int(kernel_size)
        self.seed = int(seed)
        # `nnx.data(None)` and NOT a bare `None`: an attribute first assigned `None` becomes STATIC
        # in NNX's pytree, and assigning a Module to it later RAISES *"Cannot assign data value …
        # to static attribute"*. Wrapping it is what keeps zero-arg construction compatible with
        # building at `flow()` time.
        self._model: Optional[_DnCNN] = nnx.data(None)

    @property
    def in_channels(self) -> int:
        """The channel count, read from the LAST element of the ``(H, W, C)`` knob."""
        return int(self.input_shape[-1])

    def solidify(self) -> "_DnCNN":
        """Build (once) and return the live module — confluid's ``flow()``-time hook."""
        if self._model is None:
            self._model = _DnCNN(self.in_channels, self.depth, self.width, self.kernel_size, nnx.Rngs(params=self.seed))
            logger.debug(f"Built the shared DnCNN spec in flax.nnx (depth={self.depth}, width={self.width}).")
        return self._model

    @property
    def model(self) -> "_DnCNN":
        """The live module, built on first access."""
        return self.solidify()

    def __call__(self, x: Any, training: bool = True) -> Any:
        """A restored image from an ``[N, H, W, C]`` batch — channels LAST, same shape out."""
        return self.model(x, training=training)

    @property
    def trainable_parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`distilaide.dncnn.spec` predicts.

        Flax's exclusion mechanism is a fourth one: batch norm's running statistics are
        ``nnx.BatchStat`` — a different variable TYPE in the same state tree — so counting
        ``nnx.Param`` ONLY excludes them, where torch filters buffers by ``requires_grad``, Keras
        reads ``trainable_weights`` and raw TensorFlow declares ``trainable=False``.
        """
        import jax

        params = nnx.state(self.model, nnx.Param)
        return int(sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params)))

    def spec_parameter_count(self) -> int:
        """The count the SPEC derives for this configuration."""
        return trainable_parameter_count(self.in_channels, self.depth, self.width, self.kernel_size)
