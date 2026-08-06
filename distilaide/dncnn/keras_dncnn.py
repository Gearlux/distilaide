"""``KerasDnCNN`` — the shared DnCNN spec in ``keras.layers``.

The Keras entry in the comparison :mod:`distilaide.dncnn.spec` describes, and the opposite kind of
model from the zoo wrapper: :class:`distilaide.deepinv_model.DeepInvModel` reaches fourteen
architectures, this one wraps nothing at all. It exists because **deepinv is torch-only** — there is
no Keras restoration zoo to reach for, so without this the Keras backend would have no model and the
cross-framework comparison would stop at torch.

Being written in ``keras.layers`` it runs on whichever engine Keras is bound to, so BOTH the
``keras`` and ``tensorflow`` backends wire it. That is also why it is not called
``TensorFlowDnCNN``: it is not TensorFlow-specific. The genuinely TF-native implementation, in raw
``tf.nn`` ops, is its sibling :mod:`distilaide.dncnn.tensorflow_dncnn`.

**Channels-LAST**, like every Keras model: ``call`` takes ``[N, H, W, C]`` and returns the same
shape. The ``input_shape`` knob is ``(H, W, C)`` in every implementation — here it happens to match
the tensor, where torch's does not.
"""

from typing import Any, Optional, Tuple

from confluid import configurable
from loggair import get_logger

from distilaide._keras import keras
from distilaide.dncnn.spec import DEPTH, INPUT_SHAPE, KERNEL_SIZE, WIDTH, trainable_parameter_count

logger = get_logger(__name__)

__all__ = ["KerasDnCNN"]


@configurable(task="restoration", role="model", lazy=True, framework="keras")
class KerasDnCNN(keras.Model):
    """The shared DnCNN spec (:mod:`distilaide.dncnn.spec`), implemented in ``keras.layers``.

    Args:
        input_shape: ``(H, W, C)`` — channels LAST, which is also this engine's tensor layout.
        depth: Convolution layers end to end (head + body + tail).
        width: Feature channels in every hidden layer.
        kernel_size: Square conv kernel; ``padding="same"`` keeps the spatial size.
        seed: Weight-init seed, for a reproducible comparison.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        depth: int = DEPTH,
        width: int = WIDTH,
        kernel_size: int = KERNEL_SIZE,
        seed: int = 0,
    ) -> None:
        # Lazy: store only, and `super().__init__()` FIRST — a `keras.Layer` refuses attribute
        # assignment before it.
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.depth = int(depth)
        self.width = int(width)
        self.kernel_size = int(kernel_size)
        self.seed = int(seed)
        self._model: Optional[Any] = None

    @property
    def in_channels(self) -> int:
        """The channel count, read from the LAST element of the ``(H, W, C)`` knob."""
        return int(self.input_shape[-1])

    def solidify(self) -> Any:
        """Build (once) and return the live ``keras.Model`` — confluid's ``flow()``-time hook."""
        if self._model is not None:
            return self._model
        channels = self.in_channels
        initializer = keras.initializers.GlorotUniform(seed=self.seed)
        # `(None, None, C)`, NOT the configured `(H, W, C)`. DnCNN is fully convolutional, so it
        # accepts any spatial size — and Keras's Functional API PINS the spatial dims unless they
        # are None, which would make a model built at 192 reject every other raster. Only `C` is
        # fixed, which is also the only part of `input_shape` the parameter count reads.
        inputs = keras.Input(shape=(None, None, channels))

        # Head: conv -> ReLU, deliberately NOT normalized (the spec's shape, and the paper's).
        x = keras.layers.Conv2D(self.width, self.kernel_size, padding="same", kernel_initializer=initializer)(inputs)
        x = keras.layers.ReLU()(x)

        # Body: conv -> BN -> ReLU. The bias stays even though BN makes it redundant — Keras would
        # happily take `use_bias=False`, and that "sensible default" is exactly what would break
        # the five-way parameter match.
        for _ in range(self.depth - 2):
            x = keras.layers.Conv2D(self.width, self.kernel_size, padding="same", kernel_initializer=initializer)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

        # Tail: back to the input's channel count. It predicts the NOISE.
        noise = keras.layers.Conv2D(
            channels, self.kernel_size, padding="same", kernel_initializer=initializer, name="noise"
        )(x)
        # THE residual connection — parameter-free, so it cannot make the implementations disagree.
        outputs = keras.layers.Subtract(name="restored")([inputs, noise])

        self._model = keras.Model(inputs, outputs, name="distilaide_dncnn")
        logger.debug(f"Built the shared DnCNN spec in keras (depth={self.depth}, width={self.width}).")
        return self._model

    @property
    def model(self) -> Any:
        """The live ``keras.Model``, built on first access."""
        return self.solidify()

    def build(self, input_shape: Any = None) -> None:
        """Keras's build hook — the ONE place a subclassed model may create state.

        A network created on the first ``call()`` dies with *"You cannot add new elements of state
        … to a layer that is already built"*, and ``build()`` runs before the first call, which is
        what keeps a never-flowed instance usable.
        """
        self.solidify()
        super().build(input_shape)

    def call(self, x: Any, training: bool = False) -> Any:
        """A restored image from an ``[N, H, W, C]`` batch — channels LAST, same shape out."""
        return self.model(x, training=training)

    @property
    def output_shape(self) -> Any:
        """Functional-only on a subclassed model, so it is forwarded."""
        return self.model.output_shape

    @property
    def trainable_parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`distilaide.dncnn.spec` predicts.

        BatchNorm's moving mean/variance are Keras NON-trainable weights, so reading
        ``trainable_weights`` excludes them by construction — the same exclusion the spec makes,
        reached by a different mechanism than torch's ``requires_grad`` filter.
        """
        return int(sum(int(w.size if hasattr(w, "size") else w.numpy().size) for w in self.model.trainable_weights))

    def spec_parameter_count(self) -> int:
        """The count the SPEC derives for this configuration."""
        return trainable_parameter_count(self.in_channels, self.depth, self.width, self.kernel_size)
