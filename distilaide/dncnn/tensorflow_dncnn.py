"""``TensorFlowDnCNN`` — the shared DnCNN spec in raw ``tf.nn`` ops.

The genuinely TensorFlow-native entry in the comparison, and the sibling of
:mod:`distilaide.dncnn.keras_dncnn`: that one is written in ``keras.layers`` and runs on whichever
engine Keras is bound to, this one is a ``tf.Module`` holding its own ``tf.Variable``s and calling
``tf.nn`` directly. Either can be wired into either non-torch backend; this is what "TensorFlow
native" costs, spelled out.

BatchNorm is written out rather than borrowed, because borrowing it would mean
``keras.layers.BatchNormalization`` and then this would be a Keras model wearing a TensorFlow name.
Its moving statistics are ``trainable=False`` variables, so they are STATE rather than parameters —
which is what lets the parameter count match the other implementations, since the frameworks
disagree about whether to count them.
"""

from typing import Any, List, Optional, Tuple

from confluid import configurable
from loggair import get_logger

# `distilaide._tensorflow` pins KERAS_BACKEND=tensorflow BEFORE importing keras; importing `tf`
# through it keeps that ordering in one place.
from distilaide._tensorflow import tf
from distilaide.dncnn.spec import DEPTH, INPUT_SHAPE, KERNEL_SIZE, WIDTH, trainable_parameter_count

logger = get_logger(__name__)

__all__ = ["TensorFlowDnCNN"]


def _glorot_uniform(shape: Tuple[int, ...], seed: int) -> Any:
    """Glorot-uniform weights from ``tf.random``, written out rather than taken from an initializer.

    **`tf.initializers` IS Keras**, not TensorFlow: since TF 2.16 `tf.keras` is Keras 3, and
    `tf.initializers.GlorotUniform` resolves to `keras.initializers.GlorotUniform`. Called in a
    process where Keras is bound to the TORCH backend — which is exactly this workspace, since
    `recordstream.keras` defaults `KERAS_BACKEND` to the first engine installed and torch wins — it
    returns a **torch tensor**, and `tf.Variable` then dies converting an MPS tensor to numpy
    (*"can't convert mps:0 device type tensor to numpy"*). So a module claiming to be raw `tf.nn`
    cannot reach for `tf.initializers` at all, and the lines below are what "raw TensorFlow"
    actually costs.

    ``stateless_uniform`` rather than ``tf.random.uniform``: the seed is the whole argument for a
    reproducible model, and the stateful op reads a GLOBAL seed that another library can move.

    The limit is Glorot's: ``sqrt(6 / (fan_in + fan_out))``, with the fans computed exactly as
    ``keras.src.initializers.random_initializers.compute_fans`` does — so the weights this produces
    are drawn from the same distribution the other implementations' initializers use, and "the same
    network" stays true of more than the parameter COUNT.
    """
    receptive_field = 1
    for dim in shape[:-2]:
        receptive_field *= int(dim)
    fan_in, fan_out = int(shape[-2]) * receptive_field, int(shape[-1]) * receptive_field
    limit = (6.0 / float(fan_in + fan_out)) ** 0.5
    return tf.random.stateless_uniform(shape, seed=(seed, 0), minval=-limit, maxval=limit, dtype=tf.float32)


class _DnCNN(tf.Module):
    """The spec's layer graph: conv+ReLU head, conv+BN+ReLU body, conv tail, residual subtract."""

    def __init__(
        self,
        in_channels: int,
        depth: int,
        width: int,
        kernel_size: int,
        seed: int,
        momentum: float = 0.99,
    ) -> None:
        super().__init__(name="distilaide_dncnn")
        self.depth = depth
        self.momentum = momentum
        self.kernels: List[Any] = []
        self.biases: List[Any] = []
        # Batch norm lives on the BODY layers only — indices 1 .. depth-2 — so these lists are
        # shorter than the conv lists by exactly two. Keeping them separate rather than padding
        # with None is what makes `trainable_variables` come out right without a filter.
        self.gammas: List[Any] = []
        self.betas: List[Any] = []
        self.moving_means: List[Any] = []
        self.moving_vars: List[Any] = []

        channels = in_channels
        for index in range(depth):
            out_channels = in_channels if index == depth - 1 else width
            kernel = _glorot_uniform((kernel_size, kernel_size, channels, out_channels), seed=seed + index)
            self.kernels.append(tf.Variable(kernel, name=f"kernel_{index}"))
            self.biases.append(tf.Variable(tf.zeros((out_channels,)), name=f"bias_{index}"))
            if 0 < index < depth - 1:
                self.gammas.append(tf.Variable(tf.ones((width,)), name=f"gamma_{index}"))
                self.betas.append(tf.Variable(tf.zeros((width,)), name=f"beta_{index}"))
                self.moving_means.append(tf.Variable(tf.zeros((width,)), trainable=False, name=f"moving_mean_{index}"))
                self.moving_vars.append(tf.Variable(tf.ones((width,)), trainable=False, name=f"moving_var_{index}"))
            channels = out_channels

    def __call__(self, x: Any, training: bool = False) -> Any:
        inputs = x
        for index in range(self.depth):
            x = tf.nn.conv2d(x, self.kernels[index], strides=1, padding="SAME") + self.biases[index]
            if index == self.depth - 1:
                break  # the tail is a bare convolution — no norm, no activation
            if index > 0:
                body = index - 1  # the BN lists skip the head
                if training:
                    # Batch statistics while training, folded into the moving estimates the eval
                    # path uses — the update Keras/torch do for you inside their layer.
                    mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
                    self.moving_means[body].assign(
                        self.moving_means[body] * self.momentum + mean * (1.0 - self.momentum)
                    )
                    self.moving_vars[body].assign(
                        self.moving_vars[body] * self.momentum + variance * (1.0 - self.momentum)
                    )
                else:
                    mean, variance = self.moving_means[body], self.moving_vars[body]
                x = tf.nn.batch_normalization(
                    x, mean, variance, self.betas[body], self.gammas[body], variance_epsilon=1e-3
                )
            x = tf.nn.relu(x)
        # THE residual connection: the stack predicted the NOISE.
        return inputs - x


@configurable(task="restoration", role="model", lazy=True, framework="keras")
class TensorFlowDnCNN(tf.Module):
    """The shared DnCNN spec (:mod:`distilaide.dncnn.spec`), in raw ``tf.nn`` ops.

    Tagged ``framework="keras"``, NOT ``"tensorflow"``: the axis names the API a class can be WIRED
    TO, and the only trainer that can consume this is
    :class:`~distilaide.tensorflow_restorer.TensorFlowRestorer`, which declares ``"keras"`` for the
    same reason. A truer-looking tag would hide it from the one picker that should offer it.

    Args:
        input_shape: ``(H, W, C)`` — channels LAST, which is also this engine's tensor layout.
        depth: Convolution layers end to end (head + body + tail).
        width: Feature channels in every hidden layer.
        kernel_size: Square conv kernel; ``padding="SAME"`` keeps the spatial size.
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
        super().__init__(name="distilaide_dncnn_wrapper")
        self.input_shape = tuple(input_shape)
        self.depth = int(depth)
        self.width = int(width)
        self.kernel_size = int(kernel_size)
        self.seed = int(seed)
        self._model: Optional[_DnCNN] = None

    @property
    def in_channels(self) -> int:
        """The channel count, read from the LAST element of the ``(H, W, C)`` knob."""
        return int(self.input_shape[-1])

    def solidify(self) -> "_DnCNN":
        """Build (once) and return the live module — confluid's ``flow()``-time hook."""
        if self._model is None:
            self._model = _DnCNN(self.in_channels, self.depth, self.width, self.kernel_size, self.seed)
            logger.debug(f"Built the shared DnCNN spec in raw tf.nn (depth={self.depth}, width={self.width}).")
        return self._model

    @property
    def model(self) -> "_DnCNN":
        """The live module, built on first access."""
        return self.solidify()

    def __call__(self, x: Any, training: bool = False) -> Any:
        """A restored image from an ``[N, H, W, C]`` batch — channels LAST, same shape out."""
        return self.model(x, training=training)

    @property
    def trainable_variables(self) -> Any:  # type: ignore[override]
        """The variables an optimizer updates — forwarded from the built module."""
        return self.model.trainable_variables

    @property
    def trainable_parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`distilaide.dncnn.spec` predicts.

        Raw TensorFlow's exclusion mechanism is a third one: the moving statistics are declared
        ``trainable=False``, so ``trainable_variables`` skips them — where torch filters on
        ``requires_grad`` and Keras reads ``trainable_weights``.
        """
        return int(sum(int(tf.size(v).numpy()) for v in self.trainable_variables))

    def spec_parameter_count(self) -> int:
        """The count the SPEC derives for this configuration."""
        return trainable_parameter_count(self.in_channels, self.depth, self.width, self.kernel_size)
