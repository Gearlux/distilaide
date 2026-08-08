"""``TensorFlowRestorer`` — the same restoration run on a native TensorFlow loop.

The fair question this backend answers: why, when ``KERAS_BACKEND=tensorflow distilaide train
--framework keras`` already runs the identical config? Because ``tf.data``, ``tf.GradientTape``
under ``@tf.function`` (which is what makes ``jit_compile``/XLA a knob at all) and the SavedModel
export are TensorFlow's and not Keras's. It is a SIBLING of
:class:`~distilaide.keras_restorer.KerasRestorer`, not a subclass — it narrows a precondition
its would-be parent does not have (the engine guard) — and still declares ``framework="keras"``:
the axis names the API a class is WIRED TO.

Since P5 (2026-08-07) this is ``(RestorationKerasHooks, matrainer.tensorflow.
TensorFlowRunnable)``: the loop machinery — the ``tf.data`` pipeline, the traced steps, the
hand-driven ``CallbackList``, the per-split metric objects (MERGED per split and CLONED per
instance — what owning the loop buys over ``compile()``'s union), the engine guard, the
SavedModel ``_export`` — lives in :class:`matrainer.tensorflow.loop.TensorFlowRunnable`; the
task half in :class:`distilaide.restoration_base.RestorationBase`; and the task's keras
vocabulary (L1 default loss, the PSNR shim + MSE defaults, the one-key prediction contract) in
:class:`distilaide.restoration_keras.RestorationKerasHooks` — shared verbatim with the
compile/fit sibling, which is the point. What stays here is the FULL flat constructor.

**Four behaviours the migration NORMALIZED, each a real fix** (this backend was the fleet's
outlier — it predated the patterns the machinery base pins):

1. **The per-epoch reshuffle now happens** — the old loop never called
   ``sequence.on_epoch_end()``, so every epoch trained on the same row order (invisible: the
   loss still descends).
2. **A ``callbacks:`` slot exists**, and tracking rides the family's
   ``keras_tracking_callback`` instead of a private plain-function hookup — one provenance
   derivation for the whole family.
3. **The export signature is DISCOVERED** (the model's declared ``input_shape``, else a
   measured batch) instead of hard-coded ``(None, None, None, None)``.
4. **The steps close over LOCALS, not ``self``**, the optimizer builds OUTSIDE the graph, and
   ``evaluate()`` scores the VAL split (the fleet harmonization).
"""

from typing import Iterable, List, Literal, Optional, Union

from confluid import Lazy, Mandatory, configurable
from loggair import get_logger
from matrainer.metric_names import MetricNames
from matrainer.tracking.base import Tracker
from recordstream import RecordSource

from distilaide._tensorflow import keras
from distilaide.restoration_keras import RestorationKerasHooks

# AFTER the `_tensorflow` shim import: the shim setdefaults KERAS_BACKEND before anything reads
# it (the matrainer module does the same itself — belt-and-braces, and it keeps this module's
# import story identical to every other distilaide TF-touching module).
from matrainer.tensorflow.loop import TensorFlowNetwork, TensorFlowRunnable  # isort: skip

logger = get_logger(__name__)

#: Which entry point :meth:`TensorFlowRestorer.run` dispatches to.
RestorerTask = Literal["fit", "evaluate", "test", "predict"]

#: A Keras loss / metric / optimizer: a live object, a string alias, or a deferred marker.
KerasLoss = Union[str, keras.losses.Loss]
KerasMetric = Union[str, keras.metrics.Metric]
KerasOptimizer = Union[str, keras.optimizers.Optimizer]

__all__ = [
    "KerasLoss",
    "KerasMetric",
    "KerasOptimizer",
    "RestorerTask",
    "TensorFlowNetwork",
    "TensorFlowRestorer",
]


@configurable(task="restoration", role="trainer", framework="keras")
class TensorFlowRestorer(RestorationKerasHooks, TensorFlowRunnable):
    """Train, evaluate or predict an image restorer with a native TensorFlow loop.

    Args:
        model: A ``keras.Model`` or raw ``tf.Module`` mapping ``[N, H, W, C]`` degraded images
            to restored ones — channels LAST. Wire ``!lazy:distilaide.TensorFlowDnCNN``.
        train_set: Training source (``task="fit"``).
        test_set: Held-out source (``task="test"`` / ``"predict"``).
        data_range: The span of the target images' values; injected into the default PSNR metric.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        jit_compile: Compile the traced steps with XLA.
        prefetch: ``dataset.prefetch(AUTOTUNE)`` between the sequence and the loop.
        export_dir: Where ``fit()`` writes the SavedModel. ``None`` skips the export.
        trackers: matrainer trackers — one or a list.
        callbacks: Keras callbacks, driven by hand through a ``CallbackList``.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the degraded image and the clean target.
        optimizer: Keras optimizer — an object or a name (``"adam"``).
        loss: Keras loss; ``None`` picks ``MeanAbsoluteError`` (L1, matching the torch backends).
        metrics / train_metrics / val_metrics / test_metrics: Keras metrics. This loop MERGES
            them per split (each traced step closes over its own list) — the narrowing the
            compile/fit sibling cannot do.
        metric_names: How recorded metrics are named.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[TensorFlowNetwork]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        data_range: float = 1.0,
        task: RestorerTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        jit_compile: bool = False,
        prefetch: bool = True,
        export_dir: Optional[str] = None,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[Lazy[keras.callbacks.Callback]]] = None,
        experiment_name: str = "distilaide_restoration",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        optimizer: Lazy[KerasOptimizer] = "adam",
        loss: Optional[Lazy[KerasLoss]] = None,
        metrics: Optional[List[Lazy[KerasMetric]]] = None,
        train_metrics: Optional[List[Lazy[KerasMetric]]] = None,
        val_metrics: Optional[List[Lazy[KerasMetric]]] = None,
        test_metrics: Optional[List[Lazy[KerasMetric]]] = None,
        metric_names: Optional[Lazy[MetricNames]] = None,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            jit_compile=jit_compile,
            prefetch=prefetch,
            export_dir=export_dir,
            trackers=trackers,
            callbacks=callbacks,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
        )
        self.data_range = float(data_range)
