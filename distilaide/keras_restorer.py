"""``KerasRestorer`` — the same restoration run on Keras 3.

Since P5 (2026-08-07) this is ``(RestorationKerasHooks, matrainer.keras.KerasRunnable)``: the
compile/fit machinery — build + compile, the ``History`` read-back, the entry points, the
``RecordSequence`` batch boundary, the tracking callback — lives in
:class:`matrainer.keras.trainer.KerasRunnable`, the task half (split preparation, the
trainable-model gate) in :class:`distilaide.restoration_base.RestorationBase`, and the task's
KERAS VOCABULARY (the L1 default loss, the PSNR + MSE default metrics, the one-key prediction
contract) in :class:`distilaide.restoration_keras.RestorationKerasHooks` — written once for
this concrete AND its native-TensorFlow sibling. What stays here is the FULL flat constructor
(the confluid introspection surface).

**Three real differences from the torch backends, all about the ENGINE rather than the task:**

1. **Keras is channels-LAST**, so a Keras run's ``preprocess`` chain is the shared one MINUS its
   two ``ToTensor`` steps (which produce CHW) — the family's default ``_to_arrays`` (float32 in,
   float32 target) IS this task's batch shape, so this module overrides nothing there.
2. **The model is the shared spec, not the zoo.** deepinv is torch-ONLY, so there is nothing to
   wrap here; ``KerasDnCNN`` (or ``TensorFlowDnCNN``) is what a Keras config wires. That is the
   whole reason :mod:`distilaide.dncnn` exists.
3. **The metric vocabulary is narrower, and honestly so.** ``keras.metrics`` ships no
   image-quality metric at all; ``keras.ops.psnr`` exists, so PSNR is available through this
   project's one shim (:class:`distilaide.keras_restoration.PeakSignalNoiseRatio`), but there
   is no SSIM anywhere in ``keras.ops`` and the only implementations are engine-SPECIFIC. So a
   Keras run reports PSNR and the pixel errors; SSIM stays a torch-backend column.

``metrics`` / ``train_metrics`` / ``val_metrics`` / ``test_metrics`` are UNIONED (the family's
``_compiled_metrics``): ``keras.Model.compile`` takes ONE list and reports it on train AND
validation — see the machinery base for the whole argument.

Keras 3 picks its engine from ``KERAS_BACKEND`` and this module never sets it.
"""

from typing import Iterable, List, Literal, Optional, Union

from confluid import Lazy, Mandatory, configurable
from loggair import get_logger
from matrainer.keras.runnable import KerasLoss, KerasMetric, KerasOptimizer
from matrainer.keras.trainer import KerasRunnable
from matrainer.metric_names import MetricNames
from matrainer.tracking.base import Tracker
from recordstream import RecordSource
from recordstream.keras import keras

from distilaide.restoration_keras import RestorationKerasHooks

logger = get_logger(__name__)

#: Which entry point :meth:`KerasRestorer.run` dispatches to.
RestorerTask = Literal["fit", "evaluate", "test", "predict"]

__all__ = ["KerasLoss", "KerasMetric", "KerasOptimizer", "KerasRestorer", "RestorerTask"]


@configurable(task="restoration", role="trainer", framework="keras")
class KerasRestorer(RestorationKerasHooks, KerasRunnable):
    """Train, evaluate or predict an image restorer with Keras 3.

    Args:
        model: A ``keras.Model`` mapping ``[N, H, W, C]`` degraded images to restored ones —
            channels LAST. Wire ``!lazy:distilaide.KerasDnCNN``.
        train_set: Training source (``task="fit"``).
        test_set: Held-out source (``task="test"`` / ``"predict"``).
        data_range: The span of the target images' values; injected into the default PSNR metric.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        trackers: matrainer trackers — one or a list.
        callbacks: Keras callbacks.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the degraded image and the clean target.
        optimizer: Keras optimizer — an object or a name (``"adam"``).
        loss: Keras loss; ``None`` picks ``MeanAbsoluteError`` (L1, matching the torch backends).
        metrics / train_metrics / val_metrics / test_metrics: Keras metrics. All four are UNIONED
            — see the machinery base.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. This engine has a real translation to do:
            Keras writes ``loss`` / ``val_loss`` where the canonical form is ``train/loss`` /
            ``val/loss``.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[keras.Model]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        data_range: float = 1.0,
        task: RestorerTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
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
