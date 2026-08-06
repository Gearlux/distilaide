"""``KerasRestorer`` — the same restoration run on Keras 3.

The FOURTH backend and the first that is not torch. Everything upstream of the model is
recordstream's unchanged, INCLUDING the batching: :class:`recordstream.keras.RecordSequence` is the
``keras.utils.PyDataset`` a ``DataLoader`` would have been, and ``_to_arrays`` is handed to it
exactly as a ``collate_fn`` is handed to a ``DataLoader``.

**Three real differences from the torch backends, all about the ENGINE rather than the task:**

1. **Keras is channels-LAST**, so a Keras run's ``preprocess`` chain is the shared one MINUS its two
   ``ToTensor`` steps (which produce CHW). Both halves of the pair change together — unlike the
   segmentation task, where only the image line differed and the mask was engine-neutral.
2. **The model is the shared spec, not the zoo.** deepinv is torch-ONLY, so there is nothing to
   wrap here; ``KerasDnCNN`` (or ``TensorFlowDnCNN``) is what a Keras config wires. That is the
   whole reason :mod:`distilaide.dncnn` exists.
3. **The metric vocabulary is narrower, and honestly so.** ``keras.metrics`` ships no image-quality
   metric at all; ``keras.ops.psnr`` exists, so PSNR is available through this project's one shim
   (:class:`distilaide.keras_restoration.PeakSignalNoiseRatio`), but there is no SSIM anywhere in
   ``keras.ops`` and the only implementations are engine-SPECIFIC. So a Keras run reports PSNR and
   the pixel errors; SSIM stays a torch-backend column.

``metrics`` / ``train_metrics`` / ``val_metrics`` / ``test_metrics`` are UNIONED here, not merged
(:func:`matrainer.metric_slots.union_metric_specs`): ``keras.Model.compile`` takes ONE list and
reports it on train AND validation, and there is no hook that narrows a metric to one split short of
recompiling between phases — which would discard the optimizer state a fit is using. Widening is the
honest failure; DROPPING a wired metric would be one that silently reports nothing.

Keras 3 picks its engine from ``KERAS_BACKEND`` and this module never sets it.
"""

from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import numpy as np
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.metric_slots import union_metric_specs
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.tracking.base import Tracker, as_tracker_list
from recordstream import (
    PredictionsSink,
    RecordSource,
    RestorationOutput,
    batch_metadata,
    batch_values,
    ensure_materialized,
    ensure_record_dataset,
)
from recordstream.runnable import ProgressReporting, entrypoint, run_entrypoint

from distilaide._keras import RecordSequence, keras
from distilaide.keras_restoration import PeakSignalNoiseRatio

logger = get_logger(__name__)

#: Which entry point :meth:`KerasRestorer.run` dispatches to.
RestorerTask = Literal["fit", "evaluate", "test", "predict"]

#: A Keras loss: a live object, a string alias (``"mae"``), or a deferred marker. Named so
#: `to_pydantic` has a real type to build navigaitor's form-spec / StreamStudio widgets from.
KerasLoss = Union[str, keras.losses.Loss]

#: A Keras metric, same three shapes.
KerasMetric = Union[str, keras.metrics.Metric]

#: A Keras optimizer, same three shapes.
KerasOptimizer = Union[str, keras.optimizers.Optimizer]

__all__ = ["KerasLoss", "KerasMetric", "KerasOptimizer", "KerasRestorer", "RestorerTask"]


@configurable(task="restoration", role="trainer", framework="keras")
class KerasRestorer(ProgressReporting):
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
            — see the module docstring.
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
        super().__init__()
        # `keras.Model`, not `Lazy[keras.Model]`: the DEFERRED-ness is declared once, on the ctor
        # param above, which is what `lazy_param_names` / `to_pydantic` read. Restating it here
        # would make the attribute a union no method could call through.
        self.model: keras.Model = cast(keras.Model, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: RestorerTask = task
        self.data_range = float(data_range)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.optimizer: Lazy[KerasOptimizer] = optimizer
        # Sentinels: a live default could not be told apart from a user's explicit choice.
        self.loss: Optional[Lazy[KerasLoss]] = loss
        self.metrics: Optional[List[Lazy[KerasMetric]]] = metrics
        self.train_metrics: Optional[List[Lazy[KerasMetric]]] = train_metrics
        self.val_metrics: Optional[List[Lazy[KerasMetric]]] = val_metrics
        self.test_metrics: Optional[List[Lazy[KerasMetric]]] = test_metrics
        # Stored VERBATIM (lazy-init): entries may be deferred markers, flowed where used.
        self.callbacks: Optional[List[Lazy[keras.callbacks.Callback]]] = callbacks
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Prepare the sources, then build and compile the Keras model."""
        from distilaide._keras import keras as _keras  # noqa: F401 - keeps the engine pinned early

        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        self._compile()

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process.

        ``ensure_materialized`` is the fork guard: a ``PyDataset`` may be read by worker processes,
        and on a machine where anything has set the ``fork`` start method (importing fastai does,
        process-wide) a lazy source first built in the child dies with a SIGSEGV and no traceback.
        """
        if source is None:
            return None
        return ensure_materialized(ensure_record_dataset(source))

    def _resolve_task_defaults(self) -> None:
        """Fill the unset (``None``) task-shaped slots. Idempotent; never overwrites.

        The loss is L1, the same choice the torch backends make — a config moving between engines
        must not silently change its objective. The metrics are PSNR (through this project's one
        shim, since ``keras.metrics`` has none) plus the mean squared error; SSIM is absent because
        ``keras.ops`` has no SSIM and an engine-specific one would make the two Keras-family
        backends disagree from the same config.
        """
        if self.loss is None:
            self.loss = keras.losses.MeanAbsoluteError()
        if self.metrics is None:
            self.metrics = [
                PeakSignalNoiseRatio(data_range=self.data_range, name="psnr"),
                keras.metrics.MeanSquaredError(name="mse"),
            ]

    def _compile(self) -> None:
        """Build the model, then compile it.

        The compile slots are flowed at USE: a nested ``!class:`` in a constructor slot arrives as a
        deferred ``Class`` marker, and Keras takes an unbuilt marker for a custom callable and fails
        deep inside ``compile`` rather than saying what is wrong. A bare ``flow()`` is the whole
        guard — it returns a live object, a string alias and a list unchanged.
        """
        self._resolve_task_defaults()
        self.model = flow(self.model)
        self.model.compile(
            optimizer=flow(self.optimizer),
            loss=flow(self.loss),
            metrics=[flow(m) for m in self._compiled_metrics()],
        )

    def _compiled_metrics(self) -> Sequence[Lazy[KerasMetric]]:
        """The ONE metric list this engine can compute — the UNION of the shared and split slots.

        ``keras.Model.compile`` takes a single list and reports it on training AND validation
        (prefixed ``val_``); there is no hook that narrows a metric to one split short of
        recompiling between phases, which would discard the optimizer state a fit is using. So a
        metric wired to ``val_metrics`` is computed on the training split too.

        Widening is the honest failure: a metric appearing where it was not asked for is noise a
        reader ignores, where DROPPING it would be a wired metric that silently reports nothing.
        """
        return union_metric_specs(self.metrics, self.train_metrics, self.val_metrics, self.test_metrics) or []

    # ---- the batch boundary -------------------------------------------------- #
    def _sequence(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> RecordSequence:
        """recordstream's batching adapter, wearing THIS task's batch shape."""
        return RecordSequence(
            source,
            batch_size=self.batch_size,
            shuffle=shuffle,
            transform=partial(self._to_arrays, with_targets=with_targets),
        )

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → the numpy ``(x, y)`` Keras consumes.

        The Keras twin of the Lightning backend's ``_targets``, and the shortest in the fleet:
        recordstream hands over numpy already, and for an image-to-image task BOTH halves are the
        same kind of value — so there is no dtype branch and no shaping, only a float32 cast.

        **Channels-LAST**, so a Keras config's preprocessing chain stops before the two ``ToTensor``
        steps a torch chain ends with.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="float32")

    def _callbacks(self) -> List[Any]:
        """Wired Keras callbacks plus OPTIONAL matrainer tracking (a few lines; removable)."""
        callbacks = [flow(cb) for cb in (self.callbacks or [])]
        for callback in callbacks:
            # Keras's file-writing callbacks open their path without creating its parent, so a
            # perfectly reasonable `./keras_logs/run.csv` dies with FileNotFoundError several
            # minutes into a fit — after the data walk and the model build.
            target = getattr(callback, "filename", None) or getattr(callback, "filepath", None)
            if isinstance(target, (str, Path)):
                Path(target).expanduser().parent.mkdir(parents=True, exist_ok=True)
        trackers = self._live_trackers()
        if trackers:
            from matrainer.keras import keras_tracking_callback

            # `runnable=self` rather than a params dict: matrainer derives the hyperparameters
            # from the flowed object graph AND dumps the resolved config.
            callbacks.append(
                keras_tracking_callback(trackers, runnable=self, metric_names=as_metric_names(self.metric_names))
            )
        return callbacks

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"distilaide keras training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("KerasRestorer.fit() requires a 'train_set'.")
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None
        history = self.model.fit(
            train_seq, validation_data=val_seq, epochs=self.max_epochs, callbacks=self._callbacks()
        )
        # The HISTORY keeps Keras's own names — it is Keras's object, and a `CSVLogger` beside it
        # writes those same columns, so a wired `EarlyStopping(monitor="val_loss")` must keep
        # finding them. What LEAVES the run is translated, so it reads beside another engine's.
        self._history = {k: [float(x) for x in v] for k, v in history.history.items()}
        self._computed_metrics = as_metric_names(self.metric_names)(
            {k: v[-1] for k, v in self._history.items()}, source="keras"
        )
        logger.info(f"Training complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("KerasRestorer.test() requires a 'test_set'.")
        results = self.model.evaluate(self._sequence(self.test_set), return_dict=True, callbacks=self._callbacks())
        # `split="test"`: `evaluate()` returns UNPREFIXED keys whichever split it ran on, so the
        # caller states which one this was — Keras cannot.
        self._computed_metrics = as_metric_names(self.metric_names)(
            {k: float(v) for k, v in results.items()}, source="keras", split="test"
        )
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        if self.test_set is None:
            raise ValueError("KerasRestorer.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            restored = self.model.predict(sequence[index][0], verbose=0)
            if sink is not None:
                self._write_predictions(np.asarray(restored), batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, restored: np.ndarray, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.RestorationOutput` per record, paired with its metadata.

        The contract is recordstream's and framework-neutral — generic in its array type — so the
        SAME sink that consumes the torch backends' output consumes this numpy one with no adapter.
        Unlike the classification and segmentation tasks there is nothing to derive here: the model
        emitted the answer, and the contract's one key carries it.
        """
        per_record_meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
        for i in range(len(restored)):
            out: RestorationOutput[np.ndarray] = {"image": restored[i]}
            meta = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            sink.write(out, meta)

    # ---- tracking ------------------------------------------------------------ #
    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot."""
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def trained_model(self) -> Any:
        """The live trained model (post-run ``@output``). Raises while still deferred."""
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run() — the model is still deferred.")
        return self.model

    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``."""
        if not self._computed_metrics:
            raise RuntimeError("computed_metrics is only available after run().")
        return dict(self._computed_metrics)

    @property
    @output
    def history(self) -> Dict[str, List[float]]:
        """Per-epoch metric history, as Keras's ``History`` recorded it."""
        return {key: list(values) for key, values in self._history.items()}
