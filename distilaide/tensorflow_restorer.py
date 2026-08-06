"""``TensorFlowRestorer`` — the same restoration run as a native TensorFlow training loop.

The fair question this backend answers: why, when ``KERAS_BACKEND=tensorflow distilaide train
--framework keras`` already runs the identical config? Because ``tf.data``, ``tf.GradientTape``
under ``@tf.function`` (which is what makes ``jit_compile``/XLA a knob at all) and the SavedModel
export are TensorFlow's and not Keras's. Everything else — the data, the losses, the metrics — IS
Keras 3, which is also why ``--framework tensorflow`` imports :mod:`distilaide.keras_restoration`
and there is deliberately no second loss module.

It is a SIBLING of :class:`~distilaide.keras_restorer.KerasRestorer`, not a subclass, and the reason
is a property of the code rather than taste: it **narrows a precondition** its would-be parent does
not have — ``_require_tensorflow_backend()`` raises where a ``KerasRestorer`` runs fine — so a picker
offering "any KerasRestorer" would hand out something that cannot run. It still declares
``framework="keras"``: the axis names the API a class is WIRED TO, and every one of its slots holds
a ``keras.losses`` / ``keras.metrics`` / ``keras.Model``.

Two ordering rules are load-bearing and both live in :mod:`distilaide._tensorflow`: it
``setdefault``s ``KERAS_BACKEND=tensorflow`` BEFORE importing anything that reads it, and it
re-exports ``keras`` + ``RecordSequence`` so isort cannot sort a ``recordstream.keras`` line above
it.
"""

from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union, cast

import numpy as np
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.metric_slots import merge_metric_specs
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.tracking.base import Tracker, as_tracker_list
from matrainer.tracking.provenance import capture_log_file, log_run_context
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

from distilaide._tensorflow import RecordSequence, keras, keras_backend, tf
from distilaide.keras_restoration import PeakSignalNoiseRatio

logger = get_logger(__name__)

#: Which entry point :meth:`TensorFlowRestorer.run` dispatches to.
RestorerTask = Literal["fit", "evaluate", "test", "predict"]

#: A Keras loss / metric / optimizer: a live object, a string alias, or a deferred marker.
KerasLoss = Union[str, keras.losses.Loss]
KerasMetric = Union[str, keras.metrics.Metric]
KerasOptimizer = Union[str, keras.optimizers.Optimizer]

#: The model types this loop can drive. A `keras.Model` descends `Layer -> TFLayer ->
#: KerasAutoTrackable`, so it is NOT a `tf.Module`; their only shared ancestor is a private
#: internal, which is why the honest annotation is a spelled-out union.
TensorFlowNetwork = Union[keras.Model, tf.Module]

__all__ = ["KerasLoss", "KerasMetric", "KerasOptimizer", "RestorerTask", "TensorFlowNetwork", "TensorFlowRestorer"]


@configurable(task="restoration", role="trainer", framework="keras")
class TensorFlowRestorer(ProgressReporting):
    """Train, evaluate or predict an image restorer with a native TensorFlow loop.

    Args:
        model: A ``keras.Model`` or a raw ``tf.Module`` mapping ``[N, H, W, C]`` degraded images to
            restored ones. Wire ``!lazy:distilaide.TensorFlowDnCNN`` for the TF-native spec.
        train_set: Training source (``task="fit"``).
        test_set: Held-out source (``task="test"`` / ``"predict"``).
        data_range: The span of the target images' values; injected into the default PSNR metric.
        task: Which entry point ``run()`` dispatches to.
        max_epochs / batch_size: the shared knobs.
        jit_compile: Trace the step functions with XLA. This is the knob the Keras backend cannot
            offer — it exists only because the steps are OURS.
        prefetch: Overlap input preparation with the step, through ``tf.data``'s AUTOTUNE.
        export_dir: Write a SavedModel here after ``fit()``. The one run artifact that is
            TensorFlow's and not Keras's.
        trackers / experiment_name / run_name: tracking identity and sinks.
        input_key / target_key: Record keys for the degraded image and the clean target.
        optimizer / loss: Keras objects or names.
        metrics / train_metrics / val_metrics / test_metrics: per-split metric sets. Unlike the
            Keras backend this loop CAN honour them per split — each traced step closes over its
            own list — so they are MERGED rather than unioned.
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
        self.model: TensorFlowNetwork = cast(TensorFlowNetwork, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: RestorerTask = task
        self.data_range = float(data_range)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.jit_compile = bool(jit_compile)
        self.prefetch = bool(prefetch)
        self.export_dir = export_dir
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.optimizer: Lazy[KerasOptimizer] = optimizer
        self.loss: Optional[Lazy[KerasLoss]] = loss
        self.metrics: Optional[List[Lazy[KerasMetric]]] = metrics
        self.train_metrics: Optional[List[Lazy[KerasMetric]]] = train_metrics
        self.val_metrics: Optional[List[Lazy[KerasMetric]]] = val_metrics
        self.test_metrics: Optional[List[Lazy[KerasMetric]]] = test_metrics
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self._metric_objects: Dict[str, List[Any]] = {}
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Check the engine, prepare the sources, build the model and the per-split metrics."""
        self._require_tensorflow_backend()
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        self._resolve_task_defaults()
        self.model = cast(TensorFlowNetwork, flow(self.model))
        for split in ("train", "val", "test"):
            spec = merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics"))
            # A FRESH object per split: a Keras metric is a stateful ACCUMULATOR, so two splits
            # sharing one instance would report the union of both passes.
            self._metric_objects[split] = [self._as_metric(m) for m in (spec or [])]

    @staticmethod
    def _require_tensorflow_backend() -> None:
        """This loop drives ``tf.GradientTape``; a torch- or jax-bound Keras cannot be driven by it.

        The precondition a ``KerasRestorer`` does NOT have — which is exactly why this class is a
        sibling rather than a subclass.
        """
        backend = keras_backend()
        if backend != "tensorflow":
            raise RuntimeError(
                f"TensorFlowRestorer needs Keras bound to the TensorFlow engine, but KERAS_BACKEND "
                f"is {backend!r}. Re-run with KERAS_BACKEND=tensorflow, or use --framework keras, "
                f"which runs the identical config on whichever engine is bound."
            )

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process (the fork guard)."""
        if source is None:
            return None
        return ensure_materialized(ensure_record_dataset(source))

    def _resolve_task_defaults(self) -> None:
        """Fill the unset (``None``) task-shaped slots. Idempotent; never overwrites."""
        if self.loss is None:
            self.loss = keras.losses.MeanAbsoluteError()
        if self.metrics is None:
            self.metrics = [
                PeakSignalNoiseRatio(data_range=self.data_range, name="psnr"),
                keras.metrics.MeanSquaredError(name="mse"),
            ]

    def _as_metric(self, value: Any) -> Any:
        """A FRESH metric object per split, cloning a live instance through Keras's own config.

        A Keras metric accumulates, so handing the same instance to two splits would make each
        report the other's batches too (measured on the segmentation twin: a clone reports 0.0
        where the original, after an update, reports 1.0).
        """
        built = flow(value)
        if isinstance(built, str):
            return keras.metrics.get(built)
        if isinstance(built, keras.metrics.Metric):
            return type(built).from_config(built.get_config())
        return built

    # ---- the batch boundary -------------------------------------------------- #
    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → the numpy ``(x, y)`` pair — channels LAST."""
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="float32")

    def _sequence(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> RecordSequence:
        return RecordSequence(
            source,
            batch_size=self.batch_size,
            shuffle=shuffle,
            transform=partial(self._to_arrays, with_targets=with_targets),
        )

    def _dataset(self, source: Any, shuffle: bool = False) -> Any:
        """A ``tf.data.Dataset`` over recordstream's batching adapter — TensorFlow's half.

        ``from_generator`` rather than ``from_tensor_slices``: the source is a lazy record stream,
        not an array already in memory, which is the whole point of the engine underneath.
        """
        sequence = self._sequence(source, shuffle=shuffle)

        def generate() -> Any:
            for index in range(len(sequence)):
                yield sequence[index]

        signature = (
            tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
        )
        dataset = tf.data.Dataset.from_generator(generate, output_signature=signature)
        return dataset.prefetch(tf.data.AUTOTUNE) if self.prefetch else dataset

    # ---- the traced steps ---------------------------------------------------- #
    def _network(self, x: Any, training: bool) -> Any:
        """Call the model, passing ``training`` only where the object takes it.

        A ``keras.Model`` and this project's raw ``tf.Module`` both accept it; a bare ``tf.Module``
        a user brings may not, and a TypeError three frames inside a traced function is not a
        message anyone can act on.
        """
        try:
            return self.model(x, training=training)
        except TypeError:
            return self.model(x)

    def _train_step_fn(self) -> Callable[..., Any]:
        loss_fn, optimizer = flow(self.loss), self._optimizer
        metrics = self._metric_objects["train"]

        @tf.function(jit_compile=self.jit_compile)
        def step(x: Any, y: Any) -> Any:
            with tf.GradientTape() as tape:
                predictions = self._network(x, training=True)
                loss = loss_fn(y, predictions)
            variables = self.model.trainable_variables
            optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
            for metric in metrics:
                metric.update_state(y, predictions)
            return loss

        return cast(Callable[..., Any], step)

    def _eval_step_fn(self, split: str) -> Callable[..., Any]:
        loss_fn = flow(self.loss)
        metrics = self._metric_objects[split]

        @tf.function(jit_compile=self.jit_compile)
        def step(x: Any, y: Any) -> Any:
            predictions = self._network(x, training=False)
            for metric in metrics:
                metric.update_state(y, predictions)
            return loss_fn(y, predictions)

        return cast(Callable[..., Any], step)

    @property
    def _optimizer(self) -> Any:
        """The flowed optimizer, built once and reused across epochs."""
        if not hasattr(self, "_optimizer_cache"):
            self._optimizer_cache = flow(self.optimizer)
            if isinstance(self._optimizer_cache, str):
                self._optimizer_cache = keras.optimizers.get(self._optimizer_cache)
        return self._optimizer_cache

    def _run_epoch(self, dataset: Any, step: Callable[..., Any], split: str) -> Dict[str, float]:
        """One pass, reducing the metrics ONCE at the end — never averaging per batch.

        PSNR is a logarithm of a mean squared error, so a per-batch average is not the split's
        value; the metric objects accumulate and are read here, which is what makes this column
        comparable with the torch backends'.
        """
        for metric in self._metric_objects[split]:
            metric.reset_state()
        total, batches = 0.0, 0
        for x, y in dataset:
            total += float(step(x, y))
            batches += 1
        prefix = "" if split == "train" else f"{split}_"
        results = {f"{prefix}loss": total / batches if batches else float("nan")}
        for metric in self._metric_objects[split]:
            results[f"{prefix}{metric.name}"] = float(metric.result())
        return results

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"distilaide tensorflow training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("TensorFlowRestorer.fit() requires a 'train_set'.")
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, {"jit_compile": self.jit_compile}, runnable=self) if trackers else None

        train_ds = self._dataset(self.train_set, shuffle=True)
        val_ds = self._dataset(self.val_set) if self.val_set is not None else None
        # Built ONCE and reused across epochs: a fresh `tf.function` per epoch retraces the whole
        # graph every time and ends up slower than the eager loop it replaced.
        train_step = self._train_step_fn()
        val_step = self._eval_step_fn("val") if val_ds is not None else None
        # OUTSIDE the graph: Keras builds slot variables on the first `apply_gradients`, and
        # creating variables during a RETRACE — which a short final batch triggers — is an error.
        self._optimizer.build(self.model.trainable_variables)

        names = as_metric_names(self.metric_names)
        for epoch in range(self.max_epochs):
            logs = self._run_epoch(train_ds, train_step, "train")
            if val_ds is not None and val_step is not None:
                logs.update(self._run_epoch(val_ds, val_step, "val"))
            translated = names(logs, source="keras")
            for tracker in trackers:
                tracker.log_metrics(translated, step=epoch)
            for key, value in translated.items():
                self._history.setdefault(key, []).append(value)
            logger.info(
                f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in translated.items())
            )
            self._computed_metrics = dict(translated)

        logger.info("Training complete.")
        self._export()
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("TensorFlowRestorer.test() requires a 'test_set'.")
        results = self._run_epoch(self._dataset(self.test_set), self._eval_step_fn("test"), "test")
        self._computed_metrics = as_metric_names(self.metric_names)(results, source="keras")
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        if self.test_set is None:
            raise ValueError("TensorFlowRestorer.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            restored = np.asarray(self._network(tf.convert_to_tensor(sequence[index][0]), training=False))
            if sink is not None:
                self._write_predictions(restored, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- artifacts ----------------------------------------------------------- #
    def _export(self) -> None:
        """A SavedModel — the one run artifact that is TensorFlow's and not Keras's.

        A ``tf.Module`` has NO ``keras.Model.export``, and a bare ``__call__`` is untraced Python
        that cannot be serialized — so a module export needs an explicitly traced signature.
        """
        if not self.export_dir:
            return
        directory = Path(self.export_dir).expanduser()
        directory.parent.mkdir(parents=True, exist_ok=True)
        exporter = getattr(self.model, "export", None)
        if callable(exporter):
            exporter(str(directory))
        else:
            signature = tf.function(
                lambda x: self._network(x, training=False),
                input_signature=[tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)],
            )
            tf.saved_model.save(self.model, str(directory), signatures={"serving_default": signature})
        logger.info(f"Exported a SavedModel to {directory}")

    def _write_predictions(self, restored: np.ndarray, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.RestorationOutput` per record, paired with its metadata."""
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
        """Per-epoch metric history — what a ``fit()``'s ``History`` would have accumulated."""
        return {key: list(values) for key, values in self._history.items()}
