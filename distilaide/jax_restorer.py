"""``JaxRestorer`` — the same restoration run as a jitted JAX/optax loop.

The closest sibling of :class:`~distilaide.tensorflow_restorer.TensorFlowRestorer` — both replace
``fit()`` with a traced step — where the differences between them are exactly the differences
between the two programming models. The step is a PURE FUNCTION of ``(model, batch)`` that
``nnx.value_and_grad`` differentiates and ``nnx.Optimizer`` applies in place; randomness is
EXPLICIT (the model carries its own ``nnx.Rngs``, which is what makes ``seed:`` on ``FlaxDnCNN``
real).

``_make_steps()`` is called ONCE and its result reused across epochs — a fresh ``nnx.jit`` per epoch
retraces the whole graph every time and ends up slower than the eager loop it replaced, the same
trap ``tf.function`` has. ``jit: false`` exists to read a Python traceback from inside a step.

**Losses and optimizers are NAMES, not ``!class:`` tags:** optax ships plain FUNCTIONS and
FACTORIES, so there is no class for a tag to resolve or for confluid's registry to tag — hence NO
``jax_restoration.py`` and ``FRAMEWORKS["jax"].registrations is None``.

**optax ships no L1 loss, and the default is L1 anyway.** Verified against optax's own surface: its
loss set is ``l2_loss`` / ``squared_error`` / ``huber_loss`` / ``cosine_*`` — no mean-absolute-error
anywhere. Two ways out, and the choice is not close: change THIS backend's default to something
optax has, which breaks the one claim the project makes (a config moving between engines keeps its
objective), or write the one expression out. ``jnp.abs(predictions - targets)`` is not a
reimplementation of an algorithm, so ``"l1"`` is a name this module resolves itself while every
other name passes straight through to ``optax.losses``.

**The METRICS are matrainer's, and that is the interesting part.** JAX ships no image metrics at
all, and a loop that owns its epochs cannot average PSNR per batch and stay comparable — so this
backend accumulates squared error through :mod:`matrainer.metric_functions` and reduces once, the
same helper the MLX backend uses and the same numbers ``torchmetrics`` produces.

Channels-LAST, so a JAX config's ``preprocess`` is the Keras/TensorFlow one. ``RecordSequence`` is
reused verbatim — batching a record stream is not a JAX problem.
"""

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

import jax.numpy as jnp
import numpy as np
import optax
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
from flax import nnx
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_functions import (
    accumulate_squared_error,
    mean_absolute_error,
    mean_squared_error,
    psnr,
    zero_squared_error,
)
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
from recordstream.keras import RecordSequence
from recordstream.runnable import ProgressReporting, entrypoint, run_entrypoint

logger = get_logger(__name__)

#: Which entry point :meth:`JaxRestorer.run` dispatches to.
RestorerTask = Literal["fit", "evaluate", "test", "predict"]

#: A JAX loss: the NAME of an ``optax.losses`` function, or a callable. A name because optax ships
#: functions rather than classes, so there is nothing for a `!class:` tag to resolve.
JaxLoss = Union[str, Callable[..., Any]]

#: A JAX optimizer: the NAME of an ``optax`` factory (``"adam"``, ``"adamw"``, ``"sgd"``), or one
#: already built.
JaxOptimizer = Union[str, Any]

#: The metrics this backend can compute, by NAME. Written out rather than discovered because each
#: is DERIVED from the shared squared-error accumulator — see :mod:`matrainer.metric_functions`.
JaxMetric = str

#: ``name -> a reduction over the accumulator``. The whole metric vocabulary of this backend.
_METRIC_REDUCTIONS: Dict[str, Callable[[np.ndarray, float], float]] = {
    "psnr": lambda totals, data_range: psnr(totals, data_range),
    "mse": lambda totals, _: mean_squared_error(totals),
    "mae": lambda totals, _: mean_absolute_error(totals),
}

#: Losses this module resolves ITSELF, because optax does not ship them. Only ``l1``, and only
#: because it is every other backend's default — see :meth:`JaxRestorer._loss_fn`.
_LOCAL_LOSSES: Dict[str, Callable[..., Any]] = {
    "l1": lambda predictions, targets: jnp.abs(predictions - targets),
}

#: What ``metrics`` defaults to — the same pair every non-torch backend reports.
DEFAULT_METRICS: Tuple[str, ...] = ("psnr", "mse")

__all__ = [
    "DEFAULT_METRICS",
    "JaxLoss",
    "JaxMetric",
    "JaxOptimizer",
    "JaxRestorer",
    "RestorerTask",
]


@configurable(task="restoration", role="trainer", framework="jax")
class JaxRestorer(ProgressReporting):
    """Train, evaluate or predict an image restorer with a jitted JAX/optax loop.

    Args:
        model: An ``nnx.Module`` mapping ``[N, H, W, C]`` degraded images to restored ones —
            channels LAST. Wire ``!lazy:distilaide.FlaxDnCNN``.
        train_set / test_set: sources.
        data_range: The span of the target images' values, for PSNR.
        task: Which entry point ``run()`` dispatches to.
        max_epochs / batch_size: the shared knobs.
        learning_rate: Passed to the named optax factory, so this stays the one place an LR is written.
        jit: Trace the steps with ``nnx.jit``. Turn OFF to read a Python traceback from inside a step.
        loss: An ``optax.losses`` name or a callable; ``None`` picks ``l1_loss``.
        metrics / train_metrics / val_metrics / test_metrics: metric NAMES — see
            :data:`_METRIC_REDUCTIONS` for the vocabulary.
        optimizer: An ``optax`` factory name.
        trackers / experiment_name / run_name: tracking identity and sinks.
        input_key / target_key: Record keys for the degraded image and the clean target.
        metric_names: How recorded metrics are named.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[nnx.Module]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        data_range: float = 1.0,
        task: RestorerTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        jit: bool = True,
        loss: Optional[JaxLoss] = None,
        metrics: Optional[List[JaxMetric]] = None,
        train_metrics: Optional[List[JaxMetric]] = None,
        val_metrics: Optional[List[JaxMetric]] = None,
        test_metrics: Optional[List[JaxMetric]] = None,
        optimizer: JaxOptimizer = "adam",
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        experiment_name: str = "distilaide_restoration",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metric_names: Optional[Lazy[MetricNames]] = None,
    ) -> None:
        super().__init__()
        self.model: nnx.Module = cast(nnx.Module, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: RestorerTask = task
        self.data_range = float(data_range)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = float(learning_rate)
        self.jit = bool(jit)
        self.loss: Optional[JaxLoss] = loss
        self.metrics: Optional[List[JaxMetric]] = metrics
        self.train_metrics: Optional[List[JaxMetric]] = train_metrics
        self.val_metrics: Optional[List[JaxMetric]] = val_metrics
        self.test_metrics: Optional[List[JaxMetric]] = test_metrics
        self.optimizer: JaxOptimizer = optimizer
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
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
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        self.model = cast(nnx.Module, flow(self.model))
        self._resolve_task_defaults()

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process (the fork guard)."""
        if source is None:
            return None
        return ensure_materialized(ensure_record_dataset(source))

    def _resolve_task_defaults(self) -> None:
        """Fill the unset slots. ``l1`` matches the torch backends' ``L1Loss`` exactly."""
        if self.loss is None:
            self.loss = "l1"
        if self.metrics is None:
            self.metrics = list(DEFAULT_METRICS)

    def _loss_fn(self) -> Callable[..., Any]:
        """The loss as an ELEMENTWISE callable over ``(predictions, targets)``.

        Elementwise because optax's losses are, and the caller takes the mean — which is what makes
        a named optax loss agree with its torch namesake rather than being an array-valued thing a
        loop cannot use.

        ``"l1"`` is resolved HERE rather than from optax, because optax does not have one (its loss
        set is ``l2_loss`` / ``squared_error`` / ``huber_loss`` / ``cosine_*``). Writing the one
        expression out is what keeps this backend's default the SAME as every other backend's; see
        the module docstring.
        """
        chosen = self.loss
        if callable(chosen):
            return chosen
        if str(chosen) in _LOCAL_LOSSES:
            return _LOCAL_LOSSES[str(chosen)]
        function = getattr(optax.losses, str(chosen), None)
        if function is None:
            raise ValueError(
                f"Unknown loss {chosen!r} for the jax backend. Available: {sorted(_LOCAL_LOSSES)} "
                f"(written out here) plus any name in `optax.losses` — for an image-to-image "
                f"target: squared_error, huber_loss, l2_loss."
            )
        return cast(Callable[..., Any], function)

    def _metric_names_for(self, split: str) -> List[str]:
        """The NAMES this split reports, merged over the shared set by matrainer's rule."""
        merged = merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or []
        for name in merged:
            if name not in _METRIC_REDUCTIONS:
                raise ValueError(
                    f"Unknown metric {name!r} for the jax backend. Available: "
                    f"{sorted(_METRIC_REDUCTIONS)}. (JAX ships no image metrics, so these are "
                    f"derived from matrainer's shared squared-error accumulator.)"
                )
        return list(merged)

    # ---- the batch boundary -------------------------------------------------- #
    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → the numpy ``(x, y)`` pair — channels LAST.

        recordstream hands over NUMPY, which is exactly what ``jnp.asarray`` takes, so this
        backend needs no conversion layer at all.
        """
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

    # ---- the traced steps ---------------------------------------------------- #
    def _make_steps(self, optimizer: Any) -> Tuple[Callable[..., Any], Callable[..., Any]]:
        """Build the train and eval steps ONCE, and jit them ONCE.

        A fresh ``nnx.jit`` per epoch retraces the whole graph every time and ends up slower than
        the eager loop it replaced — the same trap ``tf.function`` has. Pinned by a test that reads
        this method's presence in ``fit``'s source.
        """
        loss_fn = self._loss_fn()

        def loss_of(model: Any, x: Any, y: Any) -> Tuple[Any, Any]:
            predictions = model(x, training=True)
            # optax losses are ELEMENTWISE; the mean is what makes this agree with torch's L1Loss.
            return jnp.mean(loss_fn(predictions, y)), predictions

        def train_step(model: Any, opt: Any, x: Any, y: Any) -> Tuple[Any, Any]:
            # `has_aux=True` returns the predictions alongside the loss, so the metrics cost ONE
            # reduction per batch rather than a second forward pass.
            (loss, predictions), grads = nnx.value_and_grad(loss_of, has_aux=True)(model, x, y)
            opt.update(model, grads)
            return loss, predictions

        def eval_step(model: Any, x: Any, y: Any) -> Tuple[Any, Any]:
            predictions = model(x, training=False)
            return jnp.mean(loss_fn(predictions, y)), predictions

        if self.jit:
            return nnx.jit(train_step), nnx.jit(eval_step)
        return train_step, eval_step

    def _build_optimizer(self) -> Any:
        """An ``nnx.Optimizer`` over the named optax factory, CALLED with the learning rate."""
        chosen = self.optimizer
        if isinstance(chosen, str):
            factory = getattr(optax, chosen, None)
            if factory is None:
                raise ValueError(
                    f"Unknown optax optimizer {chosen!r}. Names come from `optax` — e.g. adam, "
                    f"adamw, sgd, rmsprop, lion."
                )
            # optax ships FACTORIES, so this CALLS where the MLX backend instantiates a class.
            chosen = factory(self.learning_rate)
        return nnx.Optimizer(self.model, chosen, wrt=nnx.Param)

    def _run_epoch(
        self, sequence: RecordSequence, step: Callable[..., Any], split: str, optimizer: Any = None
    ) -> Dict[str, float]:
        """One pass, accumulating the error and reducing ONCE — never averaging PSNR per batch."""
        totals = zero_squared_error()
        total_loss, batches = 0.0, 0
        for index in range(len(sequence)):
            x, y = sequence[index]
            jx, jy = jnp.asarray(x), jnp.asarray(y)
            if optimizer is not None:
                loss, predictions = step(self.model, optimizer, jx, jy)
            else:
                loss, predictions = step(self.model, jx, jy)
            total_loss += float(loss)
            batches += 1
            accumulate_squared_error(totals, np.asarray(predictions), y)
        prefix = "" if split == "train" else f"{split}_"
        results = {f"{prefix}loss": total_loss / batches if batches else float("nan")}
        for name in self._metric_names_for(split):
            results[f"{prefix}{name}"] = _METRIC_REDUCTIONS[name](totals, self.data_range)
        return results

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"distilaide jax training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("JaxRestorer.fit() requires a 'train_set'.")
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, {"jit": self.jit}, runnable=self) if trackers else None

        optimizer = self._build_optimizer()
        # ONCE — see `_make_steps`.
        train_step, eval_step = self._make_steps(optimizer)
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None

        names = as_metric_names(self.metric_names)
        for epoch in range(self.max_epochs):
            logs = self._run_epoch(train_seq, train_step, "train", optimizer=optimizer)
            if val_seq is not None:
                logs.update(self._run_epoch(val_seq, eval_step, "val"))
            translated = names(logs, source="keras")
            for tracker in trackers:
                tracker.log_metrics(translated, step=epoch)
            for key, value in translated.items():
                self._history.setdefault(key, []).append(value)
            logger.info(
                f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in translated.items())
            )
            self._computed_metrics = dict(translated)
            # A `fit()` would do this for you: without it every epoch sees the same batch order.
            train_seq.on_epoch_end()

        logger.info("Training complete.")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("JaxRestorer.test() requires a 'test_set'.")
        _, eval_step = self._make_steps(optimizer=None)
        results = self._run_epoch(self._sequence(self.test_set), eval_step, "test")
        self._computed_metrics = as_metric_names(self.metric_names)(results, source="keras")
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        if self.test_set is None:
            raise ValueError("JaxRestorer.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            restored = np.asarray(self.model(jnp.asarray(sequence[index][0]), training=False))
            if sink is not None:
                self._write_predictions(restored, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

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
        """Per-epoch metric history — what a ``Trainer``'s logger would have accumulated."""
        return {key: list(values) for key, values in self._history.items()}
