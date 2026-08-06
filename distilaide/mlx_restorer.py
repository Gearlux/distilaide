"""``MlxRestorer`` — the same restoration run on Apple's MLX.

The EIGHTH backend and the last of the fleet. **What MLX contributes is three facts, and each
shapes the loop rather than its spelling:**

1. **Unified memory** — CPU and GPU share it on Apple silicon, so there is NO device transfer and
   this is the one backend whose batch path does genuinely LESS work than the others.
2. **Arrays are LAZY** — ``mx.eval`` forces computation, and omitting it does NOT raise: it grows
   the graph until memory runs out. So ``fit()`` evaluates once per step over the parameters, the
   optimizer state AND the loss, deliberately and visibly.
3. **Train/eval is module STATE, not a per-call argument** — ``network.train(True)`` at the top of
   each epoch and ``train(False)`` before an eval pass, because a ``training=False`` DEFAULT on
   ``__call__`` would flip the network to eval on every forward, including the one inside
   ``nn.value_and_grad`` where there is no flag to pass, and batch norm would then never train.

Note the asymmetry with the JAX backend, which is why ``_as_optimizer`` is not shared: **optax ships
FACTORIES (``optax.adam(lr)``) while ``mlx.optimizers`` ships CLASSES**, so this one INSTANTIATES
where the JAX one CALLS.

Unlike optax, ``mlx.nn.losses`` DOES ship ``l1_loss``, so this backend's default needs nothing
written out — it names the upstream function directly and still matches every other backend.

The METRICS are matrainer's shared squared-error accumulator, the same as the JAX backend's: MLX
ships no image metrics, and a loop that owns its epochs must not average PSNR per batch.

**Apple silicon ONLY** — the extra's ``sys_platform == 'darwin' and platform_machine == 'arm64'``
marker keeps ``pip install distilaide[mlx]`` succeeding (and empty) elsewhere.
"""

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
import numpy as np
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
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

#: Which entry point :meth:`MlxRestorer.run` dispatches to.
RestorerTask = Literal["fit", "evaluate", "test", "predict"]

#: An MLX loss: the NAME of an ``mlx.nn.losses`` function, or a callable. A name because MLX ships
#: functions rather than classes — the optax situation exactly.
MlxLoss = Union[str, Callable[..., Any]]

#: An MLX optimizer: the NAME of an ``mlx.optimizers`` CLASS, or one already built.
MlxOptimizer = Union[str, Any]

#: The metrics this backend can compute, by NAME — all derived from the shared accumulator.
MlxMetric = str

_METRIC_REDUCTIONS: Dict[str, Callable[[np.ndarray, float], float]] = {
    "psnr": lambda totals, data_range: psnr(totals, data_range),
    "mse": lambda totals, _: mean_squared_error(totals),
    "mae": lambda totals, _: mean_absolute_error(totals),
}

#: What ``metrics`` defaults to — the same pair every non-torch backend reports.
DEFAULT_METRICS: Tuple[str, ...] = ("psnr", "mse")

__all__ = ["DEFAULT_METRICS", "MlxLoss", "MlxMetric", "MlxOptimizer", "MlxRestorer", "RestorerTask"]


@configurable(task="restoration", role="trainer", framework="mlx")
class MlxRestorer(ProgressReporting):
    """Train, evaluate or predict an image restorer with MLX.

    Args:
        model: An ``mlx.nn.Module`` mapping ``[N, H, W, C]`` degraded images to restored ones —
            channels LAST. Wire ``!lazy:distilaide.MlxDnCNN``.
        train_set / test_set: sources.
        data_range: The span of the target images' values, for PSNR.
        task: Which entry point ``run()`` dispatches to.
        max_epochs / batch_size / learning_rate: the shared knobs.
        loss: An ``mlx.nn.losses`` name or a callable; ``None`` picks ``l1_loss``.
        metrics / train_metrics / val_metrics / test_metrics: metric NAMES.
        optimizer: An ``mlx.optimizers`` CLASS name (``"Adam"``, ``"AdamW"``, ``"SGD"``).
        trackers / experiment_name / run_name: tracking identity and sinks.
        input_key / target_key: Record keys for the degraded image and the clean target.
        metric_names: How recorded metrics are named.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[nn.Module]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        data_range: float = 1.0,
        task: RestorerTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        loss: Optional[MlxLoss] = None,
        metrics: Optional[List[MlxMetric]] = None,
        train_metrics: Optional[List[MlxMetric]] = None,
        val_metrics: Optional[List[MlxMetric]] = None,
        test_metrics: Optional[List[MlxMetric]] = None,
        optimizer: MlxOptimizer = "Adam",
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        experiment_name: str = "distilaide_restoration",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metric_names: Optional[Lazy[MetricNames]] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = cast(nn.Module, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: RestorerTask = task
        self.data_range = float(data_range)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = float(learning_rate)
        self.loss: Optional[MlxLoss] = loss
        self.metrics: Optional[List[MlxMetric]] = metrics
        self.train_metrics: Optional[List[MlxMetric]] = train_metrics
        self.val_metrics: Optional[List[MlxMetric]] = val_metrics
        self.test_metrics: Optional[List[MlxMetric]] = test_metrics
        self.optimizer: MlxOptimizer = optimizer
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
        self.model = cast(nn.Module, flow(self.model))
        self._resolve_task_defaults()

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process (the fork guard)."""
        if source is None:
            return None
        return ensure_materialized(ensure_record_dataset(source))

    def _resolve_task_defaults(self) -> None:
        """Fill the unset slots.

        Unlike the JAX backend this needs nothing written out: ``mlx.nn.losses`` DOES ship
        ``l1_loss``, so the default names the upstream function and still matches every other
        backend's L1.
        """
        if self.loss is None:
            self.loss = "l1_loss"
        if self.metrics is None:
            self.metrics = list(DEFAULT_METRICS)

    def _loss_fn(self) -> Callable[..., Any]:
        """The loss as a callable over ``(predictions, targets)``, reduced to a scalar."""
        chosen = self.loss
        if callable(chosen):
            return chosen
        function = getattr(nn.losses, str(chosen), None)
        if function is None:
            raise ValueError(
                f"Unknown MLX loss {chosen!r}. Names come from `mlx.nn.losses` — for an "
                f"image-to-image target: l1_loss, mse_loss, huber_loss, smooth_l1_loss."
            )
        return partial(function, reduction="mean")

    def _as_optimizer(self) -> Any:
        """Build the optimizer. **INSTANTIATES**, where the JAX backend CALLS a factory.

        ``mlx.optimizers`` ships CLASSES (``optimizers.Adam(learning_rate=…)``) while optax ships
        factories (``optax.adam(lr)``) — the asymmetry that keeps the two backends' helpers apart.
        """
        chosen = self.optimizer
        if isinstance(chosen, str):
            cls = getattr(optimizers, chosen, None)
            if cls is None:
                raise ValueError(
                    f"Unknown MLX optimizer {chosen!r}. Names come from `mlx.optimizers` — e.g. "
                    f"Adam, AdamW, SGD, Adagrad, Lion."
                )
            return cls(learning_rate=self.learning_rate)
        return chosen

    def _metric_names_for(self, split: str) -> List[str]:
        merged = merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or []
        for name in merged:
            if name not in _METRIC_REDUCTIONS:
                raise ValueError(
                    f"Unknown metric {name!r} for the mlx backend. Available: "
                    f"{sorted(_METRIC_REDUCTIONS)}. (MLX ships no image metrics, so these are "
                    f"derived from matrainer's shared squared-error accumulator.)"
                )
        return list(merged)

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

    # ---- the loop ------------------------------------------------------------ #
    def _train_epoch(self, sequence: RecordSequence, optimizer: Any) -> Dict[str, float]:
        """One training pass. MLX carries train/eval as module STATE, so the mode is set ONCE."""
        loss_fn = self._loss_fn()
        network = self.model
        network.train(True)

        def objective(model: Any, x: Any, y: Any) -> Any:
            return loss_fn(model(x), y)

        value_and_grad = nn.value_and_grad(network, objective)
        totals = zero_squared_error()
        total_loss, batches = 0.0, 0
        for index in range(len(sequence)):
            x, y = sequence[index]
            mx_x, mx_y = mx.array(x), mx.array(y)
            loss, grads = value_and_grad(network, mx_x, mx_y)
            optimizer.update(network, grads)
            # MLX arrays are LAZY: without this the graph grows until memory runs out, and it does
            # NOT raise — so the eval is deliberate and visible, over the parameters, the optimizer
            # state AND the loss.
            mx.eval(network.parameters(), optimizer.state, loss)
            total_loss += float(loss)
            batches += 1
            accumulate_squared_error(totals, np.asarray(network(mx_x)), y)
        return self._reduce(totals, total_loss, batches, "train")

    def _eval_epoch(self, sequence: RecordSequence, split: str) -> Dict[str, float]:
        """One evaluation pass, with the network switched to eval MODE (not a per-call flag)."""
        loss_fn = self._loss_fn()
        network = self.model
        network.train(False)
        totals = zero_squared_error()
        total_loss, batches = 0.0, 0
        for index in range(len(sequence)):
            x, y = sequence[index]
            predictions = network(mx.array(x))
            loss = loss_fn(predictions, mx.array(y))
            mx.eval(predictions, loss)
            total_loss += float(loss)
            batches += 1
            accumulate_squared_error(totals, np.asarray(predictions), y)
        return self._reduce(totals, total_loss, batches, split)

    def _reduce(self, totals: np.ndarray, total_loss: float, batches: int, split: str) -> Dict[str, float]:
        """Reduce ONCE at the end of a split — never a per-batch PSNR average."""
        prefix = "" if split == "train" else f"{split}_"
        results = {f"{prefix}loss": total_loss / batches if batches else float("nan")}
        for name in self._metric_names_for(split):
            results[f"{prefix}{name}"] = _METRIC_REDUCTIONS[name](totals, self.data_range)
        return results

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"distilaide mlx training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("MlxRestorer.fit() requires a 'train_set'.")
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self) if trackers else None

        optimizer = self._as_optimizer()
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None

        names = as_metric_names(self.metric_names)
        for epoch in range(self.max_epochs):
            logs = self._train_epoch(train_seq, optimizer)
            if val_seq is not None:
                logs.update(self._eval_epoch(val_seq, "val"))
            translated = names(logs, source="keras")
            for tracker in trackers:
                tracker.log_metrics(translated, step=epoch)
            for key, value in translated.items():
                self._history.setdefault(key, []).append(value)
            logger.info(
                f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in translated.items())
            )
            self._computed_metrics = dict(translated)
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
            raise ValueError("MlxRestorer.test() requires a 'test_set'.")
        results = self._eval_epoch(self._sequence(self.test_set), "test")
        self._computed_metrics = as_metric_names(self.metric_names)(results, source="keras")
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        if self.test_set is None:
            raise ValueError("MlxRestorer.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        self.model.train(False)
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            restored = self.model(mx.array(sequence[index][0]))
            mx.eval(restored)
            if sink is not None:
                self._write_predictions(np.asarray(restored), batch, cast(PredictionsSink, sink))
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
