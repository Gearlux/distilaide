"""``MlxRestorer`` — the same restoration run on Apple's MLX.

Since P6 (2026-08-08) this is ``(RestorationBase, matrainer.mlx.MlxLoopRunnable)``: the loop
machinery — the ``nn.value_and_grad`` step with the per-step ``mx.eval``, the train/eval MODE
flips at the loop boundaries, the epoch driver with the per-epoch reshuffle, the
plain-function tracking hookup, the record-batch predict walk — lives in
:class:`matrainer.mlx.loop.MlxLoopRunnable`; the split preparation and the trainable-model
gate in :class:`~distilaide.restoration_base.RestorationBase`. What stays HERE is
restoration's mlx vocabulary:

* **``mlx.nn.losses`` DOES ship ``l1_loss``** (unlike optax), so the default names the
  upstream function and still matches every other backend's L1 — nothing written out.
* **The metrics are matrainer's shared squared-error accumulator** — MLX ships no image
  metrics, and a loop that owns its epochs must not average PSNR per batch.

**Apple silicon ONLY** — the extra's ``sys_platform == 'darwin' and platform_machine ==
'arm64'`` marker keeps ``pip install distilaide[mlx]`` succeeding (and empty) elsewhere.
"""

from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

import mlx.nn as nn
import numpy as np
from confluid import Lazy, Mandatory, configurable
from loggair import get_logger
from matrainer.metric_functions import (
    accumulate_squared_error,
    mean_absolute_error,
    mean_squared_error,
    psnr,
    zero_squared_error,
)
from matrainer.metric_names import MetricNames
from matrainer.metric_slots import merge_metric_specs
from matrainer.mlx.loop import MlxLoopRunnable
from matrainer.tracking.base import Tracker
from recordstream import RecordSource

from distilaide.restoration_base import RestorationBase

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
class MlxRestorer(RestorationBase, MlxLoopRunnable):
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
        model: Optional[Mandatory[Lazy[nn.Module]]] = None,
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
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            trackers=trackers,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metric_names=metric_names,
        )
        self.data_range = float(data_range)

    # ---- the task hooks ---------------------------------------------------------------------- #
    def _default_mlx_loss(self) -> Optional[MlxLoss]:
        """``l1_loss`` — the upstream name, matching every other backend's L1 default."""
        return "l1_loss"

    def _default_mlx_metrics(self) -> Optional[List[Any]]:
        return list(DEFAULT_METRICS)

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

    # ---- the metric state: ONE squared-error accumulator per split --------------------------- #
    def _new_split_state(self, split: str) -> Any:
        return zero_squared_error()

    def _score_batch(self, state: Any, outputs: Any, arrays: Tuple[Any, ...], split: str) -> None:
        accumulate_squared_error(state, np.asarray(outputs), np.asarray(arrays[1]))

    def _split_scores(self, state: Any, split: str, step: Optional[int] = None) -> Dict[str, float]:
        """Reduced ONCE per pass — never a per-batch PSNR average, which moves with the batch size."""
        return {name: _METRIC_REDUCTIONS[name](state, self.data_range) for name in self._metric_names_for(split)}

    # ---- predictions -------------------------------------------------------------------------- #
    def _prediction_output(self, preds: Any) -> Any:
        """One batched :class:`~recordstream.RestorationOutput` — the machinery slices it."""
        return {"image": cast(np.ndarray, preds)}
