"""``JaxRestorer`` — the same restoration run as a jitted JAX/optax loop.

Since P6 (2026-08-08) this is ``(RestorationBase, matrainer.jax.JaxLoopRunnable)``: the loop
machinery — the jit-ONCE step builders, the epoch driver with the per-epoch reshuffle, the
plain-function tracking hookup, the per-epoch callback fan-out, the record-batch predict
walk — lives in :class:`matrainer.jax.loop.JaxLoopRunnable`; the split preparation and the
trainable-model gate in :class:`~distilaide.restoration_base.RestorationBase`. What stays
HERE is restoration's jax vocabulary:

* **optax ships no L1 loss, and the default is L1 anyway.** Verified against optax's own
  surface: its loss set is ``l2_loss`` / ``squared_error`` / ``huber_loss`` / ``cosine_*`` —
  no mean-absolute-error anywhere. ``jnp.abs(predictions - targets)`` is not a
  reimplementation of an algorithm, so ``"l1"`` is a name :meth:`_loss_fn` resolves itself
  while every other name passes straight through to ``optax.losses`` — which is what keeps
  this backend's default the SAME as every other backend's.
* **The metrics are matrainer's shared squared-error accumulator** — JAX ships no image
  metrics, and a loop that owns its epochs cannot average PSNR per batch and stay comparable,
  so the state hooks accumulate through :mod:`matrainer.metric_functions` and reduce once.

Channels-LAST, so a JAX config's ``preprocess`` is the Keras/TensorFlow one — and the batch
boundary is the machinery's image-to-image default, untouched.
"""

from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

import jax.numpy as jnp
import numpy as np
import optax
from confluid import Lazy, Mandatory, configurable
from flax import nnx
from loggair import get_logger
from matrainer.jax.loop import JaxLoopRunnable
from matrainer.metric_functions import (
    accumulate_squared_error,
    mean_absolute_error,
    mean_squared_error,
    psnr,
    zero_squared_error,
)
from matrainer.metric_names import MetricNames
from matrainer.metric_slots import merge_metric_specs
from matrainer.tracking.base import Tracker
from recordstream import RecordSource

from distilaide.restoration_base import RestorationBase

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
class JaxRestorer(RestorationBase, JaxLoopRunnable):
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
        loss: An ``optax.losses`` name or a callable; ``None`` picks ``l1``.
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
        model: Optional[Mandatory[Lazy[nnx.Module]]] = None,
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
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            jit=jit,
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
    def _default_jax_loss(self) -> Optional[JaxLoss]:
        """``l1`` matches the torch backends' ``L1Loss`` exactly (the modern-recipe default)."""
        return "l1"

    def _default_jax_metrics(self) -> Optional[List[Any]]:
        return list(DEFAULT_METRICS)

    def _loss_fn(self) -> Callable[..., Any]:
        """The loss meaned over an ELEMENTWISE callable — with the written-out ``l1``.

        ``"l1"`` is resolved HERE rather than from optax, because optax does not have one (its
        loss set is ``l2_loss`` / ``squared_error`` / ``huber_loss`` / ``cosine_*``). Writing
        the one expression out is what keeps this backend's default the SAME as every other
        backend's; see the module docstring.
        """
        chosen = self.loss if self.loss is not None else self._default_jax_loss()
        if isinstance(chosen, str) and chosen in _LOCAL_LOSSES:
            local = _LOCAL_LOSSES[chosen]

            def compute(outputs: Any, targets: Any) -> Any:
                return jnp.mean(local(outputs, targets))

            return compute
        if isinstance(chosen, str) and getattr(optax.losses, chosen, None) is None:
            raise ValueError(
                f"Unknown loss {chosen!r} for the jax backend. Available: {sorted(_LOCAL_LOSSES)} "
                f"(written out here) plus any name in `optax.losses` — for an image-to-image "
                f"target: squared_error, huber_loss, l2_loss."
            )
        self.loss = chosen
        return super()._loss_fn()

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
