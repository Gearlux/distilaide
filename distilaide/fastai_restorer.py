"""``FastaiRestorer`` — the same restoration run with a fastai ``Learner``.

The fastai member of the restoration fleet, assembled from the workspace trainer bases
(2026-08-07): the Learner machinery — the ``DataLoaders``/``Learner`` assembly, the positional
``_to_tuple`` collate, the Recorder read-back into canonical names, the plain-function tracking
hookup — is ``matrainer.fastai.FastaiRunnable``, and the restoration task is the SAME
:mod:`distilaide.restoration_base` + :mod:`distilaide.restoration_torch` pair every torch-family
backend runs on. What remains HERE is the config surface (the full flat constructor — the
confluid introspection reads THIS signature, ``data_range`` included) and the task's fastai
VOCABULARY: the honest default metric pair (fastai ships no PSNR and no SSIM — ``mse`` / ``mae``
are what its regression family offers), and the ``psnr`` CONFIGURED ALIAS.

**The alias is the sanctioned exception to the metrics-are-names rule** — it wraps nothing
(there is no upstream fastai metric to wrap): :class:`PeakSignalNoiseRatio` supplies the
ACCUMULATOR, exactly as fastai's own ``DiceMulti`` does for its quantity, and the ARITHMETIC is
:mod:`matrainer.metric_functions`' verbatim, shared with the other owned loops so the column
means one thing in every row. PSNR is a LOGARITHM of a mean squared error, so an ``AvgMetric``
over a per-batch value would be one line and WRONG in a way that moves with the batch size.
SSIM stays absent and that is honest: there is no shared implementation to accumulate.

Run it with ``distilaide train cfg.yaml --framework fastai``.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from confluid import Lazy, Mandatory, configurable
from fastai.callback.core import Callback as FastaiCallback
from fastai.metrics import Metric
from loggair import get_logger
from matrainer.fastai import FastaiLoss, FastaiMetric, FastaiOptimizer, FastaiRunnable
from matrainer.metric_functions import accumulate_squared_error, psnr, zero_squared_error
from matrainer.metric_names import MetricNames
from matrainer.tracking.base import Tracker
from recordstream import RecordSource
from recordstream.runnable import RunnableTask

from distilaide.restoration_torch import RestorationTorchHooks

logger = get_logger(__name__)

#: Which entry point :meth:`FastaiRestorer.run` dispatches to (the standard four).
RestorerTask = RunnableTask

#: What ``metrics`` defaults to. fastai ships NO PSNR and NO SSIM, so the honest default is the
#: pixel-error pair its regression vocabulary does have — narrower than the other backends' by a
#: property of fastai, not of this project.
DEFAULT_METRICS: Tuple[str, ...] = ("mse", "mae")

__all__ = [
    "DEFAULT_METRICS",
    "FastaiMetric",
    "FastaiOptimizer",
    "FastaiRestorer",
    "PeakSignalNoiseRatio",
    "RestorerTask",
]


@configurable(task="restoration", role="trainer", framework="torch")
class FastaiRestorer(RestorationTorchHooks, FastaiRunnable):
    """Train, evaluate or predict an image restorer with a fastai ``Learner``.

    Args:
        model: Restoration model mapping ``[N, C, H, W]`` degraded images to restored ones.
        train_set: Training source (``task="fit"``).
        test_set: Held-out source (``task="test"`` / ``"predict"``).
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        num_workers: DataLoader worker processes.
        learning_rate: Peak LR for ``fit_one_cycle``, or the flat LR for ``fit``.
        weight_decay: fastai's ``wd``; ``None`` is the ``Learner`` default.
        one_cycle: Use ``fit_one_cycle`` (the fastai recipe) instead of a flat ``fit``.
        opt_func: fastai optimizer — a callable or a name in ``fastai.optimizer``.
        loss: Loss callable; ``None`` picks ``torch.nn.L1Loss`` (the task default — an
            image-to-image loss has no class dimension, so fastai's ``*Flat`` wrappers change
            nothing and the plain ``torch.nn`` loss every other backend uses stays).
        metrics: fastai metrics (callables or names). ``None`` picks :data:`DEFAULT_METRICS`.
        val_metrics / test_metrics: That split's OWN metrics, combined over ``metrics`` (the
            fit path UNIONS — a Learner reports one list on both passes — and an eval path
            narrows).
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. Unlike the torch backends this one
            has a real translation to do: a ``Recorder`` writes ``valid_loss`` and leaves its
            metric columns UNPREFIXED, where every other engine's bare key is a training one.
        callbacks: fastai callbacks.
        trackers: matrainer trackers — one or a list.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the degraded image.
        target_key: Record key holding the clean target image.
        data_range: The span of the target images' values — injected into the ``psnr`` alias;
            fastai's own vocabulary (mse / mae / rmse / msle) is scale-free and ignores it.

    Optional OVERRIDE slots — every one is already constructed with a sensible default in
    ``__init__``, so nothing here NEEDS wiring; a same-named top-level YAML key replaces one
    via broadcasting when you want to: ``val_set``, ``train/val/test_loader``, ``dataloaders``,
    ``learner``, ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Optional[Mandatory[Lazy[torch.nn.Module]]] = None,
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        task: RestorerTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        num_workers: int = 0,
        learning_rate: float = 1e-3,
        weight_decay: Optional[float] = None,
        one_cycle: bool = True,
        opt_func: FastaiOptimizer = "Adam",
        loss: Optional[Lazy[FastaiLoss]] = None,
        metrics: Optional[List[FastaiMetric]] = None,
        val_metrics: Optional[List[FastaiMetric]] = None,
        test_metrics: Optional[List[FastaiMetric]] = None,
        metric_names: Optional[Lazy[MetricNames]] = None,
        callbacks: Optional[List[Lazy[FastaiCallback]]] = None,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        experiment_name: str = "distilaide_restoration",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        data_range: float = 1.0,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            trackers=trackers,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metrics=metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            one_cycle=one_cycle,
            opt_func=opt_func,
            loss=loss,
            callbacks=callbacks,
        )
        self.data_range = float(data_range)

    def _default_fastai_metrics(self) -> List[FastaiMetric]:
        return list(DEFAULT_METRICS)

    def _metric_aliases(self) -> Dict[str, Callable[[], Any]]:
        return {"psnr": lambda: PeakSignalNoiseRatio(data_range=self.data_range)}

    def _metric_vocabulary_hint(self) -> str:
        return (
            "For an image-to-image target: mse, mae, rmse, msle — plus 'psnr', which this backend "
            "supplies because fastai ships no scale-aware image metric. Note there is still NO ssim "
            "here; use the lightning or torch backend for that."
        )


class PeakSignalNoiseRatio(Metric):
    """PSNR over the whole split — the one metric fastai has no name for.

    ``fastai.metrics`` publishes ``mse`` / ``mae`` / ``rmse`` / ``msle`` and nothing scale-aware, so
    the quantity every OTHER backend here reports as ``psnr`` was simply unreachable by NAME. In a
    cross-backend sweep that is the single blank cell in the headline column, which is what made it
    worth closing.

    **Why a CLASS when the module docstring says metrics are names.** That rule is about not
    wrapping an upstream metric to make it registrable, and this wraps nothing: it is the
    ACCUMULATOR fastai does not ship for this quantity. The arithmetic is
    :mod:`matrainer.metric_functions`', shared verbatim with the other owned loops, so the column
    means one thing in every row.

    **The accumulation is the whole point, not tidiness.** PSNR is a LOGARITHM of a mean squared
    error, so the mean of per-batch PSNRs is not the PSNR of the split — an `AvgMetric` over a
    per-batch value would have been one line and wrong, in a way that MOVES WITH THE BATCH SIZE.
    Accumulating the squared error and converting once is exact, and it is the shape fastai's own
    ``DiceMulti`` uses for the same reason.

    The name is ``psnr`` because that is what the other seven engines call it; a metric reported
    under its own class name would open a column no other row fills.
    """

    def __init__(self, data_range: float = 1.0) -> None:
        self.data_range = float(data_range)

    def reset(self) -> None:
        self.totals = zero_squared_error()

    def accumulate(self, learn: Any) -> None:
        # `.detach().cpu()` is REQUIRED, not defensive. `matrainer.metric_functions` is deliberately
        # framework-free — numpy is its only import — and `np.asarray` on a live accelerator tensor
        # raises *"can't convert mps:0 device type tensor to numpy"* (hit for real on the first
        # sweep; a CPU-tensor unit test passes happily). The other owned loops that share this
        # accumulator hand it arrays numpy can already read, so the conversion belongs at the ONE
        # call site that is torch — teaching matrainer about `.cpu()` would mean teaching a tracking
        # library what a device is.
        accumulate_squared_error(self.totals, learn.pred.detach().cpu(), learn.y.detach().cpu())

    @property
    def value(self) -> float:
        return psnr(self.totals, self.data_range)

    @property
    def name(self) -> str:
        return "psnr"
