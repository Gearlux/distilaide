"""LightningRestorer: a standard PyTorch-Lightning image restorer (train + eval in one class).

The Lightning member of the restoration fleet, assembled from the workspace trainer bases
(2026-08-06): the Lightning machinery — loaders, the deferred ``pl.Trainer`` slot, the step
functions, tracking — is ``matrainer.lightning.LightningRunnable``, and the restoration task —
the image-pair target, the ``data_range``-sized PSNR/SSIM defaults, the trainable-model gate —
is :mod:`distilaide.restoration_base` + :mod:`distilaide.restoration_torch`. What remains HERE
is only the config surface: the full flat constructor (the confluid introspection reads THIS
signature) storing the one task knob, ``data_range``.

Minimal flat YAML (top-level keys wired by Confluid broadcasting)::

    runnable: !class:distilaide.LightningRestorer
      task: fit
    model: !lazy:distilaide.DeepInvModel { arch: dncnn }
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    val_set:   !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    batch_size: 8
    max_epochs: 20

    distilaide train train.yaml      # or: recordstream run train.yaml
"""

from typing import Iterable, Optional, Union

import torch.nn as nn
from confluid import Mandatory, Partial, configurable
from loggair import get_logger
from matrainer.lightning import LightningRunnable
from matrainer.metric_names import MetricNames
from matrainer.torch.metrics import MetricsSpec
from matrainer.tracking.base import Tracker
from recordstream import RecordSource
from recordstream.runnable import RunnableTask

from distilaide.restoration_torch import RestorationTorchHooks

logger = get_logger(__name__)

#: Which entry point :meth:`LightningRestorer.run` dispatches to (the standard four).
RestorerTask = RunnableTask


@configurable(task="restoration", role="trainer", framework="torch")
class LightningRestorer(RestorationTorchHooks, LightningRunnable):
    """A standard Lightning image restorer — train, evaluate, test, or predict via ``task``.

    Args:
        model: Restoration model mapping ``[N, C, H, W]`` degraded images to restored ones of the
            same shape (wire ``!lazy:`` so the build happens at run time rather than at config
            load, i.e. so a config that is merely inspected downloads nothing).
        train_set: Training dataset / Stream (for ``task="fit"``).
        test_set: Held-out eval dataset / Stream (for ``task="test"`` / ``"predict"``; also tested
            after ``fit`` when set).
        data_range: The span of the target images' values — ``1.0`` for the ``[0, 1]`` floats a
            ``Scale`` op produces, ``255.0`` for raw ``uint8``. Injected
            into every metric template, and NOT a formality: PSNR is defined relative to it, so a
            wrong value shifts every score by a constant nobody notices (measured on one batch of
            ``[0, 1]`` images — 20.46 dB at ``1.0``, **68.59 dB** at ``255.0``, same tensors).
        task: Entry point :meth:`run` dispatches to — ``"fit"`` / ``"evaluate"`` / ``"test"`` / ``"predict"``.
        max_epochs / batch_size / num_workers: ergonomic flat-config knobs.
        persistent_workers: Override the pairing derived from ``num_workers`` (``None`` derives
            ``num_workers != 0``). An explicit value is for a HOST fact no derivation can see —
            macOS terminates persistent workers slowly, so a config there sets ``false`` while
            keeping its workers. ``True`` with ``num_workers=0`` is refused.
        trackers: matrainer trackers — one or a list. They ARE this run's metric sinks: each is
            adapted to Lightning's logger interface, so there is no separate ``logger:`` slot of
            ``pytorch_lightning.loggers``.
        experiment_name / run_name: tracking identity.
        input_key: Record key carrying the DEGRADED model input (default ``"image"``).
        target_key: Record key carrying the CLEAN target image (default ``"target"``).
        metrics: Single-metric templates flowed per split (default PSNR + SSIM).
        train_metrics / val_metrics / test_metrics: That split's OWN templates, merged over
            ``metrics`` — how a metric is added to one split only (a costly MS-SSIM on validation).
            Each slot holds the spec until :meth:`initialize` builds it, and the live prefixed
            ``MetricCollection`` after.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. The ``self.log`` calls already write the
            canonical ``<split>/<name>``, so the default translates nothing; it is handed to each
            ``TrackerLogger``, where a wired ``target:`` records the same run in another engine's
            spelling.

    Optional OVERRIDE slots — every one is already constructed with a sensible default in
    ``__init__`` (the task-shaped ones, ``loss_fn`` + the metric slots, resolve to this task's
    defaults in ``initialize()``), so nothing here NEEDS wiring; a same-named top-level YAML key
    replaces one via broadcasting when you want to. ``val_set`` is the one a config typically
    does wire (there is no sensible default validation split): ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``lightning``, ``loss_fn``, ``predictions_sink``,
    ``metrics_sidecar_path``, ``log_system_metrics``.
    """

    def __init__(
        self,
        model: Optional[Mandatory[Partial[nn.Module]]] = None,
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        data_range: float = 1.0,
        task: RestorerTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        num_workers: int = 0,
        persistent_workers: Optional[bool] = None,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        experiment_name: str = "distilaide_restoration",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metrics: MetricsSpec = None,
        train_metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
        metric_names: Optional[Partial[MetricNames]] = None,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            trackers=trackers,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metrics=metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
        )
        self.data_range = float(data_range)


__all__ = ["LightningRestorer", "MetricsSpec", "RestorerTask"]
