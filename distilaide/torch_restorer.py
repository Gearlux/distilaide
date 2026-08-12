"""``TorchRestorer`` — the same restoration run as a PLAIN PyTorch training loop.

The control-experiment member of the restoration fleet, assembled from the workspace trainer
bases (2026-08-06): the hand-written loop machinery — device bracketing, the epoch driver,
scoring, checkpointing, the plain-function tracking hookup — is
``matrainer.torch.TorchLoopRunnable``, and the restoration task is the SAME
:mod:`distilaide.restoration_base` + :mod:`distilaide.restoration_torch` pair the Lightning
backend runs on — which is the control experiment executed structurally: the two backends now
share the task by construction, so "swap the runnable, keep the config" cannot drift. What
remains HERE is only the config surface: the full flat constructor (the confluid introspection
reads THIS signature) storing the one task knob, ``data_range``.

Minimal flat YAML — the Lightning block with one key changed::

    runnable: !class:distilaide.TorchRestorer
    model: !lazy:distilaide.DeepInvModel { arch: dncnn }
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    data_range: 1.0
    max_epochs: 5

Run it with ``distilaide train cfg.yaml --framework torch``.
"""

from typing import Iterable, Optional, Union

import torch.nn as nn
from confluid import Mandatory, Partial, configurable
from loggair import get_logger
from matrainer.metric_names import MetricNames
from matrainer.torch import Device, TorchLoopRunnable
from matrainer.torch.metrics import MetricsSpec
from matrainer.tracking.base import Tracker
from recordstream import RecordSource
from recordstream.runnable import RunnableTask

from distilaide.restoration_torch import RestorationTorchHooks

logger = get_logger(__name__)

#: Which entry point :meth:`TorchRestorer.run` dispatches to (the standard four).
RestorerTask = RunnableTask

__all__ = ["Device", "MetricsSpec", "RestorerTask", "TorchRestorer"]


@configurable(task="restoration", role="trainer", framework="torch")
class TorchRestorer(RestorationTorchHooks, TorchLoopRunnable):
    """Train, evaluate or predict an image restorer with a hand-written PyTorch loop.

    Args:
        model: Restoration model mapping ``[N, C, H, W]`` degraded images to restored ones.
        train_set: Training source (``task="fit"``).
        test_set: Held-out source (``task="test"`` / ``"predict"``; also tested after a fit).
        data_range: The span of the target images' values — ``1.0`` for ``[0, 1]`` floats,
            ``255.0`` for raw ``uint8``. Injected into every metric template; a wrong value shifts
            every PSNR by a constant (measured: 20.46 dB vs 68.59 dB on the same tensors).
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        num_workers: DataLoader worker processes. Persistent workers follow from this, exactly as
            in the Lightning backend — there is no separate knob.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the degraded image.
        target_key: Record key holding the clean target image.
        metrics: torchmetrics templates, flowed per split. ``None`` picks PSNR + SSIM.
        train_metrics / val_metrics / test_metrics: That split's OWN templates, merged over
            ``metrics`` — how a metric is added to one split only (a costly MS-SSIM on validation).
        device: Where to run — see :data:`Device`.
        checkpoint_dir: Directory for ``model.pt`` (a plain ``state_dict``). ``None`` writes no
            weights, which is what a smoke run wants.
        grad_clip: Clip gradients to this global norm before each ``step()``. ``None`` disables.
        trackers: matrainer trackers — one or a list.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. This loop already writes the canonical
            ``<split>/<name>``, so the default translates nothing; wiring ``target:`` is what
            makes it record in another engine's spelling.

    Optional OVERRIDE slots — every one is already constructed with a sensible default in
    ``__init__`` (the task-shaped ones, ``loss_fn`` + the metric slots, resolve to this task's
    defaults in ``initialize()``), so nothing here NEEDS wiring; a same-named top-level YAML key
    replaces one via broadcasting when you want to: ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``loss_fn``, ``predictions_sink``, ``metrics_sidecar_path``.
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
        experiment_name: str = "distilaide_restoration",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metrics: MetricsSpec = None,
        train_metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
        device: Device = "auto",
        checkpoint_dir: Optional[str] = None,
        grad_clip: Optional[float] = None,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
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
            device=device,
            checkpoint_dir=checkpoint_dir,
            grad_clip=grad_clip,
        )
        self.data_range = float(data_range)
