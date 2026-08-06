"""LightningRestorer: a standard PyTorch-Lightning image restorer (train + eval in one class).

A self-contained ``pytorch_lightning.LightningModule`` — its own ``training_step`` /
``validation_step`` / ``test_step`` / ``predict_step`` / ``configure_optimizers``, and a ``run()``
that builds the loaders + ``pl.Trainer`` and calls ``fit`` / ``validate`` / ``test`` / ``predict``
per the ``task`` knob. It reads like a textbook Lightning example; matrainer is used ONLY for
optional experiment tracking and a couple of small utilities. Remove matrainer and this class still
trains — you lose the rich tracking, not the training.

Image-to-image specifics it owns: reading a TARGET IMAGE rather than a label, resolving the
``data_range`` every full-reference metric needs, an L1 pixel loss, and the PSNR / SSIM pair sized
against that range.

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

from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.lightning import (
    TrackerLogger,
    TrackingCallback,
    activate_for_lightning,
    attach_lightning_progress,
    build_metric_collection,
    scalar_metrics,
)
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.metric_slots import merge_metric_specs
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.tracking.base import Tracker, as_tracker_list
from recordstream import (
    PredictionsSink,
    RecordSource,
    RestorationOutput,
    batch_metadata,
    batch_tensor,
    collate_records,
    ensure_materialized,
    ensure_record_dataset,
    restoration_output,
)
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint, run_entrypoint
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

logger = get_logger(__name__)

#: Which entry point :meth:`LightningRestorer.run` dispatches to.
RestorerTask = Literal["fit", "evaluate", "test", "predict"]

#: What a metrics slot admits: a ``{name: template}`` mapping, a ready-made collection, or ``None``
#: for "not chosen". Named because FOUR slots share it (the shared set + one per split) and
#: `to_pydantic` reads the annotation for navigaitor's form-spec / StreamStudio widgets.
MetricsSpec = Optional[Union[Dict[str, Union[Metric, LazyClass[Metric]]], MetricCollection]]


def _per_record_images(preds: Any) -> List[Any]:
    """Split a batched model output into ONE entry per record, for the per-record sink contract.

    A restorer emits the whole batch at once (``image`` ``[N, C, H, W]``) while
    ``PredictionsSink.write(prediction, metadata)`` is per record — so a batched
    :class:`~recordstream.RestorationOutput` is sliced row-wise here. Handing the batch over whole
    writes ONE entry for an N-record batch, with the sink reading row 0 as if it were the whole
    prediction.
    """
    if isinstance(preds, list):
        return preds
    if isinstance(preds, dict):
        lengths = {len(v) for v in preds.values() if hasattr(v, "__len__")}
        if len(lengths) == 1:
            n = lengths.pop()
            return [{k: v[i] for k, v in preds.items()} for i in range(n)]
    return [preds]


@configurable(task="restoration", role="trainer", framework="torch")
class LightningRestorer(TorchRunner, ProgressReporting, L.LightningModule):
    """A standard Lightning image restorer — train, evaluate, test, or predict via ``task``.

    Args:
        model: Restoration model mapping ``[N, C, H, W]`` degraded images to restored ones of the
            same shape (wire ``!lazy:`` so the build happens at run time rather than at config
            load, i.e. so a config that is merely inspected downloads nothing).
        train_set: Training dataset / Stream (for ``task="fit"``).
        test_set: Held-out eval dataset / Stream (for ``task="test"`` / ``"predict"``; also tested
            after ``fit`` when set).
        data_range: The span of the target images' values — ``1.0`` for the ``[0, 1]`` floats a
            ``ToTensor(normalize=True)`` pipeline produces, ``255.0`` for raw ``uint8``. Injected
            into every metric template, and NOT a formality: PSNR is defined relative to it, so a
            wrong value shifts every score by a constant nobody notices (measured on one batch of
            ``[0, 1]`` images — 20.46 dB at ``1.0``, **68.59 dB** at ``255.0``, same tensors).
        task: Entry point :meth:`run` dispatches to — ``"fit"`` / ``"evaluate"`` / ``"test"`` / ``"predict"``.
        max_epochs / batch_size / num_workers: ergonomic flat-config knobs.
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

    Post-construction slots (YAML / broadcasting): ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``lightning``, ``loss_fn``, ``predictions_sink``,
    ``metrics_sidecar_path``, ``log_system_metrics``.
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
        num_workers: int = 0,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        experiment_name: str = "distilaide_restoration",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metrics: MetricsSpec = None,
        train_metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
        metric_names: Optional[Lazy[MetricNames]] = None,
    ) -> None:
        super().__init__()
        self.model: Lazy[nn.Module] = model
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: RestorerTask = task
        self.data_range = float(data_range)
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.max_epochs = max_epochs
        # Deferred runtime slots — `Lazy[T]` DECLARES that the run methods flow this later, usually
        # with a kwarg that does not exist yet (`params=self.parameters()`, `dataset=`). For a
        # CONSTRUCTOR param the marker is machine-read (`confluid.lazy_param_names` reports it, and
        # liquifai's auto deep-flow SKIPS it instead of eagerly building it); for the BODY slots
        # below it is inert — deferral is enforced by the slot holding a `LazyClass` — but it still
        # earns its place by giving `to_pydantic` a real type where `Any` typed nothing at all.
        #
        # Sentinels: the metrics need `data_range` at BUILD time (torchmetrics 1.9 makes it a
        # required argument of PeakSignalNoiseRatio), and the loss is a choice the config may make.
        # `None` means "not chosen" — a live default here could not be told apart from a user's
        # explicit choice, so an explicitly wired MSE loss would be silently replaced.
        self.loss_fn: Optional[Lazy[nn.Module]] = None
        self.optimizer: Lazy[Optimizer] = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        loader_kw = dict(
            collate_fn=collate_records,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        # No `logger=` here: the trackers ARE the metric sinks, adapted to Lightning's logger
        # interface in `_build_trainer` (a Tracker may still be a deferred Fluid at this point, so
        # the adapter is built at run time, not now).
        self.lightning: Lazy[L.Trainer] = LazyClass(L.Trainer, max_epochs=max_epochs)
        self.metrics: Any = metrics
        # Per-split slots, each dual-purpose the way `model` is: they hold that split's metrics
        # SPEC until `initialize()` builds it, and the live prefixed `MetricCollection` after.
        # `_metrics_built` is what tells the two states apart — `is None` cannot, once a slot may
        # legitimately arrive carrying a spec.
        self.train_metrics: Any = train_metrics
        self.val_metrics: Any = val_metrics
        self.test_metrics: Any = test_metrics
        self._metrics_built: bool = False
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        # Stored VERBATIM (lazy-init mandate: the constructor does no work). It may hold a single
        # tracker, a list, or deferred `Fluid` markers; `_live_trackers()` normalizes and flows it
        # at run time.
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        # Stored VERBATIM for the same reason `trackers` is: wired from YAML it arrives as a
        # deferred marker, and `as_metric_names()` builds it where the metrics are reported.
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        self.log_system_metrics: bool = True

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override of the marker
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table — a hand-written
        # {task: method} dict restates them and drifts (recordstream `docs/runnable.md`).
        run_entrypoint(self, self.task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, cast(nn.Module, self.model)(x))

    # ---- standard Lightning step functions ---------------------------------- #
    def _targets(self, batch: Any) -> torch.Tensor:
        """The clean target IMAGE the loss compares against — ``[N, C, H, W]`` float.

        The one line that says image-to-image out loud. Every other task in this workspace reads a
        LABEL here (a class id, a per-pixel id map, a box set); a restoration target is another
        image, in the same layout, dtype and value range as the input — which is why the batch
        reader is the SAME `batch_tensor` call the input uses, with no dtype argument: the pipeline
        that produced the input produced this the same way, so naming a dtype here could only
        introduce a disagreement between the two halves of one pair.
        """
        return batch_tensor(batch, self.target_key, self.device)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self._resolve_task_defaults()  # no-op once resolved; keeps a directly-called step working
        x = batch_tensor(batch, self.input_key, self.device)
        y = self._targets(batch)
        preds = self(x)
        loss: torch.Tensor = cast(nn.Module, self.loss_fn)(preds, y)
        # TWO calls, one series each, and that is what keeps the name canonical: Lightning appends
        # `_step` / `_epoch` when ONE name carries both, so a single `on_step=True, on_epoch=True`
        # call records `train/loss_epoch` — a name no other backend writes, in the column a
        # cross-engine comparison uses. The per-step series is still worth having (nothing else
        # shows the loss moving WITHIN an epoch) and keeps the progress bar's live number, so it
        # gets its own explicit name instead.
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=x.shape[0])
        if isinstance(self.train_metrics, MetricCollection):
            self.train_metrics.update(preds, y)
        return loss

    def on_train_epoch_start(self) -> None:
        if isinstance(self.train_metrics, MetricCollection):
            self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self._report_epoch_metrics(self.train_metrics)

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self._resolve_task_defaults()  # no-op once resolved; keeps a directly-called step working
        x = batch_tensor(batch, self.input_key, self.device)
        y = self._targets(batch)
        preds = self(x)
        loss: torch.Tensor = cast(nn.Module, self.loss_fn)(preds, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        if isinstance(self.val_metrics, MetricCollection):
            self.val_metrics.update(preds, y)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        x = batch_tensor(batch, self.input_key, self.device)
        preds = self(x)
        if isinstance(self.test_metrics, MetricCollection):
            self.test_metrics.update(preds, self._targets(batch))
        if self.predictions_sink is not None:
            self._write_predictions(restoration_output(preds), batch)
        return preds

    def predict_step(self, batch: Any, batch_idx: int) -> RestorationOutput[torch.Tensor]:
        """Predict a batch as a :class:`~recordstream.RestorationOutput`.

        Returning the CONTRACT rather than the bare tensor is what makes the output consumable: a
        sink reads ``image`` by name and is told by the contract that the value is a picture in the
        input's range, not logits and not a residual.
        """
        out = restoration_output(self(batch_tensor(batch, self.input_key, self.device)))
        if self.predictions_sink is not None:
            self._write_predictions(out, batch)
        return out

    # ---- epoch-end metric reporting ----------------------------------------- #
    # Reset happens at epoch START, not end, so the LAST epoch's state survives the run —
    # `_write_metrics_sidecar` and the `computed_metrics` @output both `compute()` after
    # `trainer.validate()` / `.test()` has returned.
    def on_validation_epoch_start(self) -> None:
        if isinstance(self.val_metrics, MetricCollection):
            self.val_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self._report_epoch_metrics(self.val_metrics)

    def on_test_epoch_start(self) -> None:
        if isinstance(self.test_metrics, MetricCollection):
            self.test_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self._report_epoch_metrics(self.test_metrics)

    def _report_epoch_metrics(self, collection: Any) -> None:
        """Compute one split's metrics ONCE at epoch end and log the scalars.

        Accumulating over the split and reducing once is not a style choice for this task: PSNR is
        a logarithm of a mean squared error, so the mean of per-batch PSNRs is not the PSNR of the
        split, and the gap grows with how much the batches differ. Updating per batch and computing
        here is what makes the number comparable between two runs at different batch sizes.

        Every full-reference image metric here is a SCALAR, so — unlike the per-pixel and
        classification tasks — there is no vector or matrix to route around ``log_dict``.
        ``scalar_metrics`` still guards the call, because a config may wire one that is not.
        """
        if not isinstance(collection, MetricCollection):
            return
        self.log_dict(scalar_metrics(collection.compute()))

    def configure_optimizers(self) -> Any:
        return flow(self.optimizer, params=self.parameters())

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"distilaide training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("LightningRestorer.fit() requires a 'train_set'.")
        train_ds = ensure_record_dataset(self.train_set)
        val_ds = ensure_record_dataset(self.val_set) if self.val_set is not None else None
        test_ds = ensure_record_dataset(self.test_set) if self.test_set is not None else None
        t_loader = flow(self.train_loader, dataset=train_ds)
        v_loader = flow(self.val_loader, dataset=val_ds) if val_ds is not None else None
        trainer = self._build_trainer()
        trainer.fit(model=self, train_dataloaders=t_loader, val_dataloaders=v_loader)
        logger.info("Training complete.")
        if test_ds is not None:
            trainer.test(model=self, dataloaders=flow(self.test_loader, dataset=test_ds))

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self._run_eval("validate", "val_set", "val_loader", "val_metrics")

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        self._run_eval("test", "test_set", "test_loader", "test_metrics")

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        logger.info(f"distilaide predicting: {self.experiment_name} ({self.run_name})")
        if self.test_set is None:
            raise ValueError("LightningRestorer.predict() requires a 'test_set'.")
        dataset = ensure_record_dataset(self.test_set)
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)
        loader = flow(self.test_loader, dataset=dataset)
        trainer = self._build_trainer()
        trainer.predict(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()

    def _run_eval(self, stage: str, dataset_slot: str, loader_slot: str, metrics_slot: str) -> None:
        logger.info(f"distilaide evaluating ({stage}): {self.experiment_name} ({self.run_name})")
        source = getattr(self, dataset_slot)
        if source is None:
            raise ValueError(f"LightningRestorer.{stage}() requires a '{dataset_slot}'.")
        dataset = ensure_record_dataset(source)
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)
        loader = flow(getattr(self, loader_slot), dataset=dataset)
        trainer = self._build_trainer()
        runner = trainer.validate if stage == "validate" else trainer.test
        runner(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()
        self._write_metrics_sidecar(getattr(self, metrics_slot))

    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot.

        Written back rather than returned fresh each call: ``flow()`` on a LIVE object returns that
        same object (idempotent), but a deferred ``Fluid`` would be REBUILT on every call — two
        MLflow runs for one training run.
        """
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    def _build_trainer(self) -> L.Trainer:
        """Flow the ``pl.Trainer`` and attach the OPTIONAL matrainer tracking (a few lines; removable).

        The trackers ARE the Trainer's loggers: each is flowed (it may still be a deferred
        ``Fluid``) and adapted through ``TrackerLogger``, so Lightning's per-step metrics land in
        the same backends as the provenance and the manifest. There is no separate ``logger:``
        slot — one wiring, one place a run's output goes.
        """
        trackers = self._live_trackers()
        names = as_metric_names(self.metric_names)
        adapters = [TrackerLogger(t, metric_names=names) for t in trackers]
        # A runtime kwarg wins over the stored one, so an explicitly wired `lightning:` block keeps
        # every knob it set and only gains the loggers.
        trainer = cast(L.Trainer, flow(self.lightning, logger=adapters or False))
        if trackers:
            # `runnable=self` rather than a params dict: matrainer derives the hyperparameters from
            # the flowed object graph AND dumps the resolved config, so no two backends can answer
            # "what were the hyperparameters?" differently.
            cb = TrackingCallback(trackers, runnable=self)
            trainer.callbacks.append(cb)  # type: ignore[attr-defined]
        attach_lightning_progress(trainer, self)
        if self.log_system_metrics:
            activate_for_lightning(trainer)
        return trainer

    # ---- restoration specifics ----------------------------------------------- #
    def initialize(self) -> None:
        """Build the model, then the loss + the metrics that depend on ``data_range``."""
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)

        self.model = flow(self.model)
        if self.task == "fit":
            self._require_a_trainable_model()
        self._resolve_task_defaults()
        if not self._metrics_built:
            self.train_metrics = self._build_split_metrics(self.train_metrics, "train/")
            self.val_metrics = self._build_split_metrics(self.val_metrics, "val/")
            self.test_metrics = self._build_split_metrics(self.test_metrics, "test/")
            self._metrics_built = True

    def _require_a_trainable_model(self) -> None:
        """Refuse ``task="fit"`` over a parameter-free model, by name and with the way out.

        The restoration zoo contains fixed ALGORITHMS beside learned networks — BM3D, total
        variation, wavelet shrinkage, a median filter — and they wear the same ``nn.Module``
        interface while having exactly zero parameters. Handing one to a ``fit`` is not a subtle
        mistake: torch reports *"optimizer got an empty parameter list"* from inside
        ``configure_optimizers``, several frames from the ``arch:`` line that caused it, and a
        reader has no reason to connect the two.

        Checked here rather than in the model wrapper because it is the TASK that makes it wrong:
        the very same object is exactly right for ``distilaide evaluate``, which is what a
        classical baseline exists for.
        """
        if any(p.requires_grad for p in cast(nn.Module, self.model).parameters()):
            return
        arch = getattr(self.model, "arch", type(self.model).__name__)
        raise ValueError(
            f"task='fit' needs a model with trainable parameters, and {arch!r} has none — it is a "
            "fixed algorithm, not a learned one. Score it with `distilaide evaluate` (that is what "
            "a classical baseline is for), or pick a learned architecture such as arch: dncnn."
        )

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process.

        ``ensure_record_dataset`` does the first; ``ensure_materialized`` does the second, and it
        matters as soon as ``num_workers`` is non-zero. A ``DataLoader`` worker may be a FORKED
        child, and if that child is the first to touch a lazy source, the download runs there —
        reaching ``_scproxy`` / CoreFoundation on macOS, which is not fork-safe. The result is a
        SIGSEGV with no Python traceback, surfacing only as ``DataLoader worker exited
        unexpectedly``. Reading one record here, in the parent, leaves the child nothing to build.
        """
        if source is None:
            return None
        return ensure_materialized(ensure_record_dataset(source))

    def _resolve_task_defaults(self) -> None:
        """Fill the ``loss_fn`` / ``metrics`` sentinels.

        A value the user DID wire is left untouched (that is what the `None` sentinel buys: "not
        chosen" is distinguishable from "chosen to be the default"), and this is idempotent so a
        directly-invoked step function works without `initialize()`.
        """
        if self.loss_fn is None:
            # L1 rather than MSE: DnCNN's original recipe was MSE, but every modern restoration
            # network reports L1 doing better on the same data — MSE's quadratic penalty is
            # dominated by a few large errors and produces visibly blurrier output at a comparable
            # PSNR. `loss_fn: !class:MSELoss@framework=torch` is the one line back.
            self.loss_fn = nn.L1Loss()
            logger.debug("loss_fn defaulted to L1Loss.")
        if self.metrics is None:
            # The restoration pair: PSNR (the field's primary score, in dB) and SSIM (the
            # perceptual one, in [0, 1]). Both are DEFERRED templates rather than instances,
            # because `data_range` is injected into them at build time — see `_build_split_metrics`.
            self.metrics = {
                "psnr": LazyClass[Metric](PeakSignalNoiseRatio),
                "ssim": LazyClass[Metric](StructuralSimilarityIndexMeasure),
            }

    def _build_split_metrics(self, spec: Any, prefix: str) -> MetricCollection:
        """One split's live, prefixed collection — its own spec merged over the shared one.

        The combination rule is :func:`matrainer.metric_slots.merge_metric_specs`, shared with every
        other backend and with the other tasks, so a config moving between them cannot change
        shape; what is restoration's alone is the flow dimension — ``data_range`` here, where a
        per-pixel or classification task injects ``num_classes``.

        **That injection is what makes a ``!lazy:`` metric buildable at all**: torchmetrics 1.9
        declares ``PeakSignalNoiseRatio(data_range, ...)`` with no default, so a template that
        never received one raises ``TypeError: missing 1 required positional argument`` rather than
        quietly scoring against an inferred range. A template that PINS its own ``data_range``
        keeps it — the injection is a floor, and a metric deliberately scored on another scale is a
        legitimate thing to write.
        """
        return build_metric_collection(merge_metric_specs(self.metrics, spec), data_range=self.data_range).clone(
            prefix=prefix
        )

    # ---- predictions + metrics ---------------------------------------------- #
    def _write_predictions(self, preds: Any, batch: Any) -> None:
        """Write ONE prediction per record to the wired sink, paired with that record's metadata."""
        per_record_meta = (
            batch_metadata(batch, exclude=(self.input_key, self.target_key)) if isinstance(batch, dict) else None
        )
        items = _per_record_images(preds)
        for i, item in enumerate(items):
            m = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            cast(PredictionsSink, self.predictions_sink).write(item, m)

    def _write_metrics_sidecar(self, metrics: Any) -> None:
        """Hand the run's final scalars to matrainer, which owns where run artifacts land."""
        if not isinstance(metrics, MetricCollection):
            return
        try:
            payload = scalar_metrics(metrics.compute())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not compute final metrics for the sidecar: {exc}")
            return
        write_metrics_sidecar(self.metrics_sidecar_path, payload, self.experiment_name, self.run_name)

    # ---- @outputs ----------------------------------------------------------- #
    @property
    @output
    def trained_model(self) -> nn.Module:
        """The trained/live restoration model (post-run ``@output``). Raises while still deferred."""
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run(): the model is still a deferred Fluid.")
        return cast(nn.Module, self.model)

    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``."""
        if self.task == "fit":
            try:
                return scalar_metrics(dict(self.trainer.callback_metrics))
            except RuntimeError as exc:
                raise RuntimeError("computed_metrics is only available after run().") from exc
        metrics = self.val_metrics if self.task == "evaluate" else self.test_metrics
        if not isinstance(metrics, MetricCollection):
            raise RuntimeError("computed_metrics is only available after run().")
        return scalar_metrics(metrics.compute())


__all__ = ["LightningRestorer", "MetricsSpec", "RestorerTask"]
