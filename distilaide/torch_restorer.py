"""``TorchRestorer`` — the same restoration run as a PLAIN PyTorch training loop.

The second backend, and the one with nothing underneath it. The Lightning backend hands the epoch
to a ``Trainer``; this one writes the loop out — ``zero_grad`` / ``backward`` / ``step``, an
explicit device, an explicit ``model.train()`` / ``model.eval()``, an explicit ``torch.no_grad()``
on the eval pass.

**Why it earns its place in a project whose point is that the config is the program.** It is the
control experiment. The claim every other backend makes is "swap the runnable, keep the config";
this one removes the training *framework* entirely and the config still does not change — same
dataset, same degradation, same ``DeepInvModel``, same ``torch.nn`` loss, same ``torchmetrics``
image metrics, same artifacts. Its ``!scope:framework=torch`` block in the example config is the
smallest in the file, which is the point. If a knob only worked because Lightning implemented it,
this backend is where that shows up.

It is also the smallest correct answer to "how do I train a denoiser on plain torch?", which is a
real question.

**What it shares with the Lightning backend, and what it does not.** The model, the losses and
metrics (:mod:`distilaide.torch_restoration`, ``framework="torch"``), the batch shape (a
``[N, C, H, W]`` degraded/clean PAIR via ``collate_records``) and the run artifacts are the SAME
objects. What differs is the loop and what a loop implies: there is no ``Trainer``, so there is no
``lightning:`` slot and no framework loggers (tracking goes through ``trackers:``, which every
backend has), no callback protocol, and no distributed strategy. Multi-GPU is deliberately out of
scope — that is precisely what Lightning is for, and pretending otherwise here would be a worse
``Trainer``.

The ``data_range`` resolution and the metric slots are DUPLICATED from the Lightning backend rather
than shared, deliberately and temporarily: an engine-neutral base is deferred until the whole
backend set exists, so it can be designed against every engine instead of one. See ``TASKS.md``.

Minimal flat YAML — the Lightning block with one key changed::

    runnable: !class:distilaide.TorchRestorer
    model: !lazy:distilaide.DeepInvModel { arch: dncnn }
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    data_range: 1.0
    max_epochs: 5

Run it with ``distilaide train cfg.yaml --framework torch``.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

import torch
import torch.nn as nn
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.metric_slots import merge_metric_specs
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.torch import Device, build_metric_collection, resolve_device, scalar_metrics
from matrainer.tracking.base import Tracker, as_tracker_list
from matrainer.tracking.provenance import capture_log_file
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

#: Which entry point :meth:`TorchRestorer.run` dispatches to — the same vocabulary every other
#: backend uses, because the verb is a property of the TASK, not of the engine.
RestorerTask = Literal["fit", "evaluate", "test", "predict"]

#: What a metrics slot admits. Declared here rather than imported from the Lightning backend for
#: the reason this whole module exists — that import would pull ``pytorch_lightning`` into a
#: deliberately framework-free one (``test_it_needs_no_training_framework`` scans for exactly it).
MetricsSpec = Optional[Union[Dict[str, Union[Metric, LazyClass[Metric]]], MetricCollection]]

#: Re-exported so ``TorchRestorer``'s ``device`` knob can be read from this module the way every
#: other name in its signature can. It is matrainer's: a ``Trainer`` picks an accelerator for you, a
#: hand-written loop has to pick one itself, and every such loop picks identically — so the choice
#: is a torch-platform helper, not this backend's private function.
__all__ = ["Device", "MetricsSpec", "RestorerTask", "TorchRestorer"]


@configurable(task="restoration", role="trainer", framework="torch")
class TorchRestorer(TorchRunner, ProgressReporting):
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

    Post-construction slots (YAML / broadcasting): ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``loss_fn``, ``predictions_sink``, ``metrics_sidecar_path``.
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
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        metric_names: Optional[Lazy[MetricNames]] = None,
    ) -> None:
        super().__init__()
        # `nn.Module`, not `Lazy[nn.Module]`: the DEFERRED-ness is declared once, on the ctor
        # param above, which is what `lazy_param_names` / `to_pydantic` read. Restating it here
        # would make the attribute a union no method could call through.
        self.model: nn.Module = cast(nn.Module, model)
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
        self.device_choice: Device = device
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        # Deferred runtime slots, built as `LazyClass` so a config can override any DataLoader
        # knob (`pin_memory`, `drop_last`, a sampler) without this class enumerating them — the
        # same construction the Lightning backend uses.
        loader_kw: Dict[str, Any] = dict(
            collate_fn=collate_records,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.optimizer: Lazy[Optimizer] = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        # Sentinels: "not chosen" must stay distinguishable from "chosen to be the default", or an
        # explicitly wired MSE loss would be silently replaced by this class's L1.
        self.loss_fn: Optional[Lazy[nn.Module]] = None
        self.metrics: Any = metrics
        # Dual-purpose the way `model` is: that split's metrics SPEC until `initialize()` builds
        # it, the live prefixed collection after. `_metrics_built` tells the two states apart —
        # `is None` cannot, once a slot may legitimately arrive carrying a spec.
        self.train_metrics: Any = train_metrics
        self.val_metrics: Any = val_metrics
        self.test_metrics: Any = test_metrics
        self._metrics_built: bool = False
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        # Stored VERBATIM (lazy-init mandate: the constructor does no work).
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        # Stored VERBATIM for the same reason `trackers` is: wired from YAML it arrives as a
        # deferred marker, and `as_metric_names()` builds it where the metrics are reported.
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}
        self._device: torch.device = torch.device("cpu")

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table (recordstream docs/runnable.md).
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Build the model and the metrics, and place everything on a device."""
        self._device = resolve_device(self.device_choice)
        logger.info(f"distilaide torch backend on {self._device}.")
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)

        # The model is BUILT by this `flow()` whether it arrived deferred or live — confluid fires
        # `solidify()` on its idempotency pass-through too, so a config wiring
        # `!class:distilaide.DeepInvModel()` cannot reach the optimizer with an unbuilt network and
        # NO parameters.
        self.model = cast(nn.Module, flow(self.model))
        if not isinstance(self.model, nn.Module):
            raise TypeError(
                f"TorchRestorer.model must be a torch.nn.Module, got {type(self.model).__name__}. "
                "Wire `!lazy:distilaide.DeepInvModel` (or any nn.Module) into the model slot."
            )
        if self.task == "fit":
            self._require_a_trainable_model()
        self.model.to(self._device)
        self._resolve_task_defaults()
        if not self._metrics_built:
            self.train_metrics = self._build_split_metrics(self.train_metrics, "train/")
            self.val_metrics = self._build_split_metrics(self.val_metrics, "val/")
            self.test_metrics = self._build_split_metrics(self.test_metrics, "test/")
            self._metrics_built = True
        loss = flow(self.loss_fn)
        if isinstance(loss, nn.Module):
            # A loss holding state (a learned perceptual loss carries a whole network) must be on
            # the device its inputs are, or it raises mid-batch rather than at setup.
            self.loss_fn = loss.to(self._device)

    def _require_a_trainable_model(self) -> None:
        """Refuse ``task="fit"`` over a parameter-free model, by name and with the way out.

        Identical to the Lightning backend's check and for the identical reason: the restoration
        zoo holds fixed ALGORITHMS (BM3D, total variation, a median filter) wearing the
        ``nn.Module`` interface with zero parameters, and handing one to a fit surfaces as torch's
        *"optimizer got an empty parameter list"* several frames from the ``arch:`` line that
        caused it. The two backends exist to be compared, so a run refused by one and accepted by
        the other would make the comparison a lie.
        """
        if any(p.requires_grad for p in self.model.parameters()):
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

        Identical choices to the Lightning backend, and that is the point: a config moved between
        the two must not silently change its loss. Idempotent, so an explicitly wired value is
        never replaced.
        """
        if self.loss_fn is None:
            self.loss_fn = nn.L1Loss()
            logger.debug("loss_fn defaulted to L1Loss.")
        if self.metrics is None:
            self.metrics = {
                "psnr": LazyClass[Metric](PeakSignalNoiseRatio),
                "ssim": LazyClass[Metric](StructuralSimilarityIndexMeasure),
            }

    def _build_split_metrics(self, spec: Any, prefix: str) -> MetricCollection:
        """One split's live, prefixed collection — its own spec merged over the shared one.

        The combination rule is :func:`~matrainer.metric_slots.merge_metric_specs`, shared with
        every other backend so a config moving between them cannot change shape; the flow
        dimension is ``data_range``, which torchmetrics 1.9 makes a REQUIRED argument of
        ``PeakSignalNoiseRatio`` — so a ``!lazy:`` template that never received one fails to build
        rather than quietly scoring against an inferred range.
        """
        return build_metric_collection(merge_metric_specs(self.metrics, spec), data_range=self.data_range).clone(
            prefix=prefix
        )

    # ---- the batch boundary -------------------------------------------------- #
    def _targets(self, batch: Any) -> torch.Tensor:
        """The clean target IMAGE the loss compares against — ``[N, C, H, W]`` float.

        The same reader the input uses, with no dtype argument: a restoration target is another
        image produced by the same pipeline as the input, so naming a dtype here could only
        introduce a disagreement between the two halves of one pair.
        """
        return batch_tensor(batch, self.target_key, self._device)

    def _loaders(self) -> Dict[str, Optional[DataLoader[Any]]]:
        """The deferred loader slots, flowed with the datasets this run resolved."""
        return {
            "train": flow(self.train_loader, dataset=self.train_set) if self.train_set is not None else None,
            "val": flow(self.val_loader, dataset=self.val_set) if self.val_set is not None else None,
            "test": flow(self.test_loader, dataset=self.test_set) if self.test_set is not None else None,
        }

    # ---- the loop ------------------------------------------------------------ #
    def _train_epoch(self, loader: DataLoader[Any], optimizer: Optimizer) -> float:
        """One training pass, returning the mean loss. The whole reason this backend exists."""
        model, loss_fn = self.model, cast(nn.Module, flow(self.loss_fn))
        model.train()
        total, batches = 0.0, 0
        for batch in loader:
            x = batch_tensor(batch, self.input_key, self._device)
            y = self._targets(batch)
            optimizer.zero_grad(set_to_none=True)
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
            if self.train_metrics is not None:
                self.train_metrics.update(preds.detach(), y)
        return total / batches if batches else float("nan")

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader[Any], metrics: Optional[MetricCollection]) -> float:
        """One evaluation pass, returning the mean loss. ``no_grad`` because nothing here trains."""
        model, loss_fn = self.model, cast(nn.Module, flow(self.loss_fn))
        model.eval()
        total, batches = 0.0, 0
        for batch in loader:
            x = batch_tensor(batch, self.input_key, self._device)
            y = self._targets(batch)
            preds = model(x)
            total += float(loss_fn(preds, y))
            batches += 1
            if metrics is not None:
                metrics.update(preds, y)
        return total / batches if batches else float("nan")

    def _epoch_metrics(self, collection: Optional[MetricCollection]) -> Dict[str, float]:
        """Compute and RESET a split's metrics — the bookkeeping a ``Trainer`` hides.

        Accumulating over the split and reducing ONCE is load-bearing for this task specifically:
        PSNR is a logarithm of a mean squared error, so averaging per-batch PSNRs is not the PSNR
        of the split, and the two answers drift apart as the batches differ from one another.
        Updating per batch and computing here is what keeps two runs at different batch sizes
        comparable — and it is what the Lightning backend's ``_report_epoch_metrics`` does, so the
        two backends' columns mean the same thing.
        """
        if collection is None:
            return {}
        values = scalar_metrics(collection.compute())
        collection.reset()
        return values

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        """Epochs, in the open: forward, backward, step, validate, log, checkpoint."""
        logger.info(f"distilaide torch training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("TorchRestorer.fit() requires a 'train_set'.")
        loaders = self._loaders()
        train_loader = cast(DataLoader[Any], loaders["train"])
        optimizer = cast(Optimizer, flow(self.optimizer, params=self.model.parameters()))
        trackers = self._live_trackers()
        manifest = self._log_run_context(trackers)
        self._move_metrics()

        for epoch in range(self.max_epochs):
            logs: Dict[str, float] = {"train/loss": self._train_epoch(train_loader, optimizer)}
            logs.update(self._epoch_metrics(self.train_metrics))
            if loaders["val"] is not None:
                logs["val/loss"] = self._eval_epoch(cast(DataLoader[Any], loaders["val"]), self.val_metrics)
                logs.update(self._epoch_metrics(self.val_metrics))
            logs = as_metric_names(self.metric_names)(logs)
            for tracker in trackers:
                tracker.log_metrics(logs, step=epoch)
            for key, value in logs.items():
                self._history.setdefault(key, []).append(value)
            logger.info(f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))
            self._computed_metrics = dict(logs)

        logger.info("Training complete.")
        self._save_checkpoint()
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        if loaders["test"] is not None:
            self._computed_metrics.update(self._score(cast(DataLoader[Any], loaders["test"])))
        # Uploads the log file, stamps the manifest's terminal status AND finalizes every tracker.
        # Not optional once a manifest exists: left unfinished it stays `running`, and matrainer's
        # atexit hook then calls a successful run `interrupted`.
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        """Score the VALIDATION split — the `evaluate` / `test` split is the only difference."""
        if self.val_set is None:
            raise ValueError("TorchRestorer.evaluate() requires a 'val_set'.")
        self._move_metrics()
        loader = cast(DataLoader[Any], self._loaders()["val"])
        self._computed_metrics = self._score(loader, split="val", metrics=self.val_metrics)
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("TorchRestorer.test() requires a 'test_set'.")
        self._move_metrics()
        self._computed_metrics = self._score(cast(DataLoader[Any], self._loaders()["test"]))
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def _score(
        self, loader: DataLoader[Any], split: str = "test", metrics: Optional[MetricCollection] = None
    ) -> Dict[str, float]:
        """Evaluate over ``loader`` and report — shared by ``test``, ``evaluate`` and the post-fit pass."""
        trackers = self._live_trackers()
        collection = metrics if metrics is not None else self.test_metrics
        results = {f"{split}/loss": self._eval_epoch(loader, collection)}
        results.update(self._epoch_metrics(collection))
        results = as_metric_names(self.metric_names)(results)
        for tracker in trackers:
            tracker.log_metrics(results)
        logger.info(f"Evaluation complete: {results}")
        return results

    @entrypoint("predict", role="predictor", primary=True)
    @torch.no_grad()
    def predict(self) -> None:
        """Stream per-record restored images to the sink, pairing each batch with its metadata."""
        logger.info(f"distilaide torch predicting: {self.experiment_name} ({self.run_name})")
        if self.test_set is None:
            raise ValueError("TorchRestorer.predict() requires a 'test_set'.")
        self.model.eval()
        sink = flow(self.predictions_sink)
        loader = cast(DataLoader[Any], self._loaders()["test"])
        for batch in loader:
            out = restoration_output(self.model(batch_tensor(batch, self.input_key, self._device)))
            if sink is not None:
                self._write_predictions(out, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, out: RestorationOutput[torch.Tensor], batch: Any, sink: PredictionsSink) -> None:
        """One prediction per record, sliced from the batched output the model emits.

        The model returns ``image`` ``[N, C, H, W]`` while the sink contract is per-record, so
        handing the batch over whole would write ONE entry for N records with the sink reading row
        0 as if it were the whole prediction.
        """
        per_record_meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
        for i in range(len(out["image"])):
            item: RestorationOutput[torch.Tensor] = {"image": out["image"][i]}
            meta = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            sink.write(item, meta)

    # ---- artifacts + tracking ----------------------------------------------- #
    def _save_checkpoint(self) -> None:
        """A plain ``state_dict`` — no framework format, because there is no framework."""
        if not self.checkpoint_dir:
            return
        directory = Path(self.checkpoint_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.pt"
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved weights to {path}")

    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot.

        Written back rather than returned fresh each call: ``flow()`` on a LIVE object returns
        that same object (idempotent), but a deferred ``Fluid`` would be REBUILT on every call —
        two MLflow runs for one training run.
        """
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    def _log_run_context(self, trackers: List[Tracker]) -> Any:
        """Provenance through matrainer's plain FUNCTIONS — hookup style (1).

        There is no callback protocol to hang a ``TrackingCallback`` on, which is exactly the case
        matrainer's function-level API exists for. ``runnable=self`` rather than a hand-written
        params dict: matrainer derives the hyperparameters from the flowed object graph and dumps
        the resolved config, so this backend and the Lightning one cannot answer "what were the
        hyperparameters?" differently. Returns the run manifest so ``fit()`` can finish it.
        """
        if not trackers:
            return None
        from matrainer.tracking.provenance import log_run_context

        return log_run_context(trackers, {"device": str(self._device)}, runnable=self)

    def _move_metrics(self) -> None:
        """Metric state lives on the same device as the images it consumes."""
        for collection in (self.train_metrics, self.val_metrics, self.test_metrics):
            if collection is not None:
                collection.to(self._device)

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def trained_model(self) -> nn.Module:
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
