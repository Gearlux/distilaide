"""``FastaiRestorer`` — the same restoration run with a fastai ``Learner``.

The third backend, and the one that swaps the LOOP and nothing else. fastai runs on torch, so it
reuses :class:`~distilaide.deepinv_model.DeepInvModel` unchanged, the same ``[N, C, H, W]`` batch
shape, the same ``collate_records``, and the same torch losses — the ``fastai`` block in the example
config overrides ``runnable`` and ``metrics`` and nothing else.

What it brings that the others do not: one-cycle scheduling, a ``Recorder`` that reports every
epoch, and fastai's own metric vocabulary.

**What it owns is two things**, and the interesting one is a NON-problem that its segmentation
sibling has:

1. ``_to_tuple`` — the twin of the Lightning backend's ``_targets``. fastai unpacks a batch
   POSITIONALLY, so the batch must BE a tuple rather than the record dict a ``LightningModule``
   reads by key. For this task both halves are images, so it is two identical ``batch_tensor``
   calls with no dtype argument — the shortest version of this method in the workspace.
2. **The loss needs NO axis fix, and that asymmetry is the point.** A segmentation config must
   write ``CrossEntropyLossFlat(axis=1)``, because that loss's default ``axis=-1`` flattens the
   CHANNEL dimension — which for per-pixel classification carries the CLASSES — into the batch.
   An image-to-image loss has no class dimension at all: flattening a ``[N, C, H, W]`` pair the
   same way on both sides leaves a mean over the same elements, so ``L1LossFlat`` agrees with
   ``torch.nn.L1Loss`` EXACTLY (measured: 0.3320147693157196 from both on one batch), as do
   ``MSELossFlat`` and ``nn.MSELoss``. The default here is therefore the plain ``torch.nn`` loss
   every other backend uses, which is what keeps "the config does not change" true of the loss too.

**fastai ships no PSNR and no SSIM**, so this backend's metric vocabulary is genuinely narrower
than the others': ``mse`` / ``mae`` / ``rmse`` / ``msle`` are what ``fastai.metrics`` offers for a
regression target, and they are what the example config wires. That is a fact about fastai rather
than a gap in this project — and it is why ``metrics:`` MUST be restated in the fastai block: the
shared key holds torchmetrics TEMPLATES, which a ``Learner`` cannot consume.

**Metrics are NAMES, not ``!class:`` tags, and that is a fastai fact.** ``mse`` and ``mae`` are
plain functions, so there is nothing uniform for a tag to resolve or for confluid's registry to
tag. ``_as_metric`` resolves a string against ``fastai.metrics``, naming the module when it cannot.
Do NOT add a metric-wrapper class to make them registrable.

``from fastai.callback import schedule`` is load-bearing: ``Learner.fit_one_cycle`` is a ``@patch``
installed as that module's IMPORT SIDE EFFECT, and fastai's ``@patch`` returns ``None``, so
importing the function by name binds None. Without the import, ``learn.fit_one_cycle`` falls
through fastcore's attribute delegation to the MODEL and raises
``'DeepInvModel' object has no attribute 'fit_one_cycle'``.

Run it with ``distilaide train cfg.yaml --framework fastai``.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from fastai import metrics as fastai_metrics
from fastai import optimizer as fastai_optimizer

# `Learner.fit_one_cycle` is a `@patch` installed as this module's import SIDE EFFECT — see the
# module docstring. Do not "tidy" it away.
from fastai.callback import schedule  # noqa: F401
from fastai.callback.core import Callback as FastaiCallback
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.metrics import Metric
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_functions import accumulate_squared_error, psnr, zero_squared_error
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.metric_slots import merge_metric_specs
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.tracking.base import Tracker, as_tracker_list
from matrainer.tracking.provenance import capture_log_file, log_run_context
from recordstream import (
    PredictionsSink,
    RecordSource,
    batch_metadata,
    batch_tensor,
    collate_records,
    ensure_materialized,
    ensure_record_dataset,
    restoration_output,
)
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint, run_entrypoint
from torch.utils.data import DataLoader

logger = get_logger(__name__)

#: Which entry point :meth:`FastaiRestorer.run` dispatches to.
RestorerTask = Literal["fit", "evaluate", "test", "predict"]

#: A fastai metric: a live callable, or the NAME of one in ``fastai.metrics``. A NAME because
#: fastai's regression metrics are plain functions — there is no class for a `!class:` tag to
#: resolve or for confluid's registry to tag.
FastaiMetric = Union[str, Callable[..., Any]]

#: A fastai optimizer: a live ``OptimWrapper`` factory, or the NAME of one in ``fastai.optimizer``.
FastaiOptimizer = Union[str, Callable[..., Any]]

#: What ``metrics`` defaults to. fastai ships NO PSNR and NO SSIM, so the honest default is the
#: pixel-error pair its regression vocabulary does have — narrower than the other backends' by a
#: property of fastai, not of this project.
DEFAULT_METRICS: Tuple[str, ...] = ("mse", "mae")

__all__ = ["DEFAULT_METRICS", "FastaiMetric", "FastaiOptimizer", "FastaiRestorer", "RestorerTask"]


@configurable(task="restoration", role="trainer", framework="torch")
class FastaiRestorer(TorchRunner, ProgressReporting):
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
        loss: Loss callable; ``None`` picks ``torch.nn.L1Loss``.
        metrics: fastai metrics (callables or names). ``None`` picks :data:`DEFAULT_METRICS`.
        val_metrics / test_metrics: That split's OWN metrics, merged over ``metrics``. There is
            deliberately no ``train_metrics`` slot — see the class docstring.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. Unlike the torch backends this one has a
            real translation to do: a ``Recorder`` writes ``valid_loss`` and leaves its metric
            columns UNPREFIXED, where every other engine's bare key is a training one.
        callbacks: fastai callbacks.
        trackers: matrainer trackers — one or a list.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the degraded image and the clean target.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``train/val/test_loader``,
    ``dataloaders``, ``learner``, ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[torch.nn.Module]],
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
        loss: Optional[Lazy[Any]] = None,
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
        super().__init__()
        # `torch.nn.Module`, not `Lazy[...]`: the DEFERRED-ness is declared once, on the ctor
        # param above, which is what `lazy_param_names` / `to_pydantic` read. Restating it here
        # would make the attribute a union no method could call through.
        self.model: torch.nn.Module = cast(torch.nn.Module, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: RestorerTask = task
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.one_cycle = one_cycle
        self.opt_func: FastaiOptimizer = opt_func
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        # The span of the target's values — this task's `num_classes`, and the reason the
        # `psnr` alias below can exist at all. It arrived LAST of the eight backends because
        # fastai's own vocabulary (mse / mae / rmse / msle) is scale-free and never needed it.
        self.data_range = float(data_range)
        # Stored VERBATIM (lazy-init): entries may be deferred `Fluid` markers, flowed where used.
        self.callbacks: Optional[List[Lazy[FastaiCallback]]] = callbacks
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        # Stored VERBATIM for the same reason `trackers` is: wired from YAML it arrives as a
        # deferred marker, and `as_metric_names()` builds it where the metrics are reported.
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        # The loaders are the Lightning backend's construction, not a private one — same names,
        # same deferral, same overridability, so ONE config idiom drives both backends. The
        # divergence is exactly ONE kwarg: `collate_fn=self._to_tuple` where Lightning bakes
        # `collate_records`, because fastai unpacks a batch positionally.
        loader_kw: Dict[str, Any] = dict(
            collate_fn=self._to_tuple,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        # fastai's own two objects, deferred for the same reason the loaders are: everything this
        # backend CONSTRUCTS should be reachable from config, or every knob it does not restate —
        # a `DataLoaders` path, a `Learner`'s `moms` / `splitter` / `model_dir` — is unreachable
        # from YAML. `DataLoaders(*loaders, …)` takes its loaders POSITIONALLY, so `_dataloaders()`
        # supplies them as `flow()`'s positional runtime args.
        self.dataloaders: Lazy[DataLoaders] = LazyClass(DataLoaders)
        self.learner: Lazy[Learner] = LazyClass(Learner, lr=learning_rate, wd=weight_decay)
        # Sentinels, for the same reason every other backend uses them: a live default could not
        # be told apart from a user's explicit choice.
        self.loss: Optional[Lazy[Any]] = loss
        self.metrics: Optional[List[FastaiMetric]] = metrics
        self.val_metrics: Optional[List[FastaiMetric]] = val_metrics
        self.test_metrics: Optional[List[FastaiMetric]] = test_metrics
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self._computed_metrics: Dict[str, float] = {}
        self._learner: Optional[Learner] = None

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Build the model and fill the loss / metric sentinels."""
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        self.model = cast(torch.nn.Module, flow(self.model))
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError(
                f"FastaiRestorer.model must be a torch.nn.Module, got {type(self.model).__name__}. "
                "Wire `!lazy:distilaide.DeepInvModel` (or any nn.Module) into the model slot."
            )
        if self.task == "fit":
            self._require_a_trainable_model()
        self._resolve_task_defaults()

    def _require_a_trainable_model(self) -> None:
        """Refuse ``task="fit"`` over a parameter-free model, by name and with the way out.

        The same check the other two backends make, and it must be the same on all three: the
        restoration zoo holds fixed ALGORITHMS (BM3D, total variation, a median filter) with zero
        parameters, and a run refused by one backend and accepted by another would make the
        cross-backend comparison a lie. fastai's own failure without it is the least legible of
        the three — ``create_opt`` builds an optimizer over an empty parameter list inside
        ``fit_one_cycle``, well past the data walk.
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

        ``ensure_record_dataset`` does the first; ``ensure_materialized`` does the second, and for
        this backend it is not optional. **fastai sets the multiprocessing start method to
        ``fork`` at import**, so a ``DataLoader`` worker is a forked child — and if that child is
        the first to touch a lazy source, ``load_dataset`` runs there, reaching ``_scproxy`` /
        CoreFoundation, which is not fork-safe. The result is a **SIGSEGV with no Python
        traceback**, surfacing only as ``DataLoader worker exited unexpectedly``. Reading one
        record HERE, in the parent, leaves the child a source that needs nothing.

        Setting ``num_workers: 0`` also avoids it, by not forking at all — which is why the crash
        looks intermittent and gets misread as a flaky test rather than a real ordering bug.
        """
        if source is None:
            return None
        return ensure_materialized(ensure_record_dataset(source))

    def _resolve_task_defaults(self) -> None:
        """Fill the ``loss`` / ``metrics`` sentinels. Idempotent — a wired value is never replaced.

        **The default loss is the plain ``torch.nn`` one, not a fastai ``*Flat``**, and that is the
        interesting difference from the segmentation backend. There, ``CrossEntropyLossFlat``
        needed ``axis=1`` because its default flattens the CHANNEL dimension — the CLASS dimension
        for per-pixel classification — into the batch. An image-to-image loss has no class
        dimension, so flattening a ``[N, C, H, W]`` pair identically on both sides changes nothing:
        ``L1LossFlat`` and ``nn.L1Loss`` return the SAME value to full precision (measured). With
        no reason to use fastai's wrapper, the loss stays the one every other backend uses.
        """
        if self.loss is None:
            self.loss = nn.L1Loss()
            logger.debug("loss defaulted to torch.nn.L1Loss.")
        if self.metrics is None:
            self.metrics = list(DEFAULT_METRICS)

    # ---- the batch boundary -------------------------------------------------- #
    def _to_tuple(self, records: Sequence[Any], with_targets: bool = True) -> Tuple[torch.Tensor, ...]:
        """One list of records → the ``(x, y)`` tuple fastai consumes.

        The fastai twin of the Lightning backend's ``_targets``, and the shortest one in the
        workspace: recordstream does the work, and for an image-to-image task BOTH halves are the
        same kind of value, so there is no dtype to name and no shaping to do. Every other task's
        version of this method has a target branch; restoration's does not, because the target IS
        an image.
        """
        batch = collate_records(list(records))
        x = batch_tensor(batch, self.input_key)
        if not with_targets:
            return (x,)
        return x, batch_tensor(batch, self.target_key)

    def _dataloaders(self) -> DataLoaders:
        """fastai's ``DataLoaders`` — a train loader and a validation loader, always both.

        ``Learner`` validates at the end of every epoch and its ``Recorder`` reports from that
        pass, so unlike ``pytorch_lightning.Trainer`` it has no "no validation" mode. When no
        ``val_set`` is wired the training data is reused, which keeps a run working but makes the
        reported metrics training metrics — worth saying out loud, hence the warning.
        """
        train_ds = self._prepare(self.train_set)
        train = flow(self.train_loader, dataset=train_ds)
        if self.val_set is not None:
            valid = flow(self.val_loader, dataset=self._prepare(self.val_set))
            return cast(DataLoaders, flow(self.dataloaders, train, valid))
        logger.warning(
            "No 'val_set' wired: fastai's Learner always validates, so the TRAINING data is being "
            "used for validation and the reported metrics measure fit, not generalization. Wire a "
            "'val_set' to get a real held-out score."
        )
        return cast(DataLoaders, flow(self.dataloaders, train, flow(self.val_loader, dataset=train_ds)))

    # ---- the learner -------------------------------------------------------- #
    def _build_learner(self, dataloaders: DataLoaders, split: str = "val") -> Learner:
        """Assemble the ``Learner``: model, loss, metrics, callbacks, artifact path.

        ``split`` picks which metric slot is merged over the shared ``metrics`` — a ``Learner``
        holds ONE list and computes it on the validation pass, so the split is decided by which
        entry point is building it rather than by anything inside the Learner.
        """
        path = self._artifact_dir()
        learner = cast(
            Learner,
            flow(
                self.learner,
                dls=dataloaders,
                model=self.model,
                loss_func=flow(self.loss),
                opt_func=_as_opt_func(self.opt_func),
                metrics=[
                    _as_metric(m, self.data_range)
                    for m in (merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or [])
                ],
                cbs=self._callbacks(),
                path=Path(path) if path else None,
            ),
        )
        # A `Recorder` computes its metrics on the VALIDATION pass unless told otherwise, and
        # `train_metrics=True` makes it accumulate the SAME list over the TRAINING batches as they
        # go by — no second pass over the data, and the numbers describe the network as it trained.
        # Without it the backend sweep's `train/psnr` and `train/mse` columns were blank on the two
        # fastai rows alone, while every other backend filled them.
        #
        # Do NOT reach for a second eval-mode pass instead: it costs a pass AND measures a different
        # network (dropout, batch norm), so an expected train/val gap would read as a bug.
        #
        # The columns become `train_loss, train_<m>…, valid_loss, valid_<m>…`, which is exactly the
        # shape `matrainer.metric_names`' fastai vocabulary already reads, so nothing downstream
        # changes. It also works for `psnr` specifically because the alias ACCUMULATES — a per-batch
        # metric asked for a training column would report the last batch rather than the epoch.
        #
        # Set on the FIT path only: `Learner.validate()` runs no training pass, so asking for train
        # columns there would declare names no row ever fills.
        if split == "val":
            learner.recorder.train_metrics = True
        self._learner = learner
        return learner

    def _callbacks(self) -> List[Any]:
        """The wired fastai callbacks, with any file destination's parent directory created.

        fastai's ``CSVLogger`` opens ``learn.path/fname`` without creating it, so a perfectly
        reasonable ``./fastai_logs/run.csv`` would die *after* the data walk and the model build.
        """
        callbacks = [flow(cb) for cb in (self.callbacks or [])]
        base = Path(self._artifact_dir() or ".")
        for callback in callbacks:
            fname = getattr(callback, "fname", None)
            if isinstance(fname, (str, Path)):
                (base / fname).expanduser().parent.mkdir(parents=True, exist_ok=True)
        return callbacks

    def _artifact_dir(self) -> Optional[str]:
        """Where run artifacts land. fastai has no checkpoint-callback convention to read, so
        this follows the sidecar's directory when one is configured."""
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent)
        return None

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"distilaide fastai training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("FastaiRestorer.fit() requires a 'train_set'.")
        # Provenance through matrainer's plain FUNCTIONS — `runnable=self` so the hyperparameters
        # and the resolved config come from matrainer's ONE derivation rather than a dict this
        # backend transcribes for itself.
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self)
        learner = self._build_learner(self._dataloaders())
        if self.one_cycle:
            learner.fit_one_cycle(self.max_epochs, lr_max=self.learning_rate)
        else:
            learner.fit(self.max_epochs, lr=self.learning_rate)
        names = as_metric_names(self.metric_names)
        self._computed_metrics = _recorded_metrics(learner, names)
        _log_recorded_epochs(learner, trackers, names)
        logger.info(f"Training complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        # Uploads the log file, stamps the manifest's terminal status AND finalizes every tracker.
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("FastaiRestorer.test() requires a 'test_set'.")
        loader = flow(self.test_loader, dataset=self._prepare(self.test_set))
        learner = self._build_learner(DataLoaders(loader, loader), "test")
        values = learner.validate()
        # `validate()` fills the VALIDATION columns whichever split it walked, so the row is read
        # as fastai spells it and lands under `val/` — the same names a fit's last epoch reports.
        self._computed_metrics = _named_values(learner, values, as_metric_names(self.metric_names))
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        """Stream per-record restored images to the sink.

        A plain ``torch.no_grad()`` forward rather than ``Learner.get_preds``: ``get_preds``
        returns only the stacked predictions, and the sink contract needs each row paired with the
        metadata from the record it came from.

        Same ``test_loader`` slot as :meth:`test`, flowed with ``collate_fn=collate_records`` — a
        runtime kwarg overrides the stored one, so this swaps the batch shape back to the record
        dict without a second loader. The default collate is ``_to_tuple``, which has already
        dropped the metadata the sink needs; every other knob the operator set still applies.
        """
        if self.test_set is None:
            raise ValueError("FastaiRestorer.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        model = self.model.eval()
        device = next(model.parameters()).device
        records = flow(self.test_loader, dataset=self._prepare(self.test_set), collate_fn=collate_records)
        with torch.no_grad():
            for batch in records:
                preds = model(batch_tensor(batch, self.input_key, device))
                if sink is not None:
                    self._write_predictions(preds, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, preds: torch.Tensor, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.RestorationOutput` per record, paired with its metadata."""
        out = restoration_output(preds)
        per_record_meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
        for i in range(len(out["image"])):
            meta = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            sink.write({"image": out["image"][i]}, meta)

    # ---- tracking ------------------------------------------------------------ #
    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot.

        Written back rather than returned fresh each call: ``flow()`` on a LIVE object returns
        that same object (idempotent), but a deferred ``Fluid`` would be REBUILT on every call —
        two MLflow runs for one training run.
        """
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def trained_model(self) -> torch.nn.Module:
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


class PeakSignalNoiseRatio(Metric):
    """PSNR over the whole split — the one metric fastai has no name for.

    ``fastai.metrics`` publishes ``mse`` / ``mae`` / ``rmse`` / ``msle`` and nothing scale-aware, so
    the quantity every OTHER backend here reports as ``psnr`` was simply unreachable by NAME. In a
    cross-backend sweep that is the single blank cell in the headline column, which is what made it
    worth closing.

    **Why a CLASS when the module docstring says metrics are names.** That rule is about not
    wrapping an upstream metric to make it registrable, and this wraps nothing: it is the
    ACCUMULATOR fastai does not ship for this quantity. The arithmetic is
    :mod:`matrainer.metric_functions`', shared verbatim with the JAX, MLX and HuggingFace loops, so
    the column means one thing in every row.

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
        # sweep; a CPU-tensor unit test passes happily). The JAX and MLX loops that share this
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


#: Metric names distilaide CONFIGURES rather than resolves from ``fastai.metrics`` — each a factory
#: taking the restorer's ``data_range``. The trigger is the one the per-pixel project's
#: ``pixel_accuracy`` alias documents: the quantity has no upstream name here, so a bare string
#: would raise rather than report it.
_METRIC_ALIASES: Dict[str, Callable[[float], Any]] = {
    "psnr": lambda data_range: PeakSignalNoiseRatio(data_range=data_range),
}


def _as_metric(value: FastaiMetric, data_range: float = 1.0) -> Any:
    """A fastai metric from a callable, a CONFIGURED ALIAS, or a NAME in ``fastai.metrics``.

    The string form is not sugar: fastai's regression metrics are plain FUNCTIONS, so none of them
    is uniform enough for a ``!class:`` tag to name or for confluid's registry to tag. A name is
    the only thing a YAML config can write, so this resolves it and says which names exist when it
    cannot.

    :data:`_METRIC_ALIASES` is tried FIRST, since an alias exists precisely where the upstream
    lookup would fail or return the wrong quantity.
    """
    built = flow(value)
    if not isinstance(built, str):
        return built
    if built in _METRIC_ALIASES:
        return _METRIC_ALIASES[built](data_range)
    metric = getattr(fastai_metrics, built, None)
    if metric is None:
        raise ValueError(
            f"Unknown fastai metric {built!r}. Names come from `fastai.metrics` — for an "
            f"image-to-image target: mse, mae, rmse, msle — plus {sorted(_METRIC_ALIASES)}, which "
            f"this backend supplies because fastai ships no scale-aware image metric. Note there "
            f"is still NO ssim here; use the lightning or torch backend for that."
        )
    return metric


def _as_opt_func(value: FastaiOptimizer) -> Any:
    """A fastai optimizer function from a callable or its name (``"Adam"``, ``"SGD"``, …)."""
    built = flow(value)
    if not isinstance(built, str):
        return built
    opt = getattr(fastai_optimizer, built, None)
    if opt is None:
        raise ValueError(
            f"Unknown fastai optimizer {built!r}. Names come from `fastai.optimizer` — e.g. Adam, "
            f"SGD, RMSProp, RAdam, QHAdam, Lamb, ranger."
        )
    return opt


def _metric_names(learner: Learner) -> List[str]:
    """The Recorder's column names, minus the bookkeeping ones."""
    names = list(getattr(learner.recorder, "metric_names", []) or [])
    return [n for n in names if n not in ("epoch", "time")]


def _named_values(learner: Learner, values: Sequence[Any], names: MetricNames) -> Dict[str, float]:
    """Pair a Recorder row with its column names, tolerating a shorter row.

    ``Learner.validate`` returns one fewer value than a training epoch does (there is no
    ``train_loss``), so the columns are matched from the END — where ``valid_loss`` and the
    metrics line up — rather than from the front.

    The row is then translated out of fastai's vocabulary (``names``), which is what makes this
    run's record comparable with another engine's. Two things only a fastai reader knows and the
    translation encodes: ``valid_loss`` is the validation loss, and an UNPREFIXED column is a
    VALIDATION metric — the Recorder computes its metrics on the validation pass and prefixes
    none of them, where every other engine's bare key is a training one.
    """
    columns = _metric_names(learner)
    if len(columns) > len(values):
        columns = columns[len(columns) - len(values) :]
    return names({name: float(value) for name, value in zip(columns, values)}, source="fastai")


def _recorded_metrics(learner: Learner, names: MetricNames) -> Dict[str, float]:
    """The last epoch's metrics as ``{name: value}``, or ``{}`` if nothing was recorded."""
    values = list(getattr(learner.recorder, "values", []) or [])
    return _named_values(learner, values[-1], names) if values else {}


def _log_recorded_epochs(learner: Learner, trackers: Sequence[Tracker], names: MetricNames) -> None:
    """Forward EVERY recorded epoch to the trackers, not just the final one.

    ``Learner.fit`` reports per epoch through its ``Recorder`` and nothing else sees those rows,
    so without this a tracked fastai run would show one point per RUN where every other backend
    shows a curve.
    """
    if not trackers:
        return
    for epoch, row in enumerate(list(getattr(learner.recorder, "values", []) or [])):
        metrics = _named_values(learner, row, names)
        for tracker in trackers:
            tracker.log_metrics(metrics, step=epoch)
