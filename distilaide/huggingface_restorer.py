"""``HuggingFaceRestorer`` — the same restoration run driven by a ``transformers.Trainer``.

The backend that writes least while swapping most: ``Trainer`` owns the epoch, the optimizer, the
scheduler, gradient accumulation, mixed precision and distribution (via Accelerate). What distilaide
supplies is the data, the loss, the run identity and the artifacts.

**This is the ONE HuggingFace backend in the workspace that HAS a loss slot, and the reason is
measured rather than stylistic.** Its siblings (classification, segmentation) register nothing,
because an ``AutoModelFor…`` handed ``labels=`` computes its own loss and there is nothing a config
could wire. transformers' only image-to-image model does the opposite:

    Swin2SRForImageSuperResolution(pixel_values=x, labels=y)
    -> NotImplementedError: Training is not supported at the moment

So the loss must come from outside, which means (a) this backend needs a ``Trainer`` subclass
overriding ``compute_loss``, and (b) ``FRAMEWORKS["huggingface"].registrations`` points at the TORCH
loss module — a ``torch.nn`` loss does not care which loop calls it, and this loop is torch.

**The other consequence worth stating up front:** every published Swin2SR checkpoint UPSCALES (x2 /
x4), so a same-size denoising run trains from scratch. This backend therefore contributes the
training PROGRAM rather than somebody's weights — see
:mod:`distilaide.huggingface_model` for the measurement and for how to fine-tune the real
checkpoint on a super-resolution degradation instead.
"""

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import numpy as np
import torch
import torch.nn as nn
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
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
    batch_tensor,
    collate_records,
    ensure_materialized,
    ensure_record_dataset,
)
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint, run_entrypoint
from torch.utils.data import DataLoader

if TYPE_CHECKING:  # transformers is imported in the METHOD bodies — see the slots note below
    from transformers import Trainer, TrainingArguments

logger = get_logger(__name__)

#: Which entry point :meth:`HuggingFaceRestorer.run` dispatches to.
RestorerTask = Literal["fit", "evaluate", "test", "predict"]

#: The metrics this backend computes, by NAME — derived from matrainer's shared accumulator, the
#: same vocabulary the jax and mlx backends use. transformers ships no image metrics either.
_METRIC_REDUCTIONS = {
    "psnr": lambda totals, data_range: psnr(totals, data_range),
    "mse": lambda totals, _: mean_squared_error(totals),
    "mae": lambda totals, _: mean_absolute_error(totals),
}

#: What ``metrics`` defaults to.
DEFAULT_METRICS = ("psnr", "mse")

__all__ = ["DEFAULT_METRICS", "HuggingFaceRestorer", "RestorerTask"]


@configurable(task="restoration", role="trainer", framework="torch")
class HuggingFaceRestorer(TorchRunner, ProgressReporting):
    """Train, evaluate or predict an image restorer with a ``transformers.Trainer``.

    Args:
        model: A model mapping ``[N, C, H, W]`` degraded images to restored ones. Wire
            ``!lazy:distilaide.HuggingFaceRestorationModel``.
        train_set / test_set: sources.
        data_range: The span of the target images' values, for PSNR.
        task: Which entry point ``run()`` dispatches to.
        max_epochs / batch_size / num_workers / learning_rate / weight_decay: translated into
            ``TrainingArguments``.
        loss_fn: The pixel loss. Unlike this backend's siblings it is REAL here, because Swin2SR
            refuses to compute one — ``None`` picks ``torch.nn.L1Loss``, matching every other
            backend.
        metrics / val_metrics / test_metrics: metric NAMES. There is deliberately no
            ``train_metrics``: ``compute_metrics`` is a ``Trainer``'s EVALUATION hook, so the slot
            could only be honoured with a second forward pass over the whole train split at every
            epoch end — a cost every user would carry, where offering it and quietly not calling it
            is the exact failure per-split slots exist to remove.
        trackers / experiment_name / run_name: tracking identity and sinks.
        input_key / target_key: Record keys for the degraded image and the clean target.
        metric_names: How recorded metrics are named — a ``Trainer`` writes ``eval_*``.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``training_arguments``, ``trainer``,
    ``test_loader``, ``predictions_sink``, ``metrics_sidecar_path``.
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
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        loss_fn: Optional[Lazy[nn.Module]] = None,
        metrics: Optional[List[str]] = None,
        val_metrics: Optional[List[str]] = None,
        test_metrics: Optional[List[str]] = None,
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
        self.num_workers = num_workers
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.loss_fn: Optional[Lazy[nn.Module]] = loss_fn
        self.metrics: Optional[List[str]] = metrics
        self.val_metrics: Optional[List[str]] = val_metrics
        self.test_metrics: Optional[List[str]] = test_metrics
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        # All THREE transformers objects are DEFERRED SLOTS, and the targets are STRING dotted
        # paths on purpose: this module has NO top-level `import transformers` (the methods that
        # need it import it in their bodies), and confluid resolves a dotted target at flow time —
        # so the slots are declarable without dragging the library into `import distilaide`.
        self.training_arguments: Lazy[Any] = LazyClass("transformers.TrainingArguments")
        self.trainer: Lazy[Any] = LazyClass("transformers.Trainer")
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(
            DataLoader, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_records, shuffle=False
        )
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self._computed_metrics: Dict[str, float] = {}

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]
        return self.task == "fit"

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
        """Fill the unset slots. The loss is L1, the same as every other backend."""
        if self.loss_fn is None:
            self.loss_fn = nn.L1Loss()
        if self.metrics is None:
            self.metrics = list(DEFAULT_METRICS)

    # ---- the batch boundary -------------------------------------------------- #
    def _to_hf_batch(self, records: Any) -> Dict[str, Any]:
        """One list of records → the ``{"pixel_values", "labels"}`` a ``Trainer`` collates to.

        The twin of the other backends' ``_targets`` / ``_to_arrays`` / ``_to_tuple``, handed to
        ``Trainer`` as its ``data_collator`` — the same seat a ``collate_fn`` occupies elsewhere.
        The KEYS are transformers' contract, not this project's; ``labels`` never reaches the model
        (see :meth:`_RestorationTrainer.compute_loss`), it is popped and used as the target.
        """
        batch = collate_records(list(records))
        return {
            "pixel_values": batch_tensor(batch, self.input_key),
            "labels": batch_tensor(batch, self.target_key),
        }

    def _metric_names_for(self, split: str) -> List[str]:
        merged = merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics", None)) or []
        for name in merged:
            if name not in _METRIC_REDUCTIONS:
                raise ValueError(
                    f"Unknown metric {name!r} for the huggingface backend. Available: " f"{sorted(_METRIC_REDUCTIONS)}."
                )
        return list(merged)

    def _metrics_hook(self, split: str) -> Any:
        """A ``compute_metrics`` callable for ONE split.

        The split is decided where the hook is BUILT, because the hook itself cannot tell which
        dataset it was called on — the same reason its segmentation sibling does it this way.
        """
        names = self._metric_names_for(split)
        data_range = self.data_range

        def compute(eval_prediction: Any) -> Dict[str, float]:
            predictions = eval_prediction.predictions
            predictions = predictions[0] if isinstance(predictions, tuple) else predictions
            totals = zero_squared_error()
            accumulate_squared_error(totals, np.asarray(predictions), np.asarray(eval_prediction.label_ids))
            return {name: _METRIC_REDUCTIONS[name](totals, data_range) for name in names}

        return compute

    def _training_arguments(self, output_dir: str) -> "TrainingArguments":
        """The derived FLOOR, injected for every key the config did NOT pin.

        Pinning these on the constructor's ``LazyClass`` instead is what a YAML override would
        DESTROY: the slot is a whole-value replacement, so ``training_arguments: {use_cpu: true}``
        would silently drop the epoch count, the batch sizes and ``remove_unused_columns=False`` —
        the last of which surfaces as a ``KeyError`` three frames inside ``Trainer``'s own collator,
        saying nothing about the override. A key the config pinned WINS.
        """
        floor: Dict[str, Any] = {
            "output_dir": output_dir,
            "num_train_epochs": self.max_epochs,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dataloader_num_workers": self.num_workers,
            # `Trainer` auto-enables every tracker it can import, and this workspace has wandb AND
            # mlflow installed — so the default would silently start reporting to both. Tracking
            # goes through the `trackers` slot like every other backend.
            "report_to": "none",
            "run_name": self.run_name,
            "disable_tqdm": True,
            "save_strategy": "no",
            # A generic map-style source is not a `datasets.Dataset`, so `Trainer` must not drop
            # "unused" columns by inspecting the model signature — the collate above already
            # produces exactly the two keys it needs.
            "remove_unused_columns": False,
            "eval_strategy": "epoch" if self.val_set is not None else "no",
        }
        pinned = self.training_arguments.kwargs if isinstance(self.training_arguments, Fluid) else {}
        return cast(
            "TrainingArguments", flow(self.training_arguments, **{k: v for k, v in floor.items() if k not in pinned})
        )

    def _build_trainer(self, split: str) -> "Trainer":
        """Assemble the ``Trainer`` — with the subclass that supplies the loss Swin2SR will not."""
        from transformers import Trainer as HFTrainer

        loss_fn = cast(nn.Module, flow(self.loss_fn))

        class _RestorationTrainer(HFTrainer):  # type: ignore[misc,valid-type]
            """A ``Trainer`` that computes the loss ITSELF.

            Required, not stylistic: ``Swin2SRForImageSuperResolution`` raises
            ``NotImplementedError: Training is not supported at the moment`` the moment it is
            handed ``labels=``, so the target is POPPED before the model is called.
            """

            def compute_loss(
                self,
                model: Any,
                inputs: Dict[str, Any],
                return_outputs: bool = False,
                # Named rather than swept into `**kwargs`. Both of `Trainer`'s own call sites pass
                # it BY KEYWORD (`trainer.py:1919` and `:2987`), so `**kwargs` works at runtime —
                # but the base declares it positional-or-keyword, which `**kwargs` cannot accept
                # positionally, so the override is formally incompatible and mypy says so. Naming
                # it costs nothing and keeps this callable the way the base promises.
                num_items_in_batch: Any = None,
                **kwargs: Any,
            ) -> Any:
                del num_items_in_batch  # the loss here is already a mean over the batch
                targets = inputs.pop("labels")
                predictions = model(inputs["pixel_values"])
                loss = loss_fn(predictions, targets)
                return (loss, {"reconstruction": predictions}) if return_outputs else loss

        output_dir = str(self.metrics_sidecar_path or ".") if self.metrics_sidecar_path else "./hf_restoration"
        trainer = flow(
            self.trainer,
            model=self.model,
            args=self._training_arguments(output_dir),
            train_dataset=self.train_set,
            eval_dataset=self.val_set if split == "val" else self.test_set,
            data_collator=self._to_hf_batch,
            compute_metrics=self._metrics_hook(split),
        )
        # The `trainer` slot is a `LazyClass("transformers.Trainer")` so a config can replace the
        # whole Trainer; when it has NOT, the subclass above is what must be built.
        if type(trainer) is HFTrainer:
            trainer = _RestorationTrainer(
                model=self.model,
                args=self._training_arguments(output_dir),
                train_dataset=self.train_set,
                eval_dataset=self.val_set if split == "val" else self.test_set,
                data_collator=self._to_hf_batch,
                compute_metrics=self._metrics_hook(split),
            )
        return cast("Trainer", trainer)

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"distilaide huggingface training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("HuggingFaceRestorer.fit() requires a 'train_set'.")
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self) if trackers else None
        trainer = self._build_trainer("val")
        result = trainer.train()
        names = as_metric_names(self.metric_names)
        metrics = dict(result.metrics)
        if self.val_set is not None:
            metrics.update(trainer.evaluate())
        self._computed_metrics = names({k: float(v) for k, v in metrics.items() if _is_number(v)}, source="huggingface")
        _log_trainer_history(trainer, trackers, names)
        logger.info(f"Training complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("HuggingFaceRestorer.test() requires a 'test_set'.")
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self) if trackers else None
        trainer = self._build_trainer("test")
        # `metric_key_prefix="test"`: HF's `eval` default would file held-out scores in the `val/`
        # columns, where every other backend puts its validation ones.
        results = trainer.evaluate(metric_key_prefix="test")
        names = as_metric_names(self.metric_names)
        self._computed_metrics = names({k: float(v) for k, v in results.items() if _is_number(v)}, source="huggingface")
        for tracker in trackers:
            tracker.log_metrics(self._computed_metrics)
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        # Left unfinished a manifest stays `running`, and matrainer's atexit hook then labels a
        # successful evaluation `interrupted`.
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("predict", role="predictor", primary=True)
    @torch.no_grad()
    def predict(self) -> None:
        if self.test_set is None:
            raise ValueError("HuggingFaceRestorer.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        model = self.model.eval()
        device = next(model.parameters()).device
        loader = flow(self.test_loader, dataset=self.test_set)
        for batch in loader:
            restored = model(batch_tensor(batch, self.input_key, device))
            if sink is not None:
                self._write_predictions(restored, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    def _write_predictions(self, restored: torch.Tensor, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.RestorationOutput` per record, paired with its metadata."""
        per_record_meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
        for i in range(len(restored)):
            out: RestorationOutput[torch.Tensor] = {"image": restored[i]}
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


def _is_number(value: Any) -> bool:
    """Whether a ``Trainer`` metrics entry is a scalar (it also reports strings and counts)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _log_trainer_history(trainer: Any, trackers: Sequence[Tracker], names: MetricNames) -> None:
    """Forward EVERY logged step to the trackers, not just the final metrics.

    A ``Trainer`` accumulates its own ``log_history`` and hands it to whatever it was told to
    ``report_to``, which here is deliberately ``"none"`` — tracking goes through the `trackers`
    slot like every other backend. So without this the run's numbers reach the LOG and the metrics
    sidecar and nothing else: a tracked huggingface run showed no metrics at all, and in a
    cross-backend sweep it contributed a completely BLANK row while its own log printed
    ``val/psnr: 26.98`` (hit for real; this is the one backend that had no bridge).

    Every step, not just the final values, so a tracked run shows a CURVE where the others show one
    — the entries are the same per-epoch rows the console prints.
    """
    if not trackers:
        return
    for entry in getattr(trainer.state, "log_history", []) or []:
        values = names(
            {k: float(v) for k, v in entry.items() if _is_number(v) and k != "epoch"},
            source="huggingface",
        )
        if not values:
            continue
        step = int(entry.get("step", 0))
        for tracker in trackers:
            tracker.log_metrics(values, step=step)
