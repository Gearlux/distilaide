"""``HuggingFaceRestorer`` — the same restoration run driven by a ``transformers.Trainer``.

Since P6 (2026-08-08) this is ``(RestorationBase, matrainer.huggingface.
HuggingFaceRunnable)``: the machinery — the deferred ``TrainingArguments`` / ``Trainer``
slots, the run-time argument FLOOR, the entry points with the plain-function tracking hookup,
the ``log_history`` replay, the record-loader predict walk — lives in
:class:`matrainer.huggingface.trainer.HuggingFaceRunnable`; the split preparation and the
trainable-model gate in :class:`~distilaide.restoration_base.RestorationBase`.

**This is the ONE HuggingFace backend in the workspace that HAS a loss slot, and the reason is
measured rather than stylistic.** Its siblings (classification, segmentation) register nothing,
because an ``AutoModelFor…`` handed ``labels=`` computes its own loss and there is nothing a config
could wire. transformers' only image-to-image model does the opposite:

    Swin2SRForImageSuperResolution(pixel_values=x, labels=y)
    -> NotImplementedError: Training is not supported at the moment

So the loss must come from outside, which means (a) this backend supplies a ``Trainer`` subclass
overriding ``compute_loss`` through the machinery's :meth:`_trainer_factory` seam, and (b)
``FRAMEWORKS["huggingface"].registrations`` points at the TORCH loss module — a ``torch.nn`` loss
does not care which loop calls it, and this loop is torch.

**The other consequence worth stating up front:** every published Swin2SR checkpoint UPSCALES (x2 /
x4), so a same-size denoising run trains from scratch. This backend therefore contributes the
training PROGRAM rather than somebody's weights — see
:mod:`distilaide.huggingface_model` for the measurement and for how to fine-tune the real
checkpoint on a super-resolution degradation instead.
"""

from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

import numpy as np
import torch.nn as nn
from confluid import Mandatory, Partial, configurable, flow
from loggair import get_logger
from matrainer.huggingface.trainer import HuggingFaceRunnable
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
from recordstream import RecordSource, batch_tensor

from distilaide.restoration_base import RestorationBase

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
class HuggingFaceRestorer(RestorationBase, HuggingFaceRunnable):
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
        model: Optional[Mandatory[Partial[nn.Module]]] = None,
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        data_range: float = 1.0,
        task: RestorerTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        num_workers: int = 0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        loss_fn: Optional[Partial[nn.Module]] = None,
        metrics: Optional[List[str]] = None,
        val_metrics: Optional[List[str]] = None,
        test_metrics: Optional[List[str]] = None,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        experiment_name: str = "distilaide_restoration",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
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
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            trackers=trackers,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metrics=metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
        )
        self.data_range = float(data_range)
        self.loss_fn = loss_fn

    # ---- the task hooks ---------------------------------------------------------------------- #
    def _default_loss(self) -> Optional[nn.Module]:
        """L1, the same default as every other restoration backend (the modern-recipe choice)."""
        return nn.L1Loss()

    def _default_hf_metrics(self) -> Optional[List[Any]]:
        return list(DEFAULT_METRICS)

    def _metric_names_for(self, split: str) -> List[str]:
        merged = merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics", None)) or []
        for name in merged:
            if name not in _METRIC_REDUCTIONS:
                raise ValueError(
                    f"Unknown metric {name!r} for the huggingface backend. Available: " f"{sorted(_METRIC_REDUCTIONS)}."
                )
        return list(merged)

    def _metrics_hook(self, split: str) -> Any:
        """A ``compute_metrics`` callable for ONE split, over the shared squared-error accumulator."""
        names = self._metric_names_for(split)
        data_range = self.data_range

        def compute(eval_prediction: Any) -> Dict[str, float]:
            predictions = eval_prediction.predictions
            predictions = predictions[0] if isinstance(predictions, tuple) else predictions
            totals = zero_squared_error()
            accumulate_squared_error(totals, np.asarray(predictions), np.asarray(eval_prediction.label_ids))
            return {name: _METRIC_REDUCTIONS[name](totals, data_range) for name in names}

        return compute

    def _trainer_factory(self) -> Any:
        """The ``Trainer`` SUBCLASS that computes the loss ITSELF — this backend's one seam.

        Required, not stylistic: ``Swin2SRForImageSuperResolution`` raises
        ``NotImplementedError: Training is not supported at the moment`` the moment it is
        handed ``labels=``, so the target is POPPED before the model is called and the wired
        ``loss_fn`` applied to the prediction.
        """
        from transformers import Trainer as HFTrainer

        loss_fn = cast(nn.Module, flow(self.loss_fn))

        class _RestorationTrainer(HFTrainer):
            """A ``Trainer`` that computes the loss itself (see the enclosing docstring)."""

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

        return _RestorationTrainer

    # ---- predictions ------------------------------------------------------------------------- #
    def _predict_batch(self, network: Any, batch: Any, device: Any) -> Any:
        """The WRAPPER's own forward — positional images in, restored images out.

        Not the transformers keyword contract: the raw Swin2SR's forward returns a
        ``.reconstruction`` output object, while the wrapper's forward returns the restored
        tensor directly, which is the array the prediction contract wants.
        """
        return cast(nn.Module, self.model)(batch_tensor(batch, self.input_key, device))

    def _prediction_output(self, preds: Any) -> Any:
        """One batched :class:`~recordstream.RestorationOutput` — the machinery slices it.

        ONE key (``image``), and that is the contract working, not a stub: an image-to-image
        model emits the answer, in the INPUT's value range.
        """
        return {"image": preds}
