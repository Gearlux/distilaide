"""``RestorationBase`` — the task half of every distilaide backend, written once.

The image-to-image logic that used to exist per backend module: the fork-safe split
preparation, the trainable-model refusal for ``task="fit"`` over a classical algorithm, and
the single-body initialization (restoration resolves no labels and no class count — the one
run dimension is ``data_range``, injected into the metric templates by the torch hooks).

It subclasses the trainer API base (`matrainer.runnable.TrainerRunnable`) and is deliberately
**torch-free**: the keras/tensorflow/jax/mlx backends import this module too, and the
core-is-backend-free mandate forbids it dragging a framework in (the trainable check below is
duck-typed over ``model.parameters()`` for exactly that reason). The torch-family hook
IMPLEMENTATIONS live in :mod:`distilaide.restoration_torch`; a concrete backend composes
``(RestorationTorchHooks, <Framework>Runnable)``.
"""

from loggair import get_logger
from matrainer.runnable import TrainerRunnable

logger = get_logger(__name__)

__all__ = ["RestorationBase"]


class RestorationBase(TrainerRunnable):
    """The restoration task base: split preparation + the trainable-model gate.

    The attribute below is STORED by the concrete backend's own ``__init__`` (it is its
    config surface); this base only declares the name/type for the methods that read it.
    """

    log_label = "distilaide"

    #: The span of the target images' values — this task's `num_classes`. Injected into every
    #: metric template by the torch hooks; getting it wrong is SILENT (PSNR is defined
    #: relative to it — measured: 20.46 dB at 1.0 vs 68.59 dB at 255.0 on the same tensors).
    data_range: float

    def initialize(self) -> None:
        """Build the model, then the loss + the metrics that depend on ``data_range``.

        Single-body (no fit/eval split): restoration resolves no labels and no class count,
        and every split's metrics are built whatever the task — an eval run scores the same
        collections a fit run would.
        """
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)

        from confluid import flow

        self.model = flow(self.model)
        if self.task == "fit":
            self._require_a_trainable_model()
        self._resolve_task_defaults()
        self._build_fit_metrics()

    def _require_a_trainable_model(self) -> None:
        """Refuse ``task="fit"`` over a parameter-free model, by name and with the way out.

        The restoration zoo contains fixed ALGORITHMS beside learned networks — BM3D, total
        variation, wavelet shrinkage, a median filter — and they wear the same ``nn.Module``
        interface while having exactly zero parameters. Handing one to a ``fit`` is not a
        subtle mistake: torch reports *"optimizer got an empty parameter list"* from inside
        ``configure_optimizers``, several frames from the ``arch:`` line that caused it, and
        a reader has no reason to connect the two.

        Checked here rather than in the model wrapper because it is the TASK that makes it
        wrong: the very same object is exactly right for an evaluation run, which is what a
        classical baseline exists for. Duck-typed over ``parameters()`` so this module stays
        torch-free.
        """
        if any(p.requires_grad for p in self.model.parameters()):
            return
        arch = getattr(self.model, "arch", type(self.model).__name__)
        raise ValueError(
            f"task='fit' needs a model with trainable parameters, and {arch!r} has none — it is a "
            "fixed algorithm, not a learned one. Score it with `distilaide evaluate` (that is what "
            "a classical baseline is for), or pick a learned architecture such as arch: dncnn."
        )
