"""The three torch-family backends, and what must be TRUE OF ALL THREE.

The project's one claim is that the config is the program: swap the runnable, keep everything else.
That claim is only worth making if the backends agree about the things a config sets — the loss,
the metric names, the batch keys, what a fit refuses — so most of this file is parametrized over
all three rather than written once per backend. What is genuinely per-backend lives at the bottom.
"""

import ast
import inspect
from pathlib import Path
from typing import Any, Dict, List, Type, cast

import pytest
import torch
import torch.nn as nn
from recordstream import collate_records

from distilaide.deepinv_model import DeepInvModel
from distilaide.fastai_restorer import FastaiRestorer
from distilaide.lightning_restorer import LightningRestorer
from distilaide.torch_restorer import TorchRestorer

SOURCE_DIR = Path(__file__).resolve().parent.parent / "distilaide"

#: The TORCH-family backends — the ones that share a model, a loss set and a batch shape, so a
#: behavioural check can be parametrized over the live CLASSES. The other five need a framework a
#: default install may lack, so the fleet-wide rules below are SOURCE scans instead (the
#: `test_tracking_parity` reason: an import-based sweep silently skips exactly the machine where a
#: copy-pasted omission lands).
BACKENDS: List[Type[Any]] = [LightningRestorer, TorchRestorer, FastaiRestorer]

#: Every backend module on disk, for the source-level fleet checks.
ALL_BACKEND_SOURCES = sorted(SOURCE_DIR.glob("*_restorer.py"))


def _tiny_model() -> DeepInvModel:
    """A learned architecture small enough for a unit test to train."""
    return DeepInvModel(arch="dncnn", depth=2, nf=4)


# --------------------------------------------------------------------------- #
# What every backend agrees about
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", BACKENDS)
def test_the_batch_keys_are_the_same_everywhere(cls: Type[Any]) -> None:
    """A config names ``input_key`` / ``target_key`` once; three backends must read the same two.

    ``target`` rather than ``mask`` or ``class``: the target of an image-to-image run is another
    image, and the key says so.
    """
    params = inspect.signature(cls.__init__).parameters
    assert params["input_key"].default == "image"
    assert params["target_key"].default == "target"


@pytest.mark.parametrize("cls", BACKENDS)
def test_the_default_loss_is_l1_on_every_backend(cls: Type[Any]) -> None:
    """A config moved between backends must not silently change its loss.

    L1 rather than MSE across the board — DnCNN's paper used MSE, but modern restoration nets
    report L1 doing better, and the point here is that all three agree on whichever it is.
    """
    restorer = cls(model=_tiny_model())
    restorer._resolve_task_defaults()
    loss = restorer.loss_fn if hasattr(restorer, "loss_fn") else restorer.loss
    assert isinstance(loss, nn.L1Loss)


@pytest.mark.parametrize("cls", BACKENDS)
def test_an_explicitly_wired_loss_is_never_replaced(cls: Type[Any]) -> None:
    """That is what the ``None`` SENTINEL buys: "not chosen" stays distinguishable from
    "chosen to be the default"."""
    chosen = nn.MSELoss()
    kwargs = {"loss_fn": chosen} if cls is not FastaiRestorer else {"loss": chosen}
    restorer = cls(model=_tiny_model())
    for key, value in kwargs.items():
        setattr(restorer, key, value)
    restorer._resolve_task_defaults()
    assert (restorer.loss_fn if hasattr(restorer, "loss_fn") else restorer.loss) is chosen


@pytest.mark.parametrize("cls", BACKENDS)
def test_every_backend_refuses_to_fit_a_parameter_free_model(cls: Type[Any]) -> None:
    """A classical baseline is a complete EVALUATION and an impossible FIT.

    All three must refuse identically: a run accepted by one backend and refused by another would
    make the cross-backend comparison a lie. The message must name the way out, because torch's
    own failure (*"optimizer got an empty parameter list"*) lands several frames from the ``arch:``
    line that caused it.
    """
    restorer = cls(model=DeepInvModel(arch="median", kernel_size=3), task="fit")
    with pytest.raises(ValueError, match="distilaide evaluate"):
        restorer.initialize()


@pytest.mark.parametrize("cls", BACKENDS)
def test_a_classical_model_is_accepted_for_evaluation(cls: Type[Any]) -> None:
    """The converse, and the reason the refusal above is task-scoped rather than model-scoped."""
    restorer = cls(model=DeepInvModel(arch="median", kernel_size=3), task="test")
    restorer.initialize()  # must not raise
    assert restorer.model is not None


@pytest.mark.parametrize("cls", BACKENDS)
def test_every_backend_takes_the_metric_naming_knob(cls: Type[Any]) -> None:
    """One top-level ``metric_names:`` key must reach all three, or a cross-engine table cannot
    be recorded in one vocabulary."""
    params = inspect.signature(cls.__init__).parameters
    assert "metric_names" in params
    assert cls(model=_tiny_model(), metric_names=None).metric_names is None


@pytest.mark.parametrize("cls", BACKENDS)
def test_trackers_is_a_constructor_param(cls: Type[Any]) -> None:
    """Not cosmetic: confluid gates LIST broadcast on the receiving slot's ANNOTATION, so as a
    post-construction attribute a ``trackers: [...]`` key silently does NOTHING while every unit
    test that assigns ``.trackers`` in Python keeps passing."""
    params = inspect.signature(cls.__init__).parameters
    assert "trackers" in params
    annotation = str(params["trackers"].annotation)
    assert "Iterable" in annotation, f"{cls.__name__}: the annotation must admit a list"


@pytest.mark.parametrize("cls", BACKENDS)
def test_the_constructor_stores_trackers_verbatim(cls: Type[Any]) -> None:
    """Lazy-init: normalizing here would ITERATE, and a deferred ``Fluid`` is not iterable."""
    marker = object()
    assert cls(model=_tiny_model(), trackers=marker).trackers is marker  # type: ignore[arg-type]


@pytest.mark.parametrize("cls", BACKENDS)
def test_every_backend_declares_the_same_four_tasks(cls: Type[Any]) -> None:
    """The verb vocabulary is a property of the TASK, not of the engine."""
    from recordstream.runnable import runnable_entrypoints

    markers = runnable_entrypoints(cls)
    assert {marker["task"] for marker in markers.values()} == {"fit", "evaluate", "test", "predict"}


@pytest.mark.parametrize("cls", BACKENDS)
def test_only_fit_needs_autograd(cls: Type[Any]) -> None:
    """A GUI executor reads this to decide whether to run under ``inference_mode``."""
    assert cls(model=_tiny_model(), task="fit").__needs_autograd__
    assert not cls(model=_tiny_model(), task="predict").__needs_autograd__


@pytest.mark.parametrize("cls", BACKENDS)
def test_no_backend_dispatches_through_a_hand_written_task_dict(cls: Type[Any]) -> None:
    """The ``@entrypoint`` markers ARE the dispatch table; a dict restates them and drifts."""
    source = inspect.getsource(cls.run)
    assert "run_entrypoint" in source


# --------------------------------------------------------------------------- #
# What every backend in the FLEET agrees about — checked by SOURCE scan
# --------------------------------------------------------------------------- #
# These read the module text rather than importing, because five of the eight backends need a
# framework a default install may lack — and that machine is exactly where a copy-pasted omission
# lands, so an import-based sweep would skip the case it exists to catch.


@pytest.mark.parametrize("path", ALL_BACKEND_SOURCES, ids=lambda p: p.stem)
def test_every_backend_warms_its_sources_before_forking(path: Path) -> None:
    """``ensure_materialized`` in ``_prepare`` — the guard against a SIGSEGV with no traceback.

    A ``DataLoader`` worker may be a FORKED child; if that child is the first to touch a lazy
    source the download runs there, through the not-fork-safe ``_scproxy``, and the worker dies
    with no Python error at all. fastai makes it worse by setting ``fork`` process-wide at import,
    so every backend sharing that interpreter inherits the hazard.
    """
    source = path.read_text()
    if "RestorationTorchHooks" in source:
        # Built on the shared bases (2026-08-06): the warm lives in `RestorationBase.initialize`
        # (via `_prepare` → recordstream's `prepare_record_dataset`, pinned upstream). What this
        # module must not do is SHADOW that initialize with an unwarmed copy.
        assert "def initialize" not in source, f"{path.name} shadows the shared (warming) initialize"
        return
    assert "ensure_materialized" in source, f"{path.name} does not warm its sources"


@pytest.mark.parametrize("path", ALL_BACKEND_SOURCES, ids=lambda p: p.stem)
def test_no_backend_derives_its_own_hyperparameters(path: Path) -> None:
    """matrainer owns the ONE derivation (``runnable=self``); a local params dict is a second
    answer to "what were the hyperparameters?" for the same config."""
    source = path.read_text()
    assert "_config_params" not in source
    assert "_tracked_params" not in source
    if "LightningRunnable" in source or "TorchLoopRunnable" in source:
        # Built on a shared machinery base (2026-08-06), which passes `runnable=self`
        # (`_build_trainer` / `_log_run_context`) — pinned in matrainer's own suite; what the
        # module must not do is shadow that wiring.
        assert "def _build_trainer" not in source and "def _log_run_context" not in source
        return
    assert "runnable=self" in source, f"{path.name} does not hand matrainer the runnable"


@pytest.mark.parametrize("path", ALL_BACKEND_SOURCES, ids=lambda p: p.stem)
def test_every_backend_takes_the_metric_naming_knob_fleet_wide(path: Path) -> None:
    """One top-level ``metric_names:`` key must reach all eight, or a cross-engine table cannot be
    recorded in one vocabulary."""
    assert "metric_names" in path.read_text(), f"{path.name} has no metric_names slot"


#: How each backend gets its metrics onto the wired trackers. FOUR spellings, because the loop
#: differs — but every backend must have ONE, or a tracked run records nothing and a cross-backend
#: sweep shows a blank row for it.
#:
#: This is a fleet check because the failure is invisible per backend: the huggingface row printed
#: `val/psnr: 26.98` in its own log and wrote its metrics sidecar while sending NOTHING to the
#: trackers, and it took a sweep — eleven rows in one table, one of them empty — to see it.
_TRACKER_METRIC_BRIDGES = {
    # The engine's own logger interface, adapted: Lightning's `log_dict` reaches a `TrackerLogger`.
    "lightning_restorer": "TrackerLogger",
    "keras_restorer": "keras_tracking_callback",
    "tensorflow_restorer": "log_metrics",
    "fastai_restorer": "log_metrics",
    # Loops we own: an explicit call per epoch.
    "torch_restorer": "log_metrics",
    "jax_restorer": "log_metrics",
    "mlx_restorer": "log_metrics",
    # A `Trainer` reports to whatever it was told to `report_to`, which is deliberately "none" —
    # so its accumulated `log_history` has to be forwarded by hand.
    "huggingface_restorer": "_log_trainer_history",
}


@pytest.mark.parametrize("path", ALL_BACKEND_SOURCES, ids=lambda p: p.stem)
def test_every_backend_puts_its_metrics_on_the_trackers(path: Path) -> None:
    """A backend whose numbers reach only the log is invisible to every comparison.

    Found by the backend sweep: `HuggingFaceRestorer` computed `self._computed_metrics`, logged
    them and wrote the sidecar — and never called `tracker.log_metrics`, so its row in an
    eleven-run table was completely blank while its own log looked perfect.
    """
    source = path.read_text()
    if "LightningRunnable" in source or "TorchLoopRunnable" in source:
        # The bridge lives in the shared machinery base (TrackerLogger adapters / the per-epoch
        # `tracker.log_metrics` call in `fit`) — pinned in matrainer's own suite; the module
        # must not shadow the wiring that carries it.
        assert "def fit" not in source and "def _build_trainer" not in source
        return
    marker = _TRACKER_METRIC_BRIDGES[path.stem]

    assert marker in source, f"{path.name} never reaches a tracker via {marker}"


def test_the_bridge_table_covers_the_whole_fleet() -> None:
    """The converse — a ninth backend must not be added without one."""
    assert set(_TRACKER_METRIC_BRIDGES) == {path.stem for path in ALL_BACKEND_SOURCES}


#: The backends whose LOOP IS OURS, and the accumulate-then-reduce vocabulary each one uses.
#:
#: Only these four can get the reduction wrong. The other four hand the loop to an engine that
#: accumulates for them — Lightning drives a ``MetricCollection``, ``keras.Model.compile`` holds
#: stateful metric objects, a fastai ``Recorder`` accumulates over its own pass, and a
#: ``transformers.Trainer`` calls ``compute_metrics`` ONCE over the gathered split — so there is
#: nothing in their source to check and a grep over all eight would only find markers by accident.
_OWNED_LOOP_ACCUMULATORS = {
    "torch_restorer": "MetricCollection",  # update() per batch, compute() at epoch end
    "tensorflow_restorer": "reset_state",  # stateful keras.metrics objects, read once per split
    "jax_restorer": "accumulate_squared_error",  # matrainer's shared accumulator
    "mlx_restorer": "accumulate_squared_error",
}


@pytest.mark.parametrize("stem,marker", sorted(_OWNED_LOOP_ACCUMULATORS.items()))
def test_a_backend_that_owns_its_loop_accumulates_rather_than_averaging(stem: str, marker: str) -> None:
    """PSNR is a LOGARITHM of a mean, so the mean of per-batch PSNRs is not the split's PSNR.

    On a fixed-degradation run the two agree to three decimals and this looks like pedantry; the
    moment the per-batch error VARIES the average reads several dB HIGH (measured: 20.485 dB
    accumulated against 20.114 batch-averaged on the same images at mixed noise levels — +3.5 dB,
    always optimistic). So a loop we own accumulates and reduces once.
    """
    source = (SOURCE_DIR / f"{stem}.py").read_text()
    if "TorchLoopRunnable" in source or "LightningRunnable" in source:
        # Accumulate-per-batch / reduce-once lives in the shared machinery base
        # (`_epoch_metrics` computes ONCE and resets) — pinned in matrainer's own suite.
        assert "def _epoch_metrics" not in source, f"{stem} shadows the shared reduction"
        return
    assert marker in source, (
        f"{stem}.py no longer shows its accumulate-then-reduce vocabulary ({marker!r}) — "
        "a per-batch PSNR average is not the PSNR of the split."
    )


def test_the_owned_loop_list_is_every_backend_without_an_engine_loop() -> None:
    """The converse: a backend moved out of an engine's loop must join the check above."""
    engine_driven = {"lightning_restorer", "keras_restorer", "fastai_restorer", "huggingface_restorer"}
    on_disk = {path.stem for path in ALL_BACKEND_SOURCES}
    assert set(_OWNED_LOOP_ACCUMULATORS) | engine_driven == on_disk


def test_the_fleet_is_the_eight_backends_the_cli_advertises() -> None:
    """Stops a ninth backend slipping past every source check above."""
    from distilaide.cli import FRAMEWORKS

    on_disk = {path.stem.removesuffix("_restorer") for path in ALL_BACKEND_SOURCES}
    assert on_disk == set(FRAMEWORKS), f"disk {sorted(on_disk)} vs table {sorted(FRAMEWORKS)}"
    assert len(on_disk) == 8, f"expected eight backends, found {sorted(on_disk)}"


def test_the_torch_family_checked_live_is_the_torch_family_on_disk() -> None:
    """The parametrized BEHAVIOURAL checks above cover exactly the three torch backends.

    Named explicitly so adding a fourth torch backend without adding it to ``BACKENDS`` fails
    here rather than silently going unchecked.
    """
    checked = {cls.__module__.rsplit(".", 1)[-1].removesuffix("_restorer") for cls in BACKENDS}
    assert checked == {"lightning", "torch", "fastai"}


# --------------------------------------------------------------------------- #
# The Lightning backend
# --------------------------------------------------------------------------- #
def test_the_metrics_default_to_psnr_and_ssim() -> None:
    restorer = LightningRestorer(model=_tiny_model())
    restorer._resolve_task_defaults()
    assert set(restorer.metrics) == {"psnr", "ssim"}


def test_data_range_is_injected_into_every_metric_template() -> None:
    """The restoration analogue of a segmentation task's ``num_classes``, and REQUIRED rather than
    nice: torchmetrics 1.9 declares ``PeakSignalNoiseRatio(data_range, ...)`` with no default, so a
    ``!lazy:`` template that never received one raises instead of quietly scoring against an
    inferred range."""
    restorer = LightningRestorer(model=_tiny_model(), data_range=255.0)
    restorer._resolve_task_defaults()
    built = restorer._build_split_metrics(None, "val/")
    assert float(cast(Any, built["val/psnr"]).data_range) == 255.0


def test_a_template_that_pins_its_own_data_range_keeps_it() -> None:
    """The injection is a FLOOR, not an override — a metric deliberately scored on another scale
    is a legitimate thing to write."""
    from confluid import LazyClass
    from torchmetrics.image import PeakSignalNoiseRatio

    restorer = LightningRestorer(model=_tiny_model(), data_range=1.0)
    restorer.metrics = {"psnr": LazyClass(PeakSignalNoiseRatio, data_range=2.0)}
    built = restorer._build_split_metrics(None, "val/")
    assert float(cast(Any, built["val/psnr"]).data_range) == 2.0


def test_a_split_slot_adds_to_the_shared_set_for_that_split_only() -> None:
    """``metrics:`` is what every split measures; ``val_metrics:`` is what validation adds.

    ``SpectralAngleMapper`` is chosen deliberately: it takes NO ``data_range``, so this also pins
    that the sizing dimension is injected per template rather than blanket — the blanket form made
    a perfectly valid metric break the whole collection.
    """
    from confluid import LazyClass
    from torchmetrics.image import SpectralAngleMapper

    restorer = LightningRestorer(model=_tiny_model(), val_metrics={"sam": LazyClass(SpectralAngleMapper)})
    restorer._resolve_task_defaults()
    val = restorer._build_split_metrics(restorer.val_metrics, "val/")
    train = restorer._build_split_metrics(None, "train/")
    assert "val/sam" in set(val.keys())
    assert not any(str(key).endswith("sam") for key in train.keys())


def test_the_training_step_computes_a_loss_over_the_pair(pair_records: List[Dict[str, Any]]) -> None:
    restorer = LightningRestorer(model=_tiny_model())
    restorer.model = restorer.model  # the model is live already; no flow needed for a direct step
    loss = restorer.training_step(collate_records(pair_records), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_predict_step_returns_the_contract_not_a_bare_tensor(pair_records: List[Dict[str, Any]]) -> None:
    """A sink reads ``image`` by name; a raw tensor has no ``.get`` and dies on the first batch."""
    restorer = LightningRestorer(model=_tiny_model())
    out = restorer.predict_step(collate_records(pair_records), 0)
    assert set(out.keys()) == {"image"}
    assert out["image"].shape == (4, 3, 32, 32)


def test_a_batched_output_is_sliced_per_record_for_the_sink(pair_records: List[Dict[str, Any]]) -> None:
    """The model emits ``[N, C, H, W]`` while ``PredictionsSink.write`` is per record — handing
    the batch over whole writes ONE entry for an N-record batch."""
    written: List[Any] = []

    class _Sink:
        def write(self, prediction: Any, metadata: Any) -> None:
            written.append((prediction, metadata))

    restorer = LightningRestorer(model=_tiny_model())
    restorer.predictions_sink = _Sink()  # type: ignore[assignment]
    restorer.predict_step(collate_records(pair_records), 0)
    assert len(written) == 4
    assert written[0][0]["image"].shape == (3, 32, 32)
    assert written[0][1]["row_id"] == 0


# --------------------------------------------------------------------------- #
# The bare-torch backend — the control experiment
# --------------------------------------------------------------------------- #
def test_it_needs_no_training_framework() -> None:
    """The whole point of this backend: remove the framework and the config still works.

    An AST scan rather than a ``sys.modules`` check, because the import may come in through
    ``matrainer.lightning``, whose package ``__init__`` runs ``require("pytorch_lightning")``.
    """
    tree = ast.parse((SOURCE_DIR / "torch_restorer.py").read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    offenders = [name for name in imported if "lightning" in name]
    assert not offenders, f"the framework-free backend imports {offenders}"


def test_the_torch_backend_has_the_knobs_a_loop_owner_needs() -> None:
    """``device`` / ``checkpoint_dir`` / ``grad_clip`` — what a ``Trainer`` would have decided."""
    params = inspect.signature(TorchRestorer.__init__).parameters
    assert {"device", "checkpoint_dir", "grad_clip"} <= set(params)
    assert params["device"].default == "auto"


def test_the_torch_backend_exposes_a_history_output() -> None:
    """A loop that owns its epochs has to accumulate what a ``Trainer``'s logger would have."""
    restorer = TorchRestorer(model=_tiny_model())
    assert restorer.history == {}


def test_a_full_torch_loop_reduces_the_loss(pair_records: List[Dict[str, Any]]) -> None:
    """End to end through the real loop: fit over a tiny in-memory pair and check it LEARNED.

    Asserting the loss falls rather than merely that the loop runs — a training loop that
    silently applies no gradients also "runs".
    """
    restorer = TorchRestorer(
        model=_tiny_model(),
        train_set=pair_records,
        max_epochs=3,
        batch_size=2,
        device="cpu",
    )
    # `val_set` is a POST-CONSTRUCTION slot on every backend (a config reaches it by broadcasting),
    # deliberately not a constructor param — so a test wires it the same way YAML does.
    restorer.val_set = pair_records
    restorer.run()
    losses = restorer.history["train/loss"]
    assert len(losses) == 3
    assert losses[-1] < losses[0], f"the loss did not fall: {losses}"


# --------------------------------------------------------------------------- #
# The fastai backend
# --------------------------------------------------------------------------- #
def test_the_fastai_loss_needs_no_axis_fix() -> None:
    """The asymmetry with the segmentation task, measured rather than assumed.

    There, ``CrossEntropyLossFlat`` must be given ``axis=1`` because its default flattens the
    CHANNEL dimension — which carries the CLASSES — into the batch. An image-to-image loss has no
    class dimension, so flattening a ``[N, C, H, W]`` pair identically on both sides changes
    nothing: fastai's wrapper and the plain torch loss agree to full precision.
    """
    from fastai.losses import L1LossFlat

    torch.manual_seed(0)
    preds, target = torch.rand(4, 3, 16, 16), torch.rand(4, 3, 16, 16)
    assert float(L1LossFlat()(preds, target)) == float(nn.L1Loss()(preds, target))


def test_the_fastai_metrics_are_names_resolved_against_its_own_module() -> None:
    """fastai's regression metrics are plain FUNCTIONS — there is no class for a tag to resolve."""
    from distilaide.fastai_restorer import _as_metric

    assert callable(_as_metric("mse"))
    with pytest.raises(ValueError, match="fastai.metrics"):
        _as_metric("definitely_not_a_metric")


def test_psnr_is_the_ONE_name_this_backend_supplies_itself() -> None:
    """It used to raise here, and the backend sweep is what changed that.

    `fastai.metrics` has nothing scale-aware, so `psnr` — the column every other backend fills —
    was unreachable by name and left two of eleven sweep rows blank in the headline metric. It now
    resolves to a CONFIGURED ALIAS holding matrainer's accumulator; the arithmetic and the
    accumulation are pinned in `tests/test_fastai_psnr.py`.
    """
    from distilaide.fastai_restorer import PeakSignalNoiseRatio, _as_metric

    assert isinstance(_as_metric("psnr"), PeakSignalNoiseRatio)


def test_the_unknown_metric_message_says_where_SSIM_lives() -> None:
    """SSIM is still absent and still honest — there is no shared implementation to accumulate."""
    from distilaide.fastai_restorer import _as_metric

    with pytest.raises(ValueError, match="lightning or torch backend"):
        _as_metric("ssim")


def test_the_fastai_batch_is_a_positional_tuple(pair_records: List[Dict[str, Any]]) -> None:
    """fastai unpacks a batch POSITIONALLY, so the batch must BE a tuple, not a record dict."""
    restorer = FastaiRestorer(model=_tiny_model())
    x, y = restorer._to_tuple(pair_records)
    assert x.shape == y.shape == (4, 3, 32, 32)
    assert (restorer._to_tuple(pair_records, with_targets=False),)[0][0].shape == (4, 3, 32, 32)


def test_the_schedule_import_is_present() -> None:
    """``fit_one_cycle`` is a ``@patch`` installed as an import SIDE EFFECT; without the import it
    falls through fastcore's delegation to the MODEL and raises an AttributeError."""
    source = (SOURCE_DIR / "fastai_restorer.py").read_text()
    assert "from fastai.callback import schedule" in source
