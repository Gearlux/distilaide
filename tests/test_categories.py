# mypy: disable-error-code="attr-defined"
"""Discovery tags for distilaide's ``@configurable`` classes — task / role / framework.

These drive navigaitor's ``list_classes(task=, role=, framework=)`` scan, its generated task
configs, and the visual-editor form-spec picker. A dropped tag empties the corresponding picker
SILENTLY — the class still resolves from a dotted path in YAML, so nothing else notices — which is
why they are pinned here rather than left to be discovered by a user with a short menu.
"""

import pytest
from confluid.registry import get_registry

import distilaide.torch_restoration as torch_restoration
from distilaide.deepinv_model import DeepInvModel
from distilaide.lightning_restorer import LightningRestorer
from distilaide.torch_restorer import TorchRestorer

pytest.importorskip("torch")


def test_model_tags() -> None:
    assert DeepInvModel.__confluid_task__ == "restoration"
    assert DeepInvModel.__confluid_role__ == "model"
    assert DeepInvModel.__confluid_framework__ == "torch"
    #: The derived legacy category, which the older pickers still read.
    assert DeepInvModel.__confluid_category__ == "restoration_model"


def test_restorer_tags() -> None:
    """One class, ``role="trainer"`` — its evaluate/test/predict capabilities are the SAME class.

    They are declared by the ``@entrypoint``-annotated methods, not by a second class and not by a
    ``role="evaluator"`` tag, so there is intentionally no separate evaluator here.
    """
    for cls in (LightningRestorer, TorchRestorer):
        assert getattr(cls, "__confluid_task__") == "restoration"
        assert getattr(cls, "__confluid_role__") == "trainer"
        assert getattr(cls, "__confluid_framework__") == "torch"


def test_tags_enumerable_via_registry() -> None:
    registry = get_registry()
    assert "DeepInvModel" in registry.list_classes(task="restoration", role="model")
    assert "LightningRestorer" in registry.list_classes(task="restoration", role="trainer")
    assert "TorchRestorer" in registry.list_classes(task="restoration", role="trainer")


def test_the_framework_axis_separates_engines() -> None:
    """``framework`` says what a class can be WIRED TO, so a picker cannot offer a mismatch."""
    registry = get_registry()
    assert {"DeepInvModel", "LightningRestorer"} <= registry.list_classes(task="restoration", framework="torch")


def test_losses_and_metrics_are_registered_with_their_roles() -> None:
    registry = get_registry()
    losses = registry.list_classes(task="restoration", role="loss", framework="torch")
    metrics = registry.list_classes(task="restoration", role="metric", framework="torch")
    assert {"L1Loss", "MSELoss"} <= set(losses)
    assert {"PeakSignalNoiseRatio", "StructuralSimilarityIndexMeasure"} <= set(metrics)


def test_every_registered_loss_and_metric_is_named_in_dunder_all() -> None:
    """``scan_module`` filters members by ``__module__``, which for these is ``torch`` /
    ``torchmetrics`` — so the ``__all__``-gated pass is the ONLY thing carrying them into the
    palette. A class registered but unlisted is registered and invisible."""
    registered = {cls.__name__ for cls in torch_restoration._RESTORATION_LOSSES}
    registered |= {cls.__name__ for cls in torch_restoration._RESTORATION_METRICS}
    assert registered == set(torch_restoration.__all__)


def test_the_defaults_are_in_the_registered_set() -> None:
    """A run's default loss and metrics must be nameable from a config, or a user cannot restate
    what they are already getting."""
    for name in ("L1Loss", "PeakSignalNoiseRatio", "StructuralSimilarityIndexMeasure"):
        assert name in torch_restoration.__all__


def test_mse_is_registered_beside_l1() -> None:
    """DnCNN's original recipe was MSE; reproducing it must be one line, not a code change."""
    assert "MSELoss" in torch_restoration.__all__
