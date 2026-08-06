"""The shared DnCNN spec, and the FIVE implementations that must agree with it.

The point of the package is a claim about the OTHER implementations: they are the same network, so
a cross-framework comparison compares FRAMEWORKS rather than five different models. Nothing else
enforces that — different APIs cannot share code — so the derived parameter count is what holds
them together, and each implementation is checked against the CLAIM rather than against whichever
one was written first.

Every implementation needs a framework a default install may lack, so each is imported behind a
skip. `test_the_agreement_is_not_vacuous` refuses to let the whole file pass on an empty set.
"""

import importlib
from typing import Any, Dict, List, Tuple

import pytest

from distilaide.dncnn import spec

#: ``label -> (module, class)``. A list rather than a hard import, because no venv is guaranteed to
#: carry all five engines — and the machine missing one is exactly where a copy-pasted mistake
#: lands, so the file must still run there.
IMPLEMENTATIONS: List[Tuple[str, str, str]] = [
    ("torch", "distilaide.dncnn.torch_dncnn", "TorchDnCNN"),
    ("keras", "distilaide.dncnn.keras_dncnn", "KerasDnCNN"),
    ("tensorflow", "distilaide.dncnn.tensorflow_dncnn", "TensorFlowDnCNN"),
    ("flax", "distilaide.dncnn.flax_dncnn", "FlaxDnCNN"),
    ("mlx", "distilaide.dncnn.mlx_dncnn", "MlxDnCNN"),
]


def _load(module_path: str, class_name: str) -> Any:
    """Import an implementation, or skip when its framework is absent."""
    try:
        return getattr(importlib.import_module(module_path), class_name)
    except ImportError as exc:  # pragma: no cover - depends on the local extras
        pytest.skip(f"{class_name} needs a framework that is not installed here ({exc})")


def _available() -> Dict[str, Any]:
    """Every implementation this machine can actually import."""
    found = {}
    for label, module_path, class_name in IMPLEMENTATIONS:
        try:
            found[label] = getattr(importlib.import_module(module_path), class_name)
        except ImportError:  # pragma: no cover - depends on the local extras
            continue
    return found


# --------------------------------------------------------------------------- #
# The spec derives its own claim
# --------------------------------------------------------------------------- #
def test_the_derived_count_is_arithmetic_not_a_measurement() -> None:
    """Head + body + tail, written out — so it is a CLAIM, not one framework's answer recorded."""
    k = spec.KERNEL_SIZE**2
    head = k * 3 * spec.WIDTH + spec.WIDTH
    body = (spec.DEPTH - 2) * (k * spec.WIDTH * spec.WIDTH + spec.WIDTH + 2 * spec.WIDTH)
    tail = k * spec.WIDTH * 3 + 3
    assert spec.trainable_parameter_count() == head + body + tail == 57_635


def test_the_count_tracks_its_arguments() -> None:
    """A spec that ignored its knobs would make every implementation "agree" trivially."""
    assert spec.trainable_parameter_count(depth=10) > spec.trainable_parameter_count(depth=8)
    assert spec.trainable_parameter_count(width=64) > spec.trainable_parameter_count(width=32)
    assert spec.trainable_parameter_count(in_channels=1) < spec.trainable_parameter_count(in_channels=3)


def test_a_degenerate_depth_is_refused() -> None:
    """Below two there is no head-and-tail to have, and the body term would go negative."""
    with pytest.raises(ValueError, match="depth >= 2"):
        spec.trainable_parameter_count(depth=1)


def test_there_is_no_size_divisor() -> None:
    """Unlike a U-Net, DnCNN never changes the spatial size — so no raster constraint exists here.

    Worth pinning because the sibling projects' specs DO export one, and adding it by analogy
    would be a constraint this architecture does not have.
    """
    assert not hasattr(spec, "size_divisor")


# --------------------------------------------------------------------------- #
# Each implementation against the claim
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,module_path,class_name", IMPLEMENTATIONS)
def test_each_implementation_reports_the_derived_count(label: str, module_path: str, class_name: str) -> None:
    model = _load(module_path, class_name)()
    assert model.trainable_parameter_count == spec.trainable_parameter_count()
    assert model.trainable_parameter_count == model.spec_parameter_count()


@pytest.mark.parametrize("label,module_path,class_name", IMPLEMENTATIONS)
def test_each_implementation_agrees_under_CUSTOM_settings(label: str, module_path: str, class_name: str) -> None:
    """The defaults agreeing could be a coincidence of one shared constant; the knobs cannot be.

    This is also what catches an implementation that hard-codes the spec's constants instead of
    reading its own arguments.
    """
    cls = _load(module_path, class_name)
    model = cls(input_shape=(64, 64, 1), depth=5, width=8)
    assert model.trainable_parameter_count == spec.trainable_parameter_count(in_channels=1, depth=5, width=8)


@pytest.mark.parametrize("label,module_path,class_name", IMPLEMENTATIONS)
def test_each_implementation_excludes_batch_norm_state(label: str, module_path: str, class_name: str) -> None:
    """Every framework needs a DIFFERENT mechanism for the same exclusion, and that is the wrinkle.

    torch holds the running statistics as BUFFERS (``requires_grad`` filtering does it), Keras as
    non-trainable WEIGHTS, raw TensorFlow as ``trainable=False`` variables, Flax as ``nnx.BatchStat``
    (a different variable TYPE in one state tree) — and MLX keeps them in the SAME tree as the
    weights, so there the exclusion is a NAME filter. If any one of them counted them, this would
    exceed the derived total by ``2 * width`` per body layer.
    """
    model = _load(module_path, class_name)()
    body_layers = spec.DEPTH - 2
    running_stats = body_layers * 2 * spec.WIDTH
    assert model.trainable_parameter_count == spec.trainable_parameter_count()
    assert model.trainable_parameter_count != spec.trainable_parameter_count() + running_stats


@pytest.mark.parametrize("label,module_path,class_name", IMPLEMENTATIONS)
def test_the_channel_count_is_read_from_the_LAST_axis(label: str, module_path: str, class_name: str) -> None:
    """``input_shape`` is ``(H, W, C)`` in EVERY implementation, including the NCHW one.

    Reading it as NCHW would build a first convolution over 192 input channels instead of 3 —
    still runnable, and silently a different network.
    """
    model = _load(module_path, class_name)(input_shape=(192, 192, 3))
    assert model.in_channels == 3


# --------------------------------------------------------------------------- #
# ...and against each other
# --------------------------------------------------------------------------- #
def test_all_available_implementations_agree_with_the_spec_and_each_other() -> None:
    counts = {label: cls().trainable_parameter_count for label, cls in _available().items()}
    assert counts, "no implementation could be imported"
    assert set(counts.values()) == {spec.trainable_parameter_count()}, counts


def test_the_agreement_is_not_vacuous() -> None:
    """Guard: with one implementation installed, "they all agree" is not a claim about anything."""
    available = _available()
    assert len(available) >= 2, (
        f"only {sorted(available)} could be imported — the cross-framework claim needs at least two. "
        "Install another engine's extra before trusting this file."
    )
