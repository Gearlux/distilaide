"""``DeepInvModel`` — one wrapper over deepinv's whole restoration zoo.

What is pinned here is the wrapper's three claims: the ``arch`` Literal is a real map onto
``deepinv.models``, the constructor does NO work, and the knob translation is derived from the
target class's signature rather than from a table that could outdate.
"""

import inspect
from typing import Any, Dict, get_args

import pytest
import torch

from distilaide.deepinv_model import (
    _ARCH_CLASS_NAMES,
    _CHANNEL_PARAM_NAMES,
    RESTORATION_ARCHITECTURES,
    DeepInvModel,
    RestorationArchitecture,
)

deepinv_models = pytest.importorskip("deepinv.models")

#: Architectures cheap enough to BUILD in a unit test. The transformers (swinir / restormer /
#: scunet) allocate tens of millions of parameters, and `bm3d` / `wavelet` need native packages
#: that are optional extras — so the build-and-run checks use this subset while the CONTRACT
#: checks below cover every member of the Literal.
_CHEAP = ("dncnn", "unet", "tv", "median")


# --------------------------------------------------------------------------- #
# The Literal is a real map onto deepinv, in both directions
# --------------------------------------------------------------------------- #
def test_the_runtime_tuple_is_derived_from_the_literal() -> None:
    """One source of truth: a second hand-written list is what drifts."""
    assert RESTORATION_ARCHITECTURES == get_args(RestorationArchitecture)


def test_every_architecture_resolves_to_a_real_deepinv_class() -> None:
    """The check a deepinv release should fail loudly: a renamed class must not go unofferable."""
    for arch in RESTORATION_ARCHITECTURES:
        class_name = _ARCH_CLASS_NAMES[arch]
        assert hasattr(deepinv_models, class_name), f"deepinv.models has no {class_name!r} (for arch={arch!r})"


def test_the_class_table_covers_the_literal_exactly() -> None:
    """A member with no class entry raises a KeyError at build; an orphan entry is unreachable."""
    assert set(_ARCH_CLASS_NAMES) == set(RESTORATION_ARCHITECTURES)


def test_an_unknown_architecture_is_refused_by_name() -> None:
    model = DeepInvModel()
    # A bad value assigned POST-construction, which is exactly how `configure()` or a plain
    # setattr would deliver one — bypassing the pydantic check the constructor runs.
    model.arch = "not-an-architecture"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="must be one of"):
        model.solidify()


# --------------------------------------------------------------------------- #
# Partial construction
# --------------------------------------------------------------------------- #
def test_the_constructor_does_no_work() -> None:
    """Zero-arg construction works and builds NOTHING — the workspace lazy-init mandate."""
    model = DeepInvModel()
    assert model.restorer is None
    assert list(model.parameters()) == []


def test_solidify_builds_once_and_caches() -> None:
    model = DeepInvModel(arch="dncnn", depth=4)
    first = model.solidify()
    assert model.solidify() is first
    # `flow()` fires the hook on a LIVE object too, so a normal run calls it more than once.
    assert model.backbone is first


def test_forward_builds_on_demand_for_a_never_flowed_instance() -> None:
    out = DeepInvModel(arch="dncnn", depth=4)(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 3, 32, 32)


# --------------------------------------------------------------------------- #
# The knob translation is read from the target's SIGNATURE
# --------------------------------------------------------------------------- #
def test_the_channel_knob_reaches_whichever_name_the_architecture_uses() -> None:
    """The zoo spells "input channels" five ways; this wrapper declares ONE knob.

    Measured against deepinv 0.4.1: ``in_channels`` (DnCNN / DRUNet / Restormer / UNet),
    ``in_nc`` (SCUNet), ``in_chans`` (SwinIR), ``img_channels`` (FFDNet).
    """
    model = DeepInvModel(arch="dncnn", in_channels=1, depth=4)
    kwargs = model._build_kwargs(model._architecture_class())
    assert kwargs["in_channels"] == 1
    assert model(torch.rand(1, 1, 16, 16)).shape == (1, 1, 16, 16)


def test_out_channels_defaults_to_the_input_count() -> None:
    """A restoration model that changed the channel count would not be a restoration model."""
    model = DeepInvModel(arch="dncnn", in_channels=1)
    assert model._build_kwargs(model._architecture_class())["out_channels"] == 1


def test_an_explicit_kwarg_wins_over_the_derived_one() -> None:
    """The pass-through mandate: a knob the caller wrote reaches the library verbatim.

    The derivation is a ``setdefault``, never an override, so an explicit value reaches deepinv
    unchanged. What deepinv then DOES with it is deepinv's business and deliberately not asserted
    here: ``DnCNN`` is a residual network that adds its input back, so its output keeps the INPUT's
    channel count whatever ``out_channels`` says (measured — a 3-channel input still comes back
    3-channel with ``out_channels=1``). Asserting the output shape would be testing the library.
    """
    model = DeepInvModel(arch="dncnn", in_channels=3, out_channels=1, depth=4)
    assert model._build_kwargs(model._architecture_class())["out_channels"] == 1


def test_a_channel_agnostic_architecture_is_not_handed_a_channel_count() -> None:
    """The classical denoisers take no channel argument, and passing one would raise."""
    model = DeepInvModel(arch="median", kernel_size=3)
    kwargs = model._build_kwargs(model._architecture_class())
    assert not any(name in kwargs for name in _CHANNEL_PARAM_NAMES)


def test_pretrained_is_only_passed_to_architectures_that_declare_it() -> None:
    """``UNet`` takes no ``pretrained``; handing it one is a TypeError, not a no-op."""
    model = DeepInvModel(arch="unet", pretrained="download")
    assert "pretrained" not in model._build_kwargs(model._architecture_class())


def test_the_default_is_to_train_from_scratch() -> None:
    """A deliberate DIVERGENCE from deepinv's own ``pretrained="download"`` default.

    distilaide is a training project: a bare ``arch: dncnn`` must not silently start from somebody
    else's weights, and must not need the network to build.
    """
    assert DeepInvModel().pretrained is None
    assert inspect.signature(DeepInvModel.__init__).parameters["pretrained"].default is None


# --------------------------------------------------------------------------- #
# Learned vs classical — the distinction the task depends on
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arch", ["dncnn", "unet"])
def test_a_learned_architecture_reports_trainable(arch: str) -> None:
    assert DeepInvModel(arch=arch).is_trainable  # type: ignore[arg-type]


@pytest.mark.parametrize("arch", ["tv", "median"])
def test_a_classical_architecture_reports_no_parameters(arch: str) -> None:
    """Fixed ALGORITHMS wearing the ``nn.Module`` interface — zero parameters, measured."""
    model = DeepInvModel(arch=arch)  # type: ignore[arg-type]
    assert not model.is_trainable
    assert sum(p.numel() for p in model.backbone.parameters()) == 0


@pytest.mark.parametrize("arch", _CHEAP)
def test_every_cheap_architecture_maps_an_image_to_an_image_of_the_same_shape(arch: str) -> None:
    """The one contract every member of the zoo shares, learned or not."""
    per_arch: Dict[str, Dict[str, Any]] = {"dncnn": {"depth": 4}, "tv": {"n_it_max": 3, "sigma": 0.1}}
    out = DeepInvModel(arch=arch, **per_arch.get(arch, {}))(torch.rand(2, 3, 32, 32))  # type: ignore[arg-type]
    assert out.shape == (2, 3, 32, 32)


def test_a_noise_conditioned_architecture_fails_loudly_without_a_sigma() -> None:
    """Some architectures REQUIRE the noise level, and that is why ``sigma`` is a model knob.

    ``TVDenoiser`` reads the second forward argument as its regularization threshold and has no
    usable default, so it raises rather than guessing (measured: *"Regularization parameter (ths)
    was not passed at init nor at forward"*). ``DRUNet`` and ``BM3D`` are the same shape of
    requirement. A blind denoiser such as DnCNN needs no such line — which is why the knob
    defaults to ``None`` instead of to a number this wrapper would have invented.
    """
    with pytest.raises(RuntimeError, match="ths"):
        DeepInvModel(arch="tv", n_it_max=3)(torch.rand(1, 3, 16, 16))


def test_sigma_is_forwarded_positionally() -> None:
    """deepinv's zoo agrees on the POSITION of the noise level, not on its name.

    ``TVDenoiser`` calls the same slot ``ths``, so a keyword call would fail on it while passing
    for every denoiser — which is why the wrapper passes it positionally.
    """
    out = DeepInvModel(arch="tv", n_it_max=3, sigma=0.1)(torch.rand(1, 3, 16, 16))
    assert out.shape == (1, 3, 16, 16)


def test_a_per_call_sigma_overrides_the_configured_one() -> None:
    model = DeepInvModel(arch="tv", n_it_max=3, sigma=0.01)
    strong = model(torch.rand(1, 3, 16, 16), sigma=0.5)
    assert strong.shape == (1, 3, 16, 16)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_a_non_4d_input_is_refused() -> None:
    with pytest.raises(ValueError, match="4D"):
        DeepInvModel(arch="dncnn", depth=4)(torch.rand(3, 32, 32))


def test_a_channel_mismatch_is_refused_by_the_wrapper() -> None:
    with pytest.raises(ValueError, match="in_channels=3"):
        DeepInvModel(arch="dncnn", depth=4)(torch.rand(1, 1, 32, 32))
