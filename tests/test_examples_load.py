"""The example configs must LOAD — and the degradation chain must produce a real pair.

A config is a user-facing surface: if it stops loading, the project's own documentation is broken.
These build the pipeline for real on synthetic data rather than only parsing YAML, because every
interesting way this chain fails (a resize after the copy, a normalize on one half only) leaves a
document that parses perfectly and a pair that is not a pair.
"""

import inspect
from pathlib import Path

import albumentations as A
import numpy as np
import pytest
import torch
from recordstream import Stream, batch_tensor, collate_records
from recordstream.ops.image import ConvertToImage
from recordstream.ops.structure import CopyField
from recordstream.ops.torch import ToTensor

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _synthetic_source(n: int = 4) -> list:
    """Records shaped like a HuggingFace image dataset's, with no network involved."""
    rng = np.random.default_rng(0)
    return [{"image": rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)} for _ in range(n)]


def _example_chain(sigma: float = 0.1, size: int = 32) -> list:
    """The example config's ``preprocess`` chain, in the same ORDER, built in Python.

    The order is the part worth pinning: everything that must reach BOTH halves goes above the
    copy, everything that degrades only the input goes below it.
    """
    return [
        ConvertToImage(field="image", width=size, height=size, colormap="gray"),
        CopyField(src="image", dst="target"),
        A.GaussNoise(std_range=(sigma, sigma), mean_range=(0.0, 0.0), per_channel=True, p=1.0),
        ToTensor(field="image", normalize=True),
        ToTensor(field="target", normalize=True),
    ]


# --------------------------------------------------------------------------- #
# The files exist and parse
# --------------------------------------------------------------------------- #
def test_the_example_config_exists() -> None:
    assert (EXAMPLES / "train_dncnn_denoising.yaml").exists()


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")))
def test_every_example_is_valid_yaml(path: Path) -> None:
    """Tag-agnostic: this is about the document being well-formed, not about resolving classes."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor("!", lambda loader, suffix, node: None)
    assert yaml.load(path.read_text(), Loader=_Loader) is not None  # noqa: S506 - restricted local loader


# --------------------------------------------------------------------------- #
# The chain produces a genuine degraded/clean PAIR
# --------------------------------------------------------------------------- #
def test_the_chain_produces_both_halves_of_the_pair() -> None:
    records = list(Stream(source=_synthetic_source(), ops=_example_chain()))
    assert {"image", "target"} <= set(records[0])
    batch = collate_records(records)
    x, y = batch_tensor(batch, "image"), batch_tensor(batch, "target")
    assert x.shape == y.shape == (4, 3, 32, 32)
    assert x.dtype == y.dtype == torch.float32


def test_both_halves_land_in_the_same_value_range() -> None:
    """``data_range: 1.0`` is only true if BOTH halves were normalized — a normalized input
    against a uint8 target is a loss that never converges and never says why."""
    batch = collate_records(list(Stream(source=_synthetic_source(), ops=_example_chain())))
    for key in ("image", "target"):
        values = batch_tensor(batch, key)
        assert 0.0 <= float(values.min()) and float(values.max()) <= 1.0


def test_the_target_is_the_CLEAN_half_and_the_input_is_the_DEGRADED_one() -> None:
    """The direction matters and nothing else checks it: swapped, the model would learn to ADD
    noise and every metric would still compute."""
    clean = list(Stream(source=_synthetic_source(), ops=_example_chain(sigma=0.0)))
    noisy = list(Stream(source=_synthetic_source(), ops=_example_chain(sigma=0.2)))
    reference = batch_tensor(collate_records(clean), "target")
    degraded_input = batch_tensor(collate_records(noisy), "image")
    degraded_target = batch_tensor(collate_records(noisy), "target")
    # The target is untouched by the degradation; the input is not.
    assert torch.allclose(degraded_target, reference)
    assert not torch.allclose(degraded_input, reference)


def test_the_degradation_strength_follows_the_configured_sigma() -> None:
    """``std_range: [s, s]`` is a FIXED sigma, not a random draw — which is what lets a
    noise-conditioned architecture be told the level it will face."""
    residuals = {}
    for sigma in (0.05, 0.2):
        batch = collate_records(list(Stream(source=_synthetic_source(), ops=_example_chain(sigma=sigma))))
        residual = batch_tensor(batch, "image") - batch_tensor(batch, "target")
        residuals[sigma] = float(residual.std())
    assert residuals[0.05] < residuals[0.2]
    # Clipping at 0 and 1 pulls the realised sigma slightly BELOW the requested one; it must not
    # be wildly off, which would mean the op is operating on a different scale than assumed.
    assert 0.5 * 0.2 < residuals[0.2] <= 0.2


def test_the_resize_happens_before_the_copy() -> None:
    """The ordering rule the config's comments state, pinned.

    A bare albumentations transform only ever receives albumentations' OWN key vocabulary
    (``image`` / ``mask`` / …), so a geometric op placed AFTER the copy would move the input and
    leave the target at its original size — unpairing every record while still producing a
    perfectly valid-looking batch.
    """
    ops = _example_chain(size=32)
    names = [type(op).__name__ for op in ops]
    assert names.index("ConvertToImage") < names.index("CopyField")
    assert names.index("CopyField") < names.index("GaussNoise")


def test_a_resize_after_the_copy_would_unpair_the_record() -> None:
    """The counterfactual, executed — so the rule above is evidence rather than assertion."""
    broken = [
        ConvertToImage(field="image", width=48, height=48, colormap="gray"),
        CopyField(src="image", dst="target"),
        A.Resize(height=16, width=16),  # AFTER the copy: reaches `image` only
        ToTensor(field="image", normalize=True),
        ToTensor(field="target", normalize=True),
    ]
    record = next(iter(Stream(source=_synthetic_source(1), ops=broken)))
    assert record["image"].shape != record["target"].shape


class TestTheConfigIsSeeded:
    """Reproducibility took TWO knobs, and either one going missing is silent.

    Measured 2026-08-06 on two runs of the identical `--framework torch` cell: unseeded 22.7530 /
    22.5188 dB, model-seeded-only 22.1058 / 22.1707, both seeded 22.1886 / 22.1886. A config that
    silently loses one of them still trains, still reports, and stops reproducing.
    """

    def test_every_degradation_is_wrapped_in_a_seeded_compose(self) -> None:
        """A BARE albumentations transform cannot take a seed — it builds its RNG from OS entropy
        and warns that `seed` is not a valid argument — so the only way to pin the noise is a
        `Compose`, and every chain in the file needs its own."""
        text = (EXAMPLES / "train_dncnn_denoising.yaml").read_text()

        composes = text.count("!class:albumentations.Compose")
        noises = text.count("!class:albumentations.GaussNoise")

        assert noises >= 5, "expected one degradation per preprocess chain"
        assert composes == noises, f"{noises - composes} GaussNoise chain(s) are outside a seeded Compose"
        assert text.count("seed: 0") == composes

    def test_a_bare_transform_really_does_refuse_a_seed(self) -> None:
        """The PREMISE, executed — without it the wrapper above looks like ceremony."""

        assert "seed" not in inspect.signature(A.GaussNoise.__init__).parameters

    def test_two_fresh_seeded_composes_draw_identically(self) -> None:
        """What makes a RUN reproduce: a fresh process rebuilds the Compose at the same seed."""

        image = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
        build = lambda: A.Compose([A.GaussNoise(std_range=(0.1, 0.1), p=1.0)], seed=0)  # noqa: E731

        assert np.array_equal(build()(image=image.copy())["image"], build()(image=image.copy())["image"])

    def test_ONE_compose_called_twice_does_NOT_repeat_itself(self) -> None:
        """The other half, and it is what a per-record stream needs: a seeded Compose gives a
        deterministic SEQUENCE, not the same noise on every record."""

        image = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
        compose = A.Compose([A.GaussNoise(std_range=(0.1, 0.1), p=1.0)], seed=0)

        assert not np.array_equal(compose(image=image.copy())["image"], compose(image=image.copy())["image"])

    def test_a_model_seed_must_not_restore_the_rng_state(self) -> None:
        """The leak is the mechanism, not an oversight — a save-and-restore version was written
        first and measured to leave two runs 0.19 dB apart, because it fixes the WEIGHTS and leaves
        the loader's shuffle order unseeded.
        """
        for module in ("deepinv_model.py", "huggingface_model.py", "dncnn/torch_dncnn.py"):
            source = (Path(__file__).resolve().parent.parent / "distilaide" / module).read_text()
            assert "torch.manual_seed(self.seed)" in source, f"{module} does not seed its build"
            assert "set_rng_state" not in source, f"{module} restores the RNG — that reproduces nothing"
