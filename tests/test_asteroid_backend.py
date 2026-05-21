"""Tests for the asteroid backend.

Mocks ``torch`` and ``asteroid.models.BaseModel`` so the test runs without
either heavy dependency installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from distilaide import asteroid_backend


def _install_fake_torch_and_asteroid(
    monkeypatch: pytest.MonkeyPatch, output: np.ndarray
) -> tuple[MagicMock, MagicMock]:
    """Install minimal fakes for ``torch`` and ``asteroid.models.BaseModel``."""
    fake_tensor = MagicMock()
    fake_tensor.to.return_value = fake_tensor

    output_tensor = MagicMock()
    output_tensor.detach.return_value = output_tensor
    output_tensor.cpu.return_value = output_tensor
    output_tensor.numpy.return_value = output

    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = MagicMock(return_value=fake_tensor)  # type: ignore[attr-defined]

    class _NoGrad:
        def __enter__(self) -> "_NoGrad":
            return self

        def __exit__(self, *_: object) -> None:
            return None

    fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model_instance = MagicMock(return_value=output_tensor)
    model_instance.to.return_value = model_instance
    model_instance.eval.return_value = model_instance

    base_model = MagicMock()
    base_model.from_pretrained = MagicMock(return_value=model_instance)

    fake_models = types.ModuleType("asteroid.models")
    fake_models.BaseModel = base_model  # type: ignore[attr-defined]
    fake_asteroid = types.ModuleType("asteroid")
    fake_asteroid.models = fake_models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "asteroid", fake_asteroid)
    monkeypatch.setitem(sys.modules, "asteroid.models", fake_models)
    return base_model, model_instance


class TestCheckpointResolution:
    def test_known_alias(self) -> None:
        assert asteroid_backend._resolve_checkpoint("dccrn", None).startswith(
            "JorisCos/DCCRNet"
        )

    def test_explicit_checkpoint_overrides_alias(self) -> None:
        assert (
            asteroid_backend._resolve_checkpoint("dccrn", "custom/repo")
            == "custom/repo"
        )

    def test_unknown_alias_without_checkpoint_rejected(self) -> None:
        with pytest.raises(ValueError, match="asteroid backend does not serve"):
            asteroid_backend._resolve_checkpoint("not-an-alias", None)

    def test_unknown_alias_with_checkpoint_accepted(self) -> None:
        assert asteroid_backend._resolve_checkpoint("not-an-alias", "x/y") == "x/y"


class TestAvailability:
    def test_unavailable_when_torch_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        assert asteroid_backend.is_available() is False


class TestDenoise:
    def test_mono_dccrn_round_trip(
        self, monkeypatch: pytest.MonkeyPatch, mono_audio: np.ndarray
    ) -> None:
        # asteroid models return (batch, n_src, samples) — for SE n_src==1.
        out = mono_audio[None, None, :].astype(np.float32)
        base_model, _ = _install_fake_torch_and_asteroid(monkeypatch, out)
        result = asteroid_backend.denoise(mono_audio, method="dccrn")
        assert result.shape == mono_audio.shape
        assert result.dtype == mono_audio.dtype
        base_model.from_pretrained.assert_called_once_with(
            "JorisCos/DCCRNet_Libri1Mix_enhsingle_16k"
        )

    def test_separation_first_source_returned(
        self, monkeypatch: pytest.MonkeyPatch, mono_audio: np.ndarray
    ) -> None:
        # ConvTasNet returns multiple sources — we pick the first.
        two_src = np.stack(
            [mono_audio.astype(np.float32), -mono_audio.astype(np.float32)], axis=0
        )[None, ...]
        _install_fake_torch_and_asteroid(monkeypatch, two_src)
        result = asteroid_backend.denoise(mono_audio, method="convtasnet")
        assert result.shape == mono_audio.shape
        # First source should match the positive copy, not the negated one.
        np.testing.assert_allclose(result, mono_audio, atol=1e-5)

    def test_invalid_ndim_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_torch_and_asteroid(
            monkeypatch, np.zeros((1, 1, 16), dtype=np.float32)
        )
        with pytest.raises(ValueError, match="ndim=3"):
            asteroid_backend.denoise(
                np.zeros((2, 3, 4), dtype=np.float32), method="dccrn"
            )

    def test_explicit_checkpoint_used(
        self, monkeypatch: pytest.MonkeyPatch, mono_audio: np.ndarray
    ) -> None:
        out = mono_audio[None, None, :].astype(np.float32)
        base_model, _ = _install_fake_torch_and_asteroid(monkeypatch, out)
        asteroid_backend.denoise(mono_audio, method="dccrn", checkpoint="my/repo")
        base_model.from_pretrained.assert_called_once_with("my/repo")
