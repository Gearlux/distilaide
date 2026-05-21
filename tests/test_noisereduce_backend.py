"""Tests for the noisereduce backend (mocks the upstream library)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from distilaide import noisereduce_backend


def _install_fake_nr(monkeypatch: pytest.MonkeyPatch, return_value: np.ndarray) -> MagicMock:
    mock_callable = MagicMock(return_value=return_value)
    fake_module = types.ModuleType("noisereduce")
    fake_module.reduce_noise = mock_callable  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "noisereduce", fake_module)
    return mock_callable


class TestNoiseReduceAvailability:
    def test_is_available_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_nr(monkeypatch, np.zeros(4, dtype=np.float32))
        assert noisereduce_backend.is_available() is True

    def test_is_available_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "noisereduce", None)
        assert noisereduce_backend.is_available() is False


class TestNoiseReduceDenoise:
    def test_mono_signal_passes_sample_rate(self, monkeypatch: pytest.MonkeyPatch, mono_audio: np.ndarray) -> None:
        mock = _install_fake_nr(monkeypatch, mono_audio.copy())
        out = noisereduce_backend.denoise(mono_audio, sample_rate=16000)
        assert out.shape == mono_audio.shape
        assert out.dtype == mono_audio.dtype
        _, kwargs = mock.call_args
        assert kwargs["sr"] == 16000
        assert kwargs["stationary"] is True
        assert kwargs["y_noise"] is None

    def test_noise_clip_forwarded(self, monkeypatch: pytest.MonkeyPatch, mono_audio: np.ndarray) -> None:
        mock = _install_fake_nr(monkeypatch, mono_audio.copy())
        clip = mono_audio[:1000]
        noisereduce_backend.denoise(mono_audio, sample_rate=16000, noise_clip=clip)
        _, kwargs = mock.call_args
        np.testing.assert_array_equal(kwargs["y_noise"], clip.astype(np.float32))

    def test_extra_kwargs_forwarded(self, monkeypatch: pytest.MonkeyPatch, mono_audio: np.ndarray) -> None:
        mock = _install_fake_nr(monkeypatch, mono_audio.copy())
        noisereduce_backend.denoise(mono_audio, sample_rate=16000, prop_decrease=0.7)
        _, kwargs = mock.call_args
        assert kwargs["prop_decrease"] == 0.7

    def test_unknown_method_rejected(self, mono_audio: np.ndarray) -> None:
        with pytest.raises(ValueError, match="noisereduce backend"):
            noisereduce_backend.denoise(mono_audio, method="bm3d")

    def test_alias_method_accepted(self, monkeypatch: pytest.MonkeyPatch, mono_audio: np.ndarray) -> None:
        _install_fake_nr(monkeypatch, mono_audio.copy())
        out = noisereduce_backend.denoise(mono_audio, method="spectral-gate", sample_rate=16000)
        assert out.shape == mono_audio.shape
