"""Tests for the BM3D backend.

Mocks ``bm3d.bm3d`` so the test runs without the upstream package installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from distilaide import bm3d_backend


def _install_fake_bm3d(monkeypatch: pytest.MonkeyPatch, return_value: np.ndarray) -> MagicMock:
    """Install a fake ``bm3d`` module whose ``bm3d`` callable returns ``return_value``."""
    mock_callable = MagicMock(return_value=return_value)
    fake_module = types.ModuleType("bm3d")
    fake_module.bm3d = mock_callable  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bm3d", fake_module)
    return mock_callable


class TestBm3dBackendAvailability:
    def test_is_available_true_when_module_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_bm3d(monkeypatch, np.zeros((4, 4)))
        assert bm3d_backend.is_available() is True

    def test_is_available_false_when_module_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "bm3d", None)
        assert bm3d_backend.is_available() is False


class TestBm3dDenoise:
    def test_float_passthrough_keeps_dtype_and_shape(
        self, monkeypatch: pytest.MonkeyPatch, gray_image: np.ndarray
    ) -> None:
        expected = gray_image * 0.5
        mock = _install_fake_bm3d(monkeypatch, expected.astype(np.float64))
        out = bm3d_backend.denoise(gray_image, sigma=0.05)
        assert out.shape == gray_image.shape
        assert out.dtype == gray_image.dtype
        mock.assert_called_once()
        _, kwargs = mock.call_args
        assert kwargs["sigma_psd"] == 0.05

    def test_uint8_input_is_scaled_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        signal = np.full((8, 8), 128, dtype=np.uint8)
        # bm3d sees roughly 0.5 in float64 — return a slightly cleaner version.
        _install_fake_bm3d(monkeypatch, np.full((8, 8), 0.5, dtype=np.float64))
        out = bm3d_backend.denoise(signal, sigma=0.05)
        assert out.dtype == np.uint8
        assert out.shape == signal.shape
        # 0.5 * 255 → 127 or 128 depending on rounding.
        assert 126 <= int(out[0, 0]) <= 128

    def test_unknown_method_rejected(self, gray_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="bm3d backend"):
            bm3d_backend.denoise(gray_image, method="dncnn")

    def test_method_alias_accepted(self, monkeypatch: pytest.MonkeyPatch, gray_image: np.ndarray) -> None:
        _install_fake_bm3d(monkeypatch, gray_image.astype(np.float64))
        # Default kwarg method="bm3d" must round-trip.
        out = bm3d_backend.denoise(gray_image, method="bm3d", sigma=0.1)
        assert out.shape == gray_image.shape
