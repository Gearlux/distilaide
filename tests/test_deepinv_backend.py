"""Tests for the deepinv backend.

Mocks ``deepinv.models.*`` and ``torch`` so the test runs without those heavy
dependencies. Reshape helpers are tested directly (they're pure numpy).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from distilaide import deepinv_backend


class TestReshapeHelpers:
    def test_2d_to_nchw_and_back(self) -> None:
        img = np.arange(16, dtype=np.float32).reshape(4, 4)
        nchw, shape = deepinv_backend._to_nchw(img)
        assert nchw.shape == (1, 1, 4, 4)
        round_trip = deepinv_backend._from_nchw(nchw, shape)
        np.testing.assert_array_equal(round_trip, img)

    def test_3d_hwc_to_nchw_and_back(self) -> None:
        img = np.arange(48, dtype=np.float32).reshape(4, 4, 3)
        nchw, shape = deepinv_backend._to_nchw(img)
        assert nchw.shape == (1, 3, 4, 4)
        round_trip = deepinv_backend._from_nchw(nchw, shape)
        np.testing.assert_array_equal(round_trip, img)

    def test_4d_passthrough(self) -> None:
        batch = np.zeros((2, 3, 4, 4), dtype=np.float32)
        nchw, shape = deepinv_backend._to_nchw(batch)
        assert nchw.shape == batch.shape
        np.testing.assert_array_equal(deepinv_backend._from_nchw(nchw, shape), batch)

    def test_invalid_ndim_rejected(self) -> None:
        with pytest.raises(ValueError, match="ndim=1"):
            deepinv_backend._to_nchw(np.zeros(4, dtype=np.float32))


class TestAvailability:
    def test_unavailable_when_torch_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        assert deepinv_backend.is_available() is False


def _install_fake_torch_and_deepinv(
    monkeypatch: pytest.MonkeyPatch, output: np.ndarray
) -> tuple[MagicMock, MagicMock]:
    """Install minimal fakes for ``torch`` and ``deepinv`` that round-trip a numpy array."""
    fake_tensor = MagicMock()
    fake_tensor.to.return_value = fake_tensor
    fake_tensor.shape = output.shape

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
    model_class = MagicMock(return_value=model_instance)

    fake_models = types.ModuleType("deepinv.models")
    for name in ("DnCNN", "DRUNet", "SwinIR", "SCUNet", "Restormer"):
        setattr(fake_models, name, model_class)
    fake_deepinv = types.ModuleType("deepinv")
    fake_deepinv.models = fake_models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepinv", fake_deepinv)
    monkeypatch.setitem(sys.modules, "deepinv.models", fake_models)
    return model_class, model_instance


class TestDenoise:
    def test_dispatch_to_dncnn(
        self, monkeypatch: pytest.MonkeyPatch, gray_image: np.ndarray
    ) -> None:
        model_class, model_instance = _install_fake_torch_and_deepinv(
            monkeypatch, gray_image[None, None, :, :].astype(np.float32)
        )
        out = deepinv_backend.denoise(gray_image, method="dncnn")
        assert out.shape == gray_image.shape
        assert out.dtype == gray_image.dtype
        # pretrained weights flag is part of the constructor contract.
        _, kwargs = model_class.call_args
        assert kwargs["pretrained"] == "download"

    def test_sigma_forwarded_to_drunet(
        self, monkeypatch: pytest.MonkeyPatch, gray_image: np.ndarray
    ) -> None:
        _, model_instance = _install_fake_torch_and_deepinv(
            monkeypatch, gray_image[None, None, :, :].astype(np.float32)
        )
        deepinv_backend.denoise(gray_image, method="drunet", sigma=0.07)
        _, kwargs = model_instance.call_args
        assert kwargs["sigma"] == 0.07

    def test_sigma_ignored_for_blind_models(
        self, monkeypatch: pytest.MonkeyPatch, gray_image: np.ndarray
    ) -> None:
        _, model_instance = _install_fake_torch_and_deepinv(
            monkeypatch, gray_image[None, None, :, :].astype(np.float32)
        )
        deepinv_backend.denoise(gray_image, method="dncnn", sigma=0.07)
        _, kwargs = model_instance.call_args
        assert "sigma" not in kwargs

    def test_unknown_method_rejected(self, gray_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="deepinv backend"):
            deepinv_backend._build_model("not-a-real-method")

    def test_rgb_round_trips_through_dispatcher(
        self, monkeypatch: pytest.MonkeyPatch, rgb_image: np.ndarray
    ) -> None:
        nchw_out = np.transpose(rgb_image, (2, 0, 1))[None, ...].astype(np.float32)
        _install_fake_torch_and_deepinv(monkeypatch, nchw_out)
        out = deepinv_backend.denoise(rgb_image, method="swinir")
        assert out.shape == rgb_image.shape
