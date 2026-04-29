"""Tests for the unified denoise() dispatcher and the Denoiser configurable."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from distilaide import api
from distilaide.api import Denoiser, available_methods, denoise, list_methods


class TestRegistry:
    def test_registry_covers_all_documented_methods(self) -> None:
        expected = {
            "bm3d",
            "noisereduce",
            "spectral-gate",
            "dncnn",
            "drunet",
            "swinir",
            "scunet",
            "restormer",
            "dccrn",
            "dcunet",
            "convtasnet",
            "demucs",
        }
        assert expected.issubset(set(api._REGISTRY.keys()))

    def test_no_duplicate_methods_across_backends(self) -> None:
        # _build_registry would have raised at import time if there were duplicates.
        # Re-running it is the regression check.
        registry = api._build_registry()
        assert len(registry) == len({k for k in registry})


class TestListMethods:
    def test_lists_all_when_unfiltered(self) -> None:
        rows = list_methods()
        methods = {row["method"] for row in rows}
        assert "bm3d" in methods and "dncnn" in methods and "dccrn" in methods

    def test_only_available_filters_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force every backend to report unavailable.
        for module_path in api._BACKEND_MODULES:
            backend = api._load_backend(module_path)
            monkeypatch.setattr(backend, "is_available", lambda: False)
        assert list_methods(only_available=True) == []


class TestDenoiseDispatcher:
    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown denoising method"):
            denoise(np.zeros(4), method="not-a-method")

    def test_unavailable_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = api._load_backend(api._REGISTRY["bm3d"])
        monkeypatch.setattr(backend, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="not available"):
            denoise(np.zeros((4, 4)), method="bm3d")

    def test_dispatch_forwards_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = api._load_backend(api._REGISTRY["bm3d"])
        monkeypatch.setattr(backend, "is_available", lambda: True)
        spy = MagicMock(return_value=np.zeros((4, 4), dtype=np.float32))
        monkeypatch.setattr(backend, "denoise", spy)

        signal = np.ones((4, 4), dtype=np.float32)
        result = denoise(signal, method="bm3d", sigma=0.2)
        assert result.shape == (4, 4)
        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert kwargs["method"] == "bm3d"
        assert kwargs["sigma"] == 0.2


class TestDenoiserConfigurable:
    def test_construct_with_unknown_method_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown denoising method"):
            Denoiser(method="not-a-method")

    def test_backend_property(self) -> None:
        assert Denoiser(method="bm3d").backend == "bm3d"
        assert Denoiser(method="dncnn").backend == "deepinv"
        assert Denoiser(method="dccrn").backend == "asteroid"
        assert Denoiser(method="noisereduce").backend == "noisereduce"

    def test_apply_uses_construction_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = api._load_backend(api._REGISTRY["bm3d"])
        monkeypatch.setattr(backend, "is_available", lambda: True)
        spy = MagicMock(return_value=np.zeros((4, 4), dtype=np.float32))
        monkeypatch.setattr(backend, "denoise", spy)

        Denoiser(method="bm3d", sigma=0.3).apply(np.zeros((4, 4), dtype=np.float32))
        _, kwargs = spy.call_args
        assert kwargs["sigma"] == 0.3

    def test_apply_overrides_win(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = api._load_backend(api._REGISTRY["bm3d"])
        monkeypatch.setattr(backend, "is_available", lambda: True)
        spy: Any = MagicMock(return_value=np.zeros((4, 4), dtype=np.float32))
        monkeypatch.setattr(backend, "denoise", spy)

        Denoiser(method="bm3d", sigma=0.3).apply(np.zeros((4, 4)), sigma=0.05)
        _, kwargs = spy.call_args
        assert kwargs["sigma"] == 0.05

    def test_callable_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = api._load_backend(api._REGISTRY["bm3d"])
        monkeypatch.setattr(backend, "is_available", lambda: True)
        monkeypatch.setattr(backend, "denoise", MagicMock(return_value=np.zeros((4, 4))))
        d = Denoiser(method="bm3d")
        out = d(np.zeros((4, 4)))
        assert out.shape == (4, 4)


class TestAvailableMethods:
    def test_unknown_method_returns_false(self) -> None:
        assert available_methods("not-a-method") is False

    def test_no_arg_polls_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for module_path in api._BACKEND_MODULES:
            backend = api._load_backend(module_path)
            monkeypatch.setattr(backend, "is_available", lambda: False)
        assert available_methods() is False

    def test_specific_method_polls_its_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = api._load_backend(api._REGISTRY["bm3d"])
        monkeypatch.setattr(backend, "is_available", lambda: True)
        assert available_methods("bm3d") is True
