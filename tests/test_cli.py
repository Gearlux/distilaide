"""Tests for the distilaide CLI helpers (input/output IO and command wiring)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from distilaide import api, cli


class TestLoadSignal:
    def test_loads_npy(self, tmp_path: Path) -> None:
        path = tmp_path / "x.npy"
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        np.save(path, data)
        loaded, meta = cli._load_signal(path)
        np.testing.assert_array_equal(loaded, data)
        assert meta == {}

    def test_loads_wav(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Stub scipy.io.wavfile so we don't pull scipy in for unit tests.
        fake_wavfile = types.SimpleNamespace(
            read=MagicMock(return_value=(8000, np.zeros(16, dtype=np.float32)))
        )
        fake_io = types.ModuleType("scipy.io")
        fake_io.wavfile = fake_wavfile  # type: ignore[attr-defined]
        fake_scipy = types.ModuleType("scipy")
        fake_scipy.io = fake_io  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
        monkeypatch.setitem(sys.modules, "scipy.io", fake_io)
        monkeypatch.setitem(sys.modules, "scipy.io.wavfile", fake_wavfile)

        path = tmp_path / "x.wav"
        path.touch()
        signal, meta = cli._load_signal(path)
        assert signal.shape == (16,)
        assert meta["sample_rate"] == 8000

    def test_unsupported_extension_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported input"):
            cli._load_signal(tmp_path / "x.png")


class TestSaveSignal:
    def test_writes_npy(self, tmp_path: Path) -> None:
        path = tmp_path / "out.npy"
        data = np.arange(16, dtype=np.float32)
        cli._save_signal(path, data, {})
        np.testing.assert_array_equal(np.load(path), data)

    def test_writes_wav(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        write_spy = MagicMock()
        fake_wavfile = types.SimpleNamespace(write=write_spy)
        fake_io = types.ModuleType("scipy.io")
        fake_io.wavfile = fake_wavfile  # type: ignore[attr-defined]
        fake_scipy = types.ModuleType("scipy")
        fake_scipy.io = fake_io  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
        monkeypatch.setitem(sys.modules, "scipy.io", fake_io)
        monkeypatch.setitem(sys.modules, "scipy.io.wavfile", fake_wavfile)

        path = tmp_path / "out.wav"
        cli._save_signal(path, np.zeros(8, dtype=np.int16), {"sample_rate": 22050})
        args, _ = write_spy.call_args
        assert args[1] == 22050

    def test_unsupported_extension_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported output"):
            cli._save_signal(tmp_path / "x.png", np.zeros(4), {})


class TestListMethodsCommand:
    def test_prints_a_row_per_method(self, capsys: pytest.CaptureFixture[str]) -> None:
        cli.list_methods_cmd()
        out = capsys.readouterr().out
        assert "bm3d" in out
        assert "dncnn" in out
        assert "dccrn" in out


class TestDenoiseCommand:
    def test_round_trip_via_npy(
        self,
        tmp_path: Path,
        gray_image: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        in_path = tmp_path / "in.npy"
        out_path = tmp_path / "out.npy"
        np.save(in_path, gray_image)

        # Stub get_context to return the test config.
        ctx = MagicMock()
        ctx.config_data = {"input": str(in_path), "output": str(out_path)}
        monkeypatch.setattr(cli, "get_context", lambda: ctx)

        # Stub denoise() to return the input unchanged.
        backend = api._load_backend(api._REGISTRY["bm3d"])
        monkeypatch.setattr(backend, "is_available", lambda: True)
        monkeypatch.setattr(backend, "denoise", MagicMock(return_value=gray_image))

        denoiser = api.Denoiser(method="bm3d")
        cli.denoise_cmd(denoiser)
        assert out_path.exists()
        np.testing.assert_array_equal(np.load(out_path), gray_image)

    def test_missing_input_key_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ctx = MagicMock()
        ctx.config_data = {"output": "x.npy"}
        monkeypatch.setattr(cli, "get_context", lambda: ctx)
        with pytest.raises(ValueError, match="input"):
            cli.denoise_cmd(api.Denoiser(method="bm3d"))

    def test_missing_config_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "get_context", lambda: None)
        with pytest.raises(RuntimeError, match="No distilaide config"):
            cli.denoise_cmd(api.Denoiser(method="bm3d"))
