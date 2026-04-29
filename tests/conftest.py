"""Shared fixtures for distilaide tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def gray_image() -> np.ndarray:
    """A small grayscale image in [0, 1] with a stripe pattern + noise."""
    rng = np.random.default_rng(42)
    base = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (32, 1))
    noise = rng.normal(0.0, 0.05, base.shape).astype(np.float32)
    return np.asarray(np.clip(base + noise, 0.0, 1.0), dtype=np.float32)


@pytest.fixture
def rgb_image() -> np.ndarray:
    """A small RGB image in [0, 1]."""
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 1.0, (16, 16, 3)).astype(np.float32)


@pytest.fixture
def mono_audio() -> np.ndarray:
    """A 1-second mono float32 sine + noise at 16 kHz."""
    rng = np.random.default_rng(1)
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    sig = 0.4 * np.sin(2 * np.pi * 440.0 * t)
    sig += 0.05 * rng.standard_normal(sr).astype(np.float32)
    return sig
