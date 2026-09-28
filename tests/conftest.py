"""Shared fixtures for the distilaide suite — a synthetic degraded/clean image PAIR.

Every backend here consumes the same thing: a record carrying a DEGRADED image under one key and
the CLEAN target under another, both ``[C, H, W]`` float tensors in ``[0, 1]``. The fixtures build
that pair the same way the example config does (copy, then degrade the copy), so a test's data has
the property the task depends on — the two halves differ ONLY by the noise.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch

#: The noise level the fixtures use, in ``[0, 1]`` units — the same 0.1 the example config wires.
SIGMA = 0.1


@pytest.fixture
def clean_batch() -> torch.Tensor:
    """A ``[4, 3, 32, 32]`` batch of smooth, structured images in ``[0, 1]``.

    Structured rather than uniform noise on purpose: SSIM measures local covariance, so on random
    pixels it reports near-zero and a test asserting "the metric moved" would measure nothing.
    """
    ramp = torch.linspace(0.0, 1.0, 32)
    image = (ramp[None, :] * 0.6 + ramp[:, None] * 0.4).expand(3, 32, 32)
    return torch.stack([image * (0.5 + 0.15 * i) for i in range(4)]).clamp(0.0, 1.0)


@pytest.fixture
def noisy_batch(clean_batch: torch.Tensor) -> torch.Tensor:
    """``clean_batch`` plus fixed-seed Gaussian noise at :data:`SIGMA`, clipped to ``[0, 1]``."""
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(clean_batch.shape, generator=generator) * SIGMA
    return (clean_batch + noise).clamp(0.0, 1.0)


@pytest.fixture
def pair_records(clean_batch: torch.Tensor, noisy_batch: torch.Tensor) -> List[Dict[str, Any]]:
    """The batch as a list of RECORDS — what a ``Stream`` yields and a collate consumes.

    Keys are the runnables' defaults (``image`` = the degraded input, ``target`` = the clean
    ground truth) plus one metadata column, so the predictions path has something to pair with.
    """
    return [{"image": noisy_batch[i], "target": clean_batch[i], "row_id": i} for i in range(clean_batch.shape[0])]
