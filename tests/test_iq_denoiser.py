"""The IQ denoiser: a 1D U-Net over [N, 2, L] recordings and the loss it trains with."""

import pytest
import torch

from distilaide.iq_denoiser import IqDenoisingLoss, IqUNet


def _pair(n: int = 4, length: int = 1000, snr_db: float = 10.0, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    t = torch.arange(length) / length
    phase = 2 * torch.pi * (20 + 30 * torch.rand(n, 1, generator=gen)) * t
    clean = torch.stack([torch.cos(phase), torch.sin(phase)], dim=1)  # [n, 2, length]
    noise = torch.randn(clean.shape, generator=gen) * 10 ** (-snr_db / 20) / 2**0.5
    return clean + noise, clean


def test_the_constructor_builds_nothing_and_the_zero_arg_class_runs_any_length() -> None:
    model = IqUNet()
    assert model.net is None and not list(model.parameters())
    out = model(torch.randn(2, 2, 1003))  # not a multiple of 2**stages: padded inside, cut back
    assert out.shape == (2, 2, 1003) and torch.isfinite(out).all()
    assert sum(p.numel() for p in model.parameters()) > 0


def test_a_recording_of_another_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="expects"):
        IqUNet()(torch.randn(2, 3, 256))


def test_the_same_seed_builds_the_same_weights() -> None:
    a, b = IqUNet(seed=3), IqUNet(seed=3)
    x = torch.randn(1, 2, 256)
    assert torch.equal(a(x), b(x))


def test_the_loss_is_zero_on_a_recording_against_itself_and_grows_with_the_error() -> None:
    noisy, clean = _pair()
    loss = IqDenoisingLoss()
    assert float(loss(clean, clean)) == pytest.approx(0.0, abs=1e-5)
    assert float(loss(noisy, clean)) > float(loss(clean + 0.01 * (noisy - clean), clean)) > 0.0


def test_the_stft_term_counts_even_when_the_sample_term_is_off() -> None:
    noisy, clean = _pair()
    assert float(IqDenoisingLoss(time_weight=0.0)(noisy, clean)) > 0.0
    assert float(IqDenoisingLoss(time_weight=0.0, stft_weight=0.0)(noisy, clean)) == 0.0


def test_a_few_steps_of_training_lower_the_loss() -> None:
    noisy, clean = _pair(n=8, length=512)
    model = IqUNet(base_channels=8, stages=3, seed=1)
    loss_fn = IqDenoisingLoss(fft_sizes=(64, 256))
    model.solidify()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = float(loss_fn(model(noisy), clean).detach())
    for _ in range(40):
        optimizer.zero_grad()
        loss = loss_fn(model(noisy), clean)
        loss.backward()
        optimizer.step()
    assert float(loss_fn(model(noisy), clean).detach()) < 0.7 * first


def test_a_residual_network_starts_near_the_identity_and_the_direct_one_does_not() -> None:
    noisy, _ = _pair(length=512)
    near = IqUNet(base_channels=8, stages=3, residual=True, seed=2)(noisy)
    direct = IqUNet(base_channels=8, stages=3, residual=False, seed=2)(noisy)
    assert float((near - noisy).abs().mean()) < float((direct - noisy).abs().mean())


def test_the_stft_term_treats_silence_gently_at_the_default_eps() -> None:
    silent = torch.zeros(1, 2, 1024)
    tone = torch.stack([torch.cos(torch.arange(1024) / 5.0), torch.sin(torch.arange(1024) / 5.0)])[None]
    assert float(IqDenoisingLoss(time_weight=0.0)(tone, silent)) < 20.0  # bounded: a tone where silence was sent
