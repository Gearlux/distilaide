"""The IQ denoiser — a 1D U-Net with a recurrent bottleneck, and the loss it trains with.

A waveform is not an image, but the restoration task is the same: a noisy ``[N, 2, L]`` recording (the I and Q rows
as two channels, ``L`` samples) in, the clean one out. The batch is the restoration batch the trainers already read
(``input_key`` the noisy tensor, ``target_key`` the clean one), so ``TorchRestorer`` trains it unchanged.

:class:`IqUNet` follows the time-domain U-Net denoisers of the RF literature (Demucs-style: strided convolutions down,
a bidirectional LSTM at the bottleneck, transposed convolutions up, skip connections across), and predicts the clean
recording directly. :class:`IqDenoisingLoss` is a sample-domain L1 plus a multi-resolution STFT-magnitude loss, so a
signal standing under the noise is not averaged away by the sample loss alone.

Both are ``torch`` modules and stay out of the backend-free core: ``distilaide`` resolves their names lazily.
"""

from typing import Any, List, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from annotated_types import Interval
from confluid import configurable
from loggair import get_logger
from typing_extensions import Annotated

logger = get_logger(__name__)

__all__ = ["IqDenoisingLoss", "IqUNet"]


class _Block(nn.Module):
    """Convolution -> GroupNorm -> GELU, the unit both halves of the U-Net are made of."""

    def __init__(self, inputs: int, outputs: int, kernel: int, stride: int = 1) -> None:
        super().__init__()
        groups = 1 if outputs < 8 else 4  # GroupNorm: no batch statistics, so a batch of one trains and evaluates alike
        self.conv = nn.Conv1d(inputs, outputs, kernel, stride=stride, padding=kernel // 2)
        self.norm = nn.GroupNorm(groups, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class _Network(nn.Module):
    """The built U-Net: ``stages`` strided blocks down, a bidirectional LSTM, ``stages`` upsampling blocks up."""

    def __init__(self, channels: int, base: int, stages: int, kernel: int, lstm_layers: int) -> None:
        super().__init__()
        widths = [base * 2**i for i in range(stages)]
        self.stages = stages
        self.stem = _Block(channels, base, kernel)
        self.down = nn.ModuleList()
        inputs = base
        for width in widths:
            self.down.append(_Block(inputs, width, kernel, stride=2))
            inputs = width
        self.lstm = nn.LSTM(inputs, inputs, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.lstm_project = nn.Conv1d(2 * inputs, inputs, 1)
        self.up = nn.ModuleList()
        self.merge = nn.ModuleList()
        skips = [base, *widths[:-1]]  # the tensor each stage's input was, before it halved the length
        for width, skip in zip(reversed(widths), reversed(skips)):
            self.up.append(nn.ConvTranspose1d(width, skip, 2, stride=2))
            self.merge.append(_Block(2 * skip, skip, kernel))
        self.head = nn.Conv1d(base, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x)
        kept: List[torch.Tensor] = []
        for block in self.down:
            kept.append(hidden)
            hidden = block(hidden)
        recurrent, _ = self.lstm(hidden.transpose(1, 2))
        hidden = hidden + self.lstm_project(recurrent.transpose(1, 2))
        for up, merge in zip(self.up, self.merge):
            hidden = merge(torch.cat([up(hidden), kept.pop()], dim=1))
        return cast(torch.Tensor, self.head(hidden))


@configurable(task="restoration", role="model", lazy=True, framework="torch")
class IqUNet(nn.Module):
    """A 1D U-Net that maps a noisy IQ recording to a clean one.

    Args:
        channels: Rows of the recording — 2 for I and Q.
        base_channels: Width of the first block; each stage down doubles it.
        stages: Strided blocks down (and up): the recording is cut to ``2**stages`` times fewer samples at the
            bottleneck.
        kernel_size: Odd convolution kernel, in samples.
        lstm_layers: Layers of the bidirectional LSTM at the bottleneck (0 leaves it out).
        residual: Predict the noise and subtract it from the input (the network starts near the identity, the do-nothing
            baseline) instead of predicting the clean recording directly.
        seed: Optional weight-init seed, for a reproducible comparison.

    Attributes:
        net: The built network (``None`` until :meth:`solidify`), a public submodule name so a Lightning checkpoint's
            ``model.`` prefix strips as it does for every model here.
    """

    def __init__(
        self,
        channels: Annotated[int, Interval(ge=1, le=64)] = 2,
        base_channels: Annotated[int, Interval(ge=4, le=512)] = 16,
        stages: Annotated[int, Interval(ge=1, le=10)] = 4,
        kernel_size: Annotated[int, Interval(ge=1, le=63)] = 9,
        lstm_layers: Annotated[int, Interval(ge=0, le=4)] = 1,
        residual: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        # Store config ONLY — the workspace lazy-init mandate; the network is built on first use.
        self.channels = int(channels)
        self.base_channels = int(base_channels)
        self.stages = int(stages)
        self.kernel_size = int(kernel_size)
        self.lstm_layers = int(lstm_layers)
        self.residual = bool(residual)
        self.seed = int(seed)
        self.net: Any = None

    def solidify(self) -> nn.Module:
        """Build the network on first call (cached in ``self.net``) and return it."""
        if self.net is None:
            if self.kernel_size % 2 == 0:
                raise ValueError(f"IqUNet: kernel_size must be odd, got {self.kernel_size}")
            if self.stages < 1:
                raise ValueError(f"IqUNet: stages must be at least 1, got {self.stages}")
            torch.manual_seed(self.seed)
            self.net = _Network(self.channels, self.base_channels, self.stages, self.kernel_size, self.lstm_layers)
            logger.debug(f"Built IqUNet (base={self.base_channels}, stages={self.stages}).")
        return cast(nn.Module, self.net)

    @property
    def backbone(self) -> nn.Module:
        """The built network, built and cached on first access."""
        return self.solidify()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Denoise a batch.

        Args:
            images: ``[N, channels, L]`` float tensor, any ``L``: it is zero-padded to a multiple of ``2**stages`` and
                the answer cut back.

        Returns:
            ``[N, channels, L]`` — the recording as the network believes it was sent, in the input's scale (with
            ``residual``, the input less the noise the network predicts).
        """
        if images.dim() != 3 or images.shape[1] != self.channels:
            raise ValueError(f"IqUNet expects [N, {self.channels}, L] recordings, got shape {tuple(images.shape)}.")
        length = images.shape[-1]
        multiple = 2**self.stages
        padded = F.pad(images, (0, (-length) % multiple))
        out = self.backbone(padded)[..., :length]
        return cast(torch.Tensor, images - out if self.residual else out)


@configurable(task="restoration", role="loss", framework="torch")
class IqDenoisingLoss(nn.Module):
    """Sample-domain L1 plus a multi-resolution STFT-magnitude loss over complex IQ.

    The STFT term compares the magnitude of each recording's short-time spectrum at several window lengths — their
    log-magnitude L1 and a spectral convergence — so a weak signal the sample loss would average away still counts.

    Args:
        fft_sizes: The STFT window lengths, samples (hop a quarter of each).
        time_weight: The sample-domain L1's weight.
        stft_weight: The STFT-magnitude term's weight.
        eps: Added under the log and the norm so silence is finite — and, at 1e-3 against unit-rms input, so the
            spectrum of a silent stretch is held to -60 dB rather than made to matter as much as a signal.
    """

    def __init__(
        self,
        fft_sizes: Sequence[int] = (64, 256, 1024),
        time_weight: Annotated[float, Interval(ge=0.0, le=1e6)] = 1.0,
        stft_weight: Annotated[float, Interval(ge=0.0, le=1e6)] = 1.0,
        eps: Annotated[float, Interval(gt=0.0, le=1.0)] = 1e-3,
    ) -> None:
        super().__init__()
        self.fft_sizes: Tuple[int, ...] = tuple(int(n) for n in fft_sizes)
        self.time_weight = float(time_weight)
        self.stft_weight = float(stft_weight)
        self.eps = float(eps)

    def _magnitude(self, x: torch.Tensor, size: int) -> torch.Tensor:
        iq = torch.complex(x[:, 0].float(), x[:, 1].float())
        window = torch.hann_window(size, device=x.device)
        return torch.stft(
            iq, size, hop_length=max(size // 4, 1), window=window, return_complex=True, onesided=False
        ).abs()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if preds.shape != targets.shape or preds.dim() != 3 or preds.shape[1] != 2:
            raise ValueError(
                f"IqDenoisingLoss expects two [N, 2, L] recordings, got {tuple(preds.shape)} and {tuple(targets.shape)}"
            )
        loss = self.time_weight * F.l1_loss(preds, targets)
        if self.stft_weight and self.fft_sizes:
            terms = []
            for size in self.fft_sizes:
                if preds.shape[-1] < size:
                    continue  # a recording shorter than the window has no frame of it
                p, t = self._magnitude(preds, size), self._magnitude(targets, size)
                # bounded in [0, 1] and zero for equal spectra, even where the target is silent (dividing by the
                # target's norm alone made a silent clean window cost 1/eps)
                convergence = torch.linalg.norm(t - p) / (torch.linalg.norm(t) + torch.linalg.norm(p) + self.eps)
                log_magnitude = F.l1_loss(torch.log(p + self.eps), torch.log(t + self.eps))
                terms.append(convergence + log_magnitude)
            if terms:
                loss = loss + self.stft_weight * torch.stack(terms).mean()
        return loss
