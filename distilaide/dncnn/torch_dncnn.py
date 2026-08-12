"""``TorchDnCNN`` — the shared DnCNN spec in PyTorch.

Serves ALL THREE torch backends (lightning, torch, fastai): they differ in the LOOP, not in what a
model is, which is why there is no ``fastai_dncnn.py`` — exactly as there is no ``fastai_model.py``.

It IS a ``torch.nn.Module``, the same decision as :class:`~distilaide.deepinv_model.DeepInvModel`
being one, so no loop reaches THROUGH a wrapper to find the network.
"""

from typing import Any, List, Tuple, cast

import torch
import torch.nn as nn
from confluid import configurable
from loggair import get_logger

from distilaide.dncnn.spec import DEPTH, INPUT_SHAPE, KERNEL_SIZE, WIDTH, trainable_parameter_count

logger = get_logger(__name__)

__all__ = ["TorchDnCNN"]


@configurable(task="restoration", role="model", lazy=True, framework="torch")
class TorchDnCNN(nn.Module):
    """The shared DnCNN spec, in ``torch.nn``.

    Args:
        input_shape: ``(H, W, C)`` — channels LAST, like every other implementation of this spec,
            even though this one's TENSOR is NCHW. Only ``C`` reaches the network; the raster is
            declared so one config line drives any implementation. DnCNN is fully convolutional, so
            ``H`` and ``W`` are advisory.
        depth: Convolution layers end to end (head + body + tail).
        width: Feature channels in every hidden layer.
        kernel_size: Square conv kernel; ``padding`` keeps the spatial size.
        seed: Optional weight-init seed, for a reproducible comparison.

    Attributes:
        net: Partial cache for the built network (``None`` until :meth:`solidify`). A PUBLIC submodule
            name, so a state_dict saved through Lightning — which prefixes ``model.``, stripped by
            ``load_lightning_checkpoint`` — round-trips as ``net.*`` keys.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        depth: int = DEPTH,
        width: int = WIDTH,
        kernel_size: int = KERNEL_SIZE,
        seed: int = 0,
    ) -> None:
        super().__init__()
        # Store config ONLY — the workspace lazy-init / zero-arg mandate.
        self.input_shape: Tuple[int, int, int] = tuple(input_shape)  # type: ignore[assignment]
        self.depth = int(depth)
        self.width = int(width)
        self.kernel_size = int(kernel_size)
        self.seed = int(seed)
        self.net: Any = None

    @property
    def in_channels(self) -> int:
        """The channel count, read from the LAST element of the ``(H, W, C)`` knob.

        Reading it as NCHW instead would build ``Conv2d(192, …)`` rather than ``Conv2d(3, …)`` —
        still runnable, and silently a different network.
        """
        return int(self.input_shape[2])

    def solidify(self) -> nn.Module:
        """Build the network on first call (cached in ``self.net``) and return it."""
        if self.net is None:
            torch.manual_seed(self.seed)
            channels, k, pad = self.in_channels, self.kernel_size, self.kernel_size // 2
            layers: List[nn.Module] = [
                # Head: conv -> ReLU, deliberately NOT normalized (the paper's shape).
                nn.Conv2d(channels, self.width, k, padding=pad),
                nn.ReLU(inplace=True),
            ]
            for _ in range(self.depth - 2):
                # Body: conv -> BN -> ReLU. The bias is kept even though BN makes it redundant —
                # see the spec: dropping it is a per-framework "sensible default" that would break
                # the five-way parameter match.
                layers += [
                    nn.Conv2d(self.width, self.width, k, padding=pad),
                    nn.BatchNorm2d(self.width),
                    nn.ReLU(inplace=True),
                ]
            # Tail: back to the input's channel count. It predicts the NOISE.
            layers.append(nn.Conv2d(self.width, channels, k, padding=pad))
            self.net = nn.Sequential(*layers)
            logger.debug(f"Built TorchDnCNN (depth={self.depth}, width={self.width}).")
        return cast(nn.Module, self.net)

    @property
    def backbone(self) -> nn.Module:
        """The built network, built + cached on first access (confluid lazy-init rule 3)."""
        return self.solidify()

    @property
    def trainable_parameter_count(self) -> int:
        """What this implementation reports — checked against the spec's DERIVATION.

        torch holds batch norm's running statistics as BUFFERS, so filtering on ``requires_grad``
        excludes them. Each framework needs a different mechanism for the same exclusion.
        """
        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Restore a batch: predict the noise, subtract it.

        Args:
            images: ``[N, C, H, W]`` float tensor.

        Returns:
            ``[N, C, H, W]`` — a restored IMAGE in the input's value range.
        """
        if images.dim() != 4:
            raise ValueError(f"TorchDnCNN expects 4D [N, C, H, W] images, got shape {tuple(images.shape)}.")
        # THE residual connection — the architecture's whole idea, and why predicting ZERO noise
        # (the do-nothing baseline) is this network's cheapest local optimum. It is a TRAINING
        # property, not an init one: at initialization this deviates from the identity by mean 0.35
        # / max 1.86 on `[0, 1]` images (measured), because the tail convolution is initialized the
        # usual way rather than at zero.
        return cast(torch.Tensor, images - self.backbone(images))

    def spec_parameter_count(self) -> int:
        """The count the SPEC derives for this configuration (what :attr:`trainable_parameter_count` must equal)."""
        return trainable_parameter_count(self.in_channels, self.depth, self.width, self.kernel_size)
