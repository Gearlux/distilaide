"""Swin2SR wrapped for the distilaide restorer flow — transformers' ONLY image-to-image model.

``transformers`` 5.x maps exactly one architecture for this task
(``MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES == {"swin2sr": "Swin2SRForImageSuperResolution"}``), so
this wrapper names it directly rather than going through ``AutoModelForImageToImage`` — there is no
zoo to be generic over, and a factory that can only ever produce one class is a factory in name
only.

**Two facts decide everything about this wrapper, and both are measured:**

1. **Every published checkpoint UPSCALES.** All six ``caidas/swin2SR-*`` models on the Hub are
   ``x2`` or ``x4``; none does same-size restoration. So a denoising run — where the output must
   be the input's size — CANNOT use pretrained weights, and this wrapper's default is
   ``pretrained=None`` with ``upscale=1``.
2. **The architecture DOES do same-size restoration** when asked: ``upscale=1`` with an EMPTY
   ``upsampler`` gives ``[N, 3, H, W] -> [N, 3, H, W]`` (verified). That is SwinIR's own
   denoising/JPEG-artifact configuration, inherited here.

So the HuggingFace backend trains this from scratch, and what it contributes is the TRAINING
PROGRAM (a ``transformers.Trainer`` owning the epoch, optimizer, scheduler, mixed precision and
distribution) rather than somebody's weights. Set ``upscale: 2`` and ``pretrained:
caidas/swin2SR-classical-sr-x2-64`` to fine-tune the real checkpoint — but note the run is then a
SUPER-RESOLUTION run and its degradation must be a downsample, not noise.
"""

from typing import Any, List, Optional, cast

import torch
import torch.nn as nn
from confluid import configurable
from loggair import get_logger
from matrainer.uri import ModelRef, file_model_ref, hf_model_ref
from torch import Tensor

logger = get_logger(__name__)

__all__ = ["HuggingFaceRestorationModel"]


@configurable(task="restoration", role="model", lazy=True, framework="torch")
class HuggingFaceRestorationModel(nn.Module):
    """A ``Swin2SRForImageSuperResolution`` wrapped so a restorer can call it like any model.

    ``forward(images) -> restored`` — the wrapper unwraps transformers' ``reconstruction`` output so
    every backend's ``preds = self(x)`` line is unchanged.

    Args:
        model_name: Hub id to load when ``pretrained`` is true. Every published Swin2SR checkpoint
            UPSCALES (x2 / x4), so this is only meaningful for a super-resolution run.
        pretrained: Load ``model_name``'s weights. **Defaults to False**, because no published
            checkpoint does same-size restoration — see the module docstring.
        upscale: Output scale factor. ``1`` (the default) is same-size restoration, which is what a
            denoising run needs and what every checkpoint fails to provide.
        embed_dim / depths / num_heads / window_size: Architecture knobs, forwarded to
            ``Swin2SRConfig``. The defaults are a small configuration that trains in seconds; the
            published models are much larger.
        in_channels: Input channel count.
        seed: Seed for torch's global RNG, set when the model is BUILT from scratch — the same
            knob every other model wrapper here carries. It matters more in this backend than
            anywhere else, because no published Swin2SR checkpoint does same-size restoration, so
            a denoising run always starts from noise.
        checkpoint_path: Optional Lightning ``.ckpt`` / state-dict to load after the build.

    Attributes:
        restorer: Partial cache for the built model (``None`` until :meth:`solidify`), PUBLIC for the
            same reason :class:`~distilaide.deepinv_model.DeepInvModel`'s is — a Lightning
            state_dict must round-trip as ``restorer.*``.
    """

    def __init__(
        self,
        model_name: str = "caidas/swin2SR-classical-sr-x2-64",
        pretrained: bool = False,
        upscale: int = 1,
        embed_dim: int = 60,
        depths: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        window_size: int = 8,
        in_channels: int = 3,
        checkpoint_path: str = "",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Store config ONLY — no download, no build (the lazy-init / zero-arg mandate).
        self.model_name = model_name
        self.pretrained = bool(pretrained)
        self.upscale = int(upscale)
        self.embed_dim = int(embed_dim)
        self.depths = list(depths) if depths else [4, 4]
        self.num_heads = list(num_heads) if num_heads else [4, 4]
        self.window_size = int(window_size)
        self.in_channels = int(in_channels)
        self.checkpoint_path = checkpoint_path
        self.seed = int(seed)
        self.checkpoint_strict: bool = False
        self.model_kwargs = dict(kwargs)
        self.restorer: Optional[nn.Module] = None

    @property
    def model_refs(self) -> List[ModelRef]:
        """The weight artifacts this model loads — see :mod:`matrainer.uri`.

        Unlike the deepinv zoo this one HAS a browsable repository page, so it uses the shared
        ``hf_model_ref`` helper rather than minting a scheme of its own — the same spelling every
        other Hub-resolving wrapper in the workspace produces for the same weights.

        Reports an EMPTY list at the default (``pretrained=False``), which is the normal state
        here: no published checkpoint does same-size restoration.
        """
        refs: List[ModelRef] = []
        if self.pretrained:
            refs.append(hf_model_ref(self.model_name))
        if self.checkpoint_path:
            refs.append(file_model_ref(self.checkpoint_path))
        return refs

    def solidify(self) -> nn.Module:
        """Build the Swin2SR model on first call (cached in ``self.restorer``) and return it."""
        if self.restorer is None:
            from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

            if self.pretrained:
                self.restorer = Swin2SRForImageSuperResolution.from_pretrained(self.model_name)
                logger.info(f"Loaded {self.model_name} (upscale is the CHECKPOINT's, not this wrapper's).")
            else:
                config = Swin2SRConfig(
                    embed_dim=self.embed_dim,
                    depths=self.depths,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    num_channels=self.in_channels,
                    upscale=self.upscale,
                    # An EMPTY upsampler is what makes `upscale=1` mean same-size restoration —
                    # SwinIR's own denoising configuration, which Swin2SR inherits.
                    upsampler="" if self.upscale == 1 else "pixelshuffle",
                    **self.model_kwargs,
                )
                # Seeds torch's GLOBAL RNG without restoring it, exactly as `DeepInvModel` and
                # the five shared-spec implementations do — measured: restoring the state fixes the
                # weights and leaves the run unseeded, which reproduces nothing. It matters most
                # HERE, because every published Swin2SR checkpoint upscales, so this backend always
                # starts from noise; a `from_pretrained` load needs no seed at all.
                torch.manual_seed(self.seed)
                self.restorer = Swin2SRForImageSuperResolution(config)
                logger.debug(f"Built Swin2SR from scratch (upscale={self.upscale}, embed_dim={self.embed_dim}).")
            if self.checkpoint_path:
                from matrainer.torch import load_lightning_checkpoint

                load_lightning_checkpoint(self, self.checkpoint_path, strict=self.checkpoint_strict)
        return cast(nn.Module, self.restorer)

    @property
    def backbone(self) -> nn.Module:
        """The built model, built + cached on first access (confluid lazy-init rule 3)."""
        return self.solidify()

    def forward(
        self,
        images: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Restore a batch, unwrapping transformers' output object.

        Args:
            images: ``[N, C, H, W]`` float tensor — the POSITIONAL spelling every other backend in
                this project uses (``preds = self(x)``).
            pixel_values: The SAME tensor under transformers' own keyword. Accepted because a
                ``Trainer``'s evaluation path calls ``model(**inputs)`` directly, bypassing
                ``compute_loss`` — so a wrapper that only took a positional argument died with
                *"got an unexpected keyword argument 'pixel_values'"* the moment the first eval ran
                (hit for real). One model, two callers, and this is the seam between them.
            labels: ACCEPTED AND IGNORED, deliberately. ``Swin2SRForImageSuperResolution`` raises
                ``NotImplementedError: Training is not supported at the moment`` when handed one,
                so the loss is computed outside — but declaring the parameter is what makes
                ``Trainer`` recognise the batch as labelled (it derives ``label_names`` from the
                forward SIGNATURE), which is what routes evaluation through ``compute_loss``
                instead of this call.

        Returns:
            ``[N, C, H*upscale, W*upscale]`` — a restored IMAGE.
        """
        del labels  # see the docstring: declared so Trainer sees a labelled batch, never used
        values = images if images is not None else pixel_values
        if values is None:
            raise ValueError("HuggingFaceRestorationModel.forward needs `images` or `pixel_values`.")
        if values.dim() != 4:
            raise ValueError(f"HuggingFaceRestorationModel expects 4D [N, C, H, W] images, got {tuple(values.shape)}.")
        return cast(Tensor, self.backbone(pixel_values=values).reconstruction)
