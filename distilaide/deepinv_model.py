"""deepinv-backed image-restoration wrapper — every denoiser in the library, one class.

Wraps a :mod:`deepinv.models` architecture into a plain :class:`torch.nn.Module` whose
``forward(images) -> restored`` signature plugs straight into
:class:`distilaide.lightning_restorer.LightningRestorer`, which calls ``preds = self(x)`` and
``loss = self.loss_fn(preds, y)`` in its ``training_step`` — for an image-to-image task that is all
the glue needed. No bundled batch type, no step-output dict, no per-mode forward divergence.

**One wrapper, the whole zoo — including the classical baselines.** ``arch`` is a closed
:data:`RestorationArchitecture` Literal, so a picker, a form-spec and a StreamStudio widget
enumerate the choices instead of offering free text, and confluid's pydantic pass rejects a typo AT
CONSTRUCTION listing the valid set. It spans two kinds of model that this project deliberately puts
in ONE slot:

* the **learned** nets — ``dncnn`` / ``drunet`` / ``scunet`` / ``swinir`` / ``restormer`` /
  ``ffdnet`` / ``unet`` / ``srresnet``, which train;
* the **classical** ones — ``tv`` / ``tgv`` / ``wavelet`` / ``median`` / ``bilateral`` / ``bm3d``,
  which have **zero parameters** and only ever evaluate.

Putting both behind one ``model:`` key is the point rather than a convenience: it means
``distilaide evaluate`` scores BM3D and a trained DnCNN through the same PSNR/SSIM path, on the same
data, into the same tracker — which is the comparison anyone choosing a denoiser actually wants, and
the thing this project's predecessor (a ``denoise(signal, method=...)`` dispatcher) could not do
because its classical and learned methods were different code paths with no metrics at either end.
A ``task: fit`` run over a parameter-free architecture is refused by name — see :meth:`is_trainable`.

Typical wiring (YAML) — declare the model ``!lazy:`` so the restorer injects what only the run
knows at flow time::

    model: !lazy:distilaide.DeepInvModel
      arch: dncnn
      in_channels: 3
"""

import inspect
from typing import Any, Dict, List, Literal, Optional, cast, get_args
from urllib.parse import quote, urlencode

import torch
import torch.nn as nn
from confluid import configurable
from loggair import get_logger
from matrainer.torch import load_lightning_checkpoint
from matrainer.uri import ModelRef, file_model_ref
from torch import Tensor

logger = get_logger(__name__)

#: Scheme naming the deepinv zoo in a model reference. deepinv resolves an architecture's published
#: weights through THREE different places depending on the architecture — a Hugging Face repo for
#: dncnn / drunet / scunet / srresnet / restormer, a GitHub release asset for swinir, and nothing at
#: all for ffdnet (which raises *"has no downloadable weights"* when asked) — so the honest
#: identifier is the architecture + weight-set deepinv itself was asked for, with no browsable link
#: rather than a guessed one. See :attr:`DeepInvModel.model_refs`.
DEEPINV_URI_PREFIX = "deepinv://"

#: The restoration architectures this wrapper offers, as a CLOSED set — so `to_pydantic` gives
#: navigaitor's form-spec and StreamStudio an enumerable widget rather than a free-text box, and a
#: typo fails at the call site instead of inside deepinv.
#:
#: Written out rather than derived from `dir(deepinv.models)`, because a `Literal` cannot be built
#: at runtime and still be a type — and because `deepinv.models` publishes ~60 names, most of them
#: (discriminators, generators, diffusion samplers, reconstruction wrappers) not image-to-image
#: denoisers at all. `tests/test_deepinv_model.py` asserts every member resolves to a real
#: `deepinv.models` class, so a deepinv release that renames one fails here loudly.
RestorationArchitecture = Literal[
    # --- learned: these train ---
    "dncnn",
    "drunet",
    "scunet",
    "swinir",
    "restormer",
    "ffdnet",
    "unet",
    "srresnet",
    # --- classical: zero parameters, evaluation baselines ---
    "tv",
    "tgv",
    "wavelet",
    "median",
    "bilateral",
    "bm3d",
]

#: The runtime tuple, DERIVED from the Literal — never a second hand-written list.
RESTORATION_ARCHITECTURES = get_args(RestorationArchitecture)

#: ``arch`` -> the :mod:`deepinv.models` class name. A table is unavoidable here (unlike smp's
#: ``create_model``, deepinv ships no factory), but it maps a name to a name and nothing else — the
#: constructor knobs are read from the class by INSPECTION below rather than restated per entry.
_ARCH_CLASS_NAMES: Dict[str, str] = {
    "dncnn": "DnCNN",
    "drunet": "DRUNet",
    "scunet": "SCUNet",
    "swinir": "SwinIR",
    "restormer": "Restormer",
    "ffdnet": "FFDNet",
    "unet": "UNet",
    "srresnet": "SRResNet",
    "tv": "TVDenoiser",
    "tgv": "TGVDenoiser",
    "wavelet": "WaveletDenoiser",
    "median": "MedianFilter",
    "bilateral": "BilateralFilter",
    "bm3d": "BM3D",
}

#: The names deepinv's architectures use for "how many input channels", in resolution order.
#: **The zoo does not agree on one** — measured against deepinv 0.4.1: ``in_channels`` (DnCNN,
#: DRUNet, Restormer, UNet), ``in_nc`` (SCUNet), ``in_chans`` (SwinIR), ``img_channels`` (FFDNet),
#: ``channels`` (EPLL) — and the classical denoisers take none at all, being channel-agnostic by
#: construction. So this wrapper declares ONE knob and translates it by inspecting the target
#: class's signature, rather than carrying a per-architecture table that a deepinv release could
#: silently outdate.
_CHANNEL_PARAM_NAMES = ("in_channels", "in_nc", "in_chans", "img_channels", "channels")


@configurable(task="restoration", role="model", lazy=True, framework="torch")
class DeepInvModel(nn.Module):
    """A ``deepinv.models`` restoration model wrapped for the distilaide restorer flow.

    ``forward(images) -> restored`` only: an image-to-image model's output IS the prediction, so
    the restorer's standard ``loss_fn(preds, target)`` call works with no batch wrapper.

    **Lazy construction (Lazy-Init / Zero-Arg mandate).** ``__init__`` does NO functional work — it
    only stores config. The deepinv model (and any pretrained-weight download) is built on first use
    by :meth:`solidify` — called by :meth:`forward`, by confluid's auto-build on ``flow()``, or
    explicitly — and cached in ``self.restorer``, so ``DeepInvModel()`` is cheap and does no network
    I/O. Tagged ``lazy=True`` so a config wires it ``!lazy:`` and the restorer flows it at run time.

    Args:
        arch: Architecture — one of :data:`RESTORATION_ARCHITECTURES`. Defaults to ``dncnn``, the
            canonical learned denoising baseline.
        in_channels: Input channel count (3 = RGB, 1 = greyscale). Translated to whichever name the
            chosen architecture uses; architectures with no channel knob ignore it.
        out_channels: Output channel count; ``0`` (the default) means "same as ``in_channels``",
            which is what an image-to-image task always wants. Only passed to architectures that
            declare it.
        pretrained: deepinv's published weights to load — ``"download"`` fetches them, ``None`` (the
            **default here**) trains from scratch. This deliberately DIVERGES from deepinv's own
            default of ``"download"``: distilaide is a training project, so a bare
            ``arch: dncnn`` must not silently start from somebody else's weights and must not need
            the network to build. Only passed to architectures that declare it — and note
            ``ffdnet`` raises *"has no downloadable weights"* if asked, which is deepinv's message,
            not something this wrapper hides.
        sigma: Noise level in the images' own value range (``0.1`` on a ``[0, 1]`` pipeline), used
            as the second forward argument. It is a MODEL knob rather than a restorer one because
            only the architecture decides whether it matters, and the zoo splits three ways:
            ``drunet`` / ``bm3d`` REQUIRE it (a noise-level map is part of the input); ``tv`` /
            ``tgv`` require it too, reading the same slot as their regularization threshold and
            raising *"Regularization parameter (ths) was not passed at init nor at forward"*
            without one (measured); ``dncnn`` / ``unet`` are BLIND denoisers that ignore it.
            ``None`` (the default) lets each architecture's own default apply, which is why a
            blind denoiser needs no such line in its config and a conditioned one fails loudly
            rather than being handed a number this wrapper invented.
        seed: Seed for torch's global RNG, set when the network is BUILT — the same knob the five
            shared-spec implementations carry, so ``seed`` means one thing across all seven model
            wrappers here. Measured 2026-08-06 with the backend sweep, and the numbers are worth
            keeping because the obvious reading of the first pair is wrong: unseeded, two runs of
            the identical config gave 22.7530 / 22.5188 dB, and the shared spec gave BIT-IDENTICAL
            scores (20.7427 twice) — which looks like "the init is the whole variance" and is not.
            The spec implementation reproduces because it seeds the global RNG and does NOT restore
            it, so the loader's shuffle order is seeded too; a save-and-restore version of this
            knob was written, and left two runs 22.1891 / 21.9956 apart.
        checkpoint_path: Optional Lightning ``.ckpt`` / state-dict to load after the model is built;
            ``""`` = none.
        kwargs: Any further deepinv constructor knob, forwarded VERBATIM (``depth``, ``nf``,
            ``n_it_max``, ``kernel_size``, …).

    Attributes:
        checkpoint_strict: Whether ``load_state_dict`` enforces exact key matching. Defaults to
            ``False`` so a differently-configured architecture does not reject the whole load.
        restorer: Lazy cache for the built deepinv model (``None`` until :meth:`solidify`). A PUBLIC
            submodule name, so a state_dict saved through Lightning — which prefixes ``model.``,
            stripped by ``load_lightning_checkpoint`` — round-trips as ``restorer.*`` keys that
            match this wrapper's layout. Naming it ``self.model`` would make the strip chew through
            this wrapper's own attribute name too, producing keys that match nothing.
    """

    def __init__(
        self,
        arch: RestorationArchitecture = "dncnn",
        in_channels: int = 3,
        out_channels: int = 0,
        pretrained: Optional[str] = None,
        sigma: Optional[float] = None,
        checkpoint_path: str = "",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Store config ONLY — no model construction, no weight download, no checkpoint load. The
        # model is built lazily on first use (via `solidify()`, the cached `backbone` property, or
        # `forward()` — all three share the one cache), so this constructor is cheap and zero-arg
        # construction works (Lazy-Init / Zero-Arg mandate).
        self.arch: RestorationArchitecture = arch
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.pretrained = pretrained
        self.sigma = float(sigma) if sigma is not None else None
        self.checkpoint_path = checkpoint_path
        self.seed = int(seed)
        self.checkpoint_strict: bool = False
        # Forwarded verbatim to the architecture at build time (the pass-through mandate: this
        # wrapper must not shadow deepinv's names or silently rewrite its defaults).
        self.model_kwargs = dict(kwargs)
        self.restorer: Optional[nn.Module] = None

    # ---- identity ------------------------------------------------------------ #
    @property
    def model_refs(self) -> List[ModelRef]:
        """The weight artifacts this model loads — see :mod:`matrainer.uri`.

        deepinv routes an architecture's published weights through whichever place its author put
        them, and which one that is depends on the architecture (measured against 0.4.1: a Hugging
        Face repo under ``deepinv/`` for dncnn / drunet / scunet / srresnet / restormer, a GitHub
        release asset for swinir, and nothing at all for ffdnet) — so a single repository URL would
        be wrong for most of the zoo. What IS unambiguous is the pair deepinv was asked for, so that
        is the reference, with no browsable link rather than a guessed one.

        ``pretrained=None`` (this wrapper's default) builds from scratch and loads nothing, so the
        list is empty: random weights referenced no artifact, and inventing a reference would name
        something nobody loaded.
        """
        refs: List[ModelRef] = []
        if self.pretrained:
            selection = urlencode({"weights": str(self.pretrained)})
            refs.append(ModelRef(uri=f"{DEEPINV_URI_PREFIX}{quote(self.arch)}?{selection}"))
        if self.checkpoint_path:
            refs.append(file_model_ref(self.checkpoint_path))
        return refs

    # ---- build --------------------------------------------------------------- #
    def _architecture_class(self) -> Any:
        """Resolve ``arch`` to its :mod:`deepinv.models` class, importing deepinv on first build."""
        if self.arch not in RESTORATION_ARCHITECTURES:
            raise ValueError(f"DeepInvModel.arch must be one of {list(RESTORATION_ARCHITECTURES)}, got {self.arch!r}.")
        from deepinv import models as deepinv_models

        class_name = _ARCH_CLASS_NAMES[self.arch]
        cls = getattr(deepinv_models, class_name, None)
        if cls is None:  # pragma: no cover - a deepinv release that renamed the class
            raise ImportError(
                f"deepinv.models has no {class_name!r} (the class {self.arch!r} maps to). "
                "The installed deepinv may be too old or too new for this wrapper."
            )
        return cls

    def _build_kwargs(self, cls: Any) -> Dict[str, Any]:
        """The constructor kwargs for ``cls`` — this wrapper's knobs translated BY INSPECTION.

        Reading the target's signature rather than consulting a per-architecture table is what
        keeps the translation honest: the zoo spells "input channels" five different ways and only
        some architectures take ``pretrained`` at all, so a table would be a second description of
        deepinv that a release could silently outdate. An explicitly passed ``kwargs`` key always
        WINS — ``setdefault``, never an override — because the pass-through mandate says a knob the
        caller wrote reaches the library verbatim.
        """
        params = inspect.signature(cls.__init__).parameters
        kwargs: Dict[str, Any] = dict(self.model_kwargs)
        channel_param = next((name for name in _CHANNEL_PARAM_NAMES if name in params), None)
        if channel_param is not None:
            kwargs.setdefault(channel_param, self.in_channels)
        else:
            logger.debug(f"{self.arch}: takes no channel argument (channel-agnostic); in_channels not passed.")
        if "out_channels" in params:
            # `0` means "same as the input", which is what an image-to-image task always wants —
            # a restoration model that changed the channel count would not be one.
            kwargs.setdefault("out_channels", self.out_channels or self.in_channels)
        if "pretrained" in params:
            kwargs.setdefault("pretrained", self.pretrained)
        elif self.pretrained:
            logger.warning(
                f"{self.arch}: takes no 'pretrained' argument, so pretrained={self.pretrained!r} was ignored."
            )
        return kwargs

    def solidify(self) -> nn.Module:
        """Build the deepinv model on first call (cached in ``self.restorer``) and return it.

        Confluid's ``flow()`` calls this after construction, and :meth:`forward` calls it too, so
        the parameters exist before an optimizer reads them. Idempotent — later calls return the
        cache, which matters because ``flow()`` fires the hook on a live object as well, so a
        normal run calls this more than once.
        """
        if self.restorer is None:
            cls = self._architecture_class()
            # Seeds torch's GLOBAL RNG and does not restore it — the same one line every
            # shared-spec implementation runs, and the leak is the point rather than an oversight.
            # A save-and-restore version was written first and MEASURED not to work: it fixes the
            # weights and leaves the loader's shuffle order unseeded, and two runs still came out
            # 22.1891 / 21.9956 dB apart. Restoring the state gives a reproducible MODEL inside a
            # run that is still not reproducible, which is the shape of answer nobody wants.
            # Because the build happens once, before any training randomness is drawn, seeding here
            # is a run-level seed applied at the earliest useful moment.
            torch.manual_seed(self.seed)
            self.restorer = cls(**self._build_kwargs(cls))
            logger.debug(f"Built deepinv {self.arch} (pretrained={self.pretrained!r}, in_channels={self.in_channels}).")
            if self.checkpoint_path:
                load_lightning_checkpoint(self, self.checkpoint_path, strict=self.checkpoint_strict)
        return cast(nn.Module, self.restorer)

    @property
    def backbone(self) -> nn.Module:
        """The built deepinv model, built + cached on first access (confluid lazy-init rule 3).

        A read-only cached ``@property`` over the same build path as :meth:`solidify`, so a
        directly constructed, **never-flowed** instance still materializes on demand. ``solidify``
        is retained as the explicit method confluid's ``flow()`` calls — ``flow`` never touches
        property getters, so the method hook is what leaves ``self.parameters()`` ready before
        ``configure_optimizers``. Both share the one cache.
        """
        return self.solidify()

    @property
    def is_trainable(self) -> bool:
        """Whether this architecture has parameters to train — ``False`` for the classical ones.

        The distinction is REAL and not a formality: ``tv`` / ``tgv`` / ``median`` / ``bilateral`` /
        ``bm3d`` / ``wavelet`` are fixed algorithms wearing an ``nn.Module`` interface, so they have
        exactly zero parameters (measured). A ``fit`` over one would hand the optimizer an empty
        parameter list — which torch reports as *"optimizer got an empty parameter list"*, several
        frames away from the ``arch:`` line that caused it — so the restorer asks this first and
        refuses by name, pointing at ``distilaide evaluate``, which is what a classical baseline is
        FOR.

        It BUILDS the model to answer (unlike :attr:`model_refs`, which must not): the parameter
        count is a property of the constructed network, and for a classical denoiser construction is
        free.
        """
        return any(p.requires_grad for p in self.backbone.parameters())

    def forward(self, images: Tensor, sigma: Optional[float] = None) -> Tensor:
        """Restore a batch of images.

        Args:
            images: ``[N, in_channels, H, W]`` float tensor.
            sigma: Per-call override of the configured :attr:`sigma`. Unset, the constructor's
                value applies — which is what lets every restorer call this as a plain
                ``self.model(x)`` regardless of architecture, so the ``model:`` slot still accepts
                any ``nn.Module`` a user brings.

        Returns:
            ``[N, out_channels, H, W]`` — a restored IMAGE in the input's value range, not logits.
        """
        if images.dim() != 4:
            raise ValueError(f"DeepInvModel expects 4D [N, C, H, W] images, got shape {tuple(images.shape)}.")
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"DeepInvModel was built for in_channels={self.in_channels}, "
                f"got images with {images.shape[1]} channels."
            )
        network = self.backbone
        level = self.sigma if sigma is None else sigma
        # Positional, because the second parameter's NAME differs across the zoo (`sigma` for the
        # denoisers, `ths` for TVDenoiser) while its POSITION does not — that is the contract the
        # `deepinv.models.Denoiser` base declares (`forward(x, sigma)`).
        return cast(Tensor, network(images) if level is None else network(images, level))


__all__ = [
    "DEEPINV_URI_PREFIX",
    "RESTORATION_ARCHITECTURES",
    "DeepInvModel",
    "RestorationArchitecture",
]
