"""Distilaide CLI — image restoration across training backends.

Three verbs, and the training backend is a *config* dimension the CLI passes through::

    distilaide train    cfg.yaml                       # the config's default backend
    distilaide train    cfg.yaml --framework lightning # a `!scope:framework=lightning` block
    distilaide evaluate cfg.yaml
    distilaide predict  cfg.yaml
    distilaide frameworks                              # what the backends are

What stays constant across backends is the point: the same ``runnable:`` + ``task:`` config, the
same three verbs, the same tracking. Only the runnable class named in the YAML changes. (A generic
``matrainer train cfg.yaml`` or ``recordstream run cfg.yaml`` also works — the first is the same
machinery without this project's table.)

``evaluate`` carries more weight here than in a classification or segmentation project, and that is
a property of this TASK rather than of this CLI: the model zoo holds fixed ALGORITHMS (BM3D, total
variation, wavelet shrinkage, a median filter) beside learned networks, so ``distilaide evaluate``
over a classical ``arch:`` is a complete, meaningful run that trains nothing — and it produces the
same PSNR/SSIM numbers, on the same data, into the same tracker, as a trained model does.

The MACHINERY is :mod:`matrainer.cli` (2026-08-07, the P4 lift) — the verb dispatch through
``materialize_runnable``, the ``--framework`` → ``--scope`` rewrite, the keras engine pin, the
registration import and their load-bearing ORDER all live on the factory's own docstrings. What
this module owns is distilaide's DATA: the FRAMEWORKS table and the verb help texts.

Adding a backend: write ``distilaide/<name>_restorer.py``, then add a row to ``FRAMEWORKS`` below.
"""

from typing import Dict

from matrainer.cli import Framework, make_trainer_app, trainer_main

FRAMEWORKS: Dict[str, Framework] = {
    "lightning": Framework(
        "PyTorch Lightning — the reference backend (deepinv models, per-split torchmetrics).",
        "distilaide.torch_restoration",
    ),
    # Shares `lightning`'s loss/metric set, and that is the point rather than an economy: a
    # `torch.nn` loss does not care whether a Trainer or a hand-written loop calls it, so the two
    # backends differ ONLY in the loop.
    "torch": Framework(
        "PyTorch — a hand-written loop, no training framework at all (the control experiment).",
        "distilaide.torch_restoration",
    ),
    "fastai": Framework(
        "fastai — a Learner with one-cycle scheduling, over the same torch stack.",
        # The same torch losses reach it (a `BaseLoss` is a callable over torch tensors), and its
        # METRICS are names in `fastai.metrics` rather than registered classes — so there is
        # nothing extra to import for it.
        "distilaide.torch_restoration",
    ),
    "keras": Framework(
        "Keras 3 — one API over PyTorch / TensorFlow / JAX (set KERAS_BACKEND to pick).",
        "distilaide.keras_restoration",
    ),
    # Shares the Keras loss/metric set on purpose: in TF >= 2.16 `tf.keras` IS Keras 3, so a
    # separate `tensorflow_restoration.py` would register the same classes under a second tag and
    # make every bare `!class:MeanAbsoluteError` ambiguous for no gain.
    "tensorflow": Framework(
        "TensorFlow — a native tf.data + tf.GradientTape loop, with SavedModel export.",
        "distilaide.keras_restoration",
    ),
    "jax": Framework(
        "JAX — a jitted optax loop over a Flax model, the loop written out in full.",
        # No registrations: optax ships plain FUNCTIONS, so a loss is resolved by NAME and there
        # is no class for the registry to tag.
    ),
    "mlx": Framework(
        "MLX — Apple silicon, unified memory, lazy arrays (`mlx.nn` losses by NAME).",
        # `mlx.nn.losses` are functions like optax's; `mlx.optimizers` ships classes, but a config
        # names those too. Nothing to register.
    ),
    "huggingface": Framework(
        "Transformers `Trainer` + Accelerate — Swin2SR, with the loss supplied from outside.",
        # It DOES get registrations, unlike its siblings in the other tasks: Swin2SR raises
        # `NotImplementedError` when handed `labels=`, so the loss comes from outside — and a
        # `torch.nn` loss does not care which loop calls it.
        "distilaide.torch_restoration",
    ),
}

app = make_trainer_app(
    name="distilaide",
    description="Image restoration across training backends",
    frameworks=FRAMEWORKS,
    verb_docs={
        "train": "Train an image-restoration model",
        "evaluate": "Score a restoration model (learned or classical) over the test set",
        "predict": "Stream restored images from a restoration model to its sink",
    },
)


def main() -> None:
    trainer_main(app, FRAMEWORKS)


if __name__ == "__main__":
    main()
