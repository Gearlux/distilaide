# distilaide — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

## Degradations

- [ ] Widen beyond additive Gaussian noise — blur, downsample, JPEG. Each is an existing
      albumentations transform, so the work is the CONFIG surface and the docs, not new ops. Worth
      doing once a second task (deblurring, super-resolution) is actually wanted — and note the
      huggingface backend already WANTS a downsample: Swin2SR's published checkpoints all upscale,
      so `pretrained: true` is only reachable through a super-resolution degradation. @medium
- [ ] A benchmark config against `deepinv/CBSD68` (68 images, the canonical colour denoising set)
      for a publishable number, separate from the fast smoke config. @low
- [ ] Random CROPS rather than a whole-image resize for training. The denoising literature trains on
      40-64 px patches, which is both cheaper and better augmentation; it needs the crop to be joint
      across the pair, i.e. ABOVE the `CopyField` in the chain. @low

## The comparison

- [ ] **Seed the OTHER seven backends' models the way the torch family now is.** The
      reproducibility work of 2026-08-06 was measured on `--framework torch` only, and the two
      knobs it needed (a model seed that does not restore the RNG, plus `Compose(seed:)` on the
      degradation) are in place for every chain — but only the torch path was verified end to end.
      keras / tensorflow / jax / mlx each carry a model `seed:` already and each engine seeds its
      own loop differently (`keras.utils.set_random_seed`, `mx.random.seed`, an explicit
      `nnx.Rngs`), so each needs its own two-draw check. @medium

- [ ] **Give the sweep a longer default than the smoke run.** Measured 2026-08-06: at `--epochs 1` a
      from-scratch `*DnCNN` row lands BELOW the 20.5 dB do-nothing baseline (tensorflow: 8.0 dB),
      because a residual denoiser spends its first pass converging toward the identity. The driver
      says so in its header and `--epochs 5` is what the table used, but a reader running the
      script bare gets MLflow and the example's own 5 epochs, while `--epochs 1` looks like a
      broken fleet rather than an unfinished one. @low
- [ ] **`aim` and `tensorboard` are genuinely queryable and `matrainer.sweep` does not read them
      yet** — `Repo.iter_runs()` and `EventAccumulator` respectively. Cross-project: the module is
      matrainer's, and the sibling task projects would get it for free. @low

## Predictions

- [ ] There is no image-writing predictions sink, so `distilaide predict` needs the caller to bring
      one. A generic "write each `RestorationOutput` to a PNG" sink is recordstream's to own (the
      `ClassificationPredictionsSink` precedent), not this project's. @medium

## Still open from the adapter era

- [ ] Spectrogram-aware adapter (apply an image denoiser to an STFT magnitude, preserve phase). The
      bridge belongs in waivefront, which owns the signal domain — distilaide would supply only the
      model. @low
