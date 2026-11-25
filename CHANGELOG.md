# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **The licence is now the PolyForm Noncommercial License 1.0.0** (`LICENSE`, `pyproject.toml`): free for personal, academic and other noncommercial use; commercial use needs a separate licence from the author. It was MIT; the versions released under MIT keep that licence.

## [0.1.0a1] — 2026-09-28

First public pre-release. The surface it ships:

### Backends
- **Eight training engines over one config**, each a runnable class behind its own extra:
  `LightningRestorer` (`[lightning]`, the reference), `TorchRestorer` (`[torch]`, a
  hand-written loop), `FastaiRestorer` (`[fastai]`), `KerasRestorer` (`[keras]`),
  `TensorFlowRestorer` (`[tensorflow]`), `JaxRestorer` (`[jax]`), `MlxRestorer` (`[mlx]`,
  Apple silicon) and `HuggingFaceRestorer` (`[huggingface]`). Each runnable dispatches on
  `task`: `fit` / `evaluate` / `test` / `predict`.
- **Backend-free core:** `import distilaide` loads no ML framework. Touching a backend that is not
  installed raises an `ImportError` naming the extra to install.

### Models
- `DeepInvModel` — deepinv's restoration zoo by `arch`: learned models (DnCNN, DRUNet, SCUNet,
  SwinIR, Restormer, …) and classical ones (BM3D with `[bm3d]`, TV/TGV, wavelet, median,
  bilateral) in the same model slot, so a classical baseline is scored exactly like a trained
  network.
- `HuggingFaceRestorationModel` (Swin2SR).
- The `distilaide.dncnn` package: one DnCNN spec with five native implementations, for comparing
  engines on one network.

### Task, losses and metrics
- Gaussian denoising built from clean images alone: the noisy input is derived from the target by
  ops in the config, so no paired dataset is needed.
- Registered restoration losses and PSNR/SSIM metrics per engine, with per-split metric choices.

### Command line
- `distilaide train | evaluate | predict <config.yaml> [--framework <name>]` and
  `distilaide frameworks`; the same config also runs under `recordstream run`.
