# Distilaide

(loggair / confluid / liquifai) adapter over four denoising libraries — AI and classical — exposed through one `denoise()` dispatcher and one Confluid-`@configurable` `Denoiser`.

| Backend | Domain | Methods | Extra |
|---|---|---|---|
| [`bm3d`](https://pypi.org/project/bm3d/) | Classical Gaussian image denoising | `bm3d` | `[classical]` |
| [`noisereduce`](https://pypi.org/project/noisereduce/) | Classical spectral-gating audio denoising | `noisereduce`, `spectral-gate` | `[classical]` |
| [`deepinv`](https://deepinv.github.io/) | Pretrained CNN/Transformer image denoisers | `dncnn`, `drunet`, `swinir`, `scunet`, `restormer` | `[deepinv]` |
| [`asteroid`](https://asteroid-team.github.io/) | Pretrained speech enhancement / source separation | `dccrn`, `dcunet`, `convtasnet`, `demucs` | `[asteroid]` |

Distilaide is **not** a denoiser. It's a thin glue layer (~250 lines of code) over four upstream packages that already do the work, with loggair logging, Confluid YAML config, and the same liquifai CLI conventions as the rest of the workspace.

## The denoising landscape

Distilaide ships method wrappers for the architectures highlighted **in bold**. Everything else listed below is a known, useful technique for context — pick the right baseline before reaching for a model.

### Image denoising (CNN era)

| Architecture | Year | Notes | Wrapped |
|---|---|---|---|
| **DnCNN** (Zhang et al.) | 2016 | Residual learning for Gaussian denoising — canonical baseline. | ✅ `dncnn` (deepinv) |
| REDNet | 2016 | Residual encoder–decoder. | — |
| MemNet | 2017 | Persistent-memory recurrent net. | — |
| FFDNet | 2018 | Noise-level-map conditioned, fast. | — |
| CBDNet | 2019 | Blind real-noise denoising. | — |
| RIDNet | 2019 | Real image denoising, attention-based. | — |
| N3Net / NLRN | 2018 | Non-local neural denoising. | — |
| MIRNet / MIRNetV2 | 2020 / 2022 | Multi-scale residual. | — |
| MPRNet | 2021 | Multi-stage progressive restoration. | — |
| HINet | 2021 | Half-instance-norm restoration. | — |
| **DRUNet** (Zhang) | 2021 | Deep residual U-Net — plug-and-play prior workhorse. | ✅ `drunet` (deepinv) |

### Image denoising (Transformer era)

| Architecture | Year | Notes | Wrapped |
|---|---|---|---|
| **SwinIR** (Liang et al.) | 2021 | Swin transformer for restoration. | ✅ `swinir` (deepinv) |
| Uformer | 2022 | U-shaped transformer. | — |
| **Restormer** (Zamir et al.) | 2022 | Efficient high-res restoration transformer. | ✅ `restormer` (deepinv) |
| **NAFNet** | 2022 | Nonlinear-activation-free net, SOTA on SIDD/GoPro. | — (planned) |
| **SCUNet** (Zhang) | 2022 | Swin-Conv-UNet, real-image denoising. | ✅ `scunet` (deepinv) |
| MAXIM / GRL | 2022–2023 | More recent restoration backbones. | — |
| UNet (DDPM/DDIM) | 2020+ | The diffusion-model workhorse, used as a denoiser via score matching. | — (planned `[diffusion]` extra) |

### Audio / speech / spectrogram denoising

| Architecture | Notes | Wrapped |
|---|---|---|
| Wave-U-Net | Time-domain U-Net for audio. | — |
| **DCUNet** | Deep Complex U-Net, speech enhancement. | ✅ `dcunet` (asteroid) |
| **DCCRN** | Deep Complex Convolution-Recurrent, speech enhancement. | ✅ `dccrn` (asteroid) |
| **ConvTasNet** | Time-domain mask network, source separation. | ✅ `convtasnet` (asteroid) |
| **Demucs / HT-Demucs** | Meta's source separation, often used as denoiser. | ✅ `demucs` (asteroid) |
| FullSubNet | Full-band + sub-band fusion SE. | — |
| DeepFilterNet | Efficient real-time deep-filtering SE. | — |
| CMGAN, MetricGAN+, SEGAN | GAN-based speech enhancement. | — |

### Classical baselines (non-AI)

| Method | Domain | Notes | Wrapped |
|---|---|---|---|
| **BM3D** | Image | Block-matching 3D — still a strong baseline at low SNR. | ✅ `bm3d` |
| **noisereduce / spectral gating** | Audio | Spectral subtraction with adaptive thresholding. | ✅ `noisereduce`, `spectral-gate` |
| Non-local means (NLM) | Image | Patch similarity averaging. | — (use `skimage.restoration.denoise_nl_means`) |
| Wavelet shrinkage | Image / signal | Frequency-domain thresholding. | — (use `skimage.restoration`) |
| Wiener filter | Signal | MMSE-optimal LTI filter under known PSDs. | — (use `scipy.signal`) |
| Total Variation (TV) | Image | Edge-preserving smoothness prior. | — (use `skimage.restoration`) |

Anything not yet wrapped that you need — file an issue or drop in a `<name>_backend.py` (the registry pattern in `distilaide.api` is intentionally one-file-to-add).

## Installation

```bash
pip install git+https://github.com/Gearlux/distilaide.git@main
```

Backends are optional extras — install only what you need:

```bash
pip install "git+https://github.com/Gearlux/distilaide.git@main#egg=distilaide[classical]"   # bm3d + noisereduce
pip install "git+https://github.com/Gearlux/distilaide.git@main#egg=distilaide[deepinv]"     # CNN/Transformer image denoisers
pip install "git+https://github.com/Gearlux/distilaide.git@main#egg=distilaide[asteroid]"    # speech enhancement
pip install "git+https://github.com/Gearlux/distilaide.git@main#egg=distilaide[all]"         # everything
```

### Development

```bash
git clone https://github.com/Gearlux/distilaide.git
cd distilaide
pip install -e ".[dev,classical]"     # core + classical backends + dev tools
```

## Architecture

```
   YAML (Confluid)         dispatch              upstream library
   ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐
   │ denoiser:    │───▶│  Denoiser        │───▶│ bm3d.bm3d            │
   │   method:..  │    │  api.denoise()   │    │ noisereduce.reduce_  │
   │   sigma:..   │    │  registry        │    │ deepinv.models.*     │
   │ input:       │    └──────────────────┘    │ asteroid.BaseModel   │
   │ output:      │            │                └──────────────────────┘
   └──────────────┘            │
            ▲                  ▼
       liquifai CLI       loggair logger
```

Each backend lives in a single `<name>_backend.py` exposing three things:

```python
METHODS: tuple[str, ...]                 # method names this backend serves
def is_available() -> bool: ...          # whether the upstream lib is importable
def denoise(signal, method, **kw): ...   # the actual call
```

The dispatcher in `distilaide.api` builds a `method → backend` registry at import and forwards calls. Upstream libraries are imported lazily, so the package works with any subset of the optional extras installed.

## Quick start

### From Python

```python
import numpy as np
from distilaide import denoise, list_methods

print(list_methods(only_available=True))    # which backends are installed?

noisy = np.random.uniform(0, 1, (256, 256)).astype(np.float32)
clean = denoise(noisy, method="bm3d", sigma=0.05)              # classical
clean = denoise(noisy, method="dncnn")                          # pretrained CNN
clean = denoise(noisy, method="swinir")                         # pretrained transformer
```

For audio:

```python
from distilaide import denoise

speech = ...  # 1D float32 array at 16 kHz
clean = denoise(speech, method="noisereduce", sample_rate=16000)   # classical
clean = denoise(speech, method="dccrn")                             # pretrained CRN
```

### From a Confluid YAML

```yaml
# examples/denoise_bm3d.yaml
denoiser: !class:distilaide.Denoiser
  method: bm3d
  sigma: 0.05

input: examples/data/noisy_image.npy
output: out/clean_image.npy
```

```bash
distilaide denoise examples/denoise_bm3d.yaml
```

## CLI reference

| Command | Purpose |
|---|---|
| `distilaide list-methods` | Print every registered method and whether its backend is installed. |
| `distilaide denoise <yaml>` | Apply the configured `denoiser:` to `input:` and write to `output:`. |

The CLI inherits Confluid's dotted-key overrides:

```bash
distilaide denoise cfg.yaml denoiser.sigma=0.1
```

## What's in the package

| Module | Purpose |
|---|---|
| `distilaide.api.denoise` | Single dispatch entry point. |
| `distilaide.api.Denoiser` | Confluid-`@configurable` callable wrapper. |
| `distilaide.api.list_methods` | Inspect the method registry + availability. |
| `distilaide.bm3d_backend` | BM3D classical Gaussian image denoiser. |
| `distilaide.noisereduce_backend` | noisereduce classical spectral gating. |
| `distilaide.deepinv_backend` | DnCNN / DRUNet / SwinIR / SCUNet / Restormer. |
| `distilaide.asteroid_backend` | DCCRN / DCUNet / ConvTasNet / Demucs. |
| `distilaide.cli` | `list-methods`, `denoise` |

That's it. Six files plus a CLI. Backends are added by dropping a `<name>_backend.py` and registering it in `distilaide.api._BACKEND_MODULES`.

## Why a thin adapter?

The denoising landscape is well-served by mature upstream packages — `deepinv` for image priors, `asteroid` for speech, `bm3d`/`noisereduce` for the classical baselines that are still the right call at low SNR. There's no value in re-implementing them.

What distilaide adds:

- One Python entry point (`denoise(signal, method=...)`) instead of four library APIs.
- One Confluid YAML knob (`!class:distilaide.Denoiser`) for use inside RecordStream pipelines, matrainer training loops, or any other config.
- loggair integration for unified logging across the workspace.
- liquifai CLI matching the rest of the .
- Lazy imports so the package itself works with any subset of `[classical]` / `[deepinv]` / `[asteroid]` installed.

If you only need one backend, use it directly. If you want to write code that swaps denoisers via config, distilaide is the glue.

## Dependencies

| Package | Purpose | Required? |
|---|---|---|
| `confluid` | YAML config + `@configurable` | yes |
| `liquifai` | CLI framework | yes |
| `loggair` | Structured logging | yes |
| `numpy` | Array protocol | yes |
| `bm3d` | BM3D denoiser | optional (`[classical]`) |
| `noisereduce` | Spectral gating | optional (`[classical]`) |
| `scipy` | WAV IO for the CLI | optional (`[classical]`) |
| `deepinv` + `torch` | CNN/Transformer image denoisers | optional (`[deepinv]`) |
| `asteroid` + `torch` + `torchaudio` | Speech enhancement / separation | optional (`[asteroid]`) |
