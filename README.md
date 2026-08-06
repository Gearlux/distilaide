# Distilaide

Image restoration — **image → image** — across training backends, driven by one config.

```bash
distilaide train    examples/train_dncnn_denoising.yaml
distilaide train    examples/train_dncnn_denoising.yaml --framework torch
distilaide evaluate examples/train_dncnn_denoising.yaml
```

The canonical task is **Gaussian denoising**: learn to map a noisy image back to the clean one. It is
the textbook image-to-image problem, and the recipe every classic denoiser was trained under — DnCNN,
DRUNet and SCUNet all learned from clean photographs with synthetic noise added, because a real noisy
photograph has no noise-free twin.

| Backend | `--framework` | What it swaps |
|---|---|---|
| PyTorch Lightning | `lightning` (default) | the reference: a `Trainer` owns the epoch |
| PyTorch | `torch` | **nothing underneath** — the loop is written out (the control experiment) |
| fastai | `fastai` | a `Learner` with one-cycle scheduling, same torch stack |
| Keras 3 | `keras` | one API over PyTorch / TensorFlow / JAX (channels-last) |
| TensorFlow | `tensorflow` | `tf.data` + `GradientTape` under `@tf.function`, SavedModel export |
| JAX | `jax` | a jitted optax loop over a Flax model, written out in full |
| MLX | `mlx` | Apple silicon: unified memory, lazy arrays |
| Transformers | `huggingface` | a `Trainer` owns epoch, optimizer, scheduler, AMP, distribution |

## Install

```bash
pip install git+https://github.com/Gearlux/distilaide.git@main
```

The core is backend-free — you get the CLI and the config surface, and no ML framework. Pick a
backend:

```bash
pip install 'git+https://github.com/Gearlux/distilaide.git@main#egg=distilaide[lightning]'
pip install 'git+https://github.com/Gearlux/distilaide.git@main#egg=distilaide[torch]'
pip install 'git+https://github.com/Gearlux/distilaide.git@main#egg=distilaide[fastai]'
pip install 'git+https://github.com/Gearlux/distilaide.git@main#egg=distilaide[keras]'
pip install 'git+https://github.com/Gearlux/distilaide.git@main#egg=distilaide[tensorflow]'
pip install 'git+https://github.com/Gearlux/distilaide.git@main#egg=distilaide[jax]'
pip install 'git+https://github.com/Gearlux/distilaide.git@main#egg=distilaide[mlx]'
pip install 'git+https://github.com/Gearlux/distilaide.git@main#egg=distilaide[huggingface]'
pip install 'git+https://github.com/Gearlux/distilaide.git@main#egg=distilaide[bm3d]'   # + the BM3D baseline
```

### Development

```bash
git clone https://github.com/Gearlux/distilaide.git
cd distilaide
pip install -e ".[dev,lightning]"
```

## The degradation IS the task

There is no paired noisy/clean dataset here, and that is the design rather than a shortcut. A
denoising pair is **made, not found**: the clean image is the only ground truth that exists, so the
target is derived from the input by three ops in the config.

```yaml
noise_sigma: 0.1          # sigma 25 on the 0-255 scale the literature quotes

preprocess:
  - !class:recordstream.ops.image.ConvertToImage      # 1. resize — BOTH halves inherit this
    field: image
    width: 192
    height: 192

  - !class:recordstream.ops.structure.CopyField       # 2. the clean copy = the target
    src: image
    dst: target

  - !class:albumentations.GaussNoise                  # 3. degrade the INPUT only
    std_range: [0.1, 0.1]
    p: 1.0

  - !class:recordstream.ops.torch.ToTensor            # 4+5. both halves to CHW float in [0, 1]
    field: image
    normalize: true
  - !class:recordstream.ops.torch.ToTensor
    field: target
    normalize: true
```

Two consequences worth knowing:

**The dataset is interchangeable.** What makes this a denoising run is `noise_sigma`, not the source.
The example uses Oxford-IIIT Pet because it is natural photography and it is small;
[`deepinv/CBSD68`](https://huggingface.co/datasets/deepinv/CBSD68) (68 images) is the canonical colour
denoising *benchmark* and is what you want as a `test_set` for a number you intend to publish.

**Order matters, and getting it wrong is silent.** A bare albumentations transform only ever receives
albumentations' own keys (`image`, `mask`, `bboxes`, …), so it cannot see `target`. Everything that
must reach both halves goes **above** the copy; everything that degrades the input goes **below** it.
A resize placed after the copy shrinks the input and leaves the target at its original size —
unpairing every record while still producing a perfectly valid-looking batch.

## Models — learned and classical, one slot

`DeepInvModel` wraps [deepinv](https://deepinv.github.io/)'s restoration zoo. `arch:` is the whole
difference between them:

| `arch` | Kind | Notes |
|---|---|---|
| `dncnn` | learned | The canonical baseline. Residual CNN, blind (ignores `sigma`). |
| `drunet` | learned | Deep residual U-Net, the plug-and-play workhorse. **Needs `sigma`.** |
| `scunet` | learned | Swin-Conv-UNet, real-image denoising. |
| `swinir` | learned | Swin transformer for restoration. |
| `restormer` | learned | Efficient high-res restoration transformer. |
| `ffdnet` | learned | Noise-level-map conditioned, fast. No published weights in deepinv. |
| `unet` | learned | Plain U-Net. |
| `srresnet` | learned | SRResNet. |
| `bm3d` | **classical** | Block-matching 3D — still a strong baseline. Needs `distilaide[bm3d]`. |
| `tv` / `tgv` | **classical** | Total / total-generalized variation. **Need `sigma`.** |
| `wavelet` | **classical** | Wavelet shrinkage (needs deepinv's `ptwt`). |
| `median` / `bilateral` | **classical** | Fixed filters. |

The classical entries have **zero parameters**. They live in the same `model:` slot on purpose: it
means `distilaide evaluate` scores BM3D and a trained DnCNN through the same PSNR/SSIM path, on the
same data, into the same tracker — which is the comparison anyone choosing a denoiser actually wants.

```yaml
# score a classical baseline — a complete run that trains nothing
runnable: !class:distilaide.LightningRestorer
model: !lazy:distilaide.DeepInvModel
  arch: bm3d
  sigma: 0.1
```

```bash
distilaide evaluate baseline.yaml
```

Asking one to **train** is refused by name, pointing at `evaluate` — because torch's own complaint
("optimizer got an empty parameter list") lands several frames from the `arch:` line that caused it.

`pretrained` defaults to `None`, so a bare `arch: dncnn` **trains from scratch**. That deliberately
differs from deepinv's own default: a training project should not silently start from somebody else's
weights. Add `pretrained: download` to fine-tune from the published ones.

## Losses and metrics

Both are off-the-shelf upstream classes, registered for discovery rather than wrapped.

**Losses** (`torch.nn`): `L1Loss` (the default), `MSELoss`, `SmoothL1Loss`, `HuberLoss`.

L1 rather than MSE is not arbitrary — DnCNN's 2016 paper used MSE, but every modern restoration
network (Restormer, NAFNet, MPRNet) reports L1 doing better on the same data, because MSE's quadratic
penalty is dominated by a few large errors and yields visibly blurrier output at a comparable PSNR.
Reproducing the original recipe is one line:

```yaml
loss_fn: !class:MSELoss@framework=torch
```

**Metrics** (`torchmetrics.image`): `PeakSignalNoiseRatio` and `StructuralSimilarityIndexMeasure` are
the defaults, with `MultiScaleStructuralSimilarityIndexMeasure`,
`ErrorRelativeGlobalDimensionlessSynthesis`, `RootMeanSquaredErrorUsingSlidingWindow`,
`SpectralAngleMapper` and `UniversalImageQualityIndex` registered alongside.

### `data_range` — the one knob you must not get wrong

Every full-reference image metric is defined relative to the span of the target's values. Set it once,
at the top level, and the restorer injects it into every metric template:

```yaml
data_range: 1.0     # ToTensor(normalize: true) produces [0, 1]; raw uint8 would be 255.0
```

It is not a formality. Measured on one batch of `[0, 1]` images: PSNR reports **20.46 dB** at
`data_range=1.0` and **68.59 dB** at `255.0` — the same tensors, scored as if the model were nearly
perfect. torchmetrics makes it a *required* argument of `PeakSignalNoiseRatio` for exactly this
reason, which is why metrics are wired as `!lazy:` templates: a template receives the run's value,
where an eagerly-built instance would have to restate it and give the config two places to disagree.

### Per-split metrics

`metrics:` is what every split measures; `val_metrics:` is what validation measures on top.

```yaml
val_metrics:
  ms_ssim: !lazy:torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure
```

MS-SSIM on validation only, because it builds an image pyramid per pair. Note it also constrains your
raster: its default five-scale pyramid needs images **larger than 160 px** (>90 at four scales, >40 at
three), so a 128 px run dies during Lightning's sanity check.

## Reading the numbers

A restoration metric is never near zero, so the number that means something is the one against the
**do-nothing baseline** — what the degraded input already scores.

On the example config (256 training images, sigma 0.1, 5 epochs, everything from scratch), the
degraded input itself scores **PSNR 20.53 dB / SSIM 0.3719 / MS-SSIM 0.8516**. Every backend against
that floor:

| `--framework` | model | val PSNR |
|---|---|---|
| `huggingface` | Swin2SR | **27.41 dB** |
| `fastai` | deepinv DnCNN (668K) | 24.4–25.2 |
| `torch` | deepinv DnCNN (668K) | 22.2–22.9 |
| `lightning` | deepinv DnCNN (668K) | 21.5–22.5 |
| `jax` | FlaxDnCNN (the 57K shared spec) | 20.56 |
| `tensorflow` | TensorFlowDnCNN (spec) | 20.49 |
| `keras` | KerasDnCNN (spec) | 20.38 |
| `mlx` | MlxDnCNN (spec) | 20.31 |

**Read that by group, not by rank — the model differs, and that is the design.** The four shared-spec
backends agree within 0.25 dB, which is exactly what the spec is for (same network, four frameworks,
so what you are comparing is the framework). The zoo backends do better because a 668K DnCNN beats a
57K one, and Swin2SR wins because a transformer with a `Trainer`'s schedule beats both. The
like-for-like comparison — every backend on the SAME network — is what
[the sweep](#comparing-the-backends--the-sweep) below produces.

The figures above were taken before the config was seeded, so they are ranges. It is seeded now —
see [Reproducibility](#reproducibility--what-a-seed-actually-has-to-cover) — so a repeat run
reproduces exactly; treat the gain over the baseline, not the absolute number, as the result.

The epoch-0 row is the sanity check worth keeping: a residual denoiser's cheapest local optimum is
predicting zero noise, so it reaches approximate identity within one epoch and its first score
should land on the baseline. A first epoch far from the floor means the pair is wrong — not that the
model is good. (It is not an *initialization* property — at init the network is nowhere near the
identity; it gets there by training.)

### Reproducibility — what a seed actually has to cover

The config is **seeded**, and getting there took two knobs rather than one. Measured 2026-08-06,
two runs of the identical `--framework torch` cell:

| what was seeded | run 1 | run 2 | spread |
|---|---|---|---|
| nothing | 22.7530 | 22.5188 | 0.234 dB |
| the model, saving+restoring the RNG | 22.1891 | 21.9956 | 0.193 dB |
| the model, leaving the RNG seeded | 22.1058 | 22.1707 | 0.065 dB |
| **the model + the degradation** | **22.1886** | **22.1886** | **0.000** |

**Neither knob is sufficient alone, and the middle row is why.** Every model wrapper takes
`seed:`, which calls `torch.manual_seed` when the network is built and does *not* restore the
previous state — so it seeds the loader's shuffle order too. That leak is the point: a
save-and-restore version was written first, and it produced a reproducible *model* inside a run
that still was not reproducible.

The remaining piece is the degradation. A **bare** albumentations transform builds its own RNG from
OS entropy and refuses a `seed` argument outright (*"Argument(s) 'seed' are not valid for transform
GaussNoise"*), so the noise is wrapped in a one-element `Compose`, which is the only place the
library accepts one:

```yaml
- !class:albumentations.Compose
  seed: 0
  transforms:
    - !class:albumentations.GaussNoise
      std_range: [0.1, 0.1]
      p: 1.0
```

Two *fresh* seeded `Compose` objects draw identically; one `Compose` called twice does not — which
is exactly right for a per-record stream, and is what makes a whole run reproduce rather than every
record getting the same noise.

### Comparing the backends — the sweep

The table above mixes two model sizes, so it reads as a ranking when it is really two groups. The
sweep is what makes the like-for-like comparison reproducible: it trains **every installed backend**
into one tracker, the three torch ones twice — once on the deepinv zoo wrapper and once on the
shared 57,635-parameter DnCNN spec — so a column compares *frameworks* rather than networks.

```bash
examples/sweep/run_sweep.sh --epochs 5                   # every installed backend, into MLflow
examples/sweep/run_sweep.sh --tracker csv --epochs 5      # ...into the always-available CSV backend
examples/sweep/run_sweep.sh --only jax,mlx --dry-run      # print the commands, run nothing
```

Eleven cells, and the two gaps are structural rather than unfinished:

| `--framework` | zoo model | `--model dncnn` |
|---|---|---|
| `lightning` / `torch` / `fastai` | `DeepInvModel` (668K) | `TorchDnCNN` (57,635) |
| `keras` | — | `KerasDnCNN` |
| `tensorflow` | — | `TensorFlowDnCNN` |
| `jax` | — | `FlaxDnCNN` |
| `mlx` | — | `MlxDnCNN` |
| `huggingface` | Swin2SR | — |

There is **no non-torch restoration zoo to wrap** — deepinv is torch-only, which is exactly why the
shared spec exists. And **`huggingface` takes no `dncnn`**: not because of the loss (this backend
supplies its own), but because a `Trainer`'s *evaluation* path calls `model(**inputs)`, so a model
must accept `pixel_values=` as a keyword. `TorchDnCNN.forward(images)` does not — it trains fine and
dies on the first eval — and teaching the shared spec one framework's keyword names would defeat the
point of a class whose job is to be the same network in five of them.

`--model dncnn` also works on the plain example, with no sweep involved:

```bash
distilaide train examples/train_dncnn_denoising.yaml --framework torch --model dncnn
```

The report is `matrainer`'s, since naming an experiment and reading it back are queries against a
tracker rather than anything restoration knows about:

```bash
python -m matrainer.sweep report csv Thu_Aug_6_2026_1 --all
```

Verified 2026-08-06 against MLflow, `--tracker mlflow --epochs 5` — **11/11 FINISHED**, every row
carrying `val/psnr`, `val/mse` and `val/loss`, with the config seeded:

| run | val PSNR | val MSE | train PSNR | sec |
|---|---|---|---|---|
| `huggingface_zoo` | **27.1698** | 0.0019 | *(see below)* | 250 |
| `fastai_zoo` | 24.7319 | 0.0034 | 24.7259 | 25 |
| `torch_zoo` | 22.1886 | 0.0060 | 21.9302 | 37 |
| `lightning_zoo` | 22.1836 | 0.0060 | 21.9417 | 80 |
| `fastai_dncnn` | 21.4821 | 0.0071 | 21.4473 | 6 |
| `torch_dncnn` | 20.7328 | 0.0084 | 20.6109 | 19 |
| `lightning_dncnn` | 20.7023 | 0.0085 | 20.5884 | 61 |
| `jax_dncnn` | 20.6757 | 0.0086 | 21.2558 | 38 |
| `tensorflow_dncnn` | 20.4879 | 0.0089 | 21.1441 | 40 |
| `keras_dncnn` | 20.4558 | 0.0090 | 20.7653 | 60 |
| `mlx_dncnn` | 20.3854 | 0.0092 | 20.3954 | 17 |

**The two numbers worth reading are not the top of the table.**

`torch_zoo` **22.1886** against `lightning_zoo` **22.1836** — a hand-written loop and a
`pl.Trainer`, same config, same network, **0.005 dB apart**. That is the project's whole claim,
executed. (`torch_dncnn` / `lightning_dncnn` agree to 0.03 dB on the smaller network.)

And `torch_zoo`'s 22.1886 is the *third* time that exact figure has appeared — the two seeding
probes above produced it as well, in a different tracker and a different sweep. That is what a
seeded config buys.

The five `*_dncnn` rows span **20.39 – 20.73 dB**, i.e. a third of a dB across five frameworks
running the same 57,635-parameter network — which is what the shared spec exists to demonstrate.
`fastai_dncnn` sits above them at 21.48 on the same network, and that difference is the *loop*
(one-cycle scheduling), which is exactly the kind of thing a sweep is for.

The `sec` column is MLflow's, not the CSV backend's — a file-backed tracker records no run
lifecycle, so `--tracker csv` leaves `status` and `sec` empty.

The one blank cell is `huggingface`'s training columns, and it is declared rather than missing: a
`Trainer`'s `compute_metrics` is an **evaluation** hook, so a training metric would cost a second
forward pass over the whole train split every epoch.

**One caveat worth knowing before you run it.** `--epochs 1` is a smoke test, not a comparison: the
`*DnCNN` rows train from scratch, and a residual denoiser spends its first pass converging *toward*
the identity — measured, one epoch of `--framework tensorflow` lands at 8.0 dB, well under the
20.5 dB do-nothing floor. Use `--epochs 5`.

## CLI reference

| Command | Purpose |
|---|---|
| `distilaide train <yaml>` | Train (`task=fit`). |
| `distilaide evaluate <yaml>` | Score a model — learned or classical — over the test set. |
| `distilaide predict <yaml>` | Stream restored images to the configured sink. |
| `distilaide frameworks` | List the backends `--framework` accepts. |

`recordstream run <yaml>` also works — the generic runner, with no verb, using whatever `task:` the
YAML declares.

## What's in the package

| Module | Purpose |
|---|---|
| `distilaide.<engine>_restorer` | One per backend — `lightning` / `torch` / `fastai` / `keras` / `tensorflow` / `jax` / `mlx` / `huggingface`. |
| `distilaide.deepinv_model` | `DeepInvModel` — the zoo wrapper, learned and classical. |
| `distilaide.huggingface_model` | `HuggingFaceRestorationModel` — Swin2SR. |
| `distilaide.dncnn` | The shared DnCNN spec + its five native implementations. |
| `distilaide.torch_restoration` | The registered torch losses + torchmetrics image metrics. |
| `distilaide.keras_restoration` | The Keras losses + metrics, and the one PSNR shim. |
| `distilaide.cli` | `train` / `evaluate` / `predict` / `frameworks`. |

## Dependencies

| Package | Purpose | Required? |
|---|---|---|
| `confluid` | YAML config + `@configurable` | yes |
| `recordstream` | Dataset engine, ops, collate, prediction contracts | yes |
| `liquifai` | CLI framework | yes |
| `loggair` | Structured logging | yes |
| `matrainer` | Experiment tracking | yes |
| `torch` + `torchmetrics` + `deepinv` + `albumentations` | The model zoo, metrics and degradation | `[torch]` |
| `pytorch-lightning` | The reference backend | `[lightning]` |
| `fastai` | The `Learner` backend | `[fastai]` |
| `keras` (+ an engine) | Keras 3 / TensorFlow backends | `[keras]` / `[tensorflow]` |
| `jax` + `flax` + `optax` | The JAX backend | `[jax]` |
| `mlx` | The MLX backend (Apple silicon only) | `[mlx]` |
| `transformers` + `accelerate` | The Swin2SR backend | `[huggingface]` |
| `bm3d` | The BM3D classical baseline | `[bm3d]` |
