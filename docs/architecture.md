# Distilaide architecture decisions

Why the mechanisms in this project exist, what they protect, and what a change would ripple into.
User-facing usage lives in `README.md`; the agent-facing rulebook is `AGENTS.md`. These records
answer a different question from either: *why does this exist at all?*

---

## 1. The pair is MADE, not found — the degradation is a config line

*2026-08-05*

### Context

Every other task in this workspace reads its target from the dataset: a class id, a per-pixel id map,
a box set. Image restoration's target is another IMAGE, and the obvious move is to look for a paired
noisy/clean dataset. That search was run first, against the Hugging Face API rather than from memory,
and it came back empty for anything a `HuggingFaceSource` can load:

* `huggan/facades`, `huggan/maps`, `huggan/night2day` — pairs per row, but as raw `{bytes, path}`
  structs rather than `Image` features;
* `eugenesiow/Div2k` — `lr` / `hr` columns are STRINGS (a script-based dataset, dead under
  `datasets` 5.x);
* `geekyrakshit/LoL-Dataset`, `Jason0411202/USenhance2023_Aligned` — the two halves are separate ROWS
  distinguished by a `ClassLabel`, not paired;
* `deepinv/drunet_dataset` — 5 GB; `deepinv/denoising` — 6 rows.

And recordstream has no pairing source (`sources/` is base / concat / huggingface / range / split),
so stitching two repositories row-by-row is not available either.

### Decision

Derive the target. Three ops, all of which already existed:

1. `ConvertToImage` — resize, before anything is copied;
2. `CopyField(src="image", dst="target")` — the clean copy IS the ground truth;
3. `albumentations.GaussNoise` — degrade the input alone.

This is not a workaround for the missing dataset; it is what the field does. DnCNN, DRUNet and SCUNet
were all trained exactly this way, because a real noisy photograph has no noise-free twin — the clean
image is the only ground truth that can exist.

### Consequences

* **The dataset became interchangeable and the DEGRADATION became the task.** Oxford-IIIT Pet appears
  in the example because it is natural photography, small, and already cached in this workspace — not
  because it is a denoising set. Changing `noise_sigma` changes the problem; changing the source does
  not.
* **Restoration needed exactly ONE new upstream primitive**, and it was not an op: the prediction
  contract `recordstream.RestorationOutput`. No distilaide collate, no distilaide noise op.
* **Op ORDER became load-bearing, and silently so.** A bare albumentations transform receives only
  albumentations' own key vocabulary (`_ALB_KEYS`), so `target` is invisible to it. Everything that
  must reach both halves goes above the copy; everything that degrades the input goes below it.
* A real paired dataset can still be used — point `input_feature` at one column and add a
  `ConvertToImage(field=…, output="target")` for the other. Nothing in the runnables assumes the
  target was synthesised.

### Example

```yaml
preprocess:
  - !class:recordstream.ops.image.ConvertToImage {field: image, width: 192, height: 192}
  - !class:recordstream.ops.structure.CopyField  {src: image, dst: target}
  - !class:albumentations.GaussNoise             {std_range: [0.1, 0.1], p: 1.0}
  - !class:recordstream.ops.torch.ToTensor       {field: image,  normalize: true}
  - !class:recordstream.ops.torch.ToTensor       {field: target, normalize: true}
```

### What you may change

The degradation (swap `GaussNoise` for a blur, a downsample, a JPEG round-trip — the runnables know
nothing about which one ran), the dataset, the raster. What you may NOT change is the ordering rule:
a geometric op below the copy unpairs every record while still producing a valid-looking batch.

---

## 2. `data_range` is this task's `num_classes`

*2026-08-05*

### Context

A classification or per-pixel task sizes its metrics with `num_classes`, resolved at run time and
injected into every metric template. Image restoration has no class count — but it has an equally
load-bearing run-level scalar: the span of the target's values, which every full-reference image
metric is defined relative to.

Getting it wrong is silent and enormous. Measured on one batch of `[0, 1]` images:

| `data_range` | PSNR |
|---|---|
| `1.0` (correct) | 20.46 dB |
| `255.0` | **68.59 dB** |

Same tensors. The wrong value does not raise, does not warn, and reports a model that looks nearly
perfect.

### Decision

`data_range: float = 1.0` is a constructor param on every backend, injected into every metric
template through `build_metric_collection(spec, data_range=self.data_range)` — structurally identical
to a segmentation task's `num_classes=`.

Metrics are therefore wired as `!lazy:` TEMPLATES rather than eager `!class:…()` instances. torchmetrics
1.9 makes `data_range` a *required* positional argument of `PeakSignalNoiseRatio` (no default) for
precisely this reason, so a template that never received one fails to BUILD rather than quietly
scoring against an inferred range.

### Consequences

**It forced a fix one layer up, and that fix is the general rule.** `build_metric_collection` used to
flow every template with every dimension. That worked for classification by luck — every
`torchmetrics.classification` metric happens to declare `num_classes`. Only three of the seven
full-reference image metrics declare `data_range`:

| Metric | `data_range`? |
|---|---|
| `PeakSignalNoiseRatio` | REQUIRED |
| `StructuralSimilarityIndexMeasure` | yes |
| `MultiScaleStructuralSimilarityIndexMeasure` | yes |
| `ErrorRelativeGlobalDimensionlessSynthesis` | no |
| `RootMeanSquaredErrorUsingSlidingWindow` | no |
| `SpectralAngleMapper` | no |
| `UniversalImageQualityIndex` | no |

So wiring any of the last four beside the defaults broke the whole collection.
`matrainer.torch.metrics._sizing_kwargs` now filters twice — the target must DECLARE the parameter,
and must not already have PINNED it — which additionally makes the injection a FLOOR rather than an
override, so a template writing its own `data_range: 2.0` keeps it.

The declaration check is `inspect.signature`, deliberately **not** `confluid.accepts_key`: every
torchmetrics metric takes `**kwargs` (the base `Metric.__init__` uses it for `compute_on_cpu` /
`dist_sync_on_step`) and then rejects anything unrecognised, so `accepts_key` answers yes for all of
them.

### Example

```yaml
data_range: 1.0

val_metrics:
  sam: !lazy:torchmetrics.image.SpectralAngleMapper   # takes no data_range — and is left alone
```

### What you may change

The value, per run. What you may not do is pass a metric's scale twice — once here and once on the
template — or reintroduce a blanket flow of every dimension into every template.

---

## 3. Learned and classical models share ONE slot

*2026-08-05*

### Context

This project's predecessor was a `denoise(signal, method=...)` dispatcher over four libraries. Its
classical methods (BM3D, spectral gating) and its learned ones (DnCNN, DRUNet, SwinIR) were separate
code paths, and neither had a metric at either end — so the question a user actually has, *which of
these is better on my data?*, was the one thing it could not answer.

### Decision

`DeepInvModel` wraps deepinv's zoo behind one closed `arch:` Literal spanning both kinds: the learned
nets and the classical algorithms, which have exactly zero parameters and wear the `nn.Module`
interface anyway. Both go in the same `model:` slot.

### Consequences

* **`distilaide evaluate` scores BM3D and a trained DnCNN through the same PSNR/SSIM path**, on the
  same data, into the same tracker. That comparison is the whole reason both live in one slot.
* **`task: fit` over a parameter-free model is refused BY NAME on every backend**, pointing at
  `evaluate`. Without the check, torch reports "optimizer got an empty parameter list" several frames
  from the `arch:` line that caused it, and fastai's version surfaces even further away, inside
  `fit_one_cycle`. The refusal is TASK-scoped, never model-scoped — the same object is exactly right
  for an evaluation, and a run refused by one backend and accepted by another would make the
  cross-backend comparison a lie.
* **`evaluate` carries more weight in this project than in the others**, and that is a property of
  the task rather than of the CLI.

Three facts about the zoo are ENCODED rather than restated, because deepinv ships no factory
(unlike smp's `create_model`) and a per-architecture table would be a second description that a
release could outdate:

1. **The channel knob has five names** — `in_channels` / `in_nc` / `in_chans` / `img_channels` /
   `channels`, and none at all on the classical ones — so one declared knob is translated by
   inspecting the target's signature.
2. **`sigma` is a MODEL knob**, because only the architecture decides whether it matters: `drunet`,
   `bm3d`, `tv` and `tgv` require it, `dncnn` and `unet` are blind. It is passed POSITIONALLY, since
   the zoo agrees on the argument's position (`Denoiser.forward(x, sigma)`) but not its name
   (`TVDenoiser` calls it `ths`).
3. **`pretrained` defaults to `None`** — a deliberate divergence from deepinv's own `"download"`,
   because a training project must not silently start from somebody else's weights.

### Example

```yaml
# the same slot, the same evaluate path, two very different models
model: !lazy:distilaide.DeepInvModel {arch: dncnn}              # trains
model: !lazy:distilaide.DeepInvModel {arch: bm3d, sigma: 0.1}   # evaluates only
```

### What you may change

Add an architecture by extending the Literal and the class-name table — the knob translation follows
from inspection and needs no edit. What you may not do is split the classical models into a second
slot or a second class; that recreates the predecessor's two code paths.

---

## 4. A restoration metric is ACCUMULATED, never averaged per batch

*2026-08-05*

### Context

A backend whose engine owns the loop gets stateful metrics for free (`update()` per batch,
`compute()` once). A backend whose loop we own sees one batch at a time, and the obvious shape —
`fn(preds, target)` averaged over the epoch — is available and wrong.

### Decision

Every backend `update()`s per batch and `compute()`s once at epoch end.

### Consequences

PSNR is a LOGARITHM of a mean squared error, so the mean of per-batch PSNRs is not the PSNR of the
split, and the two answers drift apart as the batches differ from one another. Accumulating is what
keeps two runs at different batch sizes comparable, and what makes the `lightning` and `torch`
columns of a cross-backend table mean the same thing — which is the only reason a control experiment
is worth having.

Note the asymmetry with the per-pixel and classification tasks: every full-reference image metric is
a SCALAR, so there is no vector or matrix to route around `log_dict`, and this project has no
text/figure reporting channel. `scalar_metrics` still guards the call, because a config may wire a
metric that is not.

### Example

```python
def _report_epoch_metrics(self, collection):
    if not isinstance(collection, MetricCollection):
        return
    self.log_dict(scalar_metrics(collection.compute()))   # ONE reduction, at epoch end
```

### What you may change

Which metrics a split measures. What you may not do is "simplify" a loop to a per-batch average.

---

## 5. The bare-torch backend is the control experiment

*2026-08-05*

### Context

Every backend in this workspace claims "swap the runnable, keep the config". A claim like that needs
something to falsify it.

### Decision

`TorchRestorer` removes the training FRAMEWORK entirely — `zero_grad` / `backward` / `step` written
out, an explicit device, an explicit `model.train()` / `model.eval()`, an explicit `torch.no_grad()`
on the eval pass — and the config still does not change.

### Consequences

* Its `!scope:framework=torch` block is the smallest in the example config, which is the point.
* If a knob only worked because Lightning implemented it, this backend is where that shows up.
* **What it lacks is honest rather than missing:** no `lightning:` Trainer slot, no framework loggers,
  no callback protocol, no distributed strategy. Multi-GPU is out of scope on purpose — that is what
  Lightning is for, and a half-`Trainer` here would be a worse one.
* It gains three knobs the others have no equivalent for — `device`, `checkpoint_dir`, `grad_clip` —
  and a `history` `@output`, because a loop that owns its epochs has to accumulate what a `Trainer`'s
  logger would have.
* It must not import Lightning even transitively; `matrainer.lightning`'s package `__init__` runs
  `require("pytorch_lightning")`, so the guard is an AST scan of this module's imports rather than a
  `sys.modules` check.

### Example

```yaml
torch: !scope:framework=torch
  runnable: !class:distilaide.TorchRestorer
    device: auto
```

### What you may change

The loop's details. What you may not do is grow it a partial `Trainer`, or let a Lightning import in.


---

## 6. The shared DnCNN spec — five frameworks, one network

*2026-08-06*

### Context

Four of the eight backends (Keras, TensorFlow, JAX, MLX) have **no restoration zoo to reach for**.
deepinv is torch-only; `transformers` dropped Flax in v5; nothing publishes MLX denoising weights.
So a cross-framework comparison built on zoos could only ever include the torch backends, and if the
other four each wrapped whatever their ecosystem happened to have, the table would compare four
different networks and tell you nothing about the frameworks.

### Decision

Specify ONE architecture — DnCNN, the canonical learned denoiser — in `distilaide/dncnn/spec.py`,
and implement it natively five times. The spec owns the constants AND **derives**
`trainable_parameter_count`, so each implementation is checked against the CLAIM rather than against
whichever was written first.

### Consequences

* **All five report 57,635 parameters**, including under custom `depth` / `width` — that number is
  the only thing holding them together, because different APIs cannot share code and the layer order
  is therefore prose.
* **Excluding batch norm's running statistics takes a different mechanism in every framework**, and
  that is the wrinkle worth knowing:

  | framework | how the running stats are excluded |
  |---|---|
  | torch | they are BUFFERS — filter on `requires_grad` |
  | Keras | non-trainable WEIGHTS — read `trainable_weights` |
  | raw TensorFlow | declared `trainable=False` |
  | Flax | `nnx.BatchStat` — a different variable TYPE in one state tree |
  | **MLX** | **the SAME tree as the weights — a NAME filter is the only way** |

* **The residual connection is the architecture's idea and costs nothing**: DnCNN predicts the noise
  and subtracts it, which is why approximate identity is its cheapest local optimum and a run's first
  epoch lands on the do-nothing baseline. (A training property, not an initialization one — measured.)
* Two per-framework traps are silent when wrong: MLX filters underscore-prefixed attributes out of
  `parameters()`, so the built network lives in the PUBLIC `net` (as `_model` the optimizer would see
  an empty tree and train nothing, forever, without raising); and an nnx attribute first assigned
  `None` becomes STATIC in Flax's pytree, so the lazy build is wrapped in `nnx.data`.
* Unlike a U-Net there is **no size divisor** — DnCNN never changes the spatial size.

### Example

```python
from distilaide.dncnn import spec
from distilaide.dncnn.mlx_dncnn import MlxDnCNN

assert MlxDnCNN().trainable_parameter_count == spec.trainable_parameter_count() == 57_635
```

### What you may change

The constants, the depth, the width — the derivation follows. What you may not do is let one
implementation drift, or add a sixth that is not an instance of its framework's own module type.

---

## 7. Every engine reports PSNR, and the vocabularies differ underneath

*2026-08-06*

### Context

PSNR is the field's primary score, and a cross-engine table is worthless without it in every row.
But **no non-torch engine ships it**: `keras.metrics` has no image metric at all, optax has none,
`mlx.nn` has none, and a `transformers.Trainer` computes whatever its `compute_metrics` hook returns.

### Decision

Three different answers, one per engine family, chosen by what each ecosystem actually provides:

| family | how PSNR is obtained |
|---|---|
| torch (lightning / torch / fastai) | `torchmetrics.image.PeakSignalNoiseRatio` — upstream, registered |
| Keras (keras / tensorflow) | a **shim** over `keras.ops.psnr`, holding an upstream `MeanSquaredError` |
| owned loops (jax / mlx / huggingface) | `matrainer.metric_functions` — a shared squared-error accumulator |

All three agree to six decimal places on the same arrays, which is what makes one column called
`psnr` mean one thing.

### Consequences

* **The Keras shim is the one metric class this project owns**, and it is the sanctioned "generic
  shim" exception: `keras.ops.psnr` is a FUNCTION, so confluid's `register` (which tags classes)
  cannot reach it. It holds an upstream accumulator and converts once — no error arithmetic of its
  own.
* **SSIM is deliberately absent from the Keras family.** `keras.ops` has none and the only
  implementations are engine-SPECIFIC, so wiring `tf.image.ssim` would make `--framework tensorflow`
  compute a metric `--framework keras` cannot, from the same config. A narrower vocabulary is the
  honest answer.
* The accumulator moved UP to matrainer rather than being written twice, under that package's
  existing rule for metrics an owned loop cannot average.

### Example

```python
from matrainer.metric_functions import zero_squared_error, accumulate_squared_error, psnr

totals = zero_squared_error()
for restored, clean in batches:
    accumulate_squared_error(totals, restored, clean)
print(psnr(totals, data_range=1.0))     # one reduction, at the end
```

### What you may change

Which metrics a split reports. What you may not do is give one engine a metric another cannot
compute from the same config, or average PSNR per batch (see record 4).

---

## 8. Swin2SR refuses to train itself, which gives this task's HuggingFace backend a loss slot

*2026-08-06*

### Context

The HuggingFace backends in this workspace's sibling projects register NO losses, and the reason is
structural: an `AutoModelForImageClassification` / `…ForSemanticSegmentation` handed `labels=`
computes its own loss inside `forward`, so there is nothing a config could wire.

`transformers` 5.x maps exactly ONE architecture for image-to-image
(`MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES == {"swin2sr": …}`), and it does the opposite:

```
Swin2SRForImageSuperResolution(pixel_values=x, labels=y)
-> NotImplementedError: Training is not supported at the moment
```

### Decision

Ship a `Trainer` SUBCLASS whose `compute_loss` pops the target and applies the runnable's own
`loss_fn`, and point `FRAMEWORKS["huggingface"].registrations` at the TORCH loss module.

### Consequences

* This is the ONE HuggingFace backend in the workspace with a real `loss_fn` slot, and it is the
  model's fault rather than a design choice.
* A `torch.nn` loss serves it unchanged — a loss does not care which loop calls it — so no new
  registration module was needed.
* **The wrapper's `forward` takes both `images` and `pixel_values`, and declares `labels` only to
  ignore it.** A `Trainer`'s evaluation path calls `model(**inputs)` DIRECTLY, bypassing
  `compute_loss` (a positional-only wrapper died with *"got an unexpected keyword argument
  'pixel_values'"* on the first eval), and `Trainer` derives `label_names` from the forward
  SIGNATURE — so declaring the parameter is what routes evaluation back through `compute_loss`.
* **A second measured fact decides the model**: every published Swin2SR checkpoint upscales (x2/x4),
  so a same-size denoising run cannot use pretrained weights. The architecture does same-size
  restoration at `upscale=1` with an empty upsampler, so the backend trains from scratch — and it
  still wins the fleet at 27.41 dB, because what it contributes is the training PROGRAM.
* There is deliberately no `train_metrics` slot: `compute_metrics` is an evaluation hook, so
  honouring one would cost a second forward pass over the train split every epoch.

### Example

```yaml
huggingface: !scope:framework=huggingface
  runnable: !class:distilaide.HuggingFaceRestorer
    learning_rate: 0.0002
  model: !lazy:distilaide.HuggingFaceRestorationModel
    upscale: 1          # same-size restoration; every published checkpoint upscales
```

### What you may change

The loss, the `TrainingArguments` (a config key WINS over the derived floor), the whole `Trainer`
through its slot. What you may not do is pass `labels=` to the model.


## 9. The sweep is an overlay, and the model is a second dimension

*2026-08-06*

### Context

The project's claim is that the config is the program: one file trains under eight engines and only
a `!scope:framework=` block changes. Nothing EXECUTED that claim — each backend had been verified
alone, and the headline results table was assembled by hand from eight separate runs.

Worse, that table mixes two model sizes. The torch backends wire the deepinv zoo (a 668K DnCNN) and
the rest wire the shared 57,635-parameter spec, because deepinv is torch-only — so the table reads
as a ranking of frameworks when the biggest term in it is the network.

### Decision

`examples/sweep/all_backends.yaml` — an overlay that `include:`s the example and adds exactly two
things: a `tracker` dimension and one metric vocabulary. Plus a `!scope:model=dncnn` dimension, which
lives in the **example** rather than the overlay, so the shared spec is reachable on a default
checkout.

### Consequences

* **Eleven cells, and both gaps are structural.** No non-torch restoration zoo exists to wrap. And
  `huggingface` takes no `dncnn` — for a NARROWER reason than the sibling per-pixel project's, which
  is worth knowing: there the obstacle is that a `Trainer` drives models computing their own loss;
  here that is not it, since this backend supplies the loss itself. What remains is the CALLING
  CONVENTION — `Trainer.prediction_step` calls `model(**inputs)`, and `TorchDnCNN.forward(images)`
  takes no `pixel_values=` keyword. Measured: it trains, then dies on the first eval.
* **The gap is made UNREACHABLE, not merely documented.** The huggingface block is spliced after the
  top-level `dncnn:` one, so `--model dncnn` leaves that row alone.
* **It found a real bug in the CLI within minutes.** `distilaide train --framework tensorflow` died
  on a default checkout, because the registration import binds Keras to the torch engine before
  `distilaide._tensorflow` can set its own. The fleet's own tensorflow number had been taken with
  `KERAS_BACKEND` already exported, so nothing caught it — a sweep is what runs a backend the way a
  reader would.
* **And a second one in the metric it exists to compare.** fastai could not report PSNR at all
  (`fastai.metrics` has nothing scale-aware), leaving the headline column blank on two of eleven
  rows. Closing it took a CONFIGURED ALIAS holding matrainer's squared-error accumulator — and the
  first version of that alias crashed on MPS, because `matrainer.metric_functions` is deliberately
  framework-free and `np.asarray` cannot read a live accelerator tensor. A CPU-tensor unit test
  passed the whole time; the sweep is what ran it on a device.
* **No row pins an engine, and that is a result.** The per-pixel sibling must export
  `KERAS_BACKEND=tensorflow` on its keras row, because `keras.metrics.MeanIoU` scatters per PIXEL
  through a Python loop on the torch engine. Restoration's Keras metrics are `keras.ops.psnr` and a
  mean squared error — no scatter — so the keras row runs on whichever engine is bound.
* The report is NOT here: `matrainer.sweep` owns naming the experiment and reading it back, because
  both are queries against a tracker (matrainer's `docs/architecture.md` record 8).

### Example

```bash
examples/sweep/run_sweep.sh --tracker csv --epochs 5
python -m matrainer.sweep report csv Thu_Aug_6_2026_1 --all
```

### What you may change

The cells, the shared metric pair, the tracker. What you may not do is ask an engine for a metric
another cannot produce from the same config — that turns a comparison back into a list of runs.


## 10. A model seed that restores the RNG reproduces nothing

*2026-08-06*

### Context

The sweep's value is that a row means something, which requires the row to repeat. It did not:
two runs of the identical `--framework torch` cell gave 22.7530 and 22.5188 dB.

The obvious diagnosis was wrong in an instructive way. `torch_dncnn` was already bit-identical run
to run while `torch_zoo` was not, and the only difference between those cells is the MODEL — which
reads as "the initialization is the whole variance, and the data order and the noise draw are
irrelevant." Both halves of that are false.

### Decision

Two seeds, neither sufficient alone:

1. **`seed:` on a model wrapper calls `torch.manual_seed` when the network is built, and does NOT
   restore the previous state.**
2. **The degradation is wrapped in a one-element `albumentations.Compose` carrying `seed:`.**

### Consequences

* **The measurements are the argument:**

  | seeded | run 1 | run 2 | spread |
  |---|---|---|---|
  | nothing | 22.7530 | 22.5188 | 0.234 dB |
  | model, saving+restoring the RNG | 22.1891 | 21.9956 | 0.193 dB |
  | model, leaving the RNG seeded | 22.1058 | 22.1707 | 0.065 dB |
  | model + degradation | **22.1886** | **22.1886** | **0** |

* **The leak is the mechanism.** A save-and-restore version was written FIRST, because seeding a
  model looked like it should not decide the rest of the run's randomness. It gives a reproducible
  MODEL inside a run that still is not reproducible — the shape of answer nobody wants. The build
  happens once, before any training randomness is drawn, so seeding there is a run-level seed
  applied at the earliest useful moment; it fixes the loader's shuffle order too.
* **`torch_dncnn` reproduced for that same reason**, not because init is the whole story:
  `TorchDnCNN.solidify` already seeded the global RNG and never restored it. A measurement that
  looks like a clean isolation can be an accident of another class's implementation.
* **A bare albumentations transform cannot be seeded at all.** It builds its RNG from OS entropy
  and refuses the argument outright — *"Argument(s) 'seed' are not valid for transform
  GaussNoise"*. Only a `Compose` accepts one, which is why the noise is wrapped in a one-element
  Compose in every chain rather than given a knob.
* **Two fresh seeded Composes draw identically; one Compose called twice does not.** That is
  exactly right for a per-record stream — the run reproduces, and every record still gets its own
  noise.
* Kernel nondeterminism is NOT a factor at this size: CPU and MPS were both non-reproducible before
  the degradation seed and both exact after.

### Example

```yaml
model: !lazy:distilaide.DeepInvModel {arch: dncnn, seed: 0}

preprocess:
  - !class:albumentations.Compose
    seed: 0
    transforms:
      - !class:albumentations.GaussNoise {std_range: [0.1, 0.1], p: 1.0}
```

### What you may change

The seed values, obviously. What you may not do is make a model seed restore the RNG state, or move
a `GaussNoise` back out of its `Compose` — each looks tidier and each silently unseeds the run.

---

## 11. The Lightning backend runs on the workspace trainer bases (2026-08-06)

**Context.** The four task projects were a four-way mirror: the same merged
train/evaluate/test/predict runnable per engine, with the Lightning modules 40–61%
byte-identical ACROSS projects (tracker wiring, trainer construction, step skeletons, epoch
reporting, sidecars, outputs) and the task logic near-identical WITHIN each project across its
engines. Both projects' own comments called the duplication deliberate-and-temporary, deferred
until the fleet existed to design a base against. By 2026-08-06 it did.

**Decision.** The trainer machinery moved UP to matrainer (its `docs/architecture.md` record 9
carries the full shape): a neutral abstract API (`TrainerRunnable`), a torch family layer
pinning the types (`TorchTrainerRunnable`), and the Lightning machinery base
(`LightningRunnable`). THIS project keeps only its task half, written once for the whole engine
fleet — `restoration_base.py` (torch-free, importable by every backend) + `restoration_torch.py` (the
torch-typed hook implementations) — and the concrete Lightning class becomes the composition
plus its own full flat constructor (the confluid introspection surface, signature unchanged).

**Consequences.** One task behaviour = one hook implementation instead of one copy per engine;
the fleet drift tests now check "inherits the shared base and does not shadow its wiring"
instead of per-module source patterns; matrainer is load-bearing for training (the old
"removing it keeps the class training" claim is dead by design). The other engine backends
still carry their private copies until their family's migration phase lands (root workspace
backlog).

**Example.**

```python
@configurable(task="restoration", role="trainer", framework="torch")
class LightningRestorer(RestorationTorchHooks, LightningRunnable):
    def __init__(self, model, train_set=None, ..., data_range=1.0):
        super().__init__(model=model, train_set=train_set, ...)  # shared slots
        self.data_range = float(data_range)                                          # task-specific state
```

**Amendment (2026-08-07).** The bare-torch backend joined the same shape: ``TorchRestorer`` is `(RestorationTorchHooks, matrainer.torch.TorchLoopRunnable)` — the same task hooks, a different machinery base (the owned epoch loop instead of a `pl.Trainer`), and one stored task knob (`data_range`).

**What you may change.** Hook implementations and task state freely; the concrete signature
(it is the config surface). What you may not do: shadow the inherited machinery
(`_build_trainer` / `_live_trackers` / `_run_eval` / `_report_epoch_metrics`) — extend through
the hooks, or change the base in matrainer with every consumer updated in the same workspace
change.
