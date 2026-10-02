# Distilaide Mandates

Rules only. The reasons are in [`docs/architecture.md`](docs/architecture.md) (cited as §N),
and usage is in the [README](README.md). Root `AGENTS.md` rules are not repeated here.

## Current state

Image restoration (image → image) on eight backends, all driven by one config: `lightning`,
`torch` (a hand-written loop), `fastai`, `keras`, `tensorflow`, `jax`, `mlx`, `huggingface`
(Swin2SR). The canonical task is Gaussian denoising. On PyPI as pre-release `0.1.0a1`; the repo
is private.

Each runnable `<backend>_restorer.py` = task hooks (here) + a matrainer machinery base + its own
full flat `__init__` (§11):

| backends | task hooks | matrainer base |
|---|---|---|
| lightning · torch · fastai | `RestorationTorchHooks` | `LightningRunnable` · `TorchLoopRunnable` · `FastaiRunnable` |
| keras · tensorflow | `RestorationKerasHooks` | `KerasRunnable` · `TensorFlowRunnable` |
| huggingface · jax · mlx | `RestorationBase` | `HuggingFaceRunnable` · `JaxLoopRunnable` · `MlxLoopRunnable` |

Other modules:
- `restoration_base.py` is torch-free: split preparation, the trainable-model gate, `data_range`.
- `restoration_torch.py` / `restoration_keras.py`: the per-family hooks. Not to be confused with
  `torch_restoration.py` / `keras_restoration.py`, which register the losses and metrics.
- Zoo wrappers: `deepinv_model.py` (`DeepInvModel`, 14 `arch:` values, learned and classical)
  and `huggingface_model.py` (`HuggingFaceRestorationModel`, Swin2SR).
- `dncnn/`: `spec.py` plus five implementations of one 57,635-parameter DnCNN.
- `_keras.py` / `_tensorflow.py`: Keras import shims. `cli.py`: the `FRAMEWORKS` table only, on
  `matrainer.cli`. `__init__.py`: lazy exports.

Batches are record dicts: `input_key="image"` is the degraded input, `target_key="target"` the
clean ground truth. Example config: `examples/train_dncnn_denoising.yaml`. Sweep:
`examples/sweep/`.

Read a score against the do-nothing baseline: the noisy input alone scores about 20.5 dB PSNR on
the example. A first epoch far from that floor means the pair is wrong, not that the model is
good.

**Who owns what:**
- recordstream: records, the ops that build the pair, collate, batch readers,
  `RestorationOutput`, source warming.
- matrainer: trainer machinery, tracking, metric naming and merging, the squared-error
  accumulator, the CLI factory, the sweep driver.
- distilaide: the restoration hooks, models, the DnCNN spec, registrations and configs.

Never add a collate, a noise op, a batch reader or a CLI helper here.

## Backends

- Keep each concrete backend's FULL flat `__init__`, because introspection reads it.
- Extend through the hooks. Never shadow the machinery (`initialize`, `fit`, `_build_trainer`,
  `_log_run_context`, `_callbacks`, `_epoch_metrics`, `_run_epoch`, `_live_trackers`,
  `_run_eval`, `_report_epoch_metrics`), and never derive hyperparameters locally. A machinery
  change goes into matrainer, with every consumer updated. (fleet source scans in
  `tests/test_restorers.py`) §11.
- A new backend is added to `FRAMEWORKS`, an extra, `_BACKEND_NAMES`, an entry point, and the
  fleet tables in `tests/test_restorers.py` and `tests/test_sweep_config.py::MATRIX`; the tables
  are checked against `*_restorer.py` on disk.
- `task: fit` over a parameter-free model is refused by name on every backend
  (`RestorationBase._require_a_trainable_model`); the same model is accepted for evaluation. The
  gate skips a `parameters()` that is not torch's.
  (`::test_every_backend_refuses_to_fit_a_parameter_free_model`,
  `::test_the_trainable_gate_skips_a_non_torch_parameters_contract`) §3.
- `TorchRestorer` imports no training framework, not even via `matrainer.lightning`
  (`::test_it_needs_no_training_framework`). It keeps `device`, `checkpoint_dir`, `grad_clip`
  and a `history` output; no callback protocol, no distributed strategy. §5.
- `fastai`: metrics are names. `psnr` is the one configured alias
  (`fastai_restorer.PeakSignalNoiseRatio`), tried before `fastai.metrics`: it accumulates the
  squared error, converts once, reports as `psnr`, and calls `.detach().cpu()` before matrainer's
  numpy accumulator (`tests/test_fastai_psnr.py`) §7. The default loss stays plain
  `torch.nn.L1Loss`.
- `jax`: `"l1"` is resolved in `JaxRestorer._loss_fn` because optax has none; every other name
  goes to `optax.losses`. `mlx` uses the upstream `l1_loss`. Losses and optimizers are names on
  both; never add a wrapper class. *(unpinned)*
- `huggingface`: a `Trainer` subclass applies `loss_fn`, because Swin2SR refuses `labels=`, and
  `registrations` is the torch loss module. The model `forward` accepts `images` and
  `pixel_values` and declares `labels` only to ignore it; never pass `labels=` to the model. No
  `train_metrics` slot. §8.

## Config and CLI

All pinned in `tests/test_cli_frameworks.py`.

- The backend is a flag: `--framework X` → `!scope:framework=X`. No backend subcommands, and no
  CLI machinery in `cli.py` (it lives in `matrainer.cli`). The verbs `train`/`evaluate`/`predict`
  set the task `fit`/`test`/`predict` and win over the YAML's `task:`.
- `--framework tensorflow` setdefaults `KERAS_BACKEND=tensorflow` BEFORE the registration
  import; no other framework touches it. `tensorflow` and `keras` share
  `distilaide.keras_restoration`; there is no second Keras loss module.
- `runnable`, `model`, `metrics` and `preprocess` are TOP-LEVEL siblings. `runnable:` sits inside
  each `!scope:framework=` block, and `default_scopes: [framework=lightning]` picks the default.
  Never nest `model:` in `runnable:`.
- A non-torch block sets `val_metrics: null` and never inherits the torchmetrics mapping. The
  fastai block restates `metrics:` as names.
- Every metric is a `!lazy:` template, never an eager `!class:...()`, so `data_range` can be
  injected; a template that sets its own `data_range` keeps it
  (`tests/test_restorers.py::test_a_template_that_pins_its_own_data_range_keeps_it`) §2.
- MS-SSIM is validation-only and is why the example raster is 192 px (five scales need more than
  160 px). `betas` cannot be trimmed from YAML: torchmetrics wants a tuple.

## The pair

- The target is built in the config: resize (`ConvertToImage`) → `CopyField(src=image,
  dst=target)` → degrade the input. Anything that must reach both halves goes ABOVE the copy;
  anything that degrades only the input goes BELOW it.
  (`tests/test_examples_load.py::test_a_resize_after_the_copy_would_unpair_the_record`) §1.
- Both halves end in `[0, 1]`, so `data_range: 1.0` is true on every engine. `ToTensor` only
  reorders to CHW. The torch chain puts `recordstream.ops.numpy.Scale {}` before its two
  `ToTensor` steps; a channels-last chain puts `albumentations.Normalize` (`std: 1`,
  `max_pixel_value: 255`) above the copy. (`::test_both_halves_land_in_the_same_value_range`,
  torch chain only) §1.
- Every `GaussNoise` sits inside a one-element `albumentations.Compose` with `seed:`, the only
  place albumentations accepts a seed. A model wrapper's `seed` seeds torch at build time and
  never restores the RNG. (`::TestTheConfigIsSeeded`) §10.

## Registrations and discovery

- Register upstream classes; never reimplement one. This project owns exactly two metric
  classes: the Keras PSNR shim (`keras_restoration.PeakSignalNoiseRatio`, which holds an upstream
  `MeanSquaredError` and converts once in `result()`; *unpinned*) and the fastai alias. SSIM
  stays out of the Keras family. §7.
- One registration module per engine family: `torch_restoration.py`, `keras_restoration.py`;
  jax and mlx register nothing. Every registered class is in the literal `__all__`
  (`tests/test_categories.py`). Never register `LearnedPerceptualImagePatchSimilarity` (it
  downloads weights) or the no-reference and distribution metrics.
- Every registering module gets its `confluid.configurables` entry point in the SAME change.
  Entries are modules, never `module:attr`. Then `uv pip install -e ./distilaide --no-deps`.
  (`tests/test_entry_points.py`)
- Tags: `task="restoration"`, `role` = `trainer`/`model`/`loss`/`metric`, `framework` = the API
  the class is wired to: `torch` (torch family, both zoo wrappers), `keras` (keras and
  tensorflow), `jax`, `mlx`. One class per backend serves evaluate/test/predict through its
  `@entrypoint` methods; no `role="evaluator"` class. (`tests/test_categories.py`)
- `run()` is matrainer's and dispatches through `run_entrypoint`. Never override it with a
  `{task: method}` dict (`::test_no_backend_dispatches_through_a_hand_written_task_dict`).

## Slots, metrics, outputs

Pinned in `tests/test_restorers.py` unless noted.

- Every slot flowed at run time is `Partial[T]` (`Mandatory[Partial[T]]` when required): never
  `Any`, never `Union[X, Fluid]`. The `model` slot names its type (`Partial[nn.Module]`), never
  `Partial[Any]`. Dataset slots use `recordstream.RecordSource`. (`tests/test_lazy_slots.py`)
- `trackers` is a constructor param, `Optional[Union[Partial[Tracker],
  Iterable[Partial[Tracker]]]]`, stored verbatim. There is no `logger:` slot. §12.
- `loss_fn` (`loss` on fastai) and `metrics` default to `None`, meaning "not chosen"; a wired
  value is never replaced. The default loss is L1 on all eight backends; the torch default
  metrics are PSNR + SSIM.
- Combine per-split metrics only via `matrainer.metric_slots.merge_metric_specs`. Every backend
  takes `metric_names` and puts its metrics on the trackers.
- Accumulate a metric over the split and reduce once; never average PSNR per batch. A loop we
  own must show its accumulator. §4.
- Predictions are `RestorationOutput` (one key, `image`, in the input's value range), never a
  bare tensor.
- Sources are warmed in the parent by the inherited `initialize`, through
  `recordstream.prepare_record_dataset`: a forked worker that builds a lazy source can SIGSEGV
  with no traceback. Never shadow `initialize`, and never "fix" a worker crash with
  `num_workers: 0`.

## Models

- `DeepInvModel` (`tests/test_deepinv_model.py`) §3:
  - No work in `__init__`. It builds once into the public `restorer`, not `model` (the Lightning
    checkpoint strip would eat that name); `solidify()`, `backbone` and `forward` share the cache.
  - `arch` is a closed Literal; every member resolves to a real `deepinv.models` class.
  - One channel knob, translated by inspecting the target's signature; never a
    per-architecture table.
  - `sigma` is a model knob, passed positionally. `pretrained` defaults to `None` (from scratch).
- `model_refs` reads stored config only and never builds. It returns `[]` when nothing external
  was loaded and uses `file_model_ref` for a checkpoint. The zoo URI is
  `deepinv://<arch>?weights=<set>`, with no `url`. (`tests/test_model_refs.py`)
- `dncnn/` holds the spec implementations; zoo wrappers stay top-level `<library>_model.py`
  (`tests/test_dncnn_spec.py`, `tests/test_sweep_config.py`) §6:
  - Each implementation is `<framework>_dncnn.py` and reports the count `spec.py` derives, also
    under custom `depth`/`width`, excluding batch-norm running statistics.
  - All take `input_shape` as `(H, W, C)` and a `seed`.
  - `TorchDnCNN` serves all three torch backends; there is no `fastai_dncnn.py`.
  - MLX keeps the network in the public `net`; Flax wraps the lazy build in `nnx.data`.
  - `__init__.py` imports nothing. There is no size divisor.
  - `huggingface` gets no DnCNN; never teach the spec transformers' keyword names.

## Packaging, release, tests

- `pip install distilaide` installs no framework. Every backend name in `__init__.py` is lazy
  (`matrainer._exports.lazy_exports`) and names the smallest extra that supplies it; keep
  `_BACKEND_NAMES` in step with the extras. `bm3d` is its own extra. `torch_restoration.py`
  imports torch at module level on purpose. (`release.yml` smoke test)
- Releases go through the hand-written `release.yml` (never generated) from a private repo.
  - Workspace specs on distilaide read `>=0.1.0a1` (aisland
    `tests/test_workspace_dependency_specs.py`).
  - README links are absolute GitHub URLs (`tests/test_docs_links.py`).
  - The newest CHANGELOG entry is the declared version; an `[Unreleased]` section above it is
    allowed (`tests/test_packaging.py`).
  - Dependency floors are what a clean, PyPI-only venv runs against.
- CI runs `mypy .` here with only its own extras. List every extra-gated import in
  `[[tool.mypy.overrides]]`; `warn_unused_ignores` fails a dead `# type: ignore` the root run
  passes. Reproduce CI in a Python 3.12 venv with CI's install line before pushing.
- Fleet rules are source or config scans, so they run without every framework installed. Write
  new fleet checks the same way.

## Sweep

- `examples/sweep/sweep.yaml` is the 11-cell matrix and `all_backends.yaml` its overlay. The
  driver is `matrainer sweep backends|trackers|all examples/sweep/sweep.yaml`.
  (`tests/test_sweep_config.py`) §9.
- In the overlay:
  - Restate a framework block in full, because `include:` does not deep-merge scope blocks.
  - The fastai and huggingface metric lists are runnable constructor kwargs, not sibling keys:
    a top-level key only the overlay adds lands after the block and wins.
  - The `!scope:model=dncnn` block lives in the example. The non-torch blocks come after it and
    wire their own DnCNN; that order is pinned.
  - Tracker blocks use `!ref:experiment_name` / `!ref:run_name`; output goes under `./runs/`.
  - The shared columns are `psnr` + `mse`; SSIM and MS-SSIM are left out. Never ask an engine
    for a metric another cannot produce. A blank cell is a backend defect to fix, not a column
    to drop.
  - No row pins an engine, and `num_workers` stays above 0 so the fork path is exercised.
