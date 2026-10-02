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

- [ ] **Check run-to-run reproducibility on keras / tensorflow / jax / mlx.** Every backend's
      model already carries a build-time `seed:` and every degradation is wrapped in a seeded
      `Compose` (`tests/test_examples_load.py::TestTheConfigIsSeeded`), but two identical runs
      were measured only on `--framework torch` (unseeded 22.7530 / 22.5188 dB, both seeded
      22.1886 / 22.1886). Each other engine seeds its own loop differently
      (`keras.utils.set_random_seed`, `mx.random.seed`, an explicit `nnx.Rngs`), so each needs its
      own two-identical-runs check. @medium

## Defects and drift

- [ ] **`distilaide evaluate examples/train_dncnn_denoising.yaml` (`README.md:8`) raises** @small @bug — `evaluate` runs the `test` task (`matrainer/matrainer/cli.py:72`, `VERB_TASKS`), which scores `test_set` and raises `ValueError: … .test() requires a 'test_set'.` when none is wired (`matrainer/matrainer/lightning/module.py:283-284`); `examples/train_dncnn_denoising.yaml` wires only `train_set` (`:112`) and `val_set` (`:119`). Read from the code, not run (a run downloads the dataset). Fix: wire a `test_set:` into the example (`deepinv/CBSD68`, which the README already recommends), or change the README command.
- [ ] **The ordering pins never read the example config** @small @tests — `tests/test_examples_load.py:31-43` (`_example_chain`) rebuilds the example's `preprocess` chain in Python, so `test_the_resize_happens_before_the_copy` and `test_both_halves_land_in_the_same_value_range` pass whatever `examples/train_dncnn_denoising.yaml` says: reordering the YAML, or moving a channels-last chain's `Normalize` below the copy, fails nothing, and the four channels-last chains are unpinned. Load the YAML's chains instead.
- [ ] **Five restorer modules restate the closed task set** @low @cleanup — `distilaide/{keras,tensorflow,jax,mlx,huggingface}_restorer.py` (lines 50, 55, 50, 44, 54) each write `RestorerTask = Literal["fit", "evaluate", "test", "predict"]`, while `lightning_restorer.py:41`, `torch_restorer.py:41` and `fastai_restorer.py:43` alias `recordstream.runnable.RunnableTask`: one closed set restated five times, against the workspace rule to derive it from one source.
