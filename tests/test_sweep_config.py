"""``examples/sweep/all_backends.yaml`` — the overlay that makes eleven runs comparable.

The sweep is an OVERLAY: `include:` deep-merges the neighbouring example under it and it overrides
only what a sweep needs. Three things about that composition are easy to get wrong and silent when
wrong, and all three are checked here rather than discovered eleven training runs later.

**A scope block REPLACES, it does not merge.** `include:` deep-merges two *mappings*, and a
`!scope:` block is a `ScopeBlock` object rather than a dict — so an overlay block with a key the
included file already uses discards that block wholesale. A `tensorflow:` block holding only
`export_dir` would drop the runnable, the model and the channels-LAST preprocess chain, and
`--framework tensorflow` would train a `LightningRestorer` over CHW tensors under the right name
with nothing raising.

**A deep-merged key keeps its ORIGINAL POSITION, so a new top-level key wins over it.** The
overlay's `metrics:` is new here and lands at the end of the merged document, while `fastai:` is a
key the example already has and stays where the example put it — early. As a sibling key inside that
block, `metrics:` would be silently overwritten by the top-level one. An explicit CONSTRUCTOR kwarg
is immune.

**A metric set must reach only the engines that can consume it.** The mechanism is confluid's
broadcast gate, which classifies by the RECEIVING slot's annotation — so the check below is on the
annotation, read statically.

These are CONFIG-level on purpose: no machine need carry all eight frameworks, so a mistake fails
where it is made rather than only where the extra happens to be installed.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import confluid
import pytest

SWEEP = Path(__file__).resolve().parent.parent / "examples" / "sweep" / "all_backends.yaml"

#: (backend, variant) -> (runnable class, model class). The matrix the driver script walks.
#:
#: A variant an engine structurally cannot have is simply ABSENT, and the two gaps have different
#: reasons. There is no non-torch restoration ZOO to wrap (deepinv is torch-only, which is why the
#: shared spec exists). And `huggingface` takes no dncnn for a NARROWER reason than the per-pixel
#: project's: the loss is not the obstacle here — this backend supplies its own — but a `Trainer`'s
#: EVALUATION path calls `model(**inputs)`, and `TorchDnCNN.forward(images)` accepts no
#: `pixel_values=` keyword. Measured: it trains, then dies on the first eval.
MATRIX: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("lightning", "zoo"): ("LightningRestorer", "DeepInvModel"),
    ("lightning", "dncnn"): ("LightningRestorer", "TorchDnCNN"),
    ("torch", "zoo"): ("TorchRestorer", "DeepInvModel"),
    ("torch", "dncnn"): ("TorchRestorer", "TorchDnCNN"),
    ("fastai", "zoo"): ("FastaiRestorer", "DeepInvModel"),
    ("fastai", "dncnn"): ("FastaiRestorer", "TorchDnCNN"),
    ("keras", "dncnn"): ("KerasRestorer", "KerasDnCNN"),
    ("tensorflow", "dncnn"): ("TensorFlowRestorer", "TensorFlowDnCNN"),
    ("huggingface", "zoo"): ("HuggingFaceRestorer", "HuggingFaceRestorationModel"),
    ("jax", "dncnn"): ("JaxRestorer", "FlaxDnCNN"),
    ("mlx", "dncnn"): ("MlxRestorer", "MlxDnCNN"),
}

TRACKERS = {
    "mlflow": "MlflowTracker",
    "tensorboard": "TensorBoardTracker",
    "csv": "CsvTracker",
    "aim": "AimTracker",
    "clearml": "ClearMLTracker",
    "wandb": "WandbTracker",
}

#: The engines whose metrics are NAMES or their own engine's objects, so the shared torchmetrics
#: mapping must not reach them.
NON_TORCH = ("keras", "tensorflow", "fastai", "jax", "mlx", "huggingface")


def resolve(framework: str, variant: str = "zoo", tracker: str = "mlflow") -> Dict[str, Any]:
    scopes = [f"framework={framework}", f"tracker={tracker}"]
    if variant == "dncnn":
        scopes.append("model=dncnn")
    document = confluid.load(confluid.load_config(SWEEP), flow=False, scopes=scopes)
    assert isinstance(document, dict)
    return document


def target_name(value: Any) -> Optional[str]:
    """The class name behind a `!class:` / `!lazy:` marker, without importing it."""
    target = getattr(value, "target", None)
    if target is None:
        return None
    return str(target).rsplit(".", 1)[-1]


def constructor_annotation(module: str, parameter: str) -> str:
    """The source text of one `__init__` parameter's annotation, read STATICALLY.

    `ast` rather than `typing.get_type_hints`: resolving a hint imports the module, and a module
    whose framework is not installed here is exactly the one whose annotation nobody would check.
    """
    source = (Path(__file__).resolve().parent.parent / "distilaide" / module).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for argument in [*node.args.args, *node.args.kwonlyargs]:
                if argument.arg == parameter and argument.annotation is not None:
                    return ast.unparse(argument.annotation)
    raise AssertionError(f"{module} declares no `{parameter}` constructor parameter")


class TestTheMatrix:
    """Eleven cells, each resolving to its own runnable and model."""

    @pytest.mark.parametrize("cell", sorted(MATRIX), ids=lambda c: f"{c[0]}-{c[1]}")
    def test_every_cell_resolves_to_its_own_runnable_and_model(self, cell: Tuple[str, str]) -> None:
        """The regression pin for the replace-not-merge trap described in the module docstring."""
        framework, variant = cell
        expected_runnable, expected_model = MATRIX[cell]
        document = resolve(framework, variant)

        assert target_name(document["runnable"]) == expected_runnable
        assert target_name(document.get("model")) == expected_model

    def test_the_matrix_covers_every_backend_the_cli_advertises(self) -> None:
        """A ninth backend must not slip into the CLI without a row in the sweep.

        The sweep is how the claim "one config, every backend" is EXECUTED, so a backend the CLI
        offers and the sweep never runs is a claim nothing checks.
        """
        from distilaide.cli import FRAMEWORKS

        assert {backend for backend, _ in MATRIX} == set(FRAMEWORKS)

    def test_the_sweep_spec_declares_exactly_this_matrix(self) -> None:
        """The declarative spec (`matrainer sweep`'s input) and this table must not drift — a
        cell in one and not the other is either a run nobody checks or a check nothing runs."""
        from matrainer.sweep import load_spec

        spec = load_spec(SWEEP.parent / "sweep.yaml")
        default = spec.models[0].name if spec.models else None
        declared = {
            (row.name, model) for row in spec.frameworks for model in (row.models or ([default] if default else []))
        }
        assert declared == set(MATRIX)

    def test_the_torch_backed_backends_share_one_implementation_of_the_spec(self) -> None:
        """Three backends, one `TorchDnCNN` — the example's top-level block, inherited unchanged.

        If one of them ever resolves to something else, a per-backend block has been added where
        the shared one was doing the job, which is the duplication `dncnn/spec.py` exists to avoid.
        """
        names = {target_name(resolve(fw, "dncnn")["model"]) for fw in ("lightning", "torch", "fastai")}

        assert names == {"TorchDnCNN"}

    def test_each_non_torch_engine_gets_its_own_implementation_of_the_spec(self) -> None:
        """The point of the sweep: the SAME architecture, in each framework's own module type.

        Nothing in the overlay arranges this — each framework block wires its implementation
        unconditionally and is spliced AFTER the example's top-level `dncnn:` block, so the block
        that gives the torch backends `TorchDnCNN` is simply overridden. That is a property of
        document ORDER, which is exactly the kind of thing an edit can reverse silently, so it is
        pinned here: without it, `--framework mlx --model dncnn` would train a torch model under
        the MLX name.
        """
        resolved = {fw: target_name(resolve(fw, "dncnn")["model"]) for fw in ("keras", "tensorflow", "jax", "mlx")}

        assert resolved == {
            "keras": "KerasDnCNN",
            "tensorflow": "TensorFlowDnCNN",
            "jax": "FlaxDnCNN",
            "mlx": "MlxDnCNN",
        }

    def test_the_huggingface_row_keeps_its_own_model_under_the_spec_scope(self) -> None:
        """The gap is made UNREACHABLE rather than reachable-and-broken.

        `TorchDnCNN` trains fine under a `Trainer` — this backend supplies the loss itself — and
        then dies on the first EVAL, because `Trainer.prediction_step` calls `model(**inputs)` and
        the shared spec takes no `pixel_values=` keyword (measured). Since the huggingface block is
        spliced after the top-level `dncnn:` one, `--model dncnn` simply leaves that row alone,
        which is what keeps the failure impossible instead of merely undocumented.
        """
        assert target_name(resolve("huggingface", "dncnn")["model"]) == "HuggingFaceRestorationModel"


class TestTheTrackerDimension:
    """Where the results go — one block per tracker, named from the run."""

    @pytest.mark.parametrize("tracker", sorted(TRACKERS))
    def test_each_tracker_scope_selects_exactly_that_backend(self, tracker: str) -> None:
        trackers = resolve("lightning", tracker=tracker)["trackers"]

        assert [target_name(t) for t in trackers] == [TRACKERS[tracker]]

    @pytest.mark.parametrize("tracker", sorted(TRACKERS))
    def test_the_tracker_takes_its_names_from_the_run(self, tracker: str) -> None:
        """`!ref:` rather than a literal, so the ONE pair the driver sets reaches the tracker too.

        A literal here would drift from whatever the run is actually called — the one thing a sweep
        must not do, since the tracker is where the runs are compared.
        """
        wired = resolve("lightning", tracker=tracker)["trackers"][0]

        for key in ("experiment_name", "run_name"):
            assert type(wired.kwargs[key]).__name__ == "Reference", f"{tracker}: {key} is not a !ref:"

    def test_the_sweep_writes_everything_under_one_tree(self) -> None:
        """Eleven runs must not scatter exports and Trainer checkpoints across the project root.

        The tree is `runs/` — the ONE gitignored output root every project carries (workspace
        `AGENTS.md`, "Run Output Lands In A Gitignored `runs/`"). `sweep_logs/` keeps only the
        driver's per-cell `.log` files; everything a RUN produces belongs here.
        """
        exported = resolve("tensorflow", "dncnn")["runnable"].kwargs["export_dir"]
        hf_output = resolve("huggingface")["runnable"].kwargs["output_dir"]

        assert str(exported).startswith("./runs/")
        assert str(hf_output).startswith("./runs/")


class TestOneVocabulary:
    """The columns a reader compares, and the gate that keeps a metric set off the wrong engine."""

    def test_a_metric_set_never_reaches_an_engine_that_cannot_consume_it(self) -> None:
        """The shared `metrics:` is a torchmetrics TEMPLATE MAPPING, which only torch can consume.

        Unlike a per-engine `metrics: null`, what keeps it away from the other six is confluid's
        BROADCAST GATE: it classifies by the receiving slot's annotation, and every non-torch
        restorer declares `metrics: Optional[List[...]]`, which a mapping does not satisfy. So the
        check is on the ANNOTATION rather than on the document key — the key is present for every
        framework and that is fine.
        """
        document = resolve("lightning")
        assert isinstance(document["metrics"], dict), "the shared set must be a torchmetrics template MAPPING"
        assert set(document["metrics"]) == {"psnr", "mse"}

        for framework in NON_TORCH:
            declaration = constructor_annotation(f"{framework}_restorer.py", "metrics")
            assert declaration.startswith("Optional[List["), (
                f"{framework} declares `metrics: {declaration}` — a mapping-admitting annotation "
                f"would let the sweep's torchmetrics templates broadcast into an engine that "
                f"cannot build them"
            )

    def test_every_row_reports_psnr(self) -> None:
        """The one quantity all eleven runs share, and the reason the sweep is a comparison.

        The NAME is what a table lines up on, so this checks what each block ASKS FOR rather than
        the metric object. Seven engines produce the word `psnr` directly — it is the torch
        template's key here and a registry name in the keras / jax / mlx / huggingface backends —
        and fastai could not, until `_as_metric` grew the CONFIGURED ALIAS that supplies the
        accumulator `fastai.metrics` has no name for.
        """
        assert "psnr" in resolve("lightning")["metrics"]
        assert "psnr" in resolve("huggingface")["runnable"].kwargs["metrics"]
        assert "psnr" in resolve("fastai")["runnable"].kwargs["metrics"]

    def test_every_row_reports_mse_too(self) -> None:
        """The second full column, and the control: an error the metric's own definition is built
        on, so a row whose PSNR looks odd can be checked against it."""
        assert "mse" in resolve("lightning")["metrics"]
        assert "mse" in resolve("huggingface")["runnable"].kwargs["metrics"]
        assert "mse" in resolve("fastai")["runnable"].kwargs["metrics"]

    def test_ssim_is_deliberately_NOT_in_the_shared_set(self) -> None:
        """It can never be more than a 2-of-11 column: `keras.ops` ships no SSIM and the only
        implementations are engine-SPECIFIC, so wiring one would make `--framework tensorflow`
        compute a metric `--framework keras` cannot from the same config.

        The example's own default pair IS PSNR + SSIM, so this is the overlay doing work rather
        than inheriting — which is exactly why it is worth a test.
        """
        assert "ssim" not in resolve("lightning")["metrics"]

    def test_the_expensive_validation_metric_is_dropped_for_the_sweep(self) -> None:
        """MS-SSIM is the example's validation-only extra and no non-torch engine has it, so it is
        another 2-of-11 column — and the costly one, an image pyramid per pair.

        `null` restores the SENTINEL meaning "not chosen", which is NOT the same as an empty list:
        an empty list would say the user chose no metrics.
        """
        assert resolve("lightning")["val_metrics"] is None

    def test_the_fastai_metrics_are_a_CONSTRUCTOR_kwarg_not_a_sibling_key(self) -> None:
        """The deep-merge position trap: a key only the overlay introduces lands at the END of the
        merged document, so the top-level `metrics:` mapping beats a sibling `metrics:` inside a
        block however far up the file that block is written. An explicit constructor kwarg is
        immune to document order.
        """
        block_metrics = resolve("fastai")["runnable"].kwargs["metrics"]

        assert block_metrics == ["psnr", "mse"]
        assert all(isinstance(entry, str) for entry in block_metrics), "a Learner consumes NAMES"

    def test_no_row_asks_for_a_metric_its_engine_cannot_produce(self) -> None:
        """The rule the sweep config keeps: ask only for what every engine here can compute.

        `psnr` and `mse` are it. A backend-specific metric is worth wiring when you want to look at
        THAT backend, and worth leaving out when you want to compare backends.
        """
        for framework in ("fastai", "huggingface"):
            named = resolve(framework)["runnable"].kwargs["metrics"]
            assert set(named) <= {"psnr", "mse"}, f"{framework} asks for something outside the shared pair"
