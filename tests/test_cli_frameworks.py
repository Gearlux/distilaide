"""The framework axis — the CLI flag, the FRAMEWORKS table, and the example config's blocks.

These are CONFIG-level on purpose: they load the YAML and inspect what each ``--framework`` value
resolves to, without needing that engine installed. So a mistake fails on the machine that makes
it rather than only on one that happens to have the extra.

The MACHINERY (the rewrite, the keras engine pin, the dispatch, the pre-step ORDER) is
`matrainer.cli`'s since 2026-08-07 and is pinned in matrainer's own `tests/test_cli.py`; what
stays here is distilaide's half of each seam — the flag against THIS project's spellings, the
pin against THIS project's tensorflow row, and the table itself.
"""

import inspect
import os
from pathlib import Path
from typing import Any, Dict, List

import confluid
import pytest
import yaml
from matrainer.cli import VERB_TASKS, normalize_framework_flag, pin_keras_backend, requested_framework, trainer_main

from distilaide.cli import FRAMEWORKS, main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
CONFIG = EXAMPLES / "train_dncnn_denoising.yaml"


def _raw_document() -> Dict[str, Any]:
    """The example config parsed with tags IGNORED — enough to inspect its block structure.

    Loading it through confluid would resolve ``!class:`` targets and need every backend's
    framework installed; this test is about the SHAPE of the document.
    """

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor("!", lambda loader, suffix, node: {"__tag__": suffix, **_plain(loader, node)})
    document: Dict[str, Any] = yaml.load(CONFIG.read_text(), Loader=_Loader)  # noqa: S506 - restricted loader
    return document


def _plain(loader: Any, node: Any) -> Dict[str, Any]:
    """A tagged node's own mapping, or an empty one for a scalar/sequence tag."""
    if isinstance(node, yaml.MappingNode):
        return dict(loader.construct_mapping(node, deep=True))
    return {}


def _framework_blocks() -> Dict[str, Dict[str, Any]]:
    """The ``!scope:framework=<name>`` blocks the example config declares, by name."""
    blocks = {}
    for key, value in _raw_document().items():
        if isinstance(value, dict) and str(value.get("__tag__", "")).startswith("scope:framework="):
            blocks[value["__tag__"].split("=", 1)[1]] = value
    return blocks


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def test_every_advertised_framework_is_real() -> None:
    """``FRAMEWORKS`` is a promise: every name in it must have a module behind it.

    An unbuilt backend is simply ABSENT from the table — there is nowhere for a stub to hang once
    the backend is a flag rather than a subcommand.
    """
    source_dir = Path(__file__).resolve().parent.parent / "distilaide"
    for name in FRAMEWORKS:
        assert (source_dir / f"{name}_restorer.py").exists(), f"--framework {name} has no backend module"


def test_the_backend_table_is_the_whole_fleet() -> None:
    """Every ``*_restorer.py`` on disk is advertised, and nothing is advertised that is not there.

    The check that stops a ninth backend being written and silently unreachable, and the converse
    — an entry whose module was deleted or renamed.
    """
    source_dir = Path(__file__).resolve().parent.parent / "distilaide"
    on_disk = {path.stem.removesuffix("_restorer") for path in source_dir.glob("*_restorer.py")}
    assert on_disk == set(FRAMEWORKS), f"table {sorted(FRAMEWORKS)} vs disk {sorted(on_disk)}"


def test_the_torch_FAMILY_shares_one_loss_module() -> None:
    """A ``torch.nn`` loss does not care whether a Trainer, a hand-written loop or a fastai Learner
    calls it — so pointing all three at one registration module is the point rather than an economy.

    ``huggingface`` joins them, and that is this task's oddity: its siblings in the other projects
    register NOTHING (their models compute their own loss from ``labels=``), but Swin2SR raises
    ``NotImplementedError`` when handed one, so the loss comes from outside — and the loop is torch.
    """
    torch_family = {"lightning", "torch", "fastai", "huggingface"}
    assert {FRAMEWORKS[name].registrations for name in torch_family} == {"distilaide.torch_restoration"}


def test_the_keras_family_shares_the_keras_loss_module() -> None:
    """In TF >= 2.16 ``tf.keras`` IS Keras 3, so a second ``tensorflow_restoration.py`` would
    register the same classes under a second tag and make every bare ``!class:MeanAbsoluteError``
    ambiguous for no gain."""
    assert {FRAMEWORKS[name].registrations for name in ("keras", "tensorflow")} == {"distilaide.keras_restoration"}


def test_jax_and_mlx_register_nothing_and_that_is_a_real_state() -> None:
    """optax and ``mlx.nn.losses`` ship plain FUNCTIONS, so a loss is resolved by NAME and there is
    no class for confluid's registry to tag. ``None`` is the honest answer, not an oversight."""
    assert FRAMEWORKS["jax"].registrations is None
    assert FRAMEWORKS["mlx"].registrations is None


def test_the_verbs_map_onto_the_runnable_tasks() -> None:
    assert VERB_TASKS == {"train": "fit", "evaluate": "test", "predict": "predict"}


# --------------------------------------------------------------------------- #
# The flag
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "argv,expected",
    [
        (["distilaide", "train", "c.yaml", "--framework", "torch"], "torch"),
        (["distilaide", "train", "c.yaml", "--framework=torch"], "torch"),
        (["distilaide", "train", "c.yaml", "--scope", "framework=torch"], "torch"),
        (["distilaide", "train", "c.yaml"], None),
    ],
)
def test_the_requested_framework_is_read_in_every_spelling(argv: List[str], expected: Any) -> None:
    """Read from ``argv`` because the caller needs it BEFORE the config is loaded — the eager
    ``!class:Foo()`` instance form is built during load, earlier than any handler runs."""
    assert requested_framework(argv) == expected


def test_the_flag_is_rewritten_never_appended() -> None:
    """liquifai binds ``--KEY VAL`` to scope activation only for dimensions the config DECLARES;
    against a config with no such block a bare ``--framework X`` is read as a config OVERRIDE."""
    argv = ["distilaide", "train", "c.yaml", "--framework", "torch"]
    normalize_framework_flag(argv)
    assert argv == ["distilaide", "train", "c.yaml", "--scope", "framework=torch"]
    assert "--framework" not in argv


def test_an_explicit_scope_spelling_is_left_alone() -> None:
    argv = ["distilaide", "train", "c.yaml", "--scope", "framework=torch"]
    normalize_framework_flag(argv)
    assert argv == ["distilaide", "train", "c.yaml", "--scope", "framework=torch"]


# --------------------------------------------------------------------------- #
# The example config declares a block per backend
# --------------------------------------------------------------------------- #
def test_every_framework_the_cli_advertises_has_a_block_in_the_example() -> None:
    """``--framework <name>`` must reach something. A name the config does not declare is an
    ERROR from confluid listing what it does — which is the check a backend SUBCOMMAND could
    never do, because a verb never selected anything."""
    assert set(_framework_blocks()) == set(FRAMEWORKS)


def test_the_default_backend_is_a_block_named_by_default_scopes() -> None:
    """Every backend is a `!scope:framework=<name>` block and `default_scopes:` names the one a
    bare run uses — no top-level `runnable:` is privileged; confluid's declared-value check binds
    the default too."""
    assert "lightning" in _framework_blocks()
    assert _raw_document()["default_scopes"] == ["framework=lightning"]
    bare = confluid.load(CONFIG, until="document")
    assert bare["runnable"].target == "distilaide.LightningRestorer"


def test_every_block_names_a_runnable() -> None:
    """A scope block REPLACES a key and takes its whole subtree, so a block that forgets the
    runnable trains the previous backend under the new name — silently."""
    for name, block in _framework_blocks().items():
        assert "runnable" in block, f"the {name!r} block declares no runnable"


def test_the_framework_specific_keys_are_top_level_siblings() -> None:
    """Flatness is load-bearing: a ``model:`` nested INSIDE ``runnable:`` VANISHES the moment a
    block swaps the runnable, and the run then dies on a missing required argument."""
    document = _raw_document()
    assert "model" in document, "the shared model is a TOP-LEVEL key every backend block inherits"
    assert "runnable" not in document, "the runnable lives in each framework block — `default_scopes:` picks one"
    for name, block in _framework_blocks().items():
        runnable = block["runnable"]
        if isinstance(runnable, dict):
            assert "model" not in runnable, f"{name!r} nests the model INSIDE its runnable — keep it a sibling"


def test_the_fastai_block_restates_its_metrics() -> None:
    """Not style: the shared ``metrics:`` key holds torchmetrics TEMPLATES, which a fastai
    ``Learner`` cannot consume — and fastai ships no PSNR or SSIM to inherit anyway."""
    fastai_block = _framework_blocks()["fastai"]
    assert "metrics" in fastai_block
    assert all(isinstance(entry, str) for entry in fastai_block["metrics"])


def test_no_framework_inherits_a_metric_set_its_engine_cannot_consume() -> None:
    """The converse of the check above, over every block."""
    document = _raw_document()
    shared_is_torchmetrics = "val_metrics" in document
    assert shared_is_torchmetrics
    for name, block in _framework_blocks().items():
        if name == "fastai":
            continue
        assert "metrics" not in block or not all(isinstance(e, str) for e in block.get("metrics", []))


class TestTheKerasEngineIsPinnedForTensorFlow:
    """`--framework tensorflow` must bind Keras to TensorFlow before anything imports keras.

    Found by the backend sweep, which is what put this backend through this CLI for the first time
    since it grew its registration step: `distilaide train --framework tensorflow` died on a default
    checkout with *"needs Keras bound to the TensorFlow engine, but KERAS_BACKEND is 'torch'"*. The
    fleet's own tensorflow number was taken with the variable already exported, so nothing caught it.
    """

    def test_the_tensorflow_backend_pins_the_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keras 3 reads `KERAS_BACKEND` ONCE, at import, and the default here is the wrong one.

        `recordstream.keras` defaults it to the first engine INSTALLED, which is torch on any
        machine that also carries `distilaide[lightning]` — and a torch-backed `keras.Model` cannot
        be trained with `tf.GradientTape`.
        """
        monkeypatch.delenv("KERAS_BACKEND", raising=False)
        pin_keras_backend(requested_framework(["distilaide", "train", "c.yaml", "--framework", "tensorflow"]))

        assert os.environ.get("KERAS_BACKEND") == "tensorflow"

    def test_an_explicit_keras_backend_is_never_overridden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`setdefault`, so a deliberate choice survives and the restorer reports the mismatch."""
        monkeypatch.setenv("KERAS_BACKEND", "jax")
        pin_keras_backend("tensorflow")

        assert os.environ["KERAS_BACKEND"] == "jax"

    @pytest.mark.parametrize("framework", ["lightning", "torch", "keras", "fastai", "jax", "mlx", "huggingface", None])
    def test_no_other_framework_touches_it(self, monkeypatch: pytest.MonkeyPatch, framework: Any) -> None:
        """`keras` included: it runs on whichever engine is loaded, which is the whole point of it."""
        monkeypatch.delenv("KERAS_BACKEND", raising=False)
        pin_keras_backend(framework)

        assert "KERAS_BACKEND" not in os.environ

    def test_the_pin_runs_BEFORE_the_registration_import(self) -> None:
        """Order is the whole fix: the registration import is what binds the wrong engine.

        `FRAMEWORKS["tensorflow"].registrations` names `distilaide.keras_restoration` (there is
        deliberately no second loss module), so registering imports `distilaide._keras` ->
        `recordstream.keras` and the torch default is bound before `distilaide._tensorflow` can
        setdefault its own. Read from the SOURCE of `main()` rather than by calling it — calling it
        would run the app. The order now lives in the shared `trainer_main` (matrainer pins
        it too); what THIS test keeps is the project's stake in it — `main()` must delegate
        there, and the shared body must still pin before it registers.
        """
        assert "trainer_main" in inspect.getsource(main), "main() must run the shared pre-step order"
        body = inspect.getsource(trainer_main).split('"' * 3)[-1]

        assert body.index("pin_keras_backend") < body.index("register_frameworks")

    def test_the_engine_the_pin_names_is_the_one_the_registration_module_would_bind_wrong(self) -> None:
        """The premise, so a future edit that gives tensorflow its own loss module fails here."""
        assert FRAMEWORKS["tensorflow"].registrations == FRAMEWORKS["keras"].registrations
