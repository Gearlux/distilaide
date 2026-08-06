"""``DeepInvModel`` names the weight artifacts it loads (``matrainer.uri``)."""

from pathlib import Path

from matrainer.uri import model_refs

from distilaide import DeepInvModel


def test_the_weight_selection_is_the_reference() -> None:
    """deepinv routes published weights through three different places depending on the
    architecture — a Hugging Face repo under ``deepinv/`` for dncnn / drunet / scunet / srresnet /
    restormer, a GitHub release asset for swinir, and nothing at all for ffdnet (which raises
    *"has no downloadable weights"*) — so the honest identifier is the pair deepinv itself was
    asked for, not a repository URL that would be wrong for most of the zoo."""
    refs = model_refs(DeepInvModel(arch="dncnn", pretrained="download"))
    assert [r.uri for r in refs] == ["deepinv://dncnn?weights=download"]
    assert refs[0].url is None  # no single repository to point at; none is invented
    assert refs[0].role == "pretrained"


def test_the_architecture_appears_in_the_uri() -> None:
    refs = model_refs(DeepInvModel(arch="drunet", pretrained="download"))
    assert refs[0].uri == "deepinv://drunet?weights=download"


def test_a_checkpoint_is_a_second_entry_with_its_own_role(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch=3.ckpt"
    refs = model_refs(DeepInvModel(pretrained="download", checkpoint_path=str(checkpoint)))
    assert [r.role for r in refs] == ["pretrained", "checkpoint"]
    assert refs[1].uri == checkpoint.resolve().as_uri()


def test_a_model_trained_from_scratch_reports_nothing() -> None:
    """This project's DEFAULT: random weights referenced no artifact, so naming one would name
    something nobody loaded."""
    assert model_refs(DeepInvModel(arch="dncnn")) == []


def test_asking_does_not_build_the_model() -> None:
    """Provenance must be readable from a wrapper the restorer has not solidified — the property
    is asked of models that may never run."""
    model = DeepInvModel(arch="dncnn", pretrained="download")
    assert model_refs(model)
    # The lazy cache is `restorer`, deliberately NOT `model` (see the class docstring on how the
    # Lightning `model.` prefix strip would chew through a slot of that name).
    assert model.restorer is None
