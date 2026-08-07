"""Every slot a runnable flows at run time must DECLARE itself deferred.

``Lazy[T]`` is what tells a deep-flow walker to leave a slot alone until the runnable flows it with
the kwarg that does not exist yet (``params=``, ``dataset=``). Annotating such a slot ``Any`` — or
hand-writing ``Union[X, Fluid]``, which is the ``Lazy`` alias minus its marker — declares nothing,
and nothing catches it: the class keeps working only because the slot happens to hold a
``LazyClass`` value.

This derives the expectation from the SOURCE rather than restating a list, so a slot added tomorrow
is covered without editing this file.
"""

import ast
import inspect
import re
from pathlib import Path
from typing import Dict, List, Set

from confluid import lazy_param_names

from distilaide.lightning_restorer import LightningRestorer

SOURCE_DIR = Path(__file__).resolve().parent.parent / "distilaide"


def _mro_source() -> str:
    """The runnable's WHOLE implementation: its module plus every base's, workspace-owned only.

    Since the 2026-08-06 consolidation the machinery lives in shared bases (matrainer's
    trainer API + this project's task bases), so the flowed-slot ↔ declared-lazy pairing must
    be checked over the union — a module-local scan would find nothing and pass vacuously.
    """
    chunks, seen = [], set()
    for klass in LightningRestorer.__mro__:
        module = inspect.getmodule(klass)
        if module is None or module.__name__ in seen:
            continue
        if not module.__name__.split(".")[0] in {"distilaide", "matrainer"}:
            continue
        seen.add(module.__name__)
        chunks.append(inspect.getsource(module))
    return "\n".join(chunks)


def test_every_flowed_slot_is_declared_lazy() -> None:
    source = _mro_source()
    flowed = set(re.findall(r"flow\(\s*self\.([a-z_]+)", source))
    assert flowed, "found no flow(self.<slot>) calls anywhere in the MRO — has the dispatch changed?"

    declared = _declared_lazy_slots(source) | lazy_param_names(LightningRestorer)
    undeclared = sorted(flowed - declared)

    assert not undeclared, (
        f"{undeclared} are flowed at run time but not annotated Lazy[T]. "
        "An auto-flow walker would build them eagerly — for a slot needing a runtime "
        "kwarg, that raises; for the rest it silently discards the deferral."
    )


def test_the_lazy_scan_sees_this_class_at_all() -> None:
    """Guard: the body scan reads ``__init__`` SOURCE, so a packaged deployment sees none.

    Without this, the assertion above would pass vacuously wherever the scan is blind.
    """
    assert lazy_param_names(LightningRestorer), "no lazy slots detected — the body scan may be blind here"


def test_no_restorer_declares_its_model_slot_as_lazy_any() -> None:
    """``Lazy[Any]`` is HALF a declaration — it defers, and says nothing about what fits.

    A reader cannot tell what the slot admits, mypy checks nothing through it, and ``to_pydantic``
    hands navigaitor / StreamStudio an open field. Read from the SOURCE with ``ast`` rather than by
    importing, because a backend may need a framework this machine does not have — which is
    exactly where a copy-pasted slot lands.
    """
    offenders = []
    for path in sorted(SOURCE_DIR.glob("*_restorer.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "__init__"):
                continue
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if arg.arg != "model" or arg.annotation is None:
                    continue
                text = ast.unparse(arg.annotation)
                if "Lazy[Any]" in text:
                    offenders.append(f"{path.name}: model: {text}")
    assert not offenders, f"name the type the loop requires, not Any: {offenders}"


def _declared_lazy_slots(source: str) -> Set[str]:
    """Slot names a module annotates ``Lazy[...]`` — as a ctor param OR an ``__init__`` body attr."""
    params = set(re.findall(r"\b([a-z_]+): (?:Mandatory\[)?(?:Optional\[)?Lazy\[", source))
    body = set(re.findall(r"self\.([a-z_]+): (?:Optional\[)?Lazy\[", source))
    return params | body


def test_every_backend_declares_the_slots_it_flows() -> None:
    """The same rule applied to EVERY backend by reading their SOURCE.

    ``lazy_param_names`` needs the class IMPORTED, and a backend may need a framework a default
    install lacks — so an import-based sweep would skip exactly where a copy-pasted slot lands. A
    regex over the module text covers every backend on every machine.
    """
    offenders: Dict[str, List[str]] = {}
    checked = 0
    for path in sorted(SOURCE_DIR.glob("*_restorer.py")):
        source = path.read_text()
        flowed = set(re.findall(r"flow\(\s*self\.([a-z_]+)", source))
        if not flowed:
            # A module with no flow calls must be one built on the shared bases (the
            # machinery — and its flows — live there; covered by the MRO-union test above).
            assert (
                "RestorationTorchHooks" in source
            ), f"{path.name}: no flow(self.<slot>) calls and no shared base — has the dispatch changed?"
            checked += 1
            continue

        undeclared = sorted(flowed - _declared_lazy_slots(source))
        if undeclared:
            offenders[path.name] = undeclared
        checked += 1

    assert checked >= 3, f"expected every backend to be scanned, saw {checked}"
    assert not offenders, (
        f"{offenders} are flowed at run time but not annotated Lazy[T]. "
        "An auto-flow walker would build them eagerly — for a slot needing a runtime kwarg "
        "(or a positional runtime arg) that raises; for the rest it silently discards the deferral."
    )
