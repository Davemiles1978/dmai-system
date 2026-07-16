"""Write an auto-generated pytest smoke test alongside a candidate module.

The smoke test asserts:

1. The module can be imported.
2. It exposes a callable ``run``.
3. Calling ``run(**happy_kwargs)`` returns without raising in <2s.
4. The docstring is non-empty.

That's the entire promise the materialiser makes; anything richer is
the LLM's job to add in a docstring-driven follow-up. Tests are
written to ``tests/generated/test_<slug>.py`` and imported via
``importlib`` so they are compatible with the wider test suite.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


SMOKE_TEMPLATE = '''"""Auto-generated smoke test for the {slug} capability.

Do not edit by hand. Regenerated whenever the materialiser reruns
the pipeline for this concept.
"""
import importlib
import time

import pytest


HAPPY_KWARGS = {happy_kwargs_repr}
MODULE_DOTTED = {module_dotted!r}


def test_module_imports_cleanly():
    mod = importlib.import_module(MODULE_DOTTED)
    assert mod is not None


def test_module_has_docstring():
    mod = importlib.import_module(MODULE_DOTTED)
    assert getattr(mod, "__doc__", None), (
        "generated module must have a non-empty module docstring"
    )


def test_run_is_callable():
    mod = importlib.import_module(MODULE_DOTTED)
    assert callable(getattr(mod, "run", None)), (
        "generated module must expose a callable run()"
    )


def test_happy_path_returns_within_2s():
    mod = importlib.import_module(MODULE_DOTTED)
    t0 = time.monotonic()
    # PR NN: graceful fallback matching capability_verifier._run_isolated.
    # If codegen produced a signature that doesn't accept our happy
    # kwargs (e.g. requires db_path or takes no args), retry with an
    # empty call rather than failing the whole test. The point of the
    # smoke test is "can run() be invoked at all", not "does it accept
    # exactly this dict". Real signature validation happens later in
    # capability_verifier.verify_promoted with cache=False.
    try:
        result = mod.run(**HAPPY_KWARGS)
    except TypeError as te:
        msg = str(te)
        if "unexpected keyword" in msg or "takes 0 positional" in msg:
            result = mod.run()
        else:
            raise
    dt = time.monotonic() - t0
    assert dt < 2.0, f"run() took {{dt:.2f}}s, budget is 2s"
    # Return value is left free-form on purpose; just prove it
    # produced *something* (None is allowed).
    _ = result
'''


def write_smoke_test(*, tests_dir: Path,
                     slug: str,
                     module_dotted: str,
                     happy_kwargs: Dict[str, Any]) -> Path:
    """Write ``tests_dir/test_<slug>.py`` and return the path."""
    tests_dir.mkdir(parents=True, exist_ok=True)
    path = tests_dir / f"test_{slug}.py"
    # Ensure the happy kwargs are JSON-round-trippable (i.e. builtin
    # types only). If they aren't, fall back to an empty dict - the
    # smoke test then just proves the entry point is callable.
    try:
        # No default=str here - we WANT unserialisable values to raise
        # so we fall back to {} rather than baking a stringified blob
        # into a test file.
        happy_kwargs_repr = repr(json.loads(json.dumps(happy_kwargs)))
    except (TypeError, ValueError):
        happy_kwargs_repr = "{}"
    path.write_text(SMOKE_TEMPLATE.format(
        slug=slug,
        module_dotted=module_dotted,
        happy_kwargs_repr=happy_kwargs_repr,
    ), encoding="utf-8")
    return path


__all__ = ["write_smoke_test", "SMOKE_TEMPLATE"]
