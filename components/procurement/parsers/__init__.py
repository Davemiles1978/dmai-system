"""Vendor-specific HTML parser stubs for the procurement skill (PR K).

Each module in this package is a *judge-accepted stub*: it carries a
module-level :data:`SEED_CAPABILITY_METADATA` dict shaped for the seed
capability promoter / PR H materialiser, a ``parse(html) -> list[dict]``
function that raises ``NotImplementedError`` (its docstring is the
generation brief the materialiser hands to the LLM), and a hand-written
``seed_fallback() -> list[dict]`` returning 2-3 known real rows so the
research pipeline works from day 1 without waiting for the LLM.

The researcher calls ``parse(html)`` first and falls back to
``seed_fallback()`` when the body is still a stub.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, List

__all__ = ["load_source_rows", "REQUIRED_FIELDS"]

# Minimum fields every parser row must provide before normalisation.
REQUIRED_FIELDS = ("name", "cpu")


def load_source_rows(module_path: str, html: str = "") -> List[Dict[str, Any]]:
    """Return rows for a source module.

    Tries the materialised ``parse(html)`` first; on ``NotImplementedError``
    (stub body not yet generated) falls back to ``seed_fallback()``.
    """
    mod = import_module(module_path)
    try:
        rows = mod.parse(html)  # type: ignore[attr-defined]
        if rows:
            return list(rows)
        return list(mod.seed_fallback())  # type: ignore[attr-defined]
    except NotImplementedError:
        return list(mod.seed_fallback())  # type: ignore[attr-defined]
