"""Regression tests for the AutonomousTrader once-per-instance table guard.

Before this fix, every public method call ran the CREATE TABLE IF NOT EXISTS
loop through the KeepOpenProxy — each pass briefly took the process-wide
write mutex on dmai_knowledge.db. Under any concurrent writer, that made
even a plain GET /api/trader/at-mode queue for up to the 30-second
write_mutex_timeout (observed: 36.7 s in production 2026-07-12).

These tests pin the new behaviour: `_ensure_tables` runs at most once per
instance, regardless of how many public methods you call.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import types
from unittest import mock

import pytest

# Skip if the full trader module can't import in the sandbox (heavy deps)
try:
    from components.wealth.autonomous_trader import AutonomousTrader
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False


pytestmark = pytest.mark.skipif(
    not _IMPORT_OK,
    reason="autonomous_trader failed to import in this sandbox",
)


class _StubTrader:
    """Minimal stand-in so AutonomousTrader.__init__ doesn't blow up on trader access."""

    conservative_pairs: list = []
    trading_pairs: list = []
    prediction_engine = None


def _make_trader(tmp_path):
    """Return an AutonomousTrader wired to a tmp DB. _ensure_tables runs in __init__."""
    db = str(tmp_path / "dmai_knowledge.db")
    # __init__ calls _init_db() and _ensure_state_row() which do their own creates;
    # we care about _ensure_tables specifically (called by public methods).
    return AutonomousTrader(db_path=db, trader=_StubTrader())


def test_ensure_tables_runs_at_most_once(tmp_path):
    """Multiple public method calls -> _ensure_tables invoked at most once.

    After the read-only fast-path was added, calls skip _ensure_tables
    entirely when the schema is already present (the common prod case).
    Force the readonly check to fail so we exercise the fallback and can
    prove the guard still bounds the call count.
    """
    at = _make_trader(tmp_path)
    at._tables_ensured = False
    with mock.patch.object(at, "_schema_ready_readonly", return_value=False):
        with mock.patch.object(
            at, "_ensure_tables", wraps=at._ensure_tables
        ) as spy:
            at._ensure_tables_once()
            at._ensure_tables_once()
            at._ensure_tables_once()
    assert spy.call_count == 1, (
        f"_ensure_tables should run at most once per instance, got {spy.call_count}"
    )
    assert at._tables_ensured is True


def test_ensure_tables_once_no_op_after_first(tmp_path):
    """Second call must not touch the DB at all (mocked _ensure_tables never invoked)."""
    at = _make_trader(tmp_path)
    at._tables_ensured = True  # simulate first-call already ran
    with mock.patch.object(at, "_ensure_tables") as spy:
        at._ensure_tables_once()
        at._ensure_tables_once()
    assert spy.call_count == 0


def test_public_methods_use_once_guard(tmp_path):
    """status/get_at_mode/set_at_mode should share the same guard.

    Calling all three should still result in a single _ensure_tables run.
    """
    at = _make_trader(tmp_path)
    at._tables_ensured = False
    with mock.patch.object(
        at, "_ensure_tables", wraps=at._ensure_tables
    ) as spy:
        # Call all the public methods that used to trigger _ensure_tables.
        try:
            at.status()
        except Exception:
            pass
        try:
            at.get_at_mode()
        except Exception:
            pass
        try:
            at.set_at_mode("paper")
        except Exception:
            pass
    # At most one real _ensure_tables invocation across all three calls.
    assert spy.call_count <= 1, (
        f"public methods should share the once-guard, got {spy.call_count} calls"
    )


def test_schema_ready_readonly_skips_ensure_when_schema_already_present(tmp_path):
    """Once the schema is set up, _ensure_tables_once must skip _ensure_tables
    entirely — no write-mutex acquisition on the hot request path. This is
    the exact behaviour observed in prod 2026-07-12 that gated GET
    /api/trader/at-mode for 30 seconds.
    """
    at = _make_trader(tmp_path)
    # __init__ ran _init_db which laid down at_state with the mode column.
    at._tables_ensured = False
    # readonly check should return True since schema is fully in place
    assert at._schema_ready_readonly() is True
    with mock.patch.object(at, "_ensure_tables") as spy:
        at._ensure_tables_once()
    assert spy.call_count == 0, (
        "Once schema is in place, _ensure_tables_once must skip _ensure_tables "
        "to keep the write mutex free. Called it %d time(s)." % spy.call_count
    )
    assert at._tables_ensured is True


def test_ensure_tables_once_falls_back_when_readonly_check_fails(tmp_path):
    """If the read-only check returns False (schema truly missing) the
    once-guard must fall back to the full _ensure_tables path so we self-heal.
    """
    at = _make_trader(tmp_path)
    at._tables_ensured = False
    with mock.patch.object(at, "_schema_ready_readonly", return_value=False):
        with mock.patch.object(at, "_ensure_tables") as spy:
            at._ensure_tables_once()
    assert spy.call_count == 1
    assert at._tables_ensured is True


def test_source_has_no_direct_ensure_tables_in_hot_paths():
    """Static guard: no remaining `self._ensure_tables()` calls in the public
    method bodies of autonomous_trader.py — everything must go through the
    once-guard. `_ensure_tables_once` and the definition of `_ensure_tables`
    itself are allowed to reference the raw method.
    """
    import components.wealth.autonomous_trader as at_mod
    src_path = at_mod.__file__
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # A raw "self._ensure_tables()" call is only allowed inside _ensure_tables_once.
    # We accept the reference in docstrings and comments.
    inside_once = False
    offenders: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("def _ensure_tables_once"):
            inside_once = True
            continue
        if inside_once and stripped.startswith("def ") and "_ensure_tables_once" not in stripped:
            inside_once = False
        if inside_once:
            continue
        # ignore comments / docstrings — they aren't executable calls
        if stripped.startswith("#") or stripped.startswith('"'):
            continue
        if "self._ensure_tables()" in line:
            offenders.append((idx, stripped))
    assert not offenders, (
        "Public methods must call self._ensure_tables_once() (not raw self._ensure_tables()):\n"
        + "\n".join(f"  line {n}: {t}" for n, t in offenders)
    )
