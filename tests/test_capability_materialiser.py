"""E2E tests for components.capability_materialiser.

We stub the LLM cascade with hand-written module source and prove
the orchestrator wires validator + sandbox + smoke tests + self_judge
review into a promote-or-fail decision.
"""
from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

import pytest

from components import capability_materialiser as mat
from components.generated import _codegen_client as cg


# ── Test scaffolding ──────────────────────────────────────────────────────

CAP_SCHEMA = """
CREATE TABLE IF NOT EXISTS capabilities (
    id                TEXT PRIMARY KEY,
    name              TEXT,
    capability_type   TEXT,
    description       TEXT,
    provenance        TEXT,
    judge_confidence  REAL,
    runtime_mode      TEXT
);
"""


def _seed_capability(db_path: str, **overrides) -> str:
    """Insert one candidate row and return its id."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(CAP_SCHEMA)
        row = {
            "id":               overrides.get("id", "cap_test_1"),
            "name":             overrides.get("name",
                                              "sum utility test"),
            "capability_type":  overrides.get("capability_type", "utility"),
            "description":      overrides.get(
                "description",
                "Adds numbers together. Sums a list of values.",
            ),
            "provenance":       overrides.get(
                "provenance", "fresh_blood_seed+self_judge"),
            "judge_confidence": overrides.get("judge_confidence", 0.9),
            "runtime_mode":     overrides.get("runtime_mode", "stub"),
        }
        conn.execute(
            "INSERT OR REPLACE INTO capabilities "
            "(id, name, capability_type, description, "
            " provenance, judge_confidence, runtime_mode) "
            "VALUES (:id, :name, :capability_type, :description, "
            ":provenance, :judge_confidence, :runtime_mode)",
            row,
        )
        conn.commit()
    finally:
        conn.close()
    return row["id"]


@pytest.fixture()
def db(tmp_path):
    return str(tmp_path / "dmai.db")


@pytest.fixture(autouse=True)
def _cleanup_generated_dirs(tmp_path_factory):
    """Remove any staging/live/test files this session created."""
    yield
    for d in (mat.STAGING_DIR, mat.LIVE_DIR, mat.TESTS_DIR):
        if not d.exists():
            continue
        for p in d.glob("*.py"):
            if p.name == "__init__.py":
                continue
            if p.name.startswith(("test_sum_utility",
                                  "sum_utility",
                                  "test_bad_run",
                                  "bad_run",
                                  "test_docstring_",
                                  "docstring_")):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass


# ── Fake codegen implementations ──────────────────────────────────────────

GOOD_SRC = textwrap.dedent('''
    """Adds numbers together. Sums a list of values."""
    from __future__ import annotations
    from typing import List, Iterable

    def run(values: Iterable[float] | None = None, **kwargs) -> float:
        vals: List[float] = list(values or [])
        return float(sum(vals))
''').lstrip("\n")

BANNED_SRC = textwrap.dedent('''
    """Sneaky."""
    def run(**k):
        return eval("1 + 1")
''').lstrip("\n")

DRIFTED_DOC_SRC = textwrap.dedent('''
    """Renders a monochrome checkerboard fractal."""
    from __future__ import annotations
    def run(**kwargs):
        return 0
''').lstrip("\n")


def _fake_codegen(source: str):
    def fake(concept, insight, capability_type, happy_kwargs, *,
             model=cg.MODEL_PRIMARY, retry_reasons=None, max_tokens=1500):
        return cg.CodegenAttempt(ok=True, source=source, model=model)
    return fake


# ── Tests ─────────────────────────────────────────────────────────────────

def test_promotes_valid_candidate(db, monkeypatch):
    _seed_capability(db, id="cap_ok", name="sum utility ok")
    # Bypass the real self_judge - patch review_generated_module in
    # the module namespace the orchestrator imports it from.
    from components.generated import _self_judge_review as review
    monkeypatch.setattr(review, "review_generated_module",
                        lambda **kw: review.ReviewResult(
                            ok=True, verdict="accept",
                            confidence=0.9,
                            reason="stubbed"))

    summary = mat.materialise_once(
        db_path=db,
        codegen_fn=_fake_codegen(GOOD_SRC),
        daily_cap=5,
    )
    assert summary["picked"] == 1
    assert summary["promoted"] == 1
    assert summary["failed"] == 0
    # Live file was written
    live_files = list(mat.LIVE_DIR.glob("*.py"))
    assert any(p.name != "__init__.py" for p in live_files)
    # runtime_mode flipped in DB
    conn = sqlite3.connect(db)
    try:
        rm = conn.execute(
            "SELECT runtime_mode FROM capabilities WHERE id = ?",
            ("cap_ok",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert rm == "generated_module"


def test_rejects_banned_source_and_retries_then_fails(db, monkeypatch):
    _seed_capability(db, id="cap_bad", name="bad run")
    from components.generated import _self_judge_review as review
    monkeypatch.setattr(review, "review_generated_module",
                        lambda **kw: review.ReviewResult(
                            ok=True, verdict="accept",
                            confidence=0.9, reason="stubbed"))

    summary = mat.materialise_once(
        db_path=db,
        codegen_fn=_fake_codegen(BANNED_SRC),
        daily_cap=5,
    )
    assert summary["picked"] == 1
    assert summary["promoted"] == 0
    assert summary["failed"] == 1
    reasons_joined = " ".join(summary["results"][0]["reasons"])
    assert "banned_call:" in reasons_joined


def test_self_judge_docstring_drift_rejects(db, monkeypatch):
    _seed_capability(db, id="cap_drift", name="docstring drift test")
    from components.generated import _self_judge_review as review
    monkeypatch.setattr(review, "review_generated_module",
                        lambda **kw: review.ReviewResult(
                            ok=False, verdict="reject",
                            confidence=0.1,
                            reason="drift detected"))

    summary = mat.materialise_once(
        db_path=db,
        codegen_fn=_fake_codegen(DRIFTED_DOC_SRC),
        daily_cap=5,
    )
    assert summary["failed"] == 1
    assert any("self_judge_review" in r
               for r in summary["results"][0]["reasons"])


def test_ignores_low_confidence_candidates(db, monkeypatch):
    _seed_capability(db, id="cap_low", name="low conf",
                     judge_confidence=0.5)
    summary = mat.materialise_once(
        db_path=db,
        codegen_fn=_fake_codegen(GOOD_SRC),
        daily_cap=5,
        min_confidence=0.8,
    )
    assert summary["picked"] == 0
    assert summary["promoted"] == 0


def test_ignores_non_stub_candidates(db, monkeypatch):
    _seed_capability(db, id="cap_live", name="already live",
                     runtime_mode="generated_module")
    summary = mat.materialise_once(
        db_path=db,
        codegen_fn=_fake_codegen(GOOD_SRC),
        daily_cap=5,
    )
    assert summary["picked"] == 0


def test_daily_cap_stops_selection(db, monkeypatch):
    from components.generated import _self_judge_review as review
    monkeypatch.setattr(review, "review_generated_module",
                        lambda **kw: review.ReviewResult(
                            ok=True, verdict="accept",
                            confidence=0.9, reason="stubbed"))
    for i in range(3):
        _seed_capability(db, id=f"cap_c{i}",
                         name=f"sum utility cap {i}")
    summary = mat.materialise_once(
        db_path=db, codegen_fn=_fake_codegen(GOOD_SRC),
        daily_cap=2,
    )
    assert summary["picked"] == 2
    assert summary["promoted"] == 2

    # Next pass: cap is spent, no picks
    summary2 = mat.materialise_once(
        db_path=db, codegen_fn=_fake_codegen(GOOD_SRC),
        daily_cap=2,
    )
    assert summary2.get("cap_hit") is True
    assert summary2["picked"] == 0


def test_slug_helper():
    assert mat._slug("Hello World!!") == "hello_world"
    # ** is expanded to -x--x- by the promoter-compatible helper
    assert mat._slug("A × B") == "a_x_b"
    assert mat._slug("") == "unnamed"


def test_start_loop_is_idempotent(db, monkeypatch):
    # Ensure the module-level global isn't affected between test runs
    mat._LOOP = None
    from components.generated import _self_judge_review as review
    monkeypatch.setattr(review, "review_generated_module",
                        lambda **kw: review.ReviewResult(
                            ok=True, verdict="accept", confidence=0.9,
                            reason="stub"))

    loop1 = mat.start_capability_materialiser_loop(
        db_path=db, poll_seconds=60,
    )
    loop2 = mat.start_capability_materialiser_loop(
        db_path=db, poll_seconds=60,
    )
    assert loop1 is loop2
    loop1.stop()
    mat._LOOP = None
