"""Tests for PR DD: widened materialiser input queue.

Focus areas:
- Multi-provenance picker (fresh_blood + promoter_path + gap_driven)
- Per-pool quotas honoured, rollover works
- Lowered confidence floor 0.60 (was 0.80)
- runtime_mode='stub_reverted' rows are re-pickable
- runtime_mode='quarantined' rows are permanently excluded
- Gap seeder inserts rows shaped for the picker
- seed_gaps=False disables the live gap fetcher (test hygiene)
"""
from __future__ import annotations

import sqlite3

import pytest

from components import capability_materialiser as mat


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


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "pr_dd.db")
    conn = sqlite3.connect(path)
    conn.executescript(CAP_SCHEMA)
    conn.commit()
    conn.close()
    return path


def _insert(db_path: str, **row):
    row.setdefault("capability_type", "utility")
    row.setdefault("description", row.get("name", "cap"))
    row.setdefault("runtime_mode", "stub")
    row.setdefault("judge_confidence", 0.9)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO capabilities "
        "(id, name, capability_type, description, provenance, "
        " judge_confidence, runtime_mode) "
        "VALUES (:id, :name, :capability_type, :description, "
        ":provenance, :judge_confidence, :runtime_mode)",
        row,
    )
    conn.commit()
    conn.close()


# ── Defaults widened ──────────────────────────────────────────────────────

def test_defaults_are_widened() -> None:
    assert mat.DEFAULT_DAILY_CAP == 10, "PR DD raised cap to 10"
    assert mat.DEFAULT_MIN_JUDGE_CONFIDENCE == pytest.approx(0.60), (
        "PR DD lowered confidence floor to 0.60"
    )
    assert set(mat.PICKER_QUOTAS.keys()) == {
        "fresh_blood_seed+self_judge",
        "promoter_path+self_judge",
        "gap_driven",
    }
    assert sum(mat.PICKER_QUOTAS.values()) == mat.DEFAULT_DAILY_CAP


# ── Per-pool picking + rollover ───────────────────────────────────────────

def test_picker_honours_quotas_and_rolls_over(db: str) -> None:
    # Seed 3 fresh_blood (quota 5), 5 promoter_path (quota 3),
    # 5 gap_driven (quota 2). Rollover math:
    #   fresh_blood: takes 3/5 -> 2 rollover
    #   promoter_path: quota 3 + rollover 2 = 5, takes 5 -> 0 rollover
    #   gap_driven: quota 2 + rollover 0 = 2, takes 2
    #   total = 3 + 5 + 2 = 10
    for i in range(3):
        _insert(db, id=f"fb_{i}", name=f"fb_{i}",
                provenance="fresh_blood_seed+self_judge",
                judge_confidence=0.9)
    for i in range(5):
        _insert(db, id=f"pp_{i}", name=f"pp_{i}",
                provenance="promoter_path+self_judge",
                judge_confidence=0.75)
    for i in range(5):
        _insert(db, id=f"gd_{i}", name=f"gd_{i}",
                provenance="gap_driven",
                judge_confidence=0.65)

    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS materialisation_log "
        "(capability_id TEXT, outcome TEXT, "
        " created_at TEXT DEFAULT (datetime('now')))"
    )
    picks = mat._pick_candidates(
        conn, min_confidence=0.60, limit=10,
    )
    conn.close()

    by_prov = {}
    for p in picks:
        by_prov[p["provenance"]] = by_prov.get(p["provenance"], 0) + 1

    assert len(picks) == 10, f"expected 10 picks, got {len(picks)}"
    assert by_prov["fresh_blood_seed+self_judge"] == 3
    assert by_prov["promoter_path+self_judge"] == 5  # 3 + 2 rollover
    assert by_prov["gap_driven"] == 2


def test_picker_confidence_floor_060(db: str) -> None:
    # A cap at 0.55 is below the new floor and must be excluded.
    _insert(db, id="below", name="below_floor",
            provenance="fresh_blood_seed+self_judge",
            judge_confidence=0.55)
    _insert(db, id="above", name="above_floor",
            provenance="fresh_blood_seed+self_judge",
            judge_confidence=0.65)

    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS materialisation_log "
        "(capability_id TEXT, outcome TEXT, "
        " created_at TEXT DEFAULT (datetime('now')))"
    )
    picks = mat._pick_candidates(conn, min_confidence=0.60, limit=10)
    conn.close()

    ids = [p["id"] for p in picks]
    assert "above" in ids
    assert "below" not in ids


def test_picker_includes_stub_reverted_excludes_quarantined(db: str) -> None:
    _insert(db, id="stub_cap", name="stub_cap",
            provenance="fresh_blood_seed+self_judge",
            runtime_mode="stub")
    _insert(db, id="rev_cap", name="rev_cap",
            provenance="fresh_blood_seed+self_judge",
            runtime_mode="stub_reverted")
    _insert(db, id="quar_cap", name="quar_cap",
            provenance="fresh_blood_seed+self_judge",
            runtime_mode="quarantined")
    _insert(db, id="live_cap", name="live_cap",
            provenance="fresh_blood_seed+self_judge",
            runtime_mode="generated_module")

    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS materialisation_log "
        "(capability_id TEXT, outcome TEXT, "
        " created_at TEXT DEFAULT (datetime('now')))"
    )
    picks = mat._pick_candidates(conn, min_confidence=0.60, limit=10)
    conn.close()

    ids = {p["id"] for p in picks}
    assert "stub_cap" in ids
    assert "rev_cap" in ids
    assert "quar_cap" not in ids
    assert "live_cap" not in ids


def test_picker_ignores_unknown_provenance(db: str) -> None:
    _insert(db, id="mystery", name="mystery",
            provenance="hand_crafted", judge_confidence=0.99)
    _insert(db, id="legit", name="legit",
            provenance="fresh_blood_seed+self_judge",
            judge_confidence=0.7)

    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS materialisation_log "
        "(capability_id TEXT, outcome TEXT, "
        " created_at TEXT DEFAULT (datetime('now')))"
    )
    picks = mat._pick_candidates(conn, min_confidence=0.60, limit=10)
    conn.close()

    ids = {p["id"] for p in picks}
    assert "legit" in ids
    assert "mystery" not in ids


# ── Gap seeder ────────────────────────────────────────────────────────────

def test_seed_gap_capabilities_inserts_and_is_idempotent(
    db: str, monkeypatch,
) -> None:
    class FakeGap:
        def __init__(self, name, description, priority=3):
            self.name = name
            self.description = description
            self.priority = priority

    fake_gaps = [
        FakeGap("skill_acquisition", "Learn a new skill", priority=1),
        FakeGap("kpi_recovery", "Recover a low KPI", priority=2),
    ]
    monkeypatch.setattr(
        "components.gap_fetcher.iter_capability_gaps",
        lambda fresh=False: iter(fake_gaps),
    )

    conn = sqlite3.connect(db)
    n1 = mat._seed_gap_capabilities(conn, max_new=5)
    assert n1 == 2

    # Rows landed with the expected shape
    rows = conn.execute(
        "SELECT id, name, provenance, judge_confidence, runtime_mode "
        "FROM capabilities WHERE provenance = 'gap_driven'"
    ).fetchall()
    assert len(rows) == 2
    for _id, _name, prov, conf, mode in rows:
        assert prov == "gap_driven"
        assert conf >= 0.60  # picker floor
        assert mode == "stub"

    # Second call must not double-insert
    n2 = mat._seed_gap_capabilities(conn, max_new=5)
    assert n2 == 0
    conn.close()


def test_seed_gaps_disabled_for_tmp_dbs(db: str, monkeypatch) -> None:
    """materialise_once auto-disables gap seeding for non-prod DBs."""
    called = {"count": 0}

    def spy(conn, max_new=5):
        called["count"] += 1
        return 999  # obviously wrong value we'd never expect to see

    monkeypatch.setattr(mat, "_seed_gap_capabilities", spy)

    # tmp_path DB (no 'dmai_knowledge' in path) -> should NOT call the seeder
    summary = mat.materialise_once(db_path=db, daily_cap=1)
    assert called["count"] == 0
    assert summary.get("gaps_seeded", 0) == 0

    # Same DB, explicit seed_gaps=True -> should call it
    called["count"] = 0
    summary = mat.materialise_once(
        db_path=db, daily_cap=1, seed_gaps=True,
    )
    assert called["count"] == 1
    assert summary.get("gaps_seeded") == 999


def test_seed_gaps_enabled_for_prod_path(tmp_path, monkeypatch) -> None:
    """A path that contains 'dmai_knowledge' auto-enables seeding."""
    called = {"count": 0}

    def spy(conn, max_new=5):
        called["count"] += 1
        return 3

    monkeypatch.setattr(mat, "_seed_gap_capabilities", spy)

    prod_like = tmp_path / "dmai_knowledge.db"
    conn = sqlite3.connect(str(prod_like))
    conn.executescript(CAP_SCHEMA)
    conn.commit()
    conn.close()

    summary = mat.materialise_once(db_path=str(prod_like), daily_cap=1)
    assert called["count"] == 1
    assert summary.get("gaps_seeded") == 3


# ── Summary shape ─────────────────────────────────────────────────────────

def test_summary_includes_new_observability_fields(db: str) -> None:
    _insert(db, id="obs_1", name="obs_1",
            provenance="fresh_blood_seed+self_judge",
            judge_confidence=0.9)

    # No codegen fn passed - it'll try the real client and fail on API,
    # which is fine: we only care about summary shape.
    from components.generated import _codegen_client as cg

    def fake_codegen(**kw):
        return cg.CodegenAttempt(ok=False, model="stub", reason="stub")

    summary = mat.materialise_once(
        db_path=db, codegen_fn=fake_codegen, daily_cap=5,
        seed_gaps=False,
    )
    assert "quotas" in summary
    assert "min_confidence" in summary
    assert "provenance_breakdown" in summary
    assert "gaps_seeded" in summary
    assert summary["min_confidence"] == pytest.approx(0.60)
    assert summary["gaps_seeded"] == 0
