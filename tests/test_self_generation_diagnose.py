"""Tests for components/self_generation_diagnose.py"""

from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

from components import self_generation_diagnose as sgd


@pytest.fixture()
def temp_db(monkeypatch):
    """Build a temp SQLite DB shaped like production's capabilities+log."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.executescript(
        """
        CREATE TABLE capabilities(
            id TEXT PRIMARY KEY,
            name TEXT,
            capability_type TEXT,
            description TEXT,
            provenance TEXT,
            judge_confidence REAL,
            runtime_mode TEXT
        );
        CREATE TABLE materialisation_log(
            capability_id TEXT,
            outcome TEXT,
            created_at TEXT
        );
        CREATE TABLE insights(
            id INTEGER PRIMARY KEY,
            provenance TEXT
        );
        """
    )
    # Seed representative rows: one row that should pass all filters,
    # one that fails on runtime_mode, one that fails on confidence,
    # one that's already promoted (log row).
    conn.executemany(
        "INSERT INTO capabilities(id,name,capability_type,description,"
        "provenance,judge_confidence,runtime_mode) VALUES (?,?,?,?,?,?,?)",
        [
            ("ok_1",     "ok_1",     "utility", "",
             "gap_driven",              0.80, "stub"),
            ("live_1",   "live_1",   "utility", "",
             "gap_driven",              0.80, "live"),
            ("lowconf",  "lowconf",  "utility", "",
             "gap_driven",              0.30, "stub"),
            ("done",     "done",     "utility", "",
             "gap_driven",              0.90, "stub"),
            ("fb_ok",    "fb_ok",    "utility", "",
             "fresh_blood_seed+self_judge", 0.70, "stub"),
        ],
    )
    conn.execute(
        "INSERT INTO materialisation_log(capability_id,outcome,created_at) "
        "VALUES ('done','promoted','2026-01-01T00:00:00+00:00')"
    )
    conn.executemany(
        "INSERT INTO insights(provenance) VALUES (?)",
        [("fresh_blood/github",), ("fresh_blood/diversity",),
         ("other",)],
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("DMAI_KNOWLEDGE_DB", tmp.name)
    yield tmp.name
    os.unlink(tmp.name)


def test_diagnose_returns_ok_and_expected_shape(temp_db):
    r = sgd.diagnose_self_generation()
    assert r["ok"] is True
    assert "capabilities_table" in r
    assert "gap_seeder" in r
    assert "fresh_blood" in r
    assert "capability_promoter" in r
    assert "verdict" in r


def test_capabilities_table_counts_correct(temp_db):
    r = sgd.diagnose_self_generation()
    ct = r["capabilities_table"]
    assert ct["total_rows"] == 5
    assert ct["by_runtime_mode"].get("stub") == 4
    assert ct["by_runtime_mode"].get("live") == 1


def test_per_pool_filter_chain(temp_db):
    r = sgd.diagnose_self_generation(min_confidence=0.60)
    pools = r["capabilities_table"]["per_pool"]
    gap = pools["gap_driven"]
    # 4 rows have gap_driven provenance
    assert gap["total_with_provenance"] == 4
    # 3 are in stub state (ok_1, lowconf, done); live_1 excluded
    assert gap["stub_or_reverted"] == 3
    # ok_1 (0.80) and done (0.90) clear 0.60; lowconf (0.30) doesn't
    assert gap["above_confidence_floor"] == 2


def test_min_confidence_override_changes_pool_counts(temp_db):
    r_low = sgd.diagnose_self_generation(min_confidence=0.20)
    r_high = sgd.diagnose_self_generation(min_confidence=0.85)
    pl_low = r_low["capabilities_table"]["per_pool"]["gap_driven"]
    pl_high = r_high["capabilities_table"]["per_pool"]["gap_driven"]
    assert pl_low["above_confidence_floor"] == 3   # 0.80,0.90,0.30 all pass
    assert pl_high["above_confidence_floor"] == 1  # only 0.90


def test_materialisation_log_summary(temp_db):
    r = sgd.diagnose_self_generation()
    outcomes = r["capabilities_table"]["materialisation_log_outcomes"]
    assert outcomes.get("promoted") == 1


def test_fresh_blood_insight_count(temp_db):
    r = sgd.diagnose_self_generation()
    fb = r["fresh_blood"]
    # Module may or may not import; if it did, the SQL count is 2
    if "error" not in fb:
        assert fb.get("sql_fresh_blood_insights") == 2


def test_verdict_reasons_generated(temp_db):
    r = sgd.diagnose_self_generation()
    reasons = r["verdict"]["per_pool_reasons"]
    # Should have one line per accepted provenance
    assert len(reasons) == 3
    assert any("gap_driven" in x for x in reasons)


def test_missing_db_reports_error(monkeypatch, tmp_path):
    fake = tmp_path / "nope.db"
    monkeypatch.setenv("DMAI_KNOWLEDGE_DB", str(fake))
    r = sgd.diagnose_self_generation()
    assert r["ok"] is False
    assert "not found" in r["error"]


def test_never_raises_on_broken_capabilities_shape(monkeypatch, tmp_path):
    """If the capabilities table is the wrong shape, we should degrade
    gracefully, not raise."""
    tmp = tmp_path / "broken.db"
    conn = sqlite3.connect(tmp)
    # Wrong shape: missing provenance / runtime_mode columns
    conn.execute("CREATE TABLE capabilities(id TEXT PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setenv("DMAI_KNOWLEDGE_DB", str(tmp))
    r = sgd.diagnose_self_generation()
    assert r["ok"] is True
    # Errors should surface in the sub-sections but the top-level
    # response is still a well-formed dict
    assert "capabilities_table" in r
