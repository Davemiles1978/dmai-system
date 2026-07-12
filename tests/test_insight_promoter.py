"""Tests for the JSONL -> SQL insight promoter (PR B).

Root cause covered: si_core writes discoveries to
data/research/insights.jsonl, but the admin panel + KPI derivation code
reads from the ``insights`` SQL table in dmai_knowledge.db. Nothing
promoted JSONL -> SQL, so the admin panel showed a single bootstrap
insight even after DMAI had accumulated 18k+ JSONL rows.

These tests pin the promoter's behaviour end-to-end using a real SQLite
file so we exercise the exact code path used in production.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from components import insight_promoter as ip


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """A pre-created SQLite file with the ``insights`` and ``system_state``
    tables laid out the same way dmai_core_complete does at boot.
    """
    db_path = tmp_path / "dmai_knowledge.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT,
            insight_text TEXT,
            confidence REAL DEFAULT 0.5,
            domain TEXT,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS system_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()
    return str(db_path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _sql_count(db_path: str, table: str = "insights") -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _get_offset(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        r = conn.execute(
            "SELECT value FROM system_state WHERE key = ?", (ip.OFFSET_KEY,)
        ).fetchone()
        return int(r[0]) if r and r[0] is not None else 0
    finally:
        conn.close()


# ── Behaviour ─────────────────────────────────────────────────────────────

def test_promote_backfills_existing_rows(tmp_db: str, tmp_path: Path):
    jsonl = tmp_path / "insights.jsonl"
    _write_jsonl(jsonl, [
        {"id": "insight_1", "domain": "betting", "concept": "value drift",
         "source": "orch", "confidence": 0.9,
         "timestamp": "2026-07-12T10:00:00+00:00"},
        {"id": "insight_2", "domain": "trading", "concept": "mean reversion",
         "source": "orch", "confidence": 0.7,
         "timestamp": "2026-07-12T10:01:00+00:00"},
    ])
    result = ip.promote_once(jsonl_path=jsonl, db_path=tmp_db)
    assert result["promoted"] == 2
    assert result["skipped"] == 0
    assert result["reset_from"] is None
    assert result["new_offset"] == jsonl.stat().st_size
    assert _sql_count(tmp_db) == 2


def test_promote_is_idempotent(tmp_db: str, tmp_path: Path):
    jsonl = tmp_path / "insights.jsonl"
    _write_jsonl(jsonl, [
        {"domain": "d", "concept": "c", "confidence": 0.5, "source": "s"},
    ])
    ip.promote_once(jsonl_path=jsonl, db_path=tmp_db)
    # Second call must not re-promote.
    r = ip.promote_once(jsonl_path=jsonl, db_path=tmp_db)
    assert r["promoted"] == 0
    assert r["new_offset"] == jsonl.stat().st_size
    assert _sql_count(tmp_db) == 1


def test_promote_picks_up_new_appends(tmp_db: str, tmp_path: Path):
    jsonl = tmp_path / "insights.jsonl"
    _write_jsonl(jsonl, [{"domain": "d", "concept": "one"}])
    ip.promote_once(jsonl_path=jsonl, db_path=tmp_db)
    assert _sql_count(tmp_db) == 1

    _write_jsonl(jsonl, [
        {"domain": "d", "concept": "two"},
        {"domain": "d", "concept": "three"},
    ])
    r = ip.promote_once(jsonl_path=jsonl, db_path=tmp_db)
    assert r["promoted"] == 2
    assert _sql_count(tmp_db) == 3


def test_promote_skips_malformed_lines(tmp_db: str, tmp_path: Path):
    jsonl = tmp_path / "insights.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"domain": "d", "concept": "good"}) + "\n")
        f.write("NOT_JSON_LINE\n")
        f.write(json.dumps({"domain": "d"}) + "\n")  # no concept/insight_text
        f.write(json.dumps({"domain": "d", "concept": "also good"}) + "\n")

    r = ip.promote_once(jsonl_path=jsonl, db_path=tmp_db)
    assert r["promoted"] == 2
    assert r["skipped"] == 2
    assert _sql_count(tmp_db) == 2


def test_promote_handles_truncation(tmp_db: str, tmp_path: Path):
    jsonl = tmp_path / "insights.jsonl"
    _write_jsonl(jsonl, [{"domain": "d", "concept": f"c{i}"} for i in range(5)])
    ip.promote_once(jsonl_path=jsonl, db_path=tmp_db)
    assert _sql_count(tmp_db) == 5
    prev_offset = _get_offset(tmp_db)
    assert prev_offset > 0

    # Simulate rotation: shrink the file to 1 line.
    _write_jsonl(  # overwrite
        jsonl := (jsonl if jsonl.exists() else jsonl),
        [],
    )
    jsonl.write_text(json.dumps({"domain": "d", "concept": "fresh"}) + "\n")

    r = ip.promote_once(jsonl_path=jsonl, db_path=tmp_db)
    assert r["reset_from"] == prev_offset
    # After reset the 1 new row is promoted (SQL now has 5+1).
    assert r["promoted"] == 1
    assert _sql_count(tmp_db) == 6


def test_promote_uses_small_batches(tmp_db: str, tmp_path: Path):
    """A large JSONL must promote every row across multiple commits."""
    jsonl = tmp_path / "insights.jsonl"
    rows = [{"domain": "d", "concept": f"c{i}"} for i in range(1500)]
    _write_jsonl(jsonl, rows)
    r = ip.promote_once(jsonl_path=jsonl, db_path=tmp_db, batch_rows=500)
    assert r["promoted"] == 1500
    assert _sql_count(tmp_db) == 1500
    assert _get_offset(tmp_db) == jsonl.stat().st_size


def test_promote_no_op_when_file_missing(tmp_db: str, tmp_path: Path):
    r = ip.promote_once(jsonl_path=tmp_path / "does_not_exist.jsonl",
                        db_path=tmp_db)
    assert r == {"promoted": 0, "skipped": 0, "new_offset": 0,
                 "reset_from": None}
    assert _sql_count(tmp_db) == 0


def test_row_mapping_accepts_both_shapes():
    """si_core has emitted both {concept: ...} and {insight_text: ...} rows."""
    p1 = ip._row_to_insight_params({"concept": "x", "domain": "d"})
    p2 = ip._row_to_insight_params({"insight_text": "y", "domain": "d"})
    p3 = ip._row_to_insight_params({"domain": "d"})  # neither
    assert p1 is not None and p1[0] == "x"
    assert p2 is not None and p2[0] == "y"
    assert p3 is None


def test_row_mapping_coerces_bad_confidence():
    p = ip._row_to_insight_params(
        {"concept": "x", "domain": "d", "confidence": "not_a_number"}
    )
    assert p is not None
    # 3rd field (0-indexed 2) is confidence
    assert p[2] == 0.5


def test_row_mapping_clips_oversize_fields():
    p = ip._row_to_insight_params(
        {"concept": "a" * 5000, "insight_text": "b" * 10000,
         "domain": "d" * 1000, "source": "s" * 2000, "confidence": 0.5}
    )
    assert p is not None
    assert len(p[0]) == 2000
    assert len(p[1]) == 5000
    assert len(p[3]) == 200
    assert len(p[4]) == 500


def test_start_promoter_loop_is_idempotent(tmp_db: str, tmp_path: Path,
                                           monkeypatch):
    """Calling start_promoter_loop twice must not spawn two threads."""
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    # Reset module singleton so the test is self-contained.
    ip._LOOP = None
    jsonl = tmp_path / "insights.jsonl"
    _write_jsonl(jsonl, [{"domain": "d", "concept": "x"}])
    l1 = ip.start_promoter_loop(jsonl_path=jsonl, db_path=tmp_db,
                                poll_seconds=60)
    l2 = ip.start_promoter_loop(jsonl_path=jsonl, db_path=tmp_db,
                                poll_seconds=60)
    try:
        assert l1 is l2
        # Backfill promoted the initial row.
        assert _sql_count(tmp_db) == 1
    finally:
        l1.stop()
        ip._LOOP = None
