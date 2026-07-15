"""Tests for components/capability_schema_migration.py"""

from __future__ import annotations

import os
import sqlite3

import pytest

from components import capability_schema_migration as mig


def _make_legacy_db(path: str) -> None:
    """Build a DB with the legacy capabilities shape prod has."""
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE capabilities (
                id            TEXT PRIMARY KEY,
                name          TEXT,
                capability_type TEXT,
                description   TEXT,
                runtime_mode  TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO capabilities VALUES (?,?,?,?,?)",
            [
                ("od_1", "od one", "utility", "", "ondemand"),
                ("od_2", "od two", "utility", "", "ondemand"),
                ("au_1", "au one", "utility", "", "autonomous"),
                ("odd", "odd row", "utility", "", None),
            ],
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def legacy_db(tmp_path, monkeypatch):
    p = tmp_path / "leg.db"
    _make_legacy_db(str(p))
    monkeypatch.setenv("DMAI_KNOWLEDGE_DB", str(p))
    return str(p)


def test_dry_run_reports_plan_without_changes(legacy_db):
    r = mig.migrate_capabilities_schema(dry_run=True)
    assert r["ok"] is True
    assert r["dry_run"] is True
    assert "planned" in r
    # No side effects: provenance column must still be absent
    conn = sqlite3.connect(legacy_db)
    try:
        cols = [c[1] for c in conn.execute(
            "PRAGMA table_info(capabilities)").fetchall()]
    finally:
        conn.close()
    assert "provenance" not in cols
    assert "judge_confidence" not in cols


def test_migration_adds_missing_columns(legacy_db):
    r = mig.migrate_capabilities_schema()
    assert r["ok"] is True
    cols = r["columns_after"]
    assert "provenance" in cols
    assert "judge_confidence" in cols


def test_backfill_only_touches_legacy_rows_and_leaves_others_null(legacy_db):
    mig.migrate_capabilities_schema()
    conn = sqlite3.connect(legacy_db)
    try:
        rows = dict(conn.execute(
            "SELECT id, provenance FROM capabilities").fetchall())
    finally:
        conn.close()
    assert rows["od_1"] == "legacy_ondemand"
    assert rows["od_2"] == "legacy_ondemand"
    assert rows["au_1"] == "legacy_autonomous"
    # NULL runtime_mode row not backfilled
    assert rows["odd"] is None


def test_migration_is_idempotent(legacy_db):
    r1 = mig.migrate_capabilities_schema()
    r2 = mig.migrate_capabilities_schema()
    assert r1["ok"] is True
    assert r2["ok"] is True
    # Second run: columns not added again
    steps2 = {s["name"]: s for s in r2["steps"]}
    assert steps2["add_provenance_column"]["changed"] is False
    assert steps2["add_judge_confidence_column"]["changed"] is False
    # Backfill: 0 rows updated on second run
    assert steps2["backfill_legacy_ondemand"]["rows_updated"] == 0
    assert steps2["backfill_legacy_autonomous"]["rows_updated"] == 0


def test_runtime_mode_values_never_overwritten(legacy_db):
    conn = sqlite3.connect(legacy_db)
    try:
        before = dict(conn.execute(
            "SELECT id, runtime_mode FROM capabilities").fetchall())
    finally:
        conn.close()
    mig.migrate_capabilities_schema()
    conn = sqlite3.connect(legacy_db)
    try:
        after = dict(conn.execute(
            "SELECT id, runtime_mode FROM capabilities").fetchall())
    finally:
        conn.close()
    assert before == after


def test_materialisation_log_created(legacy_db):
    mig.migrate_capabilities_schema()
    conn = sqlite3.connect(legacy_db)
    try:
        cols = [c[1] for c in conn.execute(
            "PRAGMA table_info(materialisation_log)").fetchall()]
    finally:
        conn.close()
    assert set(cols) >= {"capability_id", "outcome", "created_at"}


def test_new_stub_rows_are_pickable_after_migration(legacy_db):
    """After migration, a fresh stub+gap_driven row should be visible to
    the materialiser's picker."""
    mig.migrate_capabilities_schema()
    conn = sqlite3.connect(legacy_db)
    try:
        conn.execute(
            "INSERT INTO capabilities "
            "(id,name,capability_type,description,provenance,"
            " judge_confidence,runtime_mode) "
            "VALUES ('gap_test','test cap','utility','',"
            "'gap_driven',0.75,'stub')"
        )
        conn.commit()
        rows = conn.execute(
            "SELECT id FROM capabilities "
            "WHERE runtime_mode IN ('stub','stub_reverted') "
            "  AND provenance IN ('gap_driven','fresh_blood_seed+self_judge',"
            "                     'promoter_path+self_judge') "
            "  AND judge_confidence >= 0.60"
        ).fetchall()
    finally:
        conn.close()
    assert ("gap_test",) in rows


def test_missing_db_reports_error(monkeypatch, tmp_path):
    monkeypatch.setenv("DMAI_KNOWLEDGE_DB", str(tmp_path / "nope.db"))
    r = mig.migrate_capabilities_schema()
    assert r["ok"] is False
    assert "not found" in r["error"]


def test_missing_capabilities_table_reports_error(tmp_path, monkeypatch):
    p = tmp_path / "empty.db"
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE unrelated(id TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setenv("DMAI_KNOWLEDGE_DB", str(p))
    r = mig.migrate_capabilities_schema()
    assert r["ok"] is False
    assert "capabilities table missing" in r["error"]
