"""Tests for the capability registry.json -> SQL promoter (PR D).

Root cause covered: CapabilityIntegrator writes 20k+ capabilities to
data/capabilities/registry.json but the SQL capabilities table only has a
single bootstrap row because the SQL-mirror path in _save_registry is
guarded on hasattr(si_core, 'sqlite'), which is always False. This blocks
stage progression at Baby (Child needs >=500 capabilities in SQL).

These tests exercise the standalone promoter end-to-end with a real
SQLite file and real JSON registry files, matching the code path used at
boot.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from components import capability_promoter as cp


# --- Fixtures -------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """A pre-created SQLite file with capabilities + system_state tables
    matching the CORE schema used at boot."""
    db_path = tmp_path / "dmai_knowledge.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS capabilities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'function',
            capability_type TEXT NOT NULL DEFAULT 'general',
            description TEXT,
            source_url TEXT,
            source_repo TEXT,
            file_path TEXT,
            runtime_mode TEXT,
            language TEXT,
            methods TEXT,
            is_async INTEGER DEFAULT 0,
            args TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            integrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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


def _write_registry(path: Path, caps: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "capabilities": caps,
        "sources": {},
        "last_updated": "2026-07-12T20:00:00",
        "total_capabilities": len(caps),
        "fully_incorporated": [],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _sample_cap(cid: str, name: str = None, **overrides) -> dict:
    base = {
        "id": cid,
        "name": name or f"cap_{cid}",
        "type": "function",
        "capability_type": "utility",
        "description": f"Test capability {cid}",
        "source_url": "https://example.com/repo",
        "source_repo": "example/repo",
        "file_path": f"components/capabilities/{cid}.py",
        "runtime_mode": "autonomous",
        "language": "python",
        "methods": ["run", "stop"],
        "is_async": False,
        "args": ["ctx"],
        "integrated_at": "2026-07-12T20:00:00",
    }
    base.update(overrides)
    return base


def _sql_count(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0]
    finally:
        conn.close()


# --- Tests ----------------------------------------------------------------

def test_promote_once_backfills_all_capabilities(tmp_path, tmp_db):
    reg = tmp_path / "capabilities" / "registry.json"
    _write_registry(reg, {f"cap_{i}": _sample_cap(f"cap_{i}") for i in range(25)})

    result = cp.promote_once(registry_path=reg, db_path=tmp_db, force=True)

    assert result["promoted"] == 25
    assert result["skipped"] == 0
    assert result["total_in_registry"] == 25
    assert _sql_count(tmp_db) == 25


def test_promote_once_no_op_when_registry_missing(tmp_path, tmp_db):
    reg = tmp_path / "capabilities" / "registry.json"  # not created
    result = cp.promote_once(registry_path=reg, db_path=tmp_db, force=True)
    assert result["promoted"] == 0
    assert result["skipped"] == 0
    assert result.get("note") == "registry_missing"
    assert _sql_count(tmp_db) == 0


def test_promote_once_skips_rows_missing_id_or_name(tmp_path, tmp_db):
    reg = tmp_path / "capabilities" / "registry.json"
    caps = {
        "good_1": _sample_cap("good_1"),
        # This key becomes the id — but name is missing from the row, so skip.
        "bad_missing_name": {"id": "bad_missing_name", "type": "function"},
        "good_2": _sample_cap("good_2"),
        # A row where value is not a dict at all.
        "bad_shape": "not a dict",
        # A row missing the "name" key entirely.
        "bad_no_name_2": {"type": "function", "capability_type": "utility"},
    }
    _write_registry(reg, caps)

    result = cp.promote_once(registry_path=reg, db_path=tmp_db, force=True)

    assert result["promoted"] == 2
    assert result["skipped"] == 3
    assert result["total_in_registry"] == 5
    assert _sql_count(tmp_db) == 2


def test_promote_once_is_idempotent(tmp_path, tmp_db):
    reg = tmp_path / "capabilities" / "registry.json"
    _write_registry(reg, {f"c_{i}": _sample_cap(f"c_{i}") for i in range(10)})

    r1 = cp.promote_once(registry_path=reg, db_path=tmp_db, force=True)
    r2 = cp.promote_once(registry_path=reg, db_path=tmp_db, force=True)

    assert r1["promoted"] == 10
    assert r2["promoted"] == 10  # re-upserts same 10 rows
    assert _sql_count(tmp_db) == 10  # no dupes — INSERT OR REPLACE keyed on id


def test_mtime_skip_avoids_redundant_work(tmp_path, tmp_db):
    reg = tmp_path / "capabilities" / "registry.json"
    _write_registry(reg, {f"c_{i}": _sample_cap(f"c_{i}") for i in range(5)})

    # First run with force=False populates the mtime marker.
    r1 = cp.promote_once(registry_path=reg, db_path=tmp_db, force=False)
    assert r1["promoted"] == 5
    assert r1["mtime_unchanged"] is False

    # Second run without touching the file should skip due to mtime match.
    r2 = cp.promote_once(registry_path=reg, db_path=tmp_db, force=False)
    assert r2["mtime_unchanged"] is True
    assert r2["promoted"] == 0

    # Modifying the registry should trigger a fresh sync.
    time.sleep(0.05)  # ensure mtime advances on filesystems with 1s granularity
    _write_registry(reg, {f"c_{i}": _sample_cap(f"c_{i}") for i in range(7)})
    # Bump mtime explicitly to sidestep coarse fs timestamps in CI.
    now = time.time()
    os.utime(reg, (now, now))
    r3 = cp.promote_once(registry_path=reg, db_path=tmp_db, force=False)
    assert r3["mtime_unchanged"] is False
    assert r3["promoted"] == 7
    assert _sql_count(tmp_db) == 7


def test_upsert_overwrites_existing_row_fields(tmp_path, tmp_db):
    reg = tmp_path / "capabilities" / "registry.json"
    _write_registry(reg, {"c1": _sample_cap("c1", name="original", description="v1")})
    cp.promote_once(registry_path=reg, db_path=tmp_db, force=True)

    _write_registry(reg, {"c1": _sample_cap("c1", name="updated", description="v2")})
    cp.promote_once(registry_path=reg, db_path=tmp_db, force=True)

    conn = sqlite3.connect(tmp_db)
    try:
        row = conn.execute(
            "SELECT name, description FROM capabilities WHERE id = 'c1'"
        ).fetchone()
    finally:
        conn.close()
    assert row == ("updated", "v2")
    assert _sql_count(tmp_db) == 1  # still one row


def test_malformed_registry_json_is_handled(tmp_path, tmp_db):
    reg = tmp_path / "capabilities" / "registry.json"
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text("{ this is not valid json ")

    result = cp.promote_once(registry_path=reg, db_path=tmp_db, force=True)
    assert result["promoted"] == 0
    assert result.get("note", "").startswith("registry_unreadable")
    assert _sql_count(tmp_db) == 0


def test_registry_shape_invalid_is_handled(tmp_path, tmp_db):
    """registry.json exists but capabilities is a list, not a dict."""
    reg = tmp_path / "capabilities" / "registry.json"
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps({"capabilities": ["not", "a", "dict"]}))

    result = cp.promote_once(registry_path=reg, db_path=tmp_db, force=True)
    assert result["promoted"] == 0
    assert result.get("note") == "registry_shape_invalid"
    assert _sql_count(tmp_db) == 0


def test_batching_processes_large_registry(tmp_path, tmp_db):
    """Confirm the batch loop commits everything, including a partial final batch."""
    reg = tmp_path / "capabilities" / "registry.json"
    # 1250 = 2 full 500-batches + one partial 250-batch
    _write_registry(reg, {f"cap_{i}": _sample_cap(f"cap_{i}") for i in range(1250)})

    result = cp.promote_once(
        registry_path=reg, db_path=tmp_db,
        force=True, batch_rows=500, yield_ms=0,
    )

    assert result["promoted"] == 1250
    assert result["skipped"] == 0
    assert _sql_count(tmp_db) == 1250


def test_methods_and_args_persist_as_json(tmp_path, tmp_db):
    reg = tmp_path / "capabilities" / "registry.json"
    _write_registry(reg, {
        "c1": _sample_cap(
            "c1",
            methods=["do_thing", "undo_thing"],
            args=["arg1", "arg2"],
            is_async=True,
        )
    })
    cp.promote_once(registry_path=reg, db_path=tmp_db, force=True)

    conn = sqlite3.connect(tmp_db)
    try:
        row = conn.execute(
            "SELECT methods, args, is_async FROM capabilities WHERE id = 'c1'"
        ).fetchone()
    finally:
        conn.close()
    assert json.loads(row[0]) == ["do_thing", "undo_thing"]
    assert json.loads(row[1]) == ["arg1", "arg2"]
    assert row[2] == 1


def test_kdb_path_and_registry_path_honour_data_path(tmp_path, monkeypatch):
    """The module resolves both paths under DATA_PATH — mirroring every other
    DB-touching component and Render's persistent-disk mount."""
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    assert cp._kdb_path() == str(tmp_path / "dmai_knowledge.db")
    assert cp._registry_path() == tmp_path / "capabilities" / "registry.json"
