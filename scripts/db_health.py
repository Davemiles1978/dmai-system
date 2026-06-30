#!/usr/bin/env python3
"""DMAI manual DB health-check suite.

Pure-stdlib. Usable two ways:
  1. As a library — import the module-level ``check_*`` functions or
     ``run_all_checks`` (the Flask admin endpoint /api/admin/db-health does this).
  2. As a CLI:
       python scripts/db_health.py [--db PATH] [--schema PATH] [--json|--pretty]

Four checks, each returning {"name", "status": "ok|warn|fail", "details": {...}}:
  - check_integrity      PRAGMA integrity_check + PRAGMA foreign_key_check
  - check_schema_drift   tables in schema.sql vs tables in the live DB
  - check_row_counts     per-table row count + latest timestamp column value
  - check_wal_state      journal mode, checkpoint lag, on-disk file sizes

Exit code reflects the worst result: 0=ok, 1=warn, 2=fail.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path


def _jsonable(obj):
    """Recursively coerce a value into something ``json.dumps`` can serialize.

    SQLite can hand back ``bytes`` (e.g. ``MAX()`` over a BLOB-affinity column,
    or raw pragma rows), which Flask's ``jsonify`` rejects with
    ``TypeError: Object of type bytes is not JSON serializable``. Decode bytes,
    flatten ``sqlite3.Row``, and stringify other non-primitive types so the
    health-check payload is always serializable.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, bytearray):
        return bytes(obj).decode("utf-8", errors="replace")
    if isinstance(obj, sqlite3.Row):
        return {k: _jsonable(obj[k]) for k in obj.keys()}
    if isinstance(obj, dict):
        return {(_jsonable(k) if not isinstance(k, str) else k): _jsonable(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)

# Worst-wins ordering for combining statuses.
_STATUS_RANK = {"ok": 0, "warn": 1, "fail": 2}
_EXIT_CODE = {"ok": 0, "warn": 1, "fail": 2}

# A table is considered "stale-detectable" if it has one of these columns; we
# report MAX(col) as the last-write marker (check_row_counts).
_TS_COLUMNS = ("updated_at", "created_at", "ts", "recorded_at", "added_at", "loaded_at")

# Warn threshold for the -wal sidecar file (64 MiB).
_WAL_WARN_BYTES = 64 * 1024 * 1024


def _worst(statuses) -> str:
    worst = "ok"
    for s in statuses:
        if _STATUS_RANK.get(s, 0) > _STATUS_RANK[worst]:
            worst = s
    return worst


def _connect(db_path: str) -> sqlite3.Connection:
    # Read-only-ish: we never write in health checks. A short busy timeout
    # avoids hanging behind the app's write locks.
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def _list_tables(conn: sqlite3.Connection):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Check 1 — integrity + foreign keys
# ---------------------------------------------------------------------------
def check_integrity(db_path: str) -> dict:
    name = "integrity"
    if not os.path.exists(db_path):
        return {"name": name, "status": "fail",
                "details": {"error": f"db not found: {db_path}"}}
    details: dict = {}
    try:
        conn = _connect(db_path)
        try:
            integ = [r[0] for r in conn.execute("PRAGMA integrity_check").fetchall()]
            fk_rows = conn.execute("PRAGMA foreign_key_check").fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        return {"name": name, "status": "fail",
                "details": {"error": f"{type(e).__name__}: {e}"}}

    fk_violations = [dict(r) if isinstance(r, sqlite3.Row) else list(r) for r in fk_rows]
    integrity_ok = integ == ["ok"]
    details["integrity_check"] = integ
    details["foreign_key_violations"] = fk_violations
    if integrity_ok and not fk_violations:
        status = "ok"
    elif integrity_ok and fk_violations:
        # Structurally sound but referential integrity is off — actionable, not fatal.
        status = "warn"
    else:
        status = "fail"
    return {"name": name, "status": status, "details": details}


# ---------------------------------------------------------------------------
# Check 2 — schema drift vs scripts/schema.sql
# ---------------------------------------------------------------------------
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'`]?(\w+)",
    re.IGNORECASE,
)


def _schema_table_names(schema_sql_path: str):
    text = Path(schema_sql_path).read_text(encoding="utf-8")
    # Strip line comments so commented-out TODO table names don't count.
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("--")]
    cleaned = "\n".join(lines)
    return sorted({m.group(1).lower() for m in _CREATE_TABLE_RE.finditer(cleaned)})


def check_schema_drift(db_path: str, schema_sql_path: str) -> dict:
    name = "schema_drift"
    if not os.path.exists(db_path):
        return {"name": name, "status": "fail",
                "details": {"error": f"db not found: {db_path}"}}
    if not os.path.exists(schema_sql_path):
        return {"name": name, "status": "warn",
                "details": {"error": f"schema file not found: {schema_sql_path}"}}
    try:
        expected = set(_schema_table_names(schema_sql_path))
    except Exception as e:
        return {"name": name, "status": "warn",
                "details": {"error": f"could not parse schema: {e}"}}
    try:
        conn = _connect(db_path)
        try:
            actual = {t.lower() for t in _list_tables(conn)}
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        return {"name": name, "status": "fail",
                "details": {"error": f"{type(e).__name__}: {e}"}}

    missing = sorted(expected - actual)   # in schema.sql, absent from DB
    extra = sorted(actual - expected)     # in DB, absent from schema.sql
    details = {
        "expected_table_count": len(expected),
        "actual_table_count": len(actual),
        "missing_tables": missing,
        "extra_tables": extra,
    }
    # Missing tables are the dangerous drift (GET routes crash). Extra tables
    # are usually benign (scratch/legacy) → warn only.
    if missing:
        status = "fail"
    elif extra:
        status = "warn"
    else:
        status = "ok"
    return {"name": name, "status": status, "details": details}


# ---------------------------------------------------------------------------
# Check 3 — row counts + last write per table
# ---------------------------------------------------------------------------
def check_row_counts(db_path: str) -> dict:
    name = "row_counts"
    if not os.path.exists(db_path):
        return {"name": name, "status": "fail",
                "details": {"error": f"db not found: {db_path}"}}
    try:
        conn = _connect(db_path)
    except sqlite3.DatabaseError as e:
        return {"name": name, "status": "fail",
                "details": {"error": f"{type(e).__name__}: {e}"}}
    tables_info: dict = {}
    errors = []
    try:
        for table in _list_tables(conn):
            entry: dict = {"rows": None, "last_write_column": None, "last_write": None}
            try:
                entry["rows"] = conn.execute(
                    f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                cols = {r[1] for r in conn.execute(
                    f'PRAGMA table_info("{table}")').fetchall()}
                for tc in _TS_COLUMNS:
                    if tc in cols:
                        val = conn.execute(
                            f'SELECT MAX("{tc}") FROM "{table}"').fetchone()[0]
                        entry["last_write_column"] = tc
                        entry["last_write"] = val
                        break
            except sqlite3.DatabaseError as e:
                errors.append(f"{table}: {e}")
                entry["error"] = str(e)
            tables_info[table] = entry
    finally:
        conn.close()
    details = {
        "table_count": len(tables_info),
        "total_rows": sum(v["rows"] or 0 for v in tables_info.values()),
        "tables": tables_info,
    }
    if errors:
        details["errors"] = errors
    # Per-table read failures suggest localized corruption → warn (integrity
    # check is the authority on hard failure).
    status = "warn" if errors else "ok"
    return {"name": name, "status": status, "details": details}


# ---------------------------------------------------------------------------
# Check 4 — WAL state
# ---------------------------------------------------------------------------
def check_wal_state(db_path: str) -> dict:
    name = "wal_state"
    if not os.path.exists(db_path):
        return {"name": name, "status": "fail",
                "details": {"error": f"db not found: {db_path}"}}

    def _size(p):
        try:
            return os.path.getsize(p)
        except OSError:
            return 0

    db_size = _size(db_path)
    wal_size = _size(db_path + "-wal")
    shm_size = _size(db_path + "-shm")
    details = {
        "db_size_bytes": db_size,
        "wal_size_bytes": wal_size,
        "shm_size_bytes": shm_size,
        "wal_warn_threshold_bytes": _WAL_WARN_BYTES,
    }
    try:
        conn = _connect(db_path)
        try:
            details["journal_mode"] = conn.execute("PRAGMA journal_mode").fetchone()[0]
            # PASSIVE checkpoint never blocks writers; returns (busy, log, checkpointed).
            cp = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            if cp is not None:
                details["wal_checkpoint"] = {
                    "busy": cp[0], "log_frames": cp[1], "checkpointed_frames": cp[2]}
                details["checkpoint_lag_frames"] = (cp[1] or 0) - (cp[2] or 0)
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        return {"name": name, "status": "fail",
                "details": {**details, "error": f"{type(e).__name__}: {e}"}}

    status = "warn" if wal_size > _WAL_WARN_BYTES else "ok"
    if status == "warn":
        details["warning"] = (
            f"WAL file {wal_size} bytes exceeds {_WAL_WARN_BYTES} threshold; "
            "checkpoint may be starved by long-lived readers/writers")
    return {"name": name, "status": status, "details": details}


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def run_all_checks(db_path: str, schema_sql_path: str) -> dict:
    checks = [
        check_integrity(db_path),
        check_schema_drift(db_path, schema_sql_path),
        check_row_counts(db_path),
        check_wal_state(db_path),
    ]
    overall = _worst(c["status"] for c in checks)
    return _jsonable({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "db_path": db_path,
        "schema_sql_path": schema_sql_path,
        "checks": checks,
        "overall_status": overall,
    })


def _default_schema_path() -> str:
    return str(Path(__file__).resolve().parent / "schema.sql")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DMAI manual DB health checks")
    parser.add_argument("--db", default="data/dmai_knowledge.db",
                        help="path to the SQLite DB (default: data/dmai_knowledge.db)")
    parser.add_argument("--schema", default=_default_schema_path(),
                        help="path to schema.sql (default: alongside this script)")
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="compact JSON (default)")
    fmt.add_argument("--pretty", action="store_true", help="pretty-printed JSON")
    args = parser.parse_args(argv)

    result = run_all_checks(args.db, args.schema)
    if args.pretty:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(json.dumps(result, default=str))
    return _EXIT_CODE[result["overall_status"]]


if __name__ == "__main__":
    sys.exit(main())
