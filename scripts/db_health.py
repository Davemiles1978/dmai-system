#!/usr/bin/env python3
"""DMAI manual DB health-check suite.

Pure-stdlib. Usable two ways:
  1. As a library — import ``run_all_checks`` (or the individual ``check_*``
     helpers). The Flask admin endpoint ``POST /api/admin/db-health`` does this.
  2. As a CLI:
       python scripts/db_health.py [--db PATH] [--json|--pretty]

Four checks, each returning a ``CheckResult``:
  1. integrity     PRAGMA integrity_check + PRAGMA foreign_key_check
  2. schema_diff   objects in scripts/schema.sql vs the live DB's sqlite_master
  3. row_counts    per-table COUNT(*) + last-write timestamp (pure reporting)
  4. wal_state     journal mode, checkpoint lag, on-disk file sizes

Exit code = worst status across the four checks: 0=ok/info, 1=warn, 2=error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Status = Literal["ok", "info", "warn", "error"]

# Worst-wins ordering. ``info`` is strictly informational and never fails a run.
_STATUS_RANK = {"ok": 0, "info": 1, "warn": 2, "error": 3}
_EXIT_CODE = {"ok": 0, "info": 0, "warn": 1, "error": 2}

# A table is "stale-detectable" if it has one of these columns; we report
# MAX(col) as the last-write marker in check_row_counts.
_TS_COLUMNS = ("updated_at", "created_at", "ts", "recorded_at", "added_at", "loaded_at")

# Warn threshold for the -wal sidecar file (64 MiB).
_WAL_WARN_BYTES = 64 * 1024 * 1024


@dataclass
class CheckResult:
    name: str
    status: Status
    details: dict
    duration_ms: float = 0.0


def worst_status(results) -> Status:
    """Combine statuses worst-wins. Accepts CheckResult objects or status strings."""
    worst: Status = "ok"
    for r in results:
        s = r.status if isinstance(r, CheckResult) else r
        if _STATUS_RANK.get(s, 0) > _STATUS_RANK[worst]:
            worst = s  # type: ignore[assignment]
    return worst


def exit_code_for(status: Status) -> int:
    return _EXIT_CODE.get(status, 2)


def _connect(db_path: str) -> sqlite3.Connection:
    # Health checks never write. A short busy timeout avoids hanging behind the
    # app's write locks.
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def _list_tables(conn: sqlite3.Connection):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _default_schema_path() -> str:
    return str(Path(__file__).resolve().parent / "schema.sql")


# ---------------------------------------------------------------------------
# Check 1 — integrity + foreign keys
# ---------------------------------------------------------------------------
def check_integrity(db_path: str) -> CheckResult:
    name = "integrity"
    t0 = time.perf_counter()

    def done(status, details):
        return CheckResult(name, status, details,
                           round((time.perf_counter() - t0) * 1000, 3))

    if not os.path.exists(db_path):
        return done("error", {"error": f"db not found: {db_path}"})
    try:
        conn = _connect(db_path)
        try:
            integ = [r[0] for r in conn.execute("PRAGMA integrity_check").fetchall()]
            fk_rows = conn.execute("PRAGMA foreign_key_check").fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        return done("error", {"error": f"{type(e).__name__}: {e}"})

    fk_violations = [dict(r) if isinstance(r, sqlite3.Row) else list(r) for r in fk_rows]
    details = {"integrity_check": integ, "foreign_key_violations": fk_violations}
    # Spec: integrity_check anything other than ["ok"] => error. FK violations on
    # an otherwise-sound DB are actionable-but-not-fatal => warn.
    if integ != ["ok"]:
        status: Status = "error"
    elif fk_violations:
        status = "warn"
    else:
        status = "ok"
    return done(status, details)


# ---------------------------------------------------------------------------
# Check 2 — schema diff vs scripts/schema.sql
# ---------------------------------------------------------------------------
_OBJECT_RES = {
    "table": re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'`]?(\w+)", re.IGNORECASE),
    "index": re.compile(
        r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'`]?(\w+)",
        re.IGNORECASE),
    "view": re.compile(
        r"CREATE\s+(?:TEMP\s+|TEMPORARY\s+)?VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'`]?(\w+)",
        re.IGNORECASE),
    "trigger": re.compile(
        r"CREATE\s+(?:TEMP\s+|TEMPORARY\s+)?TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'`]?(\w+)",
        re.IGNORECASE),
}


def _schema_objects(schema_sql_path: str) -> dict:
    """Return {kind: set(names)} parsed from schema.sql (line comments stripped)."""
    text = Path(schema_sql_path).read_text(encoding="utf-8")
    cleaned = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("--"))
    return {kind: {m.group(1).lower() for m in rx.finditer(cleaned)}
            for kind, rx in _OBJECT_RES.items()}


def _db_objects(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT type, name FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%'").fetchall()
    out = {"table": set(), "index": set(), "view": set(), "trigger": set()}
    for r in rows:
        t = (r[0] or "").lower()
        if t in out:
            out[t].add((r[1] or "").lower())
    return out


def check_schema_diff(db_path: str, schema_sql_path: str | None = None) -> CheckResult:
    name = "schema_diff"
    t0 = time.perf_counter()
    schema_sql_path = schema_sql_path or _default_schema_path()

    def done(status, details):
        return CheckResult(name, status, details,
                           round((time.perf_counter() - t0) * 1000, 3))

    if not os.path.exists(db_path):
        return done("error", {"error": f"db not found: {db_path}"})
    if not os.path.exists(schema_sql_path):
        return done("warn", {"error": f"schema file not found: {schema_sql_path}"})
    try:
        expected = _schema_objects(schema_sql_path)
    except Exception as e:
        return done("warn", {"error": f"could not parse schema: {e}"})
    try:
        conn = _connect(db_path)
        try:
            actual = _db_objects(conn)
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        return done("error", {"error": f"{type(e).__name__}: {e}"})

    missing, extra = {}, {}
    any_missing = any_extra = False
    for kind in ("table", "index", "view", "trigger"):
        m = sorted(expected[kind] - actual[kind])   # in schema.sql, absent from DB
        x = sorted(actual[kind] - expected[kind])    # in DB, absent from schema.sql
        missing[kind] = m
        extra[kind] = x
        any_missing = any_missing or bool(m)
        any_extra = any_extra or bool(x)

    details = {
        "expected_counts": {k: len(v) for k, v in expected.items()},
        "actual_counts": {k: len(v) for k, v in actual.items()},
        "missing": missing,   # present in schema.sql, absent from DB
        "extra": extra,       # present in DB, absent from schema.sql
    }
    # Spec: missing => warn, extras => info, otherwise ok.
    if any_missing:
        status: Status = "warn"
    elif any_extra:
        status = "info"
    else:
        status = "ok"
    return done(status, details)


# ---------------------------------------------------------------------------
# Check 3 — row counts + last write per table (pure reporting)
# ---------------------------------------------------------------------------
def check_row_counts(db_path: str) -> CheckResult:
    name = "row_counts"
    t0 = time.perf_counter()

    def done(status, details):
        return CheckResult(name, status, details,
                           round((time.perf_counter() - t0) * 1000, 3))

    if not os.path.exists(db_path):
        return done("error", {"error": f"db not found: {db_path}"})
    try:
        conn = _connect(db_path)
    except sqlite3.DatabaseError as e:
        return done("error", {"error": f"{type(e).__name__}: {e}"})

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
    # Spec: pure reporting — no warn/error thresholds.
    return done("ok", details)


# ---------------------------------------------------------------------------
# Check 4 — WAL state
# ---------------------------------------------------------------------------
def check_wal_state(db_path: str) -> CheckResult:
    name = "wal_state"
    t0 = time.perf_counter()

    def done(status, details):
        return CheckResult(name, status, details,
                           round((time.perf_counter() - t0) * 1000, 3))

    if not os.path.exists(db_path):
        return done("error", {"error": f"db not found: {db_path}"})

    def _size(p):
        try:
            return os.path.getsize(p)
        except OSError:
            return 0

    wal_size = _size(db_path + "-wal")
    details = {
        "db_size_bytes": _size(db_path),
        "wal_size_bytes": wal_size,
        "shm_size_bytes": _size(db_path + "-shm"),
        "wal_warn_threshold_bytes": _WAL_WARN_BYTES,
    }
    try:
        conn = _connect(db_path)
        try:
            details["journal_mode"] = conn.execute(
                "PRAGMA journal_mode").fetchone()[0]
            # PASSIVE never blocks writers; returns (busy, log, checkpointed).
            cp = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            if cp is not None:
                details["wal_checkpoint"] = {
                    "busy": cp[0], "log_frames": cp[1], "checkpointed_frames": cp[2]}
                details["checkpoint_lag_frames"] = (cp[1] or 0) - (cp[2] or 0)
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        return done("error", {**details, "error": f"{type(e).__name__}: {e}"})

    if wal_size > _WAL_WARN_BYTES:
        details["warning"] = (
            f"WAL file {wal_size} bytes exceeds {_WAL_WARN_BYTES} threshold; "
            "checkpoint may be starved by long-lived readers/writers")
        return done("warn", details)
    return done("ok", details)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def run_all_checks(db_path: str) -> list[CheckResult]:
    """Run all four checks in spec order and return the results list."""
    schema_path = _default_schema_path()
    return [
        check_integrity(db_path),
        check_schema_diff(db_path, schema_path),
        check_row_counts(db_path),
        check_wal_state(db_path),
    ]


def _jsonable(obj):
    """Coerce SQLite/dataclass values into something ``json.dumps`` accepts.

    SQLite can return ``bytes`` (e.g. MAX() over a BLOB-affinity column) which
    Flask's jsonify rejects. Kept for the admin endpoint / callers that need a
    guaranteed-serializable payload.
    """
    if isinstance(obj, CheckResult):
        return _jsonable(asdict(obj))
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("utf-8", errors="replace")
    if isinstance(obj, sqlite3.Row):
        return {k: _jsonable(obj[k]) for k in obj.keys()}
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else str(k)): _jsonable(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def cli_main() -> int:
    parser = argparse.ArgumentParser(description="DMAI manual DB health checks")
    parser.add_argument("--db", default="data/dmai_knowledge.db",
                        help="path to the SQLite DB (default: data/dmai_knowledge.db)")
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="compact JSON (default)")
    fmt.add_argument("--pretty", action="store_true", help="pretty-printed JSON")
    args = parser.parse_args()

    results = run_all_checks(args.db)
    overall = worst_status(results)
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "db_path": args.db,
        "overall_status": overall,
        "checks": [_jsonable(r) for r in results],
    }
    if args.pretty:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(json.dumps(payload, default=str))
    return exit_code_for(overall)


if __name__ == "__main__":
    sys.exit(cli_main())
