#!/usr/bin/env python3
"""DMAI DB migration orchestrator.

Dry-run by default — mutations require ``--apply``. Reuses the health-check
functions from db_health so there is one source of truth for "is this DB ok".

Order of operations (strict; mutations only happen with --apply):
  1. integrity check   — refuse to proceed if not ok (VACUUM on a corrupt DB
                         destroys data).
  2. foreign-key check — dry-run continues on violations; --apply refuses.
  3. backup            — sqlite3 Connection.backup() (+ copy of -wal). --apply only.
  4. VACUUM            — --apply only, and only if integrity is ok.
  5. recreate missing tables from schema.sql (CREATE TABLE IF NOT EXISTS).
  6. ANALYZE.

Every run (dry-run and apply) appends one JSON line to
data/migrations/migration_log.jsonl.

CLI:
  python scripts/db_migrate.py [--db PATH] [--apply] [--json|--pretty]

Exit code: 0=ok, 1=warn, 2=error/refused.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# Single source of truth with the CLI/endpoint health suite.
try:
    from db_health import check_integrity, _default_schema_path, _list_tables, _connect
except ImportError:  # invoked from repo root / as scripts.db_migrate
    from scripts.db_health import (  # type: ignore
        check_integrity, _default_schema_path, _list_tables, _connect)

_EXIT_CODE = {"ok": 0, "warn": 1, "error": 2, "refused": 2}
_STATUS_RANK = {"ok": 0, "warn": 1, "error": 2}
_LOG_PATH = os.path.join("data", "migrations", "migration_log.jsonl")
_BACKUP_ROOT = os.path.join("data", "migrations", "backups")

# Statuses that a step may report but that do not count against the overall roll-up.
_NEUTRAL_STEP = {"skipped", "info"}


@dataclass
class StepResult:
    name: str
    status: str
    duration_ms: float
    details: dict = field(default_factory=dict)


@dataclass
class MigrationReport:
    started_at: str
    finished_at: str
    mode: Literal["dry-run", "apply"]
    db_path: str
    steps: list = field(default_factory=list)
    backup_path: str | None = None
    overall_status: Literal["ok", "warn", "error", "refused"] = "ok"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _worst(statuses) -> str:
    worst = "ok"
    for s in statuses:
        if _STATUS_RANK.get(s, 0) > _STATUS_RANK[worst]:
            worst = s
    return worst


def _split_statements(schema_sql: str):
    """Split schema.sql into individual statements (line comments stripped)."""
    body = "\n".join(
        ln for ln in schema_sql.splitlines() if not ln.lstrip().startswith("--"))
    return [s.strip() for s in body.split(";") if s.strip()]


def _backup(db_path: str) -> str:
    """Back up the DB via the SQLite backup API (not a file copy). Returns dest path."""
    dbname = os.path.basename(db_path)
    ts = _now_iso()
    dest_dir = os.path.join(_BACKUP_ROOT, f"{dbname}_{ts}")
    os.makedirs(dest_dir, exist_ok=True)
    dest_db = os.path.join(dest_dir, dbname)
    src = sqlite3.connect(db_path, timeout=30.0)
    try:
        dst = sqlite3.connect(dest_db)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    # Copy the -wal sidecar too, if present (point-in-time completeness).
    wal = db_path + "-wal"
    if os.path.exists(wal):
        shutil.copy2(wal, os.path.join(dest_dir, dbname + "-wal"))
    return dest_db


def _timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, round((time.perf_counter() - t0) * 1000, 3)


def run_migration(db_path: str, apply: bool) -> MigrationReport:
    mode = "apply" if apply else "dry-run"
    report = MigrationReport(started_at=_now_iso(), finished_at="",
                             mode=mode, db_path=db_path)
    schema_sql_path = _default_schema_path()

    def add(name, status, details=None, duration_ms=0.0):
        report.steps.append(StepResult(name, status, duration_ms, details or {}))

    def finish(overall):
        report.overall_status = overall
        report.finished_at = _now_iso()
        _write_log(report)
        return report

    if not os.path.exists(db_path):
        add("preflight", "error", {"error": f"db not found: {db_path}"})
        return finish("error")

    # 1. integrity — refuse to mutate a corrupt DB -------------------------
    integ = check_integrity(db_path)
    add("integrity", integ.status, integ.details, integ.duration_ms)
    if integ.status != "ok":
        # integrity error OR fk-warn from integrity? Only integrity!=ok blocks.
        if integ.details.get("integrity_check") != ["ok"]:
            add("refuse", "refused",
                {"reason": "integrity_check is not ok; VACUUM/backup on a corrupt "
                           "DB can destroy data. Aborting."})
            return finish("refused")

    # 2. foreign-key check -------------------------------------------------
    fk_violations = integ.details.get("foreign_key_violations", [])
    if fk_violations:
        if apply:
            add("foreign_keys", "refused",
                {"violations": fk_violations,
                 "reason": "foreign-key violations present; refusing to mutate in "
                           "--apply mode. Re-run dry-run to inspect."})
            return finish("refused")
        add("foreign_keys", "warn",
            {"violations": fk_violations,
             "note": "dry-run: continuing despite FK violations (would refuse in --apply)"})
    else:
        add("foreign_keys", "ok", {"violations": []})

    # Which tables are missing (drives step 5 reporting) -------------------
    try:
        conn = _connect(db_path)
        try:
            existing = {t.lower() for t in _list_tables(conn)}
        finally:
            conn.close()
        schema_sql = Path(schema_sql_path).read_text(encoding="utf-8")
        import re as _re
        schema_tables = {m.group(1).lower() for m in _re.finditer(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'`]?(\w+)",
            "\n".join(l for l in schema_sql.splitlines()
                      if not l.lstrip().startswith("--")), _re.IGNORECASE)}
        missing_tables = sorted(schema_tables - existing)
    except Exception as e:
        missing_tables = []
        add("schema_scan", "warn", {"error": str(e)})

    # ---- dry-run: report intentions, mutate nothing ---------------------
    if not apply:
        add("backup", "skipped", {"reason": "dry-run"})
        add("vacuum", "skipped", {"reason": "dry-run"})
        add("recreate_tables", "skipped",
            {"reason": "dry-run", "would_create": missing_tables})
        add("analyze", "skipped", {"reason": "dry-run"})
        overall = _worst([s.status for s in report.steps
                          if s.status not in _NEUTRAL_STEP and s.status not in ("skipped",)])
        return finish(overall)

    # ---- APPLY path -----------------------------------------------------
    # 3. backup
    try:
        (dest, dur) = _timed(lambda: _backup(db_path))
        report.backup_path = dest
        add("backup", "ok", {"backup_path": dest}, dur)
    except Exception as e:
        add("backup", "error", {"error": str(e)})
        return finish("error")

    # 4. VACUUM (integrity is ok — guaranteed, else we'd have refused)
    try:
        def _vacuum():
            c = sqlite3.connect(db_path, timeout=60.0)
            try:
                c.execute("VACUUM")
                c.commit()
            finally:
                c.close()
        _, dur = _timed(_vacuum)
        add("vacuum", "ok", {}, dur)
    except Exception as e:
        add("vacuum", "error", {"error": str(e)})

    # 5. recreate missing tables from schema.sql (idempotent)
    try:
        def _recreate():
            c = sqlite3.connect(db_path, timeout=60.0)
            stmt_errors = []
            try:
                before = {r[0].lower() for r in c.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'").fetchall()}
                for stmt in _split_statements(schema_sql):
                    if not stmt.upper().lstrip().startswith("CREATE TABLE"):
                        continue
                    try:
                        c.execute(stmt)
                    except sqlite3.OperationalError as e:
                        stmt_errors.append(f"{stmt[:60]}...: {e}")
                c.commit()
                after = {r[0].lower() for r in c.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'").fetchall()}
            finally:
                c.close()
            return sorted(after - before), stmt_errors
        (created, stmt_errors), dur = _timed(_recreate)
        add("recreate_tables", "warn" if stmt_errors else "ok",
            {"created": created, "stmt_errors": stmt_errors}, dur)
    except Exception as e:
        add("recreate_tables", "error", {"error": str(e)})

    # 6. ANALYZE
    try:
        def _analyze():
            c = sqlite3.connect(db_path, timeout=60.0)
            try:
                c.execute("ANALYZE")
                c.commit()
            finally:
                c.close()
        _, dur = _timed(_analyze)
        add("analyze", "ok", {}, dur)
    except Exception as e:
        add("analyze", "error", {"error": str(e)})

    overall = _worst([s.status for s in report.steps
                      if s.status not in _NEUTRAL_STEP and s.status != "skipped"])
    return finish(overall)


def _write_log(report: MigrationReport) -> None:
    try:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dataclasses.asdict(report), default=str) + "\n")
    except Exception as e:
        # Logging must never crash the migration.
        sys.stderr.write(f"[warn] could not append migration log: {e}\n")


def cli_main() -> int:
    parser = argparse.ArgumentParser(description="DMAI DB migration orchestrator")
    parser.add_argument("--db", default="data/dmai_knowledge.db")
    parser.add_argument("--apply", action="store_true",
                        help="perform mutations (default: dry-run)")
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="compact JSON (default)")
    fmt.add_argument("--pretty", action="store_true", help="pretty-printed JSON")
    args = parser.parse_args()

    report = run_migration(args.db, apply=args.apply)
    payload = dataclasses.asdict(report)
    if args.pretty:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(json.dumps(payload, default=str))
    return _EXIT_CODE.get(report.overall_status, 2)


if __name__ == "__main__":
    sys.exit(cli_main())
