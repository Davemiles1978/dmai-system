#!/usr/bin/env python3
"""DMAI DB migration orchestrator.

Dry-run by default — mutations require ``--apply``. Reuses the health-check
functions from db_health so there is one source of truth for "is this DB ok".

Order of operations (mutations only happen with --apply):
  1. integrity check          (abort before mutating a corrupt DB)
  2. foreign-key check
  3. backup                   sqlite3 Connection.backup() + copy of -wal/-shm
  4. VACUUM                   refused unless integrity == "ok"
  5. CREATE TABLE IF NOT EXISTS from schema.sql   (idempotent)
  6. ANALYZE

Every run appends one JSON line to data/migrations/migration_log.jsonl.

CLI:
  python scripts/db_migrate.py [--db PATH] [--schema PATH] [--apply] [--json]

Exit code: 0=ok, 1=warn, 2=fail/abort.
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
from pathlib import Path

# Import the shared check functions — single source of truth with the CLI/endpoint.
try:
    from db_health import (
        check_integrity,
        check_schema_drift,
        _schema_table_names,
        _list_tables,
        _connect,
    )
except ImportError:  # when invoked as `python -m scripts.db_migrate` or from repo root
    from scripts.db_health import (  # type: ignore
        check_integrity,
        check_schema_drift,
        _schema_table_names,
        _list_tables,
        _connect,
    )

_EXIT_CODE = {"ok": 0, "warn": 1, "fail": 2}
_LOG_PATH = os.path.join("data", "migrations", "migration_log.jsonl")
_BACKUP_ROOT = os.path.join("data", "migrations", "backups")


@dataclass
class MigrationReport:
    ts: str
    db_path: str
    apply: bool
    steps: list = field(default_factory=list)
    backup_path: str | None = None
    started_integrity: str | None = None
    final_integrity: str | None = None
    tables_created: list = field(default_factory=list)
    vacuum_ran: bool = False
    analyze_ran: bool = False
    errors: list = field(default_factory=list)
    overall_status: str = "ok"

    def step(self, name: str, status: str, **info) -> None:
        self.steps.append({"name": name, "status": status, **info})


def check_foreign_keys(db_path: str) -> dict:
    """FK-only view derived from check_integrity's foreign_key_check result."""
    res = check_integrity(db_path)
    fk = res["details"].get("foreign_key_violations", [])
    status = "ok" if not fk else "warn"
    if res["status"] == "fail":
        status = "fail"
    return {"name": "foreign_keys", "status": status,
            "details": {"violations": fk}}


def _split_statements(schema_sql: str):
    """Split schema.sql into individual statements (comments stripped)."""
    lines = [ln for ln in schema_sql.splitlines() if not ln.lstrip().startswith("--")]
    body = "\n".join(lines)
    return [s.strip() for s in body.split(";") if s.strip()]


def _backup(db_path: str, ts: str, report: MigrationReport) -> str:
    dbname = os.path.basename(db_path)
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
    # Also copy the WAL/SHM sidecars if present (point-in-time completeness).
    for suffix in ("-wal", "-shm"):
        side = db_path + suffix
        if os.path.exists(side):
            shutil.copy2(side, os.path.join(dest_dir, dbname + suffix))
    report.step("backup", "ok", backup_path=dest_db)
    return dest_db


def run_migration(db_path: str, schema_sql_path: str, apply: bool) -> MigrationReport:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report = MigrationReport(ts=ts, db_path=db_path, apply=apply)

    if not os.path.exists(db_path):
        report.errors.append(f"db not found: {db_path}")
        report.step("preflight", "fail", error=f"db not found: {db_path}")
        report.overall_status = "fail"
        return report

    # 1. integrity --------------------------------------------------------
    integ = check_integrity(db_path)
    report.started_integrity = integ["status"]
    report.step("integrity", integ["status"], details=integ["details"])
    if integ["status"] == "fail":
        report.errors.append("integrity_check failed — refusing to mutate a corrupt DB")
        report.overall_status = "fail"
        # Do NOT proceed to VACUUM/backup-mutate on a corrupt DB.
        _write_log(report)
        return report

    # 2. foreign keys -----------------------------------------------------
    fk = check_foreign_keys(db_path)
    report.step("foreign_keys", fk["status"], details=fk["details"])

    # schema drift (informational; drives which tables we will create) ----
    drift = check_schema_drift(db_path, schema_sql_path)
    report.step("schema_drift_initial", drift["status"], details=drift["details"])
    missing_tables = drift["details"].get("missing_tables", []) if drift else []

    statuses = [integ["status"], fk["status"], drift["status"]]

    if not apply:
        # Dry-run: report what WOULD be created, mutate nothing.
        report.step("backup", "skipped", reason="dry-run")
        report.step("vacuum", "skipped", reason="dry-run")
        report.step("create_tables", "skipped", reason="dry-run",
                    would_create=missing_tables)
        report.step("analyze", "skipped", reason="dry-run")
        report.overall_status = _worst(statuses)
        _write_log(report)
        return report

    # ---- APPLY path -----------------------------------------------------
    # 3. backup
    try:
        report.backup_path = _backup(db_path, ts, report)
    except Exception as e:
        report.errors.append(f"backup failed: {e}")
        report.step("backup", "fail", error=str(e))
        report.overall_status = "fail"
        _write_log(report)
        return report

    # 4. VACUUM (only if integrity ok — guaranteed here since we'd have aborted)
    if integ["status"] == "ok":
        try:
            conn = sqlite3.connect(db_path, timeout=60.0)
            try:
                conn.execute("VACUUM")
                conn.commit()
            finally:
                conn.close()
            report.vacuum_ran = True
            report.step("vacuum", "ok")
        except Exception as e:
            report.errors.append(f"vacuum failed: {e}")
            report.step("vacuum", "fail", error=str(e))
    else:
        report.step("vacuum", "skipped", reason=f"integrity={integ['status']}")

    # 5. CREATE TABLE IF NOT EXISTS from schema.sql
    try:
        before = set(t.lower() for t in _list_tables(_connect(db_path)))
        schema_sql = Path(schema_sql_path).read_text(encoding="utf-8")
        conn = sqlite3.connect(db_path, timeout=60.0)
        stmt_errors = []
        try:
            for stmt in _split_statements(schema_sql):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    stmt_errors.append(f"{stmt[:60]}...: {e}")
            conn.commit()
            after = {r[0].lower() for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'").fetchall()}
        finally:
            conn.close()
        report.tables_created = sorted(after - before)
        # Tables are created with IF NOT EXISTS; individual index statements may
        # fail against a pre-existing drifted table (missing column). That is a
        # warn-level signal for manual attention, not a hard migration failure —
        # so it is recorded in the step but NOT added to report.errors (which
        # would force overall=fail).
        step_status = "ok" if not stmt_errors else "warn"
        report.step("create_tables", step_status,
                    created=report.tables_created, stmt_errors=stmt_errors)
    except Exception as e:
        report.errors.append(f"create_tables failed: {e}")
        report.step("create_tables", "fail", error=str(e))

    # 6. ANALYZE
    try:
        conn = sqlite3.connect(db_path, timeout=60.0)
        try:
            conn.execute("ANALYZE")
            conn.commit()
        finally:
            conn.close()
        report.analyze_ran = True
        report.step("analyze", "ok")
    except Exception as e:
        report.errors.append(f"analyze failed: {e}")
        report.step("analyze", "fail", error=str(e))

    # final integrity + drift re-check (post-mutation state is what matters now)
    final = check_integrity(db_path)
    report.final_integrity = final["status"]
    report.step("final_integrity", final["status"], details=final["details"])
    drift_final = check_schema_drift(db_path, schema_sql_path)
    report.step("schema_drift", drift_final["status"], details=drift_final["details"])

    # Overall reflects the resulting state: the pre-mutation drift ("schema_drift_initial")
    # is the reason we ran and is intentionally excluded.
    counted = {"integrity", "foreign_keys", "create_tables", "vacuum",
               "analyze", "final_integrity", "schema_drift"}
    all_statuses = [s["status"] for s in report.steps
                    if s["name"] in counted and s["status"] in _EXIT_CODE]
    report.overall_status = "fail" if report.errors else _worst(all_statuses)
    _write_log(report)
    return report


def _worst(statuses) -> str:
    rank = {"ok": 0, "warn": 1, "fail": 2}
    worst = "ok"
    for s in statuses:
        if rank.get(s, 0) > rank[worst]:
            worst = s
    return worst


def _write_log(report: MigrationReport) -> None:
    try:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dataclasses.asdict(report), default=str) + "\n")
    except Exception as e:
        # Logging must never crash the migration.
        sys.stderr.write(f"[warn] could not append migration log: {e}\n")


def _default_schema_path() -> str:
    return str(Path(__file__).resolve().parent / "schema.sql")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DMAI DB migration orchestrator")
    parser.add_argument("--db", default="data/dmai_knowledge.db")
    parser.add_argument("--schema", default=_default_schema_path())
    parser.add_argument("--apply", action="store_true",
                        help="perform mutations (default: dry-run)")
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    args = parser.parse_args(argv)

    report = run_migration(args.db, args.schema, apply=args.apply)
    payload = dataclasses.asdict(report)
    if args.json:
        print(json.dumps(payload, default=str))
    else:
        print(json.dumps(payload, indent=2, default=str))
    return _EXIT_CODE.get(report.overall_status, 2)


if __name__ == "__main__":
    sys.exit(main())
