#!/usr/bin/env python3
"""DMAI quarantined-DB row recovery tool.

Report-first recovery of rows from corrupt/quarantined SQLite files back into
the live DB. Pure-stdlib. Two ways to use it:

  1. As a library — import ``recover`` (or the per-source ``recover_source``).
  2. As a CLI:
       python scripts/db_recover.py --sources PATH [PATH ...] \\
           [--target data/dmai_knowledge.db] [--apply] [--json|--pretty] \\
           [--log-dir data/migrations/recover_log]

Model (mirrors PR #168 db_health): dry-run is the default, ``--apply`` gates
all mutations, every run appends a JSONL audit line, and the source files are
NEVER touched — we always work from a temp copy.

For each source: copy → ``sqlite3 <copy> .recover`` → load recovered SQL into an
in-memory scratch DB → for each scratch table that also exists in target,
diff primary keys to find rows the target is missing. In ``--apply`` mode we
snapshot the target first, then ``INSERT OR IGNORE`` the missing rows.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

DEFAULT_TARGET = "data/dmai_knowledge.db"
DEFAULT_LOG_DIR = "data/migrations/recover_log"
_RECOVER_TIMEOUT_S = 60


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TableRecovery:
    table: str
    status: Literal[
        "recoverable", "no_new_rows", "orphan_table", "skipped_no_pk", "insert_failed"]
    source_rows: int
    target_rows: int
    recoverable_rows: int
    inserted_rows: int  # only populated in --apply mode
    error: str | None = None


@dataclass
class SourceRecovery:
    source_path: str
    source_size_bytes: int
    recover_status: Literal["ok", "partial", "unrecoverable"]
    recover_sql_path: str | None
    tables: list[TableRecovery]
    total_recoverable: int
    total_inserted: int
    error: str | None = None


@dataclass
class RecoveryReport:
    started_at: str  # UTC ISO
    finished_at: str
    mode: Literal["dry-run", "apply"]
    target_path: str
    sources: list[SourceRecovery]
    overall_recoverable: int
    overall_inserted: int
    backup_path: str | None


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _pk_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Primary-key columns in PK order. Empty list => no declared PK."""
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    pk = [(r[5], r[1]) for r in rows if r[5]]  # (pk_index, name) where pk_index>0
    pk.sort(key=lambda t: t[0])
    return [name for _, name in pk]


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [r[1] for r in rows]


def _run_recover(temp_source: str, out_sql_path: str) -> tuple[bool, str]:
    """Run ``sqlite3 <temp_source> .recover`` capturing stdout to a file.

    Returns (produced_output, error_message). Never raises for a bad DB;
    only returns False so the caller can mark the source unrecoverable.
    """
    try:
        proc = subprocess.run(
            ["sqlite3", temp_source, ".recover"],
            capture_output=True, text=True, timeout=_RECOVER_TIMEOUT_S,
        )
    except FileNotFoundError:
        return False, "sqlite3 CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return False, f".recover timed out after {_RECOVER_TIMEOUT_S}s"
    except Exception as e:  # pragma: no cover - defensive
        return False, f"{type(e).__name__}: {e}"

    stdout = proc.stdout or ""
    Path(out_sql_path).write_text(stdout, encoding="utf-8")
    if proc.returncode != 0 and not stdout.strip():
        return False, (proc.stderr or "").strip()[:500] or f"exit {proc.returncode}"
    if not stdout.strip():
        return False, "empty .recover output"
    return True, ""


def _load_scratch(recover_sql: str) -> tuple[sqlite3.Connection, str | None]:
    """Load recovered SQL into an in-memory DB.

    Returns (conn, load_error). If executescript fails partway, we still return
    the connection with whatever objects loaded before the error — some data is
    better than none. ``load_error`` is set when the load was partial.
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(recover_sql)
        return conn, None
    except sqlite3.DatabaseError as e:
        return conn, f"{type(e).__name__}: {e}"


def _target_pk_set(target: str, table: str, pk_cols: list[str]) -> set:
    """Read the target's PK tuples for a table via a read-only connection."""
    ro = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=10.0)
    try:
        cols = ", ".join(f'"{c}"' for c in pk_cols)
        rows = ro.execute(f'SELECT {cols} FROM "{table}"').fetchall()
        return {tuple(r) for r in rows}
    finally:
        ro.close()


def _target_tables(target: str) -> set[str]:
    ro = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=10.0)
    try:
        return set(_list_tables(ro))
    finally:
        ro.close()


def _backup_target(target: str, dest: str) -> None:
    """Snapshot the target DB using the SQLite backup API. Raises on failure."""
    src = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=30.0)
    try:
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


# ---------------------------------------------------------------------------
# Per-source recovery
# ---------------------------------------------------------------------------
def recover_source(
    source_path: str,
    target_path: str,
    log_dir: str,
    apply: bool,
) -> SourceRecovery:
    """Recover rows from a single quarantined source. Read-only w.r.t. source."""
    size = os.path.getsize(source_path) if os.path.exists(source_path) else 0
    basename = os.path.basename(source_path)
    recover_sql_path = str(Path(log_dir) / f"{basename}.recover.sql")

    if not os.path.exists(source_path):
        return SourceRecovery(
            source_path=source_path, source_size_bytes=0,
            recover_status="unrecoverable", recover_sql_path=None,
            tables=[], total_recoverable=0, total_inserted=0,
            error=f"source not found: {source_path}")

    # 1. Copy to a temp file — never operate on the quarantined original.
    tmp_dir = tempfile.mkdtemp(prefix="db_recover_")
    temp_source = os.path.join(tmp_dir, basename + ".copy")
    try:
        shutil.copy2(source_path, temp_source)

        # 2. sqlite3 .recover → capture SQL.
        produced, rec_err = _run_recover(temp_source, recover_sql_path)
        if not produced:
            return SourceRecovery(
                source_path=source_path, source_size_bytes=size,
                recover_status="unrecoverable", recover_sql_path=recover_sql_path,
                tables=[], total_recoverable=0, total_inserted=0,
                error=rec_err or "unrecoverable")

        recover_sql = Path(recover_sql_path).read_text(encoding="utf-8")

        # 3. Load into in-memory scratch DB (partial load tolerated).
        scratch, load_err = _load_scratch(recover_sql)
        try:
            # .recover emits a boilerplate BEGIN/PRAGMA skeleton even for pure
            # garbage input, so non-empty stdout is not proof of recovery.
            # Zero recovered tables => nothing to salvage => unrecoverable.
            if not _list_tables(scratch):
                return SourceRecovery(
                    source_path=source_path, source_size_bytes=size,
                    recover_status="unrecoverable",
                    recover_sql_path=recover_sql_path, tables=[],
                    total_recoverable=0, total_inserted=0,
                    error=load_err or "no tables recovered")
            recover_status: Literal["ok", "partial", "unrecoverable"] = (
                "partial" if load_err else "ok")
            table_reports, total_recoverable, total_inserted = _process_scratch(
                scratch, target_path, apply)
        finally:
            scratch.close()

        return SourceRecovery(
            source_path=source_path, source_size_bytes=size,
            recover_status=recover_status, recover_sql_path=recover_sql_path,
            tables=table_reports, total_recoverable=total_recoverable,
            total_inserted=total_inserted, error=load_err)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _process_scratch(
    scratch: sqlite3.Connection,
    target_path: str,
    apply: bool,
) -> tuple[list[TableRecovery], int, int]:
    target_tables = _target_tables(target_path)
    reports: list[TableRecovery] = []
    total_recoverable = 0
    total_inserted = 0

    for table in _list_tables(scratch):
        try:
            source_rows = scratch.execute(
                f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        except sqlite3.DatabaseError as e:
            reports.append(TableRecovery(
                table=table, status="insert_failed", source_rows=0,
                target_rows=0, recoverable_rows=0, inserted_rows=0,
                error=f"count failed: {e}"))
            continue

        # 6. Orphan table: exists in scratch, not in target → do not insert.
        if table not in target_tables:
            reports.append(TableRecovery(
                table=table, status="orphan_table", source_rows=source_rows,
                target_rows=0, recoverable_rows=0, inserted_rows=0))
            continue

        pk_cols = _pk_columns(scratch, table)

        # No PK: report-only via ROWID; skip entirely in --apply mode.
        if not pk_cols:
            if apply:
                reports.append(TableRecovery(
                    table=table, status="skipped_no_pk", source_rows=source_rows,
                    target_rows=0, recoverable_rows=0, inserted_rows=0,
                    error="no primary key; refused in apply mode"))
                continue
            reports.append(_report_no_pk(scratch, target_path, table, source_rows))
            total_recoverable += reports[-1].recoverable_rows
            continue

        try:
            src_pks = {
                tuple(r) for r in scratch.execute(
                    f'SELECT {", ".join(chr(34)+c+chr(34) for c in pk_cols)} '
                    f'FROM "{table}"').fetchall()}
            tgt_pks = _target_pk_set(target_path, table, pk_cols)
            target_rows = len(tgt_pks)
        except sqlite3.DatabaseError as e:
            reports.append(TableRecovery(
                table=table, status="insert_failed", source_rows=source_rows,
                target_rows=0, recoverable_rows=0, inserted_rows=0,
                error=f"pk diff failed: {e}"))
            continue

        missing_pks = src_pks - tgt_pks
        recoverable = len(missing_pks)
        total_recoverable += recoverable

        if recoverable == 0:
            reports.append(TableRecovery(
                table=table, status="no_new_rows", source_rows=source_rows,
                target_rows=target_rows, recoverable_rows=0, inserted_rows=0))
            continue

        if not apply:
            reports.append(TableRecovery(
                table=table, status="recoverable", source_rows=source_rows,
                target_rows=target_rows, recoverable_rows=recoverable,
                inserted_rows=0))
            continue

        # --apply: insert the missing rows for this table.
        inserted, ins_err = _insert_missing(
            scratch, target_path, table, pk_cols, missing_pks)
        total_inserted += inserted
        reports.append(TableRecovery(
            table=table,
            status="insert_failed" if ins_err else "recoverable",
            source_rows=source_rows, target_rows=target_rows,
            recoverable_rows=recoverable, inserted_rows=inserted, error=ins_err))

    return reports, total_recoverable, total_inserted


def _report_no_pk(
    scratch: sqlite3.Connection, target_path: str, table: str, source_rows: int,
) -> TableRecovery:
    """Reporting-only PK-less diff using ROWID as a fallback identity."""
    try:
        src_ids = {
            r[0] for r in scratch.execute(f'SELECT ROWID FROM "{table}"').fetchall()}
        tgt_ids = _target_pk_set(target_path, table, ["ROWID"])
        tgt_ids = {t[0] for t in tgt_ids}
        target_rows = len(tgt_ids)
        recoverable = len(src_ids - tgt_ids)
    except sqlite3.DatabaseError as e:
        return TableRecovery(
            table=table, status="skipped_no_pk", source_rows=source_rows,
            target_rows=0, recoverable_rows=0, inserted_rows=0,
            error=f"rowid diff failed: {e}")
    return TableRecovery(
        table=table, status="skipped_no_pk", source_rows=source_rows,
        target_rows=target_rows, recoverable_rows=recoverable, inserted_rows=0,
        error="no primary key; ROWID-based report only")


def _insert_missing(
    scratch: sqlite3.Connection,
    target_path: str,
    table: str,
    pk_cols: list[str],
    missing_pks: set,
) -> tuple[int, str | None]:
    """INSERT OR IGNORE the missing rows into the target, in one transaction.

    Returns (inserted_count, error). On failure we roll back this table's
    inserts and return an error string; the caller continues to the next table.
    """
    cols = _table_columns(scratch, table)
    col_list = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join("?" for _ in cols)
    where = " AND ".join(f'"{c}" IS ?' for c in pk_cols)
    select_sql = f'SELECT {col_list} FROM "{table}" WHERE {where}'

    rows_to_insert = []
    for pk in missing_pks:
        rows_to_insert.extend(scratch.execute(select_sql, pk).fetchall())

    conn = sqlite3.connect(target_path, timeout=30.0)
    try:
        conn.execute("BEGIN")
        conn.executemany(
            f'INSERT OR IGNORE INTO "{table}" ({col_list}) VALUES ({placeholders})',
            rows_to_insert)
        inserted = conn.total_changes
        conn.commit()
        return inserted, None
    except sqlite3.DatabaseError as e:
        conn.rollback()
        return 0, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def recover(
    sources: list[str],
    target_path: str = DEFAULT_TARGET,
    apply: bool = False,
    log_dir: str = DEFAULT_LOG_DIR,
) -> RecoveryReport:
    """Recover rows from all ``sources`` into ``target_path``. Report-first."""
    os.makedirs(log_dir, exist_ok=True)
    started_at = _utc_iso()
    mode: Literal["dry-run", "apply"] = "apply" if apply else "dry-run"

    backup_path: str | None = None
    if apply:
        # Snapshot the target first. If backup fails, refuse to proceed.
        backup_path = str(
            Path(log_dir) / f"target_pre_recover_{_utc_iso().replace(':', '')}.db")
        _backup_target(target_path, backup_path)

    source_reports: list[SourceRecovery] = []
    overall_recoverable = 0
    overall_inserted = 0
    for src in sources:
        rep = recover_source(src, target_path, log_dir, apply)
        source_reports.append(rep)
        overall_recoverable += rep.total_recoverable
        overall_inserted += rep.total_inserted

    report = RecoveryReport(
        started_at=started_at, finished_at=_utc_iso(), mode=mode,
        target_path=target_path, sources=source_reports,
        overall_recoverable=overall_recoverable, overall_inserted=overall_inserted,
        backup_path=backup_path)

    # Always append one JSONL audit line.
    with open(Path(log_dir) / "recovery_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(report), default=str) + "\n")

    return report


def cli_main() -> int:
    parser = argparse.ArgumentParser(
        description="DMAI quarantined-DB row recovery (report-first)")
    parser.add_argument("--sources", nargs="+", required=True,
                        help="quarantined DB file(s) to recover rows from")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help=f"live DB to insert into (default: {DEFAULT_TARGET})")
    parser.add_argument("--apply", action="store_true",
                        help="actually insert missing rows (default: report only)")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR,
                        help=f"reports + .recover dumps (default: {DEFAULT_LOG_DIR})")
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="compact JSON (default)")
    fmt.add_argument("--pretty", action="store_true", help="pretty-printed JSON")
    args = parser.parse_args()

    try:
        report = recover(args.sources, args.target, args.apply, args.log_dir)
    except Exception as e:
        payload = {"error": f"{type(e).__name__}: {e}", "mode":
                   "apply" if args.apply else "dry-run"}
        print(json.dumps(payload, indent=2 if args.pretty else None))
        return 2

    payload = asdict(report)
    if args.pretty:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(json.dumps(payload, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
