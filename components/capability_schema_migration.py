"""Additive, idempotent schema migration for the ``capabilities`` table.

Brings a legacy prod ``capabilities`` table (old registry vocabulary:
``runtime_mode`` in ``'ondemand'``/``'autonomous'``, no provenance /
confidence columns) into the shape the self-generation materialiser
expects (``provenance``, ``judge_confidence``, and ``runtime_mode`` in
``'stub'``/``'stub_reverted'``/``'live'``).

Safety guarantees:

- **Additive only.** No columns are dropped, no rows are deleted, and
  no existing values are overwritten in destructive ways.
- **Idempotent.** Every step checks whether it has already been done
  and is a no-op if so, so re-running the migration is safe.
- **Legacy rows stay pickable-off.** The 20k legacy ``ondemand`` /
  ``autonomous`` rows are tagged with ``provenance='legacy_ondemand'``
  or ``'legacy_autonomous'`` and left in their original runtime_mode,
  so the materialiser's picker (which only picks
  ``stub``/``stub_reverted``) never touches them. New gap-seeded rows
  land as ``runtime_mode='stub'``, ``provenance='gap_driven'``.

Callable as a library (returns a report dict) and as a Flask endpoint
(``POST /api/admin/capabilities/migrate-schema``, cron-secret gated).
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import sqlite3
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _resolve_db_path() -> str:
    return os.environ.get("DMAI_KNOWLEDGE_DB", "data/dmai_knowledge.db")


def _existing_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in rows]


def _add_column_if_missing(conn: sqlite3.Connection,
                           table: str,
                           column: str,
                           column_def: str,
                           ) -> bool:
    """Add a column iff not present. Returns True if it was added."""
    if column in _existing_columns(conn, table):
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")
    return True


def migrate_capabilities_schema(*,
                                dry_run: bool = False,
                                db_path: str | None = None,
                                ) -> Dict[str, Any]:
    """Bring the capabilities table up to materialiser shape.

    Steps performed (each idempotent):
      1. ADD COLUMN ``provenance TEXT`` if missing
      2. ADD COLUMN ``judge_confidence REAL`` if missing
      3. Backfill ``provenance='legacy_ondemand'`` for rows where
         ``runtime_mode='ondemand'`` and ``provenance IS NULL``
      4. Backfill ``provenance='legacy_autonomous'`` for rows where
         ``runtime_mode='autonomous'`` and ``provenance IS NULL``
      5. Create index on (runtime_mode, provenance) for picker perf
      6. Ensure a ``materialisation_log`` table exists with the shape
         the picker's 24h-backoff scan reads

    Never modifies runtime_mode values on existing rows.

    Returns a report dict describing what happened.
    """
    path = db_path or _resolve_db_path()
    report: Dict[str, Any] = {
        "ok": True,
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "db_path": path,
        "dry_run": bool(dry_run),
        "steps": [],
    }

    if not os.path.exists(path):
        report["ok"] = False
        report["error"] = f"db not found at {path}"
        return report

    # Route through safe_open_kdb so we participate in the same
    # WAL/busy_timeout/write-lock regime as every other DMAI writer.
    # Fall back to a well-configured bare connect for unit tests with
    # isolated tmp_path DBs.
    try:
        from components.db import safe_open_kdb  # noqa
        conn = safe_open_kdb(path, timeout=30.0)
    except Exception:  # noqa: BLE001
        conn = safe_open_kdb(path, timeout=30.0)
    try:
        # Snapshot before
        cols_before = _existing_columns(conn, "capabilities")
        report["columns_before"] = cols_before
        try:
            report["total_rows_before"] = int(
                conn.execute("SELECT COUNT(*) FROM capabilities")
                .fetchone()[0]
            )
        except sqlite3.OperationalError as e:
            report["ok"] = False
            report["error"] = f"capabilities table missing: {e}"
            return report

        if dry_run:
            # Just report what would happen without touching anything
            plan: List[str] = []
            if "provenance" not in cols_before:
                plan.append("ADD COLUMN provenance TEXT")
            if "judge_confidence" not in cols_before:
                plan.append("ADD COLUMN judge_confidence REAL")
            plan.append(
                "backfill provenance='legacy_ondemand' where "
                "runtime_mode='ondemand' AND provenance IS NULL"
            )
            plan.append(
                "backfill provenance='legacy_autonomous' where "
                "runtime_mode='autonomous' AND provenance IS NULL"
            )
            plan.append(
                "CREATE INDEX IF NOT EXISTS idx_caps_mode_prov "
                "ON capabilities(runtime_mode, provenance)"
            )
            plan.append("CREATE TABLE IF NOT EXISTS materialisation_log (...)")
            report["planned"] = plan
            return report

        # Step 1: provenance column
        added_prov = _add_column_if_missing(
            conn, "capabilities", "provenance", "TEXT",
        )
        report["steps"].append({
            "name": "add_provenance_column", "changed": added_prov,
        })

        # Step 2: judge_confidence column
        added_conf = _add_column_if_missing(
            conn, "capabilities", "judge_confidence", "REAL",
        )
        report["steps"].append({
            "name": "add_judge_confidence_column", "changed": added_conf,
        })

        # Step 3 & 4: backfill provenance for legacy rows only where NULL
        bf1 = conn.execute(
            "UPDATE capabilities SET provenance='legacy_ondemand' "
            "WHERE runtime_mode='ondemand' AND provenance IS NULL"
        ).rowcount
        report["steps"].append({
            "name": "backfill_legacy_ondemand", "rows_updated": bf1,
        })

        bf2 = conn.execute(
            "UPDATE capabilities SET provenance='legacy_autonomous' "
            "WHERE runtime_mode='autonomous' AND provenance IS NULL"
        ).rowcount
        report["steps"].append({
            "name": "backfill_legacy_autonomous", "rows_updated": bf2,
        })

        # Step 5: index for picker's WHERE runtime_mode IN (...) AND
        # provenance IN (...) query
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_caps_mode_prov "
            "ON capabilities(runtime_mode, provenance)"
        )
        report["steps"].append({
            "name": "create_picker_index", "changed": True,
        })

        # Step 6: ensure materialisation_log exists
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS materialisation_log (
                capability_id  TEXT,
                outcome        TEXT,
                created_at     TEXT,
                detail         TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_matlog_cap "
            "ON materialisation_log(capability_id)"
        )
        report["steps"].append({
            "name": "ensure_materialisation_log", "changed": True,
        })

        conn.commit()

        # Snapshot after
        report["columns_after"] = _existing_columns(conn, "capabilities")
        report["total_rows_after"] = int(
            conn.execute("SELECT COUNT(*) FROM capabilities")
            .fetchone()[0]
        )
        # Breakdown by (runtime_mode, provenance) after backfill so
        # we can eyeball the result
        breakdown = conn.execute(
            "SELECT runtime_mode, provenance, COUNT(*) FROM capabilities "
            "GROUP BY runtime_mode, provenance ORDER BY 3 DESC"
        ).fetchall()
        report["breakdown_after"] = [
            {"runtime_mode": r[0], "provenance": r[1], "count": int(r[2])}
            for r in breakdown
        ]

    except sqlite3.Error as e:
        conn.rollback()
        report["ok"] = False
        report["error"] = f"{type(e).__name__}: {e}"
    finally:
        conn.close()

    return report
