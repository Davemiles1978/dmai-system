"""Seed the capabilities table with rows from a JSONL backlog file.

This is the ingestion side of the "unified DMAI backlog" pattern
introduced in docs/planning/DMAI_COLLATED_REQUIREMENTS_AND_ROADMAP.md
(2026-07-16 snapshot). It reads ``data/self_gen_backlog.jsonl`` (or a
caller-supplied path) and inserts each row into the capabilities table
as a stub with the right provenance/confidence/runtime_mode so the
materialiser's ``_pick_candidates`` picks them up on the next tick.

Idempotent: uses ``INSERT OR IGNORE`` keyed by ``id`` (the backlog row's
``id`` field, e.g. ``gap_seed_backlog_ingestion_endpoint``). Re-running
the ingest never duplicates rows.

Never overwrites an existing row (does not modify runtime_mode /
judge_confidence / provenance of prior inserts). To re-seed a specific
item, delete it from the capabilities table first.

Companion to /api/admin/self-generation/seed-backlog endpoint in
dmai_core_complete.py.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Same DB path the materialiser uses so we write into the same registry.
_DEFAULT_JSONL = "data/self_gen_backlog.jsonl"
_DEFAULT_DB = os.environ.get(
    "DMAI_CAPABILITIES_DB",
    "data/dmai_knowledge.db",
)

# Fields the picker filters on. Every seeded row uses these fixed values.
_SEED_PROVENANCE = "gap_driven"
_SEED_RUNTIME_MODE = "stub"

# The materialiser's ACCEPTED_PROVENANCES tuple must include 'gap_driven'
# (per PR DD which widened the picker to 3 provenance pools + gap seeder).
# We do not import that constant to keep this module cheap to load.


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"backlog file not found: {path}")
    rows = []
    with p.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"invalid JSON at {path}:{line_no}: {e}"
                ) from e
    return rows


def _validate_row(r: Dict[str, Any]) -> Optional[str]:
    """Return an error string if the row is invalid, else None."""
    required = ("id", "name", "capability_type", "description",
                "priority", "judge_confidence")
    for field in required:
        if field not in r:
            return f"missing required field: {field}"
    if not isinstance(r["id"], str) or not r["id"]:
        return "id must be a non-empty string"
    if not isinstance(r["judge_confidence"], (int, float)):
        return "judge_confidence must be numeric"
    if r["priority"] not in (1, 2, 3):
        return "priority must be 1, 2, or 3"
    return None


def seed_backlog(
    jsonl_path: str = _DEFAULT_JSONL,
    db_path: str = _DEFAULT_DB,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Read the JSONL backlog and INSERT OR IGNORE each row into the
    capabilities table as a gap-driven stub.

    Returns a summary dict:
      {
        "ok": bool,
        "read": N,          # rows read from JSONL
        "valid": N,         # rows that passed schema validation
        "inserted": N,      # rows actually inserted (0 if dry_run)
        "already_present": N,
        "invalid": [{"id": ..., "error": ...}, ...],
        "dry_run": bool,
        "db_path": str,
        "jsonl_path": str,
      }
    """
    rows = _read_jsonl(jsonl_path)
    valid_rows: List[Dict[str, Any]] = []
    invalid: List[Dict[str, str]] = []
    for r in rows:
        err = _validate_row(r)
        if err:
            invalid.append({"id": r.get("id", "?"), "error": err})
        else:
            valid_rows.append(r)

    summary: Dict[str, Any] = {
        "ok": True,
        "read": len(rows),
        "valid": len(valid_rows),
        "inserted": 0,
        "already_present": 0,
        "invalid": invalid,
        "dry_run": dry_run,
        "db_path": db_path,
        "jsonl_path": jsonl_path,
    }

    if dry_run:
        # Simulate: check which ids are already present without writing.
        conn = sqlite3.connect(db_path)
        try:
            existing = {row[0] for row in conn.execute(
                "SELECT id FROM capabilities"
            ).fetchall()}
        finally:
            conn.close()
        would_insert = [r for r in valid_rows if r["id"] not in existing]
        summary["would_insert"] = len(would_insert)
        summary["already_present"] = len(valid_rows) - len(would_insert)
        summary["preview"] = [
            {"id": r["id"], "name": r["name"], "priority": r["priority"]}
            for r in would_insert[:10]
        ]
        return summary

    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        # Discover the actual capabilities table columns so we only insert
        # into columns that exist. Post-PR HH the shape is:
        # id, name, type, capability_type, description, provenance,
        # judge_confidence, runtime_mode, ...
        cols = {row[1] for row in conn.execute(
            "PRAGMA table_info(capabilities)"
        ).fetchall()}

        inserted = 0
        already = 0
        for r in valid_rows:
            # Build INSERT respecting only present columns.
            payload: Dict[str, Any] = {
                "id": r["id"],
                "name": r["name"],
                "capability_type": r["capability_type"],
                "description": r["description"],
                "provenance": _SEED_PROVENANCE,
                "judge_confidence": float(r["judge_confidence"]),
                "runtime_mode": _SEED_RUNTIME_MODE,
            }
            # 'type' column exists on prod as a legacy alias for
            # capability_type — mirror it if the column is present.
            if "type" in cols:
                payload["type"] = r["capability_type"]

            fields = [k for k in payload if k in cols]
            placeholders = ",".join("?" for _ in fields)
            col_list = ",".join(fields)
            values = tuple(payload[k] for k in fields)

            cur = conn.execute(
                f"INSERT OR IGNORE INTO capabilities ({col_list}) "
                f"VALUES ({placeholders})",
                values,
            )
            if cur.rowcount:
                inserted += 1
            else:
                already += 1
        conn.commit()
        summary["inserted"] = inserted
        summary["already_present"] = already
    except sqlite3.OperationalError as e:
        summary["ok"] = False
        summary["error"] = f"sqlite: {e}"
        log.exception("seed_backlog failed")
    finally:
        conn.close()

    return summary
