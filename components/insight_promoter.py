"""JSONL -> SQL insight promoter (PR B).

Problem this fixes
------------------
``si_core.add_insight`` appends every discovered insight to
``data/research/insights.jsonl`` (currently ~18k rows across 16 domains,
664 discoveries in the last day). The admin panel, however, reads from the
``insights`` SQL table in ``dmai_knowledge.db`` — and that table has just a
single bootstrap row because no writer ever promoted the JSONL entries
into SQL.

Symptoms in the admin panel:
    - "Study Progress" stuck at 0
    - Stage "Baby" even after weeks of learning
    - Discoveries Today = 0 even when the JSONL grows by hundreds/day
    - Insight count = 1

This module fixes that gap without changing ``si_core`` (which is on the
hot path). Instead, we tail the JSONL file: a background daemon promotes
new rows into SQL as they are appended, and a one-shot backfill on boot
catches up on everything already written.

Design choices
--------------
* **Idempotent.** We persist the byte offset we last promoted in
  ``system_state`` (``insight_promoter.jsonl_offset``). Restarts resume
  from there — no re-promotion of the same rows.
* **Crash-safe.** Only advance the offset AFTER the SQL insert commits.
* **Truncation-safe.** If the file shrinks (rotate / rewrite), we reset
  the offset to 0 and re-promote from the start.
* **Backpressure-safe.** Each batch is capped (500 rows) so the initial
  ~18k backfill runs in ~40 small commits rather than one giant write.
* **No new deps.** Uses stdlib + the existing ``safe_open_kdb`` helper.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
DEFAULT_JSONL = Path("data/research/insights.jsonl")
BATCH_ROWS    = 500                # rows per commit — bounds write burst
POLL_SECONDS  = 30                 # tail poll interval
OFFSET_KEY    = "insight_promoter.jsonl_offset"
STATE_KEY_LAST_ID = "insight_promoter.last_insight_id"


def _kdb_path() -> str:
    """Locate ``dmai_knowledge.db`` using the same DATA_PATH convention as
    ``dmai_core_complete``. Defers env lookup so tests can override."""
    data = os.environ.get("DATA_PATH", "data/").rstrip("/").rstrip("\\")
    return os.path.join(data, "dmai_knowledge.db")


# ── State helpers ─────────────────────────────────────────────────────────

def _get_offset(conn) -> int:
    row = conn.execute(
        "SELECT value FROM system_state WHERE key = ?", (OFFSET_KEY,)
    ).fetchone()
    if not row or row[0] is None:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0


def _set_offset(conn, offset: int) -> None:
    conn.execute(
        "INSERT INTO system_state (key, value, updated_at) "
        "VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
        "updated_at = CURRENT_TIMESTAMP",
        (OFFSET_KEY, str(offset)),
    )


# ── Row mapping ───────────────────────────────────────────────────────────

def _row_to_insight_params(obj: Dict[str, Any]) -> Optional[tuple]:
    """Map a JSONL row to the (concept, insight_text, confidence, domain,
    source, created_at) tuple expected by the ``insights`` table.

    Returns ``None`` for malformed rows (missing both concept and
    insight_text). We accept the union of shapes ``si_core.add_insight``
    can emit plus the legacy variants seen in the file.
    """
    concept   = obj.get("concept") or obj.get("insight_text")
    if not concept:
        return None
    insight_text = obj.get("insight_text") or obj.get("concept") or ""
    confidence   = obj.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    domain     = obj.get("domain") or "general"
    source     = obj.get("source") or ""
    created_at = obj.get("timestamp") or obj.get("date") or None
    # Clip to sensible column sizes to guard against runaway text
    concept      = str(concept)[:2000]
    insight_text = str(insight_text)[:5000]
    domain       = str(domain)[:200]
    source       = str(source)[:500]
    import hashlib
    row_id = hashlib.md5((concept + insight_text).encode()).hexdigest()[:16]
    return (row_id, concept, insight_text, confidence, domain, source, created_at)


INSERT_SQL = (
    "INSERT INTO insights (id, concept, insight_text, confidence, domain, source,"
    "                      created_at) "
    "VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))"
)


# ── Core promotion pass ───────────────────────────────────────────────────

def promote_once(
    jsonl_path: Optional[Path] = None,
    db_path: Optional[str] = None,
    *,
    batch_rows: int = BATCH_ROWS,
) -> Dict[str, int]:
    """Do a single promotion sweep: read new bytes since last offset,
    insert every parseable row, advance the offset.

    Returns a summary dict::
        {"promoted": N, "skipped": M, "new_offset": O, "reset_from": R|None}

    ``reset_from`` is present if truncation was detected — the value is
    the previous offset (before we reset to 0).
    """
    jsonl_path = Path(jsonl_path) if jsonl_path else DEFAULT_JSONL
    if not jsonl_path.exists():
        return {"promoted": 0, "skipped": 0, "new_offset": 0, "reset_from": None}

    db_p = db_path or _kdb_path()
    reset_from: Optional[int] = None
    conn = safe_open_kdb(db_p)
    try:
        # Ensure the state table exists (bootstrap-safe).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS system_state ("
            "  key TEXT PRIMARY KEY,"
            "  value TEXT,"
            "  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        # PR AA: force-commit any pending txn from the CREATE TABLE above so
        # the KeepOpenProxy releases its write lock before we do heavier work
        # or exit. Without this, _txn_held can linger and block the projector
        # rebuild on the next tick.
        try:
            conn.commit()
        except Exception:
            pass
        offset = _get_offset(conn)
        size   = jsonl_path.stat().st_size

        if offset > size:
            # File was truncated / rotated. Re-promote from start.
            logger.warning(
                "insight_promoter: JSONL truncation detected "
                "(offset=%d > size=%d). Restarting from 0.",
                offset, size,
            )
            reset_from = offset
            offset = 0

        promoted = 0
        skipped  = 0
        cur_offset = offset

        with jsonl_path.open("r", encoding="utf-8") as f:
            f.seek(offset)
            batch: list[tuple] = []
            batch_end_offset = offset
            for line in f:
                batch_end_offset += len(line.encode("utf-8"))
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                params = _row_to_insight_params(obj)
                if params is None:
                    skipped += 1
                    continue
                batch.append(params)
                if len(batch) >= batch_rows:
                    conn.executemany(INSERT_SQL, batch)
                    _set_offset(conn, batch_end_offset)
                    conn.commit()
                    promoted += len(batch)
                    cur_offset = batch_end_offset
                    batch = []
            if batch:
                conn.executemany(INSERT_SQL, batch)
                _set_offset(conn, batch_end_offset)
                conn.commit()
                promoted += len(batch)
                cur_offset = batch_end_offset

        # PR AA: final commit + explicit lock release before returning.
        # Belt-and-braces — ensures the RLock is dropped so background
        # readers (graph projector, admin queries) don't have to wait
        # through the next poll cycle for the promoter to notice.
        try:
            conn.commit()
        except Exception:
            pass

        return {
            "promoted": promoted,
            "skipped": skipped,
            "new_offset": cur_offset,
            "reset_from": reset_from,
        }
    finally:
        # KeepOpenProxy.close() is a no-op — connection is pool-owned.
        # A final commit here defends against a code path that returned
        # without committing (should be impossible after the checks above,
        # but cheap insurance against the wedged-lock behaviour we've
        # observed on prod).
        try:
            conn.commit()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


# ── Background loop ───────────────────────────────────────────────────────

class InsightPromoterLoop:
    """Thin wrapper so the boot sequence can start / stop the poller."""

    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        db_path: Optional[str] = None,
        poll_seconds: int = POLL_SECONDS,
    ) -> None:
        self._jsonl_path = Path(jsonl_path) if jsonl_path else DEFAULT_JSONL
        self._db_path    = db_path
        self._poll       = int(poll_seconds)
        self._stop       = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_summary: Dict[str, Any] = {}

    def start(self) -> None:
        # Immediate backfill pass — catches everything already in the file.
        try:
            summary = promote_once(self._jsonl_path, self._db_path)
            self.last_summary = summary
            if summary["promoted"]:
                logger.info(
                    "InsightPromoter backfill: promoted=%d skipped=%d offset=%d",
                    summary["promoted"], summary["skipped"], summary["new_offset"],
                )
        except Exception as e:
            logger.warning("InsightPromoter backfill failed: %s", e)

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="dmai-insight-promoter"
        )
        self._thread.start()
        logger.info(
            "InsightPromoter started (poll=%ds, jsonl=%s)",
            self._poll, self._jsonl_path,
        )

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        # PR V-fast: exponential backoff on lock contention so we stop
        # hammering a busy DB every poll interval. Reset on success.
        backoff = 0.0
        while not self._stop.is_set():
            self._stop.wait(self._poll + backoff)
            if self._stop.is_set():
                return
            try:
                self.last_summary = promote_once(self._jsonl_path, self._db_path)
                backoff = 0.0  # success — reset backoff
                if self.last_summary["promoted"]:
                    logger.info(
                        "InsightPromoter tick: promoted=%d skipped=%d offset=%d",
                        self.last_summary["promoted"],
                        self.last_summary["skipped"],
                        self.last_summary["new_offset"],
                    )
            except Exception as e:
                msg = str(e).lower()
                if "lock" in msg or "mutex_timeout" in msg:
                    # Cap at ~5 min so we still recover once the storm clears.
                    backoff = min(backoff * 2 + 5.0, 300.0)
                logger.warning("InsightPromoter tick error: %s (backoff=%.0fs)", e, backoff)


# Module-level singleton (started by dmai_core_complete boot sequence).
_LOOP: Optional[InsightPromoterLoop] = None


def start_promoter_loop(
    jsonl_path: Optional[Path] = None,
    db_path: Optional[str] = None,
    poll_seconds: int = POLL_SECONDS,
) -> InsightPromoterLoop:
    """Idempotent boot hook. Safe to call multiple times — returns the
    existing loop if one is already running."""
    global _LOOP
    if _LOOP is not None and _LOOP._thread and _LOOP._thread.is_alive():
        return _LOOP
    _LOOP = InsightPromoterLoop(jsonl_path, db_path, poll_seconds)
    _LOOP.start()
    return _LOOP


def get_promoter_loop() -> Optional[InsightPromoterLoop]:
    return _LOOP
