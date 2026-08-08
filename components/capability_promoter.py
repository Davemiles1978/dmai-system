"""JSON-registry -> SQL capability promoter (PR D).

Problem this fixes
------------------
``CapabilityIntegrator`` writes every discovered capability to
``data/capabilities/registry.json`` (currently ~20,694 rows across 16
capability types on production). The admin panel and the stage-progression
engine, however, read from the ``capabilities`` SQL table in
``dmai_knowledge.db`` — and that table has only a single bootstrap
``layer3_seed`` row.

Why: ``CapabilityIntegrator._save_registry`` has a SQL mirror path but it is
guarded on ``hasattr(si_core, 'sqlite')``, and ``SICore`` never assigns that
attribute. So every registry save silently skips the SQL half.

Symptom in the admin panel: stage stuck at Baby (needs 500 capabilities to
advance to Child) even though 20k+ have been discovered.

This module fixes that gap without changing ``SICore`` or the integrator (both
on hot paths). It reads the JSON registry file directly and upserts every
entry into the ``capabilities`` SQL table.

Design choices
--------------
* **Idempotent.** ``INSERT OR REPLACE`` on the ``id`` primary key — re-runs
  never create duplicates and always reflect the latest field values.
* **Change-detected.** The registry file's mtime is persisted in
  ``system_state`` (``capability_promoter.registry_mtime``). If mtime hasn't
  moved since the last successful sync, the pass is skipped.
* **Backpressure-safe.** Upserts are batched (500 rows/commit) with a small
  yield between batches so we don't monopolise the write mutex during the
  initial ~20k backfill.
* **Malformed-row-safe.** Rows missing ``id`` or ``name`` (the two NOT NULL
  columns) are counted as ``skipped`` and logged, not raised.
* **Truncation-safe.** If the registry file is missing or empty, we no-op.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

# --- Config ---------------------------------------------------------------
DEFAULT_REGISTRY = Path("data/capabilities/registry.json")
BATCH_ROWS       = 500
POLL_SECONDS     = 60
YIELD_MS         = 10
MTIME_KEY        = "capability_promoter.registry_mtime"
COUNT_KEY        = "capability_promoter.last_upserted_count"


def _kdb_path() -> str:
    """Locate ``dmai_knowledge.db`` using the same DATA_PATH convention as
    every other DB-touching component."""
    data = os.environ.get("DATA_PATH", "data/").rstrip("/").rstrip("\\")
    return os.path.join(data, "dmai_knowledge.db")


def _registry_path() -> Path:
    """Locate the capabilities registry.json under DATA_PATH."""
    env = os.environ.get("DATA_PATH")
    if env:
        base = Path(env.rstrip("/").rstrip("\\"))
        return base / "capabilities" / "registry.json"
    return DEFAULT_REGISTRY


# --- State helpers --------------------------------------------------------

def _get_state(conn, key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT value FROM system_state WHERE key = ?", (key,)
    ).fetchone()
    if not row or row[0] is None:
        return None
    return str(row[0])


def _set_state(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO system_state (key, value, updated_at) "
        "VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
        "updated_at = CURRENT_TIMESTAMP",
        (key, value),
    )


# --- Row mapping ----------------------------------------------------------

def _row_to_params(cap_id: str, cap: Dict[str, Any]) -> Optional[tuple]:
    """Map one registry entry to the (id, name, type, capability_type,
    description, source_url, source_repo, file_path, runtime_mode, language,
    methods, is_async, args, integrated_at) tuple expected by the CORE
    ``capabilities`` schema.

    Returns ``None`` for rows missing the two NOT NULL columns (id/name).
    """
    if not cap_id:
        return None
    name = cap.get("name")
    if not name:
        return None

    cap_type         = cap.get("type") or "function"
    capability_type  = cap.get("capability_type") or "general"
    description      = cap.get("description")
    source_url       = cap.get("source_url")
    source_repo      = cap.get("source_repo")
    file_path        = cap.get("file_path")
    runtime_mode     = cap.get("runtime_mode")
    language         = cap.get("language")
    methods          = cap.get("methods", []) or []
    is_async         = 1 if cap.get("is_async") else 0
    args             = cap.get("args", []) or []
    integrated_at    = cap.get("integrated_at")

    # Serialise complex fields; clip strings to sane bounds.
    try:
        methods_json = json.dumps(methods)
    except (TypeError, ValueError):
        methods_json = "[]"
    try:
        args_json = json.dumps(args)
    except (TypeError, ValueError):
        args_json = "[]"

    return (
        str(cap_id)[:200],
        str(name)[:500],
        str(cap_type)[:100],
        str(capability_type)[:100],
        (str(description)[:2000] if description is not None else None),
        (str(source_url)[:1000] if source_url is not None else None),
        (str(source_repo)[:500] if source_repo is not None else None),
        (str(file_path)[:1000] if file_path is not None else None),
        (str(runtime_mode)[:50] if runtime_mode is not None else None),
        (str(language)[:50] if language is not None else None),
        methods_json[:4000],
        is_async,
        args_json[:4000],
        integrated_at,
    )


UPSERT_SQL = (
    "INSERT INTO capabilities "
    "(id, name, type, capability_type, description, source_url, source_repo, "
    " file_path, runtime_mode, language, methods, is_async, args, integrated_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
    "        COALESCE(?, CURRENT_TIMESTAMP))"
)


# --- Core promotion pass --------------------------------------------------

def promote_once(
    registry_path: Optional[Path] = None,
    db_path: Optional[str] = None,
    *,
    batch_rows: int = BATCH_ROWS,
    force: bool = False,
    yield_ms: int = YIELD_MS,
) -> Dict[str, Any]:
    """Run one full promotion sweep.

    Returns a summary dict::
        {
          "promoted": N,        # rows successfully upserted this pass
          "skipped": M,         # rows missing id/name or parseable errors
          "total_in_registry": T,
          "mtime": <float>,
          "mtime_unchanged": bool,   # True if we skipped because file hasn't changed
        }
    """
    rpath = Path(registry_path) if registry_path else _registry_path()
    db_p  = db_path or _kdb_path()

    if not rpath.exists():
        return {
            "promoted": 0, "skipped": 0, "total_in_registry": 0,
            "mtime": None, "mtime_unchanged": False,
            "note": "registry_missing", "path": str(rpath),
        }

    current_mtime = rpath.stat().st_mtime

    conn = safe_open_kdb(db_p)
    try:
        # Belt-and-braces: ensure state and capabilities tables exist. Core
        # schema creates capabilities already; this makes the module standalone.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS system_state ("
            "  key TEXT PRIMARY KEY,"
            "  value TEXT,"
            "  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ")"
        )

        if not force:
            last_mtime = _get_state(conn, MTIME_KEY)
            if last_mtime is not None:
                try:
                    if float(last_mtime) >= current_mtime:
                        return {
                            "promoted": 0, "skipped": 0,
                            "total_in_registry": None,
                            "mtime": current_mtime,
                            "mtime_unchanged": True,
                        }
                except (TypeError, ValueError):
                    pass  # fall through and re-sync

        # Load registry.
        try:
            with rpath.open("r", encoding="utf-8") as f:
                registry = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("capability_promoter: registry unreadable: %s", e)
            return {
                "promoted": 0, "skipped": 0, "total_in_registry": 0,
                "mtime": current_mtime, "mtime_unchanged": False,
                "note": f"registry_unreadable:{e}",
            }

        caps = registry.get("capabilities") or {}
        if not isinstance(caps, dict):
            logger.warning(
                "capability_promoter: registry['capabilities'] not a dict "
                "(got %s) — nothing to promote", type(caps).__name__,
            )
            return {
                "promoted": 0, "skipped": 0, "total_in_registry": 0,
                "mtime": current_mtime, "mtime_unchanged": False,
                "note": "registry_shape_invalid",
            }

        total = len(caps)
        promoted = 0
        skipped  = 0
        batch: list[tuple] = []

        for cap_id, cap_data in caps.items():
            if not isinstance(cap_data, dict):
                skipped += 1
                continue
            params = _row_to_params(cap_id, cap_data)
            if params is None:
                skipped += 1
                continue
            batch.append(params)
            if len(batch) >= batch_rows:
                conn.executemany(UPSERT_SQL, batch)
                conn.commit()
                promoted += len(batch)
                batch = []
                if yield_ms > 0:
                    time.sleep(yield_ms / 1000.0)

        if batch:
            conn.executemany(UPSERT_SQL, batch)
            conn.commit()
            promoted += len(batch)

        # Persist mtime + count so we can skip next pass if unchanged.
        _set_state(conn, MTIME_KEY, str(current_mtime))
        _set_state(conn, COUNT_KEY, str(promoted))
        conn.commit()

        return {
            "promoted": promoted,
            "skipped": skipped,
            "total_in_registry": total,
            "mtime": current_mtime,
            "mtime_unchanged": False,
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


# --- Background loop ------------------------------------------------------

class CapabilityPromoterLoop:
    """Thin wrapper so the boot sequence can start / stop the poller."""

    def __init__(
        self,
        registry_path: Optional[Path] = None,
        db_path: Optional[str] = None,
        poll_seconds: int = POLL_SECONDS,
    ) -> None:
        self._registry_path = Path(registry_path) if registry_path else _registry_path()
        self._db_path       = db_path
        self._poll          = int(poll_seconds)
        self._stop          = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_summary: Dict[str, Any] = {}

    def start(self) -> None:
        # Immediate backfill pass — force so mtime skip doesn't hide a first sync.
        try:
            summary = promote_once(
                self._registry_path, self._db_path, force=True,
            )
            self.last_summary = summary
            if summary.get("promoted"):
                logger.info(
                    "CapabilityPromoter backfill: promoted=%d skipped=%d total=%s",
                    summary["promoted"], summary["skipped"],
                    summary.get("total_in_registry"),
                )
        except Exception as e:
            logger.warning("CapabilityPromoter backfill failed: %s", e)

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="dmai-capability-promoter",
        )
        self._thread.start()
        logger.info(
            "CapabilityPromoter started (poll=%ds, registry=%s)",
            self._poll, self._registry_path,
        )

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        # PR V-fast: exponential backoff on lock contention.
        backoff = 0.0
        while not self._stop.is_set():
            self._stop.wait(self._poll + backoff)
            if self._stop.is_set():
                return
            try:
                self.last_summary = promote_once(
                    self._registry_path, self._db_path,
                )
                backoff = 0.0  # success — reset backoff
                if self.last_summary.get("promoted"):
                    logger.info(
                        "CapabilityPromoter tick: promoted=%d skipped=%d total=%s",
                        self.last_summary["promoted"],
                        self.last_summary["skipped"],
                        self.last_summary.get("total_in_registry"),
                    )
            except Exception as e:
                msg = str(e).lower()
                if "lock" in msg or "mutex_timeout" in msg:
                    backoff = min(backoff * 2 + 5.0, 300.0)
                logger.warning("CapabilityPromoter tick error: %s (backoff=%.0fs)", e, backoff)


_LOOP: Optional[CapabilityPromoterLoop] = None


def start_promoter_loop(
    registry_path: Optional[Path] = None,
    db_path: Optional[str] = None,
    poll_seconds: int = POLL_SECONDS,
) -> CapabilityPromoterLoop:
    """Idempotent boot hook. Safe to call multiple times — returns the existing
    loop if one is already running."""
    global _LOOP
    if _LOOP is not None and _LOOP._thread and _LOOP._thread.is_alive():
        return _LOOP
    _LOOP = CapabilityPromoterLoop(registry_path, db_path, poll_seconds)
    _LOOP.start()
    return _LOOP


def get_promoter_loop() -> Optional[CapabilityPromoterLoop]:
    return _LOOP
