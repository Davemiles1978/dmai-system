"""Self-generation loop diagnostic — one-shot introspection of why the
materialiser + upstream feeders are producing zero picks even though the
capabilities table holds ~20k rows.

Exposes a single function :func:`diagnose_self_generation` that returns a
structured dict describing, for each stage of the loop:

- the raw count of candidates the stage *could* pick from
- how many were filtered out and *why* (grouped by reason)
- a small sample of concrete filtered rows so we can see real data,
  not just numbers

This is a read-only introspection module: it opens a fresh connection,
runs SELECT-only statements, and never touches production state.

Wired into ``/api/admin/self-generation/diagnose`` (see dmai_core_complete).
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Match the materialiser's own defaults so the diagnostic sees exactly
# what the loop would see. If the loop's constants ever move, we import
# them at call time rather than pinning here.
_DEFAULT_QUOTAS = {
    "fresh_blood_seed+self_judge": 5,
    "promoter_path+self_judge":    3,
    "gap_driven":                  2,
}
_DEFAULT_MIN_CONF = 0.60


def _resolve_db_path() -> str:
    """Find the same SQLite path the materialiser uses."""
    return os.environ.get(
        "DMAI_KNOWLEDGE_DB",
        "data/dmai_knowledge.db",
    )


def _safe_execute(conn: sqlite3.Connection,
                  sql: str,
                  args: Tuple = (),
                  ) -> Tuple[Optional[List[Tuple]], Optional[str]]:
    """Execute a read query, returning (rows, error). Never raises."""
    try:
        return list(conn.execute(sql, args).fetchall()), None
    except sqlite3.OperationalError as e:
        return None, f"sqlite_operational_error: {e}"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _diagnose_capabilities_table(conn: sqlite3.Connection,
                                 min_conf: float,
                                 quotas: Dict[str, int],
                                 ) -> Dict[str, Any]:
    """Walk the capabilities table with the same lens the picker uses,
    but *report every filter step* instead of returning only the winners.
    """
    out: Dict[str, Any] = {}

    # 1) Total rows
    rows, err = _safe_execute(conn, "SELECT COUNT(*) FROM capabilities")
    if err:
        return {"error": err, "reason": "capabilities table unreadable"}
    out["total_rows"] = int(rows[0][0]) if rows else 0

    # 2) Breakdown by runtime_mode
    rows, err = _safe_execute(
        conn,
        "SELECT runtime_mode, COUNT(*) FROM capabilities "
        "GROUP BY runtime_mode ORDER BY 2 DESC",
    )
    out["by_runtime_mode"] = {
        (r[0] or "<null>"): int(r[1]) for r in (rows or [])
    }
    if err:
        out["by_runtime_mode_error"] = err

    # 3) Breakdown by provenance
    rows, err = _safe_execute(
        conn,
        "SELECT provenance, COUNT(*) FROM capabilities "
        "GROUP BY provenance ORDER BY 2 DESC",
    )
    out["by_provenance"] = {
        (r[0] or "<null>"): int(r[1]) for r in (rows or [])
    }
    if err:
        out["by_provenance_error"] = err

    # 4) For each accepted provenance, count how many pass each filter
    #    stage individually. This tells us exactly which filter is the
    #    block.
    per_pool: Dict[str, Any] = {}
    for prov, quota in quotas.items():
        pool_out: Dict[str, Any] = {"quota": quota}

        # a) rows with this provenance
        rows, err = _safe_execute(
            conn,
            "SELECT COUNT(*) FROM capabilities WHERE provenance = ?",
            (prov,),
        )
        pool_out["total_with_provenance"] = int(rows[0][0]) if rows else 0

        # b) ... AND runtime_mode IN stub/stub_reverted
        rows, err = _safe_execute(
            conn,
            "SELECT COUNT(*) FROM capabilities "
            "WHERE provenance = ? "
            "  AND runtime_mode IN ('stub', 'stub_reverted')",
            (prov,),
        )
        pool_out["stub_or_reverted"] = int(rows[0][0]) if rows else 0

        # c) ... AND judge_confidence >= floor
        rows, err = _safe_execute(
            conn,
            "SELECT COUNT(*) FROM capabilities "
            "WHERE provenance = ? "
            "  AND runtime_mode IN ('stub', 'stub_reverted') "
            "  AND judge_confidence IS NOT NULL "
            "  AND judge_confidence >= ?",
            (prov, float(min_conf)),
        )
        pool_out["above_confidence_floor"] = int(rows[0][0]) if rows else 0

        # d) Confidence distribution for THIS provenance so we can see
        #    if the floor is what's cutting things
        rows, err = _safe_execute(
            conn,
            "SELECT ROUND(COALESCE(judge_confidence, 0.0), 1) AS bucket, "
            "       COUNT(*) FROM capabilities WHERE provenance = ? "
            "GROUP BY bucket ORDER BY bucket",
            (prov,),
        )
        pool_out["confidence_histogram"] = {
            str(float(r[0])): int(r[1]) for r in (rows or [])
        }

        # e) Sample of eligible rows (before log-based ineligibility)
        rows, err = _safe_execute(
            conn,
            "SELECT id, name, judge_confidence, runtime_mode "
            "FROM capabilities "
            "WHERE provenance = ? "
            "  AND runtime_mode IN ('stub', 'stub_reverted') "
            "  AND judge_confidence >= ? "
            "ORDER BY judge_confidence DESC LIMIT 5",
            (prov, float(min_conf)),
        )
        pool_out["sample_eligible_pre_log"] = [
            {"id": r[0], "name": r[1],
             "judge_confidence": r[2], "runtime_mode": r[3]}
            for r in (rows or [])
        ]

        per_pool[prov] = pool_out

    out["per_pool"] = per_pool

    # 5) How many rows are ineligible due to materialisation_log
    log_rows, err = _safe_execute(
        conn,
        "SELECT outcome, COUNT(*) FROM materialisation_log "
        "GROUP BY outcome ORDER BY 2 DESC",
    )
    out["materialisation_log_outcomes"] = {
        (r[0] or "<null>"): int(r[1]) for r in (log_rows or [])
    }
    if err:
        out["materialisation_log_error"] = err

    # 6) 24h-backoff row count (failed/rejected in last 24h)
    cutoff = (_dt.datetime.now(_dt.timezone.utc)
              - _dt.timedelta(hours=24)).isoformat()
    rows, err = _safe_execute(
        conn,
        "SELECT COUNT(DISTINCT capability_id) FROM materialisation_log "
        "WHERE outcome IN ('failed','rejected_review') "
        "  AND created_at > ?",
        (cutoff,),
    )
    out["in_24h_backoff"] = int(rows[0][0]) if rows else 0

    return out


def _diagnose_gap_seeder(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Run iter_capability_gaps() and report what it would have seeded."""
    out: Dict[str, Any] = {}
    try:
        from components.gap_fetcher import iter_capability_gaps  # noqa
    except Exception as e:  # noqa: BLE001
        return {"error": f"gap_fetcher import failed: {e}"}

    try:
        gaps = list(iter_capability_gaps(fresh=False))
    except Exception as e:  # noqa: BLE001
        return {"error": f"iter_capability_gaps failed: {e}"}

    out["gaps_available"] = len(gaps)
    out["gaps_sample"] = [
        {"name": getattr(g, "name", None),
         "priority": getattr(g, "priority", None),
         "target_kpi": getattr(g, "target_kpi", None)}
        for g in gaps[:5]
    ]

    # For each gap, is there already a capability row for it?
    already_seeded = 0
    would_insert = 0
    for g in gaps[:20]:
        name = getattr(g, "name", "") or ""
        cap_id = f"gap_{name}"
        rows, err = _safe_execute(
            conn,
            "SELECT id, provenance, runtime_mode, judge_confidence "
            "FROM capabilities WHERE id = ? OR name = ? LIMIT 1",
            (cap_id, name),
        )
        if rows:
            already_seeded += 1
        else:
            would_insert += 1

    out["already_present_in_capabilities"] = already_seeded
    out["would_insert_on_next_tick"] = would_insert
    return out


def _diagnose_fresh_blood(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Ask fresh-blood *why* it emitted 0 last tick."""
    out: Dict[str, Any] = {}
    try:
        from components.fresh_blood_injector import (  # noqa
            get_fresh_blood_status,
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"fresh_blood_injector import failed: {e}"}

    try:
        status = get_fresh_blood_status()
    except Exception as e:  # noqa: BLE001
        return {"error": f"get_fresh_blood_status failed: {e}"}

    last = status.get("last_summary", {}) or {}
    out["last_emitted"] = last.get("emitted")
    out["last_skipped"] = last.get("skipped")
    out["last_channels_used"] = last.get("channels_used") or []
    out["last_picked"] = last.get("picked") or []
    out["last_ts"] = last.get("ts")

    # Count fresh-blood insights in SQL vs how many became capabilities
    rows, err = _safe_execute(
        conn,
        "SELECT COUNT(*) FROM insights WHERE provenance LIKE 'fresh_blood%'",
    )
    out["sql_fresh_blood_insights"] = int(rows[0][0]) if rows else 0
    if err:
        out["sql_fresh_blood_insights_error"] = err

    rows, err = _safe_execute(
        conn,
        "SELECT COUNT(*) FROM capabilities "
        "WHERE provenance LIKE 'fresh_blood%'",
    )
    out["capabilities_from_fresh_blood"] = int(rows[0][0]) if rows else 0
    if err:
        out["capabilities_from_fresh_blood_error"] = err

    return out


def _diagnose_capability_promoter(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Why is the capability promoter reporting mtime_unchanged for months?"""
    out: Dict[str, Any] = {}
    try:
        from components.capability_promoter import (  # noqa
            get_capability_promoter_status,
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"capability_promoter import failed: {e}"}

    try:
        status = get_capability_promoter_status()
    except Exception as e:  # noqa: BLE001
        return {"error": f"get_capability_promoter_status failed: {e}"}

    last = status.get("last_summary", {}) or {}
    out["last_promoted"] = last.get("promoted")
    out["last_skipped"] = last.get("skipped")
    out["last_mtime_unchanged"] = last.get("mtime_unchanged")
    out["last_mtime"] = last.get("mtime")
    out["registry_path"] = status.get("registry_path")

    # If we can stat the registry file, report its actual mtime
    reg = status.get("registry_path")
    if reg and os.path.exists(reg):
        try:
            st = os.stat(reg)
            out["registry_mtime_now"] = st.st_mtime
            out["registry_size_bytes"] = st.st_size
        except OSError as e:
            out["registry_stat_error"] = str(e)
    else:
        out["registry_present"] = bool(reg) and os.path.exists(reg or "")

    return out


def diagnose_self_generation(min_confidence: Optional[float] = None,
                             quotas: Optional[Dict[str, int]] = None,
                             ) -> Dict[str, Any]:
    """Top-level diagnostic. Returns a structured dict; never raises."""
    min_conf = float(min_confidence
                     if min_confidence is not None else _DEFAULT_MIN_CONF)
    q = dict(quotas or _DEFAULT_QUOTAS)

    db_path = _resolve_db_path()
    result: Dict[str, Any] = {
        "ok": True,
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "db_path": db_path,
        "min_confidence": min_conf,
        "quotas": q,
    }

    if not os.path.exists(db_path):
        result["ok"] = False
        result["error"] = f"db not found at {db_path}"
        return result

    try:
        # Prefer safe_open_kdb so we don't force a rollback-journal /
        # WAL flip on a hot DB. Fall back to a well-configured bare
        # connect for tests using isolated tmp_path DBs.
        try:
            from components.db import safe_open_kdb  # noqa
            conn = safe_open_kdb(db_path, timeout=30.0, read_only=True)
        except Exception:  # noqa: BLE001
            conn = sqlite3.connect(db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
    except sqlite3.OperationalError as e:
        result["ok"] = False
        result["error"] = f"connect failed: {e}"
        return result

    try:
        result["capabilities_table"] = _diagnose_capabilities_table(
            conn, min_conf, q,
        )
        result["gap_seeder"] = _diagnose_gap_seeder(conn)
        result["fresh_blood"] = _diagnose_fresh_blood(conn)
        result["capability_promoter"] = _diagnose_capability_promoter(conn)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    # Roll-up verdict: which stage is the block?
    caps = result.get("capabilities_table") or {}
    pools = caps.get("per_pool") or {}
    verdict_reasons: List[str] = []
    for prov, pool in pools.items():
        if pool.get("total_with_provenance", 0) == 0:
            verdict_reasons.append(
                f"{prov}: no rows exist with this provenance"
            )
        elif pool.get("stub_or_reverted", 0) == 0:
            verdict_reasons.append(
                f"{prov}: rows exist but none in stub/stub_reverted state"
            )
        elif pool.get("above_confidence_floor", 0) == 0:
            verdict_reasons.append(
                f"{prov}: rows exist and are stubs, but none clear "
                f"the {min_conf} confidence floor"
            )
        else:
            eligible = pool.get("above_confidence_floor", 0)
            verdict_reasons.append(
                f"{prov}: {eligible} eligible rows found — "
                f"picker should be able to consume these"
            )
    result["verdict"] = {
        "min_confidence": min_conf,
        "per_pool_reasons": verdict_reasons,
    }
    return result
