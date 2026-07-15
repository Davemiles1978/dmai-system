"""Self-generation status aggregator (PR EE).

Single-call snapshot of the DMAI self-generation loop, powering the
``/api/self-generation/status`` endpoint and the dashboard widget.

Aggregates:

* **materialiser**  — day counter, cap, quotas, last-tick summary,
                       provenance breakdown, gaps_seeded
* **verifier**      — totals, 7-day success rate, recent runs,
                       quarantined + stub_reverted counts
* **queue**         — how many candidates are eligible per pool right
                       now (before quotas apply)
* **live_modules**  — count of ``runtime_mode='generated_module'``
                       capabilities and the last N promotions
* **gaps**          — how many capability gaps the scanner sees, top 5
* **health**        — one-word verdict + short human-readable reasons

The module is import-safe: no I/O at import time, all queries wrapped
in try/except so a bad row or a missing column degrades gracefully to
a ``partial`` health status instead of a 500 on the endpoint.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = REPO_ROOT / "components" / "generated" / "live"
QUARANTINE_DIR = REPO_ROOT / "components" / "generated" / "quarantine"


# ── SQLite helper ─────────────────────────────────────────────────────────

def _connect(db_path: str) -> Optional[sqlite3.Connection]:
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:  # noqa: BLE001
        logger.warning("self_generation_status: connect failed: %s", e)
        return None


def _safe_scalar(conn: sqlite3.Connection, sql: str,
                 params: tuple = ()) -> Optional[Any]:
    try:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None
    except Exception as e:  # noqa: BLE001
        logger.info("safe_scalar failed on %r: %s", sql[:60], e)
        return None


def _safe_rows(conn: sqlite3.Connection, sql: str,
               params: tuple = ()) -> List[sqlite3.Row]:
    try:
        return list(conn.execute(sql, params).fetchall())
    except sqlite3.OperationalError:
        return []
    except Exception as e:  # noqa: BLE001
        logger.info("safe_rows failed on %r: %s", sql[:60], e)
        return []


# ── Section builders ──────────────────────────────────────────────────────

def _materialiser_section(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Read the last materialiser tick summary from system_state."""
    try:
        from components.capability_materialiser import (
            STATE_KEY_LAST_RUN,
            STATE_KEY_LAST_SUMMARY,
            DEFAULT_DAILY_CAP,
            DEFAULT_MIN_JUDGE_CONFIDENCE,
            PICKER_QUOTAS,
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"materialiser import failed: {e}"}

    last_run = _safe_scalar(
        conn,
        "SELECT value FROM system_state WHERE key = ?",
        (STATE_KEY_LAST_RUN,),
    )
    summary_raw = _safe_scalar(
        conn,
        "SELECT value FROM system_state WHERE key = ?",
        (STATE_KEY_LAST_SUMMARY,),
    )
    last_summary: Dict[str, Any] = {}
    if summary_raw:
        try:
            last_summary = json.loads(summary_raw)
        except Exception:  # noqa: BLE001
            last_summary = {"error": "summary_parse_failed"}

    # Whether the loop thread is alive at all.
    # CapabilityMaterialiserLoop wraps a thread in ._thread, it isn't
    # itself a Thread subclass. Mirror the shape used by
    # /api/admin/capability-materialiser-status so both endpoints agree.
    running = False
    try:
        from components.capability_materialiser import _LOOP as _MAT_LOOP
        thr = getattr(_MAT_LOOP, "_thread", None) if _MAT_LOOP else None
        running = bool(thr and thr.is_alive())
    except Exception:  # noqa: BLE001
        running = False

    return {
        "running": running,
        "last_run_ts": last_run,
        "config": {
            "daily_cap": DEFAULT_DAILY_CAP,
            "min_confidence": DEFAULT_MIN_JUDGE_CONFIDENCE,
            "quotas": dict(PICKER_QUOTAS),
        },
        "last_tick": {
            "ts": last_summary.get("ts"),
            "picked": last_summary.get("picked", 0),
            "promoted": last_summary.get("promoted", 0),
            "failed": last_summary.get("failed", 0),
            "day_count": last_summary.get("day_count", 0),
            "cap_hit": last_summary.get("cap_hit", False),
            "gaps_seeded": last_summary.get("gaps_seeded", 0),
            "provenance_breakdown":
                last_summary.get("provenance_breakdown", {}),
        },
    }


def _verifier_section(conn: sqlite3.Connection,
                      window_days: int = 7) -> Dict[str, Any]:
    """Verifier totals, recent success rate, quarantined counts."""
    # Ensure the verifier table exists so first-tick before verifier runs
    # doesn't blow up. We use a bare try so we don't ping the schema.
    totals = {
        "successes": 0,
        "failures": 0,
        "total": 0,
    }
    row = _safe_rows(conn,
        "SELECT "
        "  SUM(CASE WHEN ok=1 THEN 1 ELSE 0 END), "
        "  SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END), "
        "  COUNT(*) "
        "FROM verification_log")
    if row and row[0][2] is not None:
        totals["successes"] = int(row[0][0] or 0)
        totals["failures"] = int(row[0][1] or 0)
        totals["total"] = int(row[0][2] or 0)

    # 7-day success rate
    since_iso = (
        _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)
        - _dt.timedelta(days=window_days)
    ).strftime("%Y-%m-%d %H:%M:%S")
    window_rows = _safe_rows(conn,
        "SELECT ok, COUNT(*) FROM verification_log "
        "WHERE created_at >= ? GROUP BY ok",
        (since_iso,))
    window_ok = 0
    window_fail = 0
    for r in window_rows:
        if r[0] == 1:
            window_ok = int(r[1])
        else:
            window_fail = int(r[1])
    window_total = window_ok + window_fail
    window_rate = (window_ok / window_total) if window_total else None

    # runtime_mode counts
    quarantined = _safe_scalar(conn,
        "SELECT COUNT(*) FROM capabilities WHERE runtime_mode = 'quarantined'")
    reverted = _safe_scalar(conn,
        "SELECT COUNT(*) FROM capabilities WHERE runtime_mode = 'stub_reverted'")

    recent = _safe_rows(conn,
        "SELECT capability_id, slug, stage, ok, reason, duration_ms, "
        "       created_at "
        "FROM verification_log ORDER BY id DESC LIMIT 10")

    return {
        "totals": totals,
        "window_days": window_days,
        "window_success_rate": window_rate,
        "window_counts": {"ok": window_ok, "failed": window_fail,
                           "total": window_total},
        "runtime_mode_counts": {
            "quarantined": int(quarantined or 0),
            "stub_reverted": int(reverted or 0),
        },
        "recent": [
            {
                "capability_id": r["capability_id"],
                "slug": r["slug"],
                "stage": r["stage"],
                "ok": bool(r["ok"]),
                "reason": r["reason"],
                "duration_ms": r["duration_ms"],
                "created_at": r["created_at"],
            }
            for r in recent
        ],
    }


def _queue_section(conn: sqlite3.Connection,
                   min_confidence: float = 0.60) -> Dict[str, Any]:
    """How many candidates are eligible per pool right now."""
    try:
        from components.capability_materialiser import PICKER_QUOTAS
    except Exception:  # noqa: BLE001
        PICKER_QUOTAS = {"fresh_blood_seed+self_judge": 5,
                         "promoter_path+self_judge": 3,
                         "gap_driven": 2}

    depth: Dict[str, int] = {}
    for prov in PICKER_QUOTAS.keys():
        n = _safe_scalar(conn,
            "SELECT COUNT(*) FROM capabilities "
            "WHERE runtime_mode IN ('stub','stub_reverted') "
            "  AND provenance = ? "
            "  AND judge_confidence >= ?",
            (prov, float(min_confidence)))
        depth[prov] = int(n or 0)

    total = sum(depth.values())
    return {"by_provenance": depth, "total": total,
            "min_confidence": min_confidence}


def _live_modules_section(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Count of live generated modules + last 5 promotions."""
    live_count = _safe_scalar(conn,
        "SELECT COUNT(*) FROM capabilities "
        "WHERE runtime_mode = 'generated_module'") or 0

    recent = _safe_rows(conn,
        "SELECT capability_id, concept, slug, outcome, created_at "
        "FROM materialisation_log "
        "WHERE outcome = 'promoted' "
        "ORDER BY id DESC LIMIT 5")

    # Files on disk vs DB rows can drift after quarantine; expose both
    disk_live = 0
    disk_quar = 0
    try:
        if LIVE_DIR.exists():
            disk_live = len([p for p in LIVE_DIR.glob("*.py")
                             if not p.name.startswith("_")])
        if QUARANTINE_DIR.exists():
            disk_quar = len(list(QUARANTINE_DIR.glob("*.py")))
    except Exception:  # noqa: BLE001
        pass

    return {
        "live_count_db": int(live_count),
        "live_count_disk": disk_live,
        "quarantine_count_disk": disk_quar,
        "recent_promotions": [
            {
                "capability_id": r["capability_id"],
                "concept": r["concept"],
                "slug": r["slug"],
                "created_at": r["created_at"],
            }
            for r in recent
        ],
    }


def _gaps_section() -> Dict[str, Any]:
    """How many capability gaps the scanner sees + top 5."""
    try:
        from components.gap_fetcher import iter_capability_gaps
    except Exception as e:  # noqa: BLE001
        return {"error": f"gap_fetcher import failed: {e}",
                "count": 0, "top": []}

    try:
        gaps = list(iter_capability_gaps(fresh=False))
    except Exception as e:  # noqa: BLE001
        return {"error": f"iter_capability_gaps failed: {e}",
                "count": 0, "top": []}

    gaps.sort(key=lambda g: int(getattr(g, "priority", 5) or 5))
    top = [
        {
            "name": getattr(g, "name", ""),
            "description": (getattr(g, "description", "") or "")[:200],
            "priority": int(getattr(g, "priority", 5) or 5),
            "target_kpi": getattr(g, "target_kpi", ""),
        }
        for g in gaps[:5]
    ]
    return {"count": len(gaps), "top": top}


def _health_verdict(materialiser: Dict[str, Any],
                    verifier: Dict[str, Any],
                    queue: Dict[str, Any],
                    live: Dict[str, Any]) -> Dict[str, Any]:
    """Roll everything up into a green/yellow/red verdict."""
    reasons: List[str] = []
    level = "green"

    if not materialiser.get("running", False):
        level = "red"
        reasons.append("materialiser loop is not running")
    elif materialiser.get("last_tick", {}).get("cap_hit"):
        # Cap hit is fine — actually a good sign we're producing
        reasons.append(
            f"daily cap hit "
            f"({materialiser['last_tick']['day_count']}/"
            f"{materialiser['config']['daily_cap']})"
        )
    else:
        picked = materialiser.get("last_tick", {}).get("picked", 0)
        if picked == 0 and queue.get("total", 0) == 0:
            if level == "green":
                level = "yellow"
            reasons.append("queue empty across all provenance pools")

    win = verifier.get("window_success_rate")
    if win is not None and win < 0.5:
        level = "red"
        reasons.append(
            f"verification success rate {win:.0%} < 50% over last "
            f"{verifier.get('window_days', 7)}d"
        )
    elif win is not None and win < 0.75:
        if level == "green":
            level = "yellow"
        reasons.append(
            f"verification success rate {win:.0%} < 75% over last "
            f"{verifier.get('window_days', 7)}d"
        )

    if verifier.get("runtime_mode_counts", {}).get("quarantined", 0) >= 5:
        if level == "green":
            level = "yellow"
        reasons.append(
            f"{verifier['runtime_mode_counts']['quarantined']} "
            f"capabilities permanently quarantined"
        )

    if live.get("live_count_db", 0) == 0:
        if level == "green":
            level = "yellow"
        reasons.append("no live generated_module capabilities yet")

    if not reasons:
        reasons.append("all systems nominal")

    return {"level": level, "reasons": reasons}


# ── Public API ────────────────────────────────────────────────────────────

def build_status(db_path: str) -> Dict[str, Any]:
    """Return the full self-generation status snapshot."""
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat()

    if not os.path.exists(db_path):
        return {
            "ok": False,
            "ts": ts,
            "error": f"knowledge db not found at {db_path}",
            "health": {"level": "red",
                       "reasons": ["knowledge db missing on disk"]},
        }

    conn = _connect(db_path)
    if conn is None:
        return {
            "ok": False,
            "ts": ts,
            "error": "could not open knowledge db",
            "health": {"level": "red",
                       "reasons": ["sqlite connect failed"]},
        }

    try:
        materialiser = _materialiser_section(conn)
        min_conf = (materialiser.get("config") or {}).get(
            "min_confidence", 0.60,
        )
        verifier = _verifier_section(conn)
        queue = _queue_section(conn, min_confidence=min_conf)
        live = _live_modules_section(conn)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    gaps = _gaps_section()
    health = _health_verdict(materialiser, verifier, queue, live)

    return {
        "ok": True,
        "ts": ts,
        "health": health,
        "materialiser": materialiser,
        "verifier": verifier,
        "queue": queue,
        "live_modules": live,
        "gaps": gaps,
    }


__all__ = ["build_status"]
