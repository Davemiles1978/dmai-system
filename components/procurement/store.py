"""Store wrapping the procurement DB (PR K).

Mirrors the style of :mod:`components.treasury.treasury_ledger`: WAL,
busy_timeout, row_factory, and a small state kv-table. All methods accept
an optional ``db_path`` so tests can point at a tmp file.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from components.procurement.config import default_procurement_path
from components.procurement.schema import SCHEMA

logger = logging.getLogger(__name__)

STATE_INSTALL_TS   = "procurement:install_ts"
STATE_LAST_RUN_TS  = "procurement:last_run_ts"
STATE_LAST_SUMMARY = "procurement:last_summary"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProcurementStore:
    """Thin wrapper around ``data/dmai_procurement.db``."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or default_procurement_path()

    # ── connection ──────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(self.db_path, timeout=30.0)
        c.row_factory = sqlite3.Row
        return c

    # ── init / state ─────────────────────────────────────────────────────────

    def init_db(self) -> Dict[str, Any]:
        with self._conn() as c:
            for ddl in SCHEMA:
                c.execute(ddl)
            if self._state_get(c, STATE_INSTALL_TS) is None:
                self._state_set(c, STATE_INSTALL_TS, _now_iso())
            c.commit()
            return {"install_ts": self._state_get(c, STATE_INSTALL_TS)}

    @staticmethod
    def _state_get(c: sqlite3.Connection, key: str) -> Optional[str]:
        row = c.execute(
            "SELECT value FROM procurement_state WHERE key = ?", (key,),
        ).fetchone()
        return row["value"] if row else None

    @staticmethod
    def _state_set(c: sqlite3.Connection, key: str, value: str) -> None:
        c.execute(
            "INSERT INTO procurement_state (key, value, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at",
            (key, value, _now_iso()),
        )

    def set_state(self, key: str, value: str) -> None:
        with self._conn() as c:
            self._state_set(c, key, value)
            c.commit()

    def get_state(self, key: str) -> Optional[str]:
        with self._conn() as c:
            return self._state_get(c, key)

    # ── catalog ────────────────────────────────────────────────────────────

    def insert_catalog(self, row: Dict[str, Any]) -> int:
        """Insert one normalised hardware row, return its id."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO hardware_catalog "
                "(source, url, name, cpu, cpu_passmark, tdp_w, idle_w, "
                " ram_gb, storage_gb, price_gbp, currency_orig, price_orig, "
                " fx_used, fetched_ts, raw_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row.get("source"),
                    row.get("url"),
                    row.get("name"),
                    row.get("cpu"),
                    row.get("cpu_passmark"),
                    row.get("tdp_w"),
                    row.get("idle_w"),
                    row.get("ram_gb"),
                    row.get("storage_gb"),
                    row.get("price_gbp"),
                    row.get("currency_orig"),
                    row.get("price_orig"),
                    row.get("fx_used"),
                    row.get("fetched_ts") or _now_iso(),
                    json.dumps(row.get("raw_json"))
                    if not isinstance(row.get("raw_json"), str)
                    else row.get("raw_json"),
                ),
            )
            c.commit()
            return int(cur.lastrowid)

    def list_catalog(self, *, limit: int = 200) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM hardware_catalog ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_catalog(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM hardware_catalog")
            c.commit()

    def clear_shortlist(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM procurement_shortlist")
            c.commit()

    # ── shortlist ────────────────────────────────────────────────────────────

    def insert_shortlist_row(self, row: Dict[str, Any]) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO procurement_shortlist "
                "(run_ts, rank, hardware_id, tco_gbp_3yr, capex_gbp, "
                " opex_3yr_gbp, headroom_ram_x, headroom_cpu_x, verdict, "
                " notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    row.get("run_ts") or _now_iso(),
                    int(row.get("rank")),
                    int(row.get("hardware_id")),
                    float(row.get("tco_gbp_3yr")),
                    float(row.get("capex_gbp")),
                    float(row.get("opex_3yr_gbp")),
                    row.get("headroom_ram_x"),
                    row.get("headroom_cpu_x"),
                    row.get("verdict"),
                    row.get("notes"),
                ),
            )
            c.commit()
            return int(cur.lastrowid)

    def latest_run_ts(self) -> Optional[str]:
        with self._conn() as c:
            row = c.execute(
                "SELECT run_ts FROM procurement_shortlist "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row["run_ts"] if row else None

    def get_shortlist(self, run_ts: Optional[str] = None,
                      ) -> List[Dict[str, Any]]:
        """Return shortlist rows joined to their catalog hardware.

        If ``run_ts`` is None, returns the latest run's rows.
        """
        with self._conn() as c:
            if run_ts is None:
                latest = c.execute(
                    "SELECT run_ts FROM procurement_shortlist "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if latest is None:
                    return []
                run_ts = latest["run_ts"]
            rows = c.execute(
                "SELECT s.*, h.name AS hw_name, h.source AS hw_source, "
                "       h.cpu AS hw_cpu, h.cpu_passmark AS hw_cpu_passmark, "
                "       h.ram_gb AS hw_ram_gb, h.idle_w AS hw_idle_w, "
                "       h.price_gbp AS hw_price_gbp, h.url AS hw_url "
                "FROM procurement_shortlist s "
                "JOIN hardware_catalog h ON h.id = s.hardware_id "
                "WHERE s.run_ts = ? ORDER BY s.rank ASC",
                (run_ts,),
            ).fetchall()
        return [dict(r) for r in rows]


__all__ = [
    "ProcurementStore",
    "STATE_INSTALL_TS",
    "STATE_LAST_RUN_TS",
    "STATE_LAST_SUMMARY",
]
