"""Layer 4 chunk L4-5 — Self-Generation Autonomy Tracker.

Rolling 7-day-window metric used to decide whether the human operator can
hand off oversight of the Layer 4 self-generation loop.

    score = auto_approved_capabilities_with_passing_tests / capability_gaps_detected
            (both counts within the trailing 7 calendar days)

The tracker stores its events in ``data/dmai_knowledge.db`` table
``sg_autonomy_log`` (schema auto-created on first use). The companion Flask
endpoint ``GET /api/self-generation/autonomy-score`` is registered in
``dmai_core_complete.py`` via the standard additive try/except block.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List


class SelfGenAutonomyTracker:
    """Track Layer 4's autonomy score over a rolling 7-day window."""

    HANDOFF_THRESHOLD = 0.7
    WINDOW_DAYS = 7

    def __init__(self, data_path: str = "data") -> None:
        self.data_path = data_path
        self.db_path = str(Path(data_path) / "dmai_knowledge.db")
        self._ensure_table()

    # ------------------------------------------------------------------
    # DB plumbing — uses safe_open_kdb to satisfy preflight check 0.
    # ------------------------------------------------------------------

    def _conn(self):
        from components.db import safe_open_kdb
        return safe_open_kdb(self.db_path, timeout=5)

    def _ensure_table(self) -> None:
        try:
            conn = self._conn()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sg_autonomy_log (
                        id SERIAL PRIMARY KEY,
                        ts TEXT NOT NULL,
                        event TEXT NOT NULL,
                        gap_name TEXT,
                        edit_id TEXT,
                        meta TEXT
                    );
                    CREATE INDEX IF NOT EXISTS ix_sg_autonomy_log_ts
                        ON sg_autonomy_log(ts);
                    CREATE INDEX IF NOT EXISTS ix_sg_autonomy_log_event
                        ON sg_autonomy_log(event, ts);
                    """
                )
                conn.commit()
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            # Non-fatal — first compute_score() call will see empty counts.
            return

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record_gap_detected(self, gap_name: str, meta: Dict[str, Any] | None = None) -> None:
        self._record("gap_detected", gap_name=gap_name, edit_id=None, meta=meta)

    def record_capability_approved(
        self,
        gap_name: str,
        edit_id: str,
        meta: Dict[str, Any] | None = None,
    ) -> None:
        self._record("capability_approved", gap_name=gap_name, edit_id=edit_id, meta=meta)

    def _record(
        self,
        event: str,
        gap_name: str | None,
        edit_id: str | None,
        meta: Dict[str, Any] | None,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        try:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO sg_autonomy_log (ts, event, gap_name, edit_id, meta) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (ts, event, gap_name, edit_id,
                     json.dumps(meta) if meta else None),
                )
                conn.commit()
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            return

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def compute_score(self) -> Dict[str, Any]:
        """Return the JSON-ready payload for the autonomy-score endpoint."""
        window_start = datetime.now(timezone.utc) - timedelta(days=self.WINDOW_DAYS)
        window_start_iso = window_start.isoformat()

        gaps = self._count_event_since("gap_detected", window_start_iso)
        approved = self._count_event_since("capability_approved", window_start_iso)
        rolled_back = self._count_rolled_back_commits()

        score = (approved / gaps) if gaps > 0 else 0.0
        sustained = self._sustained_days_above(self.HANDOFF_THRESHOLD)
        handoff_ready = bool(
            score >= self.HANDOFF_THRESHOLD
            and rolled_back == 0
            and sustained >= self.WINDOW_DAYS
        )

        return {
            "score": round(score, 4),
            "window_days": self.WINDOW_DAYS,
            "gaps_detected": gaps,
            "auto_approved": approved,
            "rolled_back_commits": rolled_back,
            "handoff_ready": handoff_ready,
            "sustained_days": sustained,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    def _count_event_since(self, event: str, since_iso: str) -> int:
        try:
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM sg_autonomy_log "
                    "WHERE event=? AND ts >= ?",
                    (event, since_iso),
                ).fetchone()
                return int(row[0]) if row else 0
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            return 0

    def _count_rolled_back_commits(self) -> int:
        """Count revert commits in git log over the rolling window."""
        try:
            result = subprocess.run(
                [
                    "git", "log", "--oneline",
                    f"--since={self.WINDOW_DAYS} days ago",
                    "--grep=Revert",
                ],
                capture_output=True, text=True, cwd=".", timeout=10,
            )
            return len([l for l in result.stdout.splitlines() if l.strip()])
        except Exception:  # noqa: BLE001
            return 0

    def _sustained_days_above(self, threshold: float) -> int:
        """Return consecutive most-recent days where daily score >= threshold.

        Daily score = approved_today / gaps_today (0 if gaps_today == 0,
        which breaks the streak — a day with no gaps cannot prove autonomy).
        """
        try:
            conn = self._conn()
            try:
                # Pull events grouped by date for the last WINDOW_DAYS days.
                cutoff = (
                    datetime.now(timezone.utc) - timedelta(days=self.WINDOW_DAYS)
                ).isoformat()
                rows = conn.execute(
                    "SELECT substr(ts, 1, 10) AS day, event, COUNT(*) "
                    "FROM sg_autonomy_log WHERE ts >= ? "
                    "GROUP BY day, event ORDER BY day DESC",
                    (cutoff,),
                ).fetchall()
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            return 0

        # Build {day: {gap_detected: n, capability_approved: m}}
        by_day: Dict[str, Dict[str, int]] = {}
        for day, event, count in rows:
            by_day.setdefault(day, {})[event] = int(count)

        # Walk back from today; count consecutive days meeting threshold.
        streak = 0
        for i in range(self.WINDOW_DAYS):
            day = (
                datetime.now(timezone.utc) - timedelta(days=i)
            ).date().isoformat()
            counts = by_day.get(day, {})
            gaps_today = counts.get("gap_detected", 0)
            approved_today = counts.get("capability_approved", 0)
            if gaps_today == 0:
                break  # No evidence today — can't prove autonomy.
            ratio = approved_today / gaps_today
            if ratio < threshold:
                break
            streak += 1
        return streak
