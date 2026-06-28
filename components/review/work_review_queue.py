"""
WorkReviewQueue — SQLite-backed human review gate for long-form Alex output.

Policy (per user directive 2026-06-24):
  "Any books created under Alex need review before auto publishing until
   the appropriate level of skill is accomplished. ... The writing skill
   would also need to be assessed for research papers, articles, TV
   Scripts, etc.."

Behaviour:
  - Every long-form submission is scored by SkillAssessor and parked in
    status='pending'. It does NOT auto-publish.
  - User reviews via dashboard / API. Choices: approve, reject, request
    revisions.
  - User can later 'graduate' a work_type once skill is consistently high
    (assessor enforces eligibility). Graduated types may bypass the queue
    in the publishing orchestrator (config flag — defaults to STILL
    queuing unless explicitly enabled per type).

Tables:
  work_review_queue (id, work_type, title, payload_json, summary,
    status, scores_json, overall_score, submitted_at, decided_at,
    decided_by, decision_notes, source_component, persona)

Statuses: pending | approved | rejected | revise | published
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from .skill_assessor import SkillAssessor, WORK_TYPES, get_skill_assessor
from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

VALID_STATUSES = {"pending", "approved", "rejected", "revise", "published"}


class WorkReviewQueue:
    """Human review gate for Alex long-form work."""

    def __init__(self, data_path: str | Path = "data", assessor: Optional[SkillAssessor] = None):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.data_path / "dmai_knowledge.db")
        self._lock = threading.RLock()
        self.assessor = assessor or SkillAssessor(data_path=self.data_path)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = safe_open_kdb(self.db_path, timeout=10, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS work_review_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_uid TEXT UNIQUE,
                    work_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    summary TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    scores_json TEXT,
                    overall_score REAL,
                    passed INTEGER,
                    submitted_at TEXT NOT NULL DEFAULT (datetime('now')),
                    decided_at TEXT,
                    decided_by TEXT,
                    decision_notes TEXT,
                    source_component TEXT,
                    persona TEXT
                )
                """
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_wrq_status ON work_review_queue(status)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_wrq_type ON work_review_queue(work_type)")
            c.commit()

    # ---------- submission ----------
    def submit(
        self,
        work_type: str,
        title: str,
        payload: Dict[str, Any],
        summary: Optional[str] = None,
        source_component: Optional[str] = None,
        persona: Optional[str] = None,
        run_assessment: bool = True,
    ) -> Dict[str, Any]:
        """Park a work for review. Scores it via SkillAssessor unless suppressed."""
        if work_type not in WORK_TYPES:
            raise ValueError(
                f"Unknown work_type '{work_type}'. Valid: {sorted(WORK_TYPES.keys())}"
            )

        submission_uid = f"sub_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        scores: Dict[str, float] = {}
        overall: Optional[float] = None
        passed: Optional[int] = None
        assessment: Dict[str, Any] = {}
        if run_assessment:
            try:
                assessment = self.assessor.assess(
                    submission_id=submission_uid,
                    work_type=work_type,
                    payload=payload,
                ) or {}
                scores = assessment.get("scores") or {}
                overall = assessment.get("overall")
                passed = 1 if assessment.get("passed") else 0
            except Exception as e:  # pragma: no cover
                logger.warning(f"SkillAssessor failed on submit: {e}")

        with self._lock, self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO work_review_queue
                  (submission_uid, work_type, title, payload_json, summary,
                   status, scores_json, overall_score, passed,
                   source_component, persona)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    submission_uid,
                    work_type,
                    title,
                    json.dumps(payload, default=str),
                    summary,
                    json.dumps(scores) if scores else None,
                    overall,
                    passed,
                    source_component,
                    persona,
                ),
            )
            row_id = cur.lastrowid
            c.commit()

        logger.info(
            f"WorkReviewQueue: submitted #{row_id} ({work_type}) "
            f"score={overall} title={title!r}"
        )
        item = self.get(row_id) or {}
        item["assessment"] = assessment
        return item

    # ---------- queries ----------
    def get(self, item_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM work_review_queue WHERE id = ?", (item_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list(
        self,
        status: Optional[str] = "pending",
        work_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        q = "SELECT * FROM work_review_queue WHERE 1=1"
        args: List[Any] = []
        if status:
            q += " AND status = ?"
            args.append(status)
        if work_type:
            q += " AND work_type = ?"
            args.append(work_type)
        q += " ORDER BY submitted_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_pending(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.list(status="pending", limit=limit)

    # ---------- decisions ----------
    def approve(self, item_id: int, notes: str = "", decided_by: str = "user") -> Dict[str, Any]:
        return self._decide(item_id, "approved", notes, decided_by)

    def reject(self, item_id: int, notes: str = "", decided_by: str = "user") -> Dict[str, Any]:
        return self._decide(item_id, "rejected", notes, decided_by)

    def request_revisions(
        self, item_id: int, notes: str = "", decided_by: str = "user"
    ) -> Dict[str, Any]:
        return self._decide(item_id, "revise", notes, decided_by)

    def mark_published(self, item_id: int, notes: str = "") -> Dict[str, Any]:
        return self._decide(item_id, "published", notes, decided_by="system")

    def _decide(
        self, item_id: int, status: str, notes: str, decided_by: str
    ) -> Dict[str, Any]:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'")
        with self._lock, self._conn() as c:
            cur = c.execute(
                """
                UPDATE work_review_queue
                SET status = ?, decided_at = datetime('now'),
                    decided_by = ?, decision_notes = ?
                WHERE id = ?
                """,
                (status, decided_by, notes, item_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Review item #{item_id} not found")
            c.commit()
        item = self.get(item_id)
        logger.info(
            f"WorkReviewQueue: #{item_id} -> {status} by {decided_by} notes={notes!r}"
        )
        return item

    # ---------- stats ----------
    def stats(self) -> Dict[str, Any]:
        with self._conn() as c:
            by_status = {
                r["status"]: r["n"]
                for r in c.execute(
                    "SELECT status, COUNT(*) AS n FROM work_review_queue GROUP BY status"
                ).fetchall()
            }
            by_type = {
                r["work_type"]: r["n"]
                for r in c.execute(
                    "SELECT work_type, COUNT(*) AS n FROM work_review_queue GROUP BY work_type"
                ).fetchall()
            }
            pending_count = by_status.get("pending", 0)
            recent = c.execute(
                "SELECT id, work_type, title, status, overall_score, submitted_at "
                "FROM work_review_queue ORDER BY submitted_at DESC LIMIT 10"
            ).fetchall()
        # Add per-type graduation flag
        graduation = {}
        for wt in WORK_TYPES.keys():
            try:
                graduation[wt] = bool(self.assessor.is_graduated(wt))
            except Exception:
                graduation[wt] = False
        return {
            "pending_count": pending_count,
            "by_status": by_status,
            "by_type": by_type,
            "graduation": graduation,
            "recent": [dict(r) for r in recent],
        }

    # ---------- helpers ----------
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for k in ("payload_json", "scores_json"):
            if d.get(k):
                try:
                    d[k.replace("_json", "")] = json.loads(d[k])
                except Exception:
                    pass
        return d


_singleton: Optional[WorkReviewQueue] = None


def get_work_review_queue(data_path: str = "data") -> WorkReviewQueue:
    global _singleton
    if _singleton is None:
        _singleton = WorkReviewQueue(data_path=data_path, assessor=get_skill_assessor(data_path))
    return _singleton
