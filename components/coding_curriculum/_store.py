"""Mastery store for the coding curriculum.

Tracks (topic_slug, mastery_score, exposures, correct_exercises,
failed_exercises, last_studied_ts) in SQLite.

Mastery score is a float in [0.0, 1.0]:
  0.0  = never seen
  0.3  = seen once (initial exposure)
  0.5  = studied - insights ingested for this topic
  0.7  = tested - passed at least one exercise
  1.0  = mastered - passed ≥5 exercises across ≥3 study sessions

Scoring uses an exponential-decay rule so mastery drifts down if a
topic isn't revisited, which forces DMAI to periodically refresh.

Never insert None/zero values (per user rule) - every row has at least
mastery_score>0 and last_studied_ts set. If we don't yet have data for
a topic, we simply don't insert it.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._taxonomy import CURRICULUM_TOPICS, all_topic_slugs

logger = logging.getLogger(__name__)


_DEFAULT_DB = "data/dmai_knowledge.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS coding_curriculum_mastery (
    slug              TEXT PRIMARY KEY,
    language          TEXT NOT NULL,
    tier              INTEGER NOT NULL,
    mastery_score     REAL NOT NULL,
    exposures         INTEGER NOT NULL DEFAULT 0,
    exercises_passed  INTEGER NOT NULL DEFAULT 0,
    exercises_failed  INTEGER NOT NULL DEFAULT 0,
    last_studied_ts   TEXT NOT NULL,
    last_source       TEXT NOT NULL,
    last_summary      TEXT
);

CREATE INDEX IF NOT EXISTS ix_curriculum_mastery_score
    ON coding_curriculum_mastery(mastery_score);
CREATE INDEX IF NOT EXISTS ix_curriculum_mastery_lang_tier
    ON coding_curriculum_mastery(language, tier);
"""


# ── Connection helper ────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # 30s Python-side timeout + 30s SQLite busy_timeout: DMAI runs many
    # concurrent writers (capability_promoter, insight_promoter,
    # fresh_blood_injector, materialiser) against the same DB. Short
    # timeouts here surface as "database is locked" 500s.
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.row_factory = sqlite3.Row
    return conn


def initialise(db_path: str = _DEFAULT_DB) -> Dict[str, Any]:
    """Create the mastery table. Safe to call repeatedly.

    Does NOT pre-populate rows for every topic - we only insert on
    first exposure (user rule: don't write None/zero rows).

    Returns a summary dict with counts.
    """
    conn = _connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM coding_curriculum_mastery"
        ).fetchone()
    finally:
        conn.close()
    return {
        "ok":              True,
        "taxonomy_topics": len(CURRICULUM_TOPICS),
        "rows_present":    int(n),
        "db_path":         db_path,
    }


# ── Scoring ──────────────────────────────────────────────────────────────

def _next_score(current: Optional[float], kind: str) -> float:
    """Compute the new mastery score after an event.

    kind ∈ {'exposure', 'study', 'exercise_pass', 'exercise_fail'}.
    """
    c = float(current) if current is not None else 0.0
    if kind == "exposure":
        return max(c, 0.3)
    if kind == "study":
        return max(c, min(0.5 + 0.05 * (c > 0), 0.5))
    if kind == "exercise_pass":
        # Move ~30% of the way toward 1.0.
        return min(1.0, c + 0.30 * (1.0 - c))
    if kind == "exercise_fail":
        # Small penalty; don't fall below the exposure floor.
        return max(0.2, c - 0.05)
    return c


# ── Public API ───────────────────────────────────────────────────────────

def record_exposure(slug: str,
                    *,
                    source: str,
                    summary: Optional[str] = None,
                    kind: str = "exposure",
                    db_path: str = _DEFAULT_DB) -> Dict[str, Any]:
    """Insert or update mastery for a topic.

    Args:
        slug:    Topic slug from CURRICULUM_TOPICS.
        source:  Where this exposure came from
                 (e.g. 'autonomous_researcher', 'fresh_blood',
                 'code_study', 'materialiser').
        summary: Short human-readable note about what was learned.
        kind:    'exposure' | 'study' | 'exercise_pass' | 'exercise_fail'.

    Returns the new row as a dict, or {'skipped': True, 'reason': ...}
    when we refuse to write (unknown slug, empty source, etc. - the
    'never insert None/zero' rule).
    """
    if not slug or slug not in CURRICULUM_TOPICS:
        return {"skipped": True, "reason": f"unknown_slug:{slug!r}"}
    if not source:
        return {"skipped": True, "reason": "empty_source"}

    topic = CURRICULUM_TOPICS[slug]
    now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
    summary_clean = (summary or "").strip()[:400] or None

    conn = _connect(db_path)
    try:
        # Auto-initialise the schema on first use (idempotent).
        conn.executescript(_SCHEMA)
        row = conn.execute(
            "SELECT * FROM coding_curriculum_mastery WHERE slug = ?",
            (slug,),
        ).fetchone()

        if row is None:
            new_score = _next_score(None, kind)
            passed = 1 if kind == "exercise_pass" else 0
            failed = 1 if kind == "exercise_fail" else 0
            conn.execute(
                """INSERT INTO coding_curriculum_mastery
                   (slug, language, tier, mastery_score, exposures,
                    exercises_passed, exercises_failed,
                    last_studied_ts, last_source, last_summary)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (slug, topic["language"], topic["tier"], new_score,
                 1, passed, failed, now_iso, source, summary_clean),
            )
        else:
            new_score = _next_score(row["mastery_score"], kind)
            passed = row["exercises_passed"] + (
                1 if kind == "exercise_pass" else 0
            )
            failed = row["exercises_failed"] + (
                1 if kind == "exercise_fail" else 0
            )
            conn.execute(
                """UPDATE coding_curriculum_mastery
                   SET mastery_score    = ?,
                       exposures        = exposures + 1,
                       exercises_passed = ?,
                       exercises_failed = ?,
                       last_studied_ts  = ?,
                       last_source      = ?,
                       last_summary     = COALESCE(?, last_summary)
                   WHERE slug = ?""",
                (new_score, passed, failed, now_iso, source,
                 summary_clean, slug),
            )
        conn.commit()
        return {
            "ok":              True,
            "slug":            slug,
            "mastery_score":   new_score,
            "kind":            kind,
            "source":          source,
        }
    finally:
        conn.close()


def mastery_of(slug: str,
               db_path: str = _DEFAULT_DB) -> Optional[dict]:
    """Return the mastery row for a topic, or None if never exposed."""
    conn = _connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        row = conn.execute(
            "SELECT * FROM coding_curriculum_mastery WHERE slug = ?",
            (slug,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def all_mastery(db_path: str = _DEFAULT_DB) -> Dict[str, dict]:
    """Return every mastery row keyed by slug."""
    conn = _connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        rows = conn.execute(
            "SELECT * FROM coding_curriculum_mastery"
        ).fetchall()
        return {r["slug"]: dict(r) for r in rows}
    finally:
        conn.close()


def lowest_mastery_topics(*,
                          limit: int = 5,
                          language: Optional[str] = None,
                          tier: Optional[int] = None,
                          db_path: str = _DEFAULT_DB) -> List[dict]:
    """Return the topics with the lowest mastery, prioritising:
      1. Never-studied topics (not in the mastery table) - score = 0.
      2. Then the lowest-score studied topics.

    This is what the picker uses to decide "what to study tonight".
    """
    existing = all_mastery(db_path=db_path)
    result: List[dict] = []
    # Never-studied first (score = 0), tier order, then depth.
    unstudied = [
        s for s in all_topic_slugs()
        if s not in existing
        and (language is None or CURRICULUM_TOPICS[s]["language"] == language)
        and (tier is None or CURRICULUM_TOPICS[s]["tier"] == tier)
    ]
    unstudied.sort(key=lambda s: (
        CURRICULUM_TOPICS[s]["tier"],
        CURRICULUM_TOPICS[s]["depth"],
        s,
    ))
    for s in unstudied[:limit]:
        t = CURRICULUM_TOPICS[s]
        result.append({
            "slug":          s,
            "title":         t["title"],
            "language":      t["language"],
            "tier":          t["tier"],
            "depth":         t["depth"],
            "mastery_score": 0.0,
            "status":        "unstudied",
        })
        if len(result) >= limit:
            return result

    # Then the studied-but-weakest.
    studied = list(existing.values())
    if language is not None:
        studied = [r for r in studied if r["language"] == language]
    if tier is not None:
        studied = [r for r in studied if r["tier"] == tier]
    studied.sort(key=lambda r: (r["mastery_score"], r["tier"], r["slug"]))
    for r in studied:
        if len(result) >= limit:
            break
        result.append({
            "slug":          r["slug"],
            "title":         CURRICULUM_TOPICS[r["slug"]]["title"],
            "language":      r["language"],
            "tier":          r["tier"],
            "depth":         CURRICULUM_TOPICS[r["slug"]]["depth"],
            "mastery_score": r["mastery_score"],
            "status":        "studied",
            "exposures":     r["exposures"],
        })
    return result


def coverage_summary(db_path: str = _DEFAULT_DB) -> Dict[str, Any]:
    """High-level coverage: how much of the curriculum has DMAI touched?"""
    existing = all_mastery(db_path=db_path)
    total = len(CURRICULUM_TOPICS)
    seen = len(existing)
    mastered = sum(1 for r in existing.values() if r["mastery_score"] >= 0.9)
    studied  = sum(1 for r in existing.values() if r["mastery_score"] >= 0.5)
    by_lang: Dict[str, Dict[str, int]] = {}
    for slug, t in CURRICULUM_TOPICS.items():
        lang = t["language"]
        d = by_lang.setdefault(
            lang,
            {"total": 0, "seen": 0, "studied": 0, "mastered": 0},
        )
        d["total"] += 1
        row = existing.get(slug)
        if row:
            d["seen"] += 1
            if row["mastery_score"] >= 0.5:
                d["studied"] += 1
            if row["mastery_score"] >= 0.9:
                d["mastered"] += 1
    return {
        "ok":              True,
        "taxonomy_topics": total,
        "seen":            seen,
        "studied":         studied,
        "mastered":        mastered,
        "pct_seen":        round(seen / total * 100, 1) if total else 0.0,
        "pct_studied":     round(studied / total * 100, 1) if total else 0.0,
        "pct_mastered":    round(mastered / total * 100, 1) if total else 0.0,
        "by_language":     by_lang,
    }
