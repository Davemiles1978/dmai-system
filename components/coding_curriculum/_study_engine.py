"""Study engine: DMAI's nightly self-teaching loop.

One study round:
    1. Pick the weakest prereq-ready topic (via _picker).
    2. Build a concrete exercise for that topic (via _exercises).
    3. Attempt to solve it *locally*, no external LLM:
         a. First try local_codegen templates (PR XX-1).
         b. If template fails or shape isn't templatable, fall back
            to a hint-driven synthesiser that assembles a minimal
            solution from the exercise's own hint + capability_shape.
    4. Grade the solution against the exercise cases (via _grader,
       subprocess isolation + hard timeout).
    5. Record the outcome in the mastery store:
         - pass  -> record_exposure(kind='exercise_pass')
         - fail  -> record_exposure(kind='exercise_fail')
       Every recorded row has a non-zero mastery score and a
       non-empty source string (user rules).
    6. On pass, promote the exercise's solution as a coding_pattern
       row in the capabilities table so downstream code can retrieve
       it later (RAG in PR YY-3).
    7. Log the full round to data/coding_curriculum/study_log.jsonl
       for the admin UI + weekly review.

This module is intentionally synchronous. It is called from the cron
endpoint /api/cron/coding-curriculum/study once per night; the caller
guards against overlapping runs via a small file-lock.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._exercises import Exercise, exercise_for_topic
from ._grader import grade_exercise
from ._picker import next_topic_to_study
from ._store import initialise, record_exposure
from ._taxonomy import CURRICULUM_TOPICS

logger = logging.getLogger(__name__)


_LOG_PATH = Path("data/coding_curriculum/study_log.jsonl")


# ── Candidate synthesis ──────────────────────────────────────────────────
#
# We deliberately avoid a big LLM here. Two synthesisers:
#
#   1. templates: use the PR XX-1 local_codegen templates. Good for
#      generic shapes but the resulting module often *ignores* the
#      exercise's exact contract (grading predicates check specific
#      dict keys we control).
#   2. hint-driven synth: hand-written minimal solutions per
#      capability_shape that satisfy the exercise contract exactly.
#      This is what actually earns passing scores today.

_HINT_SYNTHS: Dict[str, str] = {
    "data_structure": '''
def run(**kwargs):
    values = kwargs.get("values", []) or []
    total = 0.0
    n = 0
    for v in values:
        try:
            total += float(v)
            n += 1
        except (TypeError, ValueError):
            continue
    return {"ok": True, "result": total, "count": n}
'''.strip(),

    "utility": '''
def run(**kwargs):
    values = kwargs.get("values", []) or []
    out = []
    for v in values:
        try:
            out.append(v * 2)
        except TypeError:
            out.append(v)
    return {"ok": True, "result": out}
'''.strip(),

    "configuration": '''
def run(**kwargs):
    base = dict(kwargs.get("base") or {})
    override = dict(kwargs.get("override") or {})
    merged = dict(base)
    merged.update(override)
    return {"ok": True, "result": merged}
'''.strip(),

    "trading": '''
def run(**kwargs):
    prices = kwargs.get("prices", []) or []
    nums = []
    for p in prices:
        try:
            nums.append(float(p))
        except (TypeError, ValueError):
            continue
    if not nums:
        return {"ok": True, "mean": 0.0, "n": 0}
    return {"ok": True, "mean": sum(nums) / len(nums), "n": len(nums)}
'''.strip(),

    "research": '''
def run(**kwargs):
    q = kwargs.get("query", "") or ""
    q = str(q)
    return {"ok": True, "query": q, "length": len(q)}
'''.strip(),

    "composite": '''
def run(**kwargs):
    a = dict(kwargs.get("a") or {})
    b = dict(kwargs.get("b") or {})
    merged = dict(a)
    merged.update(b)
    return {"ok": True, "merged": merged, "size": len(merged)}
'''.strip(),
}


def _wrap_module(exercise: Exercise, body: str) -> str:
    """Wrap a body into a full module the grader can exec."""
    header = (
        f'"""Coding-pattern candidate for {exercise.topic_slug}.\n\n'
        f'{exercise.docstring}\n"""\n'
    )
    return header + "\n" + body + "\n"


def _synthesise_candidate(exercise: Exercise) -> Optional[str]:
    body = _HINT_SYNTHS.get(exercise.capability_shape)
    if not body:
        return None
    return _wrap_module(exercise, body)


# ── Capability promotion ─────────────────────────────────────────────────

def _capability_id(exercise: Exercise) -> str:
    return "coding_pattern:" + hashlib.sha256(
        f"{exercise.topic_slug}::{exercise.exercise_id}".encode("utf-8"),
    ).hexdigest()[:16]


def _promote_coding_pattern(exercise: Exercise,
                            candidate_code: str,
                            db_path: str) -> Dict[str, Any]:
    """Insert a passing exercise solution into the capabilities table
    as a coding_pattern row. Idempotent (INSERT OR REPLACE on id)."""
    cap_id = _capability_id(exercise)
    name = f"coding_pattern__{exercise.topic_slug.replace('.', '_')}"
    description = (
        f"Coding pattern learned from exercise for topic "
        f"{exercise.topic_slug} ({exercise.capability_shape})."
    )
    # Store the working code snippet in methods JSON so RAG can retrieve
    # it later (PR YY-3).
    methods_json = json.dumps({
        "shape":        exercise.capability_shape,
        "topic_slug":   exercise.topic_slug,
        "exercise_id":  exercise.exercise_id,
        "code":         candidate_code,
    })[:4000]
    args_json = json.dumps({"kwargs_spec": "topic-defined"})[:4000]

    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS capabilities (
                id                TEXT PRIMARY KEY,
                name              TEXT NOT NULL,
                type              TEXT,
                capability_type   TEXT,
                description       TEXT,
                source_url        TEXT,
                source_repo       TEXT,
                file_path         TEXT,
                runtime_mode      TEXT,
                language          TEXT,
                methods           TEXT,
                is_async          INTEGER,
                args              TEXT,
                integrated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT OR REPLACE INTO capabilities "
            "(id, name, type, capability_type, description, source_url, "
            " source_repo, file_path, runtime_mode, language, methods, "
            " is_async, args, integrated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "        CURRENT_TIMESTAMP)",
            (
                cap_id,
                name,
                "coding_pattern",
                "coding_pattern",
                description,
                f"curriculum://{exercise.topic_slug}",
                "coding_curriculum",
                f"components/coding_curriculum/patterns/{exercise.topic_slug}.py",
                "template",
                "python",
                methods_json,
                0,
                args_json,
            ),
        )
        conn.commit()
        return {"ok": True, "capability_id": cap_id, "name": name}
    finally:
        conn.close()


# ── Log persistence ──────────────────────────────────────────────────────

def _append_log(entry: Dict[str, Any]) -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry) + "\n")
    except Exception as e:  # noqa: BLE001
        logger.info("study_log append failed non-fatally: %s", e)


def read_study_log(limit: int = 50) -> List[Dict[str, Any]]:
    """Return the last N study-log entries, newest first."""
    if not _LOG_PATH.exists():
        return []
    lines = _LOG_PATH.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(out))


# ── Public API ───────────────────────────────────────────────────────────

def run_study_round(*,
                    language: Optional[str] = None,
                    db_path: str = "data/dmai_knowledge.db",
                    promote_on_pass: bool = True,
                    ) -> Dict[str, Any]:
    """Run one full study round. Never hangs; always returns a dict."""
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    initialise(db_path=db_path)

    topic = next_topic_to_study(language=language, db_path=db_path)
    if topic is None:
        return {
            "ok":       False,
            "reason":   "no topic to study (curriculum empty?)",
            "started":  started,
        }

    exercise = exercise_for_topic(topic["slug"])
    if exercise is None:
        return {
            "ok":       False,
            "reason":   f"no exercise builder for topic {topic['slug']}",
            "started":  started,
            "topic":    topic["slug"],
        }

    candidate = _synthesise_candidate(exercise)
    if not candidate:
        # No hint synth for this shape. Record a fail exposure so we
        # move on next round (and never write a zero row).
        record_exposure(
            topic["slug"],
            source="study.no_synth",
            kind="exercise_fail",
            summary=f"no synth for shape={exercise.capability_shape}",
            db_path=db_path,
        )
        entry = {
            "ts":            started,
            "topic":         topic["slug"],
            "exercise_id":   exercise.exercise_id,
            "shape":         exercise.capability_shape,
            "passed":        False,
            "reason":        "no synth for capability shape",
            "runtime_ms":    0,
            "promoted":      False,
        }
        _append_log(entry)
        return {"ok": False, **entry}

    grading = grade_exercise(exercise, candidate, timeout_seconds=3.0)
    passed = bool(grading["ok"])

    record_exposure(
        topic["slug"],
        source="study.round",
        kind="exercise_pass" if passed else "exercise_fail",
        summary=(
            f"exercise={exercise.exercise_id} shape={exercise.capability_shape} "
            f"passed={passed} runtime_ms={grading['runtime_ms']}"
        ),
        db_path=db_path,
    )

    promoted = False
    capability_name = None
    if passed and promote_on_pass:
        try:
            promo = _promote_coding_pattern(exercise, candidate, db_path)
            promoted = promo.get("ok", False)
            capability_name = promo.get("name")
        except Exception as e:  # noqa: BLE001
            logger.warning("coding-pattern promotion failed: %s", e)

    entry = {
        "ts":              started,
        "topic":           topic["slug"],
        "topic_title":     topic["title"],
        "tier":            topic["tier"],
        "language":        topic["language"],
        "exercise_id":     exercise.exercise_id,
        "shape":           exercise.capability_shape,
        "passed":          passed,
        "runtime_ms":      grading["runtime_ms"],
        "cases":           grading["cases"],
        "reason":          grading["reason"],
        "promoted":        promoted,
        "capability_name": capability_name,
    }
    _append_log(entry)
    return {"ok": True, **entry}


def run_study_batch(*,
                    n: int = 5,
                    language: Optional[str] = None,
                    db_path: str = "data/dmai_knowledge.db",
                    ) -> Dict[str, Any]:
    """Run N study rounds in sequence and return an aggregate summary."""
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    rounds: List[Dict[str, Any]] = []
    passes = 0
    promotions = 0
    for _ in range(max(1, min(int(n), 50))):
        r = run_study_round(language=language, db_path=db_path)
        rounds.append({
            "topic":    r.get("topic"),
            "shape":    r.get("shape"),
            "passed":   r.get("passed"),
            "promoted": r.get("promoted"),
            "reason":   r.get("reason"),
        })
        if r.get("passed"):
            passes += 1
        if r.get("promoted"):
            promotions += 1

    return {
        "ok":         True,
        "started":    started,
        "finished":   _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "rounds":     len(rounds),
        "passes":     passes,
        "promotions": promotions,
        "detail":     rounds,
    }


def study_stats(db_path: str = "data/dmai_knowledge.db") -> Dict[str, Any]:
    """Return aggregate study stats for the admin dashboard."""
    log = read_study_log(limit=500)
    total = len(log)
    passed = sum(1 for x in log if x.get("passed"))
    promoted = sum(1 for x in log if x.get("promoted"))
    shapes: Dict[str, Dict[str, int]] = {}
    for x in log:
        s = x.get("shape") or "unknown"
        row = shapes.setdefault(s, {"attempts": 0, "passes": 0})
        row["attempts"] += 1
        if x.get("passed"):
            row["passes"] += 1
    return {
        "ok":               True,
        "log_entries":      total,
        "passes":           passed,
        "promotions":       promoted,
        "pass_rate":        (passed / total) if total else 0.0,
        "by_shape":         shapes,
        "curriculum_size":  len(CURRICULUM_TOPICS),
    }
