"""LLM-driven capability materialiser (PR H).

Turns accepted concept stubs into runnable Python modules.

Pipeline for one candidate capability::

    pick candidate  -->  request_code (gpt-4o-mini)
                          |
                          v
                      _validator.validate_source
                          |  ok
                          v
                    write staging/<slug>.py
                    write tests/generated/test_<slug>.py
                          |
                          v
                    _sandbox.run_pytest_file  (30s cap)
                          |
                          v
                    _sandbox.run_happy_path   (5s cap)
                          |
                          v
                    _self_judge_review        (docstring vs concept)
                          |
                          v
                promote  staging -> live
                mark capabilities.runtime_mode = "generated_module"
                record materialisation_log row

Any gate failure retries once via ``request_code`` with the fallback
model (Claude Sonnet 4.5) and the failure reasons attached as
guidance. If retry also fails, the candidate is marked ``failed``
with the last set of reasons.

Daily cap: :data:`DEFAULT_DAILY_CAP` (5). Only picks capabilities
where ``provenance = 'fresh_blood_seed+self_judge'`` and
``judge_confidence >= 0.80``.

External surface mirrors ``seed_capability_promoter``:

* :func:`materialise_once` - single pass, callable from tests.
* :class:`CapabilityMaterialiserLoop` - background thread.
* :func:`start_capability_materialiser_loop` - idempotent bootstrap
  matching the fresh_blood / promoter loops.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from components.generated import _codegen_client as codegen
from components.generated import _sandbox as sandbox
from components.generated import _self_judge_review as reviewer
from components.generated import _smoke_writer as smoke_writer
from components.generated import _validator as validator

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────

DEFAULT_DAILY_CAP           = 5
DEFAULT_MIN_JUDGE_CONFIDENCE = 0.80
POLL_SECONDS                = 300          # every 5 minutes
DEFAULT_DB_PATH             = "data/dmai_knowledge.db"

REPO_ROOT      = Path(__file__).resolve().parents[1]
STAGING_DIR    = REPO_ROOT / "components" / "generated" / "staging"
LIVE_DIR       = REPO_ROOT / "components" / "generated" / "live"
TESTS_DIR      = REPO_ROOT / "tests" / "generated"

STATE_KEY_DAY_BUCKET  = "capability_materialiser:day_bucket"
STATE_KEY_DAY_COUNT   = "capability_materialiser:day_count"
STATE_KEY_LAST_RUN    = "capability_materialiser:last_run_ts"
STATE_KEY_LAST_SUMMARY = "capability_materialiser:last_summary"


# ── Slug helper (mirrors promoter) ────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def _slug(text: str) -> str:
    text = text.replace("×", "-x-").replace("*", "-x-")
    return _SLUG_RE.sub("_", text.strip()).strip("_").lower()[:80] or "unnamed"


# ── SQLite helpers ────────────────────────────────────────────────────────

def _safe_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=15.0)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS materialisation_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          capability_id TEXT,
          concept       TEXT,
          slug          TEXT,
          outcome       TEXT,      -- promoted | failed | rejected_review
          model_used    TEXT,
          reasons       TEXT,      -- JSON list
          judge_confidence REAL,
          duration_sec  REAL,
          created_at    TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_matlog_created
          ON materialisation_log(created_at);
        CREATE INDEX IF NOT EXISTS idx_matlog_capability
          ON materialisation_log(capability_id);

        CREATE TABLE IF NOT EXISTS system_state (
          key TEXT PRIMARY KEY,
          value TEXT,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()


def _state_get(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT value FROM system_state WHERE key = ?", (key,),
    ).fetchone()
    return row[0] if row else None


def _state_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO system_state (key, value, updated_at) "
        "VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
        "updated_at=CURRENT_TIMESTAMP",
        (key, value),
    )


def _today_bucket() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def _day_counter(conn: sqlite3.Connection) -> int:
    bucket = _today_bucket()
    stored = _state_get(conn, STATE_KEY_DAY_BUCKET)
    if stored != bucket:
        _state_set(conn, STATE_KEY_DAY_BUCKET, bucket)
        _state_set(conn, STATE_KEY_DAY_COUNT, "0")
        conn.commit()
        return 0
    try:
        return int(_state_get(conn, STATE_KEY_DAY_COUNT) or "0")
    except ValueError:
        return 0


def _bump_day(conn: sqlite3.Connection, by: int = 1) -> int:
    n = _day_counter(conn) + by
    _state_set(conn, STATE_KEY_DAY_COUNT, str(n))
    conn.commit()
    return n


# ── Candidate selection ───────────────────────────────────────────────────

def _pick_candidates(conn: sqlite3.Connection,
                     *,
                     min_confidence: float,
                     limit: int) -> List[Dict[str, Any]]:
    """Read the capabilities table for stubs eligible for materialisation.

    A capability is eligible when:
    - ``runtime_mode = 'stub'``
    - ``provenance   = 'fresh_blood_seed+self_judge'``
    - ``judge_confidence >= min_confidence``
    - There is no ``materialisation_log`` row for this cap_id with
      ``outcome = 'promoted'`` (already done) or with a failed row
      created in the last 24 hours (backoff).
    """
    # capabilities table shape in the registry-mirror on prod:
    # (id, name, type, capability_type, description, provenance,
    #  judge_confidence, runtime_mode, ...)
    try:
        rows = conn.execute(
            """
            SELECT id, name, capability_type, description,
                   provenance, judge_confidence, runtime_mode
            FROM capabilities
            WHERE runtime_mode = 'stub'
              AND provenance   = 'fresh_blood_seed+self_judge'
              AND (judge_confidence IS NOT NULL
                   AND judge_confidence >= ?)
            LIMIT ?
            """,
            (float(min_confidence), int(limit) * 4),  # over-fetch;
        ).fetchall()
    except sqlite3.OperationalError as e:
        # Column not present in this DB. Return empty and let the
        # caller record the reason.
        logger.info("capability_materialiser: capabilities table not "
                    "materialiser-shaped: %s", e)
        return []

    # Filter out already-promoted and freshly-failed (24h) ids.
    ineligible: set = set()
    log_rows = conn.execute(
        """
        SELECT capability_id, outcome, created_at
        FROM materialisation_log
        WHERE capability_id IN ({})
        """.format(",".join("?" * len(rows)) or "''"),
        tuple(r[0] for r in rows) or ("",),
    ).fetchall() if rows else []
    now = _dt.datetime.now(_dt.timezone.utc)
    for cid, outcome, created in log_rows:
        if outcome == "promoted":
            ineligible.add(cid)
            continue
        try:
            when = _dt.datetime.fromisoformat(created).replace(
                tzinfo=_dt.timezone.utc,
            )
        except ValueError:
            when = now
        if outcome in ("failed", "rejected_review") \
                and (now - when).total_seconds() < 24 * 3600:
            ineligible.add(cid)

    picks: List[Dict[str, Any]] = []
    for r in rows:
        if r[0] in ineligible:
            continue
        picks.append({
            "id":              r[0],
            "name":            r[1],
            "capability_type": r[2],
            "description":     r[3],
            "provenance":      r[4],
            "judge_confidence": r[5],
            "runtime_mode":    r[6],
        })
        if len(picks) >= limit:
            break
    return picks


def _happy_kwargs_for(capability_type: str) -> Dict[str, Any]:
    """Pick a benign default kwargs dict per capability_type.

    The generator gets this in the prompt as "MUST succeed with
    these" so it plans a signature that accepts it.
    """
    return {
        "utility":            {"values": [1, 2, 3, 4]},
        "configuration":      {"config": {}},
        "data_structure":     {"items": []},
        "trading":            {"prices": [100.0, 101.5, 99.75]},
        "blockchain":         {"payload": {}},
        "interface":          {"request": {}},
        "research":           {"query": "test"},
        "integration":        {"payload": {}},
        "composite":          {"a": {}, "b": {}},
        "frontier":           {"seed": 0},
        "diversity_nudge":    {"seed": 0},
        "ai_provider_update": {"release": {"tag": "v0.0.0"}},
        "concept":            {"input": None},
    }.get(str(capability_type or "").lower(), {"input": None})


# ── One-shot materialisation ──────────────────────────────────────────────

@dataclass
class MaterialisationResult:
    capability_id:  str
    slug:           str
    outcome:        str            # promoted | failed | rejected_review
    model_used:     str = ""
    reasons:        List[str] = field(default_factory=list)
    judge_confidence: float = 0.0
    duration_sec:   float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capability_id":   self.capability_id,
            "slug":            self.slug,
            "outcome":         self.outcome,
            "model_used":      self.model_used,
            "reasons":         list(self.reasons),
            "judge_confidence": round(self.judge_confidence, 4),
            "duration_sec":    round(self.duration_sec, 3),
        }


def _materialise_candidate(cap: Dict[str, Any],
                           db_path: str,
                           *,
                           codegen_fn: Callable[..., codegen.CodegenAttempt]
                                 = codegen.request_code,
                           ) -> MaterialisationResult:
    """Run the full pipeline for one candidate."""
    t0 = time.monotonic()
    slug = _slug(str(cap["name"]))
    module_dotted = f"components.generated.staging.{slug}"

    happy_kwargs = _happy_kwargs_for(cap.get("capability_type") or "")
    concept = str(cap.get("name") or "")
    insight = str(cap.get("description") or "")
    cap_type = str(cap.get("capability_type") or "concept")

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    staged_path = STAGING_DIR / f"{slug}.py"

    reasons_all: List[str] = []
    model_used = ""

    def _attempt(model: str, retry_reasons: Optional[List[str]]
                 ) -> Tuple[bool, List[str], codegen.CodegenAttempt]:
        att = codegen_fn(
            concept=concept, insight=insight,
            capability_type=cap_type,
            happy_kwargs=happy_kwargs,
            model=model, retry_reasons=retry_reasons,
        )
        if not att.ok or not att.source:
            return False, [att.reason or "codegen_failed"], att

        # -- AST + policy
        report = validator.validate_source(att.source)
        if not report.ok:
            return False, list(report.reasons), att

        # -- Write staged module + smoke test
        staged_path.write_text(att.source, encoding="utf-8")
        smoke_writer.write_smoke_test(
            tests_dir=TESTS_DIR,
            slug=slug,
            module_dotted=module_dotted,
            happy_kwargs=happy_kwargs,
        )

        # -- Pytest gate
        smoke_result = sandbox.run_pytest_file(
            TESTS_DIR / f"test_{slug}.py",
        )
        if not smoke_result.ok:
            return False, [
                f"pytest_failed: {smoke_result.reason}",
                (smoke_result.stdout or smoke_result.stderr or "")[-500:],
            ], att

        # -- Happy-path gate
        happy = sandbox.run_happy_path(module_dotted, happy_kwargs)
        if not happy.ok:
            return False, [
                f"happy_path_failed: {happy.reason}",
                (happy.stderr or happy.stdout or "")[-500:],
            ], att

        # -- self_judge re-eval on docstring
        review = reviewer.review_generated_module(
            concept=concept,
            channel=str(cap.get("capability_type") or "concept"),
            docstring=report.docstring or "",
            db_path=db_path,
        )
        if not review.ok:
            return False, [
                f"self_judge_review: {review.verdict} "
                f"({review.confidence:.3f}) - {review.reason}",
            ], att

        # -- All green
        return True, [], att

    # Primary attempt
    ok, reasons, att = _attempt(codegen.MODEL_PRIMARY, None)
    model_used = att.model
    if not ok:
        reasons_all.extend(reasons)
        # Retry with fallback + failure hints
        ok2, reasons2, att2 = _attempt(codegen.MODEL_FALLBACK, reasons)
        model_used = att2.model or model_used
        if not ok2:
            reasons_all.extend(reasons2)
            # Clean up any partial staging file/test
            for p in (staged_path,
                      TESTS_DIR / f"test_{slug}.py"):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            return MaterialisationResult(
                capability_id=str(cap["id"]), slug=slug,
                outcome="failed", model_used=model_used,
                reasons=reasons_all,
                duration_sec=time.monotonic() - t0,
            )

    # Promote staging -> live
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    live_path = LIVE_DIR / f"{slug}.py"
    live_path.write_text(staged_path.read_text(encoding="utf-8"),
                         encoding="utf-8")
    # Leave the staging copy in place - the tests import from it.

    return MaterialisationResult(
        capability_id=str(cap["id"]), slug=slug,
        outcome="promoted", model_used=model_used,
        reasons=[], judge_confidence=1.0,
        duration_sec=time.monotonic() - t0,
    )


# ── Public one-pass entry point ───────────────────────────────────────────

def materialise_once(*,
                     db_path: str = DEFAULT_DB_PATH,
                     daily_cap: int = DEFAULT_DAILY_CAP,
                     min_confidence: float = DEFAULT_MIN_JUDGE_CONFIDENCE,
                     codegen_fn: Optional[
                         Callable[..., codegen.CodegenAttempt]] = None,
                     ) -> Dict[str, Any]:
    """Run one pass. Returns a summary dict for the admin endpoint."""
    codegen_fn = codegen_fn or codegen.request_code
    conn = _safe_connect(db_path)
    try:
        _ensure_tables(conn)
        day_count = _day_counter(conn)
        remaining = max(0, daily_cap - day_count)
        if remaining <= 0:
            summary = {
                "picked": 0, "promoted": 0, "failed": 0,
                "cap_hit": True, "day_count": day_count,
                "daily_cap": daily_cap,
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
            _state_set(conn, STATE_KEY_LAST_RUN, summary["ts"])
            _state_set(conn, STATE_KEY_LAST_SUMMARY,
                       json.dumps(summary, default=str))
            conn.commit()
            return summary

        candidates = _pick_candidates(
            conn, min_confidence=min_confidence, limit=remaining,
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass

    picked = len(candidates)
    promoted = 0
    failed = 0
    results: List[MaterialisationResult] = []

    for cap in candidates:
        r = _materialise_candidate(cap, db_path, codegen_fn=codegen_fn)
        results.append(r)
        if r.outcome == "promoted":
            promoted += 1
        else:
            failed += 1

    # Persist log rows + counters
    conn = _safe_connect(db_path)
    try:
        _ensure_tables(conn)
        for r in results:
            conn.execute(
                "INSERT INTO materialisation_log "
                "(capability_id, concept, slug, outcome, model_used, "
                " reasons, judge_confidence, duration_sec) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (r.capability_id,
                 next((c["name"] for c in candidates
                       if c["id"] == r.capability_id), ""),
                 r.slug, r.outcome, r.model_used,
                 json.dumps(r.reasons, default=str)[:4000],
                 r.judge_confidence, r.duration_sec),
            )
            if r.outcome == "promoted":
                _bump_day(conn, 1)
                # Flip runtime_mode in the capabilities table so
                # downstream consumers see the live module.
                try:
                    conn.execute(
                        "UPDATE capabilities SET runtime_mode = ? "
                        "WHERE id = ?",
                        ("generated_module", r.capability_id),
                    )
                except sqlite3.OperationalError:
                    pass  # column absent on the test schema

        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        summary = {
            "picked":    picked,
            "promoted":  promoted,
            "failed":    failed,
            "cap_hit":   False,
            "day_count": _day_counter(conn),
            "daily_cap": daily_cap,
            "results":   [r.as_dict() for r in results],
            "ts":        now,
        }
        _state_set(conn, STATE_KEY_LAST_RUN, now)
        _state_set(conn, STATE_KEY_LAST_SUMMARY,
                   json.dumps(summary, default=str)[:20000])
        conn.commit()
        return summary
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Background loop ───────────────────────────────────────────────────────

class CapabilityMaterialiserLoop:
    """Long-running poll loop that mirrors the promoter / injector."""

    def __init__(self,
                 db_path:        str  = DEFAULT_DB_PATH,
                 daily_cap:      int  = DEFAULT_DAILY_CAP,
                 min_confidence: float = DEFAULT_MIN_JUDGE_CONFIDENCE,
                 poll_seconds:   int  = POLL_SECONDS):
        self._db_path = db_path
        self._daily_cap = int(daily_cap)
        self._min_confidence = float(min_confidence)
        self._poll = int(poll_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_summary: Dict[str, Any] = {}

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.last_summary = materialise_once(
                    db_path=self._db_path,
                    daily_cap=self._daily_cap,
                    min_confidence=self._min_confidence,
                )
            except Exception as e:
                logger.exception(
                    "capability_materialiser: pass crashed: %s", e,
                )
                self.last_summary = {"error": str(e)}
            self._stop.wait(self._poll)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name="capability-materialiser",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


_LOOP: Optional[CapabilityMaterialiserLoop] = None


def start_capability_materialiser_loop(
    *,
    db_path:        str   = DEFAULT_DB_PATH,
    daily_cap:      int   = DEFAULT_DAILY_CAP,
    min_confidence: float = DEFAULT_MIN_JUDGE_CONFIDENCE,
    poll_seconds:   int   = POLL_SECONDS,
) -> CapabilityMaterialiserLoop:
    """Idempotent bootstrap. Rebuilds the loop after a dead-thread fork."""
    global _LOOP
    live = _LOOP is not None and getattr(_LOOP, "_thread", None) is not None \
        and _LOOP._thread.is_alive()
    if live:
        return _LOOP  # type: ignore[return-value]
    loop = CapabilityMaterialiserLoop(
        db_path=db_path, daily_cap=daily_cap,
        min_confidence=min_confidence, poll_seconds=poll_seconds,
    )
    loop.start()
    _LOOP = loop
    return loop


__all__ = [
    "MaterialisationResult",
    "materialise_once",
    "CapabilityMaterialiserLoop",
    "start_capability_materialiser_loop",
    "DEFAULT_DAILY_CAP",
    "DEFAULT_MIN_JUDGE_CONFIDENCE",
    "STATE_KEY_LAST_SUMMARY",
    "STATE_KEY_LAST_RUN",
]
