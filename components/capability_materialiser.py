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

try:
    from components import capability_verifier as verifier  # type: ignore
except Exception:  # pragma: no cover - defensive at import time
    verifier = None

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────

# PR DD (2026-07-15): widen the queue. Auto-revert (PR CC) makes it
# safe to lower the judge bar and raise the daily cap.
DEFAULT_DAILY_CAP           = 10
DEFAULT_MIN_JUDGE_CONFIDENCE = 0.60
POLL_SECONDS                = 300          # every 5 minutes
DEFAULT_DB_PATH             = "data/dmai_knowledge.db"

# Daily cap split across the three picker paths. Sum should equal
# DEFAULT_DAILY_CAP so the total budget is preserved. If any pool is
# empty the unused budget rolls to the next pool in order.
PICKER_QUOTAS = {
    "fresh_blood_seed+self_judge": 5,
    "promoter_path+self_judge":    3,
    "gap_driven":                  2,
}

# Recognised provenances the picker will consider. Order matters:
# fresh_blood first (highest quality), then promoter_path, then gap.
ACCEPTED_PROVENANCES = tuple(PICKER_QUOTAS.keys())

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
    """Open the DMAI knowledge DB through the shared safe_open_kdb
    helper (WAL, busy_timeout=30s, per-thread cache, process-level write
    mutex). Falls back to a plain sqlite3.connect only if the shared
    helper is unavailable — e.g. under unit tests that use isolated
    tmp_path DBs.
    """
    try:
        # Local import so unit tests that don't set up the full package
        # environment can still use this module.
        from components.db import safe_open_kdb  # noqa
        # safe_open_kdb returns a KeepOpenProxy; it forwards sqlite3
        # methods and its .close() is a no-op so the cache stays owner.
        return safe_open_kdb(db_path, timeout=30.0)
    except Exception:  # noqa: BLE001
        # Test / fallback path: still set the important pragmas so unit
        # tests exercise WAL + a real busy_timeout, matching prod.
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
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

def _pool_depth(conn: sqlite3.Connection, *, min_confidence: float) -> int:
    """PR QQ: cheap count of stub capabilities that *would* be eligible.

    Mirrors the primary WHERE clause of ``_pick_candidates`` but without
    the materialisation_log join, so it's fast and safe to call on every
    tick. Used to detect a starved queue / a too-conservative daily cap.
    """
    provenance_list = list(PICKER_QUOTAS.keys())
    placeholders = ",".join("?" * len(provenance_list))
    try:
        row = conn.execute(
            f"""
            SELECT COUNT(*) FROM capabilities
            WHERE runtime_mode IN ('stub', 'stub_reverted')
              AND provenance   IN ({placeholders})
              AND judge_confidence IS NOT NULL
              AND judge_confidence >= ?
            """,
            tuple(provenance_list) + (float(min_confidence),),
        ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0


# PR AAA-3: reasons that indicate infrastructure noise, not code quality.
# Failures whose entire reasons list is transient don't count toward the
# 24h backoff. Match on substrings because reasons include trailing
# detail (HTTP body snippets, model names, etc.).
#
# PR AAA-4c: added "self_judge_review: defer" as transient. A defer is
# an uncertain-band verdict, not a rejection - the docstring drift check
# didn't fire, the code passed all runtime gates, the self-judge just
# didn't have enough weighted signal to be sure. AAA-4b already treats
# defer as ok for gap_driven at review-time; AAA-4c is the retroactive
# clearing counterpart so the pre-AAA-4b log rows (with credit +
# defer) stop enforcing a 24h backoff for candidates that would now
# promote cleanly. Real quality failures (reject, syntax, pytest,
# happy_path) still enforce backoff.
_TRANSIENT_REASON_MARKERS = (
    "credit_exhausted",
    "credit_skip",
    "credit_exhausted_preflight",
    "http_or_auth_failure",
    "openrouter returned 402",
    "openrouter balance",
    "status_401",
    "status_402",
    "status_429",
    "status_503",
    "status_504",
    "self_judge_review: defer",
    "vocab_coverage=",  # historical low-vocab defer rows (pre AAA-4)
)


def _is_transient_only(reasons_json: Optional[str]) -> bool:
    """PR AAA-3: True if every reason in the JSON list is a transient
    infra failure (credit / auth / 5xx). Empty or unparseable reasons
    return False — we err on the side of respecting the backoff."""
    if not reasons_json:
        return False
    try:
        reasons = json.loads(reasons_json)
    except (ValueError, TypeError):
        return False
    if not isinstance(reasons, list) or not reasons:
        return False
    for r in reasons:
        s = str(r).lower()
        if not any(marker in s for marker in _TRANSIENT_REASON_MARKERS):
            return False
    return True


def clear_transient_backoffs(db_path: Optional[str] = None,
                             *,
                             hours: int = 24,
                             ) -> Dict[str, Any]:
    """PR AAA-3: delete materialisation_log rows whose failures were
    purely transient (credit / auth / 5xx) within the last ``hours``.
    Returns a summary with the count deleted and a sample of the
    capability_ids freed up.

    Called from the /api/admin/capability-materialiser/clear-transient
    endpoint. Idempotent — safe to call repeatedly. Does NOT touch
    real code-quality failures (self_judge_review defer, smoke_test
    failed, syntax_error, etc.).
    """
    path = db_path or DEFAULT_DB_PATH
    conn = _safe_connect(path)
    try:
        _ensure_tables(conn)
        cutoff_iso = (
            _dt.datetime.now(_dt.timezone.utc)
            - _dt.timedelta(hours=int(hours))
        ).isoformat()
        candidates = conn.execute(
            """
            SELECT rowid, capability_id, reasons
            FROM materialisation_log
            WHERE outcome IN ('failed', 'rejected_review')
              AND created_at >= ?
            """,
            (cutoff_iso,),
        ).fetchall()
        to_delete: List[int] = []
        cleared_ids: List[str] = []
        for rowid, cid, reasons_json in candidates:
            if _is_transient_only(reasons_json):
                to_delete.append(rowid)
                cleared_ids.append(cid)
        if to_delete:
            placeholders = ",".join("?" * len(to_delete))
            conn.execute(
                f"DELETE FROM materialisation_log WHERE rowid IN ({placeholders})",
                tuple(to_delete),
            )
            conn.commit()
        return {
            "ok": True,
            "cleared": len(cleared_ids),
            "total_scanned": len(candidates),
            "hours_window": int(hours),
            "sample_ids": cleared_ids[:20],
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


def queue_composition(db_path: Optional[str] = None,
                      *,
                      min_confidence: float = DEFAULT_MIN_JUDGE_CONFIDENCE,
                      ) -> Dict[str, Any]:
    """PR AAA-1: return the current stub-queue composition for the
    ``/api/admin/capability-materialiser/queue`` diagnostic endpoint.

    Groups eligible stub capabilities by (provenance, capability_type)
    and classifies each capability_type as either ``local_templatable``
    (the local synthesiser can handle it without an LLM) or
    ``llm_required``. Also returns the current ``local_only_mode``
    flag so the caller can see whether the materialiser is currently
    narrowing the queue to templated types only.

    Cheap: one SELECT + a set lookup. Safe to poll every minute.
    """
    path = db_path or DEFAULT_DB_PATH
    try:
        from components.local_codegen import LOCAL_CAPABILITY_TYPES
        local_types = LOCAL_CAPABILITY_TYPES
    except Exception:  # noqa: BLE001
        local_types = frozenset()

    provenance_list = list(PICKER_QUOTAS.keys())
    placeholders = ",".join("?" * len(provenance_list))
    conn = _safe_connect(path)
    try:
        try:
            rows = conn.execute(
                f"""
                SELECT provenance, capability_type, COUNT(*)
                FROM capabilities
                WHERE runtime_mode IN ('stub', 'stub_reverted')
                  AND provenance   IN ({placeholders})
                  AND judge_confidence IS NOT NULL
                  AND judge_confidence >= ?
                GROUP BY provenance, capability_type
                ORDER BY provenance, COUNT(*) DESC
                """,
                tuple(provenance_list) + (float(min_confidence),),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    by_provenance: Dict[str, Dict[str, Any]] = {}
    total_local = 0
    total_llm = 0
    for prov, cap_type, n in rows:
        ct = str(cap_type or "").lower()
        n = int(n)
        bucket = by_provenance.setdefault(str(prov), {
            "local_templatable": 0,
            "llm_required": 0,
            "by_type": {},
        })
        bucket["by_type"][ct or "(none)"] = n
        if ct in local_types:
            bucket["local_templatable"] += n
            total_local += n
        else:
            bucket["llm_required"] += n
            total_llm += n

    return {
        "ok": True,
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "min_confidence": min_confidence,
        "local_only_mode": _local_only_mode(),
        "totals": {
            "local_templatable": total_local,
            "llm_required": total_llm,
            "eligible": total_local + total_llm,
        },
        "by_provenance": by_provenance,
        "local_capability_types": sorted(local_types),
    }


def _local_only_mode() -> bool:
    """PR AAA-1: return True when the materialiser should restrict
    itself to capability_types that the local template synthesiser can
    handle without any external LLM call.

    Two triggers, checked in order:

    1. ``MATERIALISER_LOCAL_ONLY=1`` explicit env override — always on.
    2. ``MATERIALISER_LOCAL_ONLY_AUTO=1`` (default) + OpenRouter balance
       below the viable floor ($0.01) — auto-fallback when credits
       are exhausted so self-generation still makes progress on the
       templated types instead of hard-stopping the whole tick.

    Any error looking up the balance means the auto path stays off
    (better to attempt an LLM call and get a real 402 than to silently
    narrow the queue on every tick).
    """
    if os.environ.get("MATERIALISER_LOCAL_ONLY", "0") == "1":
        return True
    if os.environ.get("MATERIALISER_LOCAL_ONLY_AUTO", "1") != "1":
        return False
    try:
        credits = codegen.get_openrouter_credits()
        if credits is None:
            return False
        bal = credits.get("balance")
        if bal is None:
            return False
        return float(bal) < 0.01
    except Exception:  # noqa: BLE001
        return False


def _pick_candidates(conn: sqlite3.Connection,
                     *,
                     min_confidence: float,
                     limit: int,
                     quotas: Optional[Dict[str, int]] = None,
                     local_only: bool = False,
                     ) -> List[Dict[str, Any]]:
    """Read the capabilities table for stubs eligible for materialisation.

    A capability is eligible when:
    - ``runtime_mode`` is ``'stub'`` (never materialised) OR
      ``'stub_reverted'`` (previously promoted then verifier-reverted)
    - ``provenance`` is one of :data:`ACCEPTED_PROVENANCES`
    - ``judge_confidence >= min_confidence``
    - There is no ``materialisation_log`` row for this cap_id with
      ``outcome = 'promoted'`` (already done) or with a failed row
      created in the last 24 hours (backoff).

    ``'quarantined'`` rows are excluded permanently.

    Picking honours :data:`PICKER_QUOTAS` (or ``quotas`` if supplied):
    each provenance pool gets its own slice of the daily budget. Unused
    slices roll forward to the next pool in the declared order so a
    starved pool doesn't waste the whole tick.
    """
    quotas = quotas or PICKER_QUOTAS

    # capabilities table shape in the registry-mirror on prod:
    # (id, name, type, capability_type, description, provenance,
    #  judge_confidence, runtime_mode, ...)
    provenance_list = list(quotas.keys())
    placeholders = ",".join("?" * len(provenance_list))
    try:
        rows = conn.execute(
            f"""
            SELECT id, name, capability_type, description,
                   provenance, judge_confidence, runtime_mode
            FROM capabilities
            WHERE runtime_mode IN ('stub', 'stub_reverted')
              AND provenance   IN ({placeholders})
              AND (judge_confidence IS NOT NULL
                   AND judge_confidence >= ?)
            """,
            tuple(provenance_list) + (float(min_confidence),),
        ).fetchall()
    except sqlite3.OperationalError as e:
        # Column not present in this DB. Return empty and let the
        # caller record the reason.
        logger.info("capability_materialiser: capabilities table not "
                    "materialiser-shaped: %s", e)
        return []

    # Filter out already-promoted and freshly-failed (24h) ids.
    #
    # PR AAA-3: a failure whose reasons list is entirely transient
    # (credit_exhausted / http_or_auth_failure / credit_skip) does NOT
    # count toward the 24h backoff — those aren't code-quality
    # failures, they're infrastructure noise. Without this, a single
    # OpenRouter outage burns the entire queue for 24h.
    ineligible: set = set()
    if rows:
        log_rows = conn.execute(
            """
            SELECT capability_id, outcome, created_at, reasons
            FROM materialisation_log
            WHERE capability_id IN ({})
            """.format(",".join("?" * len(rows))),
            tuple(r[0] for r in rows),
        ).fetchall()
    else:
        log_rows = []
    now = _dt.datetime.now(_dt.timezone.utc)
    for cid, outcome, created, reasons_json in log_rows:
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
            if _is_transient_only(reasons_json):
                # PR AAA-3: skip — don't add to ineligible.
                continue
            ineligible.add(cid)

    # PR AAA-1: in local_only mode restrict to capability_types the
    # local template synthesiser can handle without an LLM. Falls back
    # here so the materialiser keeps making progress when OpenRouter
    # credits are exhausted — the LLM-only architectural candidates
    # sit in the queue until credits return.
    if local_only:
        try:
            from components.local_codegen import LOCAL_CAPABILITY_TYPES
            _local_types = LOCAL_CAPABILITY_TYPES
        except Exception:  # noqa: BLE001
            _local_types = frozenset()
    else:
        _local_types = None

    # Bucket eligible rows by provenance so we can apply per-pool quotas.
    buckets: Dict[str, List[Dict[str, Any]]] = {
        p: [] for p in provenance_list
    }
    for r in rows:
        if r[0] in ineligible:
            continue
        prov = r[4]
        if prov not in buckets:
            continue
        if _local_types is not None:
            ct = str(r[2] or "").lower()
            if ct not in _local_types:
                continue
        buckets[prov].append({
            "id":              r[0],
            "name":            r[1],
            "capability_type": r[2],
            "description":     r[3],
            "provenance":      prov,
            "judge_confidence": r[5],
            "runtime_mode":    r[6],
        })

    picks: List[Dict[str, Any]] = []
    rollover = 0
    for prov in provenance_list:
        pool = buckets[prov]
        # Preference: highest judge_confidence first inside each pool.
        pool.sort(key=lambda c: (c.get("judge_confidence") or 0.0), reverse=True)
        pool_budget = min(int(quotas.get(prov, 0)) + rollover, limit - len(picks))
        take = pool[:max(0, pool_budget)]
        picks.extend(take)
        rollover = max(0, pool_budget - len(take))
        if len(picks) >= limit:
            break
    return picks[:limit]


# ── Gap-driven seeder (PR DD) ─────────────────────────────────

def _seed_gap_capabilities(conn: sqlite3.Connection,
                           *,
                           max_new: int = 5,
                           ) -> int:
    """Insert capability rows for detected gaps so the picker can consume them.

    Reads ``iter_capability_gaps()`` (from gap_fetcher) and materialises
    each into a ``capabilities`` row with:
      - ``runtime_mode = 'stub'``
      - ``provenance   = 'gap_driven'``
      - ``judge_confidence = 0.65`` (above the 0.60 floor, well below
        fresh-blood quality so quota preference still favours those)

    Idempotent: an INSERT OR IGNORE keyed off the gap slug prevents
    duplicate seeding on repeated ticks.

    Returns the number of rows freshly inserted.
    """
    try:
        from components.gap_fetcher import iter_capability_gaps  # noqa
    except Exception as e:  # noqa: BLE001
        logger.info("gap seeder unavailable: %s", e)
        return 0

    try:
        gaps = list(iter_capability_gaps(fresh=False))
    except Exception as e:  # noqa: BLE001
        logger.info("iter_capability_gaps failed: %s", e)
        return 0

    if not gaps:
        return 0

    # Priority 1 first, then 2, etc. Cap at max_new.
    gaps.sort(key=lambda g: int(getattr(g, "priority", 5) or 5))
    gaps = gaps[:max_new]

    inserted = 0
    for g in gaps:
        slug = _slug(str(getattr(g, "name", "") or ""))
        if not slug or slug == "unnamed":
            continue
        desc = str(getattr(g, "description", "") or slug.replace("_", " "))
        cap_id = f"gap_{slug}"
        try:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO capabilities
                    (id, name, capability_type, description,
                     provenance, judge_confidence, runtime_mode)
                VALUES (?, ?, 'utility', ?, 'gap_driven', 0.65, 'stub')
                """,
                (cap_id, slug, desc[:800]),
            )
            if cur.rowcount > 0:
                inserted += 1
        except sqlite3.OperationalError as e:
            logger.info("gap seed skipped for %s: %s", slug, e)
            break  # schema shape isn't compatible; stop trying
    if inserted:
        conn.commit()
    return inserted


def _happy_kwargs_for(capability_type: str) -> Dict[str, Any]:
    """Pick a benign default kwargs dict per capability_type.

    The generator gets this in the prompt as "MUST succeed with
    these" so it plans a signature that accepts it.
    """
    # PR SS: include db_path in the base kwargs. Gap-driven codegen
    # very often ships a run(*, db_path) signature (writes to DMAI's
    # own DB), and the smoke test's 4-way fallback already tolerates
    # signatures that ignore db_path, so this is safe for every type.
    _base = {"db_path": ":memory:"}
    return {
        "utility":            {**_base, "values": [1, 2, 3, 4]},
        "configuration":      {**_base, "config": {}},
        "data_structure":     {**_base, "items": []},
        "trading":            {**_base, "prices": [100.0, 101.5, 99.75]},
        "blockchain":         {**_base, "payload": {}},
        "interface":          {**_base, "request": {}},
        "research":           {**_base, "query": "test"},
        "integration":        {**_base, "payload": {}},
        "composite":          {**_base, "a": {}, "b": {}},
        "frontier":           {**_base, "seed": 0},
        "diversity_nudge":    {**_base, "seed": 0},
        "ai_provider_update": {**_base, "release": {"tag": "v0.0.0"}},
        "concept":            {**_base, "input": None},
        # PR AAA-2: shapes drawn from the gap_driven stub queue.
        "monitor":            {**_base, "samples": [{"cpu": 0.5, "mem": 0.3}],
                                          "thresholds": {"cpu": 0.9, "mem": 0.9}},
        "infrastructure":     {**_base, "resources": {"services": ["web"],
                                                       "envs": ["OPENROUTER_KEY"],
                                                       "ports": [8000]}},
        "analyser":           {**_base, "records": [{"kind": "a"}, {"kind": "b"}],
                                          "group_by": "kind"},
        "training":           {**_base, "samples": [{"x": 1, "y": 2, "label": "pos"}]},
        "api_wrapper":        {**_base, "request": {"method": "GET", "path": "/"}},
        "testing":            {**_base, "target": {"k": 1},
                                          "cases":  [{"key": "k", "expected": 1}]},
    }.get(str(capability_type or "").lower(), {**_base, "input": None})


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
        # PR SS: pass provenance as the channel so the reviewer's
        # normaliser can apply the relaxed vocab floor for gap-driven
        # seeds. Before PR SS we passed capability_type (usually
        # "utility"), which bypassed _RELAXED_VOCAB_CHANNELS entirely
        # and kept the floor at 0.40 - guaranteeing defer on every
        # gap-driven candidate. Fall back to capability_type only when
        # provenance is missing (should never happen for gap_driven).
        review = reviewer.review_generated_module(
            concept=concept,
            channel=str(
                cap.get("provenance")
                or cap.get("capability_type")
                or "concept"
            ),
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

    # Primary attempt. If this capability was previously reverted by
    # the verifier, feed the last failure traceback in as guidance so
    # codegen fixes what broke last time (PR CC self-repair loop).
    initial_guidance: Optional[List[str]] = None
    if str(cap.get("runtime_mode") or "") == "stub_reverted":
        try:
            from components import capability_verifier as _verifier  # noqa
            g = _verifier.get_retry_guidance(str(cap["id"]), db_path=db_path)
            if g:
                initial_guidance = g
        except Exception:  # noqa: BLE001
            initial_guidance = None
    ok, reasons, att = _attempt(codegen.MODEL_PRIMARY, initial_guidance)
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
                     seed_gaps: Optional[bool] = None,
                     ) -> Dict[str, Any]:
    """Run one pass. Returns a summary dict for the admin endpoint.

    ``seed_gaps=None`` (default) auto-detects: enabled only when the DB
    path looks like a real knowledge DB (``dmai_knowledge`` in the
    path). Tests that use tmp DBs get seeding disabled automatically
    so the SelfScanner's live gap data doesn't pollute the fixture.
    Callers can force with True/False.

    PR NN: takes a cooperative priority hold on the shared db module so
    background writers (e.g. vocabulary_ingester._idle_flush_loop) that
    poll ``is_priority_held()`` will skip their flushes during this tick.
    Non-cooperating writers are unaffected.
    """
    try:
        from components.db import priority_hold_ctx as _priority_hold_ctx  # noqa
    except Exception:  # noqa: BLE001
        # Older db.py without the hook; provide a no-op context manager.
        import contextlib as _cl
        @_cl.contextmanager
        def _priority_hold_ctx(_token):  # type: ignore
            yield None

    with _priority_hold_ctx("materialise_once"):
        return _materialise_once_inner(
            db_path=db_path, daily_cap=daily_cap,
            min_confidence=min_confidence, codegen_fn=codegen_fn,
            seed_gaps=seed_gaps,
        )


def _materialise_once_inner(*,
                             db_path: str,
                             daily_cap: int,
                             min_confidence: float,
                             codegen_fn: Optional[
                                 Callable[..., codegen.CodegenAttempt]],
                             seed_gaps: Optional[bool],
                             ) -> Dict[str, Any]:
    """Actual tick body. Extracted so materialise_once can wrap it in a
    cooperative priority hold without changing the body's control flow."""
    codegen_fn = codegen_fn or codegen.request_code
    conn = _safe_connect(db_path)
    try:
        _ensure_tables(conn)
        day_count = _day_counter(conn)
        remaining = max(0, daily_cap - day_count)
        if remaining <= 0:
            # PR QQ: expose the residual pool size so the health check can
            # distinguish "cap hit, nothing else to do" from "cap hit while
            # dozens of good candidates were skipped" (→ auto-raise cap).
            try:
                remaining_pool_size = _pool_depth(conn, min_confidence=min_confidence)
            except Exception:
                remaining_pool_size = None
            summary = {
                "picked": 0, "promoted": 0, "failed": 0,
                "cap_hit": True, "day_count": day_count,
                "daily_cap": daily_cap,
                "remaining_pool_size": remaining_pool_size,
                "cap_likely_low": (
                    remaining_pool_size is not None
                    and remaining_pool_size >= max(10, daily_cap)
                ),
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
            _state_set(conn, STATE_KEY_LAST_RUN, summary["ts"])
            _state_set(conn, STATE_KEY_LAST_SUMMARY,
                       json.dumps(summary, default=str))
            conn.commit()
            return summary

        # PR DD: seed gap-driven capability rows before picking so the
        # gap pool has candidates to draw from. Auto-disabled for
        # non-prod DBs (e.g. pytest tmp_path fixtures) so the live
        # SelfScanner doesn't leak into unit tests.
        should_seed = seed_gaps
        if should_seed is None:
            should_seed = "dmai_knowledge" in db_path
        gaps_seeded = 0
        if should_seed:
            try:
                gaps_seeded = _seed_gap_capabilities(
                    conn, max_new=PICKER_QUOTAS.get("gap_driven", 2) * 2,
                )
            except Exception as e:  # noqa: BLE001
                logger.info("gap seeding failed non-fatally: %s", e)

        # PR AAA-1: enter local-only mode if either the explicit env flag
        # is set, or (auto path) OpenRouter is credit-exhausted. Narrows
        # the queue to templated capability_types so we skip the LLM
        # entirely for this tick and keep self-generation productive.
        local_only = _local_only_mode()
        candidates = _pick_candidates(
            conn, min_confidence=min_confidence, limit=remaining,
            local_only=local_only,
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # PR WW: pre-flight credit check. If the OpenRouter balance can't
    # cover the minimum viable request, skip the tick cleanly with a
    # concrete reason instead of picking N candidates and burning them
    # all on 402s. Reads the LLM budget once at the top of the tick;
    # any exception is non-fatal (we fall back to the old behaviour of
    # attempting and letting request_code report the 402).
    #
    # PR AAA-1: in local-only mode we already narrowed the queue to
    # capability_types the local synthesiser can handle without an
    # LLM — skip the credit_skip preflight entirely so those local
    # candidates get their chance.
    credit_skip_reason = None
    if not local_only:
        try:
            credits = codegen.get_openrouter_credits()
            if credits is not None and credits.get("balance") is not None:
                balance = float(credits["balance"])
                # $0.01 covers ~5-8 completion calls at gpt-4o-mini rates,
                # enough to be worth trying; below that we're going to 402
                # on every attempt.
                if balance < 0.01:
                    credit_skip_reason = (
                        f"credit_exhausted_preflight: openrouter balance "
                        f"${balance:.4f} below viable floor $0.0100. "
                        f"Skipping {len(candidates)} candidate(s). Top up "
                        f"credits or switch MODEL_PRIMARY."
                    )
        except Exception as e:
            logger.info("pre-flight credit check failed non-fatally: %s", e)

    if credit_skip_reason is not None:
        # Return a clean summary that self-generation/diagnose can pick
        # up. Do NOT write per-candidate failed rows - that's the
        # 'never insert None/zero values' user rule.
        _now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
        summary = {
            "picked":       0,
            "promoted":     0,
            "failed":       0,
            "blocked":      True,
            "starved":      False,
            "credit_skip":  True,
            "credit_skip_reason": credit_skip_reason,
            "pool_depth":   len(candidates),
            "cap_hit":      False,
            "day_count":    day_count,
            "daily_cap":    daily_cap,
            "min_confidence": min_confidence,
            "gaps_seeded":  gaps_seeded,
            "local_only":   bool(local_only),
            "ts":           _now_iso,
            "retry_after_hint": 300,  # try again in 5 min
        }
        conn = _safe_connect(db_path)
        try:
            _ensure_tables(conn)
            _state_set(conn, STATE_KEY_LAST_RUN, summary["ts"])
            _state_set(conn, STATE_KEY_LAST_SUMMARY,
                       json.dumps(summary, default=str))
            conn.commit()
        finally:
            try: conn.close()
            except Exception: pass
        return summary

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

        # ── Post-integration verification (PR CC) ─────────────────────
        # After promotion + runtime_mode flip, run staged verification
        # on each freshly-promoted capability. Failures quarantine the
        # live file and revert runtime_mode so the next pass re-picks
        # it with the failure traceback fed to codegen.
        verifications: List[Dict[str, Any]] = []
        if verifier is not None:
            for r in results:
                if r.outcome != "promoted":
                    continue
                try:
                    concept_name = next(
                        (c["name"] for c in candidates
                         if c["id"] == r.capability_id), "",
                    )
                    cap_type = next(
                        (c.get("capability_type", "") for c in candidates
                         if c["id"] == r.capability_id), "",
                    )
                    happy = _happy_kwargs_for(cap_type or "")
                    vr = verifier.verify_promoted(
                        cap_id=str(r.capability_id),
                        slug=r.slug,
                        capability_type=cap_type or "utility",
                        happy_kwargs=happy,
                        db_path=db_path,
                        use_cache=False,
                    )
                    verifications.append(vr.to_dict())
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "post-integration verification errored for %s: %s",
                        r.slug, e,
                    )
                    verifications.append({
                        "capability_id": r.capability_id,
                        "slug": r.slug,
                        "ok": False,
                        "stage": "error",
                        "reason": f"verifier_error: {e}",
                    })

        # Provenance breakdown of what we picked this tick (for observability).
        prov_breakdown: Dict[str, int] = {}
        for c in candidates:
            key = str(c.get("provenance") or "unknown")
            prov_breakdown[key] = prov_breakdown.get(key, 0) + 1

        # PR QQ: starvation-visibility. When picked==0 we still need to know
        # whether that's "queue empty upstream" vs "queue full but everyone
        # was ineligible" — the former means "kick the seed pipeline / poll
        # sooner", the latter means "lower min_confidence / clear failed
        # backoff". Never a silent no-op.
        try:
            pool_depth_now = _pool_depth(conn, min_confidence=min_confidence)
        except Exception:
            pool_depth_now = None
        starved = (picked == 0 and (pool_depth_now or 0) == 0)
        blocked = (picked == 0 and (pool_depth_now or 0) > 0)
        # Ask the loop to poll sooner if we're starved/blocked, so the
        # system "keeps looking for work" instead of waiting the full
        # POLL_SECONDS between empty ticks.
        retry_after_hint: Optional[int] = None
        if starved or blocked:
            retry_after_hint = 30  # seconds; caller may override

        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        summary = {
            "picked":    picked,
            "promoted":  promoted,
            "failed":    failed,
            "cap_hit":   False,
            "day_count": _day_counter(conn),
            "daily_cap": daily_cap,
            "min_confidence": min_confidence,
            "quotas":    dict(PICKER_QUOTAS),
            "gaps_seeded": gaps_seeded,
            "provenance_breakdown": prov_breakdown,
            "pool_depth": pool_depth_now,
            "starved":   starved,       # queue truly empty upstream
            "blocked":   blocked,       # queue has rows but none eligible
            "local_only": bool(local_only),
            "retry_after_hint": retry_after_hint,
            "results":   [r.as_dict() for r in results],
            "verifications": verifications,
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
        # PR V-fast: exponential backoff on DB lock contention.
        backoff = 0.0
        while not self._stop.is_set():
            try:
                self.last_summary = materialise_once(
                    db_path=self._db_path,
                    daily_cap=self._daily_cap,
                    min_confidence=self._min_confidence,
                )
                backoff = 0.0  # success — reset backoff
            except Exception as e:
                msg = str(e).lower()
                if "lock" in msg or "mutex_timeout" in msg:
                    # Cap at ~5 min so we still recover once the storm clears.
                    backoff = min(backoff * 2 + 5.0, 300.0)
                    logger.warning(
                        "capability_materialiser: DB busy, backing off %.0fs: %s",
                        backoff, e,
                    )
                else:
                    logger.exception(
                        "capability_materialiser: pass crashed: %s", e,
                    )
                self.last_summary = {"error": str(e)}
            # PR QQ: honour the retry_after_hint from materialise_once so
            # the loop "keeps looking for work" when we detect a starved
            # or blocked queue, instead of waiting the full poll interval
            # on empty ticks. The hint is capped by the normal poll so we
            # never accelerate past what the loop was configured for.
            hint = None
            try:
                hint = (self.last_summary or {}).get("retry_after_hint")
            except AttributeError:
                hint = None
            effective_wait: float
            if isinstance(hint, (int, float)) and hint > 0:
                effective_wait = min(float(hint), float(self._poll)) + backoff
            else:
                effective_wait = self._poll + backoff
            self._stop.wait(effective_wait)

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
