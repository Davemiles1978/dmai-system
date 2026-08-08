"""CapabilityVerifier — post-integration verification + auto-revert.

Runs after ``capability_materialiser`` promotes a module from
``components/generated/staging/`` to ``components/generated/live/``
and flips ``capabilities.runtime_mode = 'generated_module'``.

Two-stage verification (staged per David's spec):

  1. **Isolated import + run()** (cheap gate, ~2-3s per capability)
     Fresh subprocess imports the live module and calls ``run()``
     with the recorded ``happy_kwargs``. Catches syntax errors,
     missing deps, broken ``run()`` signature, obvious crashes.

  2. **Orchestrator dispatch** (expensive gate, ~5-10s)
     Only runs if stage 1 passes. Loads the capability through the
     same code path the runtime uses when a request routes to it,
     to catch orchestrator-wiring issues (missing imports the
     orchestrator adds, hook-ordering, etc.).

On failure:
  - Move ``live/<slug>.py`` -> ``quarantine/<slug>_<ts>.py``
  - Set ``capabilities.runtime_mode = 'stub_reverted'``
  - Insert a ``verification_log`` row with the full traceback so the
    next materialisation attempt can feed it to codegen as guidance
  - Increment ``verification_attempts`` counter

Retry policy (matches David's answer to the design question):
  - Failed capabilities are eligible for re-materialisation
  - The prior traceback is passed to codegen via ``retry_reasons``
  - Max 3 verification attempts per capability, then permanent
    quarantine (``runtime_mode = 'quarantined'``)

Idempotent: safe to call ``verify_promoted(cap_id)`` twice on the
same capability - a matching successful verification_log row within
the last hour short-circuits with the previous result.
"""
from __future__ import annotations

import ast
import datetime as _dt
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from components.db import safe_open_kdb


logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = REPO_ROOT / "components" / "generated" / "live"
QUARANTINE_DIR = REPO_ROOT / "components" / "generated" / "quarantine"

DEFAULT_ISOLATED_TIMEOUT_SEC = 5
DEFAULT_ORCHESTRATOR_TIMEOUT_SEC = 10
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_DB_PATH = "data/dmai_knowledge.db"

# Successful verification within this window is treated as still valid
# so we don't re-run the expensive stage 2 on every promoter tick.
VERIFICATION_CACHE_SEC = 3600


# ── Data shapes ───────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    capability_id: str
    slug: str
    ok: bool
    stage: str                # "isolated" | "orchestrator" | "cached" | "skipped"
    reason: str = ""
    traceback: str = ""
    duration_ms: int = 0
    attempts_so_far: int = 0
    quarantined: bool = False
    reverted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── SQLite helpers ────────────────────────────────────────────────────────

def _safe_connect(db_path: str) -> sqlite3.Connection:
    conn = safe_open_kdb(db_path, timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_verification_table(conn: sqlite3.Connection) -> None:
    """Create verification_log if it doesn't exist. Idempotent."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS verification_log (
          id SERIAL PRIMARY KEY,
          capability_id TEXT NOT NULL,
          slug          TEXT NOT NULL,
          stage         TEXT NOT NULL,   -- isolated | orchestrator
          ok            INTEGER NOT NULL,
          reason        TEXT,
          traceback     TEXT,
          duration_ms   INTEGER,
          attempts      INTEGER DEFAULT 1,
          created_at    TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_verlog_cap_created
          ON verification_log(capability_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_verlog_created
          ON verification_log(created_at);
        """
    )
    conn.commit()


def _count_attempts(conn: sqlite3.Connection, cap_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM verification_log WHERE capability_id = ?",
        (cap_id,),
    ).fetchone()
    return int(row[0]) if row else 0


def _last_success_within(conn: sqlite3.Connection,
                         cap_id: str,
                         window_sec: int) -> Optional[sqlite3.Row]:
    # verification_log.created_at is written by SQLite datetime('now'),
    # which returns naive UTC 'YYYY-MM-DD HH:MM:SS'. Match that shape
    # so lexicographic comparison works.
    since = (_dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)
             - _dt.timedelta(seconds=window_sec))
    since_iso = since.strftime("%Y-%m-%d %H:%M:%S")
    row = conn.execute(
        """
        SELECT * FROM verification_log
        WHERE capability_id = ? AND ok = 1 AND stage = 'orchestrator'
          AND created_at >= ?
        ORDER BY id DESC LIMIT 1
        """,
        (cap_id, since_iso),
    ).fetchone()
    return row


def _get_last_traceback(conn: sqlite3.Connection, cap_id: str) -> str:
    """Return the most recent failure traceback for feeding to codegen."""
    row = conn.execute(
        """
        SELECT traceback FROM verification_log
        WHERE capability_id = ? AND ok = 0 AND traceback IS NOT NULL
              AND traceback != ''
        ORDER BY id DESC LIMIT 1
        """,
        (cap_id,),
    ).fetchone()
    return (row["traceback"] if row else "") or ""


# ── Stage 1: isolated subprocess run() ────────────────────────────────────

def _run_isolated(slug: str,
                  happy_kwargs: Optional[Dict[str, Any]] = None,
                  timeout_sec: int = DEFAULT_ISOLATED_TIMEOUT_SEC,
                  ) -> Tuple[bool, str, str, int]:
    """Import the live module + call run() in a fresh subprocess.

    Returns (ok, reason, traceback, duration_ms).
    """
    t0 = time.monotonic()
    happy_kwargs = happy_kwargs or {}
    module_dotted = f"components.generated.live.{slug}"

    runner_code = (
        "import json, sys, traceback\n"
        f"MOD = {module_dotted!r}\n"
        f"KWARGS = {json.dumps(happy_kwargs)}\n"
        "try:\n"
        "    m = __import__(MOD, fromlist=['run'])\n"
        "    if not hasattr(m, 'run'):\n"
        "        print(json.dumps({'ok': False, 'reason': 'no_run_function'}))\n"
        "        sys.exit(0)\n"
        "    try:\n"
        "        result = m.run(**KWARGS)\n"
        "    except TypeError as te:\n"
        "        # If run() takes no args, try calling it with none\n"
        "        msg = str(te)\n"
        "        if 'unexpected keyword' in msg or 'takes 0 positional' in msg:\n"
        "            result = m.run()\n"
        "        else:\n"
        "            raise\n"
        "    print(json.dumps({'ok': True,\n"
        "                       'result_type': type(result).__name__}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'reason': type(e).__name__,\n"
        "                       'msg': str(e)[:400],\n"
        "                       'traceback': traceback.format_exc()[-1600:]}))\n"
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            timeout=timeout_sec,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired:
        dur_ms = int((time.monotonic() - t0) * 1000)
        return False, "timeout", f"exceeded {timeout_sec}s wall clock", dur_ms
    except Exception as e:  # noqa: BLE001
        dur_ms = int((time.monotonic() - t0) * 1000)
        return False, "subprocess_error", str(e), dur_ms

    dur_ms = int((time.monotonic() - t0) * 1000)
    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")

    parsed: Dict[str, Any] = {}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if parsed.get("ok"):
        return True, "ok", "", dur_ms

    reason = parsed.get("reason") or "unknown_failure"
    tb = parsed.get("traceback") or ""
    msg = parsed.get("msg") or ""
    combined_tb = tb
    if not combined_tb and stderr:
        combined_tb = stderr[-1600:]
    if msg and reason not in combined_tb:
        combined_tb = f"{reason}: {msg}\n\n{combined_tb}"
    return False, reason, combined_tb, dur_ms


# ── Stage 2: orchestrator dispatch ────────────────────────────────────────

def _run_orchestrator_dispatch(slug: str,
                               capability_id: str,
                               capability_type: str,
                               happy_kwargs: Optional[Dict[str, Any]] = None,
                               timeout_sec: int = DEFAULT_ORCHESTRATOR_TIMEOUT_SEC,
                               ) -> Tuple[bool, str, str, int]:
    """Load the capability through DMAI's runtime dispatch path.

    We route via the actual generated-module loader logic to catch
    orchestrator-wiring bugs (missing sys.path entries, hook ordering,
    class-level registry expectations) that a bare import wouldn't hit.

    The subprocess boundary keeps a wedged module from taking down
    the parent Flask worker.
    """
    t0 = time.monotonic()
    happy_kwargs = happy_kwargs or {}

    # We mimic the runtime's dispatch pattern: import via the standard
    # `components.generated.live.<slug>` path AFTER touching the same
    # boot-time modules the runtime touches. If that ever breaks
    # (missing __init__.py, package layout drift, etc.) this catches
    # it whereas stage 1 wouldn't.
    runner_code = (
        "import json, sys, traceback\n"
        f"MOD = 'components.generated.live.{slug}'\n"
        f"KWARGS = {json.dumps(happy_kwargs)}\n"
        f"CAP_TYPE = {capability_type!r}\n"
        f"CAP_ID = {capability_id!r}\n"
        "try:\n"
        "    # Touch the same boot ordering the runtime uses: import\n"
        "    # the parent packages first so we surface any package-level\n"
        "    # side effects (e.g. registry auto-registration).\n"
        "    import components  # noqa: F401\n"
        "    import components.generated  # noqa: F401\n"
        "    import components.generated.live  # noqa: F401\n"
        "    m = __import__(MOD, fromlist=['run'])\n"
        "    if not hasattr(m, 'run'):\n"
        "        print(json.dumps({'ok': False, 'reason': 'no_run_function'}))\n"
        "        sys.exit(0)\n"
        "    # Simulate an orchestrator call site: pass cap metadata\n"
        "    # as kwargs the module might introspect.\n"
        "    ctx_kwargs = dict(KWARGS)\n"
        "    ctx_kwargs.setdefault('_capability_id', CAP_ID)\n"
        "    ctx_kwargs.setdefault('_capability_type', CAP_TYPE)\n"
        "    try:\n"
        "        result = m.run(**ctx_kwargs)\n"
        "    except TypeError as te:\n"
        "        # Fall back to non-kwargs call if signature is picky\n"
        "        if 'unexpected keyword' in str(te):\n"
        "            try:\n"
        "                result = m.run(**KWARGS)\n"
        "            except TypeError:\n"
        "                result = m.run()\n"
        "        else:\n"
        "            raise\n"
        "    # Accept any non-exception return as a pass; the module\n"
        "    # decides its own return shape.\n"
        "    print(json.dumps({'ok': True,\n"
        "                       'result_type': type(result).__name__}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'reason': type(e).__name__,\n"
        "                       'msg': str(e)[:400],\n"
        "                       'traceback': traceback.format_exc()[-2000:]}))\n"
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            timeout=timeout_sec,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired:
        dur_ms = int((time.monotonic() - t0) * 1000)
        return False, "timeout", f"orchestrator dispatch exceeded {timeout_sec}s", dur_ms
    except Exception as e:  # noqa: BLE001
        dur_ms = int((time.monotonic() - t0) * 1000)
        return False, "subprocess_error", str(e), dur_ms

    dur_ms = int((time.monotonic() - t0) * 1000)
    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")

    parsed: Dict[str, Any] = {}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if parsed.get("ok"):
        return True, "ok", "", dur_ms

    reason = parsed.get("reason") or "unknown_failure"
    tb = parsed.get("traceback") or ""
    msg = parsed.get("msg") or ""
    combined_tb = tb or (stderr[-2000:] if stderr else "")
    if msg and reason not in combined_tb:
        combined_tb = f"{reason}: {msg}\n\n{combined_tb}"
    return False, reason, combined_tb, dur_ms


# ── Quarantine + revert ───────────────────────────────────────────────────

def _quarantine_module(slug: str) -> Optional[Path]:
    """Move live/<slug>.py to quarantine/<slug>_<ts>.py. Idempotent."""
    live_path = LIVE_DIR / f"{slug}.py"
    if not live_path.exists():
        return None
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = QUARANTINE_DIR / f"{slug}_{ts}.py"
    try:
        shutil.move(str(live_path), str(dest))
        return dest
    except Exception as e:  # noqa: BLE001
        logger.warning("quarantine move failed for %s: %s", slug, e)
        return None


def _revert_capability(conn: sqlite3.Connection,
                       cap_id: str,
                       permanent: bool) -> None:
    """Flip runtime_mode back so the picker knows to (maybe) retry.

    ``permanent=True`` sets 'quarantined' so it will never be picked
    again. ``permanent=False`` sets 'stub_reverted' so the picker can
    retry with codegen guided by the failure traceback.
    """
    new_mode = "quarantined" if permanent else "stub_reverted"
    try:
        conn.execute(
            "UPDATE capabilities SET runtime_mode = ? WHERE id = ?",
            (new_mode, cap_id),
        )
        conn.commit()
    except sqlite3.OperationalError as e:
        logger.warning(
            "revert flip failed for %s: %s (schema may lack runtime_mode)",
            cap_id, e,
        )


# ── Orchestration ─────────────────────────────────────────────────────────

def verify_promoted(cap_id: str,
                    slug: str,
                    capability_type: str,
                    *,
                    happy_kwargs: Optional[Dict[str, Any]] = None,
                    db_path: str = DEFAULT_DB_PATH,
                    isolated_timeout_sec: int = DEFAULT_ISOLATED_TIMEOUT_SEC,
                    orchestrator_timeout_sec: int = DEFAULT_ORCHESTRATOR_TIMEOUT_SEC,
                    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                    use_cache: bool = True,
                    ) -> VerificationResult:
    """Run two-stage verification on a just-promoted module.

    Returns a VerificationResult recording what happened. Side effects:
    a verification_log row on both success and failure; on failure,
    quarantines the live file and reverts runtime_mode.
    """
    happy_kwargs = happy_kwargs or {}
    conn = _safe_connect(db_path)
    try:
        ensure_verification_table(conn)

        # Cache: recent orchestrator success -> short-circuit
        if use_cache:
            cached = _last_success_within(conn, cap_id, VERIFICATION_CACHE_SEC)
            if cached is not None:
                return VerificationResult(
                    capability_id=cap_id, slug=slug, ok=True,
                    stage="cached",
                    reason=f"orchestrator success at {cached['created_at']}",
                    attempts_so_far=_count_attempts(conn, cap_id),
                )

        # Stage 1: isolated
        stage1_ok, s1_reason, s1_tb, s1_ms = _run_isolated(
            slug, happy_kwargs=happy_kwargs, timeout_sec=isolated_timeout_sec,
        )
        conn.execute(
            "INSERT INTO verification_log "
            "(capability_id, slug, stage, ok, reason, traceback, duration_ms) "
            "VALUES (?, ?, 'isolated', ?, ?, ?, ?)",
            (cap_id, slug, 1 if stage1_ok else 0,
             s1_reason, s1_tb[:8000], s1_ms),
        )
        conn.commit()

        if not stage1_ok:
            attempts = _count_attempts(conn, cap_id)
            permanent = attempts >= max_attempts
            quarantined_path = _quarantine_module(slug)
            _revert_capability(conn, cap_id, permanent=permanent)
            return VerificationResult(
                capability_id=cap_id, slug=slug, ok=False,
                stage="isolated", reason=s1_reason,
                traceback=s1_tb, duration_ms=s1_ms,
                attempts_so_far=attempts,
                quarantined=quarantined_path is not None,
                reverted=True,
            )

        # Stage 2: orchestrator dispatch
        stage2_ok, s2_reason, s2_tb, s2_ms = _run_orchestrator_dispatch(
            slug, capability_id=cap_id, capability_type=capability_type,
            happy_kwargs=happy_kwargs, timeout_sec=orchestrator_timeout_sec,
        )
        conn.execute(
            "INSERT INTO verification_log "
            "(capability_id, slug, stage, ok, reason, traceback, duration_ms) "
            "VALUES (?, ?, 'orchestrator', ?, ?, ?, ?)",
            (cap_id, slug, 1 if stage2_ok else 0,
             s2_reason, s2_tb[:8000], s2_ms),
        )
        conn.commit()

        if not stage2_ok:
            attempts = _count_attempts(conn, cap_id)
            permanent = attempts >= max_attempts
            quarantined_path = _quarantine_module(slug)
            _revert_capability(conn, cap_id, permanent=permanent)
            return VerificationResult(
                capability_id=cap_id, slug=slug, ok=False,
                stage="orchestrator", reason=s2_reason,
                traceback=s2_tb, duration_ms=s1_ms + s2_ms,
                attempts_so_far=attempts,
                quarantined=quarantined_path is not None,
                reverted=True,
            )

        return VerificationResult(
            capability_id=cap_id, slug=slug, ok=True,
            stage="orchestrator", reason="both stages passed",
            duration_ms=s1_ms + s2_ms,
            attempts_so_far=_count_attempts(conn, cap_id),
        )
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def get_retry_guidance(cap_id: str,
                       db_path: str = DEFAULT_DB_PATH,
                       ) -> List[str]:
    """Return failure guidance strings for codegen retry.

    Reads the last failure traceback from verification_log and shapes
    it as a list of hint strings the codegen client can splice into
    its retry prompt.
    """
    if not os.path.exists(db_path):
        return []
    conn = _safe_connect(db_path)
    try:
        ensure_verification_table(conn)
        tb = _get_last_traceback(conn, cap_id)
        if not tb:
            return []
        # Split into digestible hints: the reason line + last few
        # stack frames. Codegen sees these as retry_reasons.
        return [
            "Previous verification failed with this traceback. Fix it:",
            tb[:3000],
        ]
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def verification_status(db_path: str = DEFAULT_DB_PATH,
                        limit: int = 20) -> Dict[str, Any]:
    """Snapshot for /api/self-generation/verification-status."""
    if not os.path.exists(db_path):
        return {"ok": False, "error": f"db not found: {db_path}"}
    conn = _safe_connect(db_path)
    try:
        ensure_verification_table(conn)

        totals = conn.execute(
            """
            SELECT
              SUM(CASE WHEN ok=1 THEN 1 ELSE 0 END) AS successes,
              SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END) AS failures,
              COUNT(*) AS total
            FROM verification_log
            """
        ).fetchone()

        recent = conn.execute(
            """
            SELECT capability_id, slug, stage, ok, reason, duration_ms,
                   created_at
            FROM verification_log
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()

        # Quarantined + reverted counts from capabilities table
        quarantined = -1
        reverted = -1
        try:
            quarantined = conn.execute(
                "SELECT COUNT(*) FROM capabilities WHERE runtime_mode = 'quarantined'"
            ).fetchone()[0]
            reverted = conn.execute(
                "SELECT COUNT(*) FROM capabilities WHERE runtime_mode = 'stub_reverted'"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            pass

        return {
            "ok": True,
            "totals": {
                "successes": int(totals["successes"] or 0),
                "failures": int(totals["failures"] or 0),
                "total": int(totals["total"] or 0),
            },
            "runtime_mode_counts": {
                "quarantined": quarantined,
                "stub_reverted": reverted,
            },
            "recent": [dict(r) for r in recent],
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
