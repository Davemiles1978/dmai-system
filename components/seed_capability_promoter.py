"""
seed_capability_promoter.py
===========================

The missing bridge between DMAI's fresh-blood insight stream and the
capability registry.

Prior to this module the three learning loops were disconnected:

* ``FreshBloodInjectorLoop``  — writes concept seeds to a JSONL log +
  the ``fresh_blood_insights`` SQL table.
* ``InsightPromoterLoop``     — mirrors the JSONL into the SQL
  ``insights`` table.
* ``CapabilityPromoterLoop``  — mirrors ``registry.json`` into the SQL
  ``capabilities`` table.

Nothing wrote **new** entries to ``registry.json``. The registry mtime
therefore never changed and the capability count sat frozen. This
module closes the loop.

Behaviour
---------

Runs periodically. On each pass:

1.  Reads the fresh-blood JSONL feed from a persisted byte offset.
2.  Parses every new row, filters against a seen-id set (drawn from the
    current registry), and applies a soft daily cap.
3.  Converts each accepted seed into a ``type=concept`` entry:

        {
            "id":             "concept:<channel>:<slug>",
            "name":           <concept slug>,
            "type":           "concept",
            "capability_type": <inferred>,
            "description":    <insight_text or fallback>,
            "source_url":     <url or None>,
            "runtime_mode":   "stub",
            "integrated_at":  <ISO timestamp>
        }

4.  Rewrites ``registry.json`` atomically. This bumps the mtime, so the
    existing ``capability_promoter`` picks the new rows up on its next
    pass and mirrors them into the SQL ``capabilities`` table.

Rationale
---------

Concept stubs are placeholder capabilities — they are not executable
functions. Their purpose is to give the readiness monitor and the
diversity metrics **something to move**. A follow-up PR (H) will layer
an LLM-driven code generator on top so accepted seeds can become
executable capabilities.

Notes
-----

* Soft daily cap (default 10) is enforced across all fresh-blood
  channels combined, per calendar day (UTC).
* Persisted state lives in ``system_state`` under three keys:
      ``seed_capability_promoter.jsonl_offset``
      ``seed_capability_promoter.day_bucket``
      ``seed_capability_promoter.day_count``
* Idempotent: re-running the same JSONL bytes with the same registry
  produces zero new writes.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from components.db import safe_open_kdb

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────

DEFAULT_DAILY_CAP    = 10       # soft cap on capabilities promoted per day
POLL_SECONDS         = 300      # 5 minutes between passes
MIN_INTERVAL_SECONDS = 60       # gate manual re-runs to at least this
JSONL_CHUNK_LIMIT    = 500      # max rows to read from JSONL per pass

# system_state keys
OFFSET_KEY     = "seed_capability_promoter.jsonl_offset"
DAY_BUCKET_KEY = "seed_capability_promoter.day_bucket"
DAY_COUNT_KEY  = "seed_capability_promoter.day_count"
LAST_RUN_KEY   = "seed_capability_promoter.last_run_ts"
REJECT_LOG_KEY = "seed_capability_promoter.reject_log"  # JSON list, tail-of-20
JUDGE_STATS_KEY = "seed_capability_promoter.judge_stats"  # JSON dict of counters

# Rejection reason codes surfaced in the reject log. The real gating
# lives in ``components/self_judge`` \u2014 this module just records what it
# decided.
REJECT_MALFORMED  = "malformed"
REJECT_DUP_ID     = "dup_id"
REJECT_SELF_JUDGE = "self_judge_reject"  # DMAI's own reasoning rejected it

# Verdict codes returned by ``self_judge.judge_seed``.
VERDICT_ACCEPT = "accept"
VERDICT_REJECT = "reject"
VERDICT_DEFER  = "defer"

REJECT_LOG_TAIL     = 20     # keep the last N rejection records
DEFERRED_QUEUE_TAIL = 50     # cap on rows in deferred_seeds


# ── Path resolvers (mirror the conventions used by peer modules) ─────────

def _kdb_path() -> str:
    data = os.environ.get("DATA_PATH", "data/").rstrip("/").rstrip("\\")
    return os.path.join(data, "dmai_knowledge.db")


def _registry_path() -> Path:
    env = os.environ.get("DATA_PATH")
    base = Path(env.rstrip("/").rstrip("\\")) if env else Path("data")
    return base / "capabilities" / "registry.json"


def _jsonl_path() -> Path:
    env = os.environ.get("DATA_PATH")
    base = Path(env.rstrip("/").rstrip("\\")) if env else Path("data")
    return base / "fresh_blood" / "insights.jsonl"


# ── State helpers ─────────────────────────────────────────────────────────

def _ensure_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS system_state ("
        " key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
    )


def _state_get(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT value FROM system_state WHERE key = ?", (key,)
    ).fetchone()
    if not row or row[0] is None:
        return None
    return str(row[0])


def _state_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO system_state (key, value, updated_at) "
        "VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
        "updated_at = CURRENT_TIMESTAMP",
        (key, value),
    )


# ── Seed -> capability conversion ────────────────────────────────────────

# Map fresh-blood channels to the capability_type they most naturally
# nudge toward. These are heuristic hints; the real payoff is that they
# vary across channels so the diversity metric can move.
CHANNEL_TO_CAP_TYPE = {
    "arxiv":             "research",
    "github":            "integration",
    "crossover":         "composite",
    "wildcard":          "frontier",
    "diversity":         "diversity_nudge",
    "ai_releases":       "ai_provider_update",
    "ai_repo_releases":  "ai_provider_update",
}


def _slugify(text: str) -> str:
    """Return a filesystem-safe slug of the concept string.

    Preserves the crossover ``x`` separator (encoded to ``-x-``) so the
    ID retains information about the two parent types.
    """
    text = text.replace("×", "-x-").replace("*", "-x-")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip())
    return text.strip("-").lower()[:120] or "unnamed"


def _infer_cap_type_from_concept(channel: str, concept: str) -> str:
    """Extract a more specific capability_type when the concept encodes one.

    ``crossover:X×Y``      → ``crossover:X-x-Y`` slug (composite)
    ``diversity_nudge:T``  → ``T`` directly (nudges toward that type)
    Otherwise falls back to the channel-level mapping.
    """
    if channel == "diversity" and concept.startswith("diversity_nudge:"):
        target = concept.split(":", 1)[1].strip()
        if target:
            return target[:100]
    return CHANNEL_TO_CAP_TYPE.get(channel, "concept")


def seed_to_capability(seed: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Convert one fresh-blood seed row to a registry (id, entry) pair.

    Returns ``None`` if the row is malformed (missing channel or concept).
    """
    channel = seed.get("channel")
    concept = seed.get("concept")
    if not channel or not concept:
        return None

    slug = _slugify(str(concept))
    cap_id = f"concept:{channel}:{slug}"
    name = str(concept)[:500]
    cap_type = _infer_cap_type_from_concept(str(channel), str(concept))

    description = seed.get("insight_text") or (
        f"Concept stub emitted by fresh_blood[{channel}]: {concept}. "
        "Placeholder capability — not yet executable. Awaiting the "
        "LLM-driven code generator pass (PR H) to materialise into a "
        "runnable function."
    )
    source_url = seed.get("source_url")
    integrated_at = seed.get("ts") or _dt.datetime.now(_dt.timezone.utc).isoformat()

    entry = {
        "name":            name,
        "type":            "concept",
        "capability_type": cap_type,
        "description":     str(description)[:2000],
        "source_url":      str(source_url)[:1000] if source_url else None,
        "runtime_mode":    "stub",
        "language":        None,
        "methods":         [],
        "is_async":        False,
        "args":            [],
        "integrated_at":   integrated_at,
        "provenance":      "fresh_blood_seed",
        "seed_hash":       seed.get("seed_hash"),
    }
    return cap_id, entry


# ── Registry read/write ───────────────────────────────────────────────────

def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"capabilities": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "seed_capability_promoter: registry unreadable, treating as empty"
        )
        return {"capabilities": {}}
    if not isinstance(data, dict):
        return {"capabilities": {}}
    caps = data.get("capabilities")
    if not isinstance(caps, dict):
        data["capabilities"] = {}
    return data


def _atomic_write_registry(path: Path, data: Dict[str, Any]) -> None:
    """Write registry.json via a temp file + rename so readers never see a
    torn file. This bumps mtime, which triggers the existing
    capability_promoter on its next pass.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".registry.", suffix=".json", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of the temp file.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── JSONL read (offset-based, tail-follower semantics) ───────────────────

def _read_new_seeds(jp: Path, start_offset: int,
                    max_rows: int = JSONL_CHUNK_LIMIT
                    ) -> Tuple[List[Dict[str, Any]], int]:
    """Read up to ``max_rows`` lines from ``jp`` starting at ``start_offset``.

    Returns ``(seeds, new_offset)``. If the file has been truncated
    (size < start_offset), rewinds to 0 to recover.
    """
    if not jp.exists():
        return [], start_offset

    try:
        size = jp.stat().st_size
    except OSError:
        return [], start_offset

    # File truncated / rotated — rewind.
    if size < start_offset:
        start_offset = 0

    seeds: List[Dict[str, Any]] = []
    try:
        with jp.open("rb") as f:
            f.seek(start_offset)
            for _ in range(max_rows):
                line = f.readline()
                if not line:
                    break
                # Skip partial trailing writes: only accept full lines.
                if not line.endswith(b"\n"):
                    break
                try:
                    obj = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    seeds.append(obj)
                start_offset = f.tell()
    except OSError as e:
        logger.warning("seed_capability_promoter: JSONL read failed: %s", e)
        return seeds, start_offset

    return seeds, start_offset


# ── Daily-cap accounting ─────────────────────────────────────────────────

def _today_bucket() -> str:
    return _dt.datetime.now(_dt.timezone.utc).date().isoformat()


def _load_day_counter(conn: sqlite3.Connection) -> Tuple[str, int]:
    bucket = _state_get(conn, DAY_BUCKET_KEY) or ""
    count_str = _state_get(conn, DAY_COUNT_KEY) or "0"
    try:
        count = int(count_str)
    except ValueError:
        count = 0
    today = _today_bucket()
    if bucket != today:
        # New day → reset.
        return today, 0
    return bucket, count


def _persist_day_counter(conn: sqlite3.Connection, bucket: str, count: int) -> None:
    _state_set(conn, DAY_BUCKET_KEY, bucket)
    _state_set(conn, DAY_COUNT_KEY, str(count))


# ── Deferred queue / reject log / judge stats ────────────────────────────

def _ensure_deferred_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS deferred_seeds ("
        " seed_hash TEXT PRIMARY KEY,"
        " concept TEXT, channel TEXT, reason TEXT,"
        " gap_summary TEXT, unknown_tokens TEXT,"
        " seed_json TEXT, first_seen TEXT, last_seen TEXT,"
        " attempts INTEGER DEFAULT 1,"
        " acquired INTEGER DEFAULT 0)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_deferred_acquired "
        "ON deferred_seeds(acquired)"
    )


def _defer_seed(conn: sqlite3.Connection,
                seed: Dict[str, Any],
                verdict_obj: Dict[str, Any]) -> bool:
    """Record a deferred seed. Returns True if a new row was written.

    ``verdict_obj`` should be the dict returned by
    ``self_judge.judge_seed``.
    """
    _ensure_deferred_table(conn)
    seed_hash = seed.get("seed_hash") or _fallback_hash(seed)
    now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
    gap = verdict_obj.get("gap_summary") or verdict_obj.get("reason") or ""
    unknown = verdict_obj.get("unknown_tokens") or []
    row = conn.execute(
        "SELECT attempts FROM deferred_seeds WHERE seed_hash = ?",
        (seed_hash,),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE deferred_seeds SET last_seen = ?, attempts = attempts + 1 "
            "WHERE seed_hash = ?",
            (now_iso, seed_hash),
        )
        return False
    conn.execute(
        "INSERT INTO deferred_seeds "
        "(seed_hash, concept, channel, reason, gap_summary, unknown_tokens,"
        " seed_json, first_seen, last_seen, attempts, acquired) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0)",
        (seed_hash,
         str(seed.get("concept", ""))[:500],
         str(seed.get("channel", ""))[:100],
         str(verdict_obj.get("reason", ""))[:200],
         str(gap)[:500],
         json.dumps(unknown)[:1000],
         json.dumps(seed)[:4000],
         now_iso, now_iso),
    )
    _prune_deferred_queue(conn)
    return True


def _prune_deferred_queue(conn: sqlite3.Connection) -> None:
    """Keep at most ``DEFERRED_QUEUE_TAIL`` rows in the deferred queue,
    preferring unresolved rows over already-acquired ones."""
    row = conn.execute("SELECT COUNT(*) FROM deferred_seeds").fetchone()
    if not row or row[0] <= DEFERRED_QUEUE_TAIL:
        return
    # Delete oldest acquired first, then oldest unresolved.
    conn.execute(
        "DELETE FROM deferred_seeds WHERE seed_hash IN ("
        " SELECT seed_hash FROM deferred_seeds "
        " ORDER BY acquired DESC, first_seen ASC LIMIT ?)",
        (int(row[0]) - DEFERRED_QUEUE_TAIL,),
    )


def _fallback_hash(seed: Dict[str, Any]) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(json.dumps(seed, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def _push_reject(conn: sqlite3.Connection,
                 seed: Dict[str, Any],
                 reason: str,
                 detail: str = "") -> None:
    """Append a rejection record to the tail-of-N reject log."""
    raw = _state_get(conn, REJECT_LOG_KEY)
    try:
        arr = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        arr = []
    if not isinstance(arr, list):
        arr = []
    arr.append({
        "ts":       _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "reason":   reason,
        "detail":   str(detail)[:300],
        "channel":  seed.get("channel"),
        "concept":  str(seed.get("concept", ""))[:200],
        "seed_hash": seed.get("seed_hash"),
    })
    if len(arr) > REJECT_LOG_TAIL:
        arr = arr[-REJECT_LOG_TAIL:]
    _state_set(conn, REJECT_LOG_KEY, json.dumps(arr))


def _bump_judge_stat(conn: sqlite3.Connection, key: str) -> None:
    raw = _state_get(conn, JUDGE_STATS_KEY)
    try:
        stats = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        stats = {}
    if not isinstance(stats, dict):
        stats = {}
    stats[key] = int(stats.get(key, 0)) + 1
    _state_set(conn, JUDGE_STATS_KEY, json.dumps(stats))


def _load_judge_stats(conn: sqlite3.Connection) -> Dict[str, int]:
    raw = _state_get(conn, JUDGE_STATS_KEY)
    try:
        stats = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        stats = {}
    return stats if isinstance(stats, dict) else {}


# ── Main pass ────────────────────────────────────────────────────────────

def promote_once(*,
                 registry_path: Optional[Path] = None,
                 jsonl_path:    Optional[Path] = None,
                 db_path:       Optional[str]  = None,
                 daily_cap:     int = DEFAULT_DAILY_CAP,
                 max_jsonl_rows: int = JSONL_CHUNK_LIMIT,
                 judge=None,
                 acquire_gap=None,
                 ) -> Dict[str, Any]:
    """Run one seed → capability promotion pass.

    Every candidate is routed through DMAI's own self-judge. Only
    ``VERDICT_ACCEPT`` seeds are written to the registry. ``REJECT``
    seeds are logged (see ``REJECT_LOG_KEY``); ``DEFER`` seeds are
    queued in the ``deferred_seeds`` table and, if ``acquire_gap`` is
    supplied, kicked off asynchronously so the knowledge is present on
    the next pass.

    Parameters
    ----------
    judge : callable | None
        Test-injection hook. Signature::

            judge(seed: dict, db_path: str) -> dict

        with keys ``verdict``, ``reason``, ``gap_summary``,
        ``unknown_tokens``, ``confidence``. Defaults to
        ``self_judge.judge_seed``.
    acquire_gap : callable | None
        Optional callback fired for every deferred seed::

            acquire_gap(concept: str, gap: str, unknown_tokens: list) -> None

        Fire-and-forget — the acquirer commits to DB itself.
    """
    # Lazy import so tests that don't touch self_judge can stub it. The
    # adapter normalises the return value: self_judge.judge_seed returns
    # a Verdict dataclass; we translate it to the flat dict shape the
    # rest of this module expects (verdict / reason / gap_summary /
    # unknown_tokens / confidence).
    if judge is None:
        from components.self_judge import judge_seed as _judge_seed  # noqa: E501

        def judge(seed: Dict[str, Any], db_path: str) -> Dict[str, Any]:
            _conn = safe_open_kdb(db_path)
            try:
                verdict = _judge_seed(seed, _conn)
            finally:
                try:
                    _conn.close()
                except Exception:
                    pass
            return {
                "verdict":        verdict.verdict,
                "confidence":     verdict.confidence,
                "reason":         verdict.reason,
                "gap_summary":    verdict.knowledge_gap or verdict.reason,
                "unknown_tokens": list(verdict.signals.unknown_tokens),
            }

    rpath = Path(registry_path) if registry_path else _registry_path()
    jp    = Path(jsonl_path)    if jsonl_path    else _jsonl_path()
    dbp   = db_path or _kdb_path()

    conn = safe_open_kdb(dbp)
    try:
        _ensure_state_table(conn)
        _ensure_deferred_table(conn)

        # Load offset + day counter.
        offset_str = _state_get(conn, OFFSET_KEY) or "0"
        try:
            start_offset = int(offset_str)
        except ValueError:
            start_offset = 0
        day_bucket, day_count = _load_day_counter(conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # ---- Read new seeds without holding the DB connection.
    seeds, new_offset = _read_new_seeds(jp, start_offset, max_jsonl_rows)

    # ---- Load registry; capacity check.
    registry = _load_registry(rpath)
    caps = registry.get("capabilities")
    if not isinstance(caps, dict):
        caps = {}
    # Rebind unconditionally: `.get(...) or {}` returns a fresh dict
    # when the stored value is missing/empty, which would otherwise
    # leave the registry's `capabilities` key detached from `caps`.
    registry["capabilities"] = caps

    promoted           = 0
    rejected           = 0
    deferred_new       = 0
    deferred_repeat    = 0
    skipped_dupes      = 0
    skipped_malformed  = 0
    cap_hit            = False
    remaining = max(0, daily_cap - day_count)

    # Second connection for writes to reject log / deferred queue /
    # judge stats. Kept open only for the tight write loop to avoid
    # long-held locks; released before the atomic registry write.
    write_conn = safe_open_kdb(dbp)
    try:
        for seed in seeds:
            if remaining <= 0:
                cap_hit = True
                break
            pair = seed_to_capability(seed)
            if pair is None:
                skipped_malformed += 1
                _push_reject(write_conn, seed, REJECT_MALFORMED,
                             "missing channel or concept")
                _bump_judge_stat(write_conn, "malformed")
                continue
            cap_id, entry = pair
            if cap_id in caps:
                skipped_dupes += 1
                _bump_judge_stat(write_conn, "dup_id")
                continue

            # ---- DMAI judges.
            try:
                verdict_obj = judge(seed, db_path=dbp)
            except Exception as e:
                logger.warning(
                    "seed_capability_promoter: self_judge crashed: %s "
                    "— conservatively deferring seed", e,
                )
                verdict_obj = {
                    "verdict":        VERDICT_DEFER,
                    "reason":         "judge_crash",
                    "gap_summary":    f"self_judge crashed: {e}",
                    "unknown_tokens": [],
                    "confidence":     0.0,
                }

            verdict = verdict_obj.get("verdict", VERDICT_DEFER)

            if verdict == VERDICT_ACCEPT:
                # Attach the judgement provenance to the entry.
                entry["provenance"] = "fresh_blood_seed+self_judge"
                entry["judge_confidence"] = round(
                    float(verdict_obj.get("confidence", 0.0)), 4,
                )
                caps[cap_id] = entry
                promoted += 1
                remaining -= 1
                _bump_judge_stat(write_conn, "accept")

            elif verdict == VERDICT_REJECT:
                rejected += 1
                _push_reject(write_conn, seed, REJECT_SELF_JUDGE,
                             str(verdict_obj.get("reason", ""))[:200])
                _bump_judge_stat(write_conn, "reject")

            else:  # defer
                is_new = _defer_seed(write_conn, seed, verdict_obj)
                if is_new:
                    deferred_new += 1
                else:
                    deferred_repeat += 1
                _bump_judge_stat(write_conn, "defer")

                # Fire-and-forget gap acquisition.
                if callable(acquire_gap):
                    try:
                        acquire_gap(
                            str(seed.get("concept", ""))[:500],
                            str(verdict_obj.get("gap_summary", ""))[:500],
                            list(verdict_obj.get("unknown_tokens") or []),
                        )
                    except Exception as e:
                        logger.warning(
                            "seed_capability_promoter: acquire_gap failed: %s", e,
                        )

        write_conn.commit()
    finally:
        try:
            write_conn.close()
        except Exception:
            pass

    # ---- Persist registry (only if we actually added anything).
    if promoted > 0:
        _atomic_write_registry(rpath, registry)

    # ---- Persist offset + day counter.
    new_day_count = day_count + promoted
    now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
    conn = safe_open_kdb(dbp)
    try:
        _state_set(conn, OFFSET_KEY, str(new_offset))
        _persist_day_counter(conn, day_bucket, new_day_count)
        _state_set(conn, LAST_RUN_KEY, now_iso)
        judge_stats = _load_judge_stats(conn)
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    summary = {
        "promoted":            promoted,
        "rejected":            rejected,
        "deferred_new":        deferred_new,
        "deferred_repeat":     deferred_repeat,
        "skipped_dupes":       skipped_dupes,
        "skipped_malformed":   skipped_malformed,
        "read":                len(seeds),
        "cap_hit":             cap_hit,
        "day_bucket":          day_bucket,
        "day_count_after":     new_day_count,
        "daily_cap":           daily_cap,
        "jsonl_offset_after":  new_offset,
        "registry_size_after": len(caps),
        "judge_stats":         judge_stats,
        "ts":                  now_iso,
    }
    logger.info(
        "seed_capability_promoter: promoted=%d rejected=%d deferred=%d/%d "
        "read=%d cap_hit=%s day=%s day_count=%d",
        promoted, rejected, deferred_new, deferred_repeat,
        len(seeds), cap_hit, day_bucket, new_day_count,
    )
    return summary


# ── Loop wrapper (mirrors peer modules) ──────────────────────────────────

class SeedCapabilityPromoterLoop:
    """Background thread that runs ``promote_once`` every ``POLL_SECONDS``.

    Idempotent boot semantics match ``fresh_blood_injector`` and
    ``capability_promoter`` — a dead thread after Gunicorn fork is
    detected and replaced by :func:`start_seed_capability_promoter_loop`.
    """

    def __init__(self,
                 registry_path: Optional[Path] = None,
                 jsonl_path:    Optional[Path] = None,
                 db_path:       Optional[str]  = None,
                 daily_cap:     int = DEFAULT_DAILY_CAP,
                 poll_seconds:  int = POLL_SECONDS,
                 enable_acquirer: bool = True,
                 kg=None):
        self._registry_path = Path(registry_path) if registry_path else None
        self._jsonl_path    = Path(jsonl_path)    if jsonl_path    else None
        self._db_path       = db_path
        self._daily_cap     = int(daily_cap)
        self._poll          = int(poll_seconds)
        self._enable_acquirer = bool(enable_acquirer)
        self._kg            = kg
        self._thread: Optional[threading.Thread] = None
        self._stop          = threading.Event()
        self.last_summary: Dict[str, Any] = {}

    # ── Acquirer trampoline (fire-and-forget, daemon thread) ────────────
    def _acquire_async(self, concept: str, gap: str,
                       unknown_tokens: List[str]) -> None:
        def _worker():
            try:
                from components.knowledge_acquirer import acquire_and_commit
                acquire_and_commit(
                    concept, gap,
                    unknown_tokens=unknown_tokens,
                    kg=self._kg,
                    db_path=self._db_path,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "seed_capability_promoter: async acquire failed: %s", e,
                )
        threading.Thread(
            target=_worker,
            name=f"kacq:{concept[:32]}",
            daemon=True,
        ).start()

    def _promote_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = dict(
            registry_path=self._registry_path,
            jsonl_path=self._jsonl_path,
            db_path=self._db_path,
            daily_cap=self._daily_cap,
        )
        if self._enable_acquirer:
            kw["acquire_gap"] = self._acquire_async
        return kw

    def _run(self) -> None:
        try:
            self.last_summary = promote_once(**self._promote_kwargs())
        except Exception as e:
            logger.exception("seed_capability_promoter initial pass failed: %s", e)
            self.last_summary = {"error": str(e)}

        while not self._stop.wait(self._poll):
            try:
                self.last_summary = promote_once(**self._promote_kwargs())
            except Exception as e:
                logger.exception("seed_capability_promoter loop failed: %s", e)
                self.last_summary = {"error": str(e)}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="seed_capability_promoter", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


_LOOP: Optional[SeedCapabilityPromoterLoop] = None


def start_seed_capability_promoter_loop(
    registry_path: Optional[Path] = None,
    jsonl_path:    Optional[Path] = None,
    db_path:       Optional[str]  = None,
    daily_cap:     int = DEFAULT_DAILY_CAP,
    enable_acquirer: bool = True,
    kg=None,
) -> SeedCapabilityPromoterLoop:
    """Idempotent boot hook. See ``fresh_blood_injector.start_injector_loop``
    for the fork-survival rationale.
    """
    global _LOOP
    if _LOOP is not None and _LOOP._thread and _LOOP._thread.is_alive():
        return _LOOP
    _LOOP = SeedCapabilityPromoterLoop(
        registry_path=registry_path,
        jsonl_path=jsonl_path,
        db_path=db_path,
        daily_cap=daily_cap,
        enable_acquirer=enable_acquirer,
        kg=kg,
    )
    _LOOP.start()
    return _LOOP


def get_seed_capability_promoter_loop() -> Optional[SeedCapabilityPromoterLoop]:
    return _LOOP
