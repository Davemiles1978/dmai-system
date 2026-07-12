"""Fresh Blood Injector (PR E).

Purpose
=======
DMAI's evolution engine ingests candidates that come from DMAI's own
learnings. Over time this becomes an echo chamber — every generation's
input pool converges on the previous generation's output pool, and the
Shannon entropy of the capability_type distribution collapses toward a
single dominant type. At the time this module was written, ``utility``
alone was 57.6% of all capabilities (11,922 / 20,694), and the overall
diversity was 53.6% of the theoretical maximum.

The v2.x DMAI specification calls this defence mechanism "Fresh Blood
Injection", but it was never actually implemented — a docstring in
``components/unified_learning_orchestrator.py`` claims to trigger it, but
no fresh-blood component exists in the codebase and that orchestrator is
itself orphaned (never imported anywhere at runtime).

What this module does
=====================
Once per boot (backfill) and then every ``POLL_SECONDS`` (default 6h),
run one injection round. Each round picks up to ``PER_ROUND_CHANNELS``
of the following channels — weighted by which is currently *least*
represented in the recent injection log — and appends the produced
insights as JSONL rows to ``data/research/insights.jsonl`` so the
existing insight_promoter (PR B) mirrors them into SQL naturally.

Every fresh-blood insight has:
    source          = "fresh_blood"
    domain          = "fresh_blood/<channel>"       e.g. fresh_blood/arxiv
    concept         = human readable seed title
    insight_text    = short prompt encouraging evolution to explore this
    provenance      = extra JSONL field ("fresh_blood") — carried through
                      to any downstream logging, ignored by the promoter's
                      column mapping

Channels
--------
- ``arxiv``        — pulls the newest cs.AI abstracts from arXiv's RSS
                     feed (no key). Deduped against previous injections.
- ``github``       — pulls today's GitHub Trending Python repos via
                     ``https://github.com/trending/python?since=daily``
                     (public, no key). Deduped by repo full name.
- ``crossover``    — picks two distant capability_types from the SQL
                     table and emits a "explore intersection of X and Y"
                     concept.
- ``wildcard``     — draws from a curated frontier vocabulary that
                     covers domains DMAI currently has zero exposure to
                     (biology, music theory, cognitive science, geology,
                     linguistics …).
- ``diversity``    — measures the current capability_type distribution's
                     Shannon entropy; if any single type exceeds
                     ``DIVERSITY_THRESHOLD`` share of the total, emits a
                     targeted concept nudging evolution to prioritise the
                     three most under-represented types.

All external fetches are best-effort with short timeouts. A failed fetch
counts as ``skipped``, never aborts the round.

State
=====
``system_state`` key ``fresh_blood.last_run_ts`` — ISO UTC of the last
successful round. Used for the admin diagnostic and to skip re-runs on
process restart within ``MIN_INTERVAL_SECONDS``.

``system_state`` key ``fresh_blood.injection_log`` — JSON list of the
last 200 ``(channel, seed_hash, ts)`` triples. Used for dedup and for
the round's channel-weighting decision.

Zero new tables — everything piggybacks on ``insights`` (via the JSONL
tail) and ``system_state``.

Callers
=======
``dmai_core_complete.py`` boot hook: ``start_injector_loop()`` runs a
first ``inject_once(force=True)`` synchronously then starts the poll
thread. Same pattern as insight_promoter (PR B) and capability_promoter
(PR D).

Admin diagnostic
----------------
``GET /api/admin/fresh-blood-status`` returns::

    {
      "ok": true,
      "running": true,
      "last_run_ts": "2026-07-13T02:00:00+00:00",
      "sql_fresh_blood_insights": 47,
      "capability_type_entropy": 2.143,
      "capability_type_max_entropy": 4.000,
      "capability_type_diversity_ratio": 0.536,
      "top_types": [["utility", 11922], ...],
      "recent_injections": [{"channel": "arxiv", "concept": "...", "ts": "..."}, ...],
      "last_summary": {"emitted": 4, "skipped": 1, "channels_used": ["arxiv","wildcard"]},
      "ts": "..."
    }
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
import os
import random
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
DEFAULT_JSONL           = Path("data/research/insights.jsonl")
LAST_RUN_KEY            = "fresh_blood.last_run_ts"
LOG_KEY                 = "fresh_blood.injection_log"
POLL_SECONDS            = 6 * 3600      # every 6 hours
MIN_INTERVAL_SECONDS    = 60 * 60       # never inject twice within 1h
PER_ROUND_CHANNELS      = 2             # pick 2 channels per round
INSIGHTS_PER_CHANNEL    = 3             # up to 3 seeds per channel
LOG_MAX_ENTRIES         = 200
DIVERSITY_THRESHOLD     = 0.40          # any single type >40% share triggers a diversity nudge
HTTP_TIMEOUT            = 10

# Curated frontier vocabulary — domains DMAI's capability_type table has
# zero or near-zero coverage in. Deliberately non-technical: the goal is
# to widen the evolution search space, not to teach DMAI new API tricks.
WILDCARD_VOCABULARY: Tuple[str, ...] = (
    # Biology / cognition
    "synaptic plasticity", "gene regulatory networks", "swarm intelligence",
    "circadian rhythm", "phenotypic plasticity", "mycelial networks",
    "morphogenesis", "epigenetic inheritance",
    # Physics / systems
    "phase transitions", "self-organised criticality", "conservation laws",
    "thermodynamic entropy", "chaos theory", "emergent behaviour",
    # Music / language
    "counterpoint", "polyrhythm", "harmonic tension", "prosody",
    "code-switching", "phonemic contrast", "poetic meter",
    # Geology / ecology
    "sedimentation", "keystone species", "trophic cascade",
    "biogeochemical cycles", "island biogeography",
    # Cognitive / social
    "collective memory", "mental models", "cognitive load", "framing effects",
    "reciprocity norms", "narrative transportation",
    # Mathematics
    "topological invariants", "group symmetries", "measure theory",
    "algebraic varieties", "Ramsey theory", "graph colouring",
)

ARXIV_RSS_URL   = "https://export.arxiv.org/rss/cs.AI"
GITHUB_TRENDING = "https://github.com/trending/python?since=daily"


# ── Path helpers ──────────────────────────────────────────────────────────

def _kdb_path() -> str:
    """Same DATA_PATH convention as every other DB-touching module."""
    data = os.environ.get("DATA_PATH", "data/").rstrip("/").rstrip("\\")
    return os.path.join(data, "dmai_knowledge.db")


def _jsonl_path() -> Path:
    """Location of the JSONL the insight_promoter tails."""
    data = os.environ.get("DATA_PATH", "data").rstrip("/").rstrip("\\")
    return Path(data) / "research" / "insights.jsonl"


# ── State helpers ─────────────────────────────────────────────────────────

def _ensure_state(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS system_state ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT,"
        "  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ")"
    )


def _state_get(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT value FROM system_state WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row else None


def _state_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO system_state (key, value, updated_at) "
        "VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
        "updated_at = CURRENT_TIMESTAMP",
        (key, value),
    )


def _load_log(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    raw = _state_get(conn, LOG_KEY)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [e for e in parsed if isinstance(e, dict)]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _save_log(conn: sqlite3.Connection, entries: List[Dict[str, Any]]) -> None:
    trimmed = entries[-LOG_MAX_ENTRIES:]
    _state_set(conn, LOG_KEY, json.dumps(trimmed))


# ── Channel implementations ───────────────────────────────────────────────

def _seed_hash(channel: str, concept: str) -> str:
    return hashlib.sha256(f"{channel}::{concept}".encode("utf-8")).hexdigest()[:16]


def _http_get(url: str, timeout: int = HTTP_TIMEOUT) -> Optional[str]:
    """Best-effort HTTP GET. Returns None on any failure."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DMAI-FreshBlood/1.0 (+https://dmai-web.onrender.com)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            OSError, ValueError) as e:
        logger.debug("fresh_blood http_get failed for %s: %s", url, e)
        return None


def _arxiv_seeds(seen: set, limit: int) -> List[Dict[str, Any]]:
    """Pull up to ``limit`` fresh arXiv cs.AI titles."""
    body = _http_get(ARXIV_RSS_URL)
    if not body:
        return []
    seeds: List[Dict[str, Any]] = []
    # RSS 2.0 format — each <item> block has <title> and <link>.
    for m in re.finditer(
        r"<item[^>]*>.*?<title[^>]*>(.*?)</title>.*?<link[^>]*>(.*?)</link>",
        body, flags=re.DOTALL,
    ):
        title = re.sub(r"^\s*|\s*$", "",
                       re.sub(r"<[^>]+>", "", m.group(1))).strip()
        link  = m.group(2).strip()
        if not title:
            continue
        # arXiv sometimes wraps titles as "Title (arXiv:XXXX.YYYYY vN)" — clean.
        title = re.sub(r"\s*\(arXiv:[^)]+\)\s*$", "", title).strip()
        h = _seed_hash("arxiv", title)
        if h in seen:
            continue
        seeds.append({
            "channel": "arxiv",
            "concept": title[:180],
            "insight_text": f"Explore the ideas in the arXiv paper '{title}'. "
                            f"Look for concepts that could inform DMAI's evolution.",
            "source_url": link,
            "seed_hash": h,
        })
        if len(seeds) >= limit:
            break
    return seeds


def _github_trending_seeds(seen: set, limit: int) -> List[Dict[str, Any]]:
    """Pull today's GitHub Trending Python repos."""
    body = _http_get(GITHUB_TRENDING)
    if not body:
        return []
    seeds: List[Dict[str, Any]] = []
    # Trending page marks repos with <h2 class="h3 lh-condensed"> ... <a href="/owner/repo">
    for m in re.finditer(
        r'<h2\s+class="h3 lh-condensed">\s*<a\s+href="/([^/"]+/[^/"]+)"',
        body,
    ):
        repo = m.group(1).strip()
        if not repo or "/" not in repo:
            continue
        h = _seed_hash("github", repo)
        if h in seen:
            continue
        seeds.append({
            "channel": "github",
            "concept": f"github_trending:{repo}",
            "insight_text": f"Trending Python repo '{repo}' on GitHub today. "
                            f"Consider whether its patterns are worth absorbing "
                            f"into DMAI's capability set.",
            "source_url": f"https://github.com/{repo}",
            "seed_hash": h,
        })
        if len(seeds) >= limit:
            break
    return seeds


def _capability_type_distribution(conn: sqlite3.Connection) -> List[Tuple[str, int]]:
    """Read the current capability_type distribution from SQL."""
    try:
        rows = conn.execute(
            "SELECT capability_type, COUNT(*) FROM capabilities "
            "GROUP BY capability_type ORDER BY 2 DESC"
        ).fetchall()
        return [(str(r[0] or "unknown"), int(r[1] or 0)) for r in rows]
    except sqlite3.OperationalError:
        return []


def _crossover_seeds(conn: sqlite3.Connection, seen: set, limit: int,
                     rng: random.Random) -> List[Dict[str, Any]]:
    """Emit "explore intersection of X and Y" seeds where X and Y are
    distant (differently-sized) capability_types."""
    dist = _capability_type_distribution(conn)
    if len(dist) < 2:
        return []
    heavy = [t for t, _ in dist[: max(1, len(dist) // 2)]]
    light = [t for t, _ in dist[max(1, len(dist) // 2) :]]
    seeds: List[Dict[str, Any]] = []
    attempts = 0
    while len(seeds) < limit and attempts < limit * 5:
        attempts += 1
        if not heavy or not light:
            break
        a = rng.choice(heavy)
        b = rng.choice(light)
        if a == b:
            continue
        concept = f"crossover:{a}×{b}"
        h = _seed_hash("crossover", concept)
        if h in seen:
            continue
        seeds.append({
            "channel": "crossover",
            "concept": concept,
            "insight_text": (
                f"Explore the intersection of the '{a}' and '{b}' capability "
                f"types. What primitives could unify them? Where do their "
                f"invariants collide productively?"
            ),
            "source_url": None,
            "seed_hash": h,
        })
    return seeds


def _wildcard_seeds(seen: set, limit: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Pull from the curated frontier vocabulary."""
    seeds: List[Dict[str, Any]] = []
    pool = list(WILDCARD_VOCABULARY)
    rng.shuffle(pool)
    for term in pool:
        h = _seed_hash("wildcard", term)
        if h in seen:
            continue
        seeds.append({
            "channel": "wildcard",
            "concept": term,
            "insight_text": (
                f"Frontier concept '{term}' — DMAI currently has near-zero "
                f"exposure to this domain. Ask what analogues, structural "
                f"parallels, or unexpected primitives it suggests for "
                f"autonomous systems."
            ),
            "source_url": None,
            "seed_hash": h,
        })
        if len(seeds) >= limit:
            break
    return seeds


def _diversity_metric(dist: List[Tuple[str, int]]) -> Dict[str, Any]:
    total = sum(c for _, c in dist)
    if not total or not dist:
        return {"entropy": 0.0, "max_entropy": 0.0, "ratio": 0.0,
                "dominant": None, "dominant_share": 0.0, "underrepresented": []}
    entropy = -sum(
        (c / total) * math.log2(c / total) for _, c in dist if c > 0
    )
    max_entropy = math.log2(len(dist)) if len(dist) > 1 else 0.0
    ratio = (entropy / max_entropy) if max_entropy > 0 else 0.0
    dominant, dominant_c = dist[0]
    dominant_share = dominant_c / total
    # Bottom-3 by count (excluding zero-count) — targets for nudging.
    non_zero = [(t, c) for t, c in dist if c > 0]
    underrepresented = [t for t, _ in sorted(non_zero, key=lambda x: x[1])[:3]]
    return {
        "entropy": entropy,
        "max_entropy": max_entropy,
        "ratio": ratio,
        "dominant": dominant,
        "dominant_share": dominant_share,
        "underrepresented": underrepresented,
    }


def _diversity_seeds(conn: sqlite3.Connection, seen: set, limit: int,
                     rng: random.Random) -> List[Dict[str, Any]]:
    dist = _capability_type_distribution(conn)
    metric = _diversity_metric(dist)
    if metric["dominant_share"] < DIVERSITY_THRESHOLD:
        return []
    seeds: List[Dict[str, Any]] = []
    pool = metric["underrepresented"] or [t for t, _ in dist[-3:]]
    rng.shuffle(pool)
    for t in pool[:limit]:
        concept = f"diversity_nudge:{t}"
        h = _seed_hash("diversity", concept)
        if h in seen:
            continue
        seeds.append({
            "channel": "diversity",
            "concept": concept,
            "insight_text": (
                f"Capability distribution is dominated by "
                f"'{metric['dominant']}' at {metric['dominant_share']:.0%}. "
                f"Grow the under-represented '{t}' type. What primitives, "
                f"integrations, or missing coverage would raise its share?"
            ),
            "source_url": None,
            "seed_hash": h,
        })
    return seeds


# ── Channel picker ────────────────────────────────────────────────────────

CHANNELS = ("arxiv", "github", "crossover", "wildcard", "diversity")


def _pick_channels(log: List[Dict[str, Any]], k: int,
                   rng: random.Random) -> List[str]:
    """Pick ``k`` channels, weighted inversely by recent usage.

    A channel with zero recent uses gets a large weight; one used a lot
    gets a small weight. Ensures every channel gets exercised over time.
    """
    recent = [e.get("channel") for e in log[-40:]]  # last 40 injections
    counts = {c: 1 + recent.count(c) for c in CHANNELS}
    # Invert: turn "recent uses + 1" into a weight.
    weights = {c: 1.0 / counts[c] for c in CHANNELS}
    chosen: List[str] = []
    pool = list(CHANNELS)
    for _ in range(min(k, len(CHANNELS))):
        w = [weights[c] for c in pool]
        pick = rng.choices(pool, weights=w, k=1)[0]
        chosen.append(pick)
        pool.remove(pick)
    return chosen


# ── JSONL emission ────────────────────────────────────────────────────────

def _emit_row(fp, seed: Dict[str, Any]) -> None:
    row = {
        "concept":      seed["concept"],
        "insight_text": seed["insight_text"],
        "confidence":   0.4,     # fresh blood is exploratory, not authoritative
        "domain":       f"fresh_blood/{seed['channel']}",
        "source":       "fresh_blood",
        "provenance":   "fresh_blood",
        "channel":      seed["channel"],
        "source_url":   seed.get("source_url"),
        "seed_hash":    seed["seed_hash"],
        "timestamp":    _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    fp.write(json.dumps(row) + "\n")


# ── Core injection pass ───────────────────────────────────────────────────

def inject_once(
    *,
    jsonl_path: Optional[Path] = None,
    db_path:    Optional[str]  = None,
    force:      bool = False,
    channels_override: Optional[List[str]] = None,
    per_channel: int = INSIGHTS_PER_CHANNEL,
    per_round_channels: int = PER_ROUND_CHANNELS,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Run one injection round.

    Returns a summary::
        {"emitted": int, "skipped": int,
         "channels_used": [str, ...], "seed_hashes": [str, ...],
         "note": Optional[str]}

    ``note`` is set to ``"cooldown"`` when we bail early because we
    injected within ``MIN_INTERVAL_SECONDS`` and ``force`` is False.
    """
    jp   = Path(jsonl_path) if jsonl_path else _jsonl_path()
    dbp  = db_path or _kdb_path()
    rng  = rng or random.Random()

    conn = safe_open_kdb(dbp)
    try:
        _ensure_state(conn)
        # Cooldown guard.
        last = _state_get(conn, LAST_RUN_KEY)
        if last and not force:
            try:
                last_dt = _dt.datetime.fromisoformat(last)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=_dt.timezone.utc)
                age = (_dt.datetime.now(_dt.timezone.utc) - last_dt).total_seconds()
                if age < MIN_INTERVAL_SECONDS:
                    return {
                        "emitted": 0, "skipped": 0,
                        "channels_used": [], "seed_hashes": [],
                        "note": "cooldown",
                        "age_seconds": age,
                    }
            except (ValueError, TypeError):
                pass  # bad timestamp — treat as no cooldown

        log = _load_log(conn)
        seen_hashes = {e.get("seed_hash") for e in log if e.get("seed_hash")}

        # Pick channels.
        picks = channels_override or _pick_channels(log, per_round_channels, rng)

        collected: List[Dict[str, Any]] = []
        skipped = 0
        for ch in picks:
            if ch == "arxiv":
                got = _arxiv_seeds(seen_hashes, per_channel)
            elif ch == "github":
                got = _github_trending_seeds(seen_hashes, per_channel)
            elif ch == "crossover":
                got = _crossover_seeds(conn, seen_hashes, per_channel, rng)
            elif ch == "wildcard":
                got = _wildcard_seeds(seen_hashes, per_channel, rng)
            elif ch == "diversity":
                got = _diversity_seeds(conn, seen_hashes, per_channel, rng)
            else:
                skipped += 1
                continue
            if not got:
                skipped += 1
                continue
            for seed in got:
                seen_hashes.add(seed["seed_hash"])
            collected.extend(got)

        # Emit to JSONL.
        if collected:
            jp.parent.mkdir(parents=True, exist_ok=True)
            with jp.open("a", encoding="utf-8") as fp:
                for seed in collected:
                    _emit_row(fp, seed)

        # Update log + last_run.
        now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
        for seed in collected:
            log.append({
                "channel":   seed["channel"],
                "concept":   seed["concept"],
                "seed_hash": seed["seed_hash"],
                "ts":        now_iso,
            })
        _save_log(conn, log)
        _state_set(conn, LAST_RUN_KEY, now_iso)
        conn.commit()

        channels_used = sorted({s["channel"] for s in collected})
        summary = {
            "emitted":        len(collected),
            "skipped":        skipped,
            "channels_used":  channels_used,
            "seed_hashes":    [s["seed_hash"] for s in collected],
            "picked":         list(picks),
            "ts":             now_iso,
        }
        logger.info(
            "🩸 fresh_blood: emitted=%d skipped=%d channels=%s",
            summary["emitted"], summary["skipped"], channels_used,
        )
        return summary
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Loop wrapper ──────────────────────────────────────────────────────────

class FreshBloodInjectorLoop:
    """Background thread that runs ``inject_once`` every ``POLL_SECONDS``."""

    def __init__(self, jsonl_path: Optional[Path] = None,
                 db_path: Optional[str] = None,
                 poll_seconds: int = POLL_SECONDS):
        self._jsonl_path   = Path(jsonl_path) if jsonl_path else None
        self._db_path      = db_path
        self._poll         = int(poll_seconds)
        self._thread: Optional[threading.Thread] = None
        self._stop         = threading.Event()
        self.last_summary: Dict[str, Any] = {}

    def _run(self) -> None:
        # First pass on boot — force=True so we don't respect cooldown
        # (a freshly-restarted process should always inject once).
        try:
            self.last_summary = inject_once(
                jsonl_path=self._jsonl_path,
                db_path=self._db_path,
                force=True,
            )
        except Exception as e:
            logger.exception("fresh_blood initial inject failed: %s", e)
            self.last_summary = {"error": str(e)}

        while not self._stop.wait(self._poll):
            try:
                self.last_summary = inject_once(
                    jsonl_path=self._jsonl_path,
                    db_path=self._db_path,
                    force=False,
                )
            except Exception as e:
                logger.exception("fresh_blood loop inject failed: %s", e)
                self.last_summary = {"error": str(e)}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="fresh_blood_injector", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


_LOOP: Optional[FreshBloodInjectorLoop] = None


def start_injector_loop(jsonl_path: Optional[Path] = None,
                        db_path: Optional[str] = None) -> FreshBloodInjectorLoop:
    """Idempotent boot hook. Safe to call multiple times.

    Returns the existing loop only when its background thread is still
    alive; otherwise rebuilds a fresh ``FreshBloodInjectorLoop`` and
    starts it. Matches the survival semantics used by
    ``insight_promoter.start_promoter_loop`` and
    ``capability_promoter.start_promoter_loop``.

    Motivation: Gunicorn's ``preload_app`` boots the module in the
    master process (starting the daemon thread) then forks workers.
    Threads do not survive ``os.fork()``, so each forked worker
    inherits ``_LOOP`` as a non-None reference to a **dead** thread.
    A guard of ``if _LOOP is None`` therefore never re-starts the
    thread in any worker, leaving ``running: False`` forever even
    though the module-level state looks initialised.
    """
    global _LOOP
    if _LOOP is not None and _LOOP._thread and _LOOP._thread.is_alive():
        return _LOOP
    _LOOP = FreshBloodInjectorLoop(jsonl_path=jsonl_path, db_path=db_path)
    _LOOP.start()
    return _LOOP


def get_injector_loop() -> Optional[FreshBloodInjectorLoop]:
    return _LOOP
