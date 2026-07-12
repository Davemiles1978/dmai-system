"""Pick Settler — close the outcome loop on every model pick.

The betting_advisor inserts every model pick into ``mon_tracking_picks``
regardless of the EV gate. This is the *training* record: we score
the model's picks against actual race outcomes without staking money.

``betting_advisor.settle_tracking_pick(event_name, market,
winning_selection)`` exists as a primitive but **nothing was calling
it in a loop** — picks accumulated ``outcome='pending'`` forever. This
module fixes that by polling OpticOdds ``/results`` for each pending
row and calling the existing settle primitive.

Design
======

- Polls ``mon_tracking_picks`` every ``POLL_SECONDS`` (default 15 min).
- For each pending pick with ``created_at`` older than
  ``PICK_MIN_AGE_SECONDS`` (default 30 min), fetches the event result
  from OpticOdds.
- Passes the winning selection to
  ``betting_advisor.settle_tracking_pick`` which handles the actual
  outcome + paper_pl computation. Idempotent (rechecks pending
  status).
- Missing results are silently skipped — a race that hasn't finished
  yet, or a result that OpticOdds doesn't have, stays pending. We
  will retry on the next tick.
- OpticOdds integration is via the ``opticodds`` connector when
  available; falls back to direct API call via ``OPTICODDS_API_KEY``
  env var. If both routes are unavailable, the loop still runs and
  reports ``no_result`` per pick — no crash.

Manual override
===============
A separate ``/api/monetisation/picks/<id>/settle`` endpoint (defined
in dmai_core_complete.py) allows David to manually settle picks that
OpticOdds doesn't cover (voids, protests, unusual markets).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── tunables ────────────────────────────────────────────────────────────────

POLL_SECONDS = int(os.getenv("PICK_SETTLER_POLL_SECONDS", "900"))    # 15 min
PICK_MIN_AGE_SECONDS = int(os.getenv("PICK_SETTLER_MIN_AGE", "1800"))  # 30 min
MAX_PENDING_PER_ROUND = 100
FETCH_TIMEOUT_SECONDS = 8


# ── helpers ─────────────────────────────────────────────────────────────────

def _now_ts() -> float:
    return time.time()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _http_get_json(url: str, headers: Optional[Dict[str, str]] = None,
                   timeout: int = FETCH_TIMEOUT_SECONDS) -> Optional[Any]:
    try:
        import urllib.request
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                return None
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        logger.debug("http_get_json failed %s: %s", url, e)
        return None


# ── opticodds fetch ─────────────────────────────────────────────────────────

def fetch_winning_selection(event_name: str, market: str,
                            *, fetcher: Optional[Callable[..., Any]] = None) -> Optional[str]:
    """Return the winning selection string for ``(event_name, market)``.

    ``fetcher`` is an override hook for tests. When None, we try the
    OpticOdds public API via ``OPTICODDS_API_KEY``. Returns None when
    the event isn't graded yet or when credentials are missing.

    Result-string convention: whatever OpticOdds returns as the
    winning ``selection`` — the settle primitive does a
    case-insensitive strip-compare against the pick's ``selection``,
    so casing/whitespace don't matter.
    """
    if fetcher is not None:
        try:
            return fetcher(event_name=event_name, market=market)
        except Exception as e:
            logger.debug("fetcher hook failed: %s", e)
            return None

    key = os.getenv("OPTICODDS_API_KEY") or os.getenv("OPTIC_ODDS_API_KEY")
    if not key:
        return None
    # OpticOdds v3: /results?event_name=...&market=...
    from urllib.parse import quote
    url = ("https://api.opticodds.com/api/v3/results"
           f"?event_name={quote(event_name)}&market={quote(market)}")
    data = _http_get_json(url, headers={"X-Api-Key": key})
    if not data:
        return None
    # Response schema (per OpticOdds docs): {"data": [{"selections":
    # [{"name": "...", "is_winner": true}, ...]}]}
    try:
        for event in data.get("data", []):
            for sel in event.get("selections", []):
                if sel.get("is_winner"):
                    return str(sel.get("name") or "").strip()
    except Exception as e:
        logger.debug("opticodds parse failed for %s/%s: %s", event_name, market, e)
    return None


# ── settle ──────────────────────────────────────────────────────────────────

def _load_pending_picks(advisor: Any, limit: int = MAX_PENDING_PER_ROUND
                        ) -> List[Dict[str, Any]]:
    """Fetch pending picks old enough to be worth settling."""
    cutoff = _now_ts() - PICK_MIN_AGE_SECONDS
    with advisor._conn() as c:
        rows = c.execute(
            "SELECT id, event_name, market, selection, created_at "
            "FROM mon_tracking_picks WHERE outcome='pending' AND created_at <= ? "
            "ORDER BY created_at ASC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def settle_once(advisor: Any, *,
                fetcher: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """One full pass through pending picks. Returns a summary."""
    if advisor is None:
        return {"error": "advisor_missing", "ts": _now_iso()}
    try:
        picks = _load_pending_picks(advisor)
    except Exception as e:
        logger.exception("load_pending_picks failed: %s", e)
        return {"error": str(e), "ts": _now_iso()}

    summary = {
        "checked":    len(picks),
        "settled":    0,
        "no_result":  0,
        "errors":     0,
        "results":    [],
        "ts":         _now_iso(),
    }
    for pick in picks:
        try:
            winner = fetch_winning_selection(
                pick["event_name"], pick["market"], fetcher=fetcher,
            )
            if not winner:
                summary["no_result"] += 1
                summary["results"].append({
                    "id": pick["id"], "status": "no_result",
                    "event": pick["event_name"], "market": pick["market"],
                })
                continue
            outcome = advisor.settle_tracking_pick(
                event_name=pick["event_name"],
                market=pick["market"],
                winning_selection=winner,
            )
            if outcome.get("status") == "no_pending_pick":
                # Race between two settle attempts — someone else got it.
                summary["results"].append({
                    "id": pick["id"], "status": "race_lost",
                })
                continue
            summary["settled"] += 1
            summary["results"].append({
                "id": pick["id"],
                "status": "settled",
                "outcome": outcome.get("outcome"),
                "paper_pl": outcome.get("paper_pl"),
                "winner": winner,
            })
        except Exception as e:
            logger.exception("settle failed for pick %s: %s", pick.get("id"), e)
            summary["errors"] += 1
            summary["results"].append({
                "id": pick.get("id"), "status": "error", "error": str(e),
            })
    return summary


# ── loop wrapper ────────────────────────────────────────────────────────────

class PickSettlerLoop:
    """Background thread that runs settle_once every POLL_SECONDS.

    Takes an ``advisor_getter`` callable rather than the advisor
    itself, because the advisor is instantiated late in
    dmai_core_complete's boot sequence (after the loop starts). The
    getter lets the loop pick up the advisor as soon as it's alive
    and skip iterations before then.
    """

    def __init__(self, advisor_getter: Callable[[], Any],
                 poll_seconds: int = POLL_SECONDS,
                 fetcher: Optional[Callable[..., Any]] = None):
        self._get_advisor = advisor_getter
        self._poll     = int(poll_seconds)
        self._fetcher  = fetcher
        self._thread: Optional[threading.Thread] = None
        self._stop     = threading.Event()
        self.last_summary: Dict[str, Any] = {}

    def _run(self) -> None:
        while not self._stop.wait(self._poll):
            try:
                advisor = self._get_advisor()
                if advisor is None:
                    self.last_summary = {"status": "advisor_not_ready",
                                          "ts": _now_iso()}
                    continue
                self.last_summary = settle_once(advisor, fetcher=self._fetcher)
            except Exception as e:
                logger.exception("pick_settler loop pass failed: %s", e)
                self.last_summary = {"error": str(e), "ts": _now_iso()}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="pick_settler", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


_LOOP: Optional[PickSettlerLoop] = None


def start_settler_loop(advisor_getter: Callable[[], Any],
                       poll_seconds: int = POLL_SECONDS,
                       fetcher: Optional[Callable[..., Any]] = None
                       ) -> PickSettlerLoop:
    """Idempotent boot hook. Uses the alive-check pattern (matches the
    fresh_blood respawn-guard fix from PR F1 and the insight/capability
    promoter loops) so worker fork() can't leave us with a dead thread.
    """
    global _LOOP
    if _LOOP is not None and _LOOP._thread and _LOOP._thread.is_alive():
        return _LOOP
    _LOOP = PickSettlerLoop(advisor_getter, poll_seconds=poll_seconds,
                            fetcher=fetcher)
    _LOOP.start()
    return _LOOP


def get_settler_loop() -> Optional[PickSettlerLoop]:
    return _LOOP
