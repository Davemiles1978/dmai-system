"""Racecard runner — pulls UK & IRE racecards from theracingapi.com and feeds
each runner through the BettingAdvisor.generate_tip() pipeline.

Endpoints used (Standard plan):
  GET /v1/racecards/standard         — today's racecards with 20+ bookmaker odds
  GET /v1/results?start_date=YYYY-MM-DD — settle previously-issued tips

Auth: HTTP Basic with username + password from env vars
  THERACINGAPI_USERNAME
  THERACINGAPI_PASSWORD

Rate limit: 2 req/s (we make at most 2 racecards calls and ~1 results call per cycle)

Behaviour:
  - Every cycle (default 300s) fetch today's standard racecards.
  - For each runner: take the best (highest) decimal_odds from any bookmaker
    (we want the best price available to the punter).
  - Skip runners where we already issued a tip in mon_tips today (dedupe).
  - Call advisor.generate_tip() — if it passes the EV gate it persists; else
    we just log the reject.
  - Settlement: every cycle also call /v1/results for today and yesterday,
    match by event_name + selection (case-insensitive trim), and call
    advisor.settle() for any pending tip whose race has finished.
"""

from __future__ import annotations

import logging
import os
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

BASE_URL = "https://api.theracingapi.com"
USER_AGENT = "DMAI-Tipster/1.0 (+https://dmai-web.onrender.com)"


def _creds() -> Optional[HTTPBasicAuth]:
    u = os.environ.get("THERACINGAPI_USERNAME", "").strip()
    p = os.environ.get("THERACINGAPI_PASSWORD", "").strip()
    if not (u and p):
        return None
    return HTTPBasicAuth(u, p)


def _get(path: str, params: Optional[Dict[str, Any]] = None, *, timeout: int = 20) -> Optional[Any]:
    auth = _creds()
    if not auth:
        logger.warning("racecard_runner: THERACINGAPI_USERNAME/PASSWORD not set; skipping")
        return None
    url = f"{BASE_URL}{path}"
    try:
        r = requests.get(url, auth=auth, params=params or {},
                         headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                         timeout=timeout)
        if r.status_code == 401:
            logger.error("racecard_runner: 401 from theracingapi — check credentials")
            return None
        if r.status_code == 429:
            logger.warning("racecard_runner: 429 rate-limited; backing off")
            time.sleep(2.0)
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("racecard_runner: GET %s failed: %s", path, e)
        return None


def _best_odds(runner: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return {'bookmaker': str, 'decimal': float} for the highest decimal price.

    Standard plan returns an 'odds' list per runner with entries like
    {"bookmaker": "Bet365", "fractional": "5/1", "decimal": "6.0", "ew_places": 4}.
    """
    best = None
    for o in runner.get("odds") or []:
        try:
            dec = float(o.get("decimal") or 0)
        except (TypeError, ValueError):
            continue
        if dec <= 1.01:
            continue
        if best is None or dec > best["decimal"]:
            best = {"bookmaker": o.get("bookmaker", ""), "decimal": dec}
    return best


def _event_label(race: Dict[str, Any]) -> str:
    course = race.get("course") or race.get("course_id") or "?"
    off = race.get("off_time") or race.get("off_dt") or ""
    name = race.get("race_name") or ""
    return f"{course} {off} {name}".strip()


class RacecardRunner:
    """Background loop that turns racecards into tips and settles them."""

    def __init__(self, advisor, *, interval_seconds: int = 300,
                 min_odds: float = 1.5, max_odds: float = 30.0,
                 region_filter: Optional[List[str]] = None):
        self.advisor = advisor
        self.interval = max(60, int(interval_seconds))
        self.min_odds = float(min_odds)
        self.max_odds = float(max_odds)
        self.region_filter = region_filter or ["gb", "ire"]
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_run_at: Optional[float] = None
        self.last_status: str = "idle"
        self.last_summary: Dict[str, Any] = {}

    # ---- public lifecycle ----

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="racecard-runner")
        self._thread.start()
        logger.info("RacecardRunner: started (interval=%ds)", self.interval)

    def stop(self):
        self._stop.set()

    def status(self) -> Dict[str, Any]:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "interval_seconds": self.interval,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "last_summary": self.last_summary,
            "credentials_set": _creds() is not None,
        }

    # ---- loop ----

    def _loop(self):
        # Initial delay so we don't slam on boot
        if self._stop.wait(20):
            return
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as e:
                logger.exception("RacecardRunner: cycle crashed: %s", e)
                self.last_status = f"error: {e}"
            if self._stop.wait(self.interval):
                break

    def run_once(self) -> Dict[str, Any]:
        """One full cycle: fetch racecards, generate tips, settle results."""
        summary = {"tips_generated": 0, "tips_rejected": 0, "settled": 0,
                   "races_seen": 0, "runners_seen": 0}

        # 1. Generate tips
        data = _get("/v1/racecards/standard")
        races = (data or {}).get("racecards") if isinstance(data, dict) else None
        if races is None:
            self.last_status = "no_racecards"
            self.last_run_at = time.time()
            self.last_summary = summary
            return summary

        for race in races:
            region = (race.get("region") or "").lower()
            if self.region_filter and region not in self.region_filter:
                continue
            summary["races_seen"] += 1
            event_name = _event_label(race)
            race_time = race.get("off_dt") or race.get("off_time") or ""
            for runner in race.get("runners") or []:
                if runner.get("non_runner"):
                    continue
                summary["runners_seen"] += 1
                horse = runner.get("horse") or runner.get("name") or ""
                if not horse:
                    continue
                best = _best_odds(runner)
                if not best or not (self.min_odds <= best["decimal"] <= self.max_odds):
                    continue
                if self._already_tipped(event_name, horse):
                    continue
                seed = self._build_seed(race, runner)
                try:
                    result = self.advisor.generate_tip(
                        event_name=event_name,
                        selection=horse,
                        decimal_odds=best["decimal"],
                        market="match_winner",
                        bookmaker=best["bookmaker"],
                        seed_data=seed,
                    )
                    if isinstance(result, dict) and result.get("status") == "pending":
                        summary["tips_generated"] += 1
                    else:
                        summary["tips_rejected"] += 1
                except Exception as e:
                    logger.warning("generate_tip failed for %s/%s: %s", event_name, horse, e)

        # 2. Settle finished races
        try:
            summary["settled"] = self._settle_completed()
        except Exception as e:
            logger.warning("settlement crashed: %s", e)

        self.last_status = "ok"
        self.last_run_at = time.time()
        self.last_summary = summary
        logger.info("RacecardRunner cycle: %s", summary)
        return summary

    # ---- helpers ----

    def _build_seed(self, race: Dict[str, Any], runner: Dict[str, Any]) -> str:
        bits = []
        if race.get("going"):
            bits.append(f"Going: {race['going']}")
        if race.get("distance"):
            bits.append(f"Distance: {race['distance']}")
        if race.get("race_class"):
            bits.append(f"Class: {race['race_class']}")
        if race.get("type"):
            bits.append(f"Type: {race['type']}")
        if race.get("prize"):
            bits.append(f"Prize: {race['prize']}")
        if runner.get("jockey"):
            bits.append(f"Jockey: {runner['jockey']}")
        if runner.get("trainer"):
            bits.append(f"Trainer: {runner['trainer']}")
        if runner.get("age"):
            bits.append(f"Age: {runner['age']}")
        if runner.get("weight_lbs") or runner.get("weight"):
            bits.append(f"Weight: {runner.get('weight_lbs') or runner.get('weight')}")
        if runner.get("form"):
            bits.append(f"Recent form: {runner['form']}")
        if runner.get("ofr"):
            bits.append(f"Official rating: {runner['ofr']}")
        return "\n".join(bits)

    def _already_tipped(self, event_name: str, selection: str) -> bool:
        try:
            with self.advisor._conn() as c:
                row = c.execute(
                    "SELECT 1 FROM mon_tips WHERE event_name=? AND selection=? "
                    "AND created_at > ? LIMIT 1",
                    (event_name, selection, time.time() - 86400),
                ).fetchone()
                return row is not None
        except Exception:
            return False

    def _settle_completed(self) -> int:
        """Pull today's results and settle any matching pending tips."""
        settled = 0
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        for d in (today, yesterday):
            data = _get("/v1/results", params={"start_date": d, "end_date": d})
            results = (data or {}).get("results") if isinstance(data, dict) else None
            if not results:
                continue
            for race in results:
                race_event = _event_label(race)
                for runner in race.get("runners") or []:
                    horse = runner.get("horse") or runner.get("name") or ""
                    position = str(runner.get("position") or "").strip()
                    if not horse or not position:
                        continue
                    tip_id = self._find_pending(race_event, horse)
                    if not tip_id:
                        continue
                    won = position == "1"
                    try:
                        odds = float(runner.get("sp_dec") or runner.get("sp") or 0)
                    except (TypeError, ValueError):
                        odds = 0.0
                    # Use the stake that's stored in the tip — settle() multiplies for win
                    try:
                        with self.advisor._conn() as c:
                            row = c.execute("SELECT actual_stake, recommended_stake, decimal_odds "
                                            "FROM mon_tips WHERE id=?", (tip_id,)).fetchone()
                        stake = (row["actual_stake"] if row and row["actual_stake"] is not None
                                 else (row["recommended_stake"] if row else 0)) or 0
                        if won:
                            return_amount = stake * (odds or row["decimal_odds"] or 0)
                            self.advisor.settle(tip_id, "won", actual_return=return_amount)
                        else:
                            self.advisor.settle(tip_id, "lost", actual_return=0.0)
                        settled += 1
                    except Exception as e:
                        logger.warning("settle failed for %s: %s", tip_id, e)
        return settled

    def _find_pending(self, event_name: str, selection: str) -> Optional[str]:
        try:
            with self.advisor._conn() as c:
                row = c.execute(
                    "SELECT id FROM mon_tips WHERE event_name=? AND selection=? "
                    "AND status='pending' LIMIT 1",
                    (event_name, selection),
                ).fetchone()
                return row["id"] if row else None
        except Exception:
            return None
