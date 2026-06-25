"""Greyhound tipster runner — free, paper-mode by default.

Data flow:

  1. Today's racecards: timeform.com/greyhound-racing/racecards (HTML, free).
     We extract per-runner: trap, dog name, Timeform Master Rating, recent form,
     trainer name + 3-week strike rate.

  2. Implied odds: softmax the Master Ratings across the 6 traps in a race,
     then decimal_odds = 1 / probability. This is OUR model's fair price,
     which is exactly what the BettingAdvisor expects. (When you eventually
     subscribe to a real odds feed, swap this layer out — nothing else changes.)

  3. Settlement: api.gbgb.org.uk/api/results?date=YYYY-MM-DD (JSON, free).
     Match by (trackName, raceTime, dogName) and mark won/lost using the
     real SP field so paper P/L mirrors live bookmaker results.

Paper vs live:
  TIPSTER_LIVE=false (default)  → tips logged with notes='paper' and
                                  notifier prefixes [PAPER]
  TIPSTER_LIVE=true             → tips logged as 'live' and notifier
                                  posts "place this manually" alerts
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

TIMEFORM_BASE = "https://www.timeform.com"
GBGB_RESULTS = "https://api.gbgb.org.uk/api/results"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_RACE_HREF = re.compile(
    r'href="(/greyhound-racing/racecards/([^/]+)/(\d{3,4})/(\d{4}-\d{2}-\d{2})/(\d+))"'
)
# data-trap="N" data-mstr="MASTER_RATING"
_RUNNER_OPEN = re.compile(r'data-trap="(\d+)"\s+data-mstr="([\d.]+)">')
# rpb-greyhound-N>...DOG NAME...</a>
_DOG_NAME = re.compile(
    r'class="rpb-greyhound rpb-greyhound-(\d+)[^"]*"[^>]*>\s*([^<]+?)\s*</a>',
    re.S,
)
# Trainer span: <span title="The full name of the greyhounds trainer (and their strike rate in the last 3 weeks)">NAME  (X.XX%)</span>
_TRAINER = re.compile(
    r'title="The full name of the greyhounds trainer[^"]*">([^<]+?)</span>'
)
# Recent form span
_FORM = re.compile(
    r'title="The previous 5 finishing positions of this greyhound">([^<]+?)</span>'
)


def _ua_get(url: str, timeout: int = 20) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept": "text/html,*/*"},
                         timeout=timeout)
        if r.status_code != 200:
            logger.warning("greyhound_runner: %s -> HTTP %d", url, r.status_code)
            return None
        return r.text
    except Exception as e:
        logger.warning("greyhound_runner: GET %s failed: %s", url, e)
        return None


def _ua_get_json(url: str, timeout: int = 20) -> Optional[Any]:
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept": "application/json"},
                         timeout=timeout)
        if r.status_code != 200:
            logger.warning("greyhound_runner: %s -> HTTP %d", url, r.status_code)
            return None
        return r.json()
    except Exception as e:
        logger.warning("greyhound_runner: GET %s failed: %s", url, e)
        return None


def _softmax_prob(ratings: List[float], temperature: float = 6.0) -> List[float]:
    """Convert Master Ratings to win probabilities via softmax.

    temperature: smaller value -> sharper favourites. 6.0 is a reasonable
    starting point; tune later via backtesting.
    """
    if not ratings:
        return []
    m = max(ratings)
    exps = [math.exp((r - m) / max(0.01, temperature)) for r in ratings]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


# ─────────────────────────────────────────────────────────────────────────────
# Timeform parsing
# ─────────────────────────────────────────────────────────────────────────────

def _list_meetings_today() -> List[Dict[str, Any]]:
    """Find every race URL on the racecards index. Returns list of dicts:
    {url, track, off_hhmm, race_date, race_id}."""
    html = _ua_get(f"{TIMEFORM_BASE}/greyhound-racing/racecards")
    if not html:
        return []
    out = []
    seen = set()
    for m in _RACE_HREF.finditer(html):
        href, track, off, date, race_id = m.groups()
        if href in seen:
            continue
        seen.add(href)
        out.append({
            "url": f"{TIMEFORM_BASE}{href}",
            "track": track.replace("-", " ").title(),
            "off_hhmm": f"{off[:-2]}:{off[-2:]}" if len(off) >= 3 else off,
            "race_date": date,
            "race_id": race_id,
        })
    return out


def _parse_race(html: str) -> List[Dict[str, Any]]:
    """Return one dict per trap: trap, dog, master_rating, trainer, form."""
    runners: List[Dict[str, Any]] = []
    # Build base list from data-trap markers (in trap order)
    for m in _RUNNER_OPEN.finditer(html):
        trap = int(m.group(1))
        try:
            mstr = float(m.group(2))
        except ValueError:
            mstr = 0.0
        runners.append({"trap": trap, "master_rating": mstr})
    # Attach dog names by rpb-greyhound-N
    dogs = {int(g.group(1)): g.group(2).strip()
            for g in _DOG_NAME.finditer(html)}
    for r in runners:
        r["dog"] = dogs.get(r["trap"], "")
    # Trainers + forms appear in row order. Pull positionally.
    trainers = [m.group(1).strip() for m in _TRAINER.finditer(html)]
    forms = [m.group(1).strip() for m in _FORM.finditer(html)]
    for i, r in enumerate(runners):
        r["trainer"] = trainers[i] if i < len(trainers) else ""
        r["form"] = forms[i] if i < len(forms) else ""
    return [r for r in runners if r["dog"]]


def _attach_implied_odds(runners: List[Dict[str, Any]],
                        temperature: float = 6.0) -> None:
    if not runners:
        return
    ratings = [r["master_rating"] for r in runners]
    probs = _softmax_prob(ratings, temperature=temperature)
    for r, p in zip(runners, probs):
        r["implied_probability"] = round(p, 4)
        r["implied_decimal_odds"] = round(1.0 / max(0.01, p), 2)


# ─────────────────────────────────────────────────────────────────────────────
# GBGB settlement
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_gbgb_results(date_ymd: str) -> List[Dict[str, Any]]:
    """Return GBGB result rows for a date (one row per winner)."""
    out: List[Dict[str, Any]] = []
    page = 1
    while True:
        url = (f"{GBGB_RESULTS}?date={date_ymd}&page={page}"
               f"&itemsPerPage=200&race_type=race")
        data = _ua_get_json(url)
        if not data or "items" not in data:
            break
        items = data["items"] or []
        out.extend(items)
        meta = data.get("meta") or {}
        if page >= int(meta.get("pageCount", 1)):
            break
        page += 1
        time.sleep(0.5)  # polite pacing
    return out


def _fetch_gbgb_meeting(meeting_id: int) -> Optional[List[Dict[str, Any]]]:
    """Return ALL races at a meeting with every trap's finish position + SP."""
    url = f"{GBGB_RESULTS}/meeting/{meeting_id}"
    data = _ua_get_json(url)
    if isinstance(data, list):
        return data
    return None


def _sp_to_decimal(sp: str) -> Optional[float]:
    """Convert 'X/Y' fractional SP to decimal odds."""
    if not sp:
        return None
    sp = sp.strip().upper()
    if sp in ("EVS", "EVENS"):
        return 2.0
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", sp)
    if not m:
        try:
            return float(sp)
        except ValueError:
            return None
    n, d = int(m.group(1)), int(m.group(2))
    if d == 0:
        return None
    return round(n / d + 1.0, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class GreyhoundRunner:
    """Background loop: generate paper/live tips from Timeform, settle from GBGB."""

    def __init__(self, advisor, *, interval_seconds: int = 600,
                 min_odds: float = 1.5, max_odds: float = 25.0,
                 softmax_temperature: float = 6.0,
                 live: Optional[bool] = None):
        self.advisor = advisor
        self.interval = max(120, int(interval_seconds))
        self.min_odds = float(min_odds)
        self.max_odds = float(max_odds)
        self.softmax_temperature = float(softmax_temperature)
        if live is None:
            live = os.environ.get("TIPSTER_LIVE", "false").strip().lower() == "true"
        self.live = bool(live)
        # Tier detection — gates which features activate based on env keys present
        self._betfair_keys_present = bool(
            os.environ.get("BETFAIR_APP_KEY")
            and os.environ.get("BETFAIR_USERNAME")
            and os.environ.get("BETFAIR_PASSWORD")
        )
        self._bankroll_set = bool(os.environ.get("STAKING_BANKROLL"))
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
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="greyhound-runner")
        self._thread.start()
        logger.info("GreyhoundRunner: started (interval=%ds, mode=%s)",
                    self.interval, "LIVE" if self.live else "PAPER")

    def stop(self):
        self._stop.set()

    def status(self) -> Dict[str, Any]:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "interval_seconds": self.interval,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "last_summary": self.last_summary,
            "mode": "LIVE" if self.live else "PAPER",
            "tier": self.tier(),
            "min_odds": self.min_odds,
            "max_odds": self.max_odds,
            "softmax_temperature": self.softmax_temperature,
        }

    def tier(self) -> Dict[str, Any]:
        """Report which feature tier is active based on env keys + live flag.

        Tier 0 (paper): default. Tips generated from Timeform softmax odds,
            settled from GBGB results. No real money touched.
        Tier 1 (real odds): BETFAIR_APP_KEY/USERNAME/PASSWORD set. Runner pulls
            real Betfair Exchange prices and uses them as the EV reference.
        Tier 2 (live staking): TIPSTER_LIVE=true + Betfair keys + STAKING_BANKROLL.
            Runner is allowed to place real back bets on Betfair when EV>threshold.
        """
        if self.live and self._betfair_keys_present and self._bankroll_set:
            level, name = 2, "LIVE_STAKING"
        elif self._betfair_keys_present:
            level, name = 1, "REAL_ODDS"
        else:
            level, name = 0, "PAPER"
        return {
            "level": level,
            "name": name,
            "betfair_keys_present": self._betfair_keys_present,
            "bankroll_set": self._bankroll_set,
            "tipster_live_flag": self.live,
            "next_unlock": self._next_unlock(level),
        }

    @staticmethod
    def _next_unlock(level: int) -> Optional[str]:
        if level == 0:
            return "Set BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD to reach Tier 1 (real odds)."
        if level == 1:
            return "Set STAKING_BANKROLL and TIPSTER_LIVE=true to reach Tier 2 (live staking)."
        return None

    # ---- loop ----

    def _loop(self):
        if self._stop.wait(20):
            return
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as e:
                logger.exception("GreyhoundRunner: cycle crashed: %s", e)
                self.last_status = f"error: {e}"
            if self._stop.wait(self.interval):
                break

    def run_once(self) -> Dict[str, Any]:
        summary = {"meetings_seen": 0, "races_seen": 0, "runners_seen": 0,
                   "tips_generated": 0, "tips_rejected": 0, "settled": 0,
                   "mode": "LIVE" if self.live else "PAPER"}
        meetings = _list_meetings_today()
        summary["meetings_seen"] = len(meetings)
        for race in meetings:
            html = _ua_get(race["url"])
            if not html:
                continue
            summary["races_seen"] += 1
            runners = _parse_race(html)
            _attach_implied_odds(runners, temperature=self.softmax_temperature)
            event_name = self._event_name(race)
            race_off = self._race_off_dt(race)
            # Skip races that have already started
            if race_off and race_off < datetime.now(timezone.utc):
                continue
            for r in runners:
                summary["runners_seen"] += 1
                dec = r.get("implied_decimal_odds") or 0.0
                if not (self.min_odds <= dec <= self.max_odds):
                    continue
                if not r.get("dog"):
                    continue
                if self._already_tipped(event_name, r["dog"]):
                    continue
                seed = self._build_seed(race, r)
                try:
                    result = self.advisor.generate_tip(
                        event_name=event_name,
                        selection=r["dog"],
                        decimal_odds=dec,
                        market="trap_winner",
                        bookmaker="implied (timeform_mstr)",
                        seed_data=seed,
                    )
                    if isinstance(result, dict) and result.get("status") == "pending":
                        summary["tips_generated"] += 1
                        # Tag with mode in the notes field
                        self._tag_mode(result.get("id"))
                    else:
                        summary["tips_rejected"] += 1
                except Exception as e:
                    logger.warning("generate_tip failed for %s/%s: %s",
                                   event_name, r["dog"], e)
            time.sleep(0.4)  # be polite to timeform
        # Settlement pass — yesterday + today
        try:
            summary["settled"] = self._settle()
        except Exception as e:
            logger.warning("settlement crashed: %s", e)
        self.last_run_at = time.time()
        self.last_status = "ok"
        self.last_summary = summary
        logger.info("GreyhoundRunner cycle: %s", summary)
        return summary

    # ---- helpers ----

    def _event_name(self, race: Dict[str, Any]) -> str:
        return f"{race['track']} {race['off_hhmm']} ({race['race_date']})"

    def _race_off_dt(self, race: Dict[str, Any]) -> Optional[datetime]:
        try:
            return datetime.strptime(
                f"{race['race_date']} {race['off_hhmm']}",
                "%Y-%m-%d %H:%M",
            ).replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _build_seed(self, race: Dict[str, Any], runner: Dict[str, Any]) -> str:
        parts = [
            f"Track: {race['track']}",
            f"Off: {race['off_hhmm']}",
            f"Trap: {runner['trap']}",
            f"Timeform Master Rating: {runner['master_rating']}",
            f"Recent form: {runner.get('form', '')}",
            f"Trainer: {runner.get('trainer', '')}",
            f"Field implied probability: {runner.get('implied_probability', 0)}",
        ]
        return "\n".join(parts)

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

    def _tag_mode(self, tip_id: Optional[str]) -> None:
        if not tip_id:
            return
        tag = "live" if self.live else "paper"
        try:
            with self.advisor._conn() as c:
                c.execute("UPDATE mon_tips SET notes=? WHERE id=?", (tag, tip_id))
        except Exception:
            pass

    # ---- settlement ----

    def _settle(self) -> int:
        settled = 0
        for offset in (0, 1):
            d = (datetime.now(timezone.utc) - timedelta(days=offset)).strftime("%Y-%m-%d")
            results = _fetch_gbgb_results(d)
            if not results:
                continue
            # Group by meetingId so we fetch each meeting only once
            meeting_ids = {row["meetingId"] for row in results if row.get("meetingId")}
            for mid in meeting_ids:
                detail = _fetch_gbgb_meeting(mid)
                if not detail:
                    continue
                # detail is a list of meetings (usually one) each with races
                for meeting in detail:
                    track = meeting.get("trackName", "")
                    for race in meeting.get("races") or []:
                        settled += self._settle_one_race(track, race)
                time.sleep(0.4)
        return settled

    def _settle_one_race(self, track: str, race: Dict[str, Any]) -> int:
        try:
            t = (race.get("raceTime") or "")[:5]  # HH:MM
            d_ddmmyyyy = race.get("raceDate") or ""
            if "/" not in d_ddmmyyyy:
                return 0
            dd, mm, yyyy = d_ddmmyyyy.split("/")
            event_name = f"{track} {t} ({yyyy}-{mm}-{dd})"
        except Exception:
            return 0
        settled = 0
        for trap in race.get("traps") or []:
            dog = (trap.get("dogName") or "").strip()
            if not dog:
                continue
            tip_id = self._find_pending(event_name, dog)
            if not tip_id:
                continue
            won = (trap.get("resultPosition") == 1)
            sp_dec = _sp_to_decimal(trap.get("SP") or "") or 0.0
            try:
                with self.advisor._conn() as c:
                    row = c.execute(
                        "SELECT actual_stake, recommended_stake, decimal_odds "
                        "FROM mon_tips WHERE id=?", (tip_id,)
                    ).fetchone()
                stake = (row["actual_stake"] if row and row["actual_stake"] is not None
                         else (row["recommended_stake"] if row else 0)) or 0
                if won:
                    payout = stake * (sp_dec or row["decimal_odds"] or 0)
                    self.advisor.settle(tip_id, "won", actual_return=payout)
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
