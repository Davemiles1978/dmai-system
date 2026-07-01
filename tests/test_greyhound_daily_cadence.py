"""Tests for the greyhound daily-tip cadence (PR #162).

The runner used to poll per-race every few minutes, writing a notifier row +
a tracking-pick row synchronously under the write mutex for every candidate.
Post-#161 that path was 52/70 of all write_mutex_timeouts. PR #162 rate-limits
the runner to ONE tip per day at 08:00 Europe/London: a single global-best
candidate is selected across all races, and exactly one tracking-pick +
at most one manual-bet tip is emitted per fire.

These are unit-level tests of the scheduling helpers and the single-fire
write contract — they do not hit the network or exercise real lock contention.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.monetisation import greyhound_runner as gr
from components.monetisation.greyhound_runner import GreyhoundRunner
from components.monetisation.betting_advisor import BettingAdvisor

LONDON = ZoneInfo("Europe/London")


def _runner(advisor, tmp_path):
    return GreyhoundRunner(
        advisor,
        daily_tip_hour=8,
        day_marker_path=str(tmp_path / "greyhound_last_tip_date.txt"),
    )


def _patch_scan(monkeypatch, races, runners_per_race):
    """Stub the network-facing scan so run_once sees deterministic candidates.

    ``races`` is a list of race dicts; every race resolves to the same
    ``runners_per_race`` list. Odds are fixed (no softmax) and settlement is
    a no-op so run_once stays offline.
    """
    monkeypatch.setattr(gr, "_list_meetings_today", lambda: races)
    monkeypatch.setattr(gr, "_ua_get", lambda url, timeout=20: "<html>")
    monkeypatch.setattr(gr, "_parse_race", lambda html: [dict(r) for r in runners_per_race])

    def _fixed_odds(runners, temperature=6.0):
        for r in runners:
            r.setdefault("implied_decimal_odds", 5.0)
            r.setdefault("implied_probability", 0.2)

    monkeypatch.setattr(gr, "_attach_implied_odds", _fixed_odds)
    monkeypatch.setattr(gr, "_fetch_gbgb_results", lambda date_ymd: [])
    monkeypatch.setattr(gr.time, "sleep", lambda s: None)


def _future_race(track):
    # race_date far in the future so run_once never skips it as "already off".
    return {"track": track, "off_hhmm": "23:59", "race_date": "2099-01-01",
            "race_id": "1", "url": f"https://x/{track}"}


# ─────────────────────────────────────────────────────────────────────────────
# _next_fire_at
# ─────────────────────────────────────────────────────────────────────────────

def test_next_fire_at_returns_today_when_before_fire_hour(tmp_path):
    r = _runner(MagicMock(), tmp_path)
    now = datetime(2026, 7, 1, 6, 30, tzinfo=LONDON)  # 06:30 < 08:00
    nxt = r._next_fire_at(now)
    assert nxt == datetime(2026, 7, 1, 8, 0, tzinfo=LONDON)


def test_next_fire_at_returns_tomorrow_when_after_fire_hour(tmp_path):
    r = _runner(MagicMock(), tmp_path)
    now = datetime(2026, 7, 1, 8, 0, tzinfo=LONDON)  # exactly 08:00 -> tomorrow
    nxt = r._next_fire_at(now)
    assert nxt == datetime(2026, 7, 2, 8, 0, tzinfo=LONDON)

    later = datetime(2026, 7, 1, 15, 0, tzinfo=LONDON)  # 15:00 -> tomorrow
    assert r._next_fire_at(later) == datetime(2026, 7, 2, 8, 0, tzinfo=LONDON)


def test_fire_hour_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("GREYHOUND_DAILY_TIP_HOUR_UK", "6")
    r = GreyhoundRunner(MagicMock(), day_marker_path=str(tmp_path / "m.txt"))
    assert r.daily_tip_hour == 6
    now = datetime(2026, 7, 1, 5, 0, tzinfo=LONDON)
    assert r._next_fire_at(now) == datetime(2026, 7, 1, 6, 0, tzinfo=LONDON)


# ─────────────────────────────────────────────────────────────────────────────
# day-marker double-fire guard (survives process restart)
# ─────────────────────────────────────────────────────────────────────────────

def test_day_marker_prevents_double_fire_across_restart(tmp_path):
    marker = str(tmp_path / "greyhound_last_tip_date.txt")
    now = datetime(2026, 7, 1, 10, 0, tzinfo=LONDON)  # after 08:00

    r1 = GreyhoundRunner(MagicMock(), daily_tip_hour=8, day_marker_path=marker)
    assert r1._due_to_fire(now) is True  # nothing recorded yet -> fire

    r1._mark_tipped(now.date().isoformat())
    assert r1._due_to_fire(now) is False  # already fired today -> no re-fire

    # Simulate a process restart: a brand-new instance reads the same marker.
    r2 = GreyhoundRunner(MagicMock(), daily_tip_hour=8, day_marker_path=marker)
    assert r2._already_tipped_today(now) is True
    assert r2._due_to_fire(now) is False

    # A new calendar day clears the guard.
    tomorrow = datetime(2026, 7, 2, 10, 0, tzinfo=LONDON)
    assert r2._due_to_fire(tomorrow) is True


def test_not_due_before_fire_hour(tmp_path):
    r = _runner(MagicMock(), tmp_path)
    before = datetime(2026, 7, 1, 3, 0, tzinfo=LONDON)  # 03:00 < 08:00
    assert r._due_to_fire(before) is False


# ─────────────────────────────────────────────────────────────────────────────
# single-fire write contract
# ─────────────────────────────────────────────────────────────────────────────

def test_one_tracking_pick_per_fire_across_many_races(tmp_path, monkeypatch):
    """Even with several races/runners, run_once records exactly ONE pick."""
    advisor = MagicMock()

    def fake_analyse(**kw):
        # Vary EV by selection so there is a distinct global best; not EV-gated
        # so the generate_tip path is skipped and we isolate record_tracking_pick.
        ev = {"Dog A": 0.05, "Dog B": 0.30, "Dog C": 0.10}.get(kw["selection"], 0.0)
        return {
            "event_name": kw["event_name"], "market": kw["market"],
            "selection": kw["selection"], "decimal_odds": kw["decimal_odds"],
            "model_probability": 0.4, "confidence": 0.6, "expected_value": ev,
            "passes_ev_gate": False, "rationale": "r", "prediction_id": "p",
        }

    advisor.analyse_candidate.side_effect = fake_analyse

    races = [_future_race(t) for t in ("Romford", "Hove", "Nottingham")]
    runners = [
        {"trap": 1, "master_rating": 80.0, "dog": "Dog A"},
        {"trap": 2, "master_rating": 82.0, "dog": "Dog B"},
        {"trap": 3, "master_rating": 78.0, "dog": "Dog C"},
    ]
    _patch_scan(monkeypatch, races, runners)

    r = _runner(advisor, tmp_path)
    summary = r.run_once()

    assert advisor.record_tracking_pick.call_count == 1
    # The single pick must be the global-best-EV selection.
    kwargs = advisor.record_tracking_pick.call_args.kwargs
    assert kwargs["selection"] == "Dog B"
    assert summary["tips_generated"] == 0  # nothing passed the EV gate


def test_one_notifier_record_per_fire(tmp_path, monkeypatch):
    """A full fire routes exactly one notification through notifier._record.

    Uses a real BettingAdvisor + a spy notifier. The day's best candidate
    passes the EV gate (one generate_tip -> one hot_tip -> one _record) but is
    below STRONG threshold, so record_tracking_pick does not also notify.
    """
    advisor = BettingAdvisor(db_path=str(tmp_path / "kdb.db"))

    class SpyNotifier:
        def __init__(self):
            self._record = MagicMock()

        def hot_tip(self, tip):
            # Mirror SlackNotifier.send -> _record (1 hot_tip == 1 _record).
            self._record("tip", tip.get("title", ""), tip)

    spy = SpyNotifier()
    advisor.notifier = spy

    def fake_analyse(**kw):
        # passes_ev_gate True but NOT strong (conf < 0.70) so only generate_tip
        # notifies, never record_tracking_pick.
        return {
            "event_name": kw["event_name"], "market": kw["market"],
            "selection": kw["selection"], "bookmaker": kw.get("bookmaker", ""),
            "decimal_odds": kw["decimal_odds"], "model_probability": 0.45,
            "confidence": 0.60, "expected_value": 0.12, "kelly_fraction": 0.08,
            "recommended_stake": 5.0, "bankroll": 100.0, "passes_ev_gate": True,
            "rationale": "edge", "prediction_id": "pid-1",
        }

    monkeypatch.setattr(advisor, "analyse_candidate", fake_analyse)

    races = [_future_race(t) for t in ("Romford", "Hove")]
    runners = [
        {"trap": 1, "master_rating": 80.0, "dog": "Dog A"},
        {"trap": 2, "master_rating": 82.0, "dog": "Dog B"},
    ]
    _patch_scan(monkeypatch, races, runners)

    summary = _runner(advisor, tmp_path).run_once()

    assert spy._record.call_count == 1
    assert summary["tips_generated"] == 1
    assert summary["tracking_picks_recorded"] == 1


def test_one_generate_tip_call_per_fire(tmp_path, monkeypatch):
    """run_once calls advisor.generate_tip at most once per fire."""
    advisor = MagicMock()

    def fake_analyse(**kw):
        ev = {"Dog A": 0.10, "Dog B": 0.30}.get(kw["selection"], 0.0)
        return {
            "event_name": kw["event_name"], "market": kw["market"],
            "selection": kw["selection"], "decimal_odds": kw["decimal_odds"],
            "model_probability": 0.5, "confidence": 0.8, "expected_value": ev,
            "passes_ev_gate": True, "rationale": "r", "prediction_id": "p",
        }

    advisor.analyse_candidate.side_effect = fake_analyse
    advisor.generate_tip.return_value = {"status": "pending", "id": "tip1"}
    advisor.record_tracking_pick.return_value = {"status": "tracked"}
    # No prior tip exists -> _already_tipped must return False.
    advisor._conn.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = None

    races = [_future_race(t) for t in ("Romford", "Hove", "Sheffield")]
    runners = [
        {"trap": 1, "master_rating": 80.0, "dog": "Dog A"},
        {"trap": 2, "master_rating": 82.0, "dog": "Dog B"},
    ]
    _patch_scan(monkeypatch, races, runners)

    r = _runner(advisor, tmp_path)
    summary = r.run_once()

    assert advisor.generate_tip.call_count == 1
    assert advisor.record_tracking_pick.call_count == 1
    assert advisor.generate_tip.call_args.kwargs["selection"] == "Dog B"
    assert summary["tips_generated"] == 1
