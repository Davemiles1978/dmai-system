"""Tests for pick_settler: closes the outcome loop on every model pick
inserted into mon_tracking_picks. No live OpticOdds calls in tests -
we inject a ``fetcher`` hook that returns whatever winner we like.
"""

from __future__ import annotations

import time
import uuid
import pytest

from components.monetisation.betting_advisor import BettingAdvisor
from components.monetisation import pick_settler
from components.monetisation.revenue_allocator import RevenueAllocator


@pytest.fixture
def advisor(tmp_path, monkeypatch):
    # Isolate everything on tmp_path so nothing touches shared DBs.
    monkeypatch.setenv("DMAI_DATA_DIR", str(tmp_path))
    db = str(tmp_path / "monetisation.db")
    alloc_db = str(tmp_path / "allocator.db")
    alloc = RevenueAllocator(db_path=alloc_db)
    return BettingAdvisor(db_path=db, allocator=alloc)


def _seed_pending_pick(advisor, *, event="race_1", market="win",
                       selection="Runner A", odds=3.0, created_at=None):
    """Directly insert a pending tracking pick and backdate it."""
    if created_at is None:
        created_at = time.time() - 3600  # 1h old, passes age gate
    pick_id = str(uuid.uuid4())
    with advisor._conn() as c:
        c.execute(
            "INSERT INTO mon_tracking_picks("
            "  id, event_name, market, selection, decimal_odds, "
            "  model_probability, confidence, expected_value, "
            "  outcome, created_at, rationale) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
            (pick_id, event, market, selection, odds, 0.5, 0.6, 0.5,
             created_at, "test rationale"),
        )
        c.commit()
    return pick_id


def test_settler_marks_winner_won(advisor):
    _seed_pending_pick(advisor, event="derby", selection="Runner A", odds=4.0)

    fetcher = lambda event_name, market: "Runner A"
    summary = pick_settler.settle_once(advisor, fetcher=fetcher)

    assert summary["settled"] == 1
    assert summary["errors"] == 0
    with advisor._conn() as c:
        row = c.execute(
            "SELECT outcome, paper_pl FROM mon_tracking_picks "
            "WHERE event_name='derby'"
        ).fetchone()
    assert row["outcome"] == "won"
    # paper_pl = odds - 1 = 3.0 for a 1-unit notional stake
    assert float(row["paper_pl"]) == pytest.approx(3.0)


def test_settler_marks_loser_lost(advisor):
    _seed_pending_pick(advisor, event="derby2", selection="Runner A", odds=2.5)

    fetcher = lambda event_name, market: "Runner B"
    summary = pick_settler.settle_once(advisor, fetcher=fetcher)

    assert summary["settled"] == 1
    with advisor._conn() as c:
        row = c.execute(
            "SELECT outcome, paper_pl FROM mon_tracking_picks "
            "WHERE event_name='derby2'"
        ).fetchone()
    assert row["outcome"] == "lost"
    assert float(row["paper_pl"]) == pytest.approx(-1.0)


def test_settler_skips_when_no_result(advisor):
    _seed_pending_pick(advisor, event="pending_event", selection="Runner A")

    fetcher = lambda event_name, market: None
    summary = pick_settler.settle_once(advisor, fetcher=fetcher)

    assert summary["settled"] == 0
    assert summary["no_result"] == 1
    with advisor._conn() as c:
        row = c.execute(
            "SELECT outcome FROM mon_tracking_picks WHERE event_name='pending_event'"
        ).fetchone()
    assert row["outcome"] == "pending"


def test_settler_respects_min_age(advisor):
    # created_at = now, too young, should be skipped
    _seed_pending_pick(advisor, event="new_race", selection="Runner A",
                       created_at=time.time())

    fetcher = lambda event_name, market: "Runner A"
    summary = pick_settler.settle_once(advisor, fetcher=fetcher)

    # No pick met the age gate.
    assert summary["checked"] == 0
    with advisor._conn() as c:
        row = c.execute(
            "SELECT outcome FROM mon_tracking_picks WHERE event_name='new_race'"
        ).fetchone()
    assert row["outcome"] == "pending"


def test_settler_idempotent(advisor):
    """Second pass finds no pending picks and returns settled=0."""
    _seed_pending_pick(advisor, event="derby3", selection="Runner A", odds=3.0)

    fetcher = lambda event_name, market: "Runner A"
    s1 = pick_settler.settle_once(advisor, fetcher=fetcher)
    s2 = pick_settler.settle_once(advisor, fetcher=fetcher)

    assert s1["settled"] == 1
    assert s2["settled"] == 0


def test_start_settler_loop_alive_check_pattern(monkeypatch):
    """Respawn-guard pattern: an already-alive thread is not duplicated."""
    monkeypatch.setattr(pick_settler, "settle_once",
                        lambda advisor, **kw: {"settled": 0})
    pick_settler._LOOP = None
    getter = lambda: object()
    l1 = pick_settler.start_settler_loop(advisor_getter=getter, poll_seconds=60)
    try:
        assert l1 is not None
        assert l1._thread.is_alive()
        l2 = pick_settler.start_settler_loop(advisor_getter=getter,
                                              poll_seconds=60)
        assert l2 is l1
    finally:
        l1.stop()
