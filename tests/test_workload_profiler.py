"""Tests for components.workload.workload_profiler.

Covers:
- init stamps install_ts + creates schema idempotently
- sample_now writes a row and updates last_sample_ts
- get_recent / get_latest shape
- daily rollup buckets by UTC calendar day
- get_db_growth computes MB delta and per-day rate
- graceful degradation when psutil is unavailable
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from components.workload import workload_profiler as wp


@pytest.fixture()
def db(tmp_path):
    return str(tmp_path / "workload.db")


# ── init ──────────────────────────────────────────────────────────────────

def test_init_creates_schema_and_state(db):
    state = wp.init_workload_db(db)
    assert state["install_ts"]
    # tables exist
    with sqlite3.connect(db) as c:
        cur = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('workload_samples','workload_state')"
        )
        names = sorted(r[0] for r in cur.fetchall())
    assert names == ["workload_samples", "workload_state"]


def test_init_is_idempotent(db):
    s1 = wp.init_workload_db(db)
    s2 = wp.init_workload_db(db)
    assert s1 == s2


# ── sampling ──────────────────────────────────────────────────────────────

def test_sample_now_writes_row(db):
    wp.init_workload_db(db)
    s = wp.sample_now(db)
    assert s.ts
    assert wp.get_install_ts(db)
    latest = wp.get_latest(db)
    assert latest is not None
    assert latest["ts"] == s.ts
    # counters are either numeric or None (never string)
    for k in ("cpu_seconds_total", "mem_rss_mb",
              "disk_read_mb_total", "net_sent_mb_total"):
        assert latest[k] is None or isinstance(latest[k], (int, float))


def test_sample_now_updates_last_sample_ts(db):
    wp.init_workload_db(db)
    s = wp.sample_now(db)
    with sqlite3.connect(db) as c:
        r = c.execute(
            "SELECT value FROM workload_state WHERE key = 'last_sample_ts'"
        ).fetchone()
    assert r is not None
    assert r[0] == s.ts


def test_multiple_samples_accumulate(db):
    wp.init_workload_db(db)
    for _ in range(5):
        wp.sample_now(db)
    status = wp.get_status(db)
    assert status["sample_count"] == 5


# ── recent / rollup ───────────────────────────────────────────────────────

def _inject(db, ts_iso, **fields):
    """Force-write a specific timestamp for rollup testing."""
    with sqlite3.connect(db) as c:
        cols = ["ts"] + list(fields.keys())
        vals = [ts_iso] + list(fields.values())
        placeholders = ",".join("?" * len(cols))
        c.execute(
            f"INSERT INTO workload_samples({','.join(cols)}) "
            f"VALUES ({placeholders})", vals,
        )
        c.commit()


def test_get_recent_orders_ascending_and_filters_window(db):
    wp.init_workload_db(db)
    now = datetime.now(timezone.utc)
    _inject(db, (now - timedelta(hours=48)).isoformat(), mem_rss_mb=100)
    _inject(db, (now - timedelta(hours=1)).isoformat(),  mem_rss_mb=110)
    _inject(db, (now - timedelta(hours=0.1)).isoformat(), mem_rss_mb=115)

    rows = wp.get_recent(hours=24, db_path=db)
    assert len(rows) == 2
    assert rows[0]["mem_rss_mb"] == 110
    assert rows[1]["mem_rss_mb"] == 115


def test_daily_rollup_buckets_by_utc_day(db):
    wp.init_workload_db(db)
    now = datetime.now(timezone.utc)
    day = now.date().isoformat()
    # 3 samples same day
    _inject(db, f"{day}T00:10:00+00:00",
            cpu_percent=10.0, mem_rss_mb=100.0,
            cpu_seconds_total=100.0)
    _inject(db, f"{day}T12:00:00+00:00",
            cpu_percent=30.0, mem_rss_mb=120.0,
            cpu_seconds_total=250.0)
    _inject(db, f"{day}T23:50:00+00:00",
            cpu_percent=20.0, mem_rss_mb=110.0,
            cpu_seconds_total=400.0)

    rollup = wp.get_daily_rollup(days=1, db_path=db)
    assert len(rollup) == 1
    r = rollup[0]
    assert r["day"] == day
    assert r["samples"] == 3
    assert r["avg_cpu_percent"] == pytest.approx(20.0)
    assert r["peak_cpu_percent"] == 30.0
    assert r["peak_rss_mb"] == 120.0
    assert r["cpu_seconds_delta"] == pytest.approx(300.0)


def test_get_db_growth_computes_delta_and_rate(db):
    wp.init_workload_db(db)
    now = datetime.now(timezone.utc)
    # 6 days ago rather than 7, so we're safely inside the window
    # (7-day cutoff computed at call time drifts by microseconds).
    _inject(db, (now - timedelta(days=6)).isoformat(),
            knowledge_db_mb=100.0, ledger_db_mb=5.0,
            treasury_db_mb=0.1, workload_db_mb=0.01)
    _inject(db, now.isoformat(),
            knowledge_db_mb=142.0, ledger_db_mb=6.5,
            treasury_db_mb=0.15, workload_db_mb=0.05)

    g = wp.get_db_growth(days=7, db_path=db)
    assert g["window_days"] == 7
    kg = g["growth"]["knowledge_db"]
    assert kg["delta_mb"] == pytest.approx(42.0)
    # mb_per_day = delta / days_window (7), not / spanned_days.
    assert kg["mb_per_day"] == pytest.approx(6.0)
    # Small growth also captured
    tg = g["growth"]["treasury_db"]
    assert tg["delta_mb"] == pytest.approx(0.05, abs=1e-4)


def test_get_status_shape(db):
    wp.init_workload_db(db)
    wp.sample_now(db)
    st = wp.get_status(db)
    assert st["sample_count"] == 1
    assert st["latest"] is not None
    assert isinstance(st["rollup_24h"], list)
    assert isinstance(st["rollup_7d"], list)
    assert isinstance(st["db_growth_7d"], dict)


# ── graceful degradation ─────────────────────────────────────────────────

def test_sample_survives_without_psutil(db, monkeypatch):
    monkeypatch.setattr(wp, "psutil", None)
    wp.init_workload_db(db)
    s = wp.sample_now(db)
    # All process-scoped fields are None but the sample still writes
    assert s.cpu_percent is None
    assert s.mem_rss_mb is None
    assert wp.get_latest(db) is not None
