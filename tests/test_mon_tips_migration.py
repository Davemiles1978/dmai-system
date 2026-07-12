"""Tests for the mon_tips column-migration that fixes the prod
``no such column: t.recommended_stake`` bug on
``/api/monetisation/bets/performance``.

Prod's ``mon_tips`` table pre-dates the Kelly-sizing columns. The
advisor now runs a non-destructive ALTER TABLE ADD COLUMN pass on
init that only adds columns which don't already exist.
"""

from __future__ import annotations

import sqlite3
import pytest

from components.monetisation.betting_advisor import BettingAdvisor
from components.monetisation.revenue_allocator import RevenueAllocator


def _make_advisor(tmp_path, monkeypatch):
    monkeypatch.setenv("DMAI_DATA_DIR", str(tmp_path))
    db = str(tmp_path / "monetisation.db")
    alloc = RevenueAllocator(db_path=str(tmp_path / "alloc.db"))
    return BettingAdvisor(db_path=db, allocator=alloc), db


def test_migration_adds_missing_columns_on_legacy_schema(tmp_path, monkeypatch):
    """Simulate the prod schema by dropping columns after creation
    and then re-instantiating the advisor. The migration should add
    all missing columns idempotently.
    """
    ad, db = _make_advisor(tmp_path, monkeypatch)
    # Drop a subset of columns via table rebuild to simulate legacy prod.
    with sqlite3.connect(db) as c:
        c.execute("ALTER TABLE mon_tips RENAME TO mon_tips_new")
        c.execute("""
            CREATE TABLE mon_tips (
                id           TEXT PRIMARY KEY,
                event_name   TEXT NOT NULL,
                market       TEXT NOT NULL,
                selection    TEXT NOT NULL,
                decimal_odds REAL NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                created_at   REAL
            )
        """)
        c.execute("DROP TABLE mon_tips_new")
        c.commit()

    # Instantiating a fresh advisor over the same DB re-runs _init_schema
    # which calls _migrate_mon_tips_columns.
    RevenueAllocator(db_path=str(tmp_path / "alloc2.db"))
    ad2, _ = _make_advisor(tmp_path, monkeypatch)

    with sqlite3.connect(db) as c:
        cols = {row[1] for row in c.execute("PRAGMA table_info(mon_tips)").fetchall()}

    assert "recommended_stake" in cols, "migration didn't add recommended_stake"
    assert "kelly_fraction"    in cols
    assert "profit_loss"       in cols
    assert "settled_at"        in cols


def test_migration_idempotent(tmp_path, monkeypatch):
    """Running the migration twice on the same DB shouldn't error."""
    ad, db = _make_advisor(tmp_path, monkeypatch)
    # Second init should be a no-op - all columns already exist.
    ad._init_schema()
    ad._init_schema()
    with sqlite3.connect(db) as c:
        cols = {row[1] for row in c.execute("PRAGMA table_info(mon_tips)").fetchall()}
    assert "recommended_stake" in cols
    # No duplicate columns (obviously) - count matches unique set.
    with sqlite3.connect(db) as c:
        raw = c.execute("PRAGMA table_info(mon_tips)").fetchall()
    assert len(raw) == len(cols)


def test_recommended_stake_query_works_after_migration(tmp_path, monkeypatch):
    """The actual query pattern that was blowing up in prod -
    SELECT t.recommended_stake FROM mon_tips t - should now succeed
    even if the table pre-dates the Kelly column.
    """
    ad, db = _make_advisor(tmp_path, monkeypatch)
    with sqlite3.connect(db) as c:
        # This SELECT would have raised OperationalError pre-migration.
        rows = c.execute(
            "SELECT t.recommended_stake FROM mon_tips t LIMIT 1"
        ).fetchall()
    # Empty is fine - the assertion is that it didn't raise.
    assert rows == []
