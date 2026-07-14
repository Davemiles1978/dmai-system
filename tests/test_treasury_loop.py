"""Tests for components.treasury.treasury_loop."""
from __future__ import annotations

import time

import pytest

from components.treasury import treasury_ledger as tl
from components.treasury import treasury_loop as tloop


def test_start_treasury_loop_is_idempotent(tmp_path):
    tloop._LOOP = None  # reset module global
    tp = str(tmp_path / "t.db")
    lp = str(tmp_path / "l.db")

    loop1 = tloop.start_treasury_loop(
        treasury_db_path=tp, ledger_db_path=lp, poll_seconds=60,
    )
    loop2 = tloop.start_treasury_loop(
        treasury_db_path=tp, ledger_db_path=lp, poll_seconds=60,
    )
    try:
        assert loop1 is loop2
    finally:
        loop1.stop()
        tloop._LOOP = None


def test_start_treasury_loop_runs_sync_immediately(tmp_path):
    tloop._LOOP = None
    tp = str(tmp_path / "t.db")
    lp = str(tmp_path / "l.db")

    loop = tloop.start_treasury_loop(
        treasury_db_path=tp, ledger_db_path=lp, poll_seconds=1,
    )
    try:
        # Give the thread a moment to run one pass.
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline and not loop.last_summary:
            time.sleep(0.1)
        assert loop.last_summary, "loop never produced a summary"
        assert "balance_gbp" in loop.last_summary
    finally:
        loop.stop()
        tloop._LOOP = None
