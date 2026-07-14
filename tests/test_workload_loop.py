"""Tests for components.workload.workload_loop."""
from __future__ import annotations

import time

import pytest

from components.workload import workload_loop as wl
from components.workload import workload_profiler as wp


def test_start_workload_loop_is_idempotent(tmp_path):
    wl._LOOP = None
    p = str(tmp_path / "w.db")
    l1 = wl.start_workload_loop(workload_db_path=p, poll_seconds=60)
    l2 = wl.start_workload_loop(workload_db_path=p, poll_seconds=60)
    try:
        assert l1 is l2
    finally:
        l1.stop()
        wl._LOOP = None


def test_start_workload_loop_samples_immediately(tmp_path):
    wl._LOOP = None
    p = str(tmp_path / "w.db")
    loop = wl.start_workload_loop(workload_db_path=p, poll_seconds=1)
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not loop.last_summary:
            time.sleep(0.1)
        assert loop.last_summary, "loop never sampled"
        assert wp.get_status(p)["sample_count"] >= 1
    finally:
        loop.stop()
        wl._LOOP = None
