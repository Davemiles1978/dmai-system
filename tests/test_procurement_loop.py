"""Tests for components.procurement.loop (PR K)."""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone

import pytest

from components.procurement import loop as ploop
from components.workload import workload_profiler as wp
from components.treasury import treasury_ledger as tl


def _seed(tmp_path):
    wpath = str(tmp_path / "wl.db")
    tpath = str(tmp_path / "tr.db")
    ppath = str(tmp_path / "proc.db")
    wp.init_workload_db(wpath)
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(wpath) as c:
        for cpu, rss in [(1000.0, 200.0), (1000.0 + 43200, 420.0)]:
            c.execute(
                "INSERT INTO workload_samples"
                "(ts, cpu_percent, cpu_seconds_total, mem_rss_mb, "
                " mem_peak_rss_mb) VALUES (?,?,?,?,?)",
                (now, 10.0, cpu, rss, rss),
            )
        c.commit()
    tl.init_treasury_db(tpath)
    tl.record_manual(kind="manual_credit", amount_gbp=800.0,
                     description="seed", db_path=tpath)
    return ppath, wpath, tpath


def _make_loop(tmp_path, **kw):
    ppath, wpath, tpath = _seed(tmp_path)
    return ploop.ProcurementLoop(
        procurement_db_path=ppath,
        workload_db_path=wpath,
        treasury_db_path=tpath,
        **kw,
    )


def test_loop_respects_6h_cadence(tmp_path):
    lp = _make_loop(tmp_path, poll_seconds=1,
                    run_interval_seconds=6 * 60 * 60)
    lp.force_run()  # first run stamps _last_run_monotonic
    first_run_ts = lp.last_summary["run_ts"]
    # Cadence gate: not due again for 6h.
    assert lp._due() is False
    # A scheduled _run pass should NOT produce a new run_ts.
    assert lp.last_summary["run_ts"] == first_run_ts


def test_force_run_bypasses_cadence(tmp_path):
    lp = _make_loop(tmp_path, poll_seconds=1,
                    run_interval_seconds=6 * 60 * 60)
    s1 = lp.force_run()
    assert lp._due() is False  # cadence says wait
    s2 = lp.force_run()        # force ignores it
    assert s1["run_ts"] != s2["run_ts"]


def test_graceful_degradation_if_parser_raises(tmp_path, monkeypatch):
    lp = _make_loop(tmp_path, poll_seconds=1)

    real_load = ploop.researcher.load_source_rows

    def flaky(module_path, html=""):
        if module_path.endswith("newegg_us"):
            raise RuntimeError("boom")
        return real_load(module_path, html)

    monkeypatch.setattr(ploop.researcher, "load_source_rows", flaky)
    summary = lp.force_run()
    # The run still succeeds; the broken source is recorded, not fatal.
    assert summary["ok"] is True
    assert "newegg_us" in summary["parser_errors"]
    assert summary["candidate_count"] >= 1
