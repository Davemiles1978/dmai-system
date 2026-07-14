"""Tests for components.procurement.researcher (PR K)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from components.procurement import config as cfg
from components.procurement import researcher
from components.procurement.store import ProcurementStore
from components.treasury import treasury_ledger as tl
from components.workload import workload_profiler as wp


def _seed_workload(path, *, cpu_delta=43200.0, peak_rss=420.0):
    """Seed two workload samples so the 7d rollup yields a cpu_seconds
    delta and an RSS peak."""
    wp.init_workload_db(path)
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(path) as c:
        for cpu, rss in [(1000.0, peak_rss * 0.5),
                         (1000.0 + cpu_delta, peak_rss)]:
            c.execute(
                "INSERT INTO workload_samples"
                "(ts, cpu_percent, cpu_seconds_total, mem_rss_mb, "
                " mem_peak_rss_mb) VALUES (?,?,?,?,?)",
                (now, 10.0, cpu, rss, rss),
            )
        c.commit()


def _seed_treasury(path, balance):
    tl.init_treasury_db(path)
    if balance:
        tl.record_manual(kind="manual_credit", amount_gbp=float(balance),
                         description="seed", db_path=path)


@pytest.fixture
def paths(tmp_path):
    return {
        "procurement_db_path": str(tmp_path / "proc.db"),
        "workload_db_path":    str(tmp_path / "wl.db"),
        "treasury_db_path":    str(tmp_path / "tr.db"),
    }


def test_happy_path(paths):
    _seed_workload(paths["workload_db_path"])
    _seed_treasury(paths["treasury_db_path"], 800.0)
    summary = researcher.run_research(**paths)
    assert summary["ok"] is True
    assert summary["candidate_count"] >= 3
    sl = summary["shortlist"]
    assert len(sl) == 3
    # TCO strictly increasing with rank (lowest-TCO-first ranking).
    tcos = [row["tco_gbp_3yr"] for row in sl]
    assert tcos == sorted(tcos)
    assert tcos[0] < tcos[-1]


def test_skip_if_workload_empty(paths):
    # Init an empty workload DB (no samples).
    wp.init_workload_db(paths["workload_db_path"])
    _seed_treasury(paths["treasury_db_path"], 800.0)
    summary = researcher.run_research(**paths)
    assert summary["ok"] is False
    assert summary["skipped"] == "no_workload_data"
    assert summary["shortlist"] == []


def test_skip_if_no_candidates_pass_headroom(paths):
    # Enormous CPU demand -> required passmark far above any seed box.
    _seed_workload(paths["workload_db_path"],
                   cpu_delta=cfg.CPU_SECONDS_PER_CORE_DAY * 50,
                   peak_rss=420.0)
    _seed_treasury(paths["treasury_db_path"], 800.0)
    summary = researcher.run_research(**paths)
    assert summary["ok"] is True
    assert summary["skipped"] == "no_candidates_pass_headroom"
    assert summary["shortlist"] == []
    # Catalog was still populated (rows fetched, just none passed gates).
    assert summary["catalog_size"] > 0


def test_affordable_vs_aspirational_verdict(paths):
    _seed_workload(paths["workload_db_path"])
    # Tiny balance -> top-3 capex >> 1.5x balance -> aspirational.
    _seed_treasury(paths["treasury_db_path"], 10.0)
    store = ProcurementStore(paths["procurement_db_path"])
    researcher.run_research(**paths)
    verdicts = {r["verdict"] for r in store.get_shortlist()}
    assert "aspirational" in verdicts
    assert "affordable" not in verdicts

    # Large balance -> top-3 within 1.5x -> affordable.
    p2 = dict(paths)
    p2["procurement_db_path"] = paths["procurement_db_path"] + ".2"
    p2["treasury_db_path"] = paths["treasury_db_path"] + ".2"
    _seed_treasury(p2["treasury_db_path"], 5000.0)
    researcher.run_research(**p2)
    store2 = ProcurementStore(p2["procurement_db_path"])
    verdicts2 = {r["verdict"] for r in store2.get_shortlist()}
    assert "affordable" in verdicts2


def test_deterministic_ranking(paths):
    _seed_workload(paths["workload_db_path"])
    _seed_treasury(paths["treasury_db_path"], 800.0)
    s1 = researcher.run_research(**paths)
    p2 = dict(paths)
    p2["procurement_db_path"] = paths["procurement_db_path"] + ".2"
    s2 = researcher.run_research(**p2)
    names1 = [r["name"] for r in s1["shortlist"]]
    names2 = [r["name"] for r in s2["shortlist"]]
    assert names1 == names2


def test_fx_conversion_applied(paths):
    _seed_workload(paths["workload_db_path"])
    _seed_treasury(paths["treasury_db_path"], 800.0)
    researcher.run_research(**paths)
    store = ProcurementStore(paths["procurement_db_path"])
    usd_rows = [r for r in store.list_catalog()
                if r["currency_orig"] == "USD"]
    assert usd_rows, "expected at least one USD (Newegg) row"
    fx = cfg.fx_usd_gbp()
    for r in usd_rows:
        assert r["fx_used"] == fx
        # GBP price is the USD price scaled by FX (< original USD number).
        assert r["price_gbp"] == round(r["price_orig"] * fx, 2)
        assert r["price_gbp"] < r["price_orig"]
