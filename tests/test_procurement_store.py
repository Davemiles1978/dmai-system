"""Tests for components.procurement.store (PR K)."""
from __future__ import annotations

import sqlite3

import pytest

from components.procurement.store import ProcurementStore


@pytest.fixture
def store(tmp_path):
    s = ProcurementStore(str(tmp_path / "proc.db"))
    s.init_db()
    return s


def _catalog_row():
    return {
        "source": "serve_the_home", "url": "http://x", "name": "Box A",
        "cpu": "CPU X", "cpu_passmark": 20000, "tdp_w": 45.0, "idle_w": 12.0,
        "ram_gb": 32, "storage_gb": 1000, "price_gbp": 500.0,
        "currency_orig": "GBP", "price_orig": 500.0, "fx_used": 1.0,
        "raw_json": {"foo": "bar"},
    }


def test_schema_creation(store):
    with sqlite3.connect(store.db_path) as c:
        names = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "hardware_catalog" in names
    assert "procurement_shortlist" in names
    assert "procurement_state" in names


def test_insert_catalog_row(store):
    hid = store.insert_catalog(_catalog_row())
    assert isinstance(hid, int) and hid > 0
    rows = store.list_catalog()
    assert len(rows) == 1
    assert rows[0]["name"] == "Box A"
    assert rows[0]["price_gbp"] == 500.0


def test_insert_shortlist_row(store):
    hid = store.insert_catalog(_catalog_row())
    sid = store.insert_shortlist_row({
        "run_ts": "2026-07-14T00:00:00+00:00", "rank": 1,
        "hardware_id": hid, "tco_gbp_3yr": 570.96, "capex_gbp": 500.0,
        "opex_3yr_gbp": 70.96, "headroom_ram_x": 2.0, "headroom_cpu_x": 1.2,
        "verdict": "affordable", "notes": "n",
    })
    assert isinstance(sid, int) and sid > 0


def test_cascade_query_joins_catalog(store):
    hid = store.insert_catalog(_catalog_row())
    store.insert_shortlist_row({
        "run_ts": "2026-07-14T00:00:00+00:00", "rank": 1,
        "hardware_id": hid, "tco_gbp_3yr": 570.96, "capex_gbp": 500.0,
        "opex_3yr_gbp": 70.96, "headroom_ram_x": 2.0, "headroom_cpu_x": 1.2,
        "verdict": "affordable", "notes": "n",
    })
    joined = store.get_shortlist()
    assert len(joined) == 1
    row = joined[0]
    # Shortlist columns + joined catalog columns are both present.
    assert row["verdict"] == "affordable"
    assert row["hw_name"] == "Box A"
    assert row["hw_cpu_passmark"] == 20000
