"""Schema tests for the purchase-approval gate (PR L)."""
from __future__ import annotations

import sqlite3

import pytest

from components.purchase_gate.purchase_ledger import PurchaseGateStore


@pytest.fixture
def store(tmp_path):
    s = PurchaseGateStore(str(tmp_path / "pg.db"))
    s.init_db()
    return s


def test_tables_and_index_created(store):
    with sqlite3.connect(store.db_path) as c:
        tables = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        indexes = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
    assert "purchase_proposals" in tables
    assert "config_kv" in tables
    assert "idx_prop_state" in indexes


def test_install_ts_stamped_and_confirm_token_stable(store):
    ts = store.install_ts()
    assert ts  # non-empty ISO timestamp
    # Re-init must not re-stamp a new install_ts (token stays stable).
    token1 = store.confirm_token()
    store.init_db()
    assert store.install_ts() == ts
    assert store.confirm_token() == token1
