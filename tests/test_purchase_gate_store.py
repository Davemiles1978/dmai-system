"""Store + state-machine tests for the purchase-approval gate (PR L)."""
from __future__ import annotations

import pytest

from components.purchase_gate.purchase_ledger import PurchaseGateStore


@pytest.fixture
def store(tmp_path):
    s = PurchaseGateStore(str(tmp_path / "pg.db"))
    s.init_db()
    return s


def _row(hardware_id=1, capex=500.0):
    return {
        "hardware_id":              hardware_id,
        "hw_name":                  "Box A",
        "hw_source":                "amazon_uk",
        "hw_url":                   "http://x/dp/B0ABCDE123",
        "capex_gbp":                capex,
        "tco_gbp_3yr":              700.0,
        "treasury_at_proposal_gbp": 900.0,
        "trigger_multiplier":       1.2,
    }


def test_insert_returns_full_row(store):
    p = store.insert_proposal(_row())
    assert p["id"] > 0
    assert p["state"] == "pending"
    assert p["hardware_id"] == 1
    assert p["capex_gbp"] == 500.0


def test_get_and_list_by_state(store):
    p1 = store.insert_proposal(_row(hardware_id=1))
    store.insert_proposal(_row(hardware_id=2))
    assert store.get_proposal(p1["id"])["id"] == p1["id"]
    assert store.get_proposal(9999) is None
    pending = store.list_proposals(state="pending")
    assert len(pending) == 2
    assert store.list_proposals(state="purchased") == []


def test_valid_transition_pending_to_approved(store):
    p = store.insert_proposal(_row())
    updated = store.transition_state(p["id"], "approved")
    assert updated["state"] == "approved"


def test_invalid_transition_rejected(store):
    p = store.insert_proposal(_row())
    # pending → purchased is not allowed (must go via approved).
    with pytest.raises(ValueError):
        store.transition_state(p["id"], "purchased")
    # Unknown id is rejected too.
    with pytest.raises(ValueError):
        store.transition_state(123456, "approved")


def test_dedupe_by_hardware_id(store):
    store.insert_proposal(_row(hardware_id=7))
    assert store.has_open_proposal(7) is True
    assert store.has_open_proposal(8) is False
    # Once terminal, it is no longer "open".
    p2 = store.insert_proposal(_row(hardware_id=8))
    store.transition_state(p2["id"], "declined")
    assert store.has_open_proposal(8) is False


def test_config_kv_round_trip_and_auto_checkout_defaults(store):
    # Defaults come from module constants when nothing is set.
    assert store.auto_checkout_enabled() is False
    assert store.auto_checkout_dry_run() is True
    assert store.auto_checkout_max_gbp() == 750.0
    # Overrides persist via config_kv.
    store.config_kv_set("auto_checkout_enabled", True)
    store.config_kv_set("auto_checkout_max_gbp", 250.0)
    assert store.auto_checkout_enabled() is True
    assert store.auto_checkout_max_gbp() == 250.0
    assert store.config_kv_get("missing", "fallback") == "fallback"
