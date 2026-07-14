"""Admin API + page tests for the purchase gate (PR L).

DATA_PATH is pointed at a temp dir *before* importing the app so the
purchase-gate and treasury DBs stay isolated.
"""
from __future__ import annotations

import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="pg_endpoints_")
os.environ["DATA_PATH"] = _TMP

from components.purchase_gate.purchase_ledger import (  # noqa: E402
    PurchaseGateStore,
)


def _seed(hardware_id=1, capex=500.0):
    store = PurchaseGateStore()
    store.init_db()
    return store.insert_proposal({
        "hardware_id":              hardware_id,
        "hw_name":                  "Box A",
        "hw_source":                "amazon_uk",
        "hw_url":                   "http://x/dp/B0ABCDE123",
        "capex_gbp":                capex,
        "tco_gbp_3yr":              700.0,
        "treasury_at_proposal_gbp": 900.0,
        "trigger_multiplier":       1.2,
    })


@pytest.fixture(scope="module")
def client():
    from dmai_core_complete import app
    app.config["TESTING"] = True
    return app.test_client()


def test_list_and_detail(client):
    p = _seed(hardware_id=101)
    resp = client.get("/api/admin/purchase-proposals")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert any(x["id"] == p["id"] for x in body["proposals"])

    detail = client.get(f"/api/admin/purchase-proposals/{p['id']}")
    assert detail.status_code == 200
    assert detail.get_json()["proposal"]["id"] == p["id"]

    missing = client.get("/api/admin/purchase-proposals/999999")
    assert missing.status_code == 404

    bad = client.get("/api/admin/purchase-proposals?state=bogus")
    assert bad.status_code == 400


def test_approve_debits_treasury(client):
    from components.treasury import treasury_ledger as tl
    p = _seed(hardware_id=102, capex=400.0)
    before = tl.get_balance()
    resp = client.post(f"/api/admin/purchase-proposals/{p['id']}/approve",
                       json={"note": "ok"})
    assert resp.status_code == 200
    assert resp.get_json()["proposal"]["state"] == "approved"
    after = tl.get_balance()
    assert round(before - after, 2) == 400.0  # infra_spend -capex


def test_mark_purchased_requires_price_then_reconciles(client):
    p = _seed(hardware_id=103, capex=300.0)
    client.post(f"/api/admin/purchase-proposals/{p['id']}/approve", json={})
    # Missing actual_price_gbp → 400.
    missing = client.post(
        f"/api/admin/purchase-proposals/{p['id']}/mark-purchased", json={})
    assert missing.status_code == 400
    # With a price → purchased.
    ok = client.post(
        f"/api/admin/purchase-proposals/{p['id']}/mark-purchased",
        json={"actual_price_gbp": 320.0})
    assert ok.status_code == 200
    assert ok.get_json()["proposal"]["state"] == "purchased"


def test_autocheckout_config_requires_confirm_token(client):
    # No token → 403, but the endpoint echoes the token so the operator sees it.
    resp = client.post("/api/admin/purchase-gate/auto-checkout-config",
                       json={"enabled": True})
    assert resp.status_code == 403
    token = resp.get_json()["confirm_token"]
    assert token
    # Wrong token → still disabled.
    assert PurchaseGateStore().auto_checkout_enabled() is False
    # Correct token → applied.
    ok = client.post("/api/admin/purchase-gate/auto-checkout-config",
                    json={"enabled": True, "confirm_token": token})
    assert ok.status_code == 200
    assert ok.get_json()["enabled"] is True
    # Reset so other modules/tests see the default.
    client.post("/api/admin/purchase-gate/auto-checkout-config",
               json={"enabled": False, "confirm_token": token})


def test_status_endpoint_shape(client):
    resp = client.get("/api/admin/purchase-gate-status")
    assert resp.status_code == 200
    body = resp.get_json()
    for key in ("auto_checkout_enabled", "auto_checkout_dry_run",
                "auto_checkout_max_gbp", "streak_days_positive",
                "streak_requirement_met", "confirm_token", "adapter_map"):
        assert key in body
    assert all(v["is_implemented"] is False
               for v in body["adapter_map"].values())


def test_purchases_page_renders(client):
    _seed(hardware_id=104)
    resp = client.get("/admin/purchases")
    assert resp.status_code == 200
    assert "text/html" in resp.content_type
    body = resp.get_data(as_text=True)
    assert "Auto-checkout" in body
    assert "NotImplementedError" in body
