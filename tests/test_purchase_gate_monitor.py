"""Monitor check-cycle tests for the purchase gate (PR L).

The procurement top-1 and treasury balance are monkeypatched at the module
level so the trigger arithmetic is exercised in isolation (no procurement or
treasury DB needed). Notifications are stubbed with a fake notifier.
"""
from __future__ import annotations

import pytest

from components.purchase_gate import monitor as mon
from components.purchase_gate.monitor import PurchaseGateMonitor
from components.purchase_gate.purchase_ledger import PurchaseGateStore


class _FakeNotifier:
    def __init__(self):
        self.sent = []

    def send_new_proposal(self, proposal):
        self.sent.append(proposal["id"])
        return ["in_app"]


def _top1(hardware_id=1, capex=500.0):
    return {
        "rank": 1, "hardware_id": hardware_id, "hw_name": "Box A",
        "hw_source": "amazon_uk", "hw_url": "http://x/dp/B0ABCDE123",
        "capex_gbp": capex, "tco_gbp_3yr": 700.0,
    }


def _monitor(tmp_path, notifier=None):
    return PurchaseGateMonitor(
        purchase_db_path=str(tmp_path / "pg.db"),
        procurement_db_path=str(tmp_path / "proc.db"),
        treasury_db_path=str(tmp_path / "treas.db"),
        notifier=notifier or _FakeNotifier(),
    )


def test_no_shortlist(tmp_path, monkeypatch):
    monkeypatch.setattr(mon, "_top1_shortlist", lambda p: None)
    res = _monitor(tmp_path).check_once()
    assert res == {"triggered": False, "reason": "no_shortlist"}


def test_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(mon, "_top1_shortlist", lambda p: _top1(capex=500.0))
    monkeypatch.setattr(mon, "_treasury_balance", lambda p: 500.0)  # < 1.2*500
    res = _monitor(tmp_path).check_once()
    assert res["triggered"] is False
    assert res["reason"] == "below_threshold"


def test_trigger_inserts_proposal_and_notifies(tmp_path, monkeypatch):
    monkeypatch.setattr(mon, "_top1_shortlist", lambda p: _top1(capex=500.0))
    monkeypatch.setattr(mon, "_treasury_balance", lambda p: 650.0)  # >= 600
    fake = _FakeNotifier()
    res = _monitor(tmp_path, fake).check_once()
    assert res["triggered"] is True
    assert res["hardware_id"] == 1
    assert res["channels"] == ["in_app"]
    assert fake.sent == [res["proposal_id"]]
    # Persisted and channels recorded.
    store = PurchaseGateStore(str(tmp_path / "pg.db"))
    p = store.get_proposal(res["proposal_id"])
    assert p["state"] == "pending"
    assert "in_app" in (p["channels_notified"] or "")


def test_dedupe_open_proposal(tmp_path, monkeypatch):
    monkeypatch.setattr(mon, "_top1_shortlist", lambda p: _top1(capex=500.0))
    monkeypatch.setattr(mon, "_treasury_balance", lambda p: 650.0)
    m = _monitor(tmp_path)
    first = m.check_once()
    assert first["triggered"] is True
    second = m.check_once()
    assert second["triggered"] is False
    assert second["reason"] == "open_proposal_exists"


def test_top1_changed_note(tmp_path, monkeypatch):
    monkeypatch.setattr(mon, "_treasury_balance", lambda p: 5000.0)
    m = _monitor(tmp_path)
    # First proposal for hardware 1.
    monkeypatch.setattr(mon, "_top1_shortlist",
                        lambda p: _top1(hardware_id=1, capex=500.0))
    m.check_once()
    # Top-1 flips to hardware 2 while hw1's proposal is still open.
    monkeypatch.setattr(mon, "_top1_shortlist",
                        lambda p: _top1(hardware_id=2, capex=500.0))
    res = m.check_once()
    assert res["triggered"] is True
    assert res["hardware_id"] == 2
    assert res["note"] == "top-1 changed since last check"
