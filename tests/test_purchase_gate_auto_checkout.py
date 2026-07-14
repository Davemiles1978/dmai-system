"""Auto-checkout gate tests (PR L, PART B — feature-flagged OFF).

Exercises the eligibility ladder in
:meth:`PurchaseGateMonitor._maybe_auto_checkout`. Even when every gate is
open, no real purchase can occur: the stub adapters raise
``NotImplementedError`` from ``execute_checkout`` (invariant #3), so the live
path can only ever record an ``error:`` result — the proposal stays pending.
"""
from __future__ import annotations

import pytest

from components.purchase_gate import monitor as mon
from components.purchase_gate import checkout_adapter as ca
from components.purchase_gate.monitor import PurchaseGateMonitor
from components.purchase_gate.purchase_ledger import PurchaseGateStore


class _FakeNotifier:
    def send_new_proposal(self, proposal):
        return ["in_app"]


def _top1(capex=500.0):
    return {
        "rank": 1, "hardware_id": 1, "hw_name": "Box A",
        "hw_source": "amazon_uk", "hw_url": "http://x/dp/B0ABCDE123",
        "capex_gbp": capex, "tco_gbp_3yr": 700.0,
    }


def _run(tmp_path, monkeypatch, *, capex=500.0, streak=30):
    monkeypatch.setattr(mon, "_top1_shortlist", lambda p: _top1(capex=capex))
    monkeypatch.setattr(mon, "_treasury_balance", lambda p: capex * 5)
    monkeypatch.setattr(mon, "positive_pnl_streak_days", lambda p=None: streak)
    m = PurchaseGateMonitor(
        purchase_db_path=str(tmp_path / "pg.db"),
        procurement_db_path=str(tmp_path / "proc.db"),
        treasury_db_path=str(tmp_path / "treas.db"),
        notifier=_FakeNotifier(),
    )
    res = m.check_once()
    store = PurchaseGateStore(str(tmp_path / "pg.db"))
    return store, store.get_proposal(res["proposal_id"])


def _enable(tmp_path, **kv):
    s = PurchaseGateStore(str(tmp_path / "pg.db"))
    s.init_db()
    s.config_kv_set("auto_checkout_enabled", True)
    for k, v in kv.items():
        s.config_kv_set(k, v)


def test_disabled_by_default_no_attempt(tmp_path, monkeypatch):
    _store, prop = _run(tmp_path, monkeypatch)
    assert prop["auto_checkout_attempted"] == 0
    assert prop["auto_checkout_result"] is None


def test_over_cap_skipped(tmp_path, monkeypatch):
    _enable(tmp_path)  # capex 800 > 750 cap
    _store, prop = _run(tmp_path, monkeypatch, capex=800.0)
    assert prop["auto_checkout_attempted"] == 0


def test_streak_below_requirement_skipped(tmp_path, monkeypatch):
    _enable(tmp_path)
    _store, prop = _run(tmp_path, monkeypatch, streak=5)
    assert prop["auto_checkout_attempted"] == 0


def test_dry_run_eligible_marks_but_does_not_purchase(tmp_path, monkeypatch):
    _enable(tmp_path)  # dry_run defaults True
    monkeypatch.setattr(ca.AmazonUKAdapter, "can_checkout",
                        lambda self, p: (True, "ok"))
    _store, prop = _run(tmp_path, monkeypatch)
    assert prop["auto_checkout_attempted"] == 1
    assert prop["auto_checkout_result"] == "dry_run_eligible"
    assert prop["state"] == "pending"  # never auto-purchased in dry-run


def test_live_execute_raises_keeps_pending_with_error(tmp_path, monkeypatch):
    _enable(tmp_path, auto_checkout_dry_run=False)
    monkeypatch.setattr(ca.AmazonUKAdapter, "can_checkout",
                        lambda self, p: (True, "ok"))
    _store, prop = _run(tmp_path, monkeypatch)
    assert prop["auto_checkout_attempted"] == 1
    assert prop["auto_checkout_result"].startswith("error:")
    # Invariant #3: no real purchase — proposal remains pending.
    assert prop["state"] == "pending"


def test_ineligible_when_adapter_refuses(tmp_path, monkeypatch):
    # Real stub adapter: can_checkout is False, so result is 'ineligible:...'.
    _enable(tmp_path)
    _store, prop = _run(tmp_path, monkeypatch)
    assert prop["auto_checkout_attempted"] == 1
    assert prop["auto_checkout_result"].startswith("ineligible:")
