"""Tri-channel notifier tests for the purchase gate (PR L)."""
from __future__ import annotations

from components.purchase_gate.notifier import Notifier

_PROP = {
    "id": 1, "hw_name": "Box A", "hw_source": "amazon_uk",
    "capex_gbp": 500.0, "tco_gbp_3yr": 700.0,
    "treasury_at_proposal_gbp": 900.0,
}


def test_all_three_channels_deliver(monkeypatch):
    n = Notifier()
    monkeypatch.setattr(n, "_notify_inapp", lambda p: True)
    monkeypatch.setattr(n, "_notify_slack", lambda p: True)
    monkeypatch.setattr(n, "_notify_email", lambda p: True)
    assert n.send_new_proposal(_PROP) == ["in_app", "slack", "email"]


def test_partial_delivery_returns_subset(monkeypatch):
    n = Notifier()
    monkeypatch.setattr(n, "_notify_inapp", lambda p: True)
    monkeypatch.setattr(n, "_notify_slack", lambda p: False)
    monkeypatch.setattr(n, "_notify_email", lambda p: True)
    assert n.send_new_proposal(_PROP) == ["in_app", "email"]


def test_channel_exception_is_swallowed(monkeypatch):
    n = Notifier()

    def boom(p):
        raise RuntimeError("smtp down")

    monkeypatch.setattr(n, "_notify_inapp", lambda p: True)
    monkeypatch.setattr(n, "_notify_slack", lambda p: True)
    monkeypatch.setattr(n, "_notify_email", boom)
    # Must not raise; email is simply absent from the delivered list.
    assert n.send_new_proposal(_PROP) == ["in_app", "slack"]
