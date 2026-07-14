"""Tri-channel notifier for new purchase proposals (PR L).

Fans a new proposal out to three channels — in-app, Slack (#dmaitalk), and
email — best-effort. A failure on any one channel is logged and swallowed so
it never blocks proposal creation. :meth:`Notifier.send_new_proposal` returns
the list of channels that actually delivered, which the monitor records in
``purchase_proposals.channels_notified``.

All external integrations are imported lazily inside the per-channel helpers
so tests can monkeypatch them without importing the app.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from components.purchase_gate import config as cfg

logger = logging.getLogger(__name__)


def _pplx_send_notification(*, title: str, body: str,
                            channels: List[str],
                            template: str = "generic") -> bool:
    """Bridge to the Perplexity harness ``send_notification`` tool.

    Returns True on delivery. Best-effort: if the tool is not importable in
    this runtime it returns False (the caller logs and continues). Tests
    monkeypatch this function directly.
    """
    send = None
    try:  # pragma: no cover - resolution path varies by runtime
        from pplx_tools import send_notification as send  # type: ignore
    except Exception:
        try:
            from components.integrations.pplx import (  # type: ignore
                send_notification as send,
            )
        except Exception:
            send = None
    if send is None:
        logger.info("purchase_gate.notify: send_notification unavailable "
                    "(channels=%s)", channels)
        return False
    try:
        send(title=title, body=body, channels=channels, template=template)
        return True
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("purchase_gate.notify: send_notification failed: %s", e)
        return False


class Notifier:
    def __init__(self, *, slack_db_path: Optional[str] = None,
                 slack_webhook_url: Optional[str] = None) -> None:
        self._slack_db_path = slack_db_path
        self._slack_webhook_url = slack_webhook_url

    # ── message body ──────────────────────────────────────────────────────────
    @staticmethod
    def _title() -> str:
        return "DMAI purchase proposal"

    @staticmethod
    def _body(proposal: Dict[str, Any]) -> str:
        lines = [
            f"{proposal.get('hw_name')} — £{float(proposal.get('capex_gbp', 0)):,.2f}",
            f"source: {proposal.get('hw_source')}",
            f"3yr TCO: £{float(proposal.get('tco_gbp_3yr', 0)):,.2f}",
            f"treasury at proposal: "
            f"£{float(proposal.get('treasury_at_proposal_gbp', 0)):,.2f}",
        ]
        notes = proposal.get("notes")
        if notes:
            lines.append(str(notes))
        ac = proposal.get("auto_checkout_result")
        if ac:
            lines.append(f"auto-checkout: {ac}")
        lines.append(f"proposal #{proposal.get('id')} — approve in /admin/purchases")
        return "\n".join(lines)

    # ── per-channel helpers (return bool; monkeypatched in tests) ─────────────
    def _notify_inapp(self, proposal: Dict[str, Any]) -> bool:
        return _pplx_send_notification(
            title=self._title(), body=self._body(proposal),
            channels=["in_app"], template="generic")

    def _notify_slack(self, proposal: Dict[str, Any]) -> bool:
        try:
            from components.monetisation.notifier import SlackNotifier
        except Exception as e:
            logger.info("purchase_gate.notify: SlackNotifier unavailable: %s",
                        e)
            return False
        db_path = self._slack_db_path or cfg.default_purchase_gate_path()
        # Use a mask containing our category so the send isn't filtered out.
        notifier = SlackNotifier(db_path,
                                 webhook_url=self._slack_webhook_url,
                                 mask={"purchase"})
        return bool(notifier.send("purchase", self._title(),
                                  self._body(proposal),
                                  meta={"proposal_id": proposal.get("id")}))

    def _notify_email(self, proposal: Dict[str, Any]) -> bool:
        body = f"To: {cfg.OPERATOR_EMAIL}\n\n" + self._body(proposal)
        return _pplx_send_notification(
            title=self._title(), body=body,
            channels=["email"], template="generic")

    # ── fanout ────────────────────────────────────────────────────────────────
    def send_new_proposal(self, proposal: Dict[str, Any]) -> List[str]:
        delivered: List[str] = []
        for name, fn in (("in_app", self._notify_inapp),
                         ("slack", self._notify_slack),
                         ("email", self._notify_email)):
            try:
                if fn(proposal):
                    delivered.append(name)
            except Exception as e:
                logger.warning("purchase_gate.notify: channel %s failed: %s",
                               name, e)
        return delivered


__all__ = ["Notifier", "_pplx_send_notification"]
