"""Purchase-gate monitor (PR L).

One check cycle:
  1. Read the live procurement top-1 shortlist row (PR K) + treasury balance.
  2. If ``balance >= TRIGGER_MULTIPLIER * capex`` and there's no open proposal
     for that hardware_id, insert a new proposal (annotating if the top-1
     changed while another proposal is still open).
  3. Fire tri-channel notifications (best-effort).
  4. If auto-checkout is enabled+eligible, attempt it (dry-run by default;
     no adapter can actually execute — see checkout_adapter).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, Optional

from components.purchase_gate import config as cfg
from components.purchase_gate.checkout_adapter import ADAPTERS
from components.purchase_gate.notifier import Notifier
from components.purchase_gate.purchase_ledger import PurchaseGateStore

logger = logging.getLogger(__name__)


def _top1_shortlist(procurement_db_path: Optional[str]
                    ) -> Optional[Dict[str, Any]]:
    from components.procurement.store import ProcurementStore
    store = ProcurementStore(procurement_db_path)
    store.init_db()
    rows = store.get_shortlist()
    for r in rows:
        if int(r.get("rank", 0)) == 1:
            return r
    return rows[0] if rows else None


def _treasury_balance(treasury_db_path: Optional[str]) -> float:
    from components.treasury import treasury_ledger as tl
    return float(tl.get_balance(db_path=treasury_db_path))


def positive_pnl_streak_days(treasury_db_path: Optional[str] = None) -> int:
    """Count consecutive calendar days (ending at the most recent recorded
    day) whose realised-P&L entries net positive."""
    from components.treasury import treasury_ledger as tl
    placeholders = ",".join("?" for _ in cfg.REALISED_PNL_KINDS)
    with tl._conn(treasury_db_path) as c:  # noqa: SLF001 - internal helper reuse
        rows = c.execute(
            f"SELECT substr(ts,1,10) AS d, SUM(amount_gbp) AS s "
            f"FROM treasury_ledger WHERE kind IN ({placeholders}) "
            f"GROUP BY d ORDER BY d DESC",
            cfg.REALISED_PNL_KINDS,
        ).fetchall()
    day_sums: Dict[str, float] = {r["d"]: float(r["s"] or 0.0) for r in rows}
    if not rows:
        return 0
    try:
        cur = date.fromisoformat(rows[0]["d"])
    except (TypeError, ValueError):
        return 0
    streak = 0
    while True:
        key = cur.isoformat()
        if day_sums.get(key, 0.0) > 0:
            streak += 1
            cur = cur - timedelta(days=1)
        else:
            break
    return streak


class PurchaseGateMonitor:
    def __init__(self, *,
                 purchase_db_path: Optional[str] = None,
                 procurement_db_path: Optional[str] = None,
                 treasury_db_path: Optional[str] = None,
                 notifier: Optional[Notifier] = None,
                 trigger_multiplier: float = cfg.TRIGGER_MULTIPLIER) -> None:
        self.purchase_db_path = purchase_db_path
        self.procurement_db_path = procurement_db_path
        self.treasury_db_path = treasury_db_path
        self.notifier = notifier if notifier is not None else Notifier(
            slack_db_path=purchase_db_path)
        self.trigger_multiplier = float(trigger_multiplier)
        self.last_check_ts: Optional[str] = None

    # ── auto-checkout eligibility + attempt ───────────────────────────────────
    def _maybe_auto_checkout(self, store: PurchaseGateStore,
                             proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the auto-checkout gate for a freshly-created proposal.

        Returns the (possibly-updated) proposal dict. Enforces all invariants;
        no adapter can actually execute a purchase today.
        """
        if not store.auto_checkout_enabled():
            return proposal
        capex = float(proposal["capex_gbp"])
        if capex > store.auto_checkout_max_gbp():
            logger.info("purchase_gate: auto-checkout skipped — capex £%.2f "
                        "over cap £%.2f", capex, store.auto_checkout_max_gbp())
            return proposal
        streak = positive_pnl_streak_days(self.treasury_db_path)
        if streak < cfg.AUTO_CHECKOUT_REQUIRE_STREAK_DAYS:
            logger.info("purchase_gate: auto-checkout skipped — streak %d < %d",
                        streak, cfg.AUTO_CHECKOUT_REQUIRE_STREAK_DAYS)
            return proposal
        adapter_cls = ADAPTERS.get(proposal["hw_source"])
        if adapter_cls is None:
            logger.info("purchase_gate: no adapter for source %s",
                        proposal["hw_source"])
            return proposal
        adapter = adapter_cls()
        ok, reason = adapter.can_checkout(proposal)
        if not ok:
            store.update_fields(proposal["id"], auto_checkout_attempted=1,
                                auto_checkout_result=f"ineligible:{reason}")
            return store.get_proposal(proposal["id"])

        dry_run = store.auto_checkout_dry_run()
        if dry_run:
            store.update_fields(
                proposal["id"], auto_checkout_attempted=1,
                auto_checkout_result="dry_run_eligible",
                notes=(proposal.get("notes") or "") +
                " [DRY RUN] auto-checkout eligible — would have purchased "
                "if dry_run off")
            return store.get_proposal(proposal["id"])

        # Live path — no adapter implements this; execute_checkout raises.
        try:
            result = adapter.execute_checkout(proposal, dry_run=False)
            order_id = str(result.get("order_id", "?"))
            actual = float(result.get("actual_price_gbp", capex))
            from components.purchase_gate import purchase_ledger as pl
            pl.approve_proposal(proposal["id"], note="auto-checkout",
                                actor="auto_checkout",
                                purchase_db_path=self.purchase_db_path,
                                treasury_db_path=self.treasury_db_path)
            pl.mark_purchased(proposal["id"], actual_price_gbp=actual,
                              note="auto-checkout", actor="auto_checkout",
                              purchase_db_path=self.purchase_db_path,
                              treasury_db_path=self.treasury_db_path)
            store.update_fields(proposal["id"], auto_checkout_attempted=1,
                                auto_checkout_result=f"executed:{order_id}")
        except Exception as e:
            store.update_fields(proposal["id"], auto_checkout_attempted=1,
                                auto_checkout_result=f"error:{e}")
        return store.get_proposal(proposal["id"])

    # ── main check ─────────────────────────────────────────────────────────────
    def check_once(self) -> Dict[str, Any]:
        from datetime import datetime, timezone
        self.last_check_ts = datetime.now(timezone.utc).isoformat()

        top1 = _top1_shortlist(self.procurement_db_path)
        if not top1:
            return {"triggered": False, "reason": "no_shortlist"}

        balance = _treasury_balance(self.treasury_db_path)
        capex = float(top1["capex_gbp"])
        threshold = self.trigger_multiplier * capex
        if balance < threshold:
            return {"triggered": False, "reason": "below_threshold",
                    "balance_gbp": round(balance, 2),
                    "threshold_gbp": round(threshold, 2)}

        hardware_id = int(top1["hardware_id"])
        store = PurchaseGateStore(self.purchase_db_path)
        store.init_db()
        if store.has_open_proposal(hardware_id):
            return {"triggered": False, "reason": "open_proposal_exists",
                    "hardware_id": hardware_id}

        note = ""
        other_open = store.latest_open_proposal()
        if other_open and int(other_open["hardware_id"]) != hardware_id:
            note = "top-1 changed since last check"

        proposal = store.insert_proposal({
            "hardware_id":               hardware_id,
            "hw_name":                   top1.get("hw_name"),
            "hw_source":                 top1.get("hw_source"),
            "hw_url":                    top1.get("hw_url") or "",
            "capex_gbp":                 capex,
            "tco_gbp_3yr":               float(top1.get("tco_gbp_3yr") or 0.0),
            "treasury_at_proposal_gbp":  round(balance, 2),
            "trigger_multiplier":        self.trigger_multiplier,
            "notes":                     note or None,
        })

        proposal = self._maybe_auto_checkout(store, proposal)

        channels = self.notifier.send_new_proposal(proposal)
        store.set_channels_notified(proposal["id"], channels)

        return {"triggered": True, "proposal_id": proposal["id"],
                "hardware_id": hardware_id, "balance_gbp": round(balance, 2),
                "channels": channels,
                "auto_checkout_result": proposal.get("auto_checkout_result"),
                "note": note}


__all__ = ["PurchaseGateMonitor", "positive_pnl_streak_days"]
