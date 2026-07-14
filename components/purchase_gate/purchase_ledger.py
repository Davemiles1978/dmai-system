"""Store + ledger actions for the purchase-approval gate (PR L).

Flat-file style mirroring :mod:`components.treasury.treasury_ledger`: WAL,
busy_timeout, row_factory, a ``config_kv`` table, and module-level action
functions that wire treasury entries onto state transitions.

Valid proposal state machine (enforced in code)::

    pending ─approve──▶ approved ─mark-purchased─▶ purchased
       │                   │
       └─decline─▶declined └─cancel───▶ cancelled
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from components.purchase_gate import config as cfg
from components.purchase_gate.schema import SCHEMA

logger = logging.getLogger(__name__)

# Which target states each current state may transition into.
VALID_TRANSITIONS: Dict[str, set] = {
    "pending":   {"approved", "declined"},
    "approved":  {"purchased", "cancelled"},
    "purchased": set(),
    "cancelled": set(),
    "declined":  set(),
}

OPEN_STATES = ("pending", "approved")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PurchaseGateStore:
    """Thin wrapper around ``data/dmai_purchase_gate.db``."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or cfg.default_purchase_gate_path()

    # ── connection ──────────────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(self.db_path, timeout=30.0)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA busy_timeout=30000")
        c.row_factory = sqlite3.Row
        return c

    # ── init ────────────────────────────────────────────────────────────────
    def init_db(self) -> Dict[str, Any]:
        with self._conn() as c:
            for ddl in SCHEMA:
                c.execute(ddl)
            row = c.execute(
                "SELECT value FROM config_kv WHERE key = ?",
                (cfg.KV_INSTALL_TS,),
            ).fetchone()
            if row is None:
                c.execute(
                    "INSERT INTO config_kv (key, value, updated_ts) "
                    "VALUES (?, ?, ?)",
                    (cfg.KV_INSTALL_TS, _now_iso(), _now_iso()),
                )
            c.commit()
        return {"install_ts": self.install_ts()}

    # ── config_kv ─────────────────────────────────────────────────────────────
    def config_kv_get(self, key: str, default: Any = None) -> Any:
        with self._conn() as c:
            row = c.execute(
                "SELECT value FROM config_kv WHERE key = ?", (key,),
            ).fetchone()
        return row["value"] if row else default

    def config_kv_set(self, key: str, value: Any) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO config_kv (key, value, updated_ts) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_ts = excluded.updated_ts",
                (key, str(value), _now_iso()),
            )
            c.commit()

    def install_ts(self) -> str:
        return str(self.config_kv_get(cfg.KV_INSTALL_TS) or "")

    def confirm_token(self) -> str:
        return cfg.confirm_token(self.install_ts())

    # ── auto-checkout config (module const → env → config_kv) ─────────────────
    def _bool_kv(self, key: str, env_val: Optional[bool],
                 const: bool) -> bool:
        raw = self.config_kv_get(key)
        if raw is not None:
            return str(raw).strip().lower() in ("1", "true", "yes", "on")
        if env_val is not None:
            return env_val
        return const

    def auto_checkout_enabled(self) -> bool:
        return self._bool_kv(cfg.KV_AUTO_CHECKOUT_ENABLED,
                             cfg.env_auto_checkout_enabled(),
                             cfg.AUTO_CHECKOUT_ENABLED)

    def auto_checkout_dry_run(self) -> bool:
        return self._bool_kv(cfg.KV_AUTO_CHECKOUT_DRY_RUN,
                             cfg.env_auto_checkout_dry_run(),
                             cfg.AUTO_CHECKOUT_DRY_RUN)

    def auto_checkout_max_gbp(self) -> float:
        raw = self.config_kv_get(cfg.KV_AUTO_CHECKOUT_MAX_GBP)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
        env = cfg.env_auto_checkout_max_gbp()
        return env if env is not None else cfg.AUTO_CHECKOUT_MAX_GBP

    # ── proposals ─────────────────────────────────────────────────────────────
    def insert_proposal(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a proposal (state defaults to 'pending'); return the row."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO purchase_proposals "
                "(created_ts, hardware_id, hw_name, hw_source, hw_url, "
                " capex_gbp, tco_gbp_3yr, treasury_at_proposal_gbp, "
                " trigger_multiplier, state, channels_notified, "
                " auto_checkout_attempted, auto_checkout_result, notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row.get("created_ts") or _now_iso(),
                    int(row["hardware_id"]),
                    row["hw_name"],
                    row["hw_source"],
                    row["hw_url"],
                    float(row["capex_gbp"]),
                    float(row["tco_gbp_3yr"]),
                    float(row["treasury_at_proposal_gbp"]),
                    float(row.get("trigger_multiplier",
                                  cfg.TRIGGER_MULTIPLIER)),
                    row.get("state", "pending"),
                    json.dumps(row["channels_notified"])
                    if isinstance(row.get("channels_notified"), (list, dict))
                    else row.get("channels_notified"),
                    int(row.get("auto_checkout_attempted", 0)),
                    row.get("auto_checkout_result"),
                    row.get("notes"),
                ),
            )
            c.commit()
            new_id = int(cur.lastrowid)
        return self.get_proposal(new_id)  # type: ignore[return-value]

    def get_proposal(self, proposal_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM purchase_proposals WHERE id = ?",
                (int(proposal_id),),
            ).fetchone()
        return dict(row) if row else None

    def list_proposals(self, state: Optional[str] = None,
                       ) -> List[Dict[str, Any]]:
        q = "SELECT * FROM purchase_proposals"
        params: List[Any] = []
        if state:
            q += " WHERE state = ?"
            params.append(state)
        q += " ORDER BY id DESC"
        with self._conn() as c:
            rows = c.execute(q, params).fetchall()
        return [dict(r) for r in rows]

    def has_open_proposal(self, hardware_id: int) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM purchase_proposals "
                "WHERE hardware_id = ? AND state IN ('pending','approved') "
                "LIMIT 1",
                (int(hardware_id),),
            ).fetchone()
        return row is not None

    def latest_open_proposal(self) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM purchase_proposals "
                "WHERE state IN ('pending','approved') "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def transition_state(self, proposal_id: int, new_state: str, *,
                         actor: str = "operator",
                         note: Optional[str] = None,
                         actual_price_gbp: Optional[float] = None,
                         ) -> Dict[str, Any]:
        """Move a proposal to ``new_state``, enforcing the state machine.

        Raises ``ValueError`` on an unknown id or an illegal transition.
        """
        prop = self.get_proposal(proposal_id)
        if prop is None:
            raise ValueError(f"no such proposal: {proposal_id}")
        current = prop["state"]
        allowed = VALID_TRANSITIONS.get(current, set())
        if new_state not in allowed:
            raise ValueError(
                f"illegal transition {current!r} → {new_state!r} "
                f"(allowed: {sorted(allowed) or 'none'})"
            )
        with self._conn() as c:
            c.execute(
                "UPDATE purchase_proposals SET state = ?, "
                " operator_action_ts = ?, operator_note = ?, "
                " actual_price_gbp = COALESCE(?, actual_price_gbp) "
                "WHERE id = ?",
                (new_state, _now_iso(), note, actual_price_gbp,
                 int(proposal_id)),
            )
            c.commit()
        logger.info("purchase_gate: proposal %s %s→%s by %s",
                    proposal_id, current, new_state, actor)
        return self.get_proposal(proposal_id)  # type: ignore[return-value]

    def update_fields(self, proposal_id: int, **cols: Any) -> None:
        if not cols:
            return
        sets = ", ".join(f"{k} = ?" for k in cols)
        params = list(cols.values()) + [int(proposal_id)]
        with self._conn() as c:
            c.execute(
                f"UPDATE purchase_proposals SET {sets} WHERE id = ?", params,
            )
            c.commit()

    def set_channels_notified(self, proposal_id: int,
                              channels: List[str]) -> None:
        self.update_fields(proposal_id,
                           channels_notified=json.dumps(channels))

    # ── lifetime aggregates ───────────────────────────────────────────────────
    def total_purchased_gbp(self) -> float:
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(COALESCE(actual_price_gbp, capex_gbp)), "
                " 0.0) AS s FROM purchase_proposals WHERE state = 'purchased'"
            ).fetchone()
        return float(row["s"] or 0.0)


# ── ledger action functions (treasury-wiring on transition) ───────────────────
#
# The treasury_ledger CHECK constraint (PR I) only permits five kinds and has
# no ``provenance`` column, so provenance + adjustment semantics are encoded in
# the entry ``description`` and linked back via the proposal's *_entry_id
# columns. See workspace/pr_l_notes.md.

def _treasury_record(kind: str, amount_gbp: float, description: str,
                     treasury_db_path: Optional[str]) -> int:
    from components.treasury import treasury_ledger as tl
    tl.init_treasury_db(treasury_db_path)
    return tl.record_manual(kind=kind, amount_gbp=amount_gbp,
                            description=description,
                            db_path=treasury_db_path)


def approve_proposal(proposal_id: int, *, note: str = "",
                     actor: str = "operator",
                     purchase_db_path: Optional[str] = None,
                     treasury_db_path: Optional[str] = None,
                     ) -> Dict[str, Any]:
    """pending → approved, creating the auto-debit treasury entry."""
    store = PurchaseGateStore(purchase_db_path)
    prop = store.get_proposal(proposal_id)
    if prop is None:
        raise ValueError(f"no such proposal: {proposal_id}")
    capex = float(prop["capex_gbp"])
    entry_id = _treasury_record(
        "infra_spend", -capex,
        f"purchase_proposal:{proposal_id} auto-debit {prop['hw_name']}",
        treasury_db_path,
    )
    updated = store.transition_state(proposal_id, "approved",
                                     actor=actor, note=note)
    store.update_fields(proposal_id, auto_debit_entry_id=entry_id)
    updated["auto_debit_entry_id"] = entry_id
    return updated


def decline_proposal(proposal_id: int, *, note: str = "",
                     actor: str = "operator",
                     purchase_db_path: Optional[str] = None,
                     treasury_db_path: Optional[str] = None,
                     ) -> Dict[str, Any]:
    """pending → declined. No treasury change (no debit was ever made)."""
    store = PurchaseGateStore(purchase_db_path)
    return store.transition_state(proposal_id, "declined",
                                  actor=actor, note=note)


def cancel_proposal(proposal_id: int, *, note: str = "",
                    actor: str = "operator",
                    purchase_db_path: Optional[str] = None,
                    treasury_db_path: Optional[str] = None,
                    ) -> Dict[str, Any]:
    """approved → cancelled, reversing the auto-debit with a credit."""
    store = PurchaseGateStore(purchase_db_path)
    prop = store.get_proposal(proposal_id)
    if prop is None:
        raise ValueError(f"no such proposal: {proposal_id}")
    capex = float(prop["capex_gbp"])
    entry_id = _treasury_record(
        "manual_credit", +capex,
        f"purchase_proposal_cancelled:{proposal_id} reverse auto-debit",
        treasury_db_path,
    )
    updated = store.transition_state(proposal_id, "cancelled",
                                     actor=actor, note=note)
    store.update_fields(proposal_id, cancel_credit_entry_id=entry_id)
    updated["cancel_credit_entry_id"] = entry_id
    return updated


def mark_purchased(proposal_id: int, *, actual_price_gbp: float,
                   note: str = "", actor: str = "operator",
                   purchase_db_path: Optional[str] = None,
                   treasury_db_path: Optional[str] = None,
                   ) -> Dict[str, Any]:
    """approved → purchased, reconciling any delta vs the auto-debited capex.

    Reconciliation is booked as an ``infra_spend`` entry for ``-delta`` (a
    higher actual price spends more; a lower one refunds). The treasury schema
    has no dedicated adjustment kind, so the entry description is prefixed
    ``infra_spend_adjustment`` for auditability.
    """
    store = PurchaseGateStore(purchase_db_path)
    prop = store.get_proposal(proposal_id)
    if prop is None:
        raise ValueError(f"no such proposal: {proposal_id}")
    capex = float(prop["capex_gbp"])
    actual = float(actual_price_gbp)
    updated = store.transition_state(proposal_id, "purchased", actor=actor,
                                     note=note, actual_price_gbp=actual)
    delta = round(actual - capex, 2)
    if abs(delta) >= 0.005:
        entry_id = _treasury_record(
            "infra_spend", -delta,
            f"infra_spend_adjustment purchase_proposal:{proposal_id} "
            f"actual {actual:.2f} vs capex {capex:.2f}",
            treasury_db_path,
        )
        store.update_fields(proposal_id, reconcile_entry_id=entry_id)
        updated["reconcile_entry_id"] = entry_id
    return updated


__all__ = [
    "PurchaseGateStore",
    "VALID_TRANSITIONS",
    "OPEN_STATES",
    "approve_proposal",
    "decline_proposal",
    "cancel_proposal",
    "mark_purchased",
]
