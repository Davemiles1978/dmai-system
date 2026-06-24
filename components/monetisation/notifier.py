"""
Slack notifier for the monetisation hub.

Posts trader fills, tier changes, circuit-breaker halts, and daily P&L
digests to a Slack channel via incoming webhook.

Configuration (env vars):
  SLACK_WEBHOOK_URL   - Required for delivery; if unset, all sends become no-ops
                        but events are still recorded in mon_alerts for audit.
  SLACK_ALERT_MASK    - Comma-separated list of enabled categories. Default
                        "trade,tier,halt,digest,error".

Per-event categories:
  trade   - Every executed autonomous trade
  tier    - Tier promotion/demotion
  halt    - Circuit-breaker fires (drawdown, cap reached)
  digest  - End-of-session P&L summary
  error   - Unhandled exception inside the loop

All sends fail safely (logged, never raised).
"""

import os
import json
import time
import logging
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import requests

logger = logging.getLogger(__name__)


DEFAULT_MASK = ("trade", "tier", "halt", "digest", "error")

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS mon_alerts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          TEXT    NOT NULL DEFAULT (datetime('now')),
        category    TEXT    NOT NULL,
        title       TEXT    NOT NULL,
        body        TEXT,
        meta_json   TEXT,
        delivered   INTEGER NOT NULL DEFAULT 0,
        error       TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS mon_alerts_cat_ts ON mon_alerts(category, ts DESC)",
]


def _env_mask() -> set:
    raw = os.getenv("SLACK_ALERT_MASK", "").strip()
    if not raw:
        return set(DEFAULT_MASK)
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


class SlackNotifier:
    """Lightweight Slack webhook poster with SQLite audit."""

    def __init__(self, db_path: str,
                 webhook_url: Optional[str] = None,
                 mask: Optional[Iterable[str]] = None):
        self.db_path = db_path
        self._webhook = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "").strip() or None
        self._mask = set(mask) if mask else _env_mask()
        self._lock = threading.Lock()
        self._init_db()

    # ── DB ────────────────────────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as c:
            for ddl in SCHEMA:
                c.execute(ddl)
            c.commit()

    # ── Public surface ────────────────────────────────────────────────────────
    def configured(self) -> bool:
        return bool(self._webhook)

    def status(self) -> Dict[str, Any]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts, category, title, delivered, error "
                "FROM mon_alerts ORDER BY id DESC LIMIT 25"
            ).fetchall()
        return {
            "configured":   self.configured(),
            "mask":         sorted(self._mask),
            "available":    sorted(DEFAULT_MASK),
            "recent":       [dict(r) for r in rows],
        }

    def set_mask(self, mask: Iterable[str]) -> Dict[str, Any]:
        clean = {m.strip().lower() for m in mask if m and m.strip().lower() in DEFAULT_MASK}
        self._mask = clean or set(DEFAULT_MASK)
        return self.status()

    def set_webhook(self, url: Optional[str]) -> Dict[str, Any]:
        self._webhook = (url or "").strip() or None
        return self.status()

    def send(self, category: str, title: str, body: str = "",
             meta: Optional[Dict[str, Any]] = None,
             blocks: Optional[list] = None) -> bool:
        """Send (best-effort). Always records an audit row.
        Returns True if Slack accepted the post."""
        category = (category or "info").lower()
        meta = meta or {}
        if category not in self._mask:
            self._record(category, title, body, meta, delivered=0, error="masked")
            return False
        if not self._webhook:
            self._record(category, title, body, meta, delivered=0, error="no_webhook")
            return False
        payload: Dict[str, Any] = {"text": f"*{title}*\n{body}"} if not blocks else {
            "text": title, "blocks": blocks,
        }
        delivered = 0
        err: Optional[str] = None
        try:
            r = requests.post(self._webhook, json=payload, timeout=8)
            if 200 <= r.status_code < 300:
                delivered = 1
            else:
                err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            err = str(e)[:300]
        self._record(category, title, body, meta, delivered=delivered, error=err)
        return bool(delivered)

    # ── Convenience helpers used by AutonomousTrader ──────────────────────────
    def trade(self, tr: Dict[str, Any]) -> None:
        live = bool(tr.get("live"))
        mode = "LIVE" if live else "PAPER"
        title = f"{mode} buy {tr.get('symbol')} ({tr.get('tier')})"
        body = (
            f"qty={tr.get('qty')}  "
            f"confidence={float(tr.get('confidence') or 0):.0%}  "
            f"EV={float(tr.get('ev') or 0):.1%}"
        )
        self.send("trade", title, body, meta=tr)

    def tier_change(self, from_tier: str, to_tier: str, reason: str) -> None:
        arrow = "↑ promoted" if (
            ["conservative", "moderate", "aggressive"].index(to_tier) >
            ["conservative", "moderate", "aggressive"].index(from_tier)
        ) else "↓ demoted"
        self.send("tier",
                  f"Trader tier {arrow}: {from_tier} → {to_tier}",
                  f"reason: {reason}",
                  meta={"from": from_tier, "to": to_tier, "reason": reason})

    def halt(self, kind: str, detail: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.send("halt",
                  f"Trader halted: {kind}",
                  detail,
                  meta=meta or {})

    def digest(self, summary: Dict[str, Any]) -> None:
        pnl = summary.get("pnl_pct", 0) or 0
        emoji = "📈" if pnl > 0 else ("📉" if pnl < 0 else "➖")
        body = (
            f"trades: {summary.get('trades', 0)}  "
            f"win rate: {summary.get('win_rate_pct') if summary.get('win_rate_pct') is not None else '—'}  "
            f"deployed: {(summary.get('deployed_pct') or 0)*100:.1f}%\n"
            f"tier: {summary.get('tier')}  mode: {'LIVE' if summary.get('live') else 'PAPER'}\n"
            f"equity: {summary.get('equity')}  P&L: {pnl*100:.2f}%"
        )
        self.send("digest", f"{emoji} Daily trader digest ({summary.get('date')})",
                  body, meta=summary)

    def error(self, where: str, message: str) -> None:
        self.send("error", f"Trader error in {where}", message[:1500])

    # ── Internals ─────────────────────────────────────────────────────────────
    def _record(self, category: str, title: str, body: str,
                meta: Dict[str, Any], delivered: int, error: Optional[str]) -> None:
        try:
            with self._lock, self._conn() as c:
                c.execute(
                    "INSERT INTO mon_alerts(category, title, body, meta_json, "
                    "delivered, error) VALUES (?, ?, ?, ?, ?, ?)",
                    (category, title, body, json.dumps(meta)[:4000], delivered, error),
                )
                c.commit()
        except Exception as e:
            logger.debug("SlackNotifier audit insert failed: %s", e)


def get_notifier(db_path: str) -> SlackNotifier:
    return SlackNotifier(db_path=db_path)
