"""Tests for the AutonomousTrader paper/live execution mode (PR #166 Part A).

Covers the at_state.mode column and its admin surface:
  - fresh boot defaults to 'paper' (never trades real money by accident),
  - set_at_mode persists paper↔live in at_state,
  - the migration is idempotent (re-run doesn't error or clobber the value),
  - invalid mode values are rejected,
  - the tick reconciles AggressiveTrader.set_mode() from the persisted value,
  - the /api/trader/at-mode routes are registered and auth-gated.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.wealth import autonomous_trader as at_mod
from components.wealth.autonomous_trader import AutonomousTrader


def _make_trader(tmp_path):
    trader = MagicMock()
    trader.conservative_pairs = []
    trader.trading_pairs = []
    at = AutonomousTrader(
        db_path=str(tmp_path / "dmai_knowledge.db"),
        trader=trader,
        prediction_engine=None,
        notifier=None,
    )
    at.stop()  # halt the background loop; drive helpers directly
    return at


# ── boot default ────────────────────────────────────────────────────────────
def test_fresh_boot_defaults_to_paper(tmp_path):
    at = _make_trader(tmp_path)
    state = at.get_at_mode()
    assert state["mode"] == "paper"
    assert state["enabled"] is False  # A.5: never enabled at boot
    assert "next_tick_at" in state


def test_status_reports_at_mode(tmp_path):
    at = _make_trader(tmp_path)
    assert at.status()["at_mode"] == "paper"


# ── persistence ────────────────────────────────────────────────────────────
def test_set_at_mode_persists_paper_to_live_and_back(tmp_path):
    at = _make_trader(tmp_path)
    out = at.set_at_mode("live")
    assert out["mode"] == "live"
    # Re-read via a fresh instance to prove it persisted in at_state.
    at2 = _make_trader(tmp_path)
    assert at2.get_at_mode()["mode"] == "live"
    at2.set_at_mode("paper")
    assert _make_trader(tmp_path).get_at_mode()["mode"] == "paper"


def test_migration_idempotent_preserves_mode(tmp_path):
    at = _make_trader(tmp_path)
    at.set_at_mode("live")
    # _ensure_tables re-runs the ALTER TABLE migration; must not error or reset.
    at._ensure_tables()
    at._init_db()
    assert at.get_at_mode()["mode"] == "live"


def test_migration_adds_missing_mode_column_as_paper(tmp_path):
    """Pre-#166 at_state (no mode column) → migration adds it defaulting to
    'paper' so no tick ever runs without an explicit mode (A.4 safety belt)."""
    import sqlite3
    db = str(tmp_path / "legacy_knowledge.db")
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE at_state (id INTEGER PRIMARY KEY CHECK (id = 1), "
            "enabled INTEGER NOT NULL DEFAULT 0, tier TEXT NOT NULL DEFAULT 'conservative')"
        )
        conn.execute("INSERT INTO at_state(id, enabled, tier) VALUES (1, 0, 'conservative')")
        conn.commit()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(at_state)").fetchall()}
        assert "mode" not in cols  # legacy shape
        AutonomousTrader._migrate_mode_column(conn)
        conn.commit()
        row = conn.execute("SELECT mode FROM at_state WHERE id = 1").fetchone()
        assert row[0] == "paper"
    finally:
        conn.close()


# ── validation ────────────────────────────────────────────────────────────
def test_invalid_mode_rejected(tmp_path):
    at = _make_trader(tmp_path)
    with pytest.raises(ValueError):
        at.set_at_mode("turbo")
    # State unchanged after a rejected write.
    assert at.get_at_mode()["mode"] == "paper"


# ── tick reconciliation ──────────────────────────────────────────────────────
def test_tick_reconciles_trader_client_from_mode(tmp_path):
    at = _make_trader(tmp_path)
    at.set_at_mode("live")
    at.tick()
    # trader.set_mode(paper=False) must have been called for live mode.
    at.trader.set_mode.assert_called_with(paper=False)

    at.set_at_mode("paper")
    at.trader.set_mode.reset_mock()
    at.tick()
    at.trader.set_mode.assert_called_with(paper=True)


# ── API routes ────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def api_client():
    os.environ.setdefault("RENDER", "false")
    data_dir = "/tmp/dmai_at_mode_api_data"
    os.makedirs(data_dir, exist_ok=True)
    os.environ.setdefault("DATA_PATH", data_dir)
    os.environ.setdefault("MASTER_PASSWORD", "testpw")
    os.environ.setdefault("JWT_SECRET", "testjwt_at_mode_only")
    try:
        import dmai_core_complete as dcc
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"dmai_core_complete import unavailable: {e}")
    dcc.app.config["TESTING"] = True
    return dcc.app.test_client()


def test_post_at_mode_requires_auth(api_client):
    r = api_client.post("/api/trader/at-mode", json={"mode": "live"})
    assert r.status_code == 401


def test_get_at_mode_route_registered(api_client):
    g = api_client.get("/api/trader/at-mode")
    # 200 when the trader component is loaded; 503 if not booted in this env.
    assert g.status_code in (200, 503)
    assert g.status_code != 404
    if g.status_code == 200:
        body = g.get_json()
        assert body["mode"] in ("paper", "live")
        assert "enabled" in body and "next_tick_at" in body


def test_post_invalid_at_mode_rejected(api_client):
    headers = {"X-Master-Password": "testpw"}
    r = api_client.post("/api/trader/at-mode", json={"mode": "turbo"}, headers=headers)
    # 400 when trader loaded (ValueError path); 503 if not booted.
    assert r.status_code in (400, 503)
    assert r.status_code != 404


def test_post_and_get_at_mode_with_auth(api_client):
    headers = {"X-Master-Password": "testpw"}
    r = api_client.post("/api/trader/at-mode", json={"mode": "live"}, headers=headers)
    assert r.status_code in (200, 503)
    assert r.status_code != 404
    if r.status_code == 200:
        assert r.get_json()["mode"] == "live"
        g = api_client.get("/api/trader/at-mode")
        assert g.status_code == 200
        assert g.get_json()["mode"] == "live"
        # reset to paper so we don't leave shared state live
        api_client.post("/api/trader/at-mode", json={"mode": "paper"}, headers=headers)
