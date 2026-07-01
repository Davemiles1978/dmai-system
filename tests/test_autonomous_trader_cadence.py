"""Tests for AutonomousTrader cadence + live-mode toggle (PR #163).

Post-#162 the trader became the dominant write-mutex holder: a 5-min loop tick
hitting ``c.execute()`` under SQLite write pressure produced a steady
``database is locked`` storm. PR #163 rate-limits the loop to a default 2h
cadence with a dashboard-controlled ``live`` override (30s ticks) that
auto-expires back to ``scheduled`` after 4h.

These are unit-level tests of the scheduling helpers, the mode-file contract,
the heartbeat, the MANUAL-trade guardrail, and the /api/trader/mode routes.
They do not exercise real lock contention or the network.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.wealth import autonomous_trader as at_mod
from components.wealth.autonomous_trader import AutonomousTrader

CADENCE_ENV = (
    "AUTONOMOUS_TRADER_TICK_INTERVAL_SECONDS",
    "AUTONOMOUS_TRADER_LIVE_INTERVAL_SECONDS",
    "AUTONOMOUS_TRADER_LIVE_MAX_MINUTES",
)


@pytest.fixture(autouse=True)
def _clean_cadence_env(monkeypatch):
    """Each test starts from documented defaults (7200 / 30 / 240)."""
    for name in CADENCE_ENV:
        monkeypatch.delenv(name, raising=False)


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
    at.stop()  # halt the background loop; we drive helpers directly
    return at


# ── Defaults + env overrides ──────────────────────────────────────────────────

def test_default_mode_scheduled_and_2h_interval(tmp_path):
    at = _make_trader(tmp_path)
    assert at_mod._tick_interval_s() == 7200
    assert at._read_mode() == "scheduled"
    interval, mode = at._effective_interval()
    assert mode == "scheduled"
    assert interval == 7200


def test_tick_interval_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTONOMOUS_TRADER_TICK_INTERVAL_SECONDS", "60")
    at = _make_trader(tmp_path)
    assert at_mod._tick_interval_s() == 60
    interval, mode = at._effective_interval()
    assert (interval, mode) == (60, "scheduled")


def test_live_and_max_env_defaults(tmp_path):
    _make_trader(tmp_path)
    assert at_mod._live_interval_s() == 30
    assert at_mod._live_max_minutes() == 240


def test_bad_env_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTONOMOUS_TRADER_TICK_INTERVAL_SECONDS", "not-an-int")
    assert at_mod._tick_interval_s() == 7200


# ── Live mode ─────────────────────────────────────────────────────────────────

def test_live_mode_uses_live_interval(tmp_path):
    at = _make_trader(tmp_path)
    at.set_mode("live")
    interval, mode = at._effective_interval()
    assert mode == "live"
    assert interval == at_mod._live_interval_s() == 30


def test_mode_change_detected_shortens_interval(tmp_path):
    """Mirrors the loop's mid-sleep break: flipping to live yields a shorter
    interval so the sleep loop wakes and ticks promptly (within ~10s)."""
    at = _make_trader(tmp_path)
    sched_interval, sched_mode = at._effective_interval()
    assert sched_mode == "scheduled" and sched_interval == 7200

    at.set_mode("live")
    new_interval, new_mode = at._effective_interval()
    assert new_mode == "live"
    assert new_interval < sched_interval  # loop would break out of the 2h sleep


def test_live_mode_auto_expiry_reverts_and_rewrites(tmp_path):
    at = _make_trader(tmp_path)
    at.set_mode("live")
    # Age the mode file well past LIVE_MAX_MINUTES (240m).
    old = time.time() - (5 * 3600)
    os.utime(at.mode_file, (old, old))

    st = at.mode_status(mutate_expiry=True)
    assert st["mode"] == "scheduled"
    assert st["expires_at"] is None
    # File was rewritten by the auto-expiry.
    with open(at.mode_file, encoding="utf-8") as fh:
        assert fh.read().strip() == "scheduled"
    # And the effective interval is back to the 2h scheduled cadence.
    assert at._effective_interval() == (7200, "scheduled")


def test_get_mode_status_does_not_mutate(tmp_path):
    at = _make_trader(tmp_path)
    at.set_mode("live")
    old = time.time() - (5 * 3600)
    os.utime(at.mode_file, (old, old))
    # Non-mutating read reports scheduled (expired) but must NOT rewrite the file.
    st = at.mode_status(mutate_expiry=False)
    assert st["mode"] == "scheduled"
    with open(at.mode_file, encoding="utf-8") as fh:
        assert fh.read().strip() == "live"  # untouched


def test_set_mode_rejects_invalid(tmp_path):
    at = _make_trader(tmp_path)
    with pytest.raises(ValueError):
        at.set_mode("turbo")


def test_mode_status_shape_live_has_expiry(tmp_path):
    at = _make_trader(tmp_path)
    st = at.set_mode("live")
    assert set(st) == {"mode", "expires_at", "next_tick_at"}
    assert st["mode"] == "live"
    assert st["expires_at"] is not None
    # expires_at ~ now + 240m; next_tick_at ~ now + 30s. Both parse as ISO-8601.
    datetime.fromisoformat(st["expires_at"])
    datetime.fromisoformat(st["next_tick_at"])


# ── Heartbeat ─────────────────────────────────────────────────────────────────

def test_heartbeat_written_and_refreshed(tmp_path):
    at = _make_trader(tmp_path)
    at._write_heartbeat("scheduled", 120)
    assert os.path.exists(at.heartbeat_file)
    with open(at.heartbeat_file, encoding="utf-8") as fh:
        first = datetime.fromisoformat(fh.read().strip())

    time.sleep(0.01)
    at._write_heartbeat("live", 5)
    with open(at.heartbeat_file, encoding="utf-8") as fh:
        second = datetime.fromisoformat(fh.read().strip())
    assert second >= first


# ── MANUAL-trade guardrail ────────────────────────────────────────────────────

def test_tick_does_not_auto_execute_when_disabled(tmp_path):
    """The only broker-execution call reachable from _tick_inner is
    self.trader.execute_buy, gated behind the enabled flag + market-open. With
    the loop disabled (default), a tick must never place a trade."""
    at = _make_trader(tmp_path)
    # Default state row has enabled = 0.
    at.tick()
    assert at.trader.execute_buy.call_count == 0
    for banned in ("place_trade", "place_order", "submit_order", "auto_execute"):
        assert not hasattr(at, banned), (
            f"unexpected auto-execute method {banned!r} on AutonomousTrader"
        )


# ── API routes: /api/trader/mode ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_client():
    os.environ.setdefault("RENDER", "false")
    data_dir = "/tmp/dmai_cadence_api_data"
    os.makedirs(data_dir, exist_ok=True)
    os.environ.setdefault("DATA_PATH", data_dir)
    os.environ.setdefault("MASTER_PASSWORD", "testpw")
    os.environ.setdefault("JWT_SECRET", "testjwt_cadence_only")
    try:
        import dmai_core_complete as dcc
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"dmai_core_complete import unavailable: {e}")
    dcc.app.config["TESTING"] = True
    return dcc.app.test_client()


def test_post_mode_requires_auth(api_client):
    r = api_client.post("/api/trader/mode", json={"mode": "live"})
    assert r.status_code == 401


def test_post_and_get_mode_with_auth(api_client):
    headers = {"X-Master-Password": "testpw"}
    r = api_client.post("/api/trader/mode", json={"mode": "live"}, headers=headers)
    # 200 when the trader component is loaded; 503 if not booted in this env.
    assert r.status_code in (200, 503)
    assert r.status_code != 404
    if r.status_code == 200:
        assert r.get_json()["mode"] == "live"
        g = api_client.get("/api/trader/mode")
        assert g.status_code == 200
        body = g.get_json()
        assert body["mode"] in ("live", "scheduled")
        assert "expires_at" in body and "next_tick_at" in body


def test_post_invalid_mode_rejected(api_client):
    headers = {"X-Master-Password": "testpw"}
    r = api_client.post("/api/trader/mode", json={"mode": "turbo"}, headers=headers)
    # 400 when trader loaded (ValueError path); 503 if not booted.
    assert r.status_code in (400, 503)
