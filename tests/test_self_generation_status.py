"""PR EE tests — self-generation status aggregator."""
from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

# Repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components import self_generation_status as sgs  # noqa: E402


# ── DB fixture ────────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path: Path) -> str:
    """Minimal DMAI-ish schema for status queries."""
    p = tmp_path / "knowledge.db"
    conn = sqlite3.connect(p)
    conn.executescript("""
        CREATE TABLE system_state (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE capabilities (
            id TEXT PRIMARY KEY,
            name TEXT,
            provenance TEXT,
            judge_confidence REAL,
            runtime_mode TEXT,
            created_ts TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE verification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            capability_id TEXT,
            slug TEXT,
            stage TEXT,
            ok INTEGER,
            reason TEXT,
            duration_ms REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE materialisation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            capability_id TEXT,
            concept TEXT,
            slug TEXT,
            outcome TEXT,
            model_used TEXT,
            reasons TEXT,
            judge_confidence REAL,
            duration_sec REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()
    return str(p)


def _insert_cap(db: str, **kw) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO capabilities (id, name, provenance, judge_confidence, "
        "                          runtime_mode) VALUES (?, ?, ?, ?, ?)",
        (kw["id"], kw.get("name", kw["id"]),
         kw["provenance"], kw["judge_confidence"], kw["runtime_mode"]),
    )
    conn.commit()
    conn.close()


def _insert_verify(db: str, cap_id: str, ok: int,
                   stage: str = "orchestrator", reason: str = "") -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO verification_log "
        "(capability_id, slug, stage, ok, reason, duration_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (cap_id, cap_id, stage, ok, reason, 42.0),
    )
    conn.commit()
    conn.close()


def _set_state(db: str, key: str, value: str) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()
    conn.close()


# ── Structural tests ──────────────────────────────────────────────────────

def test_build_status_empty_db(db: str) -> None:
    """A brand-new DB returns ok=True with red/yellow health."""
    payload = sgs.build_status(db)
    assert payload["ok"] is True
    assert set(payload.keys()) >= {
        "ok", "ts", "health", "materialiser", "verifier",
        "queue", "live_modules", "gaps",
    }
    assert payload["health"]["level"] in ("green", "yellow", "red")
    assert payload["verifier"]["totals"]["total"] == 0
    assert payload["live_modules"]["live_count_db"] == 0
    assert payload["queue"]["total"] == 0


def test_build_status_missing_db(tmp_path: Path) -> None:
    payload = sgs.build_status(str(tmp_path / "does-not-exist.db"))
    assert payload["ok"] is False
    assert payload["health"]["level"] == "red"


# ── Verifier section ──────────────────────────────────────────────────────

def test_verifier_success_rate(db: str) -> None:
    # 8 pass, 2 fail => 80%
    for i in range(8):
        _insert_verify(db, f"ok_{i}", ok=1)
    for i in range(2):
        _insert_verify(db, f"bad_{i}", ok=0, reason="boom")

    payload = sgs.build_status(db)
    v = payload["verifier"]
    assert v["totals"] == {"successes": 8, "failures": 2, "total": 10}
    # Rate should be >= 0.75 => healthy
    assert v["window_success_rate"] == pytest.approx(0.8)
    assert len(v["recent"]) == 10


def test_verifier_low_rate_triggers_red(db: str) -> None:
    # 3 pass, 7 fail => 30% => red
    for i in range(3):
        _insert_verify(db, f"ok_{i}", ok=1)
    for i in range(7):
        _insert_verify(db, f"bad_{i}", ok=0)

    payload = sgs.build_status(db)
    assert payload["health"]["level"] == "red"
    reasons = " ".join(payload["health"]["reasons"])
    assert "verification success rate" in reasons


def test_quarantined_capabilities_flagged(db: str) -> None:
    # Insert a live module so we don't get 'no live modules' yellow
    _insert_cap(db, id="live_1", provenance="fresh_blood_seed+self_judge",
                judge_confidence=0.9, runtime_mode="generated_module")
    for i in range(5):
        _insert_cap(db, id=f"q_{i}",
                    provenance="fresh_blood_seed+self_judge",
                    judge_confidence=0.9, runtime_mode="quarantined")

    payload = sgs.build_status(db)
    assert payload["verifier"]["runtime_mode_counts"]["quarantined"] == 5
    # 5+ quarantined should push us at least to yellow
    assert payload["health"]["level"] in ("yellow", "red")


# ── Queue section ─────────────────────────────────────────────────────────

def test_queue_depth_per_pool(db: str) -> None:
    # 4 fresh_blood, 2 promoter_path, 1 gap_driven above floor
    for i in range(4):
        _insert_cap(db, id=f"fb_{i}",
                    provenance="fresh_blood_seed+self_judge",
                    judge_confidence=0.9, runtime_mode="stub")
    for i in range(2):
        _insert_cap(db, id=f"pp_{i}",
                    provenance="promoter_path+self_judge",
                    judge_confidence=0.75, runtime_mode="stub")
    _insert_cap(db, id="gd_1", provenance="gap_driven",
                judge_confidence=0.65, runtime_mode="stub")

    # Also insert something below the floor, should be excluded
    _insert_cap(db, id="low", provenance="fresh_blood_seed+self_judge",
                judge_confidence=0.30, runtime_mode="stub")

    payload = sgs.build_status(db)
    q = payload["queue"]
    assert q["by_provenance"]["fresh_blood_seed+self_judge"] == 4
    assert q["by_provenance"]["promoter_path+self_judge"] == 2
    assert q["by_provenance"]["gap_driven"] == 1
    assert q["total"] == 7
    assert q["min_confidence"] == pytest.approx(0.60)


def test_queue_includes_stub_reverted(db: str) -> None:
    _insert_cap(db, id="fb_reverted",
                provenance="fresh_blood_seed+self_judge",
                judge_confidence=0.9, runtime_mode="stub_reverted")
    payload = sgs.build_status(db)
    assert payload["queue"]["by_provenance"][
        "fresh_blood_seed+self_judge"] == 1


def test_queue_excludes_quarantined_and_live(db: str) -> None:
    _insert_cap(db, id="q1", provenance="fresh_blood_seed+self_judge",
                judge_confidence=0.9, runtime_mode="quarantined")
    _insert_cap(db, id="l1", provenance="fresh_blood_seed+self_judge",
                judge_confidence=0.9, runtime_mode="generated_module")
    payload = sgs.build_status(db)
    assert payload["queue"]["total"] == 0


# ── Materialiser section ──────────────────────────────────────────────────

def test_materialiser_reads_last_summary(db: str) -> None:
    summary = {
        "ts": "2026-07-15T17:00:00+00:00",
        "picked": 3, "promoted": 2, "failed": 1,
        "day_count": 3, "cap_hit": False,
        "gaps_seeded": 2,
        "provenance_breakdown": {"fresh_blood_seed+self_judge": 2,
                                  "gap_driven": 1},
    }
    # We need to know the key names from the real module.
    from components.capability_materialiser import (
        STATE_KEY_LAST_RUN, STATE_KEY_LAST_SUMMARY,
    )
    _set_state(db, STATE_KEY_LAST_SUMMARY, json.dumps(summary))
    _set_state(db, STATE_KEY_LAST_RUN, "2026-07-15T17:00:00+00:00")

    payload = sgs.build_status(db)
    m = payload["materialiser"]
    assert m["last_tick"]["picked"] == 3
    assert m["last_tick"]["promoted"] == 2
    assert m["last_tick"]["gaps_seeded"] == 2
    assert m["last_tick"]["provenance_breakdown"][
        "fresh_blood_seed+self_judge"] == 2
    assert m["config"]["daily_cap"] == 10           # PR DD default
    assert m["config"]["min_confidence"] == 0.60    # PR DD default
    assert set(m["config"]["quotas"].keys()) == {
        "fresh_blood_seed+self_judge",
        "promoter_path+self_judge",
        "gap_driven",
    }


# ── Live modules section ──────────────────────────────────────────────────

def test_live_modules_count_and_recent(db: str) -> None:
    for i in range(3):
        _insert_cap(db, id=f"live_{i}",
                    provenance="fresh_blood_seed+self_judge",
                    judge_confidence=0.9, runtime_mode="generated_module")
    conn = sqlite3.connect(db)
    for i in range(3):
        conn.execute(
            "INSERT INTO materialisation_log "
            "(capability_id, concept, slug, outcome, model_used) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"live_{i}", f"concept_{i}", f"live_{i}",
             "promoted", "codex"),
        )
    conn.commit()
    conn.close()

    payload = sgs.build_status(db)
    live = payload["live_modules"]
    assert live["live_count_db"] == 3
    assert len(live["recent_promotions"]) == 3


# ── Gaps section ──────────────────────────────────────────────────────────

def test_gaps_section_uses_gap_fetcher(db: str, monkeypatch) -> None:
    """Patch iter_capability_gaps and confirm we surface it."""
    class _G:
        def __init__(self, name, prio):
            self.name = name
            self.description = f"desc for {name}"
            self.priority = prio
            self.target_kpi = "kpi"

    def fake_iter(fresh: bool = False):
        return iter([_G("a", 1), _G("b", 5), _G("c", 3)])

    # Inject a fake gap_fetcher module
    fake_mod = types.ModuleType("components.gap_fetcher")
    fake_mod.iter_capability_gaps = fake_iter
    monkeypatch.setitem(sys.modules, "components.gap_fetcher", fake_mod)

    payload = sgs.build_status(db)
    g = payload["gaps"]
    assert g["count"] == 3
    # Sorted by priority ascending
    assert g["top"][0]["name"] == "a"
    assert g["top"][1]["name"] == "c"
    assert g["top"][2]["name"] == "b"


# ── Health rollup ─────────────────────────────────────────────────────────

def test_health_green_with_live_and_good_verify(db: str) -> None:
    # Live module + 80% verification + queue populated + materialiser running
    _insert_cap(db, id="live", provenance="fresh_blood_seed+self_judge",
                judge_confidence=0.9, runtime_mode="generated_module")
    _insert_cap(db, id="fb", provenance="fresh_blood_seed+self_judge",
                judge_confidence=0.9, runtime_mode="stub")
    for i in range(8):
        _insert_verify(db, f"ok_{i}", ok=1)
    for i in range(2):
        _insert_verify(db, f"bad_{i}", ok=0)

    # Pretend materialiser loop is alive.
    # CapabilityMaterialiserLoop wraps a threading.Thread in ._thread
    # — mirror that shape so the running check matches production.
    from components import capability_materialiser
    original_loop = getattr(capability_materialiser, "_LOOP", None)
    try:
        class _FakeThread:
            def is_alive(self):
                return True

        class _FakeLoop:
            _thread = _FakeThread()
        capability_materialiser._LOOP = _FakeLoop()

        payload = sgs.build_status(db)
        assert payload["materialiser"]["running"] is True
        # Should end up green — no red/yellow triggers
        assert payload["health"]["level"] == "green"
    finally:
        capability_materialiser._LOOP = original_loop


def test_health_yellow_when_queue_empty(db: str) -> None:
    _insert_cap(db, id="live", provenance="fresh_blood_seed+self_judge",
                judge_confidence=0.9, runtime_mode="generated_module")
    payload = sgs.build_status(db)
    # Queue empty + materialiser not running in this test = red-worthy,
    # but at minimum queue-empty should register as a reason.
    reasons = " ".join(payload["health"]["reasons"])
    assert "queue empty" in reasons.lower() or "not running" in reasons.lower()


# ── Endpoint smoke test via test client ───────────────────────────────────

def test_endpoint_returns_200_and_shape(monkeypatch, tmp_path):
    """Hit the Flask route directly with a stub DB path."""
    # Build a valid tiny DB
    db_path = tmp_path / "kn.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE system_state (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE capabilities (
            id TEXT PRIMARY KEY, name TEXT, provenance TEXT,
            judge_confidence REAL, runtime_mode TEXT,
            created_ts TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE verification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            capability_id TEXT, slug TEXT, stage TEXT,
            ok INTEGER, reason TEXT, duration_ms REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE materialisation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            capability_id TEXT, concept TEXT, slug TEXT, outcome TEXT,
            model_used TEXT, reasons TEXT, judge_confidence REAL,
            duration_sec REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()

    from components import capability_materialiser
    monkeypatch.setattr(capability_materialiser, "DEFAULT_DB_PATH",
                        str(db_path))

    import dmai_core_complete as dcc
    dcc.app.config["TESTING"] = True
    client = dcc.app.test_client()
    resp = client.get("/api/self-generation/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert "health" in data
    assert "materialiser" in data
    assert "verifier" in data
    assert "queue" in data
    assert "live_modules" in data
    assert "gaps" in data
