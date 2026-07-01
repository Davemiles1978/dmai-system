"""Tests for SelfHealer heartbeat-aware liveness + env-gated skip (PR #164).

Post-#162 the self-healer restarted greyhound_runner and kaizen_auto_repair in
a 7.2 min window even though both were correctly configured (greyhound sleeps
~24h between fires, kaizen is disabled). The liveness heuristic treated a
sleeping / stopped component as "dead".

Fixes covered here:
  1. A fresh ``data/<component>_heartbeat.txt`` (or ``_last_*.txt`` marker)
     proves liveness for rate-limited loops — no restart.
  2. Env-gated components (kaizen_auto_repair via KAIZEN_AUTO_REPAIR_ENABLED)
     are skipped entirely when the flag is off.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.self_management.self_healer import SelfHealer


class _FakeThread:
    def __init__(self, alive):
        self._alive = alive

    def is_alive(self):
        return self._alive


class _FakeComp:
    def __init__(self, alive):
        self._thread = _FakeThread(alive)


def _healer_past_grace(components, root):
    h = SelfHealer(components=components, repo_root=str(root))
    h._started_ts = time.time() - 100_000  # well past the 5-min grace window
    return h


def test_fresh_heartbeat_marks_component_alive(tmp_path):
    (tmp_path / "data").mkdir()
    hb = tmp_path / "data" / "greyhound_runner_heartbeat.txt"
    hb.write_text("ok")  # mtime = now
    h = SelfHealer(repo_root=str(tmp_path))
    assert h._alive_by_heartbeat("greyhound_runner") is True


def test_last_fire_marker_marks_component_alive(tmp_path):
    (tmp_path / "data").mkdir()
    marker = tmp_path / "data" / "greyhound_runner_last_tip_date.txt"
    marker.write_text("2026-07-01")
    h = SelfHealer(repo_root=str(tmp_path))
    assert h._alive_by_heartbeat("greyhound_runner") is True


def test_missing_heartbeat_is_not_alive(tmp_path):
    (tmp_path / "data").mkdir()
    h = SelfHealer(repo_root=str(tmp_path))
    assert h._alive_by_heartbeat("greyhound_runner") is False


def test_stale_heartbeat_is_not_alive(tmp_path):
    (tmp_path / "data").mkdir()
    hb = tmp_path / "data" / "learning_orchestrator_heartbeat.txt"
    hb.write_text("old")
    # learning_orchestrator max age is 30 min; push mtime an hour into the past.
    old = time.time() - 3600
    import os
    os.utime(hb, (old, old))
    h = SelfHealer(repo_root=str(tmp_path))
    assert h._alive_by_heartbeat("learning_orchestrator") is False


def _restarted_names(mock):
    # _check_threads also probes always-expected threads (ai-discovery,
    # tutor-config) that are absent in the test process; ignore those and look
    # only at which registry component keys were restarted.
    return {c.args[0] for c in mock.call_args_list}


def test_fresh_heartbeat_prevents_restart(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "greyhound_runner_heartbeat.txt").write_text("ok")
    comps = {"greyhound_runner": _FakeComp(alive=False)}  # thread looks dead
    h = _healer_past_grace(comps, tmp_path)
    h._try_restart_thread = MagicMock()
    h._check_threads()
    assert "greyhound_runner" not in _restarted_names(h._try_restart_thread)


def test_kaizen_skipped_when_env_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("KAIZEN_AUTO_REPAIR_ENABLED", raising=False)
    comps = {"kaizen_auto_repair": _FakeComp(alive=False)}
    h = _healer_past_grace(comps, tmp_path)
    h._try_restart_thread = MagicMock()
    h._check_threads()
    assert "kaizen_auto_repair" not in _restarted_names(h._try_restart_thread)


def test_kaizen_restarted_when_env_enabled_and_dead(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZEN_AUTO_REPAIR_ENABLED", "1")
    comps = {"kaizen_auto_repair": _FakeComp(alive=False)}
    h = _healer_past_grace(comps, tmp_path)
    h._try_restart_thread = MagicMock()
    h._check_threads()
    assert "kaizen_auto_repair" in _restarted_names(h._try_restart_thread)
