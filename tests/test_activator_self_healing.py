"""Tests for the self-healing AutoAPIActivator (PR T).

Covers: DB→env hydration before every scan, active→pending regression
detection + warning logging, the one-shot self-heal rescan scheduling (and that
it is NOT scheduled on a clean first scan), and the shortened check interval.

A tiny 2-provider fake catalogue keeps scans instant and offline (validation is
stubbed). Env is only ever mutated via monkeypatch so nothing leaks into the
wider pytest session.
"""
from __future__ import annotations

import logging
import os
import tempfile

import pytest

import components.integration.auto_api_activator as aaa
from components.integration.auto_api_activator import AutoAPIActivator

_FAKE_CATALOGUE = {
    "fake_provider": {
        "name":       "Fake Provider",
        "env_vars":   ["FAKE_KEY"],
        "signup_url": "https://example.com/keys",
        "free_tier":  "unlimited (fake)",
        "models":     ["fake-model-1"],
        "best_model": "fake-model-1",
        "validation": {"method": "GET", "url": "https://example.com", "headers": lambda k: {}, "body": None},
    },
    "other_provider": {
        "name":       "Other Provider",
        "env_vars":   ["OTHER_KEY"],
        "signup_url": "https://example.org/keys",
        "free_tier":  "unlimited (fake)",
        "models":     ["other-model-1"],
        "best_model": "other-model-1",
        "validation": {"method": "GET", "url": "https://example.org", "headers": lambda k: {}, "body": None},
    },
}


def _make_activator(tmp_path, monkeypatch, offline=True):
    """Build an activator over the fake catalogue with validation stubbed."""
    monkeypatch.setattr(aaa, "PROVIDER_CATALOGUE", _FAKE_CATALOGUE)
    act = AutoAPIActivator(ai_hub=None, data_path=str(tmp_path))
    # Any key that _find_key returns is treated as valid — no network.
    monkeypatch.setattr(act, "_validate",
                        lambda pid, spec, key: {"status": "active", "latency_ms": 1})
    return act


def test_check_interval_is_900():
    assert AutoAPIActivator.CHECK_INTERVAL == 900


def test_hydrate_before_scan_pulls_from_db(monkeypatch, tmp_path):
    # Point dmai_core_complete's hydration at a DB stub holding our fake key.
    import dmai_core_complete as core

    class _StubStorage:
        def get_api_key(self, provider_id):
            return "test-key-abc" if provider_id == "fake_provider" else ""

    monkeypatch.setattr(core, "_PROVIDER_REGISTRY",
                        [("fake_provider", "Fake", "FAKE_KEY", "https://example.com")])
    monkeypatch.setitem(core.components, "db_storage", _StubStorage())
    monkeypatch.delenv("FAKE_KEY", raising=False)  # auto-removes anything set during test

    act = _make_activator(tmp_path, monkeypatch)
    act.scan_and_activate()

    # Hydration should have pushed the DB key into the process env.
    assert os.environ.get("FAKE_KEY") == "test-key-abc"


def test_no_selfheal_on_first_scan(monkeypatch, tmp_path):
    act = _make_activator(tmp_path, monkeypatch)
    monkeypatch.setattr(act, "_hydrate_from_db_before_scan", lambda: None)
    monkeypatch.delenv("FAKE_KEY", raising=False)
    monkeypatch.delenv("OTHER_KEY", raising=False)

    called = []
    monkeypatch.setattr(act, "_schedule_selfheal_rescan", lambda: called.append(1))

    results = act.scan_and_activate()
    # No prior state → no regression → no self-heal.
    assert called == []
    assert "fake_provider" in results["pending"]


def test_regression_detection_logs_warning(monkeypatch, tmp_path, caplog):
    act = _make_activator(tmp_path, monkeypatch)
    monkeypatch.setattr(act, "_hydrate_from_db_before_scan", lambda: None)
    monkeypatch.setattr(act, "_schedule_selfheal_rescan", lambda: None)

    # Scan 1: key present → active.
    monkeypatch.setenv("FAKE_KEY", "sk-live")
    monkeypatch.delenv("OTHER_KEY", raising=False)
    first = act.scan_and_activate()
    assert "fake_provider" in first["activated"]

    # Scan 2: key gone → pending → regression.
    monkeypatch.delenv("FAKE_KEY", raising=False)
    with caplog.at_level(logging.WARNING, logger=aaa.logger.name):
        second = act.scan_and_activate()
    assert "fake_provider" in second["pending"]
    assert any("REGRESSION detected" in r.message for r in caplog.records)


def test_selfheal_rescan_scheduled_on_regression(monkeypatch, tmp_path):
    act = _make_activator(tmp_path, monkeypatch)
    monkeypatch.setattr(act, "_hydrate_from_db_before_scan", lambda: None)

    calls = []
    monkeypatch.setattr(act, "_schedule_selfheal_rescan", lambda: calls.append(1))

    monkeypatch.setenv("FAKE_KEY", "sk-live")
    monkeypatch.delenv("OTHER_KEY", raising=False)
    act.scan_and_activate()          # active
    assert calls == []

    monkeypatch.delenv("FAKE_KEY", raising=False)
    act.scan_and_activate()          # regressed → schedule once
    assert calls == [1]
