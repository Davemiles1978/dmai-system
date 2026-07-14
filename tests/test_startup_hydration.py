"""Tests for _bootstrap_api_key_hydration (PR O).

The helper pushes DB-stored API keys into os.environ before AutoAPIActivator
runs its first validation pass, so providers don't show pending_api_key after a
Render redeploy. These tests lock its contract: idempotent, DB→env population,
env-wins, and graceful DB-failure handling.

DATA_PATH is pointed at a temp dir *before* importing the app so boot side
effects stay isolated. Env vars are only ever mutated via monkeypatch so
nothing leaks into the wider pytest session.
"""
from __future__ import annotations

import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="startup_hydration_")
os.environ["DATA_PATH"] = _TMP

import dmai_core_complete  # noqa: E402
from dmai_core_complete import _bootstrap_api_key_hydration  # noqa: E402


class _StubStorage:
    """Minimal db_storage stub returning known keys by provider id."""

    def __init__(self, keys):
        self._keys = keys

    def get_api_key(self, provider_id):
        return self._keys.get(provider_id, "")


def test_hydration_is_idempotent(monkeypatch):
    monkeypatch.setitem(dmai_core_complete.components, "db_storage",
                        _StubStorage({"openai": "sk-openai-xyz"}))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    first = _bootstrap_api_key_hydration()
    assert first["db_ready"] is True
    assert "openai" in first["hydrated"]
    assert os.environ["OPENAI_API_KEY"] == "sk-openai-xyz"

    # Second call: env already populated -> nothing new hydrated.
    second = _bootstrap_api_key_hydration()
    assert second["db_ready"] is True
    assert second["hydrated"] == []


def test_hydration_populates_empty_env(monkeypatch):
    monkeypatch.setitem(dmai_core_complete.components, "db_storage",
                        _StubStorage({"openai": "sk-from-db"}))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    out = _bootstrap_api_key_hydration()
    assert "openai" in out["hydrated"]
    assert os.environ["OPENAI_API_KEY"] == "sk-from-db"


def test_hydration_does_not_overwrite_existing_env(monkeypatch):
    monkeypatch.setitem(dmai_core_complete.components, "db_storage",
                        _StubStorage({"openai": "sk-from-db"}))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-render")

    out = _bootstrap_api_key_hydration()
    # Env wins — the DB value must NOT clobber the pre-set env var.
    assert "openai" not in out["hydrated"]
    assert os.environ["OPENAI_API_KEY"] == "sk-from-render"


def test_hydration_handles_db_init_failure(monkeypatch):
    # No db_storage present, and pg_storage.get_storage raises.
    monkeypatch.delitem(dmai_core_complete.components, "db_storage", raising=False)

    import components.pg_storage as pg
    def _boom():
        raise RuntimeError("postgres not ready")
    monkeypatch.setattr(pg, "get_storage", _boom)

    out = _bootstrap_api_key_hydration()
    assert out["db_ready"] is False
    assert out["hydrated"] == []
