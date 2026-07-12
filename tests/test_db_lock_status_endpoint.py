"""Tests for the /api/admin/db-lock-status diagnostic endpoint and its
helper components.db.get_write_lock_status.
"""
from __future__ import annotations

import threading
import time

import pytest

from components.db import (
    _get_write_lock,
    acquire_write_lock,
    get_write_lock_status,
)


def test_get_write_lock_status_reports_no_holder_when_free(tmp_path):
    db = str(tmp_path / "test_lock_free.db")
    # Touch the lock so it exists in the registry.
    _get_write_lock(db)
    snap = get_write_lock_status()
    assert db in snap
    # No writer active -> currently_held is False, no stack.
    assert snap[db]["currently_held"] is False
    assert snap[db]["holder_stack"] == []


def test_get_write_lock_status_reports_holder_when_held(tmp_path):
    db = str(tmp_path / "test_lock_held.db")
    ready = threading.Event()
    release = threading.Event()

    def holder():
        with acquire_write_lock(db):
            ready.set()
            release.wait(timeout=5)

    t = threading.Thread(target=holder, name="test_lock_holder", daemon=True)
    t.start()
    assert ready.wait(timeout=5)
    try:
        snap = get_write_lock_status()
        assert db in snap
        assert snap[db]["currently_held"] is True
        assert snap[db]["holder_thread_ident"] == t.ident
        assert snap[db]["holder_thread_name"] == "test_lock_holder"
        # Stack should have at least one frame from our holder function.
        stack_joined = "\n".join(snap[db]["holder_stack"])
        assert "holder" in stack_joined or "acquire" in stack_joined
    finally:
        release.set()
        t.join(timeout=5)


def test_get_write_lock_status_never_raises_on_stale_ident():
    """Even if _WRITE_LOCK_HOLDERS still points at a thread that already exited,
    the snapshot must return without raising.
    """
    from components.db import _WRITE_LOCK_HOLDERS, _get_write_lock
    key, _ = _get_write_lock("/tmp/nonexistent_test.db")
    # Inject a bogus thread ident.
    _WRITE_LOCK_HOLDERS[key] = 999999999
    try:
        snap = get_write_lock_status()
        assert key in snap
        # holder_thread_name is None because the ident doesn't map to any live thread.
        assert snap[key]["holder_thread_name"] is None
    finally:
        _WRITE_LOCK_HOLDERS.pop(key, None)
