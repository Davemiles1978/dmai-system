"""PR #167: verify the write-on-change throttle in `_write_stage_to_db`.

`_write_stage_to_db` uses wallclock (`time.time()`) for its throttle window, so we
freeze `time.time` via monkeypatch and count how many times a DB connection is
opened (`safe_open_kdb`) — one open == one SQLite write attempt.
"""
import time

import pytest

import dmai_core_complete as d


class _FakeCursor:
    def fetchone(self):
        return None


class _FakeConn:
    def __init__(self):
        self.row_factory = None

    def execute(self, *a, **k):
        return _FakeCursor()

    def commit(self):
        pass

    def close(self):
        pass


_M = {"insights": 1, "capabilities": 1, "vocab": 1, "avg_kpi": 0.5}


@pytest.fixture
def patched(monkeypatch):
    """Isolate `_write_stage_to_db`: fake DB opens (counted), no sidecar I/O,
    frozen clock, and a reset in-memory cache."""
    calls = {"open": 0}

    def fake_open(*a, **k):
        calls["open"] += 1
        return _FakeConn()

    monkeypatch.setattr(d, "safe_open_kdb", fake_open)
    monkeypatch.setattr(d, "_write_stage_sidecar", lambda *a, **k: None)

    fake_now = {"t": 1000.0}
    monkeypatch.setattr(time, "time", lambda: fake_now["t"])

    d._LAST_STAGE_WRITTEN.update({"stage": None, "pct": None, "written_at": 0.0})
    return calls, fake_now


def test_unchanged_within_interval_single_write(patched):
    """Two calls with the same stage inside the interval → only ONE SQLite write."""
    calls, fake_now = patched
    d._write_stage_to_db("stage_Baby", 10.0, _M)   # None -> Baby: writes
    fake_now["t"] += 30                             # 30s later (< 300s)
    d._write_stage_to_db("stage_Baby", 11.0, _M)   # unchanged: skipped
    assert calls["open"] == 1


def test_stage_change_writes_immediately(patched):
    """A stage transition writes immediately, regardless of the interval floor."""
    calls, fake_now = patched
    d._write_stage_to_db("stage_Baby", 10.0, _M)      # writes (open == 1)
    fake_now["t"] += 5                                # only 5s later
    d._write_stage_to_db("stage_Toddler", 20.0, _M)   # changed: writes (open == 2)
    assert calls["open"] == 2


def test_periodic_touch_after_interval(patched):
    """Same stage but after _STAGE_MIN_WRITE_INTERVAL_S+1 → periodic-touch write."""
    calls, fake_now = patched
    d._write_stage_to_db("stage_Baby", 10.0, _M)                    # writes (open == 1)
    fake_now["t"] += d._STAGE_MIN_WRITE_INTERVAL_S + 1              # 301s later
    d._write_stage_to_db("stage_Baby", 12.0, _M)                   # touch: writes (open == 2)
    assert calls["open"] == 2
