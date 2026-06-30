"""Tests for VocabularyIngester single-flight idle guard + finally-block log (PR #159).

Two behaviours are guarded here:

  1. The IDLE flush path is single-flighted: if a prior idle flush is still
     running, a second idle tick is skipped (no second write-mutex acquisition,
     so no vocab-vs-vocab self-collision). The public ``flush()`` API is NOT
     single-flighted — explicit callers must always write.
  2. The ``vocab flush: ...`` success-summary INFO log fires on EVERY flush —
     the happy path AND the per-row fallback path — because it now lives in a
     ``finally:`` block.

These are fast, contract-level tests; they do not exercise real lock contention.
"""

import logging
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.knowledge import vocabulary_ingester as vi
from components.knowledge.vocabulary_ingester import VocabularyIngester


def _make_ingester(tmp_path) -> VocabularyIngester:
    """Ingester pointed at a temp DB with the background idle-flush timer
    disabled so tests drive the idle tick explicitly."""
    ing = VocabularyIngester(db_path=str(tmp_path / "kdb.db"))
    ing.flush_seconds = 0  # no background thread; tests call _idle_flush_tick directly
    return ing


def _push_rows(ing: VocabularyIngester, n: int) -> None:
    for i in range(n):
        ing.ingest_one({"word": f"word{i}", "definition": f"definition number {i}"})


def test_idle_flush_single_flight(tmp_path):
    """A second idle tick fired while the first is still flushing must skip:
    ``_write_batch`` is entered exactly once."""
    ing = _make_ingester(tmp_path)
    _push_rows(ing, 5)
    # Idle tick only flushes when the buffer has been idle >= flush_seconds.
    ing.flush_seconds = 0.0
    ing._last_add_ts = time.monotonic() - 10

    call_count = {"n": 0}
    started = threading.Event()
    real_write_batch = ing._write_batch

    def slow_write_batch(rows):
        call_count["n"] += 1
        started.set()
        time.sleep(0.5)
        return real_write_batch(rows)

    ing._write_batch = slow_write_batch

    t = threading.Thread(target=ing._idle_flush_tick, name="idle-1")
    t.start()
    assert started.wait(2.0), "first idle flush did not start"

    # Second tick while the first is mid-flight — must skip immediately.
    assert ing._flush_in_progress.is_set()
    ing._idle_flush_tick()

    t.join(2.0)
    assert not t.is_alive()
    # Only the first tick ran a write; the second was single-flighted away.
    assert call_count["n"] == 1
    assert not ing._flush_in_progress.is_set()  # event cleared after the flush


def test_explicit_flush_not_guarded(tmp_path):
    """The public ``flush()`` API is NOT single-flighted: even with the
    in-progress event set (as the idle path would), an explicit flush still
    writes its buffered rows."""
    ing = _make_ingester(tmp_path)

    # Simulate the idle guard being held by a (notional) concurrent idle flush.
    ing._flush_in_progress.set()

    for w in ("alpha", "beta", "gamma"):
        ing.ingest_one({"word": w, "definition": f"def {w}"})
    written_1 = ing.flush()
    assert written_1 == 3  # ran despite the guard being set

    for w in ("delta", "epsilon"):
        ing.ingest_one({"word": w, "definition": f"def {w}"})
    written_2 = ing.flush()
    assert written_2 == 2  # second explicit flush also runs

    check = sqlite3.connect(str(tmp_path / "kdb.db"))
    count = check.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
    check.close()
    assert count == 5


def test_finally_log_fires_on_happy_path(tmp_path, caplog):
    ing = _make_ingester(tmp_path)
    _push_rows(ing, 4)

    with caplog.at_level(logging.INFO, logger=vi.logger.name):
        written = ing.flush()

    assert written == 4
    summary = [r.getMessage() for r in caplog.records if r.getMessage().startswith("vocab flush:")]
    assert len(summary) == 1
    assert "flushed=4" in summary[0]
    assert "failed_sub_batches=0" in summary[0]


def test_finally_log_fires_on_fallback_path(tmp_path, monkeypatch, caplog):
    """When a sub-batch raises 'database is locked' and the per-row fallback
    takes over, the ``vocab flush:`` INFO line must STILL fire (with
    failed_sub_batches > 0)."""
    monkeypatch.setattr(vi, "_VOCAB_SUBBATCH_SIZE", 2)
    db_file = tmp_path / "kdb.db"
    ing = VocabularyIngester(db_path=str(db_file))
    ing.flush_seconds = 0
    _push_rows(ing, 4)  # -> 2 sub-batches

    class FlakyConn:
        """First executemany raises 'database is locked'; per-row execute works."""

        def __init__(self, real):
            self.real = real
            self.executemany_calls = 0

        def executemany(self, sql, seq):
            self.executemany_calls += 1
            if self.executemany_calls == 1:
                raise sqlite3.OperationalError("database is locked")
            return self.real.executemany(sql, seq)

        def execute(self, sql, *a, **k):
            return self.real.execute(sql, *a, **k)

        def commit(self):
            return self.real.commit()

        def rollback(self):
            return self.real.rollback()

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                self.real.commit()
            else:
                self.real.rollback()
            return False

    flaky = FlakyConn(sqlite3.connect(str(db_file)))
    monkeypatch.setattr(vi, "safe_open_kdb", lambda *a, **k: flaky)

    with caplog.at_level(logging.INFO, logger=vi.logger.name):
        written = ing.flush()

    assert written == 4  # all rows landed via fallback
    summary = [r.getMessage() for r in caplog.records if r.getMessage().startswith("vocab flush:")]
    assert len(summary) == 1
    assert "flushed=4" in summary[0]
    assert "failed_sub_batches=1" in summary[0]
