"""Tests for VocabularyIngester sub-batch write behaviour (PR #156).

These guard the contract that ``_write_batch`` acquires and releases the
process write mutex **once per sub-batch** (not once for the whole flush) and
yields between sub-batches so other writers can interleave. They are fast and
do not exercise real lock contention — we test the contract, not chaos.
"""

import contextlib
import inspect
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.knowledge import vocabulary_ingester as vi
from components.knowledge.vocabulary_ingester import VocabularyIngester


def _make_ingester(tmp_path) -> VocabularyIngester:
    """Ingester pointed at a temp DB with the idle-flush timer disabled so the
    only flush is the one the test triggers explicitly."""
    ing = VocabularyIngester(db_path=str(tmp_path / "kdb.db"))
    ing.flush_seconds = 0  # disable the background idle-flush thread
    return ing


def _push_rows(ing: VocabularyIngester, n: int) -> None:
    for i in range(n):
        ing.ingest_one({"word": f"word{i}", "definition": f"definition number {i}"})


def test_flush_releases_mutex_between_sub_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(vi, "_VOCAB_SUBBATCH_SIZE", 2)
    ing = _make_ingester(tmp_path)
    _push_rows(ing, 6)  # 6 rows / size 2 -> 3 sub-batches

    cycles = {"n": 0}
    real_acquire = vi.acquire_write_lock

    def counting_acquire(path):
        cycles["n"] += 1
        return real_acquire(path)

    monkeypatch.setattr(vi, "acquire_write_lock", counting_acquire)

    written = ing.flush()

    assert written == 6
    # One acquire/release cycle per sub-batch, not a single cycle for the flush.
    assert cycles["n"] == 3


def test_flush_yields_between_sub_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(vi, "_VOCAB_SUBBATCH_SIZE", 2)
    ing = _make_ingester(tmp_path)
    _push_rows(ing, 6)  # -> 3 sub-batches, so 2 inter-sub-batch yields

    sleeps = []
    monkeypatch.setattr(vi.time, "sleep", lambda s: sleeps.append(s))

    ing.flush()

    # N sub-batches -> N-1 yields (never after the last one).
    assert len(sleeps) == 2
    assert all(s == pytest.approx(vi._VOCAB_SUBBATCH_YIELD_MS / 1000.0) for s in sleeps)


def test_fallback_one_by_one_on_locked_sub_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(vi, "_VOCAB_SUBBATCH_SIZE", 2)
    db_file = tmp_path / "kdb.db"
    ing = VocabularyIngester(db_path=str(db_file))  # real tables created here
    ing.flush_seconds = 0
    _push_rows(ing, 6)  # 3 sub-batches

    class FlakyConn:
        """Real sqlite connection whose FIRST executemany raises
        'database is locked'; per-row execute always works."""

        def __init__(self, real):
            self.real = real
            self.executemany_calls = 0
            self.execute_calls = 0

        def executemany(self, sql, seq):
            self.executemany_calls += 1
            if self.executemany_calls == 1:
                raise sqlite3.OperationalError("database is locked")
            return self.real.executemany(sql, seq)

        def execute(self, sql, *a, **k):
            self.execute_calls += 1
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

    written = ing.flush()

    # Sub-batch 1 (2 rows) failed executemany -> per-row fallback (2 execute
    # calls). Sub-batches 2 and 3 succeeded via executemany in their own
    # acquisitions. All 6 rows landed.
    assert flaky.executemany_calls == 3
    assert flaky.execute_calls == 2
    assert written == 6

    check = sqlite3.connect(str(db_file))
    count = check.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
    check.close()
    assert count == 6


def test_flush_buffer_is_cleared_after_success(tmp_path):
    db_file = tmp_path / "kdb.db"
    ing = VocabularyIngester(db_path=str(db_file))
    ing.flush_seconds = 0
    _push_rows(ing, 5)

    assert len(ing._batch) == 5  # buffered, not yet written

    written = ing.flush()

    assert written == 5
    assert ing._batch == []
    assert ing._batch_words == set()

    check = sqlite3.connect(str(db_file))
    count = check.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
    check.close()
    assert count == 5


def test_public_api_signatures_unchanged():
    init_params = list(inspect.signature(VocabularyIngester.__init__).parameters)
    assert init_params == ["self", "db_path", "batch_size"]

    assert list(inspect.signature(VocabularyIngester.ingest_one).parameters) == [
        "self", "word_data",
    ]
    assert list(inspect.signature(VocabularyIngester.ingest_many).parameters) == [
        "self", "items",
    ]
    assert list(inspect.signature(VocabularyIngester.flush).parameters) == ["self"]
