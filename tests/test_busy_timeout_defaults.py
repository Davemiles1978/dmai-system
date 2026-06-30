"""Guards for PR #158a — reader-side busy_timeout PRAGMA overrides removed.

Five modules used to issue ``PRAGMA busy_timeout=<3000|5000>`` on connections
obtained from ``safe_open_kdb``. Because ``safe_open_kdb`` caches one connection
per thread, those PRAGMAs mutated the shared handle for the rest of the thread's
lifetime, so a later write inherited the downgraded timeout instead of the 30 s
default from ``_PER_CONNECTION_PRAGMAS`` (``components/db.py``).

These tests assert each formerly-overriding module now yields the canonical
30 000 ms default, plus one guard test confirming the vocab writer (PR #157)
still issues its legitimate per-write ``PRAGMA busy_timeout=2000`` inside
``_write_batch`` — that override is correct because it is re-applied per write,
not held for the connection's lifetime.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_DEFAULT_BUSY_TIMEOUT_MS = 30000
_VOCAB_BUSY_TIMEOUT_MS = 2000


def _busy_timeout(conn) -> int:
    return conn.execute("PRAGMA busy_timeout").fetchone()[0]


def _conn_via(cls, db_path):
    """Build an instance without running __init__ (avoids heavy ctor deps) and
    call its real ``_conn`` so we exercise the module's actual open path."""
    obj = cls.__new__(cls)
    obj.db_path = str(db_path)
    return obj._conn()


def test_memory_retrieval_uses_default_busy_timeout(tmp_path, monkeypatch):
    import components.memory_retrieval as mr

    db = tmp_path / "dmai_knowledge.db"
    sqlite3.connect(str(db)).close()  # mode=ro requires an existing file

    monkeypatch.setattr(mr, "_KNOWLEDGE_DB", db)
    monkeypatch.setattr(mr, "_KNOWLEDGE_DB_BROKEN_UNTIL", 0)

    captured = {}
    real = mr.safe_open_kdb

    def capture(*args, **kwargs):
        conn = real(*args, **kwargs)
        captured["conn"] = conn
        return conn

    monkeypatch.setattr(mr, "safe_open_kdb", capture)

    mr._search_knowledge_db("some query", 1)

    assert "conn" in captured, "read path never opened a connection"
    assert _busy_timeout(captured["conn"]) == _DEFAULT_BUSY_TIMEOUT_MS


def test_betting_advisor_uses_default_busy_timeout(tmp_path):
    from components.monetisation.betting_advisor import BettingAdvisor

    conn = _conn_via(BettingAdvisor, tmp_path / "betting.db")
    assert _busy_timeout(conn) == _DEFAULT_BUSY_TIMEOUT_MS


def test_bill_payer_uses_default_busy_timeout(tmp_path):
    from components.monetisation.bill_payer import BillPayer

    conn = _conn_via(BillPayer, tmp_path / "bills.db")
    assert _busy_timeout(conn) == _DEFAULT_BUSY_TIMEOUT_MS


def test_wealth_allocator_uses_default_busy_timeout(tmp_path):
    from components.monetisation.wealth_allocator import WealthAllocator

    conn = _conn_via(WealthAllocator, tmp_path / "wealth.db")
    assert _busy_timeout(conn) == _DEFAULT_BUSY_TIMEOUT_MS


def test_autonomous_trader_uses_default_busy_timeout(tmp_path):
    from components.wealth.autonomous_trader import AutonomousTrader

    conn = _conn_via(AutonomousTrader, tmp_path / "trader.db")
    assert _busy_timeout(conn) == _DEFAULT_BUSY_TIMEOUT_MS


def test_vocab_writer_still_uses_short_busy_timeout(tmp_path):
    """Guard: vocab's per-write override (PR #157) must survive. ``_write_batch``
    issues ``PRAGMA busy_timeout=2000`` on its cached connection before the
    executemany, so after a flush that connection reports 2000 ms."""
    from components.knowledge.vocabulary_ingester import VocabularyIngester
    from components.db import safe_open_kdb

    db = tmp_path / "kdb.db"
    ing = VocabularyIngester(db_path=str(db))
    ing.flush_seconds = 0  # disable the background idle-flush thread
    ing.ingest_one({"word": "alpha", "definition": "the first letter"})

    written = ing.flush()
    assert written == 1

    # _write_batch set busy_timeout=2000 on the per-thread cached connection.
    conn = safe_open_kdb(str(db))
    assert _busy_timeout(conn) == _VOCAB_BUSY_TIMEOUT_MS
