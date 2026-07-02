"""Guard for R2 (forensic report §9) — checkpoint WAL before integrity_check.

Root cause of the chronic false-positive quarantines: the boot self-heal path in
``dmai_core_complete.py`` opened ``dmai_knowledge.db`` with a bare (non-WAL-aware)
connection and ran ``PRAGMA integrity_check`` immediately. After a SIGKILL
(gunicorn timeout=300, common Render restart signals) the ``-wal`` sidecar often
holds committed-but-uncheckpointed transactions. A bare connection doesn't
reconcile the WAL, so integrity_check could report a non-"ok" verdict and trigger
a quarantine of a perfectly recoverable DB (which under pre-R1 code then destroyed
the data). R2 runs ``PRAGMA wal_checkpoint(TRUNCATE)`` first to fold committed
frames back into the main file; only genuine corruption survives to the check.

The checkpoint logic lives in the module-level helper
``_checkpoint_before_integrity``. Importing ``dmai_core_complete`` wholesale pulls
in the entire app (Flask, background threads, optional deps), so — like the R1
guard — we extract just that function's source via ``ast`` and exercise the exact
shipped code in an isolated namespace.
"""

import ast
import logging
import os
import sqlite3
from pathlib import Path

_CORE = Path(__file__).parent.parent / "dmai_core_complete.py"
_FUNC_NAME = "_checkpoint_before_integrity"


def _load_checkpoint_fn():
    """Extract and compile ``_checkpoint_before_integrity`` from the real source
    without importing the (very heavy) module."""
    src = _CORE.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == _FUNC_NAME:
            segment = ast.get_source_segment(src, node)
            ns = {"os": os, "logger": logging.getLogger("test.boot_checkpoint")}
            exec(compile(segment, str(_CORE), "exec"), ns)
            return ns[_FUNC_NAME]
    raise AssertionError(f"{_FUNC_NAME} not found in {_CORE}")


def _write_uncheckpointed_wal(db_path):
    """Create a WAL-mode DB with a sentinel row and leave frames in the ``-wal``.

    We disable autocheckpoint and abandon the connection object without closing
    it cleanly, so the committed row lives in the ``-wal`` sidecar rather than the
    main file — the exact state a SIGKILL leaves behind.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")  # don't fold frames back on our own
    conn.execute("CREATE TABLE sentinel (id INTEGER PRIMARY KEY, note TEXT)")
    conn.execute("INSERT INTO sentinel (note) VALUES ('committed-in-wal')")
    conn.commit()
    # Leak the connection intentionally: no conn.close(), so the checkpoint that a
    # clean close would perform never runs. The -wal keeps the committed frames.
    assert os.path.exists(db_path + "-wal"), "test setup expected a -wal sidecar"
    return conn  # returned only to keep it alive; caller may drop it


def test_committed_wal_survives_boot_check(tmp_path):
    """After checkpointing, integrity_check is 'ok' and the WAL row is present."""
    checkpoint = _load_checkpoint_fn()
    db = tmp_path / "kb.db"

    _leak = _write_uncheckpointed_wal(str(db))  # noqa: F841 (kept alive on purpose)

    checkpoint(str(db))

    # A brand-new connection sees a healthy DB and the committed sentinel row.
    conn = sqlite3.connect(str(db))
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        rows = conn.execute("SELECT note FROM sentinel").fetchall()
    finally:
        conn.close()

    assert integrity == "ok"
    assert ("committed-in-wal",) in rows


def test_missing_db_is_noop(tmp_path):
    """Pointing the helper at a non-existent path must not raise or create a file."""
    checkpoint = _load_checkpoint_fn()
    missing = tmp_path / "does_not_exist.db"

    checkpoint(str(missing))  # must not raise

    assert not missing.exists()
    assert not list(tmp_path.iterdir())  # no empty DB (or any file) was created


def test_no_wal_sidecar_is_noop(tmp_path):
    """A checkpointed DB with no ``-wal`` is left untouched and does not raise."""
    checkpoint = _load_checkpoint_fn()
    db = tmp_path / "clean.db"

    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.commit()
    conn.close()  # clean close checkpoints and removes the -wal
    assert not os.path.exists(str(db) + "-wal")

    checkpoint(str(db))  # must not raise

    conn = sqlite3.connect(str(db))
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()
    assert integrity == "ok"


def test_locked_db_does_not_raise(tmp_path):
    """A busy/locked DB is a signal, not proof of corruption — must not raise."""
    checkpoint = _load_checkpoint_fn()
    db = tmp_path / "busy.db"

    _leak = _write_uncheckpointed_wal(str(db))  # noqa: F841

    # Hold an exclusive write lock from another connection so TRUNCATE can't fully
    # complete; the helper should log a warning / partial result but never raise.
    blocker = sqlite3.connect(str(db))
    blocker.execute("BEGIN IMMEDIATE")
    blocker.execute("INSERT INTO sentinel (note) VALUES ('holder')")
    try:
        checkpoint(str(db))  # must not raise even under contention
    finally:
        blocker.rollback()
        blocker.close()
