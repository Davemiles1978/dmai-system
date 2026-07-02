"""R3: disk cleanup must never delete live -wal/-shm sidecars.

Importing ``dmai_core_complete`` triggers full Flask/app boot and imports
components that aren't always materialised, so we extract the pure
``_sidecar_is_live`` helper straight from source via AST and exec it in an
isolated namespace. This exercises the real shipped function body without the
module-level side effects.
"""
import ast
import sqlite3
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "dmai_core_complete.py"


def _load_sidecar_is_live():
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_sidecar_is_live":
            ns = {}
            exec(compile(ast.Module([node], []), str(SOURCE), "exec"), ns)
            return ns["_sidecar_is_live"]
    raise AssertionError("_sidecar_is_live not found in dmai_core_complete.py")


sidecar_is_live = _load_sidecar_is_live()


def _make_wal_db(db_path: Path):
    """Create a real WAL-mode SQLite DB with data, leaving sidecars present."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute("INSERT INTO t (v) VALUES ('hello')")
    conn.commit()
    return conn


def test_live_sidecar_is_preserved(tmp_path):
    db = tmp_path / "live.db"
    conn = _make_wal_db(db)
    try:
        wal = tmp_path / "live.db-wal"
        assert wal.exists(), "WAL sidecar should exist for an open WAL-mode DB"
        assert sidecar_is_live(wal) is True
    finally:
        conn.close()


def test_orphan_sidecar_is_deletable(tmp_path):
    wal = tmp_path / "orphan.db-wal"
    wal.write_bytes(b"\x00\x01\x02")  # no matching main .db
    assert sidecar_is_live(wal) is False


def test_sidecar_with_invalid_main_is_deletable(tmp_path):
    (tmp_path / "junk.db").write_bytes(b"not a sqlite header at all")
    wal = tmp_path / "junk.db-wal"
    wal.write_bytes(b"\x00")
    assert sidecar_is_live(wal) is False


def test_missing_sidecar_is_not_live(tmp_path):
    assert sidecar_is_live(tmp_path / "nope.db-wal") is False


def test_non_sidecar_path_is_not_live(tmp_path):
    p = tmp_path / "plain.txt"
    p.write_text("x")
    assert sidecar_is_live(p) is False


def test_shm_sidecar_liveness(tmp_path):
    db = tmp_path / "live.db"
    conn = _make_wal_db(db)
    try:
        shm = tmp_path / "live.db-shm"
        if shm.exists():
            assert sidecar_is_live(shm) is True
        # Orphan -shm with no main file is deletable.
        orphan_shm = tmp_path / "gone.db-shm"
        orphan_shm.write_bytes(b"\x00")
        assert sidecar_is_live(orphan_shm) is False
    finally:
        conn.close()


def test_cleanup_skips_live_but_deletes_orphans(tmp_path):
    """Integration-style: replicate the cleanup name_kill sweep + R3 guard and
    assert live sidecars survive while orphans are removed."""
    live_db = tmp_path / "live.db"
    conn = _make_wal_db(live_db)
    try:
        live_wal = tmp_path / "live.db-wal"
        live_shm = tmp_path / "live.db-shm"
        assert live_wal.exists()
        shm_existed = live_shm.exists()

        orphan_wal = tmp_path / "old.db-wal"
        orphan_shm = tmp_path / "old.db-shm"
        orphan_wal.write_bytes(b"\x00")
        orphan_shm.write_bytes(b"\x00")

        name_kill = ("-wal", "-shm", "-journal")
        for fp in sorted(tmp_path.iterdir()):
            f = fp.name
            if any(f.endswith(s) for s in name_kill):
                if (f.endswith("-wal") or f.endswith("-shm")) and sidecar_is_live(fp):
                    continue  # R3 guard
                fp.unlink()

        assert live_wal.exists(), "live -wal must be preserved"
        if shm_existed:
            assert live_shm.exists(), "live -shm must be preserved"
        assert live_db.exists()
        assert not orphan_wal.exists(), "orphan -wal must be deleted"
        assert not orphan_shm.exists(), "orphan -shm must be deleted"
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
