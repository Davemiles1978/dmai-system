"""Guard for R1 (forensic report §9) — boot quarantine must PRESERVE WAL/SHM.

Root cause of the chronic zero-recovery data loss: the boot self-heal path in
``dmai_core_complete.py`` renamed a malformed ``dmai_knowledge.db`` to
``.malformed_<ts>`` but then **deleted** its ``-wal``/``-shm`` sidecars. In WAL
mode the ``-wal`` holds committed-but-uncheckpointed rows, so deleting it
destroyed the real data — ``.recover`` then found zero tables on 36 of 37
quarantined files. R1 swaps the delete for a rename so the trio is preserved
under a single shared timestamp, letting future tooling pair a WAL with its main
file.

The quarantine logic lives in the module-level helper
``_quarantine_malformed_db``. Importing ``dmai_core_complete`` wholesale pulls in
the entire app (Flask, background threads, optional deps), so instead we extract
just that function's source via ``ast`` and exercise the exact shipped code in an
isolated namespace.
"""

import ast
import os
import re
from pathlib import Path

_CORE = Path(__file__).parent.parent / "dmai_core_complete.py"
_FUNC_NAME = "_quarantine_malformed_db"


def _load_quarantine_fn():
    """Extract and compile ``_quarantine_malformed_db`` from the real source
    without importing the (very heavy) module."""
    src = _CORE.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == _FUNC_NAME:
            segment = ast.get_source_segment(src, node)
            ns = {"os": os}
            exec(compile(segment, str(_CORE), "exec"), ns)
            return ns[_FUNC_NAME]
    raise AssertionError(f"{_FUNC_NAME} not found in {_CORE}")


def test_quarantine_renames_wal_shm_instead_of_deleting(tmp_path):
    quarantine = _load_quarantine_fn()

    db = tmp_path / "foo.db"
    wal = tmp_path / "foo.db-wal"
    shm = tmp_path / "foo.db-shm"

    # Distinct sentinel bytes per file so we can prove no truncation/mix-up.
    db.write_bytes(b"MAIN-DB-SENTINEL-\x00\x01\x02")
    wal.write_bytes(b"WAL-SENTINEL-committed-rows-\xde\xad\xbe\xef")
    shm.write_bytes(b"SHM-SENTINEL-\xca\xfe")

    result = quarantine(str(db))

    # Originals must be gone (renamed, not copied).
    assert not db.exists()
    assert not wal.exists()
    assert not shm.exists()

    # The returned path is the quarantined main file and must exist.
    assert result == str(db) + ".malformed_" + result.rsplit("_", 1)[1]
    assert os.path.exists(result)

    ts = result.rsplit("_", 1)[1]
    assert ts.isdigit()

    new_main = tmp_path / f"foo.db.malformed_{ts}"
    new_wal = tmp_path / f"foo.db.wal.bak_{ts}"
    new_shm = tmp_path / f"foo.db.shm.bak_{ts}"

    # All three exist under the shared-timestamp names.
    assert new_main.exists()
    assert new_wal.exists()
    assert new_shm.exists()

    # Same ts across the whole trio — the property recovery tooling relies on.
    def _ts(name):
        return re.search(r"_(\d+)$", name).group(1)

    assert _ts(new_main.name) == ts
    assert _ts(new_wal.name) == ts
    assert _ts(new_shm.name) == ts

    # Bytes preserved exactly — rename must not truncate or corrupt.
    assert new_main.read_bytes() == b"MAIN-DB-SENTINEL-\x00\x01\x02"
    assert new_wal.read_bytes() == b"WAL-SENTINEL-committed-rows-\xde\xad\xbe\xef"
    assert new_shm.read_bytes() == b"SHM-SENTINEL-\xca\xfe"


def test_quarantine_handles_missing_sidecars(tmp_path):
    """A DB with no WAL/SHM (checkpointed / never opened) must still quarantine
    cleanly without error."""
    quarantine = _load_quarantine_fn()

    db = tmp_path / "bar.db"
    db.write_bytes(b"lonely-main")

    result = quarantine(str(db))

    assert not db.exists()
    assert os.path.exists(result)
    assert Path(result).read_bytes() == b"lonely-main"
    # No stray sidecar files were created.
    assert not list(tmp_path.glob("*.wal.bak_*"))
    assert not list(tmp_path.glob("*.shm.bak_*"))


def test_explicit_shared_timestamp_across_trio(tmp_path):
    """When a caller passes an explicit ts, all three renames use it verbatim."""
    quarantine = _load_quarantine_fn()

    db = tmp_path / "baz.db"
    (tmp_path / "baz.db-wal").write_bytes(b"w")
    (tmp_path / "baz.db-shm").write_bytes(b"s")
    db.write_bytes(b"m")

    result = quarantine(str(db), ts=1234567890)

    assert result == str(db) + ".malformed_1234567890"
    assert (tmp_path / "baz.db.malformed_1234567890").exists()
    assert (tmp_path / "baz.db.wal.bak_1234567890").exists()
    assert (tmp_path / "baz.db.shm.bak_1234567890").exists()
