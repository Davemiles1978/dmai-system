"""R4: root-cause fixes for the chronic "malformed -> db not found" chain.

Three bugs fixed:
  1. Journal-mode race — several code paths opened dmai_knowledge.db with a
     bare sqlite3.connect() and never set PRAGMA journal_mode=WAL before doing
     real work, racing a sibling WAL connection and corrupting page 1.
  2. api_admin_db_rebuild quarantined the live DB and returned success, but
     nothing recreated it until the next process restart — db-salvage then
     hit "db not found" and every SELECT hit "no such table".
  3. The boot auto-heal block quarantined on ANY non-"ok" integrity_check
     verdict, including open_failed:... signals that are not proof of
     corruption (locked file, permission error, transient busy connection).

``_is_genuine_corruption`` is a small pure function, so we extract it via AST
(matching the PR #174 test pattern) to avoid importing the whole Flask app.
``_ensure_kdb_schema`` depends on module-level constants (``_CORE_SCHEMA_SQL``,
``_CORE_SCHEMA_ALTERS``) and ``components.schema_bootstrap``, so isolating it
via AST would mean re-implementing most of the module; the full import is fast
(~1-2s, no network calls at import time in this app) and stays faithful to the
exact shipped code, so we import ``dmai_core_complete`` directly for those
tests and for the flask test-client integration test.
"""

import ast
import os
import sqlite3
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "dmai_core_complete.py"


def _load_is_genuine_corruption():
    """Extract and compile ``_is_genuine_corruption`` (+ its signature tuple)
    from the real source without importing the whole module."""
    src = SOURCE.read_text()
    tree = ast.parse(src)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_GENUINE_CORRUPTION_SIGNATURES"
            for t in node.targets
        ):
            nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_is_genuine_corruption":
            nodes.append(node)
    assert len(nodes) == 2, "expected both the signatures tuple and the function"
    ns = {}
    exec(compile(ast.Module(nodes, []), str(SOURCE), "exec"), ns)
    return ns["_is_genuine_corruption"]


is_genuine_corruption = _load_is_genuine_corruption()


# ── Test env setup for the full-module-import tests ─────────────────────────
# Set required env vars BEFORE importing dmai_core_complete so module-level
# boot code (MASTER_PASSWORD checks, DATA_PATH) behaves predictably. Each
# test still points _ensure_kdb_schema at its own tmp_path DB explicitly, so
# this shared boot DATA_PATH is only a safe scratch default.
os.environ.setdefault("RENDER", "true")
# NOTE: "testpw" matches the convention already used by
# test_autonomous_trader_cadence.py / test_trader_at_mode.py (both also do
# os.environ.setdefault("MASTER_PASSWORD", "testpw")). Whichever test module
# pytest collects first "wins" the setdefault; using a different value here
# would silently break the other files' hardcoded "testpw" auth headers when
# this file is collected first in the same session.
os.environ.setdefault("MASTER_PASSWORD", "testpw")
os.environ.setdefault("DATA_PATH", "/tmp/dmai_r4_test_boot_data/")

import dmai_core_complete as dmai  # noqa: E402  (import after env setup, by design)


# ── 1. _ensure_kdb_schema creates from missing ───────────────────────────────

def test_ensure_kdb_schema_creates_from_missing(tmp_path):
    db_path = tmp_path / "nested" / "dmai_knowledge.db"
    assert not db_path.exists()

    result = dmai._ensure_kdb_schema(str(db_path))

    assert result["core_ok"] is True
    assert result["error"] is None
    assert db_path.exists()

    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for expected in ("capabilities", "insights", "system_state", "at_state"):
            assert expected in tables, f"{expected} missing after _ensure_kdb_schema"

        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()


# ── 2. Idempotent ────────────────────────────────────────────────────────────

def test_ensure_kdb_schema_idempotent(tmp_path):
    db_path = tmp_path / "dmai_knowledge.db"

    first = dmai._ensure_kdb_schema(str(db_path))
    assert first["core_ok"] is True

    conn = sqlite3.connect(str(db_path))
    try:
        tables_before = sorted(
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        )
    finally:
        conn.close()

    second = dmai._ensure_kdb_schema(str(db_path))  # must not error
    assert second["core_ok"] is True

    conn = sqlite3.connect(str(db_path))
    try:
        tables_after = sorted(
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        )
        at_state_rows = conn.execute("SELECT COUNT(*) FROM at_state").fetchone()[0]
    finally:
        conn.close()

    assert tables_before == tables_after
    assert at_state_rows == 1  # INSERT OR IGNORE must not duplicate the singleton


# ── 3. WAL mode locked first ─────────────────────────────────────────────────

def test_ensure_kdb_schema_wal_mode_locked_first(tmp_path):
    db_path = tmp_path / "dmai_knowledge.db"

    result = dmai._ensure_kdb_schema(str(db_path))
    assert result["core_ok"] is True

    conn = sqlite3.connect(str(db_path))
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"


# ── 4/5. _is_genuine_corruption gate ─────────────────────────────────────────

@pytest.mark.parametrize(
    "verdict",
    [
        "database disk image is malformed",
        "file is not a database",
        "malformed",
        "database is corrupt",
        "SOME PREFIX: malformed page 12",
    ],
)
def test_is_genuine_corruption_true_positives(verdict):
    assert is_genuine_corruption(verdict) is True


@pytest.mark.parametrize(
    "verdict",
    [
        "ok",
        "",
        None,
        "open_failed: [Errno 13] Permission denied",
        "database is locked",
    ],
)
def test_is_genuine_corruption_false_positives(verdict):
    assert is_genuine_corruption(verdict) is False


# ── 6. Flask integration: db-rebuild leaves a working DB behind ─────────────

def test_db_rebuild_leaves_working_db(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    live = data_dir / "dmai_knowledge.db"
    live.write_bytes(b"garbage")  # a genuinely malformed SQLite file

    monkeypatch.setenv("DATA_PATH", str(data_dir) + "/")
    monkeypatch.setenv("MASTER_PASSWORD", "testpw")

    client = dmai.app.test_client()
    resp = client.post(
        "/api/admin/db-rebuild",
        headers={"X-Master-Password": "testpw"},
        json={"db": "dmai_knowledge.db"},
    )
    body = resp.get_json()

    assert resp.status_code == 200, body
    assert body.get("rebuilt") is True
    assert body.get("schema_restored") is True

    assert os.path.exists(str(live)), "rebuild must leave a live DB file behind"

    conn = sqlite3.connect(str(live))
    try:
        count = conn.execute("SELECT COUNT(*) FROM capabilities").fetchone()
    finally:
        conn.close()
    assert count == (0,)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
