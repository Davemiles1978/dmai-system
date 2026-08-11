"""Schema bootstrap — eagerly create all SQLite tables at boot.

PROBLEM SOLVED:
After a DB rebuild (by watchdog or admin endpoint), the fresh dmai_knowledge.db
contains only tables that have been written to since boot. Components create
their tables lazily on first WRITE via `CREATE TABLE IF NOT EXISTS`, but
GET-only routes do SELECT first — and crash with `no such table: X`.

FIX:
At boot, after all components are loaded but before Flask serves traffic,
this module scans every components/*.py file for `CREATE TABLE IF NOT EXISTS`
and CREATE INDEX statements, then executes them in a single connection
against dmai_knowledge.db. All statements are idempotent — safe to run
on every boot, even when tables already exist.

DESIGN PRINCIPLES:
- Read-only of source code: we don't import component modules (avoids side effects).
- Wraps every operation in try/except; never raises.
- Reports counts and errors for observability.
- Targets ONLY dmai_knowledge.db (the main, often-rebuilt DB).
  Other DBs (dmai.db, trading_mastery.db) are not touched.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from typing import Dict, List
from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

# Files that target dmai_knowledge.db (the often-rebuilt main DB).
# Skip files that target other DBs (e.g. wealth/autonomous_trader.py uses
# trading_mastery.db). When in doubt INCLUDE — CREATE IF NOT EXISTS is
# idempotent and a table appearing in the wrong DB at boot is harmless
# if it's never written to.
_COMPONENT_ROOTS = ["components"]

# Skip files that use PostgreSQL-specific syntax (SERIAL, TIMESTAMPTZ, etc.)
# These are valid only against a Postgres backend and will fail in SQLite.
_SKIP_FILES = (
    "pg_storage.py",
    "P0T0_Migrate_Database_to_Production.py",
    "P0T3_Connect_local_core_to_PostgreSQL.py",
    "schema_bootstrap.py",  # don't scan ourselves — fallback strings would re-match
    "self_optimizer.py",    # uses multi-statement string concat, not portable SQLite
    "capability_schema_migration.py",  # PG-incompatible syntax
    "sqlite_persistence.py",  # uses sqlite3 directly
    "insight_promoter.py",  # has malformed CREATE TABLE
    "capability_materialiser.py",  # SQLite-specific DDL
)

# Match CREATE TABLE/INDEX IF NOT EXISTS up to the terminating `;` at end-of-statement.
# Tables can contain nested `(` `)` for column types (e.g. DECIMAL(10,2)) and `CHECK(...)`.
# We use a balanced-paren walker instead of a regex.
_CREATE_START_RE = re.compile(
    r"CREATE\s+(?:TABLE|INDEX|UNIQUE\s+INDEX)\s+IF\s+NOT\s+EXISTS\s+\w+",
    re.IGNORECASE,
)

# Some components build CREATE TABLE statements via Python string concatenation
# (e.g. brain_loader.py, persona_registry.py). The scanner cannot extract those
# because there's no contiguous SQL literal. List them explicitly here so the
# tables exist after bootstrap. Each entry is idempotent SQL.
_EXPLICIT_FALLBACK_SCHEMAS = [
    # brain_entries (components/brain/brain_loader.py)
    """CREATE TABLE IF NOT EXISTS brain_entries (
        id TEXT PRIMARY KEY,
        domain TEXT NOT NULL,
        domain_label TEXT,
        topic TEXT NOT NULL,
        content TEXT NOT NULL,
        source_url TEXT NOT NULL,
        tier TEXT DEFAULT 'canonical',
        version TEXT,
        loaded_at TEXT DEFAULT (datetime('now'))
    );""",
    "CREATE INDEX IF NOT EXISTS idx_brain_domain ON brain_entries(domain);",
    """CREATE TABLE IF NOT EXISTS brain_load_log (
        id SERIAL PRIMARY KEY,
        domain TEXT,
        entries_added INTEGER,
        ts TEXT DEFAULT (datetime('now'))
    );""",
    # personas + persona_usage (components/personas/persona_registry.py)
    """CREATE TABLE IF NOT EXISTS personas (
        name TEXT PRIMARY KEY,
        label TEXT,
        scope TEXT,
        used_by_json TEXT,
        brain_domains_json TEXT,
        model_pref_json TEXT,
        system_prompt TEXT,
        decision_rules_json TEXT,
        version TEXT,
        updated_at TEXT DEFAULT (datetime('now'))
    );""",
    # R4: this fallback previously declared persona_name/operation/metadata_json,
    # which does not match the real schema created by
    # components/personas/persona_registry.py's _init_db (persona/component/task).
    # Both are CREATE TABLE IF NOT EXISTS, so whichever ran first won permanently
    # — harmless while bootstrap_all_schemas only ran once per long-lived boot
    # DB, but _ensure_kdb_schema (R4) now calls it on every fresh-DB self-heal,
    # so a mismatched fallback would silently lock in the wrong columns for any
    # new DB. Kept in sync with persona_registry.py's real DDL.
    """CREATE TABLE IF NOT EXISTS persona_usage (
        id SERIAL PRIMARY KEY,
        ts TEXT DEFAULT (datetime('now')),
        persona TEXT,
        component TEXT,
        task TEXT
    );""",
    # suggestions (built inside dmai_core_complete.py boot fn, not in components/)
    """CREATE TABLE IF NOT EXISTS suggestions (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL DEFAULT 'user',
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        complexity TEXT DEFAULT NULL,
        plan TEXT DEFAULT NULL,
        result TEXT DEFAULT NULL,
        pr_url TEXT DEFAULT NULL,
        branch TEXT DEFAULT NULL,
        files_changed TEXT DEFAULT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT DEFAULT NULL
    );""",
]

# Tables whose schema drifts across the codebase — the legacy CREATE TABLE
# (in dmai_core_complete.py) runs at import time and 'wins' the IF NOT EXISTS
# race, leaving the modern code with INSERTs that reference columns that
# don't exist. After table creation, we run ALTER TABLE ADD COLUMN for
# any missing columns listed here. ADD COLUMN in SQLite can't use
# CURRENT_TIMESTAMP as default, so we use NULL/static defaults only and
# let application code populate real values.
_REQUIRED_COLUMNS = {
    # canonical rich schema from components/sqlite_persistence.py + INSERT in dmai_api_routes.py
    "insights": [
        # (column_name, sqlite_type_with_default)
        ("entity_type", "TEXT"),
        ("entities", "TEXT"),
        ("relationship", "TEXT"),
        ("source_topic", "TEXT"),
        ("target_topic", "TEXT"),
        ("source_url", "TEXT"),
        ("source_title", "TEXT"),
        ("source_type", "TEXT"),
        ("occurrence_count", "INTEGER DEFAULT 1"),
        ("last_used", "TIMESTAMP"),
        # legacy columns the early CREATE already has — listed here so IF MISSING
        # we still ADD them (e.g. on a Postgres-imported snapshot)
        ("content", "TEXT"),
        ("description", "TEXT"),
        ("title", "TEXT"),
    ],
    # capabilities also drifts (sqlite_persistence vs core)
    "capabilities": [
        ("capability_type", "TEXT"),
        ("runtime_mode", "TEXT"),
        ("description", "TEXT"),
        ("category", "TEXT"),
        ("proficiency", "REAL DEFAULT 0.0"),
    ],
}


def _ensure_columns(conn: sqlite3.Connection, result: Dict) -> None:
    """For each table in _REQUIRED_COLUMNS, add any missing columns.

    for ones that don't exist yet. ALTER TABLE ADD COLUMN is fast even on
    large tables in SQLite because it doesn't rewrite the table.
    """
    cur = conn.cursor()
    for table, cols in _REQUIRED_COLUMNS.items():
        try:
            # Get existing columns for this table
            try:
                cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'")
                existing = {row[0] for row in cur.fetchall()}
            except Exception:
                existing = set()
            if not existing:
                # table doesn't exist — skip (CREATE pass should have made it)
                continue
            for col_name, col_type in cols:
                if col_name in existing:
                    continue
                try:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                    result.setdefault("columns_added", 0)
                    result["columns_added"] += 1
                except Exception as oe:
                    msg = str(oe)
                    if "duplicate column name" in msg.lower():
                        continue  # already exists
                    result["errors"] += 1
                    if len(result["error_samples"]) < 8:
                        result["error_samples"].append(
                            f"alter {table}.{col_name}: {msg[:100]}"
                        )
        except Exception as e:
            result["errors"] += 1
            if len(result["error_samples"]) < 8:
                result["error_samples"].append(
                    f"ensure_cols({table}): {type(e).__name__}: {str(e)[:100]}"
                )


def _extract_statement(text: str, start_idx: int) -> str:
    """Given the position of a CREATE statement start, walk forward to find the
    statement's terminating `;`, correctly handling nested parens.

    Returns the full SQL statement including the trailing `;`, or empty string
    on failure.
    """
    n = len(text)
    depth = 0
    i = start_idx
    # First find the opening paren (CREATE TABLE foo (...) or CREATE INDEX foo ON tbl(...)).
    # Index statements have an `ON tbl` clause before `(`.
    while i < n and text[i] != "(":
        if text[i] == ";":
            # Plain `CREATE INDEX IF NOT EXISTS name ON table(col);` ends here.
            return text[start_idx:i + 1]
        i += 1
    if i >= n:
        return ""
    # Now walk parens until balanced, then find the next ;
    while i < n:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                # Find the next ; (skipping whitespace and table-options like WITHOUT ROWID)
                j = i + 1
                while j < n and text[j] != ";":
                    # Bail out if we hit another statement marker or newline-then-nonspace
                    if text[j] == "\n" and j + 1 < n and not text[j + 1].isspace():
                        # End without semicolon — still return what we have, append ;
                        return text[start_idx:i + 1].strip() + ";"
                    j += 1
                if j < n:
                    return text[start_idx:j + 1]
                return text[start_idx:i + 1].strip() + ";"
        i += 1
    return ""


def _scan_create_statements() -> List[Dict[str, str]]:
    """Walk components/ and extract every CREATE TABLE/INDEX IF NOT EXISTS stmt.

    Returns a list of {"file": str, "sql": str}.
    """
    statements: List[Dict[str, str]] = []
    for root_dir in _COMPONENT_ROOTS:
        if not os.path.isdir(root_dir):
            continue
        for dirpath, _, files in os.walk(root_dir):
            # Skip backups directory — those are old code, may have stale schemas
            if "backups" in dirpath.split(os.sep):
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                if f in _SKIP_FILES:
                    continue
                path = os.path.join(dirpath, f)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        txt = fh.read()
                except Exception:
                    continue
                for match in _CREATE_START_RE.finditer(txt):
                    sql = _extract_statement(txt, match.start())
                    sql = sql.strip()
                    if not sql or not sql.endswith(";"):
                        continue
                    statements.append({"file": path, "sql": sql})
    return statements


def bootstrap_all_schemas(db_path: str) -> Dict[str, int]:
    """Run every CREATE TABLE/INDEX IF NOT EXISTS against db_path.

    Returns: {"statements_total", "executed", "skipped", "errors", "tables_after"}.
    Never raises.
    """
    result = {
        "statements_total": 0,
        "executed": 0,
        "skipped": 0,
        "errors": 0,
        "tables_after": 0,
        "error_samples": [],
    }
    try:
        statements = _scan_create_statements()
    except Exception as e:
        logger.warning("schema_bootstrap: scan failed: %s", e)
        result["errors"] += 1
        result["error_samples"].append(f"scan: {e}")
        return result

    # Add explicit fallback schemas for components that build SQL via Python concat
    for sql in _EXPLICIT_FALLBACK_SCHEMAS:
        statements.append({"file": "<explicit_fallback>", "sql": sql})

    result["statements_total"] = len(statements)
    if not statements:
        return result

    # Ensure parent dir exists
    try:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    except Exception:
        pass

    conn = None
    try:
        conn = safe_open_kdb(db_path, timeout=30.0)
        cur = conn.cursor()
        for stmt in statements:
            try:
                cur.execute(stmt["sql"])
                result["executed"] += 1
            except Exception as oe:
                # Common: "table already exists" with different schema —
                # our IF NOT EXISTS makes that impossible, so this is real.
                # Also: column-list parse failures from non-standard syntax.
                msg = str(oe)
                if "already exists" in msg.lower():
                    result["skipped"] += 1
                else:
                    result["errors"] += 1
                    if len(result["error_samples"]) < 5:
                        result["error_samples"].append(
                            f"{os.path.basename(stmt['file'])}: {msg[:120]}"
                        )
            except Exception as e:
                result["errors"] += 1
                if len(result["error_samples"]) < 5:
                    result["error_samples"].append(
                        f"{os.path.basename(stmt['file'])}: {type(e).__name__}: {str(e)[:120]}"
                    )
        conn.commit()
        # Ensure schema-drift tables have all required columns (ALTER TABLE ADD COLUMN)
        _ensure_columns(conn, result)
        conn.commit()
        # Count tables after
        try:
            n = cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"
            ).fetchone()[0]
            result["tables_after"] = int(n)
        except Exception:
            pass
    except Exception as e:
        result["errors"] += 1
        result["error_samples"].append(f"connect/commit: {e}")
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    return result
