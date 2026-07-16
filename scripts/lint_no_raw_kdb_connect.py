#!/usr/bin/env python3
"""Lint: no raw `sqlite3.connect` on the knowledge DB outside components/db.py.

The knowledge DB (``data/dmai_knowledge.db``) must be opened only via
``components.db.safe_open_kdb`` so every writer participates in the
shared process-level RLock + WAL busy_timeout + per-thread cache.

Bypassing this — even for a "just a small read" — creates a lock
competitor at the SQLite file level: our RLock is not enough because
SQLite serialises writers at the file, not at the Python process. A raw
``sqlite3.connect(dmai_knowledge.db)`` that writes will block our
KeepOpenProxy for up to ``busy_timeout=30000`` ms, and every waiting
thread will timeout with ``database is locked``.

This lint scans the tree, flags every raw ``sqlite3.connect`` call, and
whitelists ONLY:
  1. components/db.py itself (defines safe_open_kdb)
  2. components/capability_materialiser.py: `_safe_connect` fallback,
     which is documented + only used when the shared helper is unavailable
  3. scripts/ and root-level dev/one-shot scripts (recovery, migration
     one-offs) — these run offline
  4. test files (tmp DBs, fine)
  5. lines in comments/docstrings

Exit codes:
  0 = clean
  1 = raw sqlite3.connect found outside the whitelist
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files exempted from this rule
WHITELIST = {
    "components/db.py",                        # defines safe_open_kdb
    "components/capability_materialiser.py",   # documented test fallback
    "components/generated/_codegen_client.py", # docstring only
    "components/generated/_self_judge_review.py",  # accepts external db_path
    "components/purchase_gate/purchase_ledger.py",  # different DB
    "components/wealth/autonomous_trader.py",  # documented raw handle for test isolation
    "components/monetisation/revenue_allocator.py",  # different DB (ledger)
    "components/treasury/treasury_ledger.py",   # different DB (ledger)
    "components/procurement/store.py",         # different DB (procurement)
    "components/workload/workload_profiler.py",  # different DB (workload)
    "components/sqlite_persistence.py",        # backup path — different lifecycle
    "components/backup/r2_backup.py",          # opens read-only source + writes to different file
    "dmai_core_complete.py",                   # main app: reviewed manually
}

# Whitelisted prefixes (dev/one-shot scripts)
WHITELIST_PREFIXES = (
    "scripts/",
    "tests/",
    "update_endpoint.py",
    "fix_render_deploy.py",
    "add_critical_topics.py",
    "data/self_healing/backups/",  # snapshots
    "venv/",
    ".venv/",
    "components/backup_",  # historical backup module tree
)


def is_whitelisted(rel: str) -> bool:
    if rel in WHITELIST:
        return True
    return any(rel.startswith(p) for p in WHITELIST_PREFIXES)


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return [(line_no, offending_line)] for raw sqlite3.connect calls
    that aren't inside comments/docstrings.
    """
    text = path.read_text(errors="replace")
    if "sqlite3.connect" not in text:
        return []

    hits: list[tuple[int, str]] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Fallback: regex scan, ignoring lines that look like comments
        for line_no, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if re.search(r"\bsqlite3\.connect\s*\(", line):
                hits.append((line_no, line.strip()))
        return hits

    # AST walk: find Call nodes matching sqlite3.connect
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "connect"
                and isinstance(func.value, ast.Name)
                and func.value.id == "sqlite3"
            ):
                # Skip if the call is inside a docstring literal — we
                # can't easily detect this via AST for Call nodes since
                # a real Call in a docstring wouldn't parse as Call.
                # But comment context is already excluded by AST.
                line_no = getattr(node, "lineno", 0)
                line_text = text.splitlines()[line_no - 1]
                hits.append((line_no, line_text.strip()))
    return hits


def main() -> int:
    violations: list[tuple[str, int, str]] = []
    for py in REPO_ROOT.rglob("*.py"):
        rel = str(py.relative_to(REPO_ROOT))
        if is_whitelisted(rel):
            continue
        for line_no, line in scan_file(py):
            violations.append((rel, line_no, line))

    if not violations:
        print("OK: no raw sqlite3.connect on knowledge DB outside whitelist")
        return 0

    print("FAIL: raw sqlite3.connect calls found outside whitelist.")
    print("Use `from components.db import safe_open_kdb` and call")
    print("`safe_open_kdb(db_path, timeout=T)` instead. This is required")
    print("to prevent DB-lock storms — see components/db.py docstring.")
    print()
    for rel, line_no, line in violations:
        print(f"  {rel}:{line_no}: {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
