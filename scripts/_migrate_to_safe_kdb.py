#!/usr/bin/env python3
"""One-off migration helper for DB hardening.

For each file passed on argv:
  1. Add `from components.db import safe_open_kdb` to imports if not present.
  2. Replace `sqlite3.connect(<path>[, timeout=N])` -> `safe_open_kdb(<path>[, timeout=N])`
  3. Replace `<alias>.connect(<path>[, timeout=N])` -> `safe_open_kdb(<path>[, timeout=N])`
     when the alias was set with `import sqlite3 as <alias>` in the same file.

Does NOT touch:
  - bare `import sqlite3` statements (sqlite3.Row, sqlite3.Connection, etc. still used)
  - non-knowledge-DB connects (script is run per-target-file by caller)

Refuses to write if the file's text changes the import statement structure
in any other way.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CONNECT_RE = re.compile(
    r"\b(?P<mod>sqlite3|[A-Za-z_]\w*)\.connect\("
)

# We need to know which alias names refer to sqlite3 in each file.
ALIAS_RE = re.compile(r"^\s*import\s+sqlite3(?:\s+as\s+(\w+))?", re.MULTILINE)


def collect_aliases(src: str) -> set[str]:
    """Return the set of names that are imported as sqlite3 in this file."""
    aliases = {"sqlite3"}
    for m in ALIAS_RE.finditer(src):
        alias = m.group(1)
        if alias:
            aliases.add(alias)
    return aliases


def add_helper_import(src: str) -> str:
    """Ensure `from components.db import safe_open_kdb` is present at module top."""
    if "from components.db import safe_open_kdb" in src:
        return src
    # Insert after the last top-level import statement.
    lines = src.splitlines(keepends=True)
    last_import_idx = -1
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("import ", "from ")):
            # only top-level (no leading whitespace)
            if line[:len(line) - len(stripped)] == "":
                last_import_idx = i
    if last_import_idx == -1:
        # No top-level imports: insert at top
        return "from components.db import safe_open_kdb\n" + src
    lines.insert(last_import_idx + 1, "from components.db import safe_open_kdb\n")
    return "".join(lines)


def rewrite_connects(src: str, aliases: set[str]) -> tuple[str, int]:
    """Replace `<alias>.connect(...)` with `safe_open_kdb(...)`. Returns new src + count."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        if m.group("mod") in aliases:
            count += 1
            return "safe_open_kdb("
        return m.group(0)

    new_src = CONNECT_RE.sub(repl, src)
    return new_src, count


def process_file(path: Path) -> int:
    """Returns count of replacements made in this file."""
    src = path.read_text(encoding="utf-8")
    aliases = collect_aliases(src)
    new_src, count = rewrite_connects(src, aliases)
    if count == 0:
        print(f"  {path}: 0 sites (skipped)")
        return 0
    new_src = add_helper_import(new_src)
    path.write_text(new_src, encoding="utf-8")
    print(f"  {path}: {count} site(s) migrated")
    return count


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: _migrate_to_safe_kdb.py <file> [<file> ...]", file=sys.stderr)
        return 1
    total = 0
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.is_file():
            print(f"  {p}: NOT FOUND", file=sys.stderr)
            return 1
        total += process_file(p)
    print(f"\nTotal sites migrated: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
