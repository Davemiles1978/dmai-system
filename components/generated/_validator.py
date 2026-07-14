"""AST-level validator for LLM-generated capability modules.

The generator writes candidate code into ``components/generated/staging/``.
Before *anything* imports or runs it, ``validate_source`` scans the AST
for banned imports, banned calls, and mandatory shape (``def run(...)``
plus a module docstring). Rejections are recorded with a specific
reason so the materialiser can re-prompt the LLM with a targeted
retry hint.

This module has no side effects. It parses, walks, returns a report.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


# ── Policy ────────────────────────────────────────────────────────────────

# Modules the sandbox is allowed to reach. Anything else is a hard reject.
# Kept deliberately tight: DMAI's generated capabilities operate on pure
# data + SQLite. Networking, threading, subprocess, filesystem writes
# outside the DB are all denied.
ALLOWED_IMPORTS: Set[str] = {
    # always-allowed
    "__future__",
    # stdlib data / algorithmics
    "math", "statistics", "json", "re", "hashlib", "datetime", "collections",
    "itertools", "functools", "operator", "dataclasses", "enum", "typing",
    "uuid", "random", "decimal", "string", "textwrap", "difflib", "bisect",
    "heapq", "copy",
    # DB (read-only paths only, enforced by sandbox — not this validator)
    "sqlite3",
    # DMAI internals that generated code is allowed to consume
    "components.knowledge",
    "components.self_judge",
}

# Attribute chains that indicate escape attempts even without an import.
BANNED_ATTR_CHAINS: Tuple[Tuple[str, ...], ...] = (
    ("os", "system"),
    ("os", "popen"),
    ("os", "remove"),
    ("os", "unlink"),
    ("os", "rmdir"),
    ("os", "removedirs"),
    ("subprocess", "run"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "check_output"),
    ("socket", "socket"),
    ("shutil", "rmtree"),
    ("ctypes", "CDLL"),
)

# Bare-name calls that must never appear.
BANNED_CALLS: Set[str] = {
    "eval", "exec", "compile", "__import__", "open",
    "input", "breakpoint",
}

# Optional: writing to files at import time is banned; the sandbox
# also enforces this, but catching it here gives a clearer message.
DUNDER_WRITE_ATTRS: Set[str] = {
    "write", "writelines", "truncate", "unlink", "remove",
}


# ── Result ────────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    imports_seen: List[str] = field(default_factory=list)
    calls_seen: List[str] = field(default_factory=list)
    has_run_fn: bool = False
    has_docstring: bool = False
    docstring: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "imports_seen": list(self.imports_seen),
            "calls_seen": list(self.calls_seen[:20]),
            "has_run_fn": self.has_run_fn,
            "has_docstring": self.has_docstring,
            "docstring": self.docstring,
        }


# ── Helpers ───────────────────────────────────────────────────────────────

def _root_import_name(name: str) -> str:
    """"pkg.sub.mod" -> "pkg". Also handles the "components.foo" prefix."""
    if name.startswith("components."):
        # Keep two levels so components.knowledge is treated as one unit.
        parts = name.split(".", 2)
        return ".".join(parts[:2])
    return name.split(".", 1)[0]


def _is_allowed_import(name: str) -> bool:
    if not name:
        return False
    root = _root_import_name(name)
    return root in ALLOWED_IMPORTS


def _attr_chain(node: ast.AST) -> Tuple[str, ...]:
    """For an Attribute node like os.path.join return ("os","path","join")."""
    parts: List[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.insert(0, cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.insert(0, cur.id)
    return tuple(parts)


# ── Walker ────────────────────────────────────────────────────────────────

class _Walker(ast.NodeVisitor):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report

    # -- imports
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.report.imports_seen.append(alias.name)
            if not _is_allowed_import(alias.name):
                self.report.reasons.append(
                    f"banned_import: {alias.name} (root "
                    f"{_root_import_name(alias.name)} not in allowlist)"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if node.level and node.level > 0:
            # Relative imports are rejected — generated code must be
            # self-contained.
            self.report.reasons.append(
                f"banned_import: relative import 'from {'.'*node.level}"
                f"{mod} import ...'"
            )
        else:
            self.report.imports_seen.append(mod)
            if not _is_allowed_import(mod):
                self.report.reasons.append(
                    f"banned_import: from {mod} import ... (root "
                    f"{_root_import_name(mod)} not in allowlist)"
                )
        self.generic_visit(node)

    # -- calls
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # Bare-name banned calls (eval, exec, open, ...)
        if isinstance(func, ast.Name):
            self.report.calls_seen.append(func.id)
            if func.id in BANNED_CALLS:
                self.report.reasons.append(f"banned_call: {func.id}(...)")

        # Attribute-chain banned calls (os.system, subprocess.run, ...)
        if isinstance(func, ast.Attribute):
            chain = _attr_chain(func)
            self.report.calls_seen.append(".".join(chain))
            for banned in BANNED_ATTR_CHAINS:
                if chain[-len(banned):] == banned:
                    self.report.reasons.append(
                        f"banned_call: {'.'.join(chain)}(...)"
                    )
                    break

        self.generic_visit(node)


# ── Entry point ───────────────────────────────────────────────────────────

def validate_source(source: str,
                    *,
                    require_run_fn: bool = True,
                    require_docstring: bool = True) -> ValidationReport:
    """Parse *source* and check it against the capability policy.

    Never raises. If the source is not valid Python at all, the
    returned report has ``ok=False`` and a single ``parse_error``
    reason.
    """
    report = ValidationReport(ok=False)

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        report.reasons.append(f"parse_error: {e.msg} at line {e.lineno}")
        return report

    # Module docstring + run() presence.
    docstring = ast.get_docstring(tree)
    if docstring:
        report.has_docstring = True
        report.docstring = docstring
    elif require_docstring:
        report.reasons.append("missing_docstring")

    for node in tree.body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "run"):
            report.has_run_fn = True
            break
    if require_run_fn and not report.has_run_fn:
        report.reasons.append("missing_run_function")

    # Walk everything.
    _Walker(report).visit(tree)

    report.ok = not report.reasons
    return report


__all__ = [
    "validate_source",
    "ValidationReport",
    "ALLOWED_IMPORTS",
    "BANNED_ATTR_CHAINS",
    "BANNED_CALLS",
]
