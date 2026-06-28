#!/usr/bin/env python3
"""Static decorator scan: detect any @decorator that doesn't resolve to a defined or imported name.

This catches the class of bug that crashed production on 2026-06-26:
a commit used @require_master_password which doesn't exist anywhere in the repo.
py_compile passes because decorators are looked up at import time, not compile time.

Exits 1 on first unresolved decorator. Exits 0 if all decorators resolve.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

# Decorators that are always considered safe (stdlib / Flask / common libs).
# Match by the full dotted head expression (before any call args).
ALWAYS_OK = {
    # stdlib
    "staticmethod", "classmethod", "property", "abstractmethod",
    "cached_property", "lru_cache", "wraps", "functools.wraps",
    "contextmanager", "asynccontextmanager",
    "dataclass", "dataclasses.dataclass",
    "final", "override",
    # Flask common
    "app.route", "app.before_request", "app.after_request",
    "app.errorhandler", "app.teardown_appcontext",
    "app.cli.command", "app.template_filter", "app.template_global",
    "app.context_processor", "app.before_first_request",
    "app.url_value_preprocessor", "app.url_defaults",
    # pytest
    "pytest.fixture",
    # SQLAlchemy / common
    "validates", "hybrid_property", "event.listens_for",
    # Click
    "click.command", "click.group", "click.option", "click.argument",
}

ALWAYS_OK_PREFIXES = (
    "pytest.mark.",
    "app.cli.",
    "blueprint.",
    "bp.",
)


def decorator_head(node: ast.expr) -> str:
    """Return the dotted-name 'head' of a decorator expression, stripping any Call args.

    Examples:
        @foo                  -> "foo"
        @foo.bar              -> "foo.bar"
        @app.route("/x")      -> "app.route"
        @some(arg=1)          -> "some"
        @a.b.c(x)             -> "a.b.c"
    """
    if isinstance(node, ast.Call):
        return decorator_head(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{decorator_head(node.value)}.{node.attr}"
    # Subscripts and other exotic forms: return repr as fallback
    return f"<unresolvable:{type(node).__name__}>"


def collect_defined_names(tree: ast.AST) -> set[str]:
    """Collect all top-level and nested names defined or imported in the module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                elif isinstance(t, ast.Tuple):
                    for elt in t.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def head_is_ok(head: str, defined: set[str]) -> bool:
    if head in ALWAYS_OK:
        return True
    for prefix in ALWAYS_OK_PREFIXES:
        if head.startswith(prefix):
            return True
    # Take the leftmost segment — that's the symbol that must resolve in the module
    root = head.split(".")[0]
    # `self`/`cls` decorators are bound-method calls evaluated at runtime inside
    # a method body, not at module import. They never cause boot crashes — skip.
    if root in ("self", "cls"):
        return True
    if root in defined:
        return True
    # Allow module-level references like 'os.path.join' if 'os' is imported
    return False


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return list of (lineno, decorator_head) for unresolved decorators."""
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"[WARN] cannot read {path}: {e}", file=sys.stderr)
        return []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        print(f"[FAIL] syntax error in {path}: {e}", file=sys.stderr)
        return [(e.lineno or 0, f"<syntax:{e.msg}>")]
    defined = collect_defined_names(tree)
    bad: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                head = decorator_head(dec)
                if not head_is_ok(head, defined):
                    bad.append((dec.lineno, head))
    return bad


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    targets: list[Path] = [repo_root / "dmai_core_complete.py"]
    components_dir = repo_root / "components"
    if components_dir.is_dir():
        targets.extend(sorted(components_dir.rglob("*.py")))

    total_bad = 0
    files_checked = 0
    for path in targets:
        if not path.is_file():
            continue
        files_checked += 1
        bad = scan_file(path)
        if bad:
            rel = path.relative_to(repo_root)
            for lineno, head in bad:
                print(f"[FAIL] {rel}:{lineno} unresolved decorator: @{head}")
                total_bad += 1

    if total_bad:
        print(f"\n[FAIL] decorator scan: {total_bad} unresolved decorator(s) across {files_checked} files")
        print("       → these will crash module import at @decorator evaluation time")
        return 1
    print(f"[PASS] decorator scan: 0 unresolved decorators across {files_checked} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
