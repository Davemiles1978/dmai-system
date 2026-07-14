"""Tests for components.generated._validator.

Covers:
- accept a clean module with docstring + run()
- reject when module docstring missing
- reject when run() missing
- reject banned imports (subprocess, os.system chain)
- reject banned calls (eval/exec/open)
- accept when only allow-listed imports are used
- parse errors are surfaced as parse_error reasons
"""
from __future__ import annotations

import textwrap

from components.generated._validator import (
    validate_source, ALLOWED_IMPORTS,
)


def _src(body: str) -> str:
    return textwrap.dedent(body).lstrip("\n")


def test_accepts_clean_module():
    src = _src(
        '''
        """A tiny capability that sums its inputs."""
        from __future__ import annotations
        import math
        import statistics

        def run(values):
            return math.fsum(values) + statistics.mean(values or [0])
        '''
    )
    report = validate_source(src)
    assert report.ok, report.reasons
    assert report.has_run_fn is True
    assert report.has_docstring is True
    assert "math" in report.imports_seen
    assert "statistics" in report.imports_seen


def test_missing_module_docstring_rejected():
    src = _src(
        '''
        import math

        def run():
            return math.pi
        '''
    )
    report = validate_source(src)
    assert not report.ok
    assert any(r.startswith("missing_docstring") for r in report.reasons)


def test_missing_run_function_rejected():
    src = _src(
        '''
        """No run here."""

        def helper():
            return 42
        '''
    )
    report = validate_source(src)
    assert not report.ok
    assert any(r.startswith("missing_run_function") for r in report.reasons)


def test_banned_import_subprocess():
    src = _src(
        '''
        """Malicious."""
        import subprocess

        def run():
            return subprocess.check_output(["ls"])
        '''
    )
    report = validate_source(src)
    assert not report.ok
    assert any(r.startswith("banned_import:") for r in report.reasons)


def test_banned_call_eval():
    src = _src(
        '''
        """Naughty."""
        def run(expr):
            return eval(expr)
        '''
    )
    report = validate_source(src)
    assert not report.ok
    assert any(r.startswith("banned_call:") for r in report.reasons)


def test_banned_attr_chain_os_system():
    src = _src(
        '''
        """Naughty."""
        import os

        def run():
            return os.system("echo hi")
        '''
    )
    report = validate_source(src)
    assert not report.ok
    # os itself isn't on the allow-list, so it should trip either the
    # import block or the attr-chain block. Both are acceptable.
    assert any(r.startswith(("banned_import:", "banned_call:"))
               for r in report.reasons)


def test_parse_error_surfaced():
    src = "def run(:\n    pass\n"
    report = validate_source(src)
    assert not report.ok
    assert any(r.startswith("parse_error:") for r in report.reasons)


def test_allowlist_contains_expected_modules():
    for mod in ("math", "statistics", "json", "re", "hashlib",
                "components.knowledge", "components.self_judge",
                "sqlite3"):
        assert mod in ALLOWED_IMPORTS, mod
