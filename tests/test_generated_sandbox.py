"""Tests for components.generated._sandbox.

We exercise the sandbox by writing tiny modules into
components/generated/staging/ and asking the sandbox to run them.
Everything is cleaned up afterwards.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from components.generated._sandbox import (
    run_happy_path, run_pytest_file, SandboxResult, REPO_ROOT,
)


STAGING = REPO_ROOT / "components" / "generated" / "staging"


def _write(name: str, body: str) -> Path:
    STAGING.mkdir(parents=True, exist_ok=True)
    p = STAGING / f"{name}.py"
    p.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    return p


def _cleanup(name: str) -> None:
    p = STAGING / f"{name}.py"
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def test_happy_path_returns_result():
    name = "sandbox_ok_case"
    _write(name, '''
        """Sample."""
        def run(a=0, b=0):
            return {"sum": a + b}
    ''')
    try:
        r = run_happy_path(
            f"components.generated.staging.{name}",
            {"a": 3, "b": 4},
        )
    finally:
        _cleanup(name)
    assert isinstance(r, SandboxResult)
    assert r.ok is True, (r.reason, r.stderr)
    assert r.return_value == {"sum": 7}
    assert not r.timed_out


def test_happy_path_timeout():
    name = "sandbox_timeout_case"
    # Note: no non-allow-listed imports needed for a busy loop.
    _write(name, '''
        """Slow."""
        def run(**kw):
            x = 0
            while True:
                x += 1
    ''')
    try:
        r = run_happy_path(
            f"components.generated.staging.{name}",
            {}, timeout_sec=1, cpu_sec=1,
        )
    finally:
        _cleanup(name)
    assert r.ok is False
    assert r.timed_out or r.reason.startswith(("timeout", "runtime_error",
                                               "pytest_exit_"))


def test_happy_path_runtime_error():
    name = "sandbox_boom_case"
    _write(name, '''
        """Boom."""
        def run(**kw):
            raise ValueError("nope")
    ''')
    try:
        r = run_happy_path(
            f"components.generated.staging.{name}",
            {},
        )
    finally:
        _cleanup(name)
    assert r.ok is False
    assert "runtime_error" in r.reason or r.reason.startswith("pytest_exit_")


def test_run_pytest_file_pass(tmp_path):
    tf = tmp_path / "test_dummy_ok.py"
    tf.write_text(textwrap.dedent('''
        def test_truth():
            assert 1 + 1 == 2
    ''').lstrip("\n"), encoding="utf-8")
    r = run_pytest_file(tf)
    assert r.ok is True, (r.reason, r.stdout[-400:], r.stderr[-400:])


def test_run_pytest_file_fail(tmp_path):
    tf = tmp_path / "test_dummy_fail.py"
    tf.write_text(textwrap.dedent('''
        def test_bad():
            assert 1 == 2
    ''').lstrip("\n"), encoding="utf-8")
    r = run_pytest_file(tf)
    assert r.ok is False
    assert "pytest_exit_" in r.reason or "pytest_timeout" in r.reason
