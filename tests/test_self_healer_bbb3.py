"""PR BBB-3: self_healer must not log phantom syntax errors for files
that no longer exist on disk, and must skip the materialiser's
ephemeral staging directory entirely.

Prod symptom before this fix:
    Syntax error in components/generated/staging/consistency_assertion_cron.py:
        No such file or directory

Root cause: rglob() built the file list, then the materialiser's
cleanup block deleted the staging .py before _syntax_ok() got to it,
so path.read_text() threw FileNotFoundError which the generic
Exception branch stringified into the "syntax error" log line.
"""
from __future__ import annotations

import importlib
from pathlib import Path


def test_syntax_ok_missing_file_returns_ok():
    """A vanished file is not a syntax error - it's a benign race."""
    healer = importlib.import_module("components.self_management.self_healer")
    missing = Path("/tmp/definitely_does_not_exist_bbb3_a1b2c3.py")
    if missing.exists():
        missing.unlink()
    ok, err = healer._syntax_ok(missing)
    assert ok is True, f"missing file should be ok=True, got err={err!r}"
    assert "file_missing" in err or err == "", err


def test_syntax_ok_real_syntax_error_still_fails(tmp_path):
    """Real syntax errors must still be reported so real repair fires."""
    healer = importlib.import_module("components.self_management.self_healer")
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n    pass\n")  # unmistakable syntax error
    ok, err = healer._syntax_ok(bad)
    assert ok is False
    assert "SyntaxError" in err


def test_syntax_ok_valid_file_passes(tmp_path):
    """Sanity: valid Python still parses cleanly."""
    healer = importlib.import_module("components.self_management.self_healer")
    good = tmp_path / "good.py"
    good.write_text("x = 1\ndef f():\n    return x\n")
    ok, err = healer._syntax_ok(good)
    assert ok is True
    assert err == ""


def test_staging_dir_is_excluded_from_sweep(tmp_path):
    """components/generated/staging/*.py must be excluded from the sweep."""
    healer = importlib.import_module("components.self_management.self_healer")
    root = tmp_path
    staging = root / "components" / "generated" / "staging"
    staging.mkdir(parents=True)
    staged_file = staging / "ephemeral_cap.py"
    staged_file.write_text("x = 1\n")
    assert healer._is_excluded_path(staged_file, root) is True


def test_live_dir_still_included(tmp_path):
    """components/generated/live/ is real promoted source and must be swept."""
    healer = importlib.import_module("components.self_management.self_healer")
    root = tmp_path
    live = root / "components" / "generated" / "live"
    live.mkdir(parents=True)
    live_file = live / "promoted_cap.py"
    live_file.write_text("x = 1\n")
    assert healer._is_excluded_path(live_file, root) is False


def test_backup_dirs_still_excluded(tmp_path):
    """Regression: legacy exclusions still work after adding staging."""
    healer = importlib.import_module("components.self_management.self_healer")
    root = tmp_path
    backup = root / "components" / "backup_final_20260101"
    backup.mkdir(parents=True)
    backup_file = backup / "old.py"
    backup_file.write_text("x = 1\n")
    assert healer._is_excluded_path(backup_file, root) is True


def test_pycache_still_excluded(tmp_path):
    """Regression: __pycache__ still excluded after adding staging."""
    healer = importlib.import_module("components.self_management.self_healer")
    root = tmp_path
    pyc = root / "components" / "some_mod" / "__pycache__"
    pyc.mkdir(parents=True)
    pyc_file = pyc / "some_mod.cpython-312.pyc"
    pyc_file.write_text("")  # not real .pyc but path check is what matters
    assert healer._is_excluded_path(pyc_file, root) is True
