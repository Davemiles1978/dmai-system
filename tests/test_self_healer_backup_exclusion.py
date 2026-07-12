"""Tests for SelfHealer backup / cache directory exclusion (PR A).

Root cause covered: `_sweep_components` used to `rglob("*.py")` under
components/ and treat every backup snapshot (e.g.
components/backup_final_20260316_175710/) as live code. Every sweep
filed duplicate "Auto-repair needed:" Kaizen proposals for files that
were never meant to be executed.

These tests pin:
  1. Filesystem paths under excluded dirs are NOT scanned.
  2. Real live components under components/ still ARE scanned.
  3. `_retire_excluded_kaizen_entries` cleans up stale JSONL entries
     from both kaizen_proposals.jsonl and kaizen_queue.jsonl, without
     touching entries that point at live code.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from components.self_management.self_healer import (
    EXCLUDED_DIR_NAMES,
    EXCLUDED_DIR_PREFIXES,
    SelfHealer,
    _is_excluded_path,
)


def test_is_excluded_matches_backup_final(tmp_path: Path):
    root = tmp_path
    (root / "components" / "backup_final_20260316_175710" / "phase1").mkdir(parents=True)
    target = root / "components" / "backup_final_20260316_175710" / "phase1" / "P1T4.py"
    target.write_text("nonsense not python")
    assert _is_excluded_path(target, root) is True


def test_is_excluded_matches_backup_before_renumber(tmp_path: Path):
    root = tmp_path
    (root / "components" / "backup_before_renumber_20260316_175222").mkdir(parents=True)
    target = root / "components" / "backup_before_renumber_20260316_175222" / "P.py"
    target.write_text("x")
    assert _is_excluded_path(target, root) is True


def test_is_excluded_matches_pycache_and_dist_info(tmp_path: Path):
    root = tmp_path
    (root / "components" / "foo" / "__pycache__").mkdir(parents=True)
    (root / "components" / "some.dist-info").mkdir(parents=True)
    p1 = root / "components" / "foo" / "__pycache__" / "foo.cpython-311.pyc"
    p1.write_text("")
    p2 = root / "components" / "some.dist-info" / "METADATA"
    p2.write_text("")
    assert _is_excluded_path(p1, root) is True
    assert _is_excluded_path(p2, root) is True


def test_is_excluded_does_not_match_live_component(tmp_path: Path):
    root = tmp_path
    (root / "components" / "wealth").mkdir(parents=True)
    live = root / "components" / "wealth" / "autonomous_trader.py"
    live.write_text("def foo(): pass")
    assert _is_excluded_path(live, root) is False


def test_exclusions_are_not_empty():
    # Static guard against accidentally emptying the exclusion set.
    assert "__pycache__" in EXCLUDED_DIR_NAMES
    assert "backup_final_" in EXCLUDED_DIR_PREFIXES
    assert "backup_before_" in EXCLUDED_DIR_PREFIXES


def test_retire_removes_backup_entries_but_keeps_live(tmp_path: Path):
    root = tmp_path
    data = root / "data"
    data.mkdir()
    proposals = data / "kaizen_proposals.jsonl"
    queue     = data / "kaizen_queue.jsonl"

    entries = [
        # These 3 should be REMOVED (all point at excluded dirs)
        {"file": "components/backup_final_20260316_175710/phase1/P1T4.py",
         "title": "Auto-repair needed", "status": "pending"},
        {"file_path": "components/backup_before_renumber_20260316_175222/P.py",
         "title": "Auto-repair needed", "status": "pending"},
        {"file": "components/foo/__pycache__/foo.cpython-311.pyc",
         "title": "stale pyc", "status": "pending"},
        # These 2 should be KEPT (live code)
        {"file": "components/wealth/autonomous_trader.py",
         "title": "real proposal", "status": "pending"},
        {"file_path": "dmai_core_complete.py",
         "title": "top-level real", "status": "pending"},
    ]
    for f in (proposals, queue):
        f.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    h = SelfHealer(repo_root=str(root))
    h._retire_excluded_kaizen_entries()

    for f in (proposals, queue):
        kept = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        assert len(kept) == 2, f"{f.name}: expected 2 kept, got {len(kept)}"
        titles = [e["title"] for e in kept]
        assert "real proposal" in titles
        assert "top-level real" in titles
        assert "Auto-repair needed" not in titles


def test_retire_is_idempotent(tmp_path: Path):
    root = tmp_path
    data = root / "data"
    data.mkdir()
    proposals = data / "kaizen_proposals.jsonl"
    proposals.write_text(json.dumps(
        {"file": "components/backup_final_x/y.py", "title": "stale"}
    ) + "\n")

    h = SelfHealer(repo_root=str(root))
    h._retire_excluded_kaizen_entries()
    h._retire_excluded_kaizen_entries()  # second run must not error / re-clean

    remaining = [l for l in proposals.read_text().splitlines() if l.strip()]
    assert remaining == []


def test_retire_preserves_unparseable_lines(tmp_path: Path):
    """A corrupted line must NOT be silently discarded (that's data loss)."""
    root = tmp_path
    data = root / "data"
    data.mkdir()
    proposals = data / "kaizen_proposals.jsonl"
    proposals.write_text(
        "GARBAGE_NOT_JSON\n"
        + json.dumps({"file": "components/backup_final_x/y.py", "title": "stale"}) + "\n"
        + json.dumps({"file": "components/foo/live.py", "title": "keep"}) + "\n"
    )

    h = SelfHealer(repo_root=str(root))
    h._retire_excluded_kaizen_entries()

    lines = proposals.read_text().splitlines()
    assert "GARBAGE_NOT_JSON" in lines
    assert any('"title": "keep"' in l for l in lines)
    assert not any('"title": "stale"' in l for l in lines)


def test_retire_no_op_when_files_absent(tmp_path: Path):
    root = tmp_path
    (root / "data").mkdir()
    h = SelfHealer(repo_root=str(root))
    # Should not raise even though neither kaizen_*.jsonl exists.
    h._retire_excluded_kaizen_entries()
