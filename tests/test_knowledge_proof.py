"""Tests for components.knowledge_proof.

Verify the three probes work against a hand-crafted SQLite DB shaped
like prod, and that overall_ok honours the "insights + (caps OR callable)"
rule.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import textwrap
from pathlib import Path

import pytest

from components.knowledge_proof import (
    KnowledgeProofResult,
    ProbeResult,
    probe_capabilities_stored,
    probe_capability_callable,
    probe_insights_stored,
    run_knowledge_proof,
)


@pytest.fixture
def seeded_kdb(tmp_path: Path) -> str:
    """Build a prod-shaped dmai_knowledge.db."""
    path = tmp_path / "dmai_knowledge.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE insights (
            id INTEGER PRIMARY KEY,
            source_topic TEXT,
            target_topic TEXT,
            concept TEXT,
            domain TEXT,
            confidence REAL,
            provenance TEXT,
            created_ts TEXT
        );
        INSERT INTO insights (id, source_topic, target_topic, concept, domain,
                              confidence, provenance, created_ts)
        VALUES
          (1, 'trading', 'analytics', 'trading', 'analytics', 0.9,
           'promoter_path', datetime('now')),
          (2, 'greyhounds', 'racing', 'greyhounds', 'racing', 0.85,
           'promoter_path', datetime('now', '-2 hours'));

        CREATE TABLE capabilities (
            id TEXT PRIMARY KEY,
            name TEXT,
            capability_type TEXT,
            description TEXT,
            runtime_mode TEXT,
            provenance TEXT,
            judge_confidence REAL,
            source_insight_id INTEGER
        );
        INSERT INTO capabilities VALUES
          ('cap-1', 'analyse_trade_flow', 'trading',
           'Analyse the flow of trades against a benchmark and detect anomalies.',
           'stub', 'promoter_path+self_judge', 0.75, 1),
          ('cap-2', 'summarise_greyhound_form', 'analytics',
           'Summarise recent greyhound racing form to detect trend shifts.',
           'generated_module', 'fresh_blood_seed+self_judge', 0.82, 2);
        """
    )
    conn.commit()
    conn.close()

    # Write a matching live module for cap-2 so probe_capabilities_stored
    # and probe_capability_callable find something.
    live_dir = Path(__file__).resolve().parents[1] / "components" / "generated" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    module_path = live_dir / "summarise_greyhound_form.py"
    module_path.write_text(textwrap.dedent('''
        """Summarise recent greyhound racing form to detect trend shifts.

        Aggregates form data across race meets and returns a compact digest.
        """
        def run(**kwargs):
            return {"summary": "test", "shifts": []}
    ''').lstrip(), encoding="utf-8")

    yield str(path)

    # Cleanup
    try:
        module_path.unlink()
    except FileNotFoundError:
        pass


def test_probe_insights_stored_passes_on_recent(seeded_kdb: str) -> None:
    conn = sqlite3.connect(seeded_kdb)
    conn.row_factory = sqlite3.Row
    result = probe_insights_stored(conn)
    conn.close()
    assert isinstance(result, ProbeResult)
    assert result.ok, f"probe failed: {result.detail} checks={result.checks}"
    assert result.checks["roundtrip_by_id"]
    assert result.checks["topic_present"]


def test_probe_insights_stored_falls_back_when_no_recent(tmp_path: Path) -> None:
    """No recent insight but an older one exists — should still find it."""
    path = tmp_path / "kdb.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE insights (
            id INTEGER PRIMARY KEY,
            source_topic TEXT, target_topic TEXT,
            created_ts TEXT
        );
        INSERT INTO insights VALUES
          (1, 'old', 'stuff', datetime('now', '-30 days'));
        """
    )
    conn.commit()
    conn.row_factory = sqlite3.Row
    result = probe_insights_stored(conn, lookback_hours=1)
    conn.close()
    assert result.ok
    assert "older sample" in result.detail


def test_probe_insights_stored_fails_on_empty_table(tmp_path: Path) -> None:
    path = tmp_path / "kdb.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE insights (id INTEGER PRIMARY KEY, created_ts TEXT)"
    )
    conn.commit()
    conn.row_factory = sqlite3.Row
    result = probe_insights_stored(conn)
    conn.close()
    assert not result.ok
    assert "empty" in result.detail


def test_probe_capabilities_stored_verifies_live_module(seeded_kdb: str) -> None:
    conn = sqlite3.connect(seeded_kdb)
    conn.row_factory = sqlite3.Row
    result = probe_capabilities_stored(conn)
    conn.close()
    assert result.ok
    # We should have found the generated_module and verified it parses.
    assert result.checks.get("module_file_exists") is True
    assert result.checks.get("module_file_parses") is True


def test_probe_capability_callable_executes_run(seeded_kdb: str) -> None:
    conn = sqlite3.connect(seeded_kdb)
    conn.row_factory = sqlite3.Row
    result = probe_capability_callable(conn, timeout_sec=10)
    conn.close()
    assert result.ok, (
        f"callable probe failed: {result.detail}; checks={result.checks}"
    )
    assert result.checks["import_ok"]
    assert result.checks["run_ok"]


def test_probe_capability_callable_reports_missing_gracefully(tmp_path: Path) -> None:
    """When no generated_module capabilities exist, probe reports it cleanly."""
    path = tmp_path / "kdb.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE capabilities (
            id TEXT PRIMARY KEY, name TEXT, capability_type TEXT,
            description TEXT, runtime_mode TEXT
        );
        INSERT INTO capabilities VALUES
          ('x', 'thing', 'utility', 'A thing.', 'stub');
        """
    )
    conn.commit()
    conn.row_factory = sqlite3.Row
    result = probe_capability_callable(conn)
    conn.close()
    assert not result.ok
    assert "no generated_module" in result.detail


def test_run_knowledge_proof_end_to_end(seeded_kdb: str, tmp_path: Path) -> None:
    """Full run with a healthy DB should return overall_ok=True."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # KnowledgeProof looks for dmai_knowledge.db under data_path
    target = data_dir / "dmai_knowledge.db"
    import shutil
    shutil.copy(seeded_kdb, target)

    result = run_knowledge_proof(data_path=str(data_dir),
                                 callable_timeout_sec=10)
    assert isinstance(result, KnowledgeProofResult)
    assert result.overall_ok, (
        f"expected overall_ok, got probes: "
        f"{[(p.name, p.ok, p.detail) for p in result.probes]}"
    )
    assert len(result.probes) == 3
    assert result.counts["insights_total"] == 2
    assert result.counts["capabilities_total"] == 2
    assert result.counts["generated_modules"] == 1


def test_run_knowledge_proof_handles_missing_db(tmp_path: Path) -> None:
    """No DB at path → clean failure, no exception."""
    result = run_knowledge_proof(data_path=str(tmp_path))
    assert not result.overall_ok
    assert result.probes[0].name == "setup"
    assert "not found" in result.probes[0].detail


def test_probe_capabilities_stored_flags_broken_module(seeded_kdb: str,
                                                      tmp_path: Path) -> None:
    """A generated_module row whose file is missing → probe fails."""
    conn = sqlite3.connect(seeded_kdb)
    conn.execute(
        "INSERT INTO capabilities VALUES "
        "('cap-3', 'nonexistent_module', 'utility', 'Not on disk.', "
        " 'generated_module', 'x', 0.9, NULL)"
    )
    conn.commit()
    conn.row_factory = sqlite3.Row

    # Rig RANDOM() to pick cap-3 by deleting cap-2 (only other gen module)
    conn.execute("DELETE FROM capabilities WHERE id = 'cap-2'")
    conn.commit()

    result = probe_capabilities_stored(conn)
    conn.close()
    assert not result.ok
    assert result.checks.get("module_file_exists") is False
