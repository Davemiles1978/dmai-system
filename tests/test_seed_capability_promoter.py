"""Tests for components/seed_capability_promoter.py.

Exercise the full pipeline: JSONL tail-follower -> self_judge ->
(accept / reject / defer) -> registry write + deferred queue + reject
log + judge stats + acquirer trampoline.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pytest

from components import seed_capability_promoter as scp


# --- Fixtures -------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    db = tmp_path / "kdb.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE system_state (
          key TEXT PRIMARY KEY,
          value TEXT,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE capabilities (
          id TEXT PRIMARY KEY,
          name TEXT,
          type TEXT,
          capability_type TEXT
        );
        CREATE TABLE vocabulary (
          id TEXT PRIMARY KEY,
          word TEXT UNIQUE,
          part_of_speech TEXT,
          definition TEXT,
          etymology TEXT,
          domain TEXT,
          added_at TEXT
        );
        CREATE INDEX idx_vocab_word ON vocabulary(word);
        CREATE TABLE insights (
          id TEXT PRIMARY KEY,
          insight_text TEXT,
          entity_type TEXT,
          entities TEXT,
          relationship TEXT,
          confidence REAL,
          source_topic TEXT,
          target_topic TEXT,
          source TEXT,
          created_at TEXT
        );
        """
    )
    conn.commit()
    conn.close()
    return str(db)


@pytest.fixture
def tmp_registry(tmp_path) -> Path:
    return tmp_path / "capabilities" / "registry.json"


@pytest.fixture
def tmp_jsonl(tmp_path) -> Path:
    p = tmp_path / "fresh_blood" / "insights.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_jsonl(path: Path, seeds: List[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s) + "\n")


def _make_judge(verdict_map: Dict[str, Dict[str, Any]]):
    """Test double for self_judge; keyed by seed concept."""
    default = {
        "verdict":        scp.VERDICT_ACCEPT,
        "confidence":     0.9,
        "reason":         "test-accept",
        "gap_summary":    "",
        "unknown_tokens": [],
    }

    def _judge(seed, db_path):
        concept = str(seed.get("concept", ""))
        return verdict_map.get(concept, default)
    return _judge


# --- Basic accept path ---------------------------------------------------

def test_accept_seed_writes_to_registry(tmp_db, tmp_registry, tmp_jsonl):
    seeds = [{
        "channel": "arxiv",
        "concept": "novel attention mechanism",
        "insight_text": "very novel",
        "seed_hash": "aaaaa1",
    }]
    _write_jsonl(tmp_jsonl, seeds)

    summary = scp.promote_once(
        registry_path=tmp_registry, jsonl_path=tmp_jsonl,
        db_path=tmp_db, judge=_make_judge({}),
    )
    assert summary["promoted"] == 1
    assert summary["rejected"] == 0
    assert summary["deferred_new"] == 0
    reg = json.loads(tmp_registry.read_text())
    assert any("attention" in cid for cid in reg["capabilities"])
    # The self-judge confidence and provenance are attached to the entry.
    entry = next(iter(reg["capabilities"].values()))
    assert entry["provenance"] == "fresh_blood_seed+self_judge"
    assert "judge_confidence" in entry


# --- Reject path ---------------------------------------------------------

def test_reject_seed_recorded_in_reject_log(tmp_db, tmp_registry, tmp_jsonl):
    seeds = [{
        "channel": "arxiv",
        "concept": "duplicate concept",
        "insight_text": "seen this before",
        "seed_hash": "bbbbb1",
    }]
    _write_jsonl(tmp_jsonl, seeds)

    judge = _make_judge({
        "duplicate concept": {
            "verdict":        scp.VERDICT_REJECT,
            "confidence":     0.05,
            "reason":         "near-duplicate of insight foo",
            "gap_summary":    "",
            "unknown_tokens": [],
        }
    })

    summary = scp.promote_once(
        registry_path=tmp_registry, jsonl_path=tmp_jsonl,
        db_path=tmp_db, judge=judge,
    )
    assert summary["promoted"] == 0
    assert summary["rejected"] == 1

    conn = sqlite3.connect(tmp_db)
    row = conn.execute(
        "SELECT value FROM system_state WHERE key = ?",
        (scp.REJECT_LOG_KEY,),
    ).fetchone()
    conn.close()
    entries = json.loads(row[0])
    assert entries and entries[-1]["reason"] == scp.REJECT_SELF_JUDGE
    assert "duplicate" in entries[-1]["detail"]


# --- Defer path + acquirer trampoline -----------------------------------

def test_defer_seed_queued_and_acquirer_fired(tmp_db, tmp_registry, tmp_jsonl):
    seeds = [{
        "channel": "arxiv",
        "concept": "obscure topology term",
        "insight_text": "very obscure",
        "seed_hash": "ccccc1",
    }]
    _write_jsonl(tmp_jsonl, seeds)

    judge = _make_judge({
        "obscure topology term": {
            "verdict":        scp.VERDICT_DEFER,
            "confidence":     0.15,
            "reason":         "vocab_coverage below floor",
            "gap_summary":    "unknown: quaternion, solitons",
            "unknown_tokens": ["quaternion", "solitons"],
        }
    })

    calls: List[Dict[str, Any]] = []

    def _acq(concept, gap, unknown_tokens):
        calls.append({"concept": concept, "gap": gap,
                      "unknown_tokens": list(unknown_tokens)})

    summary = scp.promote_once(
        registry_path=tmp_registry, jsonl_path=tmp_jsonl,
        db_path=tmp_db, judge=judge, acquire_gap=_acq,
    )
    assert summary["deferred_new"] == 1
    assert summary["promoted"] == 0
    assert calls and calls[0]["concept"] == "obscure topology term"
    assert calls[0]["unknown_tokens"] == ["quaternion", "solitons"]

    # Row exists in deferred_seeds.
    conn = sqlite3.connect(tmp_db)
    row = conn.execute(
        "SELECT concept, reason FROM deferred_seeds"
    ).fetchone()
    conn.close()
    assert row and row[0] == "obscure topology term"


def test_defer_second_pass_bumps_attempts_not_duplicates(
    tmp_db, tmp_registry, tmp_jsonl,
):
    seed = {
        "channel": "arxiv", "concept": "same seed",
        "insight_text": "s", "seed_hash": "dddd1",
    }
    _write_jsonl(tmp_jsonl, [seed])
    judge = _make_judge({
        "same seed": {
            "verdict":     scp.VERDICT_DEFER, "confidence": 0.2,
            "reason":      "vocab", "gap_summary": "unknown foo",
            "unknown_tokens": ["foo"],
        }
    })

    scp.promote_once(
        registry_path=tmp_registry, jsonl_path=tmp_jsonl,
        db_path=tmp_db, judge=judge,
    )
    # Re-append the same seed hash - simulates a second pass reading
    # the same JSONL row (idempotence check on the defer queue itself
    # since offset advances, but same_hash comes back).
    _write_jsonl(tmp_jsonl, [seed])
    scp.promote_once(
        registry_path=tmp_registry, jsonl_path=tmp_jsonl,
        db_path=tmp_db, judge=judge,
    )
    conn = sqlite3.connect(tmp_db)
    row = conn.execute(
        "SELECT attempts FROM deferred_seeds WHERE seed_hash = ?",
        ("dddd1",),
    ).fetchone()
    conn.close()
    assert row and row[0] >= 2


# --- Daily cap -----------------------------------------------------------

def test_daily_cap_hits_and_stops_promotion(tmp_db, tmp_registry, tmp_jsonl):
    seeds = [{
        "channel": "arxiv", "concept": f"topic-{i}",
        "insight_text": "x", "seed_hash": f"e{i:05d}",
    } for i in range(5)]
    _write_jsonl(tmp_jsonl, seeds)

    summary = scp.promote_once(
        registry_path=tmp_registry, jsonl_path=tmp_jsonl,
        db_path=tmp_db, daily_cap=3, judge=_make_judge({}),
    )
    assert summary["promoted"] == 3
    assert summary["cap_hit"] is True
    assert summary["day_count_after"] == 3


# --- Judge stats surface -------------------------------------------------

def test_judge_stats_track_verdicts(tmp_db, tmp_registry, tmp_jsonl):
    seeds = [
        {"channel": "arxiv", "concept": "a", "insight_text": "a",
         "seed_hash": "f001"},
        {"channel": "arxiv", "concept": "b", "insight_text": "b",
         "seed_hash": "f002"},
        {"channel": "arxiv", "concept": "c", "insight_text": "c",
         "seed_hash": "f003"},
    ]
    _write_jsonl(tmp_jsonl, seeds)

    judge = _make_judge({
        "b": {"verdict": scp.VERDICT_REJECT, "confidence": 0.1,
              "reason": "dup", "gap_summary": "", "unknown_tokens": []},
        "c": {"verdict": scp.VERDICT_DEFER, "confidence": 0.4,
              "reason": "vocab", "gap_summary": "gap",
              "unknown_tokens": ["x"]},
    })
    summary = scp.promote_once(
        registry_path=tmp_registry, jsonl_path=tmp_jsonl,
        db_path=tmp_db, judge=judge,
    )
    stats = summary["judge_stats"]
    assert stats.get("accept") == 1
    assert stats.get("reject") == 1
    assert stats.get("defer") == 1


# --- Judge crash safety --------------------------------------------------

def test_judge_crash_conservatively_defers(tmp_db, tmp_registry, tmp_jsonl):
    seeds = [{
        "channel": "arxiv", "concept": "crash me",
        "insight_text": "boom", "seed_hash": "g001",
    }]
    _write_jsonl(tmp_jsonl, seeds)

    def _bad_judge(seed, db_path):
        raise RuntimeError("kaboom")

    summary = scp.promote_once(
        registry_path=tmp_registry, jsonl_path=tmp_jsonl,
        db_path=tmp_db, judge=_bad_judge,
    )
    assert summary["promoted"] == 0
    assert summary["deferred_new"] == 1


# --- Real self_judge integration smoke test -----------------------------

def test_real_self_judge_end_to_end(tmp_db, tmp_registry, tmp_jsonl):
    """No mocked judge: verify the real self_judge glues in without
    exploding. With an empty vocabulary the verdict will be defer."""
    seeds = [{
        "channel": "arxiv", "concept": "novel concept",
        "insight_text": "novel", "seed_hash": "h001",
    }]
    _write_jsonl(tmp_jsonl, seeds)

    summary = scp.promote_once(
        registry_path=tmp_registry, jsonl_path=tmp_jsonl, db_path=tmp_db,
    )
    # Empty vocab -> low coverage -> defer.
    assert summary["deferred_new"] + summary["rejected"] >= 1
    assert summary["read"] == 1


# --- Malformed seed handling --------------------------------------------

def test_malformed_seed_recorded(tmp_db, tmp_registry, tmp_jsonl):
    seeds = [{"insight_text": "no channel or concept"}]
    _write_jsonl(tmp_jsonl, seeds)
    summary = scp.promote_once(
        registry_path=tmp_registry, jsonl_path=tmp_jsonl,
        db_path=tmp_db, judge=_make_judge({}),
    )
    assert summary["skipped_malformed"] == 1
    assert summary["judge_stats"].get("malformed") == 1


# --- Duplicate cap_id skipped ------------------------------------------

def test_duplicate_capability_id_skipped(tmp_db, tmp_registry, tmp_jsonl):
    seed = {"channel": "arxiv", "concept": "dup topic",
            "insight_text": "x", "seed_hash": "i001"}
    _write_jsonl(tmp_jsonl, [seed])
    scp.promote_once(
        registry_path=tmp_registry, jsonl_path=tmp_jsonl,
        db_path=tmp_db, judge=_make_judge({}),
    )
    # Second seed with the same concept but different hash - registry
    # already contains cap_id -> skipped_dupes.
    _write_jsonl(tmp_jsonl, [{**seed, "seed_hash": "i002"}])
    summary = scp.promote_once(
        registry_path=tmp_registry, jsonl_path=tmp_jsonl,
        db_path=tmp_db, judge=_make_judge({}),
    )
    assert summary["skipped_dupes"] == 1
    assert summary["promoted"] == 0
