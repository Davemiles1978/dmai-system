"""Tests for components/self_judge.py.

self_judge is DMAI's own judgement primitive. It reads from
``vocabulary`` and ``insights`` in the knowledge DB and never makes
external calls. Tests fix a small vocabulary + insight neighbourhood
per case and assert the verdict + gap description.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

import pytest

from components import self_judge as sj


# --- Fixtures -------------------------------------------------------------

@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    db = tmp_path / "kdb.sqlite"
    c = sqlite3.connect(str(db))
    c.executescript(
        """
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
    c.commit()
    return c


def _seed_vocab(conn: sqlite3.Connection, words: List[str]) -> None:
    conn.executemany(
        "INSERT INTO vocabulary (id, word, definition) VALUES (?, ?, ?)",
        [(w, w.lower(), f"def of {w}") for w in words],
    )
    conn.commit()


def _seed_insight(conn: sqlite3.Connection, id_: str, text: str,
                  source_topic: str = "test") -> None:
    conn.execute(
        "INSERT INTO insights (id, insight_text, source_topic) VALUES (?, ?, ?)",
        (id_, text, source_topic),
    )
    conn.commit()


# --- Tokeniser ------------------------------------------------------------

def test_tokenise_lowercases_and_filters_stopwords():
    toks = sj._tokenise("The Quick Brown Fox Jumps")
    assert "fox" in toks and "jumps" in toks
    assert "the" not in toks   # stopword


def test_tokenise_ignores_short_tokens():
    toks = sj._tokenise("a I go on it up")
    assert toks == []


# --- Vocab coverage -------------------------------------------------------

def test_vocab_coverage_all_known(conn):
    _seed_vocab(conn, ["quantum", "entanglement", "topology"])
    cov, unknown = sj._vocab_coverage(
        conn, ["quantum", "entanglement", "topology"],
    )
    assert cov == pytest.approx(1.0)
    assert unknown == []


def test_vocab_coverage_partial(conn):
    _seed_vocab(conn, ["quantum"])
    cov, unknown = sj._vocab_coverage(conn, ["quantum", "spooky", "action"])
    assert cov == pytest.approx(1 / 3)
    assert set(unknown) == {"spooky", "action"}


def test_vocab_coverage_empty_conn():
    cov, unknown = sj._vocab_coverage(None, ["quantum"])
    assert cov == 1.0        # neutral fallback
    assert unknown == []


# --- Insight neighbourhood ------------------------------------------------

def test_insight_neighbourhood_duplicate_detected(conn):
    _seed_vocab(conn, ["retrieval", "augmented", "generation", "pipeline"])
    _seed_insight(
        conn, "i1",
        "retrieval augmented generation pipeline for enterprise search",
    )
    overlap, nid = sj._insight_neighbourhood(
        conn, ["retrieval", "augmented", "generation", "pipeline"],
    )
    assert overlap >= 0.9
    assert nid == "i1"


def test_insight_neighbourhood_no_matches(conn):
    overlap, nid = sj._insight_neighbourhood(
        conn, ["completely", "novel", "concept"],
    )
    assert overlap == 0.0
    assert nid is None


# --- KPI linkage ----------------------------------------------------------

def test_kpi_linkage_detects_keyword():
    # Any keyword in KPI_KEYWORDS should trigger; grab one directly.
    kw = next(iter(sj.KPI_KEYWORDS))
    assert sj._kpi_linkage([kw]) == 1.0


def test_kpi_linkage_absent():
    assert sj._kpi_linkage(["banana", "walrus"]) == 0.0


# --- Diversity pressure ---------------------------------------------------

def test_diversity_pressure_favours_underrepresented():
    dist = [("utility", 60), ("configuration", 15), ("frontier", 1)]
    hi = sj._diversity_pressure({"channel": "wildcard",
                                 "concept": "frontier concept"}, dist)
    lo = sj._diversity_pressure({"channel": "arxiv",
                                 "concept": "utility concept"}, dist)
    assert hi > lo


def test_diversity_pressure_missing_dist_returns_neutral():
    p = sj._diversity_pressure({"channel": "arxiv", "concept": "x"}, None)
    assert 0.4 <= p <= 0.6


# --- End-to-end verdicts --------------------------------------------------

def test_judge_accept_high_confidence(conn):
    # All words known + KPI hit + crossover channel gives diversity boost.
    kpi_kw = next(iter(sj.KPI_KEYWORDS))
    _seed_vocab(conn, [kpi_kw, "frontier", "integration", "composite"])
    v = sj.judge_seed(
        {"concept": f"crossover:{kpi_kw}\u00d7frontier",
         "insight_text": f"crossover of {kpi_kw} and frontier integration",
         "channel": "crossover"},
        conn,
        cap_type_dist=[("utility", 60), ("configuration", 15),
                       ("frontier", 5)],
    )
    assert v.verdict == "accept", (
        f"expected accept, got {v.verdict!r}: {v.reason} "
        f"conf={v.confidence:.3f} signals={v.signals.as_dict()}"
    )


def test_judge_defer_on_low_vocab(conn):
    # Vocabulary is empty → coverage is 0, well below the floor.
    v = sj.judge_seed(
        {"concept": "quaternion holography entropy solitons",
         "insight_text": "quaternion holography entropy solitons",
         "channel": "arxiv"},
        conn,
    )
    assert v.verdict == "defer"
    assert v.signals.unknown_tokens  # DMAI can report *what* she doesn't know
    assert v.knowledge_gap and "unknown" in v.knowledge_gap.lower()


def test_judge_reject_on_near_duplicate(conn):
    _seed_vocab(conn, [
        "retrieval", "augmented", "generation", "pipeline", "enterprise",
    ])
    _seed_insight(
        conn, "i1",
        "retrieval augmented generation pipeline enterprise",
    )
    v = sj.judge_seed(
        {"concept": "retrieval augmented generation pipeline enterprise",
         "insight_text": "retrieval augmented generation pipeline enterprise",
         "channel": "arxiv"},
        conn,
    )
    assert v.verdict == "reject"
    assert "near-duplicate" in v.reason or "duplicate" in v.reason.lower()


def test_judge_missing_concept_rejected():
    v = sj.judge_seed({"insight_text": "no concept here"}, None)
    assert v.verdict == "reject"


def test_judge_never_makes_external_calls(monkeypatch, conn):
    """self_judge must never call urllib/requests/openai/etc. This is
    the core promise DMAI relies on \u2014 judgement is her own."""
    import urllib.request
    called = {"n": 0}

    def _fail(*a, **k):
        called["n"] += 1
        raise RuntimeError("self_judge should NEVER make HTTP calls")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    _seed_vocab(conn, ["capabilities", "kpi"])
    v = sj.judge_seed(
        {"concept": "capabilities kpi diversity throughput",
         "insight_text": "explore capabilities kpi"},
        conn,
    )
    assert v.verdict in {"accept", "reject", "defer"}
    assert called["n"] == 0
