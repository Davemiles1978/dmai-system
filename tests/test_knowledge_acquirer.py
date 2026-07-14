"""Tests for components/knowledge_acquirer.py.

The acquirer is a cascade: internal knowledge graph -> web -> LLM.
Every step is monkeypatched so the tests run offline.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pytest

from components import knowledge_acquirer as ka


# --- Fixtures -------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "kdb.sqlite")


@pytest.fixture
def fake_kg():
    class _KG:
        def __init__(self, hits=None):
            self._hits = hits or []
            self.added_concepts: List[Dict[str, Any]] = []
            self.added_relations: List[Dict[str, Any]] = []

        def query_knowledge(self, concept):
            return self._hits

        def add_concept(self, name, type_, metadata):
            self.added_concepts.append(
                {"name": name, "type": type_, "metadata": metadata},
            )

        def add_relationship(self, a, b, rel_type, weight=0.5):
            self.added_relations.append(
                {"a": a, "b": b, "rel": rel_type, "w": weight},
            )
    return _KG


def _parcel(concept, definition, source, related_concepts=None,
            related_kpis=None, why_useful="because"):
    return ka.KnowledgeParcel(
        concept=concept,
        definition=definition,
        why_useful=why_useful,
        related_concepts=list(related_concepts or []),
        related_kpis=list(related_kpis or []),
        source=source,
    )


# --- Cascade order --------------------------------------------------------

def test_cascade_prefers_knowledge_graph(db_path, fake_kg, monkeypatch):
    kg = fake_kg(hits=[
        {"definition": "known in the graph", "related": ["neighbour"]},
    ])
    monkeypatch.setattr(
        ka, "_try_web_search",
        lambda concept: pytest.fail("web must not be called when KG hit"),
    )
    result = ka.acquire_and_commit(
        "quantum entanglement", "gap: unknown",
        kg=kg, db_path=db_path,
    )
    assert result["resolved"] is True
    assert result["source"] == "knowledge_graph"


def test_cascade_falls_through_to_web(db_path, fake_kg, monkeypatch):
    kg = fake_kg(hits=[])
    monkeypatch.setattr(
        ka, "_try_web_search",
        lambda concept: _parcel(concept, "from the web", "web"),
    )
    monkeypatch.setattr(
        ka, "_try_llm",
        lambda concept, why: pytest.fail("LLM must not run after web hit"),
    )
    result = ka.acquire_and_commit(
        "quantum entanglement", "gap: unknown",
        kg=kg, db_path=db_path,
    )
    assert result["source"] == "web"


def test_cascade_falls_through_to_llm(db_path, fake_kg, monkeypatch):
    kg = fake_kg(hits=[])
    monkeypatch.setattr(ka, "_try_web_search", lambda concept: None)
    monkeypatch.setattr(
        ka, "_try_llm",
        lambda concept, why: _parcel(
            concept, "from the LLM", "llm",
            related_concepts=["related_thing"],
            related_kpis=["capabilities"],
        ),
    )
    result = ka.acquire_and_commit(
        "nonexistent thing", "gap: unknown",
        kg=kg, db_path=db_path,
    )
    assert result["source"] == "llm"


def test_cascade_unresolved_when_all_fail(db_path, fake_kg, monkeypatch):
    kg = fake_kg(hits=[])
    monkeypatch.setattr(ka, "_try_web_search", lambda concept: None)
    monkeypatch.setattr(ka, "_try_llm", lambda concept, why: None)
    result = ka.acquire_and_commit(
        "utterly unknown", "gap: unknown",
        kg=kg, db_path=db_path,
    )
    assert result["resolved"] is False
    assert result["source"] is None

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT resolved_source FROM learning_progress "
        "WHERE concept = ?", ("utterly unknown",),
    ).fetchone()
    conn.close()
    assert row and row[0] == "unresolved"


# --- Commits -------------------------------------------------------------

def test_commits_land_in_all_stores(db_path, fake_kg, monkeypatch):
    kg = fake_kg(hits=[])
    monkeypatch.setattr(
        ka, "_try_web_search",
        lambda concept: _parcel(
            concept,
            "A retrieval augmented generation pipeline stores embeddings "
            "and returns nearest neighbours for enterprise search.",
            "web",
            related_concepts=["embedding", "vector database"],
        ),
    )
    result = ka.acquire_and_commit(
        "retrieval augmented generation",
        "gap: unknown tokens",
        unknown_tokens=["retrieval", "augmented", "embeddings"],
        kg=kg, db_path=db_path,
    )
    counts = result["commit_counts"]
    assert counts["vocab_added"] >= 1
    assert counts["insights_added"] == 1
    assert counts["progress_added"] == 1
    assert counts["graph_added"] == 1
    assert kg.added_concepts


def test_idempotence_second_call_no_double_write(db_path, fake_kg, monkeypatch):
    kg = fake_kg(hits=[])
    monkeypatch.setattr(
        ka, "_try_web_search",
        lambda concept: _parcel(concept, "a definition", "web"),
    )
    r1 = ka.acquire_and_commit(
        "idempotent concept", "same gap",
        unknown_tokens=["idempotent"], kg=kg, db_path=db_path,
    )
    r2 = ka.acquire_and_commit(
        "idempotent concept", "same gap",
        unknown_tokens=["idempotent"], kg=kg, db_path=db_path,
    )
    assert r1["commit_counts"]["insights_added"] == 1
    assert r2["commit_counts"]["insights_added"] == 0
    assert r2["commit_counts"]["progress_added"] == 0


# --- Vocabulary expansion -------------------------------------------------

def test_vocabulary_expansion_matches_unknown_tokens(db_path, fake_kg, monkeypatch):
    kg = fake_kg(hits=[])
    monkeypatch.setattr(
        ka, "_try_web_search",
        lambda concept: _parcel(
            concept,
            "Entanglement links quantum particles across arbitrary "
            "distances. Solitons preserve their shape while moving.",
            "web",
        ),
    )
    ka.acquire_and_commit(
        "quantum concept", "gap: unknown tokens",
        unknown_tokens=["entanglement", "solitons"],
        kg=kg, db_path=db_path,
    )
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT word FROM vocabulary ORDER BY word"
    ).fetchall()
    conn.close()
    words = {r[0] for r in rows}
    assert "entanglement" in words
    assert "solitons" in words


# --- Helpers -------------------------------------------------------------

def test_parse_llm_json_extracts_object():
    obj = ka._parse_llm_json(
        'blah blah {"definition":"x","why_useful":"y"} suffix'
    )
    assert obj == {"definition": "x", "why_useful": "y"}


def test_parse_llm_json_returns_none_on_garbage():
    assert ka._parse_llm_json("no json here") is None
    assert ka._parse_llm_json("") is None


# --- Web/LLM side-effect safety ------------------------------------------

def test_llm_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert ka._try_llm("anything", "any gap") is None


def test_web_returns_none_on_http_failure(monkeypatch):
    import urllib.request

    def _fail(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    assert ka._try_web_search("anything") is None
