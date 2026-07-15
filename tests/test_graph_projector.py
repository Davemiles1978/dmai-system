"""Tests for components.graph_projector.

Builds a tiny DB matching the real capabilities + insights schema, runs
the projector, and asserts the tiered graph shape.
"""
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from components.graph_projector import GraphProjector


@pytest.fixture()
def seeded_env(tmp_path):
    db = tmp_path / "dmai_knowledge.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE capabilities (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,
            capability_type TEXT NOT NULL, description TEXT, source_url TEXT,
            source_repo TEXT, file_path TEXT, runtime_mode TEXT,
            language TEXT, methods TEXT, is_async INTEGER DEFAULT 0,
            args TEXT, integrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE insights (
            id TEXT PRIMARY KEY, insight_text TEXT NOT NULL,
            entity_type TEXT NOT NULL, entities TEXT NOT NULL,
            relationship TEXT NOT NULL, confidence REAL DEFAULT 0.5,
            source_topic TEXT NOT NULL, target_topic TEXT NOT NULL,
            source_url TEXT, source_title TEXT, source_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            occurrence_count INTEGER DEFAULT 1, last_used TIMESTAMP
        );
        """
    )
    caps = [
        ("c1", "SearchEngine", "class", "utility", "", "", "acme/tool", "a.py", "ondemand"),
        ("c2", "TradeExecutor", "class", "trading", "", "", "acme/tool", "b.py", "autonomous"),
        ("c3", "TokenSwap", "function", "trading", "", "", "coin/swap", "c.py", "ondemand"),
        ("c4", "MarketData", "class", "api", "", "", "coin/swap", "d.py", "ondemand"),
    ]
    for c in caps:
        conn.execute(
            "INSERT INTO capabilities "
            "(id,name,type,capability_type,description,source_url,source_repo,"
            "file_path,runtime_mode) VALUES (?,?,?,?,?,?,?,?,?)",
            c,
        )
    insights = [
        ("i1", "trading depends on market data", "dep", "[]", "depends_on", 0.9, "trading", "market_data"),
        ("i2", "tradeexecutor is a trading tool", "cat", "[]", "is_a", 0.8, "tradeexecutor", "trading"),
        ("i3", "market data feeds trading", "cause", "[]", "enables", 0.7, "market_data", "trading"),
        ("i4", "search engine finds data", "util", "[]", "finds", 0.6, "searchengine", "market_data"),
    ]
    for i in insights:
        conn.execute(
            "INSERT INTO insights "
            "(id,insight_text,entity_type,entities,relationship,confidence,"
            "source_topic,target_topic) VALUES (?,?,?,?,?,?,?,?)",
            i,
        )
    conn.commit()
    conn.close()

    arch = tmp_path / "graph_schema.json"
    arch.write_text(json.dumps({
        "neurons": [
            {"id": "trading_agent", "label": "Trading Agent",
             "cluster": "agents", "activation": 0.9},
            {"id": "api_gateway", "label": "API Gateway",
             "cluster": "core", "activation": 0.8},
            {"id": "background_svc", "label": "Background",
             "cluster": "core", "activation": 0.7},
        ],
        "synapses": [
            {"source": "api_gateway", "target": "trading_agent",
             "weight": 1.0, "type": "control"},
        ],
    }))
    return db, arch


def test_rebuild_produces_all_layers(seeded_env):
    db, arch = seeded_env
    gp = GraphProjector(db_path=db, arch_schema_path=arch)
    stats = gp.rebuild()
    assert stats["ok"] is True
    assert stats["arch_neurons"] == 3
    assert stats["capability_neurons"] == 4
    # utility, trading, api → 3 type-cluster neurons
    assert stats["type_neurons"] == 3
    # trading, market_data, tradeexecutor, searchengine → 4 topics
    assert stats["topic_neurons"] == 4
    assert stats["insight_synapses"] >= 4
    assert stats["arch_synapses"] == 1


def test_schema_read_returns_neurons_and_edges(seeded_env):
    db, arch = seeded_env
    gp = GraphProjector(db_path=db, arch_schema_path=arch)
    gp.rebuild()
    schema = gp.to_schema()
    assert schema["total_neurons"] == 14
    assert schema["total_synapses"] >= 10

    edges = {(s["source"], s["target"], s["type"])
             for s in schema["synapses"]}
    # Insight-derived topic↔topic edges must be present.
    assert ("topic:trading", "topic:market_data", "insight") in edges
    assert ("topic:market_data", "topic:trading", "insight") in edges


def test_capability_to_topic_link(seeded_env):
    db, arch = seeded_env
    gp = GraphProjector(db_path=db, arch_schema_path=arch)
    gp.rebuild()
    schema = gp.to_schema()
    edges = {(s["source"], s["target"], s["type"])
             for s in schema["synapses"]}
    # "TradeExecutor" (cap:c2) → topic:tradeexecutor via insight table.
    assert ("cap:c2", "topic:tradeexecutor", "cap_to_topic") in edges


def test_same_repo_edges_are_bounded(seeded_env):
    db, arch = seeded_env
    gp = GraphProjector(db_path=db, arch_schema_path=arch)
    stats = gp.rebuild()
    # Two repos, each with 2 capabilities → 2 same_repo edges total.
    assert stats["same_repo_synapses"] == 2


def test_stats_reports_layers(seeded_env):
    db, arch = seeded_env
    gp = GraphProjector(db_path=db, arch_schema_path=arch)
    gp.rebuild()
    st = gp.stats()
    assert st["ok"] is True
    assert st["built"] is True
    assert st["by_layer"]["architecture"] == 3
    assert st["by_layer"]["capability"] == 4
    assert st["by_layer"]["capability_type"] == 3
    assert st["by_layer"]["topic"] == 4


def test_rebuild_is_idempotent(seeded_env):
    db, arch = seeded_env
    gp = GraphProjector(db_path=db, arch_schema_path=arch)
    a = gp.rebuild()
    b = gp.rebuild()
    for k in ("total_neurons", "total_synapses", "arch_neurons",
              "capability_neurons", "topic_neurons"):
        assert a[k] == b[k], f"{k} differed between rebuilds"
