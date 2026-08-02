#!/usr/bin/env python3
"""
Build DMAI's knowledge graph with trading foundation nodes and edges.
This ensures the knowledge is in her memory graph, not just in the insights table.
"""

import sqlite3
import json
from pathlib import Path

DB_PATH = Path("data/dmai_knowledge.db")

# ----------------------------------------------------------------------
# 1. Define key concepts and relationships
# ----------------------------------------------------------------------
CONCEPTS = [
    # Main concepts from the trading text
    ("TradingSystemFoundation", "trading_knowledge",
     "Multi-asset algorithmic trading foundation including Forex, stocks, and cryptocurrencies."),
    ("MultiAssetTradingEngine", "python_code",
     "Class that standardizes inputs across disparate asset classes and computes indicators like log returns, EMA, ATR."),
    ("ATR_Volatility_Sizing", "trading_rule",
     "Volatility-based position sizing using Average True Range to equalize risk across assets."),
    ("LogReturns_Standardization", "trading_rule",
     "Use log returns to standardize price movements across different asset classes."),
    ("ExecutionIsolation", "architecture",
     "Separate AI signal generation from hard-coded execution logic to protect capital."),
    ("Vectorbt", "library",
     "High-performance backtesting tool using NumPy and Numba for multi-asset strategy evaluation."),
    ("Freqtrade", "library",
     "Modular open-source crypto and multi-market algorithmic trading bot with configurable strategies."),
    ("Qlib", "library",
     "Microsoft's AI-oriented quantitative investment platform bridging ML with traditional market rules."),
    ("DockerizedBackend", "infrastructure",
     "Containerized Python backend for seamless migration between commercial cloud and private servers."),
]

# Relationships: (source, target, rel_type, weight)
RELATIONSHIPS = [
    ("TradingSystemFoundation", "MultiAssetTradingEngine", "uses", 1.0),
    ("TradingSystemFoundation", "ATR_Volatility_Sizing", "implements", 1.0),
    ("TradingSystemFoundation", "LogReturns_Standardization", "implements", 1.0),
    ("TradingSystemFoundation", "ExecutionIsolation", "follows", 1.0),
    ("MultiAssetTradingEngine", "ATR_Volatility_Sizing", "uses", 1.0),
    ("MultiAssetTradingEngine", "LogReturns_Standardization", "uses", 1.0),
    ("TradingSystemFoundation", "Vectorbt", "recommends", 0.9),
    ("TradingSystemFoundation", "Freqtrade", "recommends", 0.9),
    ("TradingSystemFoundation", "Qlib", "recommends", 0.9),
    ("MultiAssetTradingEngine", "Vectorbt", "can_backtest_with", 0.8),
    ("MultiAssetTradingEngine", "Freqtrade", "can_derive_from", 0.8),
    ("MultiAssetTradingEngine", "Qlib", "can_integrate_with", 0.8),
    ("TradingSystemFoundation", "DockerizedBackend", "requires", 0.9),
    ("DockerizedBackend", "MultiAssetTradingEngine", "hosts", 1.0),
]

# ----------------------------------------------------------------------
# 2. Connect to DB and insert nodes/edges
# ----------------------------------------------------------------------
def build_graph():
    if not DB_PATH.exists():
        print(f"❌ Database {DB_PATH} not found.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Ensure tables exist (they should, but just in case)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_graph_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT NOT NULL UNIQUE,
            entity_type TEXT NOT NULL,
            description TEXT,
            properties TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            relationship TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Insert concepts
    for concept, etype, desc in CONCEPTS:
        cursor.execute('''
            INSERT OR IGNORE INTO knowledge_graph_nodes (concept, entity_type, description, properties)
            VALUES (?, ?, ?, ?)
        ''', (concept, etype, desc, json.dumps({"source": "user_provided_trading_foundation"})))

    # Insert relationships
    for src, tgt, rel, weight in RELATIONSHIPS:
        cursor.execute('''
            INSERT OR IGNORE INTO knowledge_graph_edges (source, target, relationship, weight)
            VALUES (?, ?, ?, ?)
        ''', (src, tgt, rel, weight))

    conn.commit()
    conn.close()
    print(f"✅ Added {len(CONCEPTS)} concepts and {len(RELATIONSHIPS)} relationships to DMAI's knowledge graph.")

# ----------------------------------------------------------------------
# 3. Also add the repository URLs as concepts with links
# ----------------------------------------------------------------------
def add_repo_concepts():
    repos = [
        ("Vectorbt", "https://github.com/vectorbt/vectorbt", "High-performance backtesting tool."),
        ("Freqtrade", "https://github.com/freqtrade/freqtrade", "Modular crypto and multi-market bot."),
        ("Qlib", "https://github.com/microsoft/qlib", "AI-oriented quantitative investment platform."),
    ]
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    for name, url, desc in repos:
        cursor.execute('''
            INSERT OR IGNORE INTO knowledge_graph_nodes (concept, entity_type, description, properties)
            VALUES (?, ?, ?, ?)
        ''', (name, "repository", desc, json.dumps({"url": url, "source": "user"})))

        # Link each repo to the foundation
        cursor.execute('''
            INSERT OR IGNORE INTO knowledge_graph_edges (source, target, relationship, weight)
            VALUES (?, ?, ?, ?)
        ''', ("TradingSystemFoundation", name, "recommends", 0.9))

    conn.commit()
    conn.close()
    print("✅ Added repository concepts and links.")

# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    build_graph()
    add_repo_concepts()
    print("\n🎯 DMAI's knowledge graph now contains trading foundation nodes and edges.")
    print("   She can query these concepts and relationships via her memory graph.")
