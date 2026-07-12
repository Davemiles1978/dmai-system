"""Tests for the Fresh Blood Injector (PR E).

The injector emits exploratory insights into the same JSONL the
insight_promoter tails, so these tests exercise the full path with a
real SQLite state store and a real on-disk JSONL file. External HTTP
fetches are monkeypatched out.
"""
from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

import pytest

from components import fresh_blood_injector as fb


# --- Fixtures -------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """SQLite DB pre-populated with the schema fresh_blood touches."""
    db_path = tmp_path / "dmai_knowledge.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS system_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT,
            insight_text TEXT,
            confidence REAL DEFAULT 0.5,
            domain TEXT,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS capabilities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'function',
            capability_type TEXT NOT NULL DEFAULT 'general',
            description TEXT
        );
        """
    )
    # Seed a skewed capability distribution — matches prod today
    # (utility dominant at ~58%).
    rows = (
        [("u" + str(i), "u", "utility", "utility", None) for i in range(60)]
        + [("c" + str(i), "c", "configuration", "configuration", None) for i in range(15)]
        + [("d" + str(i), "d", "data_structure", "data_structure", None) for i in range(15)]
        + [("t" + str(i), "t", "trading", "trading", None) for i in range(5)]
        + [("b" + str(i), "b", "blockchain", "blockchain", None) for i in range(2)]
        + [("i" + str(i), "i", "interface", "interface", None) for i in range(1)]
    )
    conn.executemany(
        "INSERT INTO capabilities (id, name, type, capability_type, description) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def tmp_jsonl(tmp_path: Path) -> Path:
    return tmp_path / "research" / "insights.jsonl"


@pytest.fixture
def rng():
    return random.Random(42)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _sql_state(db_path: str, key: str) -> str | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# --- wildcard channel: deterministic, no HTTP ----------------------------

def test_wildcard_channel_emits_expected_rows(tmp_db, tmp_jsonl, rng):
    result = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=True,
        channels_override=["wildcard"], per_channel=3, rng=rng,
    )
    assert result["emitted"] == 3
    assert result["channels_used"] == ["wildcard"]

    rows = _read_jsonl(tmp_jsonl)
    assert len(rows) == 3
    for r in rows:
        assert r["source"] == "fresh_blood"
        assert r["domain"] == "fresh_blood/wildcard"
        assert r["provenance"] == "fresh_blood"
        assert r["channel"] == "wildcard"
        assert r["confidence"] == 0.4
        assert r["concept"] in fb.WILDCARD_VOCABULARY
        assert "seed_hash" in r and len(r["seed_hash"]) == 16
        assert "timestamp" in r


def test_wildcard_dedup_across_rounds(tmp_db, tmp_jsonl, rng):
    """The same wildcard term should never be emitted twice."""
    r1 = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=True,
        channels_override=["wildcard"], per_channel=5, rng=rng,
    )
    r2 = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=True,
        channels_override=["wildcard"], per_channel=5, rng=random.Random(99),
    )
    hashes1 = set(r1["seed_hashes"])
    hashes2 = set(r2["seed_hashes"])
    assert hashes1.isdisjoint(hashes2)


# --- crossover channel: uses SQL capabilities ----------------------------

def test_crossover_channel_uses_capability_types(tmp_db, tmp_jsonl, rng):
    result = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=True,
        channels_override=["crossover"], per_channel=3, rng=rng,
    )
    assert result["emitted"] == 3
    rows = _read_jsonl(tmp_jsonl)
    for r in rows:
        assert r["channel"] == "crossover"
        assert r["concept"].startswith("crossover:")
        # Both halves of the crossover pair should be real capability types.
        a_b = r["concept"].split(":", 1)[1].split("×")
        assert len(a_b) == 2
        assert all(len(x) > 0 for x in a_b)


# --- diversity channel: only fires above the threshold -------------------

def test_diversity_channel_fires_when_dominant(tmp_db, tmp_jsonl, rng):
    # utility is 60/98 = 61% > 40% threshold — diversity should fire.
    result = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=True,
        channels_override=["diversity"], per_channel=3, rng=rng,
    )
    assert result["emitted"] >= 1
    rows = _read_jsonl(tmp_jsonl)
    for r in rows:
        assert r["channel"] == "diversity"
        assert r["concept"].startswith("diversity_nudge:")


def test_diversity_channel_silent_when_balanced(tmp_path, tmp_jsonl, rng):
    """If no type dominates >40%, diversity should emit nothing."""
    db_path = tmp_path / "balanced.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS system_state (
            key TEXT PRIMARY KEY, value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT, insight_text TEXT, confidence REAL DEFAULT 0.5,
            domain TEXT, source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS capabilities (
            id TEXT PRIMARY KEY, name TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'function',
            capability_type TEXT NOT NULL DEFAULT 'general',
            description TEXT
        );
        """
    )
    # 4 types with 20 each -> 25% share each, well under threshold.
    for i in range(20):
        for t in ("utility", "trading", "blockchain", "identity"):
            conn.execute(
                "INSERT INTO capabilities (id, name, type, capability_type, description) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"{t}_{i}", t, t, t, None),
            )
    conn.commit()
    conn.close()

    result = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=str(db_path), force=True,
        channels_override=["diversity"], per_channel=3, rng=rng,
    )
    # No dominant type -> no nudges emitted, whole round skipped.
    assert result["emitted"] == 0
    assert result["skipped"] == 1


# --- arxiv channel: HTTP monkeypatched -----------------------------------

_ARXIV_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item>
  <title>Emergent Reasoning in Sparse Mixture Models (arXiv:2607.12345 v1)</title>
  <link>http://arxiv.org/abs/2607.12345</link>
</item>
<item>
  <title>Continual Learning Under Distribution Shift</title>
  <link>http://arxiv.org/abs/2607.99999</link>
</item>
<item>
  <title>Autonomous Agents that Repair Themselves</title>
  <link>http://arxiv.org/abs/2607.55555</link>
</item>
</channel></rss>"""


def test_arxiv_channel_parses_titles(monkeypatch, tmp_db, tmp_jsonl, rng):
    monkeypatch.setattr(fb, "_http_get", lambda url, timeout=10: _ARXIV_SAMPLE)
    result = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=True,
        channels_override=["arxiv"], per_channel=3, rng=rng,
    )
    assert result["emitted"] == 3
    rows = _read_jsonl(tmp_jsonl)
    concepts = {r["concept"] for r in rows}
    # The arXiv suffix should be stripped.
    assert "Emergent Reasoning in Sparse Mixture Models" in concepts
    assert "Continual Learning Under Distribution Shift" in concepts
    assert "Autonomous Agents that Repair Themselves" in concepts
    for r in rows:
        assert r["source_url"].startswith("http://arxiv.org/abs/")


def test_arxiv_http_failure_emits_nothing(monkeypatch, tmp_db, tmp_jsonl, rng):
    monkeypatch.setattr(fb, "_http_get", lambda url, timeout=10: None)
    result = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=True,
        channels_override=["arxiv"], per_channel=3, rng=rng,
    )
    assert result["emitted"] == 0
    assert result["skipped"] == 1
    assert _read_jsonl(tmp_jsonl) == []


# --- github channel: HTTP monkeypatched ----------------------------------

_GITHUB_SAMPLE = """
<html><body>
<article class="Box-row">
<h2 class="h3 lh-condensed">
  <a href="/octocat/hello-world">octocat / hello-world</a>
</h2>
</article>
<article class="Box-row">
<h2 class="h3 lh-condensed">
  <a href="/facebook/react">facebook / react</a>
</h2>
</article>
</body></html>
"""


def test_github_channel_parses_repos(monkeypatch, tmp_db, tmp_jsonl, rng):
    monkeypatch.setattr(fb, "_http_get", lambda url, timeout=10: _GITHUB_SAMPLE)
    result = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=True,
        channels_override=["github"], per_channel=3, rng=rng,
    )
    assert result["emitted"] == 2  # only 2 repos in the sample
    rows = _read_jsonl(tmp_jsonl)
    repos = {r["concept"] for r in rows}
    assert "github_trending:octocat/hello-world" in repos
    assert "github_trending:facebook/react" in repos


# --- cooldown ------------------------------------------------------------

def test_cooldown_blocks_immediate_reinjection(monkeypatch, tmp_db, tmp_jsonl, rng):
    r1 = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=True,
        channels_override=["wildcard"], per_channel=1, rng=rng,
    )
    assert r1["emitted"] == 1

    r2 = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=False,
        channels_override=["wildcard"], per_channel=1, rng=random.Random(1),
    )
    assert r2["emitted"] == 0
    assert r2.get("note") == "cooldown"


def test_force_overrides_cooldown(tmp_db, tmp_jsonl, rng):
    r1 = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=True,
        channels_override=["wildcard"], per_channel=1, rng=rng,
    )
    r2 = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=True,   # force
        channels_override=["wildcard"], per_channel=1, rng=random.Random(7),
    )
    assert r1["emitted"] == 1
    assert r2["emitted"] == 1


# --- state persistence ---------------------------------------------------

def test_last_run_ts_and_log_persisted(tmp_db, tmp_jsonl, rng):
    result = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=True,
        channels_override=["wildcard"], per_channel=2, rng=rng,
    )
    assert result["emitted"] == 2
    assert _sql_state(tmp_db, fb.LAST_RUN_KEY) is not None
    log_raw = _sql_state(tmp_db, fb.LOG_KEY)
    assert log_raw is not None
    log = json.loads(log_raw)
    assert len(log) == 2
    assert {e["channel"] for e in log} == {"wildcard"}


# --- channel picker weighting -------------------------------------------

def test_pick_channels_weighted_toward_unused(rng):
    """A channel used many times recently should be picked less."""
    log = [{"channel": "wildcard"}] * 40  # heavily used
    picks = [fb._pick_channels(log, 2, random.Random(i)) for i in range(200)]
    flat = [c for p in picks for c in p]
    # wildcard should be under-represented vs its 1/5 baseline
    wildcard_share = flat.count("wildcard") / len(flat)
    assert wildcard_share < 0.15  # inverse weighting should crush it


def test_pick_channels_returns_k_distinct(rng):
    picks = fb._pick_channels([], 3, rng)
    assert len(picks) == 3
    assert len(set(picks)) == 3


# --- diversity metric ---------------------------------------------------

def test_diversity_metric_matches_expected_entropy():
    # Perfectly uniform: entropy == log2(N), ratio == 1.0
    dist = [("a", 10), ("b", 10), ("c", 10), ("d", 10)]
    m = fb._diversity_metric(dist)
    assert m["ratio"] == pytest.approx(1.0, abs=1e-6)
    assert m["dominant_share"] == pytest.approx(0.25, abs=1e-6)

    # Single-type: entropy == 0, ratio == 0.0
    dist = [("x", 100), ("y", 0), ("z", 0)]
    m = fb._diversity_metric(dist)
    assert m["entropy"] == pytest.approx(0.0, abs=1e-6)
    assert m["dominant_share"] == 1.0

    # Empty
    m = fb._diversity_metric([])
    assert m["entropy"] == 0.0
    assert m["dominant"] is None


# --- integration: multi-channel round with mocked HTTP ------------------

def test_multi_channel_round(monkeypatch, tmp_db, tmp_jsonl, rng):
    monkeypatch.setattr(fb, "_http_get",
                        lambda url, timeout=10: _ARXIV_SAMPLE
                                                if "arxiv" in url else _GITHUB_SAMPLE)
    result = fb.inject_once(
        jsonl_path=tmp_jsonl, db_path=tmp_db, force=True,
        channels_override=["arxiv", "wildcard"], per_channel=2, rng=rng,
    )
    assert result["emitted"] == 4  # 2 arxiv + 2 wildcard
    assert set(result["channels_used"]) == {"arxiv", "wildcard"}
    rows = _read_jsonl(tmp_jsonl)
    assert len({r["seed_hash"] for r in rows}) == 4  # all distinct
