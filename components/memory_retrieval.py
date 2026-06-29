"""
components/memory_retrieval.py
──────────────────────────────────────────────────────────────────────────────
DMAI Memory Retrieval — query internal knowledge BEFORE going external.

Key rule: DMAI must check her own knowledge base first.
    1. Search insights.jsonl (recent short-term memory)
    2. Search dmai_knowledge.db insights + syllabus tables (long-term memory)
    3. Search compiled_knowledge JSON files (domain expertise)
    4. Return a MemoryResult with confidence + sources

Used by: AutonomousResearcher, KPIEvaluator, CodeWriter, KaizenAutoRepair
"""

import json
import sqlite3
import logging
from components.db import safe_open_kdb
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from difflib import SequenceMatcher

logger = logging.getLogger("dmai.memory")

_DATA_PATH    = Path("data")
_INSIGHTS_FILE = _DATA_PATH / "research" / "insights.jsonl"
_KNOWLEDGE_DB  = _DATA_PATH / "dmai_knowledge.db"
_COMPILED_DIR  = _DATA_PATH / "learning" / "compiled_knowledge"
_MASTER_KNOW   = _COMPILED_DIR / "master_knowledge.json"

# Confidence threshold to consider a memory "sufficient" (skip external search)
MEMORY_HIT_THRESHOLD = 0.55


class MemoryResult:
    """Returned by every recall() call."""
    def __init__(self, query: str, hits: List[Dict], confidence: float, source: str):
        self.query      = query
        self.hits       = hits            # list of matching records
        self.confidence = confidence      # 0.0–1.0
        self.source     = source          # "insights_jsonl", "knowledge_db", "compiled", "none"
        self.sufficient = confidence >= MEMORY_HIT_THRESHOLD

    def best_text(self) -> str:
        """Return the most relevant text snippet from hits."""
        if not self.hits:
            return ""
        top = self.hits[0]
        return (top.get("concept") or top.get("insight_text") or
                top.get("summary") or top.get("content") or "")

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "confidence": self.confidence,
            "source": self.source,
            "sufficient": self.sufficient,
            "hits": self.hits[:5],
        }


def _fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _keyword_score(text: str, query: str) -> float:
    """Return fraction of query words found in text."""
    if not text:
        return 0.0
    words = [w for w in query.lower().split() if len(w) > 2]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in text.lower())
    return hits / len(words)


# ─────────────────────────────────────────────────────────────────────────────
# Primary recall function
# ─────────────────────────────────────────────────────────────────────────────

def recall(query: str, top_k: int = 5, min_confidence: float = 0.0) -> MemoryResult:
    """
    DMAI's internal memory query. Searches in order:
        1. insights.jsonl (most recent, highest priority)
        2. dmai_knowledge.db (bulk historical)
        3. compiled_knowledge JSON files

    Returns a MemoryResult. If sufficient=True, the caller should use this
    result and skip any external API call.
    """
    query = query.strip()
    if not query:
        return MemoryResult(query, [], 0.0, "none")

    # ── 1. insights.jsonl ────────────────────────────────────────────────────
    result = _search_insights_jsonl(query, top_k)
    if result.sufficient:
        logger.debug("memory recall HIT (insights.jsonl, conf=%.2f): %s", result.confidence, query[:60])
        return result

    # ── 2. dmai_knowledge.db ─────────────────────────────────────────────────
    db_result = _search_knowledge_db(query, top_k)
    if db_result.confidence > result.confidence:
        result = db_result
    if result.sufficient:
        logger.debug("memory recall HIT (knowledge_db, conf=%.2f): %s", result.confidence, query[:60])
        return result

    # ── 3. compiled_knowledge JSON files ─────────────────────────────────────
    compiled_result = _search_compiled_knowledge(query, top_k)
    if compiled_result.confidence > result.confidence:
        result = compiled_result

    if result.confidence > 0:
        logger.debug("memory recall PARTIAL (conf=%.2f, source=%s): %s",
                     result.confidence, result.source, query[:60])
    else:
        logger.debug("memory recall MISS: %s", query[:60])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Search backends
# ─────────────────────────────────────────────────────────────────────────────

def _search_insights_jsonl(query: str, top_k: int) -> MemoryResult:
    if not _INSIGHTS_FILE.exists():
        return MemoryResult(query, [], 0.0, "insights_jsonl")

    try:
        lines = _INSIGHTS_FILE.read_text().strip().splitlines()
    except Exception as e:
        logger.warning("Could not read insights.jsonl: %s", e)
        return MemoryResult(query, [], 0.0, "insights_jsonl")

    scored = []
    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        text = f"{rec.get('domain','')} {rec.get('concept','')} {rec.get('source','')}"
        score = _keyword_score(text, query)
        if score > 0:
            scored.append((score, rec))

    if not scored:
        return MemoryResult(query, [], 0.0, "insights_jsonl")

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    best_score = top[0][0]
    # Confidence: keyword coverage * recency bonus (recent = higher)
    confidence = min(best_score * 1.1, 1.0) if best_score >= 0.4 else best_score
    return MemoryResult(query, [r for _, r in top], confidence, "insights_jsonl")


_KNOWLEDGE_DB_BROKEN_UNTIL = 0  # epoch seconds; suppress retries while broken

def _search_knowledge_db(query: str, top_k: int) -> MemoryResult:
    import time as _t
    global _KNOWLEDGE_DB_BROKEN_UNTIL
    if not _KNOWLEDGE_DB.exists():
        return MemoryResult(query, [], 0.0, "knowledge_db")
    if _t.time() < _KNOWLEDGE_DB_BROKEN_UNTIL:
        return MemoryResult(query, [], 0.0, "knowledge_db")

    hits = []
    try:
        conn = safe_open_kdb(str(_KNOWLEDGE_DB), read_only=True)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=3000")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Search insights table
        query_words = [f"%{w}%" for w in query.lower().split() if len(w) > 2]
        if query_words:
            like_clause = " OR ".join(["LOWER(insight_text) LIKE ?" for _ in query_words])
            c.execute(f"""
                SELECT insight_text, confidence, source_topic, source_url, created_at
                FROM insights
                WHERE {like_clause}
                ORDER BY occurrence_count DESC, created_at DESC
                LIMIT ?
            """, query_words + [top_k * 2])
            for row in c.fetchall():
                hits.append({
                    "insight_text": row["insight_text"],
                    "confidence": row["confidence"],
                    "source_topic": row["source_topic"],
                    "source_url": row["source_url"],
                })

        # Search syllabus_content
        c.execute("""
            SELECT topic, name, content, mastery
            FROM syllabus_content
            WHERE LOWER(name) LIKE ? OR LOWER(content) LIKE ?
            ORDER BY mastery DESC
            LIMIT ?
        """, (f"%{query.lower()}%", f"%{query.lower()}%", top_k))
        for row in c.fetchall():
            hits.append({
                "topic": row["topic"],
                "concept": row["name"],
                "content": row["content"],
                "mastery": row["mastery"],
            })

        conn.close()
    except Exception as e:
        msg = str(e).lower()
        # Back off for 60s on malformed/locked to stop log spam
        if "malformed" in msg or "locked" in msg or "not a database" in msg:
            _KNOWLEDGE_DB_BROKEN_UNTIL = _t.time() + 60
            logger.warning("DB unhealthy (%s) — backing off 60s", e)
        else:
            logger.warning("DB search error: %s", e)
        return MemoryResult(query, [], 0.0, "knowledge_db")

    if not hits:
        return MemoryResult(query, [], 0.0, "knowledge_db")

    # Score by keyword coverage
    scored = []
    for h in hits:
        text = " ".join(str(v) for v in h.values() if v)
        score = _keyword_score(text, query)
        if score > 0:
            scored.append((score, h))

    if not scored:
        return MemoryResult(query, [], 0.0, "knowledge_db")

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    confidence = min(top[0][0] * 1.05, 1.0)
    return MemoryResult(query, [r for _, r in top], confidence, "knowledge_db")


def _search_compiled_knowledge(query: str, top_k: int) -> MemoryResult:
    if not _COMPILED_DIR.exists():
        return MemoryResult(query, [], 0.0, "compiled")

    hits = []
    for json_file in _COMPILED_DIR.glob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            # master_knowledge.json has a 'modules' dict
            if "modules" in data and isinstance(data["modules"], dict):
                for mod_name, mod_data in data["modules"].items():
                    text = f"{mod_name} {json.dumps(mod_data)[:500]}"
                    score = _keyword_score(text, query)
                    if score > 0.2:
                        hits.append((score, {
                            "module": mod_name,
                            "content": str(mod_data)[:300],
                            "source": json_file.name,
                        }))
            else:
                text = json.dumps(data)[:1000]
                score = _keyword_score(text, query)
                if score > 0.2:
                    hits.append((score, {
                        "content": str(data)[:300],
                        "source": json_file.name,
                    }))
        except Exception:
            continue

    if not hits:
        return MemoryResult(query, [], 0.0, "compiled")

    hits.sort(key=lambda x: x[0], reverse=True)
    top = hits[:top_k]
    return MemoryResult(query, [r for _, r in top], top[0][0] * 0.9, "compiled")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: add to SICore
# ─────────────────────────────────────────────────────────────────────────────

def recall_or_search(query: str, search_fn, top_k: int = 5) -> Tuple[bool, any]:
    """
    Call recall() first. If sufficient, return (True, MemoryResult).
    If not, call search_fn(query) and return (False, result).

    Usage:
        used_memory, result = recall_or_search(topic, lambda q: researcher.search_github(q))
    """
    mem = recall(query, top_k=top_k)
    if mem.sufficient:
        return True, mem
    external = search_fn(query)
    return False, external


# ─────────────────────────────────────────────────────────────────────────────
# Patch SICore at import time
# ─────────────────────────────────────────────────────────────────────────────

def patch_si_core(si_core) -> None:
    """
    Inject recall() as a method on an existing SICore instance.
    Call this after SICore is initialised.
    """
    si_core.recall = recall
    si_core.memory_recall = recall
    si_core.recall_or_search = recall_or_search
    logger.info("MemoryRetrieval patched into SICore")
