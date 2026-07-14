"""
knowledge_acquirer.py
=====================

DMAI's mechanism for filling her own knowledge gaps. When
``self_judge.judge_seed`` returns ``verdict="defer"`` because DMAI does
not have enough knowledge to reason about a seed, this module goes and
gets that knowledge \u2014 then commits it to her memory so the next
judgement pass can use it.

Design contract
---------------

**External LLM is a fallback, never a primary.** The acquirer follows
a three-step cascade for every gap:

    1.  DMAI's internal knowledge graph  (KnowledgeGraph.query_knowledge)
    2.  Public web search                (DuckDuckGo instant answer API)
    3.  External LLM via OpenRouter      (last resort only)

Each step is optional: missing modules, empty results, or transport
failures fall through to the next step. The output is normalised into
a ``KnowledgeParcel`` and committed to four persistent stores:

* ``vocabulary``            \u2014 one row per new token, with definition
* ``insights``              \u2014 one structured row per resolved gap
* knowledge graph JSON      \u2014 concept node + relations to keywords
* ``learning_progress``     \u2014 one row per resolved gap for auditability

Idempotence
-----------

Every write path is idempotent:

* ``vocabulary``:      ``INSERT OR IGNORE`` keyed on ``word``.
* ``insights``:        ``INSERT OR IGNORE`` keyed on a deterministic
                       hash of (source_topic, insight_text).
* knowledge graph:     ``KnowledgeGraph.add_concept`` upserts by name.
* ``learning_progress``: ``INSERT OR IGNORE`` keyed on
                       (gap_signature, resolved_source).

That means safely re-running ``acquire_and_commit`` on the same gap
does no harm \u2014 useful when a scheduled cron replays a defer queue.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

from components.db import safe_open_kdb

logger = logging.getLogger(__name__)


# ── Tunables ──────────────────────────────────────────────────────────────

HTTP_TIMEOUT_SECONDS   = 15
MAX_DEFINITION_CHARS   = 800
MAX_INSIGHT_CHARS      = 2000
MAX_VOCAB_TOKENS       = 25   # cap per acquire call

# System-prompt style hint used when the acquirer has to call an LLM.
LLM_SYSTEM_HINT = (
    "You are helping DMAI, an autonomous system, learn a new concept. "
    "Return a JSON object with keys: "
    "'definition' (<= 500 chars, plain English), "
    "'why_useful' (<= 300 chars, why an autonomous AI system might care), "
    "'related_kpis' (list of KPI keywords it could plausibly move), "
    "'related_concepts' (list of at most 5 related terms). "
    "Return only JSON."
)


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class KnowledgeParcel:
    """Everything the acquirer produced for a single gap."""
    concept:          str
    definition:       str
    why_useful:       str
    related_kpis:     List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    source:           str = "unknown"   # "knowledge_graph" | "web" | "llm"
    new_tokens:       List[Tuple[str, str]] = field(default_factory=list)
                                        # (word, short_definition)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "concept":          self.concept,
            "definition":       self.definition,
            "why_useful":       self.why_useful,
            "related_kpis":     list(self.related_kpis),
            "related_concepts": list(self.related_concepts),
            "source":           self.source,
            "new_tokens":       [{"word": w, "definition": d} for w, d in self.new_tokens],
        }


# ── Cascade step 1: internal knowledge graph ─────────────────────────────

def _try_knowledge_graph(kg, concept: str) -> Optional[KnowledgeParcel]:
    """Ask DMAI's own knowledge graph. Returns ``None`` if the graph is
    missing, empty, or has no useful hit for the concept."""
    if kg is None:
        return None

    query_fn: Optional[Callable[[str], Any]] = getattr(kg, "query_knowledge", None)
    if not callable(query_fn):
        return None

    try:
        hits = query_fn(concept) or []
    except Exception as e:
        logger.warning("knowledge_acquirer: KG query failed: %s", e)
        return None

    if not hits:
        return None

    # Coerce whatever shape the graph returned into a parcel.
    first = hits[0]
    if isinstance(first, dict):
        definition = str(first.get("definition") or first.get("text") or "")[:MAX_DEFINITION_CHARS]
        related    = first.get("related") or []
        if not definition:
            return None
        return KnowledgeParcel(
            concept=concept,
            definition=definition,
            why_useful=(
                f"'{concept}' is already known to my knowledge graph as a "
                f"related concept; reinforcing the link."
            ),
            related_kpis=[],
            related_concepts=[str(r)[:100] for r in related][:5],
            source="knowledge_graph",
        )
    return None


# ── Cascade step 2: web search ───────────────────────────────────────────

def _try_web_search(concept: str) -> Optional[KnowledgeParcel]:
    """DuckDuckGo instant-answer API. Free, keyless, best-effort.

    Returns ``None`` on any transport / parse failure so the cascade
    falls through to the LLM.
    """
    try:
        import urllib.request
    except ImportError:
        return None

    url = (
        "https://api.duckduckgo.com/?q=" + quote_plus(concept) +
        "&format=json&no_html=1&skip_disambig=1"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dmai-knowledge-acquirer"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        logger.debug("knowledge_acquirer: web search failed: %s", e)
        return None

    definition = (
        data.get("AbstractText")
        or data.get("Abstract")
        or (data.get("RelatedTopics", [{}])[0].get("Text", "") if data.get("RelatedTopics") else "")
    )
    if not definition:
        return None

    definition = str(definition)[:MAX_DEFINITION_CHARS]
    related = [
        str(r.get("Text", "")).split(" - ", 1)[0]
        for r in (data.get("RelatedTopics") or [])[:5]
        if isinstance(r, dict) and r.get("Text")
    ]
    return KnowledgeParcel(
        concept=concept,
        definition=definition,
        why_useful=(
            f"Public reference material for '{concept}'. Consider whether "
            f"any of the related concepts touch DMAI's tracked metrics."
        ),
        related_kpis=[],
        related_concepts=[r[:100] for r in related if r][:5],
        source="web",
    )


# ── Cascade step 3: external LLM via OpenRouter ──────────────────────────

def _openrouter_key() -> Optional[str]:
    return os.environ.get("OPENROUTER_API_KEY") or None


DEFAULT_OPENROUTER_MODEL = os.environ.get(
    "DMAI_ACQUIRER_MODEL", "openai/gpt-4o-mini",
)


def _try_llm(concept: str, why: str) -> Optional[KnowledgeParcel]:
    """Last-resort acquirer. Only fires when knowledge graph and web
    search both returned nothing. Requires ``OPENROUTER_API_KEY`` in
    the environment; returns ``None`` otherwise so tests / dev
    environments without a key still work.
    """
    key = _openrouter_key()
    if not key:
        return None

    try:
        import urllib.request
    except ImportError:
        return None

    payload = {
        "model": DEFAULT_OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_HINT},
            {"role": "user",
             "content": (
                 f"Concept: {concept}\n"
                 f"Why DMAI wants to learn this: {why}\n"
                 f"Return the JSON object."
             )},
        ],
        "temperature": 0.2,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://dmai-web.onrender.com",
            "X-Title":       "DMAI knowledge_acquirer",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        outer = json.loads(raw)
        content = outer["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning("knowledge_acquirer: LLM call failed: %s", e)
        return None

    parsed = _parse_llm_json(content)
    if not parsed:
        return None

    return KnowledgeParcel(
        concept=concept,
        definition=str(parsed.get("definition", ""))[:MAX_DEFINITION_CHARS],
        why_useful=str(parsed.get("why_useful", ""))[:300],
        related_kpis=[str(k)[:50] for k in (parsed.get("related_kpis") or [])][:10],
        related_concepts=[str(c)[:100] for c in (parsed.get("related_concepts") or [])][:5],
        source="llm",
    )


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_llm_json(content: str) -> Optional[Dict[str, Any]]:
    """LLMs sometimes wrap JSON in prose or code fences. Extract the
    largest ``{...}`` block and try to parse it."""
    if not content:
        return None
    m = _JSON_BLOCK_RE.search(content)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ── Vocabulary token expansion ───────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")


def _expand_new_tokens(parcel: KnowledgeParcel, unknown: List[str]) -> None:
    """For each previously-unknown token that appears in the parcel's
    definition, attach a short excerpt of the sentence containing it as
    a mini-definition. This is what gets written to the vocabulary
    table.
    """
    if not unknown or not parcel.definition:
        return

    sentences = re.split(r"(?<=[.!?])\s+", parcel.definition)
    tokens_lc = {t.lower() for t in unknown}
    seen: set = set()
    out: List[Tuple[str, str]] = []

    for sent in sentences:
        sent_tokens = {t.lower() for t in _TOKEN_RE.findall(sent)}
        matched = sent_tokens & tokens_lc
        for tok in matched:
            if tok in seen:
                continue
            seen.add(tok)
            out.append((tok, sent.strip()[:300]))
            if len(out) >= MAX_VOCAB_TOKENS:
                break
        if len(out) >= MAX_VOCAB_TOKENS:
            break

    parcel.new_tokens = out


# ── Persistence: vocabulary / insights / graph / learning_progress ──────

def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create the tables the acquirer writes to if they're not already
    present. Idempotent \u2014 safe on every call."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS vocabulary ("
        " id TEXT PRIMARY KEY, word TEXT UNIQUE, part_of_speech TEXT,"
        " definition TEXT, etymology TEXT, domain TEXT, added_at TEXT)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vocab_word ON vocabulary(word)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS insights ("
        " id TEXT PRIMARY KEY, insight_text TEXT, entity_type TEXT,"
        " entities TEXT, relationship TEXT, confidence REAL,"
        " source_topic TEXT, target_topic TEXT, source TEXT,"
        " created_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS learning_progress ("
        " id TEXT PRIMARY KEY, gap_signature TEXT, concept TEXT,"
        " resolved_source TEXT, ts TEXT, UNIQUE(gap_signature, resolved_source))"
    )


def _gap_signature(concept: str, gap: str) -> str:
    """Deterministic hash of (concept, gap description). Lets
    ``learning_progress`` idempotently record the same gap only once."""
    h = hashlib.sha256()
    h.update((concept + "\n" + gap).encode("utf-8"))
    return h.hexdigest()[:16]


def _insight_id(concept: str, text: str) -> str:
    h = hashlib.sha256()
    h.update((concept + "\n" + text).encode("utf-8"))
    return "acq:" + h.hexdigest()[:24]


def _commit_parcel(conn: sqlite3.Connection,
                   parcel: KnowledgeParcel,
                   gap: str,
                   kg=None) -> Dict[str, int]:
    """Write the parcel to all four stores. Returns counts of new rows
    per store for the caller's summary."""
    _ensure_tables(conn)
    now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()

    # 1. Vocabulary.
    vocab_added = 0
    for word, definition in parcel.new_tokens:
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO vocabulary "
                "(id, word, part_of_speech, definition, etymology, domain, added_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), word.lower(), None,
                 definition[:MAX_DEFINITION_CHARS], None,
                 f"acquired:{parcel.source}", now_iso),
            )
            if cur.rowcount:
                vocab_added += 1
        except sqlite3.OperationalError as e:
            logger.warning("knowledge_acquirer: vocab insert failed: %s", e)
            break

    # 2. Insights.
    insight_text = _format_insight_text(parcel, gap)
    iid = _insight_id(parcel.concept, insight_text)
    insights_added = 0
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO insights "
            "(id, insight_text, entity_type, entities, relationship, "
            " confidence, source_topic, target_topic, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (iid, insight_text[:MAX_INSIGHT_CHARS], "acquired_concept",
             json.dumps([parcel.concept]),
             "learned_from", 0.6, parcel.concept,
             ", ".join(parcel.related_concepts)[:200] or None,
             f"knowledge_acquirer:{parcel.source}", now_iso),
        )
        if cur.rowcount:
            insights_added += 1
    except sqlite3.OperationalError as e:
        logger.warning("knowledge_acquirer: insight insert failed: %s", e)

    # 3. Learning progress.
    prog_added = 0
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO learning_progress "
            "(id, gap_signature, concept, resolved_source, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), _gap_signature(parcel.concept, gap),
             parcel.concept, parcel.source, now_iso),
        )
        if cur.rowcount:
            prog_added += 1
    except sqlite3.OperationalError as e:
        logger.warning("knowledge_acquirer: learning_progress insert failed: %s", e)

    conn.commit()

    # 4. Knowledge graph (best effort \u2014 no DB lock).
    graph_added = 0
    if kg is not None:
        add_concept = getattr(kg, "add_concept", None)
        add_rel     = getattr(kg, "add_relationship", None)
        if callable(add_concept):
            try:
                add_concept(
                    parcel.concept[:100],
                    "acquired",
                    {"definition": parcel.definition[:200],
                     "source": parcel.source, "acquired_at": now_iso},
                )
                graph_added += 1
                if callable(add_rel):
                    for rel in parcel.related_concepts[:5]:
                        try:
                            add_rel(parcel.concept[:100], rel[:100],
                                    "related_to", weight=0.5)
                        except Exception:
                            pass
            except Exception as e:
                logger.warning("knowledge_acquirer: KG write failed: %s", e)

    return {
        "vocab_added":    vocab_added,
        "insights_added": insights_added,
        "progress_added": prog_added,
        "graph_added":    graph_added,
    }


def _format_insight_text(parcel: KnowledgeParcel, gap: str) -> str:
    parts = [
        f"[acquired via {parcel.source}] '{parcel.concept}':",
        parcel.definition,
    ]
    if parcel.why_useful:
        parts.append(f"Why useful: {parcel.why_useful}")
    if parcel.related_kpis:
        parts.append("KPIs it may touch: " + ", ".join(parcel.related_kpis))
    if parcel.related_concepts:
        parts.append("Related: " + ", ".join(parcel.related_concepts))
    parts.append(f"Gap addressed: {gap[:200]}")
    return "\n".join(parts)


# ── Public entry point ───────────────────────────────────────────────────

def _kdb_path() -> str:
    data = os.environ.get("DATA_PATH", "data/").rstrip("/").rstrip("\\")
    return os.path.join(data, "dmai_knowledge.db")


def acquire_and_commit(concept: str,
                       gap: str,
                       *,
                       unknown_tokens: Optional[List[str]] = None,
                       kg=None,
                       db_path: Optional[str] = None,
                       ) -> Dict[str, Any]:
    """Resolve a knowledge gap and commit the result to DMAI's memory.

    Cascade: knowledge graph \u2192 web \u2192 external LLM. Whichever step
    returns a non-empty parcel wins; the rest are skipped.

    Returns a summary dict::

        {
          "concept":        str,
          "resolved":       bool,
          "source":         "knowledge_graph"|"web"|"llm"|None,
          "commit_counts":  {"vocab_added": int, "insights_added": int,
                             "progress_added": int, "graph_added": int},
          "parcel":         dict | None,   # for debugging / audit
          "ts":             str,
        }
    """
    unknown = list(unknown_tokens or [])
    dbp = db_path or _kdb_path()

    # ---- Cascade.
    parcel = _try_knowledge_graph(kg, concept)
    if parcel is None:
        parcel = _try_web_search(concept)
    if parcel is None:
        parcel = _try_llm(concept, gap)

    now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()

    if parcel is None:
        # Even a full miss is a learning event \u2014 record that we tried
        # so we don't retry the exact same gap immediately.
        try:
            conn = safe_open_kdb(dbp)
            try:
                _ensure_tables(conn)
                conn.execute(
                    "INSERT OR IGNORE INTO learning_progress "
                    "(id, gap_signature, concept, resolved_source, ts) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()),
                     _gap_signature(concept, gap),
                     concept, "unresolved", now_iso),
                )
                conn.commit()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            logger.warning("knowledge_acquirer: unresolved-log write failed: %s", e)

        return {
            "concept":       concept,
            "resolved":      False,
            "source":        None,
            "commit_counts": {"vocab_added": 0, "insights_added": 0,
                              "progress_added": 0, "graph_added": 0},
            "parcel":        None,
            "ts":            now_iso,
        }

    # ---- Expand vocabulary entries from the parcel's definition.
    _expand_new_tokens(parcel, unknown)

    # ---- Commit.
    try:
        conn = safe_open_kdb(dbp)
        try:
            counts = _commit_parcel(conn, parcel, gap, kg=kg)
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as e:
        logger.warning("knowledge_acquirer: commit failed: %s", e)
        counts = {"vocab_added": 0, "insights_added": 0,
                  "progress_added": 0, "graph_added": 0}

    logger.info(
        "knowledge_acquirer: resolved '%s' via %s "
        "(vocab+%d insights+%d progress+%d graph+%d)",
        concept, parcel.source,
        counts["vocab_added"], counts["insights_added"],
        counts["progress_added"], counts["graph_added"],
    )

    return {
        "concept":       concept,
        "resolved":      True,
        "source":        parcel.source,
        "commit_counts": counts,
        "parcel":        parcel.as_dict(),
        "ts":            now_iso,
    }
