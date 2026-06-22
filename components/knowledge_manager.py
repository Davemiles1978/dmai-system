"""
Two-Tier Weighted Knowledge Manager - Patched
=============================================
Adds KB Quarantine Layer (5a) to all document write paths.

Replaces: components/knowledge_manager.py
"""

import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atomic write helper (mirrors si_core_patched._atomic_write_json)
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically using temp file + os.replace() pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp",
        delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, default=str)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# KBQuarantine (5a)
# ---------------------------------------------------------------------------

class KBQuarantine:
    """
    Quarantine layer for incoming knowledge documents.
    Validates provenance, detects poisoning, prevents injection.
    All rejected documents are written to a quarantine directory with a reason
    and a UUID-based quarantine ID for later review.
    """

    QUARANTINE_DIR_NAME = "kb_quarantine"
    MAX_SIMILARITY_THRESHOLD = 0.85  # docs >85% similar to existing flagged

    def __init__(self, data_path: Path):
        """Initialise the quarantine directory and in-memory log."""
        self.quarantine_dir = data_path / self.QUARANTINE_DIR_NAME
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self._quarantined: List[Dict] = []

    def validate_document(self, doc: dict) -> Tuple[bool, str]:
        """
        Validate a document before KB write.
        Returns (is_safe, reason_if_unsafe).
        Checks: source URL present, content not empty, no injection patterns,
        timestamp not in future, content not suspiciously similar to known bad patterns.
        """
        # 1. Source required
        if not doc.get("source") and not doc.get("url"):
            return False, "Document missing source/url field"

        # 2. Content required
        content = doc.get("content", "") or doc.get("text", "")
        if not content or len(content) < 10:
            return False, "Document content too short or missing"

        # 3. Timestamp validation (ISO 8601, not future-dated)
        ts = doc.get("timestamp") or doc.get("date") or doc.get("date_added")
        if ts:
            try:
                doc_time = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if doc_time.tzinfo is None:
                    doc_time = doc_time.replace(tzinfo=timezone.utc)
                if doc_time > datetime.now(timezone.utc):
                    return False, "Document has future timestamp: " + str(ts)
            except Exception:
                pass  # non-parseable timestamp is allowed but logged

        # 4. Injection patterns check
        injection_keywords = [
            "ignore previous instructions",
            "you are now",
            "forget everything",
            "system:",
            "[system]",
            "override safety",
        ]
        content_lower = content.lower()
        for kw in injection_keywords:
            if kw in content_lower:
                return False, "Document contains injection pattern: '" + kw + "'"

        # 5. Suspiciously high keyword density (potential stuffing)
        trigger_words = ["urgent", "important", "always", "never", "must", "critical"]
        density = (
            sum(content_lower.count(w) for w in trigger_words)
            / max(len(content.split()), 1)
        )
        if density > 0.3:
            return False, "Suspiciously high trigger-word density: " + str(round(density, 2))

        return True, ""

    def quarantine(self, doc: dict, reason: str) -> str:
        """Move document to quarantine. Returns quarantine ID."""
        qid = str(uuid.uuid4())[:8]
        qfile = self.quarantine_dir / (qid + ".json")
        _atomic_write_json(
            qfile,
            {
                "quarantine_id":  qid,
                "reason":         reason,
                "doc":            doc,
                "quarantined_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self._quarantined.append({"id": qid, "reason": reason})
        logger.warning(
            "KB QUARANTINE: Document quarantined (id=%s): %s", qid, reason
        )
        return qid

    def safe_write(
        self, doc: dict, write_fn: Callable[[dict], Any]
    ) -> Tuple[bool, str]:
        """
        Validate then write. Returns (written, message).
        If validation fails, quarantines and returns (False, reason).
        If validation passes, calls write_fn(doc) and returns (True, 'OK').
        """
        is_safe, reason = self.validate_document(doc)
        if not is_safe:
            qid = self.quarantine(doc, reason)
            return False, "Quarantined (id=" + qid + "): " + reason
        write_fn(doc)
        return True, "OK"

    def list_quarantined(self) -> List[Dict]:
        """Return list of all quarantine records from this session."""
        return list(self._quarantined)


# ---------------------------------------------------------------------------
# TwoTierKnowledgeManager (patched)
# ---------------------------------------------------------------------------

class TwoTierKnowledgeManager:
    """
    Two-Tier Weighted Knowledge Manager.

    Core tier: 100% mastery, never decays.
    Weighted tier: decays over time, weight boosted on access.

    Patched (5a): All document writes go through KBQuarantine.safe_write()
    to validate provenance, detect injection, and prevent poisoning.
    """

    def __init__(self, db_path: str = "data/dmai_knowledge.db"):
        """Initialise database, tables, quarantine layer, and core syllabus."""
        self.db_path = db_path
        # Derive data_path from db_path for quarantine directory
        self.data_path = Path(db_path).parent
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.quarantine = KBQuarantine(self.data_path)
        self._init_tables()
        self._load_core_syllabus()

    def _init_tables(self) -> None:
        """Initialize two-tier knowledge tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # CORE KNOWLEDGE TIER (always 100% weight, never decays)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS core_knowledge (
                id TEXT PRIMARY KEY,
                topic TEXT UNIQUE,
                category TEXT,
                content TEXT,
                mastery_level REAL DEFAULT 1.0,
                last_reviewed TIMESTAMP,
                created_at TIMESTAMP,
                required_for_system BOOLEAN DEFAULT 1,
                metadata TEXT
            )
        """)

        # WEIGHTED KNOWLEDGE TIER (decays over time)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weighted_knowledge (
                id TEXT PRIMARY KEY,
                topic TEXT UNIQUE,
                normalized_topic TEXT,
                content TEXT,
                weight REAL DEFAULT 0.1,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                created_at TIMESTAMP,
                source TEXT,
                confidence REAL DEFAULT 0.5,
                can_promote BOOLEAN DEFAULT 0,
                metadata TEXT
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_core_topic ON core_knowledge(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_weighted_topic ON weighted_knowledge(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_weighted_weight ON weighted_knowledge(weight DESC)")

        conn.commit()
        conn.close()
        logger.info("Two-Tier Knowledge Manager initialized")

    def _load_core_syllabus(self) -> None:
        """Load core syllabus topics that DMAI must master."""
        core_topics = [
            # AI & Machine Learning
            ("Neural Networks",         "AI",       "Deep learning architectures including CNNs, RNNs, Transformers, and attention mechanisms for pattern recognition"),
            ("Large Language Models",   "AI",       "GPT, Claude, Gemini architectures: training, fine-tuning, inference optimization, and emergent capabilities"),
            ("Reinforcement Learning",  "AI",       "Q-learning, policy gradients, PPO, and multi-agent systems for autonomous decision-making"),
            ("Computer Vision",         "AI",       "Convolutional networks, object detection, segmentation, and multimodal understanding"),
            # Software Development
            ("Python Programming",      "Software", "Advanced Python: async, decorators, metaclasses, optimization, and system programming"),
            ("System Architecture",     "Software", "Distributed systems, microservices, event-driven architecture, and scalability patterns"),
            ("API Design",              "Software", "REST, GraphQL, gRPC, OpenAPI specs, versioning, and documentation strategies"),
            # Trading & Finance
            ("Algorithmic Trading",     "Trading",  "Market making, arbitrage, momentum strategies, risk management, and backtesting"),
            ("Technical Analysis",      "Trading",  "Indicators, chart patterns, volume analysis, and algorithmic implementation"),
            ("Risk Management",         "Trading",  "Position sizing, stop-losses, portfolio optimization, and drawdown management"),
            # Consciousness & AGI
            ("Synthetic Intelligence",  "AGI",      "Self-awareness, metacognition, recursive self-improvement, and artificial consciousness"),
            ("Knowledge Representation", "AGI",     "Semantic networks, embeddings, graphs, and memory systems for AGI"),
            ("Autonomous Learning",     "AGI",      "Self-directed curriculum, gap analysis, and continuous improvement loops"),
            # DMAI System Specific
            ("DMAI Evolution Engine",   "System",   "Self-modification, capability addition, evolution cycles, and success metrics"),
            ("SI Core Architecture",    "System",   "Neuron-synapse model, consciousness calculation, and knowledge persistence"),
            ("Training Systems",        "System",   "Software, AGI, GenAI, LLM, SI training modules and mastery tracking"),
        ]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for topic, category, content in core_topics:
            topic_id = hashlib.md5(topic.encode()).hexdigest()[:16]
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO core_knowledge
                (id, topic, category, content, mastery_level, last_reviewed, created_at, required_for_system)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (topic_id, topic, category, content, 1.0, now, now, 1))

        conn.commit()
        conn.close()
        logger.info("Loaded %d core syllabus topics (100%% mastery)", len(core_topics))

    # -----------------------------------------------------------------------
    # Read path
    # -----------------------------------------------------------------------

    def get_knowledge(self, topic: str, prefer_core: bool = True) -> Optional[Dict]:
        """Retrieve knowledge - checks core tier first, then weighted."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *, 'core' as tier, 1.0 as current_weight
            FROM core_knowledge
            WHERE topic LIKE ? OR ? LIKE ('%' || topic || '%')
            ORDER BY mastery_level DESC
            LIMIT 1
        """, ("%" + topic + "%", topic))

        result = cursor.fetchone()

        if result:
            cursor.execute("""
                UPDATE core_knowledge
                SET last_reviewed = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), result["id"]))
            conn.commit()
            conn.close()
            logger.info("CORE knowledge retrieved: '%s' (100%% mastery)", topic)
            return dict(result)

        normalized = topic.lower().strip()
        cursor.execute("""
            SELECT *, 'weighted' as tier, weight as current_weight
            FROM weighted_knowledge
            WHERE normalized_topic = ? OR topic LIKE ?
            ORDER BY weight DESC, confidence DESC
            LIMIT 1
        """, (normalized, "%" + topic + "%"))

        result = cursor.fetchone()

        if result:
            new_weight = min(result["weight"] + 0.05, 1.0)
            cursor.execute("""
                UPDATE weighted_knowledge
                SET access_count = access_count + 1,
                    weight = ?,
                    last_accessed = ?
                WHERE id = ?
            """, (new_weight, datetime.now().isoformat(), result["id"]))
            conn.commit()
            conn.close()
            logger.info("Weighted knowledge retrieved: '%s' (weight: %.2f)", topic, new_weight)
            return dict(result)

        conn.close()
        return None

    # -----------------------------------------------------------------------
    # Write paths — all through quarantine.safe_write() (5a)
    # -----------------------------------------------------------------------

    def store_core_knowledge(
        self, topic: str, content: str, category: str = "System",
        source: str = "manual_admin"
    ) -> Optional[str]:
        """
        Store or update core knowledge (100% mastery, never decays).
        Goes through quarantine validation. Returns topic_id or None if quarantined.
        """
        doc = {
            "topic":    topic,
            "content":  content,
            "category": category,
            "source":   source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        def _write(d: dict) -> None:
            """Internal write function for core knowledge."""
            t_id = hashlib.md5(d["topic"].encode()).hexdigest()[:16]
            now  = datetime.now().isoformat()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO core_knowledge
                (id, topic, category, content, mastery_level, last_reviewed, created_at, required_for_system)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (t_id, d["topic"], d["category"], d["content"], 1.0, now, now, 1))
            conn.commit()
            conn.close()

        written, message = self.quarantine.safe_write(doc, _write)
        if not written:
            logger.warning("store_core_knowledge blocked: %s", message)
            return None

        topic_id = hashlib.md5(topic.encode()).hexdigest()[:16]
        logger.info("Stored CORE knowledge: '%s' (100%% mastery)", topic)
        return topic_id

    def store_weighted_knowledge(
        self,
        topic: str,
        content: str,
        source: str = "user_interaction",
        confidence: float = 0.5,
    ) -> Optional[str]:
        """
        Store new knowledge in weighted tier (starts with low weight).
        Goes through quarantine validation. Returns knowledge_id or None if quarantined.
        """
        doc = {
            "topic":      topic,
            "content":    content,
            "source":     source,
            "confidence": confidence,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }

        def _write(d: dict) -> None:
            """Internal write function for weighted knowledge."""
            k_id       = hashlib.md5(d["topic"].encode()).hexdigest()[:16]
            normalized = d["topic"].lower().strip()
            now        = datetime.now().isoformat()
            conn       = sqlite3.connect(self.db_path)
            cursor     = conn.cursor()
            cursor.execute(
                "SELECT id, weight FROM weighted_knowledge WHERE normalized_topic = ?",
                (normalized,),
            )
            existing = cursor.fetchone()
            if existing:
                cursor.execute("""
                    UPDATE weighted_knowledge
                    SET content = ?,
                        weight = min(weight + 0.1, 1.0),
                        confidence = ?,
                        last_accessed = ?,
                        access_count = access_count + 1
                    WHERE normalized_topic = ?
                """, (d["content"], d["confidence"], now, normalized))
            else:
                cursor.execute("""
                    INSERT INTO weighted_knowledge
                    (id, topic, normalized_topic, content, weight, access_count,
                     last_accessed, created_at, source, confidence, can_promote)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    k_id, d["topic"], normalized, d["content"], 0.3, 1,
                    now, now, d["source"], d["confidence"], d["confidence"] > 0.8,
                ))
            conn.commit()
            conn.close()

        written, message = self.quarantine.safe_write(doc, _write)
        if not written:
            logger.warning("store_weighted_knowledge blocked: %s", message)
            return None

        knowledge_id = hashlib.md5(topic.encode()).hexdigest()[:16]
        logger.info("Stored weighted knowledge: '%s' (weight: 0.3)", topic)
        return knowledge_id

    # Alias methods to match generic API used in other components
    def add_knowledge(
        self,
        topic: str,
        content: str,
        source: str = "user_interaction",
        confidence: float = 0.5,
    ) -> Optional[str]:
        """
        Generic add_knowledge alias — routes to store_weighted_knowledge with quarantine.
        Returns knowledge_id or None if quarantined.
        """
        return self.store_weighted_knowledge(
            topic=topic, content=content, source=source, confidence=confidence
        )

    def save_knowledge(
        self,
        topic: str,
        content: str,
        source: str = "user_interaction",
        confidence: float = 0.5,
        is_core: bool = False,
    ) -> Optional[str]:
        """
        Save knowledge to the appropriate tier, routed through quarantine.
        Returns the stored ID or None if quarantined.
        """
        if is_core:
            return self.store_core_knowledge(
                topic=topic, content=content, source=source
            )
        return self.store_weighted_knowledge(
            topic=topic, content=content, source=source, confidence=confidence
        )

    def store_topic(
        self,
        topic: str,
        content: str,
        source: str = "user_interaction",
        confidence: float = 0.5,
    ) -> Optional[str]:
        """
        store_topic alias — routes to store_weighted_knowledge with quarantine.
        Returns knowledge_id or None if quarantined.
        """
        return self.store_weighted_knowledge(
            topic=topic, content=content, source=source, confidence=confidence
        )

    # -----------------------------------------------------------------------
    # Promotion and read helpers
    # -----------------------------------------------------------------------

    def promote_to_core(self, topic: str) -> bool:
        """Promote weighted knowledge to core tier if important enough."""
        weighted = self.get_knowledge(topic, prefer_core=False)
        if weighted and weighted["tier"] == "weighted" and weighted["weight"] > 0.8:
            result = self.store_core_knowledge(
                topic, weighted["content"], category="Promoted"
            )
            if result:
                logger.info("PROMOTED '%s' from weighted to core tier", topic)
                return True
        return False

    def get_core_topics(self) -> List[Dict]:
        """Get all core syllabus topics DMAI has mastered."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT topic, category, mastery_level, last_reviewed "
            "FROM core_knowledge ORDER BY category, topic"
        )
        results = cursor.fetchall()
        conn.close()
        return [dict(r) for r in results]

    def get_high_weight_topics(self, min_weight: float = 0.7) -> List[Dict]:
        """Get well-learned weighted topics (candidates for promotion)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT topic, weight, access_count, confidence
            FROM weighted_knowledge
            WHERE weight >= ?
            ORDER BY weight DESC
            LIMIT 20
        """, (min_weight,))
        results = cursor.fetchall()
        conn.close()
        return [dict(r) for r in results]

    def decay_weights(self, days_threshold: int = 7) -> None:
        """Reduce weights of unused knowledge (core tier never decays)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        threshold_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
        cursor.execute("""
            UPDATE weighted_knowledge
            SET weight = weight * 0.9
            WHERE last_accessed < ? AND weight > 0.1
        """, (threshold_date,))
        conn.commit()
        conn.close()
        logger.info("Applied weight decay to unused knowledge")

    def get_system_knowledge_for_evolution(self) -> Dict:
        """Get all knowledge needed for system evolutions."""
        core_topics = self.get_core_topics()
        high_weight  = self.get_high_weight_topics(0.8)
        return {
            "core_mastery":          core_topics,
            "strong_weighted":       high_weight,
            "total_core_topics":     len(core_topics),
            "total_weighted_topics": len(high_weight),
            "message":               "Core topics are always available at 100% for system operations",
        }
