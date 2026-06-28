"""
ExpertBrain — loads a curated, citable knowledge seed into DMAI's core_knowledge.

On first boot (and idempotent on every restart), reads
`components/brain/seed/expert_brain_v1.json` and stores each entry via
`KnowledgeManager.store_core_knowledge`. Each entry has a real `source` URL
required by the KB quarantine layer. No synthesised facts.

Domains covered:
  - quantitative_trading
  - risk_management
  - sports_betting
  - software_engineering
  - ai_ml_engineering
  - personal_finance_uk
  - uk_real_estate
  - business_monetisation

After load, exposes search + stats. The trader, code-generator, and chat
surface can query the brain to ground responses in primary-source facts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

SEED_VERSION = "1.0.0"
SEED_PATH = Path(__file__).resolve().parent / "seed" / "expert_brain_v1.json"


class ExpertBrain:
    """Curated expert-domain knowledge brain for DMAI."""

    def __init__(
        self,
        knowledge_manager: Any = None,
        data_path: str | Path = "data",
        seed_path: Optional[Path] = None,
    ) -> None:
        self.km = knowledge_manager
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.data_path / "dmai_knowledge.db")
        self.seed_path = seed_path or SEED_PATH
        self._lock = threading.RLock()
        self._init_db()
        self._seed_data: Dict[str, Any] = self._load_seed_file()

    # ── DB ────────────────────────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        c = safe_open_kdb(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS brain_entries ("
                "id TEXT PRIMARY KEY, "
                "domain TEXT NOT NULL, "
                "domain_label TEXT, "
                "topic TEXT NOT NULL, "
                "content TEXT NOT NULL, "
                "source_url TEXT NOT NULL, "
                "tier TEXT DEFAULT 'canonical', "
                "version TEXT, "
                "loaded_at TEXT DEFAULT (datetime('now')))"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_brain_domain "
                "ON brain_entries(domain)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS brain_load_log ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ts TEXT DEFAULT (datetime('now')), "
                "seed_version TEXT, "
                "entries_loaded INTEGER, "
                "entries_skipped INTEGER, "
                "notes TEXT)"
            )
            c.commit()

    def _load_seed_file(self) -> Dict[str, Any]:
        if not self.seed_path.exists():
            logger.error("ExpertBrain: seed file missing at %s", self.seed_path)
            return {"version": "0.0.0", "domains": {}}
        try:
            with open(self.seed_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.exception("ExpertBrain: failed to read seed: %s", e)
            return {"version": "0.0.0", "domains": {}}

    # ── Loading ───────────────────────────────────────────────────────────────
    def load(self, force: bool = False) -> Dict[str, Any]:
        """
        Load (or reload) the seed into brain_entries + core_knowledge.
        Idempotent: existing entries are updated, not duplicated.
        """
        with self._lock:
            return self._load_inner(force=force)

    def _load_inner(self, force: bool) -> Dict[str, Any]:
        version = self._seed_data.get("version", SEED_VERSION)
        domains = self._seed_data.get("domains", {})

        if not domains:
            return {"loaded": 0, "skipped": 0, "error": "no seed data"}

        loaded = 0
        skipped = 0
        km_blocked = 0

        for domain_key, domain in domains.items():
            label = domain.get("label", domain_key)
            tier = domain.get("tier", "canonical")
            for entry in domain.get("entries", []):
                topic = entry.get("topic")
                content = entry.get("content")
                source = entry.get("source")
                if not (topic and content and source):
                    skipped += 1
                    continue

                entry_id = hashlib.md5(
                    f"{domain_key}::{topic}".encode("utf-8")
                ).hexdigest()[:16]

                # Persist in brain table
                try:
                    with self._conn() as c:
                        c.execute(
                            "INSERT OR REPLACE INTO brain_entries("
                            "id, domain, domain_label, topic, content, "
                            "source_url, tier, version, loaded_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                            (
                                entry_id, domain_key, label, topic, content,
                                source, tier, version,
                            ),
                        )
                        c.commit()
                except Exception as e:
                    logger.exception("brain insert failed for %s: %s", topic, e)
                    skipped += 1
                    continue

                # Mirror into core_knowledge (100% mastery, never decays)
                if self.km and hasattr(self.km, "store_core_knowledge"):
                    try:
                        full_content = (
                            f"{content}\n\nSource: {source}\nDomain: {label}"
                        )
                        ok = self.km.store_core_knowledge(
                            topic=f"[{domain_key}] {topic}",
                            content=full_content,
                            category=domain_key,
                            source=source,
                        )
                        if not ok:
                            km_blocked += 1
                    except Exception as e:
                        logger.debug("km mirror failed for %s: %s", topic, e)

                loaded += 1

        notes = f"km_quarantine_blocks={km_blocked}"
        with self._conn() as c:
            c.execute(
                "INSERT INTO brain_load_log(seed_version, entries_loaded, "
                "entries_skipped, notes) VALUES (?, ?, ?, ?)",
                (version, loaded, skipped, notes),
            )
            c.commit()

        logger.info(
            "ExpertBrain: loaded %d entries across %d domains (skipped=%d, km_blocked=%d)",
            loaded, len(domains), skipped, km_blocked,
        )
        return {
            "version": version,
            "domains": list(domains.keys()),
            "loaded": loaded,
            "skipped": skipped,
            "km_blocked": km_blocked,
        }

    # ── Query surface ─────────────────────────────────────────────────────────
    def domains(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT domain, MAX(domain_label) AS label, "
                "COUNT(*) AS entries FROM brain_entries "
                "GROUP BY domain ORDER BY domain"
            ).fetchall()
            return [dict(r) for r in rows]

    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Case-insensitive LIKE search over topic + content."""
        q = f"%{query.lower()}%"
        sql = (
            "SELECT id, domain, domain_label, topic, content, source_url "
            "FROM brain_entries WHERE (LOWER(topic) LIKE ? OR LOWER(content) LIKE ?)"
        )
        params: List[Any] = [q, q]
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        sql += " LIMIT ?"
        params.append(limit)
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM brain_entries WHERE id = ?", (entry_id,)
            ).fetchone()
            return dict(row) if row else None

    def by_domain(self, domain: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, topic, content, source_url FROM brain_entries "
                "WHERE domain = ? ORDER BY topic", (domain,),
            ).fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) AS n FROM brain_entries").fetchone()
            by_dom = c.execute(
                "SELECT domain, COUNT(*) AS n FROM brain_entries GROUP BY domain"
            ).fetchall()
            last_load = c.execute(
                "SELECT * FROM brain_load_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return {
                "total_entries": total["n"] if total else 0,
                "by_domain": {r["domain"]: r["n"] for r in by_dom},
                "seed_version": self._seed_data.get("version"),
                "last_load": dict(last_load) if last_load else None,
            }

    def context_for(
        self,
        topic_query: str,
        max_entries: int = 5,
        max_chars: int = 4000,
    ) -> str:
        """
        Build a compact context string for LLM grounding. Returns markdown
        with topic + content + source for top matching entries.
        """
        hits = self.search(topic_query, limit=max_entries)
        out: List[str] = []
        used = 0
        for h in hits:
            block = (
                f"### {h['topic']}\n"
                f"_Domain: {h['domain_label']}_\n\n"
                f"{h['content']}\n\n"
                f"Source: {h['source_url']}\n"
            )
            if used + len(block) > max_chars:
                break
            out.append(block)
            used += len(block)
        return "\n".join(out) if out else ""


def get_expert_brain(knowledge_manager=None, data_path: str = "data") -> ExpertBrain:
    return ExpertBrain(knowledge_manager=knowledge_manager, data_path=data_path)
