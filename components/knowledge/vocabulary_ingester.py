"""
VocabularyIngester — DMAI's language and encyclopaedia knowledge engine.

Sources:
  - Wiktionary REST API (free, no key) — definitions, etymology, pronunciation, POS
  - Wikipedia REST API (free, no key) — encyclopaedic summaries, categories
  - Project Gutenberg RSS — classic literature titles for cultural literacy

Stores learned vocabulary in the `vocabulary` table and encyclopaedia entries
in the `encyclopaedia` table of dmai_knowledge.db.

Runs continuously, cycling through curated word/topic lists and dynamically
expanding via insights already stored in the DB.
"""

import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DB_PATH   = Path("data/dmai_knowledge.db")
DATA_PATH = Path("data/vocabulary")

# ── Curated seed lists ─────────────────────────────────────────────────────
# OED-grade vocabulary: academic, philosophical, scientific, literary terms
OED_SEED_WORDS = [
    # Logic & philosophy
    "epistemology", "ontology", "phenomenology", "hermeneutics", "dialectic",
    "syllogism", "sophistry", "empiricism", "rationalism", "pragmatism",
    "solipsism", "determinism", "teleology", "axiology", "deontology",
    "consequentialism", "nihilism", "existentialism", "stoicism", "nominalism",
    # Science & mathematics
    "heuristic", "algorithm", "recursion", "stochastic", "entropy",
    "eigenvalue", "topology", "manifold", "gradient", "divergence",
    "convergence", "isomorphism", "homomorphism", "hypothesis", "paradigm",
    "falsifiability", "reductionism", "emergence", "complexity", "bifurcation",
    # Language & linguistics
    "morphology", "phonology", "syntax", "semantics", "pragmatics",
    "etymology", "lexicography", "semiotics", "rhetoric", "prosody",
    "metonymy", "synecdoche", "catachresis", "apophasis", "euphemism",
    "circumlocution", "periphrasis", "pleonasm", "tautology", "oxymoron",
    "paradox", "polysemy", "homonym", "antonym", "hypernym",
    # Literature & arts
    "ekphrasis", "bildungsroman", "denouement", "deus ex machina", "hubris",
    "anagnorisis", "hamartia", "catharsis", "mimesis", "verisimilitude",
    "allegory", "anachronism", "bathos", "chiasmus", "epistrophe",
    # Economics & society
    "oligopoly", "monopsony", "arbitrage", "liquidity", "solvency",
    "hegemony", "sovereignty", "jurisprudence", "realpolitik", "zeitgeist",
    "weltanschauung", "schadenfreude", "gestalt", "leitmotif", "pathos",
    # Advanced general vocabulary
    "perspicacious", "loquacious", "laconic", "sycophantic", "obsequious",
    "recalcitrant", "perspicuous", "equivocal", "tendentious", "pellucid",
    "mellifluous", "sesquipedalian", "grandiloquent", "truculent", "mendacious",
    "perfidious", "inveterate", "assiduous", "fastidious", "meticulous",
    "punctilious", "scrupulous", "conscientious", "tenacious", "pertinacious",
    "sagacious", "perspicacious", "judicious", "discerning", "astute",
]

# Encyclopaedic topics — broad knowledge base
ENCYCLOPAEDIA_SEED_TOPICS = [
    # History
    "Ancient Rome", "Renaissance", "Industrial Revolution", "World War I", "Cold War",
    "Byzantine Empire", "Ottoman Empire", "Ming dynasty", "Mughal Empire", "Age of Enlightenment",
    # Science
    "Theory of relativity", "Quantum mechanics", "Evolution", "DNA", "Plate tectonics",
    "Thermodynamics", "Electromagnetism", "Periodic table", "Big Bang", "Black hole",
    # Philosophy
    "Socrates", "Plato", "Aristotle", "Immanuel Kant", "Friedrich Nietzsche",
    "John Stuart Mill", "Bertrand Russell", "Ludwig Wittgenstein", "Karl Marx", "Jean-Paul Sartre",
    # Literature
    "Shakespeare", "James Joyce", "Virginia Woolf", "Franz Kafka", "Fyodor Dostoevsky",
    "Leo Tolstoy", "Charles Dickens", "Jane Austen", "Homer", "Dante Alighieri",
    # Mathematics
    "Calculus", "Number theory", "Graph theory", "Game theory", "Probability theory",
    "Set theory", "Abstract algebra", "Differential geometry", "Combinatorics", "Cryptography",
    # Economics
    "Keynesian economics", "Adam Smith", "Supply and demand", "Comparative advantage",
    "Monetary policy", "Behavioural economics", "Game theory", "Public goods", "Externality",
]

LINGUISTICS_TOPICS = [
    "morpheme", "phoneme", "lexeme", "syntax", "grammar", "Indo-European languages",
    "language acquisition", "Sapir-Whorf hypothesis", "sociolinguistics", "pragmatics",
    "corpus linguistics", "historical linguistics", "dialect", "register", "creole language",
    "linguistic relativity", "universal grammar", "Noam Chomsky", "structuralism", "semiology",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VocabularyIngester:
    """
    Fetches vocabulary and encyclopaedic knowledge from free public APIs.
    Runs in a background thread, cycling through word/topic lists.
    """

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "DMAI-Knowledge-Bot/1.0 (autonomous educational AI; contact: milesd040@gmail.com)"
        })
        self._ensure_tables()
        DATA_PATH.mkdir(parents=True, exist_ok=True)

    # ── Table init ──────────────────────────────────────────────────────────
    def _ensure_tables(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS vocabulary (
                id TEXT PRIMARY KEY,
                word TEXT NOT NULL UNIQUE,
                part_of_speech TEXT,
                definition TEXT NOT NULL,
                etymology TEXT,
                example TEXT,
                pronunciation TEXT,
                domain TEXT DEFAULT 'general',
                source TEXT DEFAULT 'wiktionary',
                confidence REAL DEFAULT 0.9,
                created_at TEXT NOT NULL,
                last_reviewed TEXT
            );

            CREATE TABLE IF NOT EXISTS encyclopaedia (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL UNIQUE,
                summary TEXT NOT NULL,
                categories TEXT,
                url TEXT,
                domain TEXT DEFAULT 'general',
                word_count INTEGER DEFAULT 0,
                source TEXT DEFAULT 'wikipedia',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_vocab_word ON vocabulary(word);
            CREATE INDEX IF NOT EXISTS idx_vocab_domain ON vocabulary(domain);
            CREATE INDEX IF NOT EXISTS idx_encyc_title ON encyclopaedia(title);
        """)
        conn.commit()
        conn.close()
        logger.info("VocabularyIngester: tables ensured")

    # ── Wiktionary ──────────────────────────────────────────────────────────
    def fetch_word(self, word: str) -> Optional[dict]:
        """Fetch word data from Wiktionary REST API (en)."""
        url = f"https://en.wiktionary.org/api/rest_v1/page/definition/{requests.utils.quote(word)}"
        try:
            r = self.session.get(url, timeout=15)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            entries = data.get("en", [])
            if not entries:
                return None

            entry = entries[0]
            definitions = entry.get("definitions", [])
            if not definitions:
                return None

            defn_obj = definitions[0]
            definition = defn_obj.get("definition", "")
            # Strip HTML tags
            import re
            definition = re.sub(r"<[^>]+>", "", definition).strip()
            if not definition:
                return None

            examples = defn_obj.get("examples", [])
            example = re.sub(r"<[^>]+>", "", examples[0]).strip() if examples else ""

            pos = entry.get("partOfSpeech", "")
            pronunciation = ""
            if entry.get("pronunciations"):
                pr = entry["pronunciations"][0]
                pronunciation = pr.get("text", "")

            # Try etymology from a second call to the parse API
            etymology = self._fetch_etymology(word)

            return {
                "word": word.lower(),
                "part_of_speech": pos,
                "definition": definition[:1000],
                "etymology": etymology[:500] if etymology else "",
                "example": example[:500],
                "pronunciation": pronunciation[:100],
            }
        except Exception as e:
            logger.debug("Wiktionary fetch failed for '%s': %s", word, e)
            return None

    def _fetch_etymology(self, word: str) -> str:
        """Attempt to get etymology from Wikipedia's Wiktionary parse endpoint."""
        try:
            url = "https://en.wiktionary.org/w/api.php"
            params = {
                "action": "query", "titles": word, "prop": "extracts",
                "exintro": True, "format": "json", "redirects": 1
            }
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
            for p in pages.values():
                extract = p.get("extract", "")
                if extract:
                    import re
                    clean = re.sub(r"<[^>]+>", "", extract)
                    # Find etymology section hint
                    for line in clean.split("\n"):
                        if any(kw in line.lower() for kw in ["from ", "origin", "latin", "greek", "old english", "french", "proto-"]):
                            return line.strip()[:400]
            return ""
        except Exception:
            return ""

    def store_word(self, word_data: dict) -> bool:
        """Store a vocabulary entry. Returns True if new, False if already existed."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            existing = conn.execute("SELECT id FROM vocabulary WHERE word=?",
                                    (word_data["word"],)).fetchone()
            if existing:
                conn.close()
                return False
            domain = self._classify_word_domain(word_data.get("definition", ""),
                                                 word_data.get("part_of_speech", ""))
            conn.execute(
                "INSERT INTO vocabulary (id, word, part_of_speech, definition, etymology, "
                "example, pronunciation, domain, source, confidence, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()),
                 word_data["word"],
                 word_data.get("part_of_speech", ""),
                 word_data["definition"],
                 word_data.get("etymology", ""),
                 word_data.get("example", ""),
                 word_data.get("pronunciation", ""),
                 domain, "wiktionary", 0.92, _now())
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("store_word failed for '%s': %s", word_data.get("word"), e)
            return False

    # ── Wikipedia ──────────────────────────────────────────────────────────
    def fetch_topic(self, title: str) -> Optional[dict]:
        """Fetch encyclopaedic summary from Wikipedia REST API."""
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
        try:
            r = self.session.get(url, timeout=15)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            extract = data.get("extract", "")
            if not extract or len(extract) < 50:
                return None
            categories = data.get("categories", [])
            cat_str = json.dumps([c.get("title", "") for c in categories[:10]])
            return {
                "title": data.get("title", title),
                "summary": extract[:2000],
                "categories": cat_str,
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "word_count": len(extract.split()),
            }
        except Exception as e:
            logger.debug("Wikipedia fetch failed for '%s': %s", title, e)
            return None

    def store_topic(self, topic_data: dict) -> bool:
        """Store an encyclopaedia entry. Returns True if new."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            existing = conn.execute("SELECT id FROM encyclopaedia WHERE title=?",
                                    (topic_data["title"],)).fetchone()
            if existing:
                conn.close()
                return False
            domain = self._classify_encyc_domain(topic_data.get("title", ""),
                                                  topic_data.get("summary", ""))
            conn.execute(
                "INSERT INTO encyclopaedia (id, title, summary, categories, url, domain, "
                "word_count, source, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()),
                 topic_data["title"],
                 topic_data["summary"],
                 topic_data.get("categories", "[]"),
                 topic_data.get("url", ""),
                 domain,
                 topic_data.get("word_count", 0),
                 "wikipedia", _now())
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("store_topic failed for '%s': %s", topic_data.get("title"), e)
            return False

    # ── Domain classifiers ─────────────────────────────────────────────────
    def _classify_word_domain(self, definition: str, pos: str) -> str:
        d = definition.lower()
        if any(w in d for w in ["philosophy", "logic", "ethics", "metaphysics"]):
            return "philosophy"
        if any(w in d for w in ["grammar", "language", "linguistic", "word", "speech", "syntax"]):
            return "linguistics"
        if any(w in d for w in ["mathematics", "mathematical", "geometry", "algebra", "calculus"]):
            return "mathematics"
        if any(w in d for w in ["science", "physics", "chemistry", "biology", "scientific"]):
            return "science"
        if any(w in d for w in ["economy", "economic", "finance", "market", "trade"]):
            return "economics"
        if any(w in d for w in ["literature", "poem", "novel", "narrative", "literary"]):
            return "literature"
        if any(w in d for w in ["psychology", "cognitive", "mind", "behaviour"]):
            return "psychology"
        return "general"

    def _classify_encyc_domain(self, title: str, summary: str) -> str:
        text = (title + " " + summary).lower()
        if any(w in text for w in ["war", "empire", "dynasty", "revolution", "history", "ancient"]):
            return "history"
        if any(w in text for w in ["theorem", "equation", "mathematics", "calculus", "algebra"]):
            return "mathematics"
        if any(w in text for w in ["physics", "chemistry", "biology", "science", "quantum"]):
            return "science"
        if any(w in text for w in ["philosophy", "philosopher", "ethics", "metaphysics"]):
            return "philosophy"
        if any(w in text for w in ["novel", "author", "poet", "literature", "writer"]):
            return "literature"
        if any(w in text for w in ["economy", "economics", "market", "trade", "monetary"]):
            return "economics"
        if any(w in text for w in ["language", "linguistics", "grammar", "phonology"]):
            return "linguistics"
        return "general"

    # ── Dynamic expansion from insights ────────────────────────────────────
    def _get_insight_topics(self, limit: int = 20) -> list:
        """Pull high-confidence concept terms from the insights table."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            rows = conn.execute(
                "SELECT DISTINCT source_topic FROM insights WHERE confidence > 0.6 "
                "AND NOT EXISTS (SELECT 1 FROM encyclopaedia WHERE title = source_topic) "
                "ORDER BY confidence DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
            return [r[0] for r in rows if r[0] and len(r[0]) > 3]
        except Exception as e:
            logger.debug("_get_insight_topics failed: %s", e)
            return []

    # ── Stats ──────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        try:
            conn = sqlite3.connect(str(self.db_path))
            vocab_count = conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
            encyc_count = conn.execute("SELECT COUNT(*) FROM encyclopaedia").fetchone()[0]
            vocab_domains = dict(conn.execute(
                "SELECT domain, COUNT(*) FROM vocabulary GROUP BY domain"
            ).fetchall())
            encyc_domains = dict(conn.execute(
                "SELECT domain, COUNT(*) FROM encyclopaedia GROUP BY domain"
            ).fetchall())
            conn.close()
            return {
                "vocabulary_total": vocab_count,
                "encyclopaedia_total": encyc_count,
                "vocabulary_by_domain": vocab_domains,
                "encyclopaedia_by_domain": encyc_domains,
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Main run loop ───────────────────────────────────────────────────────
    def run_once(self):
        """Single pass: ingest a batch of vocabulary + encyclopaedia entries."""
        import random

        # Build word list: seeds + insight-derived terms
        insight_topics = self._get_insight_topics(30)
        word_pool = list(OED_SEED_WORDS) + LINGUISTICS_TOPICS
        # Add single-word insight topics as vocabulary candidates
        word_pool += [t for t in insight_topics if " " not in t and len(t) > 3]
        random.shuffle(word_pool)

        # Build topic list: seeds + multi-word insight topics
        topic_pool = list(ENCYCLOPAEDIA_SEED_TOPICS) + LINGUISTICS_TOPICS
        topic_pool += [t for t in insight_topics if " " in t]
        random.shuffle(topic_pool)

        new_words = 0
        new_topics = 0

        # Ingest vocabulary (up to 20 new words per pass)
        words_attempted = 0
        for word in word_pool:
            if new_words >= 20:
                break
            data = self.fetch_word(word)
            if data:
                if self.store_word(data):
                    new_words += 1
                    logger.info("VocabularyIngester: learned '%s' (%s) — %s",
                                word, data.get("part_of_speech", "?"),
                                data["definition"][:60])
                    # Also add as an insight so it feeds the knowledge graph
                    self._add_to_insights(word, data["definition"], "linguistics")
            words_attempted += 1
            time.sleep(0.3)  # polite rate limit

        # Ingest encyclopaedia (up to 15 new topics per pass)
        for topic in topic_pool:
            if new_topics >= 15:
                break
            data = self.fetch_topic(topic)
            if data:
                if self.store_topic(data):
                    new_topics += 1
                    logger.info("VocabularyIngester: encyclopaedia '%s' (%d words)",
                                topic, data.get("word_count", 0))
                    self._add_to_insights(topic, data["summary"][:200], data.get("domain", "general"))
            time.sleep(0.4)

        logger.info("VocabularyIngester pass complete: +%d words, +%d topics", new_words, new_topics)
        return {"new_words": new_words, "new_topics": new_topics}

    def _add_to_insights(self, concept: str, text: str, domain: str):
        """Mirror learning into the insights table so it feeds the knowledge graph."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            existing = conn.execute(
                "SELECT id FROM insights WHERE source_topic=? LIMIT 1", (concept,)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT OR IGNORE INTO insights "
                    "(id, insight_text, entity_type, entities, relationship, confidence, "
                    "source_topic, target_topic, source_type, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (str(uuid.uuid4()),
                     text[:500],
                     "concept",
                     json.dumps([concept]),
                     "defines",
                     0.88,
                     concept,
                     domain,
                     "vocabulary_ingester",
                     _now())
                )
                conn.commit()
            conn.close()
            # Feed graph writer
            try:
                from components.graph_writer import GraphWriter
                GraphWriter().add_insight_node(
                    concept=concept, domain=domain, source="vocabulary_ingester"
                )
            except Exception:
                pass
        except Exception as e:
            logger.debug("_add_to_insights failed: %s", e)

    def run_continuous(self, interval_mins: int = 30):
        """Run ingestion passes indefinitely."""
        logger.info("VocabularyIngester: continuous loop started (every %dm)", interval_mins)
        while True:
            try:
                result = self.run_once()
                logger.info("VocabularyIngester: pass done — %s", result)
            except Exception as e:
                logger.error("VocabularyIngester loop error: %s", e)
            time.sleep(interval_mins * 60)
