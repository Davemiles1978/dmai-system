"""
VocabularyIngester — DMAI's language and encyclopaedic knowledge engine.

Sources (Wikipedia is NOT used — policy):
  Vocabulary:
    - Wiktionary REST API (free, no key) — definitions, POS, pronunciation
    - Merriam-Webster Collegiate API (gated on MERRIAM_API_KEY env var) — adds
      authoritative US-English definitions on top of Wiktionary data when
      available. 1,000 calls/day on the free tier.
    - Cached large word pool (dwyl/english-words, ~370k words) advanced via a
      persistent cursor across passes.
  Encyclopaedia:
    - Stanford Encyclopedia of Philosophy (plato.stanford.edu) — ~1,800
      peer-reviewed entries, modern, scholarly. Indexed from contents.html.
    - Scholarpedia (scholarpedia.org) — peer-reviewed articles in
      computational neuroscience, dynamical systems, machine learning,
      astrophysics, and physics.

Stores learned vocabulary in the `vocabulary` table and encyclopaedic entries
in the `encyclopaedia` table of dmai_knowledge.db.
"""

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from components.db import acquire_write_lock, safe_open_kdb

logger = logging.getLogger(__name__)

DB_PATH   = Path("data/dmai_knowledge.db")
DATA_PATH = Path("data/vocabulary")

# ── Batched-insert defaults (2026-06-30, lock-storm reduction) ─────────────
# Per-word INSERT+COMMIT was the dominant source of write-lock acquisitions.
# Accumulate rows and flush in transactions of N (default 200). All knobs are
# optional env overrides — no env var is required and no schema changes.
DEFAULT_BATCH_SIZE    = 200
DEFAULT_FLUSH_SECONDS = 5.0

_VOCAB_INSERT_SQL = (
    "INSERT OR IGNORE INTO vocabulary (id, word, part_of_speech, definition, "
    "etymology, example, pronunciation, domain, source, confidence, created_at) "
    "VALUES (?,?,?,?,?,?,?,?,?,?,?)"
)

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

# ── Large word pool (downloaded once, cached to disk) ──────────────────────
# dwyl/english-words: ~370k words. Filtered to alpha-only, 4-20 chars.
WORD_POOL_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
WORD_POOL_PATH = Path("data/word_pool.txt")
WORD_CURSOR_PATH = Path("data/vocab_cursor.txt")


def _load_big_word_pool() -> list:
    """Download (once) and cache a large English word list. Filter to OED-grade tokens."""
    try:
        if not WORD_POOL_PATH.exists():
            WORD_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
            logger.info("VocabularyIngester: downloading large word pool from dwyl/english-words")
            r = requests.get(WORD_POOL_URL, timeout=60)
            r.raise_for_status()
            raw = r.text
            words = []
            for w in raw.split():
                w = w.strip().lower()
                if 4 <= len(w) <= 20 and w.isalpha():
                    words.append(w)
            # dedupe preserving order
            seen = set()
            unique = []
            for w in words:
                if w not in seen:
                    seen.add(w)
                    unique.append(w)
            WORD_POOL_PATH.write_text("\n".join(unique))
            logger.info("VocabularyIngester: cached %d words to %s", len(unique), WORD_POOL_PATH)
            return unique
        else:
            words = [w.strip() for w in WORD_POOL_PATH.read_text().splitlines() if w.strip()]
            return words
    except Exception as e:
        logger.warning("VocabularyIngester: big-pool load failed (%s) — falling back to seed list", e)
        return []


def _read_cursor() -> int:
    try:
        if WORD_CURSOR_PATH.exists():
            return int(WORD_CURSOR_PATH.read_text().strip() or "0")
    except Exception:
        pass
    return 0


def _write_cursor(idx: int) -> None:
    try:
        WORD_CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        WORD_CURSOR_PATH.write_text(str(idx))
    except Exception:
        pass




def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VocabularyIngester:
    """
    Fetches vocabulary and encyclopaedic knowledge from free public APIs.
    Runs in a background thread, cycling through word/topic lists.
    """

    def __init__(self, db_path: str = None, batch_size: int = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "DMAI-Knowledge-Bot/1.0 (autonomous educational AI; contact: milesd040@gmail.com)"
        })

        # ── Batched-insert configuration ───────────────────────────────────
        # Precedence: explicit batch_size arg > VOCAB_INGEST_BATCH_SIZE env > default.
        if batch_size is not None:
            resolved = batch_size
        else:
            try:
                resolved = int(os.environ.get("VOCAB_INGEST_BATCH_SIZE", DEFAULT_BATCH_SIZE))
            except (TypeError, ValueError):
                resolved = DEFAULT_BATCH_SIZE
        self.batch_size = max(1, int(resolved))
        try:
            self.flush_seconds = float(
                os.environ.get("VOCAB_INGEST_FLUSH_SECONDS", DEFAULT_FLUSH_SECONDS)
            )
        except (TypeError, ValueError):
            self.flush_seconds = DEFAULT_FLUSH_SECONDS

        self._batch: list = []            # pending row tuples
        self._batch_words: set = set()    # in-memory dedup within the buffer
        self._batch_lock = threading.RLock()
        self._last_add_ts = time.monotonic()
        self.transactions = 0             # committed write transactions (observability/tests)
        self._failures_path = DATA_PATH / "vocab_ingest_failures.jsonl"
        self._flush_timer: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._ensure_tables()
        DATA_PATH.mkdir(parents=True, exist_ok=True)

    # ── Table init ──────────────────────────────────────────────────────────
    def _ensure_tables(self):
        conn = safe_open_kdb(str(self.db_path))
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
                source TEXT DEFAULT 'unknown',
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
        """Etymology helper. No Wikipedia/Wiktionary parse calls — policy.

        Etymology, when present, is taken from the Wiktionary REST definition
        payload itself (no extra request). Returns empty string here as a
        graceful no-op.
        """
        return ""

    def store_word(self, word_data: dict) -> bool:
        """Store a vocabulary entry. Returns True if new, False if already existed."""
        word_data = self._mw_augment(dict(word_data))
        try:
            conn = safe_open_kdb(str(self.db_path))
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
                 domain, word_data.get("source", "wiktionary"), 0.92, _now())
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("store_word failed for '%s': %s", word_data.get("word"), e)
            return False

    # ── Batched ingestion ───────────────────────────────────────────────────
    def _prepare_row(self, word_data: dict):
        """Augment + classify a word payload into an INSERT row tuple.

        Returns ``(word, row_tuple)`` or ``None`` if the payload is unusable.
        """
        wd = self._mw_augment(dict(word_data))
        word = (wd.get("word") or "").strip().lower()
        definition = wd.get("definition") or ""
        if not word or not definition:
            return None
        domain = self._classify_word_domain(definition, wd.get("part_of_speech", ""))
        row = (
            str(uuid.uuid4()),
            word,
            wd.get("part_of_speech", ""),
            definition,
            wd.get("etymology", ""),
            wd.get("example", ""),
            wd.get("pronunciation", ""),
            domain,
            wd.get("source", "wiktionary"),
            0.92,
            _now(),
        )
        return word, row

    def ingest_one(self, word_data: dict) -> None:
        """Buffer a single word for batched insertion. Auto-flushes when the
        buffer reaches ``batch_size``."""
        prepared = self._prepare_row(word_data)
        if prepared is None:
            return
        word, row = prepared
        self._maybe_start_flush_timer()
        with self._batch_lock:
            if word in self._batch_words:
                return  # dedup within the current buffer
            self._batch_words.add(word)
            self._batch.append(row)
            self._last_add_ts = time.monotonic()
            if len(self._batch) >= self.batch_size:
                self._flush_locked()

    def ingest_many(self, items) -> int:
        """Buffer many word payloads. Full batches flush automatically; any
        remainder stays buffered until ``flush()``, the idle timer, or
        shutdown. Returns the number of rows written by auto-flushes here."""
        written = 0
        for item in items:
            before = self.transactions
            self.ingest_one(item)
            # (transactions only advances on an actual flush)
            if self.transactions > before:
                written += 1
        return written

    def flush(self) -> int:
        """Flush any buffered rows in a single transaction. Returns rows written."""
        with self._batch_lock:
            return self._flush_locked()

    def _flush_locked(self) -> int:
        """Flush the buffer. Caller must hold ``self._batch_lock``."""
        if not self._batch:
            return 0
        rows = self._batch
        self._batch = []
        self._batch_words = set()
        return self._write_batch(rows)

    def _write_batch(self, rows) -> int:
        """Write rows in one transaction; on error fall back to one-by-one.

        The whole BEGIN/executemany/COMMIT is held under the process write lock
        so a batch is atomic with respect to other writers (the lock is
        reentrant, so the proxy's per-statement guards nest safely)."""
        conn = safe_open_kdb(str(self.db_path))
        try:
            with acquire_write_lock(self.db_path):
                conn.executemany(_VOCAB_INSERT_SQL, rows)
                conn.commit()
            self.transactions += 1
            return len(rows)
        except Exception as e:
            logger.warning(
                "VocabularyIngester: batch insert of %d rows failed (%s) — "
                "falling back to one-by-one", len(rows), e,
            )
            try:
                conn.rollback()
            except Exception:
                pass
            return self._write_rows_individually(rows)

    def _write_rows_individually(self, rows) -> int:
        """Per-row fallback so a single bad row doesn't drop the whole batch."""
        written = 0
        conn = safe_open_kdb(str(self.db_path))
        for row in rows:
            try:
                conn.execute(_VOCAB_INSERT_SQL, row)
                conn.commit()
                self.transactions += 1
                written += 1
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                self._record_failure(row, e)
        return written

    def _record_failure(self, row, error) -> None:
        """Append a failed row to a JSONL file (no new schema, per design)."""
        try:
            DATA_PATH.mkdir(parents=True, exist_ok=True)
            rec = {
                "ts": _now(),
                "word": row[1] if len(row) > 1 else None,
                "error": str(error),
                "row": list(row),
            }
            with open(self._failures_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception as e:
            logger.debug("could not record vocab ingest failure: %s", e)

    def _maybe_start_flush_timer(self) -> None:
        """Lazily start a daemon thread that flushes a partial buffer after it
        has been idle for ``flush_seconds``."""
        if self._flush_timer is not None or self.flush_seconds <= 0:
            return
        self._stop_event.clear()
        t = threading.Thread(
            target=self._idle_flush_loop, name="vocab-ingest-flush", daemon=True
        )
        self._flush_timer = t
        t.start()

    def _idle_flush_loop(self) -> None:
        poll = min(self.flush_seconds, 1.0) or 1.0
        while not self._stop_event.wait(poll):
            try:
                with self._batch_lock:
                    if self._batch and (
                        time.monotonic() - self._last_add_ts
                    ) >= self.flush_seconds:
                        self._flush_locked()
            except Exception as e:
                logger.debug("VocabularyIngester idle flush failed: %s", e)

    def shutdown(self) -> int:
        """Stop the idle timer and flush remaining buffered rows."""
        self._stop_event.set()
        return self.flush()

    # ── Merriam-Webster (gated on env var) ──────────────────────────
    def _mw_augment(self, word_data: dict) -> dict:
        """If MERRIAM_API_KEY is set, look up the word in Merriam-Webster's
        Collegiate Dictionary and merge a fuller definition + etymology into
        the existing payload. Silently returns the original payload on any
        error or missing key."""
        import os
        key = os.environ.get("MERRIAM_API_KEY", "").strip()
        if not key:
            return word_data
        try:
            word = word_data.get("word") or ""
            if not word:
                return word_data
            url = f"https://www.dictionaryapi.com/api/v3/references/collegiate/json/{requests.utils.quote(word)}?key={key}"
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            entries = r.json()
            if not isinstance(entries, list) or not entries:
                return word_data
            first = entries[0]
            if not isinstance(first, dict):
                return word_data
            shortdefs = first.get("shortdef") or []
            mw_def = shortdefs[0] if shortdefs else ""
            mw_etym = ""
            et = first.get("et")
            if isinstance(et, list) and et:
                for chunk in et:
                    if isinstance(chunk, list) and len(chunk) > 1 and chunk[0] == "text":
                        mw_etym = re.sub(r"\{[^}]+\}", "", chunk[1]).strip()
                        break
            mw_pos = first.get("fl") or ""
            if mw_def:
                # Prefer M-W definition (more authoritative) but keep Wiktionary
                # as a fallback inside the same field.
                base = word_data.get("definition", "")
                merged = mw_def
                if base and base not in merged:
                    merged = f"{mw_def} — (Wiktionary: {base})"
                word_data["definition"] = merged[:1000]
                word_data["source"] = "merriam_webster"
            if mw_etym and not word_data.get("etymology"):
                word_data["etymology"] = mw_etym[:500]
            if mw_pos and not word_data.get("part_of_speech"):
                word_data["part_of_speech"] = mw_pos
            return word_data
        except Exception as e:
            logger.debug("Merriam-Webster augment failed for %r: %s", word_data.get("word"), e)
            return word_data

    # ── Stanford Encyclopedia of Philosophy (SEP) ─────────────────────
    _SEP_INDEX_CACHE: Path = Path("data/sep_index.json")
    _SEP_BASE = "https://plato.stanford.edu"

    def _load_sep_index(self) -> list:
        """Return list of SEP entry slugs. Cached to disk after first fetch."""
        try:
            if self._SEP_INDEX_CACHE.exists():
                import json as _j
                return _j.loads(self._SEP_INDEX_CACHE.read_text())
        except Exception:
            pass
        try:
            r = self.session.get(f"{self._SEP_BASE}/contents.html", timeout=30)
            r.raise_for_status()
            slugs = sorted(set(re.findall(r'href="entries/([^"/]+)/"', r.text)))
            try:
                self._SEP_INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
                import json as _j
                self._SEP_INDEX_CACHE.write_text(_j.dumps(slugs))
            except Exception:
                pass
            logger.info("SEP index loaded: %d entries", len(slugs))
            return slugs
        except Exception as e:
            logger.warning("SEP index load failed: %s", e)
            return []

    def _fetch_sep_entry(self, slug: str) -> Optional[dict]:
        """Fetch a Stanford Encyclopedia of Philosophy entry by slug."""
        url = f"{self._SEP_BASE}/entries/{slug}/"
        try:
            r = self.session.get(url, timeout=25)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            html = r.text
            tm = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S)
            title = re.sub(r"<[^>]+>", "", tm.group(1)).strip() if tm else slug.replace("-", " ").title()

            m = re.search(r'<div[^>]*id="main-text"[^>]*>(.+?)</div>\s*<div[^>]*id="bibliography"', html, re.S)
            if not m:
                m = re.search(r'<div[^>]*id="aueditable"[^>]*>(.+?)</div>', html, re.S)
            body = m.group(1) if m else html
            body = re.sub(r"<script[^>]*>.*?</script>", " ", body, flags=re.S)
            body = re.sub(r"<style[^>]*>.*?</style>", " ", body, flags=re.S)
            text = re.sub(r"<[^>]+>", " ", body)
            text = re.sub(r"&[a-zA-Z]+;", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 200:
                return None
            return {
                "title": title,
                "summary": text[:4000],
                "categories": json.dumps(["philosophy", "sep"]),
                "url": url,
                "word_count": len(text.split()),
                "source": "stanford_encyclopedia_of_philosophy",
                "domain": "philosophy",
            }
        except Exception as e:
            logger.debug("SEP fetch failed for %r: %s", slug, e)
            return None

    # ── Scholarpedia ─────────────────────────────────────────────
    _SP_INDEX_CACHE: Path = Path("data/scholarpedia_index.json")
    _SP_BASE = "http://www.scholarpedia.org"
    _SP_SUB_ENCYCLOPEDIAS = [
        "Encyclopedia:Astrophysics",
        "Encyclopedia:Celestial_Mechanics",
        "Encyclopedia:Computational_intelligence",
        "Encyclopedia:Computational_neuroscience",
        "Encyclopedia:Dynamical_systems",
        "Encyclopedia:Physics",
        "Encyclopedia:Touch",
    ]

    def _load_scholarpedia_index(self) -> list:
        """Aggregate article slugs across Scholarpedia's sub-encyclopedias."""
        try:
            if self._SP_INDEX_CACHE.exists():
                import json as _j
                return _j.loads(self._SP_INDEX_CACHE.read_text())
        except Exception:
            pass
        slugs: list = []
        for enc in self._SP_SUB_ENCYCLOPEDIAS:
            try:
                r = self.session.get(f"{self._SP_BASE}/article/{enc}", timeout=20)
                if not r.ok:
                    continue
                found = re.findall(r'href="/article/([^"#?]+)"', r.text)
                for s in found:
                    if ":" in s:
                        continue  # skip namespaced (Category:, Help:, etc.)
                    if s in ("Main_Page",):
                        continue
                    slugs.append(s)
            except Exception as e:
                logger.debug("Scholarpedia index fetch failed for %s: %s", enc, e)
        # dedupe
        slugs = sorted(set(slugs))
        try:
            self._SP_INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
            import json as _j
            self._SP_INDEX_CACHE.write_text(_j.dumps(slugs))
        except Exception:
            pass
        logger.info("Scholarpedia index loaded: %d articles", len(slugs))
        return slugs

    def _fetch_scholarpedia_entry(self, slug: str) -> Optional[dict]:
        url = f"{self._SP_BASE}/article/{slug}"
        try:
            r = self.session.get(url, timeout=25)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            html = r.text
            tm = re.search(r"<h1[^>]*id=\"firstHeading\"[^>]*>(.*?)</h1>", html, re.S)
            if not tm:
                tm = re.search(r"<title>(.*?)</title>", html, re.S)
            title = re.sub(r"<[^>]+>", "", tm.group(1)).strip() if tm else slug.replace("_", " ")
            title = title.replace(" - Scholarpedia", "")

            m = re.search(r'<div[^>]*id="mw-content-text"[^>]*>(.+?)<div[^>]*class="printfooter"', html, re.S)
            if not m:
                m = re.search(r'<div[^>]*id="bodyContent"[^>]*>(.+?)<!--', html, re.S)
            body = m.group(1) if m else html
            body = re.sub(r"<script[^>]*>.*?</script>", " ", body, flags=re.S)
            body = re.sub(r"<style[^>]*>.*?</style>", " ", body, flags=re.S)
            text = re.sub(r"<[^>]+>", " ", body)
            text = re.sub(r"&[a-zA-Z]+;", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 200:
                return None
            return {
                "title": title,
                "summary": text[:4000],
                "categories": json.dumps(["scholarpedia", "peer_reviewed"]),
                "url": url,
                "word_count": len(text.split()),
                "source": "scholarpedia",
                "domain": "science",
            }
        except Exception as e:
            logger.debug("Scholarpedia fetch failed for %r: %s", slug, e)
            return None

    # ── Public API: fetch_topic dispatches across sources ──────────
    def fetch_topic(self, title: str) -> Optional[dict]:
        """Try SEP first, then Scholarpedia. No Wikipedia."""
        # Slugify a title for SEP-style URLs
        slug = title.lower().strip()
        slug = re.sub(r"[^a-z0-9 -]", "", slug)
        slug = re.sub(r"\s+", "-", slug).strip("-")
        if slug:
            d = self._fetch_sep_entry(slug)
            if d:
                return d
        # Scholarpedia uses underscores
        sp_slug = re.sub(r"\s+", "_", title.strip())
        d = self._fetch_scholarpedia_entry(sp_slug)
        if d:
            return d
        return None

    def store_topic(self, topic_data: dict) -> bool:
        """Store an encyclopaedia entry. Returns True if new."""
        try:
            conn = safe_open_kdb(str(self.db_path))
            existing = conn.execute("SELECT id FROM encyclopaedia WHERE title=?",
                                    (topic_data["title"],)).fetchone()
            if existing:
                conn.close()
                return False
            domain = topic_data.get("domain") or self._classify_encyc_domain(
                topic_data.get("title", ""), topic_data.get("summary", "")
            )
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
                 topic_data.get("source", "unknown"), _now())
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("store_topic failed for %r: %s", topic_data.get("title"), e)
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
            conn = safe_open_kdb(str(self.db_path))
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
            conn = safe_open_kdb(str(self.db_path))
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

    # ── Main run loop ─────────────────────────────────────────────────────────────────
    def run_once(self, target_new_words: int = 200, target_new_topics: int = 50):
        """Single pass: ingest a batch of vocabulary + encyclopaedia entries.

        Sources words from a large cached English word list (~370k entries from
        dwyl/english-words) advancing through it via a persistent cursor, plus
        the small curated seed list for high-signal vocabulary. Defaults to
        target 200 new words / 50 new topics per pass.
        """
        import random

        # Determine which words already exist to skip them fast
        existing_words = set()
        try:
            conn = safe_open_kdb(str(self.db_path))
            existing_words = {row[0] for row in conn.execute("SELECT word FROM vocabulary").fetchall()}
            conn.close()
        except Exception as _e:
            logger.debug("vocab dedup pre-load failed: %s", _e)

        # 1) Seed list always tried first (small, high-value)
        seed_words = list(OED_SEED_WORDS) + [w for w in LINGUISTICS_TOPICS if " " not in w]
        random.shuffle(seed_words)

        # 2) Insight-derived single-word topics
        insight_topics = self._get_insight_topics(50)
        insight_words = [t.lower() for t in insight_topics if " " not in t and len(t) > 3]

        # 3) Big pool with cursor (advance through ~370k words across passes)
        big_pool = _load_big_word_pool()
        cursor = _read_cursor()
        if big_pool:
            cursor = cursor % len(big_pool)
            # Take a window of 4000 candidates from the cursor onwards (with wrap)
            window_size = 4000
            if cursor + window_size <= len(big_pool):
                window = big_pool[cursor:cursor + window_size]
            else:
                window = big_pool[cursor:] + big_pool[:window_size - (len(big_pool) - cursor)]
            random.shuffle(window)
        else:
            window = []
            cursor = 0

        # Try seeds + insight words first, then big-pool window
        word_pool = seed_words + insight_words + window

        # Topic pool: SEP + Scholarpedia indexed slugs (not Wikipedia)
        import random as _r2
        sep_slugs = self._load_sep_index()
        sp_slugs = self._load_scholarpedia_index()
        # Convert SEP slugs to display titles for the slugify pass in fetch_topic
        sep_titles = [s.replace("-", " ").title() for s in sep_slugs]
        sp_titles = [s.replace("_", " ") for s in sp_slugs]
        topic_pool = sep_titles + sp_titles + [t for t in insight_topics if " " in t]
        _r2.shuffle(topic_pool)

        new_words = 0
        new_topics = 0
        attempted = 0

        for word in word_pool:
            if new_words >= target_new_words:
                break
            if attempted >= target_new_words * 6:  # cap network calls per pass
                break
            if word in existing_words:
                continue
            data = self.fetch_word(word)
            attempted += 1
            if data:
                # Buffer for batched insertion (INSERT OR IGNORE dedupes against
                # the table; we already skip known words via existing_words).
                self.ingest_one(data)
                new_words += 1
                existing_words.add(word)
                logger.info("VocabularyIngester: learned %r (%s)", word, data.get("part_of_speech", "?"))
                self._add_to_insights(word, data["definition"], "linguistics")
            # gentler rate-limit (Wiktionary REST has generous quotas)
            time.sleep(0.15)

        # Persist any words still buffered from this pass.
        self.flush()

        # Advance cursor by the window size so next pass moves forward
        if big_pool:
            _write_cursor((cursor + 4000) % len(big_pool))

        # Ingest encyclopaedia
        for topic in topic_pool:
            if new_topics >= target_new_topics:
                break
            data = self.fetch_topic(topic)
            if data:
                if self.store_topic(data):
                    new_topics += 1
                    logger.info("VocabularyIngester: encyclopaedia %r (%d words)",
                                topic, data.get("word_count", 0))
                    self._add_to_insights(topic, data["summary"][:200], data.get("domain", "general"))
            time.sleep(0.25)

        logger.info("VocabularyIngester pass complete: +%d words (%d attempted), +%d topics",
                    new_words, attempted, new_topics)
        return {"new_words": new_words, "new_topics": new_topics, "attempted": attempted}

    def _add_to_insights(self, concept: str, text: str, domain: str):
        """Mirror learning into the insights table so it feeds the knowledge graph."""
        try:
            conn = safe_open_kdb(str(self.db_path))
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

    def run_continuous(self, interval_mins: int = 5):
        """Run ingestion passes indefinitely."""
        logger.info("VocabularyIngester: continuous loop started (every %dm)", interval_mins)
        while True:
            try:
                result = self.run_once()
                logger.info("VocabularyIngester: pass done — %s", result)
            except Exception as e:
                logger.error("VocabularyIngester loop error: %s", e)
            time.sleep(max(60, interval_mins * 60))
