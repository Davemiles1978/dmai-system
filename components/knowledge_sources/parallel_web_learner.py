"""
DMAI Parallel Web Learner
==========================
Concurrent async URL fetcher that feeds learned content into the existing
WebCrawler URL queue and directly into SICore as insights.

Design
------
- Uses asyncio + httpx for true concurrent fetching (not threaded blocking I/O)
- Respects per-domain rate limiting (max 1 req/domain every DOMAIN_DELAY seconds)
- Extracts clean text via BeautifulSoup, discards boilerplate
- Scores each page for knowledge value before storing
- Feeds newly discovered internal links back into the queue (breadth-first)
- Saves per-page results to data/knowledge_sources/parallel_web/ as JSONL
- Thread-safe: can be started from any Flask thread via start_background()

Usage
-----
    from components.knowledge_sources.parallel_web_learner import ParallelWebLearner
    learner = ParallelWebLearner(data_path=Path("data"), si_core=si_core)
    learner.add_urls([
        ("https://arxiv.org/abs/2406.00001", "latest AI research"),
        ("https://en.wikipedia.org/wiki/Reinforcement_learning", "RL fundamentals"),
    ])
    learner.start_background()  # daemon thread, non-blocking

The learner also exposes add_url(url, reason) so the admin panel or chat
commands can dynamically inject new URLs at runtime.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

logger = logging.getLogger("dmai.parallel_web_learner")

# ── Optional deps — graceful degradation if missing ───────────────────────────
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed — ParallelWebLearner will use requests fallback")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 not installed — text extraction will be basic")

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_CONCURRENT       = 6       # simultaneous HTTP connections
DOMAIN_DELAY         = 2.0     # seconds between requests to same domain
REQUEST_TIMEOUT      = 20      # seconds per request
MAX_CONTENT_CHARS    = 8_000   # characters to keep per page (cost control)
MIN_KNOWLEDGE_SCORE  = 0.3     # discard pages scoring below this
BATCH_INTERVAL       = 900     # seconds between auto-batch runs (15 min)
MAX_QUEUE_SIZE       = 500     # cap the pending URL queue
DISCOVERED_LINK_DEPTH = 1      # how many link-hops to follow per page
MAX_LINKS_PER_PAGE   = 3       # max new links to extract per page

# Domains DMAI actively learns from — seeded on startup
SEED_DOMAINS: List[Tuple[str, str]] = [
    # AI / ML
    ("https://arxiv.org/list/cs.AI/recent",           "arXiv AI papers — latest"),
    ("https://arxiv.org/list/cs.LG/recent",           "arXiv ML papers — latest"),
    ("https://paperswithcode.com/latest",             "Papers with code — SOTA benchmarks"),
    ("https://huggingface.co/blog",                   "HuggingFace blog — model releases"),
    # General knowledge
    ("https://en.wikipedia.org/wiki/Artificial_intelligence", "AI Wikipedia"),
    ("https://en.wikipedia.org/wiki/Reinforcement_learning",  "RL Wikipedia"),
    ("https://en.wikipedia.org/wiki/Large_language_model",    "LLM Wikipedia"),
    # Coding / engineering
    ("https://realpython.com/",                       "Python best practices"),
    ("https://docs.python.org/3/whatsnew/",           "Python changelog"),
    # Business / entrepreneurship
    ("https://hbr.org/topic/technology",              "HBR Technology"),
    ("https://techcrunch.com/",                       "TechCrunch"),
    # Finance
    ("https://www.investopedia.com/",                 "Investopedia finance education"),
]

# High-value knowledge indicator phrases — used for scoring
KNOWLEDGE_INDICATORS = [
    "algorithm", "neural", "model", "training", "dataset", "research",
    "technique", "method", "system", "architecture", "framework",
    "results", "performance", "implementation", "analysis", "study",
    "theorem", "proof", "experiment", "evaluation", "benchmark",
    "revenue", "strategy", "market", "business", "invest", "finance",
    "autonomous", "agent", "reasoning", "inference", "optimization",
]

BOILERPLATE_PATTERNS = re.compile(
    r"(cookie|privacy policy|terms of service|subscribe|newsletter|"
    r"advertisement|sponsored|follow us|share this|related articles)",
    re.IGNORECASE,
)

USER_AGENT = "DMAILearner/1.0 (Educational research bot; contact: milesd040@gmail.com)"


# ─────────────────────────────────────────────────────────────────────────────
# URL queue item
# ─────────────────────────────────────────────────────────────────────────────

class URLItem:
    __slots__ = ("url", "reason", "depth", "added_at")

    def __init__(self, url: str, reason: str, depth: int = 0):
        self.url      = url
        self.reason   = reason
        self.depth    = depth
        self.added_at = datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Core learner
# ─────────────────────────────────────────────────────────────────────────────

class ParallelWebLearner:
    """
    Fetches URLs concurrently, extracts knowledge, feeds it into SICore,
    and surfaces status through the Admin Harvester panel.
    """

    def __init__(
        self,
        data_path: Path,
        si_core=None,
        web_crawler=None,           # pass existing WebCrawler if available
        seed: bool = True,          # add SEED_DOMAINS on init
    ):
        self.data_path   = Path(data_path) / "knowledge_sources" / "parallel_web"
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.si_core     = si_core
        self.web_crawler = web_crawler

        self._queue:    List[URLItem]   = []
        self._visited:  set             = self._load_visited()
        self._domain_last: Dict[str, float] = defaultdict(float)
        self._lock      = threading.Lock()
        self._stop      = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Stats
        self.pages_fetched   = 0
        self.pages_stored    = 0
        self.pages_skipped   = 0
        self.insights_created = 0
        self.last_batch_at: Optional[str] = None
        self.active = False

        if seed:
            self.add_urls(SEED_DOMAINS, depth=0)

        logger.info(
            "ParallelWebLearner initialised — %d URLs queued (httpx: %s, bs4: %s)",
            len(self._queue), HTTPX_AVAILABLE, BS4_AVAILABLE
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_url(self, url: str, reason: str = "manual", depth: int = 0):
        """Add a single URL to the learning queue (thread-safe)."""
        with self._lock:
            if len(self._queue) >= MAX_QUEUE_SIZE:
                logger.debug("Queue full (%d) — dropping %s", MAX_QUEUE_SIZE, url)
                return
            if url not in self._visited:
                self._queue.append(URLItem(url, reason, depth))
                logger.debug("Queued: %s (%s)", url, reason)

    def add_urls(self, items: List[Tuple[str, str]], depth: int = 0):
        """Add multiple (url, reason) tuples."""
        for url, reason in items:
            self.add_url(url, reason, depth)

    def start_background(self):
        """Start the learning loop as a daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.info("ParallelWebLearner already running")
            return
        self.active = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="dmai-web-learner"
        )
        self._thread.start()
        logger.info("ParallelWebLearner background thread started")

    def stop(self):
        self._stop.set()
        self.active = False

    def get_status(self) -> Dict:
        return {
            "active":          self.active,
            "queue_depth":     len(self._queue),
            "visited_count":   len(self._visited),
            "pages_fetched":   self.pages_fetched,
            "pages_stored":    self.pages_stored,
            "pages_skipped":   self.pages_skipped,
            "insights_created": self.insights_created,
            "last_batch_at":   self.last_batch_at,
            "interval_seconds": BATCH_INTERVAL,
        }

    # ── Background loop ────────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop.is_set():
            try:
                batch = self._dequeue_batch(MAX_CONCURRENT * 3)
                if batch:
                    asyncio.run(self._process_batch(batch))
                    self.last_batch_at = datetime.now(timezone.utc).isoformat()
                    logger.info(
                        "ParallelWebLearner batch done — fetched: %d, stored: %d, queue: %d",
                        self.pages_fetched, self.pages_stored, len(self._queue)
                    )
                else:
                    # Queue empty — re-seed and wait
                    self.add_urls(SEED_DOMAINS, depth=0)
            except Exception as exc:
                logger.error("ParallelWebLearner loop error: %s", exc)
            self._stop.wait(BATCH_INTERVAL)

    def _dequeue_batch(self, n: int) -> List[URLItem]:
        with self._lock:
            batch, self._queue = self._queue[:n], self._queue[n:]
        return batch

    # ── Async fetch layer ──────────────────────────────────────────────────────

    async def _process_batch(self, batch: List[URLItem]):
        sem = asyncio.Semaphore(MAX_CONCURRENT)
        tasks = [self._fetch_and_learn(item, sem) for item in batch]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_and_learn(self, item: URLItem, sem: asyncio.Semaphore):
        async with sem:
            # Per-domain rate limiting
            domain = urlparse(item.url).netloc
            wait   = DOMAIN_DELAY - (time.monotonic() - self._domain_last[domain])
            if wait > 0:
                await asyncio.sleep(wait)
            self._domain_last[domain] = time.monotonic()

            # Skip already visited
            if item.url in self._visited:
                return

            self.pages_fetched += 1
            self._visited.add(item.url)
            self._save_visited()

            result = await self._fetch(item.url)
            if result is None:
                self.pages_skipped += 1
                return

            raw_text, links = result
            clean_text = self._extract_text(raw_text)
            score      = self._score_page(clean_text, item.url)

            if score < MIN_KNOWLEDGE_SCORE:
                self.pages_skipped += 1
                logger.debug("Low-value page (%.2f): %s", score, item.url)
                return

            # Store the page
            page_data = {
                "url":        item.url,
                "reason":     item.reason,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "score":      round(score, 3),
                "depth":      item.depth,
                "text":       clean_text[:MAX_CONTENT_CHARS],
                "word_count": len(clean_text.split()),
            }
            self._save_page(page_data)
            self.pages_stored += 1

            # Create SICore insight
            self._create_insight(page_data)

            # Feed into WebCrawler's discovered_urls if available
            if self.web_crawler and hasattr(self.web_crawler, "add_url"):
                self.web_crawler.add_url(item.url, item.reason)

            # Enqueue discovered links (limited depth)
            if item.depth < DISCOVERED_LINK_DEPTH:
                count = 0
                for link in links:
                    if count >= MAX_LINKS_PER_PAGE:
                        break
                    abs_link = urljoin(item.url, link)
                    if self._is_learnable(abs_link):
                        self.add_url(abs_link, f"discovered from {item.url}", item.depth + 1)
                        count += 1

    async def _fetch(self, url: str) -> Optional[Tuple[str, List[str]]]:
        """Return (html_text, [links]) or None on failure."""
        headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
        try:
            if HTTPX_AVAILABLE:
                async with httpx.AsyncClient(
                    timeout=REQUEST_TIMEOUT,
                    follow_redirects=True,
                    headers=headers,
                ) as client:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        return None
                    ct = resp.headers.get("content-type", "")
                    if "text/html" not in ct and "text/plain" not in ct:
                        return None
                    html  = resp.text
                    links = self._extract_links(html, url)
                    return html, links
            else:
                # Fallback: blocking requests in executor
                import requests as req_lib
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(
                    None,
                    lambda: req_lib.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                )
                if resp.status_code != 200:
                    return None
                links = self._extract_links(resp.text, url)
                return resp.text, links
        except Exception as exc:
            logger.debug("Fetch error %s: %s", url, exc)
            return None

    # ── Text processing ────────────────────────────────────────────────────────

    def _extract_text(self, html: str) -> str:
        if BS4_AVAILABLE:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "form", "noscript", "iframe"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
        else:
            # Basic tag strip fallback
            text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        if not BS4_AVAILABLE:
            return []
        soup  = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                abs_url = urljoin(base_url, href)
                links.append(abs_url)
        return links

    def _score_page(self, text: str, url: str) -> float:
        """
        Score 0–1 based on knowledge indicator density.
        Short pages and boilerplate-heavy pages score low.
        """
        if len(text) < 200:
            return 0.0
        words       = text.lower().split()
        word_count  = len(words)
        if word_count < 50:
            return 0.0

        indicator_hits = sum(
            1 for word in words
            if any(ind in word for ind in KNOWLEDGE_INDICATORS)
        )
        # Density: indicator words per 100 words
        density = (indicator_hits / word_count) * 100
        score   = min(density / 5.0, 1.0)   # 5% density = score 1.0

        # Boost for high-value domains
        domain = urlparse(url).netloc
        if any(d in domain for d in ["arxiv", "wikipedia", "huggingface", "paperswithcode"]):
            score = min(score + 0.2, 1.0)

        # Penalise boilerplate-heavy pages
        boilerplate_hits = len(BOILERPLATE_PATTERNS.findall(text[:2000]))
        score -= boilerplate_hits * 0.05

        return max(score, 0.0)

    def _is_learnable(self, url: str) -> bool:
        """Filter out non-learnable URLs."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        # Skip media, archives, auth pages
        skip_exts = (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip",
                     ".mp4", ".mp3", ".exe", ".css", ".js", ".xml",
                     ".rss", ".atom", ".svg", ".ico", ".woff", ".ttf")
        if any(parsed.path.lower().endswith(ext) for ext in skip_exts):
            return False
        skip_paths = ("/login", "/signup", "/register", "/cart", "/checkout",
                      "/account", "/profile", "/logout", "/search")
        if any(parsed.path.lower().startswith(p) for p in skip_paths):
            return False
        return True

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save_page(self, data: Dict):
        ts  = datetime.now(timezone.utc).strftime("%Y%m%d")
        out = self.data_path / f"pages_{ts}.jsonl"
        try:
            with open(out, "a") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to save page: %s", exc)

    def _create_insight(self, data: Dict):
        if self.si_core is None:
            return
        try:
            snippet = data["text"][:300].replace("\n", " ")
            self.si_core.add_insight(
                insight_text=f"[WebLearner] {data['url']}: {snippet}",
                entity_type="web_page",
                entities=[urlparse(data["url"]).netloc, data["reason"]],
                relationship="learned_from",
                confidence=min(data["score"] + 0.1, 0.95),
                source_topic="parallel_web_learning",
                target_topic="general_knowledge",
                source_url=data["url"],
                source_title=data["url"],
                source_type="web_page",
            )
            self.insights_created += 1
        except Exception as exc:
            logger.debug("Insight creation failed: %s", exc)

    def _load_visited(self) -> set:
        vf = self.data_path / "visited_urls.json"
        if vf.exists():
            try:
                with open(vf) as f:
                    return set(json.load(f))
            except Exception:
                pass
        return set()

    def _save_visited(self):
        """Persist visited set every 50 new entries."""
        if len(self._visited) % 50 != 0:
            return
        vf = self.data_path / "visited_urls.json"
        try:
            tmp = vf.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(list(self._visited), f)
            tmp.replace(vf)
        except Exception as exc:
            logger.debug("Could not save visited list: %s", exc)
