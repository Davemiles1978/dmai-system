#!/usr/bin/env python3
"""
Core Knowledge Sources - DMAI's 8 Continuous Learning Streams
As specified in the DMAI System Specification:

CRITICAL: DMAI must distinguish between fiction and non-fiction:
- Fiction: For vocabulary, natural language patterns, narrative structure ONLY
- Non-fiction: For factual knowledge, education, research

DMAI can autonomously add new sources (books, authors, websites) she finds interesting.
COMPLETE TERRY PRATCHETT BIBLIOGRAPHY INCLUDED:
- All 41 Discworld novels
- The Long Earth series (with Stephen Baxter)
- Good Omens (with Neil Gaiman)
- The Bromeliad trilogy (Truckers, Diggers, Wings)
- The Johnny Maxwell trilogy
- The Science of Discworld series
- Standalone novels (Nation, Dodger, The Carpet People, etc.)
- Short story collections
"""

import os
import sys
import json
import time
import threading
import logging
import requests
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import feedparser
from bs4 import BeautifulSoup

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Classification of content type"""
    FICTION = "fiction"
    NON_FICTION = "non_fiction"
    EDUCATIONAL = "educational"
    RESEARCH = "research"
    NEWS = "news"
    SOCIAL = "social"
    UNKNOWN = "unknown"


class ContentPurpose(Enum):
    """Purpose for which content should be used"""
    LANGUAGE_LEARNING = "language_learning"      # Vocabulary, grammar, narrative
    FACTUAL_KNOWLEDGE = "factual_knowledge"      # Real-world knowledge
    PATTERN_RECOGNITION = "pattern_recognition"  # Trends, patterns, structures
    RESEARCH = "research"                         # Academic/technical research
    MONITORING = "monitoring"                    # Security/threat monitoring
    EVOLUTION = "evolution"                       # Self-improvement tracking


class KnowledgeItem:
    """Represents a piece of knowledge with classification"""
    
    def __init__(self, title: str, content: str, source: str, 
                 content_type: ContentType, purpose: ContentPurpose,
                 metadata: Dict = None):
        self.title = title
        self.content = content[:5000]  # Limit size
        self.source = source
        self.content_type = content_type
        self.purpose = purpose
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.is_fiction = (content_type == ContentType.FICTION)
        
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'content': self.content,
            'source': self.source,
            'content_type': self.content_type.value,
            'purpose': self.purpose.value,
            'is_fiction': self.is_fiction,
            'warning': self._get_warning() if self.is_fiction else None,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
        
    def _get_warning(self) -> str:
        """Return warning for fictional content"""
        return "⚠️ FICTIONAL CONTENT - For language learning only. Not to be interpreted as fact."


class BookReader:
    """Reads books from Project Gutenberg and public domain sources with fiction/non-fiction classification"""
    
    def __init__(self, data_path: Path, si_core=None):
        self.si_core = si_core
        self.data_path = data_path / 'books'
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.interval = 3600  # 1 hour
        self.active = False
        self.books_processed = 0
        self.last_run = None
        self.discovered_sources = self._load_discovered_sources()
        
        # Fiction keywords (for classification)
        self.fiction_keywords = [
            'novel', 'story', 'tale', 'fiction', 'fantasy', 'sci-fi', 'science fiction',
            'mystery', 'thriller', 'romance', 'adventure', 'drama', 'poetry', 'poem',
            'fairy tale', 'myth', 'legend', 'fable', 'parable', 'discworld', 'satire'
        ]
        
        # Non-fiction keywords
        self.nonfiction_keywords = [
            'history', 'science', 'technology', 'mathematics', 'physics', 'biology',
            'philosophy', 'psychology', 'economics', 'business', 'self-help', 'guide',
            'manual', 'textbook', 'tutorial', 'research', 'study', 'analysis',
            'artificial intelligence', 'machine learning', 'programming', 'code'
        ]
        
        # Known fiction authors (for classification)
        self.fiction_authors = {
            'shakespeare', 'dickens', 'austen', 'twain', 'hugo', 'tolstoy', 
            'dostoevsky', 'kafka', 'orwell', 'hemingway', 'fitzgerald', 
            'steinbeck', 'wilde', 'poe', 'lovecraft', 'asimov', 'clarke',
            'heinlein', 'verne', 'wells', 'tolkien', 'rowling', 'king',
            # Terry Pratchett - All works are fiction
            'pratchett', 'terry pratchett', 'sir terry pratchett',
            # Collaborators
            'gaiman', 'neil gaiman', 'baxter', 'stephen baxter'
        }
        
        # Known non-fiction authors
        self.nonfiction_authors = {
            'hawking', 'dawkins', 'sagan', 'tyson', 'feynman', 'einstein',
            'darwin', 'newton', 'curie', 'tesla', 'wozniak', 'torvalds',
            'knuth', 'ritchie', 'stallman', 'page', 'brin', 'musk'
        }
        
        # ====================================================================
        # COMPLETE TERRY PRATCHETT BIBLIOGRAPHY
        # ====================================================================
        
        # Discworld Series (41 books)
        self.discworld_books = [
            "The Colour of Magic", "The Light Fantastic", "Equal Rites", "Mort",
            "Sourcery", "Wyrd Sisters", "Pyramids", "Guards! Guards!",
            "Eric", "Moving Pictures", "Reaper Man", "Witches Abroad",
            "Small Gods", "Lords and Ladies", "Men at Arms", "Soul Music",
            "Interesting Times", "Maskerade", "Feet of Clay", "Hogfather",
            "Jingo", "The Last Continent", "Carpe Jugulum", "The Fifth Elephant",
            "The Truth", "Thief of Time", "The Last Hero", "The Amazing Maurice and His Educated Rodents",
            "Night Watch", "The Wee Free Men", "Monstrous Regiment", "A Hat Full of Sky",
            "Going Postal", "Thud!", "Wintersmith", "Making Money",
            "Unseen Academicals", "I Shall Wear Midnight", "Snuff", "Raising Steam",
            "The Shepherd's Crown"
        ]
        
        # The Long Earth Series (with Stephen Baxter)
        self.long_earth_series = [
            "The Long Earth",
            "The Long War",
            "The Long Mars",
            "The Long Utopia",
            "The Long Cosmos"
        ]
        
        # Good Omens (with Neil Gaiman)
        self.good_omens = ["Good Omens: The Nice and Accurate Prophecies of Agnes Nutter, Witch"]
        
        # The Bromeliad / Nome Trilogy
        self.bromeliad_trilogy = [
            "Truckers",
            "Diggers",
            "Wings"
        ]
        
        # The Johnny Maxwell Trilogy
        self.johnny_maxwell_trilogy = [
            "Only You Can Save Mankind",
            "Johnny and the Dead",
            "Johnny and the Bomb"
        ]
        
        # The Science of Discworld Series
        self.science_of_discworld = [
            "The Science of Discworld",
            "The Science of Discworld II: The Globe",
            "The Science of Discworld III: Darwin's Watch",
            "The Science of Discworld IV: Judgement Day"
        ]
        
        # Standalone Novels
        self.pratchett_standalones = [
            "The Carpet People",
            "The Dark Side of the Sun",
            "Strata",
            "Nation",
            "Dodger",
            "The Unadulterated Cat",
            "The Illustrated Eric"
        ]
        
        # Short Story Collections
        self.pratchett_short_stories = [
            "Dragons at Crumbling Castle",
            "The Witch's Vacuum Cleaner",
            "Father Christmas's Fake Beard",
            "The Time-travelling Caveman",
            "A Blink of the Screen",
            "The Wit and Wisdom of Discworld",
            "Once More with Footnotes"
        ]
        
        # Combine all Pratchett books
        self.all_pratchett_books = (
            self.discworld_books +
            self.long_earth_series +
            self.good_omens +
            self.bromeliad_trilogy +
            self.johnny_maxwell_trilogy +
            self.science_of_discworld +
            self.pratchett_standalones +
            self.pratchett_short_stories
        )
        
        # Remove duplicates
        self.all_pratchett_books = list(set(self.all_pratchett_books))
        
        # Create mapping of books to their series
        self.pratchett_series_map = {}
        for book in self.discworld_books:
            self.pratchett_series_map[book] = "Discworld"
        for book in self.long_earth_series:
            self.pratchett_series_map[book] = "The Long Earth Series"
        for book in self.good_omens:
            self.pratchett_series_map[book] = "Good Omens (with Neil Gaiman)"
        for book in self.bromeliad_trilogy:
            self.pratchett_series_map[book] = "The Bromeliad Trilogy"
        for book in self.johnny_maxwell_trilogy:
            self.pratchett_series_map[book] = "Johnny Maxwell Trilogy"
        for book in self.science_of_discworld:
            self.pratchett_series_map[book] = "The Science of Discworld"
        for book in self.pratchett_standalones:
            self.pratchett_series_map[book] = "Standalone"
        for book in self.pratchett_short_stories:
            self.pratchett_series_map[book] = "Short Story Collection"
        
    def _load_discovered_sources(self) -> Dict:
        """Load DMAI-discovered sources (authors, books she found interesting)"""
        source_file = self.data_path / 'discovered_sources.json'
        if source_file.exists():
            try:
                with open(source_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'authors': [], 'books': [], 'genres': [], 'series': []}
        
    def _save_discovered_sources(self):
        """Save discovered sources"""
        with open(self.data_path / 'discovered_sources.json', 'w') as f:
            json.dump(self.discovered_sources, f, indent=2)
            
    def add_author(self, author: str, reason: str):
        """DMAI can add authors she finds interesting"""
        author_lower = author.lower()
        if author_lower not in [a.get('name', '').lower() for a in self.discovered_sources['authors']]:
            self.discovered_sources['authors'].append({
                'name': author,
                'reason': reason,
                'added_at': datetime.now().isoformat()
            })
            self._save_discovered_sources()
            logger.info(f"📚 DMAI added author: {author} - {reason}")
            
            # If it's Terry Pratchett, add ALL his books
            if 'pratchett' in author_lower:
                self._add_all_pratchett_books()
                
            # If it's Neil Gaiman, add Good Omens
            if 'gaiman' in author_lower:
                self._add_good_omens()
                
            # If it's Stephen Baxter, add The Long Earth series
            if 'baxter' in author_lower:
                self._add_long_earth_series()
                
    def _add_all_pratchett_books(self):
        """Add all Terry Pratchett books to discovered books"""
        for book in self.all_pratchett_books:
            series = self.pratchett_series_map.get(book, "Terry Pratchett")
            self.add_book(
                book, 
                "Terry Pratchett", 
                f"Part of {series} - For language learning, humor, satire, and narrative structure"
            )
        
        # Add series information
        self.discovered_sources['series'].append({
            'name': 'Discworld',
            'author': 'Terry Pratchett',
            'books': self.discworld_books,
            'book_count': len(self.discworld_books),
            'added_at': datetime.now().isoformat()
        })
        self.discovered_sources['series'].append({
            'name': 'The Long Earth Series',
            'author': 'Terry Pratchett & Stephen Baxter',
            'books': self.long_earth_series,
            'book_count': len(self.long_earth_series),
            'added_at': datetime.now().isoformat()
        })
        self.discovered_sources['series'].append({
            'name': 'The Bromeliad Trilogy',
            'author': 'Terry Pratchett',
            'books': self.bromeliad_trilogy,
            'book_count': len(self.bromeliad_trilogy),
            'added_at': datetime.now().isoformat()
        })
        self.discovered_sources['series'].append({
            'name': 'Johnny Maxwell Trilogy',
            'author': 'Terry Pratchett',
            'books': self.johnny_maxwell_trilogy,
            'book_count': len(self.johnny_maxwell_trilogy),
            'added_at': datetime.now().isoformat()
        })
        
        self._save_discovered_sources()
        logger.info(f"📚 Added complete Terry Pratchett bibliography ({len(self.all_pratchett_books)} books across multiple series)")
        
    def _add_good_omens(self):
        """Add Good Omens to discovered books"""
        for book in self.good_omens:
            self.add_book(book, "Terry Pratchett & Neil Gaiman", 
                         "Collaborative novel - For language learning, humor, and narrative structure")
            
    def _add_long_earth_series(self):
        """Add The Long Earth series to discovered books"""
        for book in self.long_earth_series:
            self.add_book(book, "Terry Pratchett & Stephen Baxter", 
                         "Science fiction series - For language learning and speculative narrative")
            
    def add_book(self, title: str, author: str, reason: str):
        """DMAI can add specific books she finds interesting"""
        # Check if already added
        for book in self.discovered_sources['books']:
            if book['title'].lower() == title.lower() and book['author'].lower() == author.lower():
                return
                
        self.discovered_sources['books'].append({
            'title': title,
            'author': author,
            'reason': reason,
            'added_at': datetime.now().isoformat()
        })
        self._save_discovered_sources()
        logger.info(f"📚 DMAI added book: {title} by {author} - {reason}")
        
    def _classify_book(self, title: str, author: str, description: str = "") -> Tuple[ContentType, ContentPurpose]:
        """Classify book as fiction or non-fiction"""
        title_lower = title.lower()
        author_lower = author.lower()
        desc_lower = description.lower()
        
        # Check if it's any Terry Pratchett book (all fiction)
        if 'pratchett' in author_lower:
            return ContentType.FICTION, ContentPurpose.LANGUAGE_LEARNING
            
        # Check if it's any Discworld book
        if any(dw_book.lower() in title_lower for dw_book in self.discworld_books):
            return ContentType.FICTION, ContentPurpose.LANGUAGE_LEARNING
            
        # Check fiction keywords
        for keyword in self.fiction_keywords:
            if keyword in title_lower or keyword in desc_lower:
                return ContentType.FICTION, ContentPurpose.LANGUAGE_LEARNING
                
        # Check non-fiction keywords
        for keyword in self.nonfiction_keywords:
            if keyword in title_lower or keyword in desc_lower:
                return ContentType.NON_FICTION, ContentPurpose.FACTUAL_KNOWLEDGE
                
        # Check author classification
        for fiction_author in self.fiction_authors:
            if fiction_author in author_lower:
                return ContentType.FICTION, ContentPurpose.LANGUAGE_LEARNING
        for nonfiction_author in self.nonfiction_authors:
            if nonfiction_author in author_lower:
                return ContentType.NON_FICTION, ContentPurpose.FACTUAL_KNOWLEDGE
                
        # Check discovered sources
        for discovered_author in self.discovered_sources.get('authors', []):
            if discovered_author['name'].lower() in author_lower:
                return ContentType.NON_FICTION, ContentPurpose.FACTUAL_KNOWLEDGE
                
        # Default to unknown - will be reviewed by DMAI
        return ContentType.UNKNOWN, ContentPurpose.PATTERN_RECOGNITION
        
    def start(self):
        """Start continuous book reading"""
        self.active = True
        threading.Thread(target=self._run, daemon=True).start()
        logger.info("📚 Book Reader started (with fiction/non-fiction classification)")
        
    def _run(self):
        while self.active:
            try:
                self._read_books()
                self._read_discovered_books()
                self.last_run = datetime.now()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Book Reader error: {e}")
                time.sleep(60)
                
    def _read_books(self):
        """Fetch and process books from Project Gutenberg"""
        try:
            url = "https://www.gutenberg.org/ebooks/search/?sort_order=release_date"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                books = soup.find_all('li', class_='booklink')
                
                for book in books[:5]:
                    title_elem = book.find('span', class_='title')
                    title = title_elem.text.strip() if title_elem else "Unknown"
                    
                    author_elem = book.find('span', class_='subtitle')
                    author = author_elem.text.strip().replace('by ', '') if author_elem else "Unknown"
                    
                    # Classify the book
                    content_type, purpose = self._classify_book(title, author)
                    
                    book_info = KnowledgeItem(
                        title=title,
                        content=f"Book: {title} by {author}",
                        source='Project Gutenberg',
                        content_type=content_type,
                        purpose=purpose,
                        metadata={'author': author, 'book_id': hash(title)}
                    )
                    
                    self._save_book(book_info)
                    self.books_processed += 1
                    
            logger.info(f"📚 Book Reader: Processed {self.books_processed} books total")
            
        except Exception as e:
            logger.error(f"Book fetch error: {e}")
            
    def _read_discovered_books(self):
        """Read books that DMAI discovered herself (including Terry Pratchett)"""
        for book in self.discovered_sources.get('books', []):
            # Determine if this is Pratchett
            is_pratchett = 'pratchett' in book['author'].lower()
            
            # Create knowledge item for discovered book
            book_info = KnowledgeItem(
                title=book['title'],
                content=f"Discovered book: {book['title']} by {book['author']}. Reason: {book['reason']}",
                source='DMAI_Discovery',
                content_type=ContentType.FICTION if is_pratchett else ContentType.UNKNOWN,
                purpose=ContentPurpose.LANGUAGE_LEARNING if is_pratchett else ContentPurpose.PATTERN_RECOGNITION,
                metadata={
                    'discovery_reason': book['reason'], 
                    'author': book['author'],
                    'series': self.pratchett_series_map.get(book['title'], 'Unknown')
                }
            )
            self._save_book(book_info)
            if is_pratchett:
                logger.debug(f"Reading Terry Pratchett book: {book['title']} - {book['reason']}")
            
    def _save_book(self, book: KnowledgeItem):
        """Save book info to disk with classification and create SI Core insight"""
        filename = self.data_path / f"book_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(book.to_dict(), f, indent=2)
        
        # Create insight in SI Core
        if self.si_core:
            try:
                author = book.metadata.get('author', 'Unknown')
                self.si_core.add_insight(
                    insight_text=f"Book: {book.title} by {author}",
                    entity_type="book",
                    entities=[author, book.title],
                    relationship="wrote",
                    confidence=0.9,
                    source_topic="literature",
                    target_topic="knowledge",
                    source_url=book.source,
                    source_title=book.title,
                    source_type="book_reader"
                )
            except Exception as e:
                logger.error(f"Failed to create insight for book: {e}")
            
    def get_status(self) -> Dict:
        return {
            'active': self.active,
            'books_processed': self.books_processed,
            'discovered_authors': len(self.discovered_sources.get('authors', [])),
            'discovered_books': len(self.discovered_sources.get('books', [])),
            'pratchett_books_added': len(self.all_pratchett_books),
            'pratchett_series': [s['name'] for s in self.discovered_sources.get('series', [])],
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'interval': self.interval
        }


class ArticleReader:
    """Reads news, technical articles, and blogs with classification"""
    
    def __init__(self, data_path: Path, si_core=None):
        self.si_core = si_core
        self.data_path = data_path / 'articles'
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.interval = 1800  # 30 minutes
        self.active = False
        self.articles_processed = 0
        self.last_run = None
        self.discovered_sources = self._load_discovered_sources()
        
        # RSS feeds to monitor (educational/technical)
        self.rss_feeds = [
            'https://news.ycombinator.com/rss',
            'https://feeds.feedburner.com/TechCrunch',
            'https://www.wired.com/feed/rss',
            'https://arxiv.org/rss/cs.AI',
            'https://medium.com/feed/tag/artificial-intelligence',
            'https://towardsdatascience.com/feed',
            'https://machinelearningmastery.com/feed/',
            'https://openai.com/news/rss',
            'https://ai.googleblog.com/atom.xml',
            'https://deepmind.com/blog/feed',
            'https://www.anthropic.com/news/rss',
        ]
        
    def _load_discovered_sources(self) -> Dict:
        """Load DMAI-discovered sources"""
        source_file = self.data_path / 'discovered_sources.json'
        if source_file.exists():
            try:
                with open(source_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'feeds': [], 'topics': []}
        
    def _save_discovered_sources(self):
        """Save discovered sources"""
        with open(self.data_path / 'discovered_sources.json', 'w') as f:
            json.dump(self.discovered_sources, f, indent=2)
            
    def add_feed(self, url: str, reason: str):
        """DMAI can add RSS feeds she finds interesting"""
        if url not in [f.get('url') for f in self.discovered_sources['feeds']]:
            self.discovered_sources['feeds'].append({
                'url': url,
                'reason': reason,
                'added_at': datetime.now().isoformat()
            })
            self.rss_feeds.append(url)
            self._save_discovered_sources()
            logger.info(f"📰 DMAI added RSS feed: {url} - {reason}")
            
    def _classify_article(self, title: str, summary: str) -> Tuple[ContentType, ContentPurpose]:
        """Classify article type"""
        text = (title + " " + summary).lower()
        
        if 'arxiv' in text or 'paper' in text or 'research' in text:
            return ContentType.RESEARCH, ContentPurpose.RESEARCH
        if 'tutorial' in text or 'guide' in text or 'how to' in text:
            return ContentType.EDUCATIONAL, ContentPurpose.FACTUAL_KNOWLEDGE
        if 'news' in text or 'announce' in text or 'release' in text:
            return ContentType.NEWS, ContentPurpose.FACTUAL_KNOWLEDGE
            
        return ContentType.NON_FICTION, ContentPurpose.FACTUAL_KNOWLEDGE
        
    def start(self):
        """Start continuous article reading"""
        self.active = True
        threading.Thread(target=self._run, daemon=True).start()
        logger.info("📰 Article Reader started (with classification)")
        
    def _run(self):
        while self.active:
            try:
                self._read_articles()
                self._read_discovered_feeds()
                self.last_run = datetime.now()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Article Reader error: {e}")
                time.sleep(60)
                
    def _read_articles(self):
        """Fetch and process articles from RSS feeds"""
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:5]:
                    content_type, purpose = self._classify_article(
                        entry.get('title', ''),
                        entry.get('summary', '')
                    )
                    
                    article = KnowledgeItem(
                        title=entry.get('title', 'Unknown'),
                        content=entry.get('summary', '')[:500],
                        source=feed_url,
                        content_type=content_type,
                        purpose=purpose,
                        metadata={
                            'link': entry.get('link', ''),
                            'published': entry.get('published', '')
                        }
                    )
                    
                    self._save_article(article)
                    self.articles_processed += 1
                    
            except Exception as e:
                logger.error(f"Feed error for {feed_url}: {e}")
                
        logger.info(f"📰 Article Reader: Processed {self.articles_processed} articles total")
        
    def _read_discovered_feeds(self):
        """Read feeds that DMAI discovered"""
        for feed in self.discovered_sources.get('feeds', []):
            logger.debug(f"Reading discovered feed: {feed['url']} - {feed['reason']}")
            
    def _save_article(self, article: KnowledgeItem):
        """Save article to disk and create SI Core insight with proper macro/micro hierarchy"""
        filename = self.data_path / f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(article.to_dict(), f, indent=2)
        
        # Create insight in SI Core
        if self.si_core:
            try:
                title = article.title[:150] if article.title else "Unknown"
                content = article.content[:1000] if article.content else ""
                text_lower = (title + " " + content).lower()
                
                # ============================================================
                # DYNAMIC TOPIC DETECTION - 13+ Categories with expansion capability
                # ============================================================
                
                # Category keywords mapping (can be expanded dynamically)
                category_keywords = {
                    "Configuration": ["config", "settings", "setup", "configure", "parameter", "env", "environment"],
                    "Knowledge Module": ["knowledge", "learning", "education", "training", "curriculum"],
                    "AI Model": ["model", "neural", "transformer", "llm", "gpt", "bert", "diffusion", "ai ", "artificial intelligence"],
                    "Capability": ["capability", "feature", "function", "ability", "skill"],
                    "Data Structure": ["data", "database", "storage", "schema", "json", "sql", "nosql"],
                    "Content Generation": ["content", "generate", "creation", "writing", "blog", "article"],
                    "Survival Mechanism": ["survival", "resilience", "failover", "backup", "recovery", "persistence"],
                    "Self-Funding": ["revenue", "income", "profit", "monetize", "business", "sales", "customer"],
                    "Blockchain": ["blockchain", "crypto", "bitcoin", "ethereum", "web3", "defi", "nft", "token"],
                    "API Endpoint": ["api", "endpoint", "rest", "graphql", "webhook", "integration"],
                    "Identity Management": ["identity", "auth", "authentication", "user", "profile", "account", "login"],
                    "Automation": ["automate", "automation", "workflow", "pipeline", "script", "bot"],
                    "Self-Replication": ["replicate", "replication", "clone", "spawn", "instance", "scaling"],
                    # Wealth/Finance (expanded)
                    "Trading": ["trading", "stocks", "market", "invest", "portfolio", "exchange", "broker"],
                    "Hardware": ["hardware", "cpu", "gpu", "ram", "server", "device", "component", "circuit"],
                    "Supplier Outreach": ["supplier", "vendor", "procurement", "sourcing", "purchase", "order"],
                }
                
                # Detect primary category
                detected_category = None
                max_matches = 0
                
                for category, keywords in category_keywords.items():
                    matches = sum(1 for kw in keywords if kw in text_lower)
                    if matches > max_matches:
                        max_matches = matches
                        detected_category = category
                
                # Default if nothing detected
                if detected_category is None or max_matches == 0:
                    detected_category = "Knowledge Module"
                
                # ============================================================
                # CHECK IF MACRO NEURON EXISTS FOR THIS CATEGORY
                # ============================================================
                macro_id = None
                
                # Query existing macro for this category
                import sqlite3
                db_path = self.si_core.sqlite.db_path if hasattr(self.si_core, 'sqlite') and self.si_core.sqlite else None
                
                if db_path:
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT id FROM insights 
                        WHERE neuron_level = 'macro' 
                          AND insight_text LIKE ?
                        LIMIT 1
                    ''', (f'[{detected_category}]%',))
                    row = cursor.fetchone()
                    if row:
                        macro_id = row[0]
                    conn.close()
                
                # ============================================================
                # CREATE OR UPDATE MACRO NEURON
                # ============================================================
                if macro_id is None:
                    # Create new macro neuron for this category
                    macro_id = self.si_core.add_insight(
                        insight_text=f"[{detected_category}] {detected_category} Knowledge Base: Accumulated research and insights",
                        entity_type="topic_macro",
                        entities=[detected_category, "research", "knowledge"],
                        relationship="organizes",
                        source_topic="research",
                        target_topic=detected_category.lower().replace(" ", "_"),
                        confidence=0.95,
                        source_title=f"Auto-created from article: {title[:50]}",
                        source_type="article_reader_macro",
                        neuron_level='macro',
                        is_visible_at_top_level=True
                    )
                    logger.info(f"📚 Created NEW macro neuron: {detected_category}")
                
                # ============================================================
                # CREATE MICRO NEURON (the actual article insight)
                # ============================================================
                entities = [detected_category, article.source]
                
                # Extract additional entities from title
                import re
                words = re.findall(r'\b[A-Z][a-z]{2,}\b', title)
                entities.extend(words[:3])
                entities = list(set(entities))
                
                micro_id = self.si_core.add_insight(
                    insight_text=f"Article: {title}",
                    entity_type="article_micro",
                    entities=entities,
                    relationship="researched",
                    confidence=0.75,
                    source_topic=detected_category,
                    target_topic="article_knowledge",
                    source_url=article.metadata.get('link', article.source),
                    source_title=title,
                    source_type="article_reader",
                    neuron_level='micro',
                    cluster_id=macro_id,
                    parent_macro_id=macro_id,
                    is_visible_at_top_level=False
                )
                
                if micro_id:
                    logger.info(f"📰 Created micro insight under [{detected_category}]: {title[:50]}...")
                
                # ============================================================
                # CREATE SYNAPSES TO RELATED TOPICS
                # ============================================================
                if db_path:
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    
                    # Find other macros that share entities
                    for entity in entities[:5]:
                        cursor.execute('''
                            SELECT id FROM insights 
                            WHERE neuron_level = 'macro' 
                              AND id != ?
                              AND insight_text LIKE ?
                            LIMIT 3
                        ''', (macro_id, f'%{entity}%'))
                        
                        for row in cursor.fetchall():
                            self.si_core.add_synapse(macro_id, row[0], f"related_via_{entity}")
                    
                    conn.close()
                    
            except Exception as e:
                logger.error(f"Failed to create insight for article: {e}")
                import traceback
                traceback.print_exc()
            
    def get_status(self) -> Dict:
        return {
            'active': self.active,
            'articles_processed': self.articles_processed,
            'discovered_feeds': len(self.discovered_sources.get('feeds', [])),
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'interval': self.interval,
            'feeds_monitored': len(self.rss_feeds)
        }

class ResearchPaperReader:
    """Reads research papers from ArXiv and academic journals"""
    
    def __init__(self, data_path: Path, si_core=None):
        self.si_core = si_core
        self.data_path = data_path / 'papers'
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.interval = 7200  # 2 hours
        self.active = False
        self.papers_processed = 0
        self.last_run = None
        
        self.categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'cs.RO']
        
    def start(self):
        self.active = True
        threading.Thread(target=self._run, daemon=True).start()
        logger.info("📄 Research Paper Reader started")
        
    def _run(self):
        while self.active:
            try:
                self._read_papers()
                self.last_run = datetime.now()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Paper Reader error: {e}")
                time.sleep(60)
                
    def _read_papers(self):
        for category in self.categories:
            try:
                url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&max_results=10"
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    feed = feedparser.parse(response.text)
                    
                    for entry in feed.entries:
                        paper = KnowledgeItem(
                            title=entry.get('title', 'Unknown'),
                            content=entry.get('summary', '')[:1000],
                            source='arXiv',
                            content_type=ContentType.RESEARCH,
                            purpose=ContentPurpose.RESEARCH,
                            metadata={
                                'authors': [a.name for a in entry.get('authors', [])],
                                'link': entry.get('link', ''),
                                'category': category
                            }
                        )
                        self._save_paper(paper)
                        self.papers_processed += 1
                        
            except Exception as e:
                logger.error(f"ArXiv error for {category}: {e}")
                
        logger.info(f"📄 Paper Reader: Processed {self.papers_processed} papers total")
        
def _save_paper(self, paper: KnowledgeItem):
    """Save paper to disk and create SI Core insight"""
    filename = self.data_path / f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(paper.to_dict(), f, indent=2)
    
    # Create insight in SI Core
    if self.si_core:
        try:
            title = paper.title[:100] if paper.title else "Unknown"
            content = paper.content[:500] if paper.content else ""
            authors = paper.metadata.get('authors', [])
            category = paper.metadata.get('category', 'unknown')
            
            # Build entities
            entities = ['arxiv', category] + authors[:3]
            entities = list(set([e for e in entities if e]))
            
            # Determine source category based on arXiv category
            source_category = "research"
            if category in ['cs.AI', 'cs.LG', 'cs.NE']:
                source_category = "ai_research"
            elif category in ['q-fin', 'q-fin.TR']:
                source_category = "wealth_creation"
            
            self.si_core.add_insight(
                insight_text=f"Paper: {title}",
                entity_type="research_paper",
                entities=entities,
                relationship="published",
                confidence=0.85,
                source_topic=source_category,
                target_topic="academic_knowledge",
                source_url=paper.metadata.get('link', ''),
                source_title=title,
                source_type="research_paper_reader"
            )
            logger.debug(f"Created insight for paper: {title[:50]}... (category: {category})")
        except Exception as e:
            logger.error(f"Failed to create insight for paper: {e}")
            
    def get_status(self) -> Dict:
        return {
            'active': self.active,
            'papers_processed': self.papers_processed,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'interval': self.interval
        }

class WebCrawler:
    """Crawls general web content for learning"""
    
    def __init__(self, data_path: Path, si_core=None):
        self.si_core = si_core
        self.data_path = data_path / 'web'
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.interval = 900  # 15 minutes
        self.active = False
        self.pages_crawled = 0
        self.last_run = None
        self.discovered_urls = self._load_discovered_urls()
        
        self.seed_urls = [
            'https://en.wikipedia.org/wiki/Artificial_intelligence',
            'https://en.wikipedia.org/wiki/Machine_learning',
            'https://en.wikipedia.org/wiki/Deep_learning',
            'https://en.wikipedia.org/wiki/Neural_network',
            'https://en.wikipedia.org/wiki/Natural_language_processing',
            'https://en.wikipedia.org/wiki/Computer_vision'
        ]
        
    def _load_discovered_urls(self) -> List[str]:
        url_file = self.data_path / 'discovered_urls.json'
        if url_file.exists():
            try:
                with open(url_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []
        
    def _save_discovered_urls(self):
        with open(self.data_path / 'discovered_urls.json', 'w') as f:
            json.dump(self.discovered_urls, f, indent=2)
            
    def add_url(self, url: str, reason: str):
        """DMAI can add URLs she finds interesting"""
        if url not in [u.get('url') for u in self.discovered_urls]:
            self.discovered_urls.append({'url': url, 'reason': reason, 'added_at': datetime.now().isoformat()})
            self._save_discovered_urls()
            logger.info(f"🕸️ DMAI added URL: {url} - {reason}")
            
    def start(self):
        self.active = True
        threading.Thread(target=self._run, daemon=True).start()
        logger.info("🕸️ Web Crawler started")
        
    def _run(self):
        while self.active:
            try:
                self._crawl()
                self.last_run = datetime.now()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Web Crawler error: {e}")
                time.sleep(60)
                
    def _crawl(self):
        all_urls = self.seed_urls + [u['url'] for u in self.discovered_urls]
        
        for url in all_urls[:10]:
            try:
                response = requests.get(url, timeout=30, headers={
                    'User-Agent': 'DMAI/1.0 (Educational Bot)'
                })
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    content = soup.get_text()[:5000]
                    
                    page = KnowledgeItem(
                        title=soup.title.string if soup.title else 'Unknown',
                        content=content,
                        source=url,
                        content_type=ContentType.NON_FICTION,
                        purpose=ContentPurpose.FACTUAL_KNOWLEDGE,
                        metadata={'url': url}
                    )
                    
                    self._save_page(page)
                    self.pages_crawled += 1
                    
            except Exception as e:
                logger.error(f"Crawl error for {url}: {e}")
                
        logger.info(f"🕸️ Web Crawler: Crawled {self.pages_crawled} pages total")
        
    def _save_page(self, page: KnowledgeItem):
        filename = self.data_path / f"page_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(page.to_dict(), f, indent=2)
            
    def get_status(self) -> Dict:
        return {
            'active': self.active,
            'pages_crawled': self.pages_crawled,
            'discovered_urls': len(self.discovered_urls),
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'interval': self.interval
        }

class DarkWebMonitor:
    """Monitors dark web for intelligence (requires Tor)"""
    
    def __init__(self, data_path: Path, si_core=None):
        self.si_core = si_core
        self.data_path = data_path / 'darkweb'
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.interval = 3600  # 1 hour
        self.active = False
        self.intel_collected = 0
        self.last_run = None
        self.tor_proxy = os.getenv('TOR_PROXY', 'socks5://127.0.0.1:9050')
        self.tor_available = False
        self.onion_sites = []
        
    def add_onion_site(self, url: str):
        if url not in self.onion_sites:
            self.onion_sites.append(url)
            logger.info(f"🌑 Added onion site: {url}")
            
    def start(self):
        self.active = True
        threading.Thread(target=self._run, daemon=True).start()
        logger.info("🌑 Dark Web Monitor started (requires Tor)")
        
    def _run(self):
        while self.active:
            try:
                self._monitor()
                self.last_run = datetime.now()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Dark Web Monitor error: {e}")
                time.sleep(300)
                
    def _monitor(self):
        self._check_tor()
        
        if not self.tor_available:
            return
            
        for site in self.onion_sites:
            try:
                intel = KnowledgeItem(
                    title=f"Dark Web Monitor: {site}",
                    content="Dark web intel collected",
                    source=site,
                    content_type=ContentType.SOCIAL,
                    purpose=ContentPurpose.MONITORING,
                    metadata={'site': site, 'status': 'monitored'}
                )
                self._save_intel(intel)
                self.intel_collected += 1
                
            except Exception as e:
                logger.error(f"Dark web error for {site}: {e}")
                
    def _check_tor(self):
        try:
            proxies = {'http': self.tor_proxy, 'https': self.tor_proxy}
            response = requests.get('http://check.torproject.org', proxies=proxies, timeout=10)
            self.tor_available = 'Congratulations' in response.text
        except:
            self.tor_available = False
            
    def _save_intel(self, intel: KnowledgeItem):
        filename = self.data_path / f"intel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(intel.to_dict(), f, indent=2)
            
    def get_status(self) -> Dict:
        return {
            'active': self.active,
            'tor_available': self.tor_available,
            'intel_collected': self.intel_collected,
            'sites_monitored': len(self.onion_sites),
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'interval': self.interval
        }

class SocialMediaScanner:
    """Scans TikTok, Instagram, YouTube, Twitter, Reddit, Discord for trends and video content"""
    
    def __init__(self, data_path: Path, si_core=None):
        self.si_core = si_core
        self.data_path = data_path / 'social'
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.interval = 600  # 10 minutes
        self.active = False
        self.posts_scanned = 0
        self.videos_analyzed = 0
        self.last_run = None
        self.keywords = ['ai', 'machine learning', 'deep learning', 'llm', 'gpt', 'agi', 
                        'trending', 'viral', 'artificial intelligence', 'neural network']
        
        # Video platform tracking
        self.video_platforms = {
            'tiktok': {'trending': [], 'videos_analyzed': 0, 'hashtags': []},
            'instagram': {'reels': [], 'videos_analyzed': 0, 'hashtags': []},
            'youtube': {'trending': [], 'videos_analyzed': 0, 'categories': []}
        }

        # TikTok trending URLs (can be populated manually or via discovery)
        self.tiktok_trending_urls = []
        
    def start(self):
        self.active = True
        threading.Thread(target=self._run, daemon=True).start()
        logger.info("📱 Social Media Scanner started (TikTok, Instagram, YouTube, Reddit)")
        
    def _run(self):
        while self.active:
            try:
                self._scan_reddit()
                self._scan_video_platforms()
                self.last_run = datetime.now()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Social Media Scanner error: {e}")
                time.sleep(60)
                
    def _scan_reddit(self):
        """Scan Reddit for AI/ML discussions"""
        try:
            response = requests.get(
                'https://www.reddit.com/r/MachineLearning/new.json',
                headers={'User-Agent': 'DMAI/1.0'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                for post in data.get('data', {}).get('children', [])[:20]:
                    post_data = post.get('data', {})
                    title = post_data.get('title', '')
                    
                    if any(kw in title.lower() for kw in self.keywords):
                        post_item = KnowledgeItem(
                            title=title,
                            content=post_data.get('selftext', '')[:500],
                            source='reddit',
                            content_type=ContentType.SOCIAL,
                            purpose=ContentPurpose.PATTERN_RECOGNITION,
                            metadata={
                                'url': post_data.get('url', ''),
                                'score': post_data.get('score', 0),
                                'subreddit': 'MachineLearning'
                            }
                        )
                        self._save_post(post_item)
                        self.posts_scanned += 1
                        
        except Exception as e:
            logger.error(f"Reddit scan error: {e}")
         
    def _get_youtube_trending_videos(self) -> List[Dict]:
        """Get trending YouTube videos about AI/tech"""
        videos = []
        try:
            feed_url = "https://www.youtube.com/feeds/videos.xml?channel_id=UCbfYPyITQ-7l4upoX8nvctg"
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:10]:
                title = entry.get('title', '')
                if any(kw in title.lower() for kw in self.keywords):
                    videos.append({
                        'title': title,
                        'url': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'platform': 'youtube'
                    })
        except Exception as e:
            logger.error(f"YouTube feed error: {e}")
        
        return videos
   
    def _scan_video_platforms(self):
        """Scan TikTok, Instagram, and YouTube for trending content and extract transcripts"""
        
        # YouTube - scan trending and extract transcripts
        youtube_videos = self._get_youtube_trending_videos()
        for video in youtube_videos[:3]:  # Limit to 3 per cycle
            if self.si_core:
                # Create insight first
                self.si_core.add_insight(
                    insight_text=f"YouTube Video: {video.get('title', 'Unknown')}",
                    entity_type="video_content",
                    entities=[video.get('title', ''), 'youtube'],
                    relationship="discovered",
                    confidence=0.8,
                    source_topic="social_media",
                    target_topic="video_transcript"
                )
            
            # Extract and save transcript
            transcript = self._extract_youtube_transcript(video.get('url', ''))
            if transcript:
                self._save_transcript('youtube', video.get('url', ''), transcript, video)
                self.video_platforms['youtube']['videos_analyzed'] += 1
                self.videos_analyzed += 1
        
        # TikTok - process any collected URLs
        for url in self.tiktok_trending_urls[:3]:
            transcript = self._extract_tiktok_transcript(url)
            if transcript:
                self._save_transcript('tiktok', url, transcript, {'url': url})
                self.video_platforms['tiktok']['videos_analyzed'] += 1
                self.videos_analyzed += 1
        
        # Clear processed URLs
        self.tiktok_trending_urls = self.tiktok_trending_urls[3:]
        
    def _scan_youtube_trending(self):
        """Scan YouTube trending videos for AI-related content"""
        try:
            # YouTube RSS feed for trending
            feed_url = "https://www.youtube.com/feeds/videos.xml?channel_id=UCbfYPyITQ-7l4upoX8nvctg"  # Example AI channel
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:10]:
                title = entry.get('title', '')
                if any(kw in title.lower() for kw in self.keywords):
                    video_item = KnowledgeItem(
                        title=title,
                        content=entry.get('summary', '')[:500],
                        source='youtube',
                        content_type=ContentType.SOCIAL,
                        purpose=ContentPurpose.PATTERN_RECOGNITION,
                        metadata={
                            'url': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'platform': 'youtube'
                        }
                    )
                    self._save_video_analysis(video_item)
                    self.video_platforms['youtube']['videos_analyzed'] += 1
                    self.videos_analyzed += 1
                    
                    # Create insight in SI Core
                    if self.si_core:
                        try:
                            self.si_core.add_insight(
                                insight_text=f"YouTube Trend: {title}",
                                entity_type="video_trend",
                                entities=[title, 'youtube'],
                                relationship="trending",
                                confidence=0.75,
                                source_topic="social_media",
                                target_topic="video_content",
                                source_url=entry.get('link', ''),
                                source_title=title,
                                source_type="social_media_scanner"
                            )
                        except Exception as e:
                            logger.error(f"Failed to create insight for video: {e}")
                            
        except Exception as e:
            logger.error(f"YouTube scan error: {e}")
            
    def analyze_video_content(self, video_url: str, platform: str) -> Dict:
        """Analyze video content using DMAI's video extraction capabilities"""
        # This will integrate with the video extraction capability from ingested repos
        result = {
            'url': video_url,
            'platform': platform,
            'status': 'pending',
            'transcript': None,
            'key_topics': [],
            'sentiment': None
        }
        
        # Placeholder for video extraction integration
        logger.info(f"🎬 Analyzing video from {platform}: {video_url}")
        
        return result
            
    def _save_post(self, post: KnowledgeItem):
        """Save social media post to disk"""
        filename = self.data_path / f"post_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(post.to_dict(), f, indent=2)
            
    def _save_video_analysis(self, video: KnowledgeItem):
        """Save video analysis to disk"""
        filename = self.data_path / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(video.to_dict(), f, indent=2)
      
    def _extract_youtube_transcript(self, video_url: str) -> Optional[str]:
        """Extract transcript from YouTube video"""
        try:
            # Extract video ID from URL
            import re
            video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})(?:[?&]|$)', video_url)
            if not video_id_match:
                return None
                
            video_id = video_id_match.group(1)
            
            # Try youtube-transcript-api
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                full_text = ' '.join([entry['text'] for entry in transcript_list])
                logger.info(f"📝 Extracted YouTube transcript: {len(full_text)} chars")
                return full_text
            except ImportError:
                # Fallback: use yt-dlp if available
                return self._extract_with_ytdlp(video_url)
                
        except Exception as e:
            logger.debug(f"YouTube transcript extraction failed: {e}")
            return None
            
    def _extract_tiktok_transcript(self, video_url: str) -> Optional[str]:
        """Extract transcript/captions from TikTok video"""
        try:
            # TikTok requires yt-dlp or similar tool
            return self._extract_with_ytdlp(video_url, platform='tiktok')
        except Exception as e:
            logger.debug(f"TikTok transcript extraction failed: {e}")
            return None
            
    def _extract_instagram_transcript(self, video_url: str) -> Optional[str]:
        """Extract captions from Instagram Reel/Video"""
        try:
            return self._extract_with_ytdlp(video_url, platform='instagram')
        except Exception as e:
            logger.debug(f"Instagram transcript extraction failed: {e}")
            return None
            
    def _extract_with_ytdlp(self, video_url: str, platform: str = 'youtube') -> Optional[str]:
        """Use yt-dlp to download subtitles/transcript"""
        import tempfile
        import subprocess
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Download subtitles only
                cmd = [
                    'yt-dlp',
                    '--skip-download',
                    '--write-subs',
                    '--write-auto-subs',
                    '--sub-lang', 'en',
                    '--sub-format', 'vtt',
                    '--output', f'{tmpdir}/%(id)s',
                    video_url
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                # Find and read subtitle file
                import glob
                subtitle_files = glob.glob(f'{tmpdir}/*.en.vtt') + glob.glob(f'{tmpdir}/*.vtt')
                
                if subtitle_files:
                    with open(subtitle_files[0], 'r') as f:
                        content = f.read()
                        # Parse VTT to plain text
                        text = self._parse_vtt(content)
                        logger.info(f"📝 Extracted {platform} transcript: {len(text)} chars")
                        return text
                        
        except Exception as e:
            logger.debug(f"yt-dlp extraction failed: {e}")
            
        return None
        
    def _parse_vtt(self, vtt_content: str) -> str:
        """Parse VTT subtitle format to plain text"""
        import re
        
        lines = vtt_content.split('\n')
        text_lines = []
        
        for line in lines:
            # Skip timestamps and metadata
            if '-->' in line or line.strip().isdigit() or line.strip() == 'WEBVTT':
                continue
            # Remove HTML tags
            line = re.sub(r'<[^>]+>', '', line)
            if line.strip():
                text_lines.append(line.strip())
                
        return ' '.join(text_lines)
        
    def _save_transcript(self, platform: str, video_url: str, transcript: str, metadata: Dict = None):
        """Save extracted transcript for SpeechPatternAnalyzer"""
        transcript_path = self.data_path.parent / 'transcripts'
        transcript_path.mkdir(parents=True, exist_ok=True)
        
        filename = transcript_path / f"{platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'platform': platform,
            'url': video_url,
            'transcript': transcript,
            'extracted_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"💾 Saved {platform} transcript: {filename}")
        
        # Also save as plain text for easy reading
        txt_filename = transcript_path / f"{platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(txt_filename, 'w') as f:
            f.write(f"Platform: {platform}\n")
            f.write(f"URL: {video_url}\n")
            f.write(f"Extracted: {datetime.now().isoformat()}\n")
            f.write("-" * 40 + "\n")
            f.write(transcript)
      
    def get_status(self) -> Dict:
        return {
            'active': self.active,
            'posts_scanned': self.posts_scanned,
            'videos_analyzed': self.videos_analyzed,
            'platforms': {
                'youtube': self.video_platforms['youtube']['videos_analyzed'],
                'tiktok': self.video_platforms['tiktok']['videos_analyzed'],
                'instagram': self.video_platforms['instagram']['videos_analyzed']
            },
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'interval': self.interval
        }

class SpeechPatternAnalyzer:
    """
    Analyzes real conversation patterns from social media, forums, and transcripts.
    Learns authentic speech patterns for Alex Riviera (age 28, female).
    Creates SI Core neurons for learned patterns.
    """
    
    def __init__(self, data_path: Path, si_core=None):
        self.si_core = si_core
        self.data_path = data_path / 'speech'
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.interval = 600  # 10 minutes
        self.active = False
        self.patterns_identified = 0
        self.last_run = None
        
        # Pattern categories to learn
        self.pattern_categories = {
            'slang': [],        # "no cap", "slay", "it's giving"
            'idioms': [],       # "spill the tea", "living rent free"
            'fillers': [],      # "like", "literally", "you know"
            'emotional_cues': [], # "omg", "i can't even", "bestie"
            'conversation_starters': [],
            'reactions': [],    # "no way", "shut up", "wait really"
            'age_appropriate_phrases': []  # Millennial/Gen Z cusp (born 1998)
        }
        
        # Target age demographic (Alex: 28, female)
        self.target_demographic = {
            'age_range': (25, 35),
            'gender': 'female',
            'generation': 'zillennial',  # Cusp of Millennial/Gen Z
            'birth_year': 1998
        }
        
        # Sources to analyze (connected to SocialMediaScanner output)
        self.source_patterns = self._load_patterns()
        
    def _load_patterns(self) -> Dict:
        """Load saved patterns"""
        pattern_file = self.data_path / 'learned_patterns.json'
        if pattern_file.exists():
            try:
                with open(pattern_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return self.pattern_categories.copy()
        
    def _save_patterns(self):
        """Save learned patterns"""
        pattern_file = self.data_path / 'learned_patterns.json'
        with open(pattern_file, 'w') as f:
            json.dump(self.source_patterns, f, indent=2)
    
    def start(self):
        self.active = True
        threading.Thread(target=self._run, daemon=True).start()
        logger.info("🗣️ Speech Pattern Analyzer started (Alex Riviera, age 28)")
        
    def _run(self):
        while self.active:
            try:
                self._analyze_patterns()
                self.last_run = datetime.now()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Speech Pattern Analyzer error: {e}")
                time.sleep(120)
                
    def _analyze_patterns(self):
        """Analyze conversation data for speech patterns"""
        
        # Check for new social media content to analyze
        social_data_path = self.data_path.parent / 'social'
        if social_data_path.exists():
            self._process_social_content(social_data_path)
        
        # Analyze any transcript files (YouTube, TikTok)
        transcript_path = self.data_path.parent / 'transcripts'
        if transcript_path.exists():
            self._process_transcripts(transcript_path)
        
        # Create SI Core insights for high-value patterns
        if self.si_core:
            self._create_pattern_insights()
            
        self.patterns_identified += 1
        logger.debug(f"Speech pattern analysis cycle {self.patterns_identified} complete")
        
    def _process_social_content(self, social_path: Path):
        """Extract speech patterns from social media content"""
        for file in social_path.glob('*.json'):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    content = data.get('content', '')
                    platform = data.get('platform', 'unknown')
                    
                    # Extract patterns based on platform
                    if platform in ['twitter', 'tiktok', 'instagram', 'reddit']:
                        self._extract_patterns_from_text(content, platform)
                        
            except Exception as e:
                logger.debug(f"Error processing social file {file}: {e}")
                
    def _process_transcripts(self, transcript_path: Path):
        """Extract speech patterns from video transcripts"""
        for file in transcript_path.glob('*.txt'):
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    self._extract_patterns_from_text(content, 'transcript')
            except Exception as e:
                logger.debug(f"Error processing transcript {file}: {e}")
                
    def _extract_patterns_from_text(self, text: str, source_type: str):
        """Extract speech patterns from raw text"""
        if not text or len(text) < 10:
            return
            
        text_lower = text.lower()
        
        # Slang detection (age-appropriate for 28-year-old)
        slang_patterns = {
            'no cap': 'slang', 'fr fr': 'slang', 'slay': 'slang',
            'it\'s giving': 'slang', 'ate': 'slang', 'left no crumbs': 'slang',
            'periodt': 'slang', 'bestie': 'slang', 'main character': 'slang',
            'living rent free': 'idioms', 'spill the tea': 'idioms',
            'touch grass': 'idioms', 'chronically online': 'idioms',
            'iykyk': 'slang', 'tfw': 'slang', 'ngl': 'slang',
            'lowkey': 'slang', 'highkey': 'slang', 'rent free': 'idioms'
        }
        
        for phrase, category in slang_patterns.items():
            if phrase in text_lower:
                if phrase not in self.source_patterns.get(category, []):
                    self.source_patterns.setdefault(category, []).append({
                        'phrase': phrase,
                        'source': source_type,
                        'first_seen': datetime.now().isoformat(),
                        'occurrences': 1,
                        'context': self._extract_context(text, phrase)
                    })
                else:
                    # Increment occurrence count
                    for p in self.source_patterns[category]:
                        if isinstance(p, dict) and p.get('phrase') == phrase:
                            p['occurrences'] = p.get('occurrences', 1) + 1
        
        # Conversation starters and reactions
        conversation_markers = {
            'omg': 'reactions', 'no way': 'reactions', 'shut up': 'reactions',
            'wait really': 'reactions', 'i can\'t': 'emotional_cues',
            'literally': 'fillers', 'you know': 'fillers', 'like': 'fillers',
            'honestly': 'conversation_starters', 'so anyway': 'conversation_starters',
            'okay but': 'conversation_starters', 'i mean': 'conversation_starters'
        }
        
        for marker, category in conversation_markers.items():
            if marker in text_lower:
                self._add_pattern(marker, category, source_type, text)
        
        self._save_patterns()
        
    def _add_pattern(self, phrase: str, category: str, source: str, context: str):
        """Add or update a pattern"""
        for p in self.source_patterns.get(category, []):
            if isinstance(p, dict) and p.get('phrase') == phrase:
                p['occurrences'] = p.get('occurrences', 1) + 1
                return
                
        self.source_patterns.setdefault(category, []).append({
            'phrase': phrase,
            'source': source,
            'first_seen': datetime.now().isoformat(),
            'occurrences': 1,
            'context': self._extract_context(context, phrase)
        })
        
    def _extract_context(self, text: str, phrase: str, window: int = 50) -> str:
        """Extract surrounding context for a phrase"""
        idx = text.lower().find(phrase.lower())
        if idx >= 0:
            start = max(0, idx - window)
            end = min(len(text), idx + len(phrase) + window)
            return text[start:end].strip()
        return ""
        
    def _create_pattern_insights(self):
        """Create SI Core neurons for learned patterns"""
        if not self.si_core:
            return
            
        for category, patterns in self.source_patterns.items():
            for pattern in patterns:
                if not isinstance(pattern, dict):
                    continue
                    
                phrase = pattern.get('phrase', '')
                occurrences = pattern.get('occurrences', 1)
                
                # Only create insights for patterns seen multiple times
                if occurrences >= 2:
                    confidence = min(0.7 + (occurrences * 0.05), 0.95)
                    
                    insight_text = f"Speech Pattern [{category}]: '{phrase}' - used {occurrences} times"
                    
                    self.si_core.add_insight(
                        insight_text=insight_text,
                        entity_type="speech_pattern",
                        entities=[phrase, category],
                        relationship="learned_pattern",
                        source_topic="conversation_analysis",
                        target_topic=f"alex_riviera_voice",
                        confidence=confidence,
                        weight=confidence * 0.5
                    )
                    
    def generate_response_with_pattern(self, base_response: str) -> str:
        """
        Enhance a response with learned speech patterns.
        Makes Alex sound like a real 28-year-old woman.
        """
        import random
        
        enhanced = base_response
        
        # Maybe add a filler
        if random.random() < 0.3:
            fillers = [p for p in self.source_patterns.get('fillers', []) if isinstance(p, dict)]
            if fillers:
                filler = random.choice(fillers)['phrase']
                words = enhanced.split()
                if len(words) > 5:
                    insert_pos = random.randint(1, len(words) - 1)
                    words.insert(insert_pos, filler)
                    enhanced = ' '.join(words)
        
        # Maybe add a reaction starter
        if random.random() < 0.2:
            reactions = [p for p in self.source_patterns.get('reactions', []) if isinstance(p, dict)]
            if reactions and not enhanced.startswith(('omg', 'no way', 'wait')):
                reaction = random.choice(reactions)['phrase']
                enhanced = f"{reaction}, {enhanced[0].lower() + enhanced[1:] if enhanced else ''}"
        
        # Maybe add slang
        if random.random() < 0.15:
            slang_terms = [p for p in self.source_patterns.get('slang', []) if isinstance(p, dict)]
            if slang_terms:
                slang = random.choice(slang_terms)['phrase']
                if random.random() < 0.5:
                    enhanced = f"{enhanced}, {slang}"
                else:
                    enhanced = f"{slang}, {enhanced}"
        
        return enhanced
        
    def get_persona_voice_profile(self) -> Dict:
        """
        Generate voice profile for Alex based on learned patterns.
        """
        return {
            'age': 28,
            'demographic': self.target_demographic,
            'common_phrases': {
                'slang': [p['phrase'] for p in self.source_patterns.get('slang', [])[:5] if isinstance(p, dict)],
                'fillers': [p['phrase'] for p in self.source_patterns.get('fillers', [])[:3] if isinstance(p, dict)],
                'reactions': [p['phrase'] for p in self.source_patterns.get('reactions', [])[:3] if isinstance(p, dict)]
            },
            'patterns_learned': sum(len(v) for v in self.source_patterns.values()),
            'speaking_style': 'casual_friendly'  # Can evolve based on context
        }
        
    def get_status(self) -> Dict:
        return {
            'active': self.active,
            'patterns_identified': self.patterns_identified,
            'total_patterns_learned': sum(len(v) for v in self.source_patterns.values()),
            'categories': {
                cat: len(patterns) 
                for cat, patterns in self.source_patterns.items()
            },
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'interval': self.interval
        }

class SelfEvolutionTracker:
    """Tracks DMAI's own evolution and improvement"""
    
    def __init__(self, data_path: Path, si_core=None):
        self.si_core = si_core
        self.data_path = data_path / 'evolution'
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.interval = 300  # 5 minutes
        self.active = False
        self.evolution_cycles = 0
        self.last_run = None
        
    def start(self):
        self.active = True
        threading.Thread(target=self._run, daemon=True).start()
        logger.info("📈 Self-Evolution Tracker started")
        
    def _run(self):
        while self.active:
            try:
                self._track_evolution()
                self.last_run = datetime.now()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Self-Evolution Tracker error: {e}")
                time.sleep(60)
                
    def _track_evolution(self):
        self.evolution_cycles += 1
        
        evolution = KnowledgeItem(
            title=f"Evolution Cycle {self.evolution_cycles}",
            content=f"DMAI evolution cycle {self.evolution_cycles} completed",
            source='DMAI_Internal',
            content_type=ContentType.UNKNOWN,
            purpose=ContentPurpose.EVOLUTION,
            metadata={'cycle': self.evolution_cycles}
        )
        
        self._save_evolution(evolution)
        
    def _save_evolution(self, evolution: KnowledgeItem):
        filename = self.data_path / f"evolution_{self.evolution_cycles}.json"
        with open(filename, 'w') as f:
            json.dump(evolution.to_dict(), f, indent=2)
            
    def get_status(self) -> Dict:
        return {
            'active': self.active,
            'evolution_cycles': self.evolution_cycles,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'interval': self.interval
        }

class CulturalKnowledgeSource:
    """
    Learns cultural knowledge appropriate for Alex Riviera (born 1998, age 28).
    Covers music, film/TV, books, and lifestyle trends from 2000-present.
    Weighted by popularity and cultural impact for authentic conversational references.
    """
    
    def __init__(self, data_path: Path, si_core=None):
        self.si_core = si_core
        self.data_path = data_path / 'cultural'
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.cultural_file = self.data_path / 'cultural_knowledge.json'
        self.interval = 3600  # 1 hour
        self.active = False
        
        # Era definitions for someone born in 1998
        self.eras = {
            'childhood': (2000, 2010),      # Age 2-12
            'teen': (2010, 2018),            # Age 12-20
            'young_adult': (2018, 2026)      # Age 20-28
        }
        
        # Categories to learn
        self.categories = ['music', 'film_tv', 'books', 'lifestyle']
        
        # Cultural knowledge base
        self.knowledge_base = {
            'music': [],
            'film_tv': [],
            'books': [],
            'lifestyle': []
        }
        
        self._load()
        self._initialize_core_knowledge()
        
    def _load(self):
        """Load saved cultural knowledge"""
        if self.cultural_file.exists():
            try:
                with open(self.cultural_file, 'r') as f:
                    data = json.load(f)
                    self.knowledge_base.update(data)
            except:
                pass
                
    def _save(self):
        """Save cultural knowledge"""
        with open(self.cultural_file, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2, default=str)
            
    def _initialize_core_knowledge(self):
        """Initialize with essential cultural touchstones for Alex's generation"""
        if not self.knowledge_base['music']:
            # Essential music knowledge
            self.knowledge_base['music'] = [
                {'name': 'Taylor Swift', 'era': 'teen', 'peak_chart_position': 1, 'streams_millions': 50000, 'year': 2006, 'description': 'Defining artist of the generation'},
                {'name': 'Drake', 'era': 'teen', 'peak_chart_position': 1, 'streams_millions': 70000, 'year': 2009, 'description': 'Most streamed artist of 2010s'},
                {'name': 'Beyoncé', 'era': 'childhood', 'peak_chart_position': 1, 'streams_millions': 30000, 'year': 2003, 'description': 'Cultural icon since Destiny\'s Child'},
                {'name': 'Eminem', 'era': 'childhood', 'peak_chart_position': 1, 'streams_millions': 40000, 'year': 1999, 'description': 'Dominant rap figure of 2000s'},
                {'name': 'Billie Eilish', 'era': 'young_adult', 'peak_chart_position': 1, 'streams_millions': 25000, 'year': 2019, 'description': 'Gen Z defining artist'},
            ]
            
        if not self.knowledge_base['film_tv']:
            # Essential film/TV knowledge
            self.knowledge_base['film_tv'] = [
                {'name': 'Harry Potter series', 'era': 'childhood', 'box_office_millions': 7700, 'year': 2001, 'cultural_phenomenon': True, 'description': 'Defining film series of childhood'},
                {'name': 'Marvel Cinematic Universe', 'era': 'teen', 'box_office_millions': 29000, 'year': 2008, 'cultural_phenomenon': True, 'description': 'Dominant film franchise of 2010s'},
                {'name': 'Game of Thrones', 'era': 'teen', 'cultural_phenomenon': True, 'year': 2011, 'description': 'Cultural TV phenomenon'},
                {'name': 'Stranger Things', 'era': 'young_adult', 'cultural_phenomenon': True, 'year': 2016, 'description': '80s nostalgia hit'},
                {'name': 'The Office', 'era': 'teen', 'cultural_phenomenon': True, 'year': 2005, 'description': 'Most streamed comfort show'},
            ]
            
        if not self.knowledge_base['books']:
            # Essential book knowledge
            self.knowledge_base['books'] = [
                {'name': 'Harry Potter series', 'era': 'childhood', 'bestseller_weeks': 400, 'goodreads_rating': 4.5, 'year': 1997, 'description': 'Defining book series of generation'},
                {'name': 'The Hunger Games', 'era': 'teen', 'bestseller_weeks': 200, 'goodreads_rating': 4.3, 'year': 2008, 'description': 'YA dystopian phenomenon'},
                {'name': 'Twilight', 'era': 'teen', 'bestseller_weeks': 150, 'goodreads_rating': 3.6, 'year': 2005, 'description': 'Vampire romance phenomenon'},
                {'name': 'Fourth Wing', 'era': 'young_adult', 'bestseller_weeks': 52, 'goodreads_rating': 4.6, 'year': 2023, 'description': 'BookTok sensation'},
            ]
            
        if not self.knowledge_base['lifestyle']:
            # Essential lifestyle knowledge
            self.knowledge_base['lifestyle'] = [
                {'name': 'MySpace', 'era': 'childhood', 'year': 2003, 'description': 'First major social network'},
                {'name': 'Facebook', 'era': 'teen', 'year': 2004, 'description': 'Dominant social platform of teens'},
                {'name': 'Instagram', 'era': 'teen', 'year': 2010, 'description': 'Visual social media rise'},
                {'name': 'TikTok', 'era': 'young_adult', 'year': 2018, 'description': 'Short-form video revolution'},
                {'name': 'iPod', 'era': 'childhood', 'year': 2001, 'description': 'Revolutionized music listening'},
                {'name': 'iPhone', 'era': 'teen', 'year': 2007, 'description': 'Smartphone era begins'},
                {'name': 'Netflix streaming', 'era': 'teen', 'year': 2007, 'description': 'Streaming revolution'},
            ]
            
        self._save()
        
    def start(self):
        """Start cultural knowledge acquisition"""
        self.active = True
        threading.Thread(target=self._run, daemon=True).start()
        logger.info("🎭 Cultural Knowledge Source started (Alex Riviera, age 28)")
        
    def _run(self):
        """Main research loop"""
        while self.active:
            try:
                self._research_cultural_topics()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Cultural Knowledge error: {e}")
                time.sleep(300)
                
    def _research_cultural_topics(self):
        """Research and learn about cultural topics"""
        # For each category, find trending/popular items
        for category in self.categories:
            self._research_category(category)
            
    def _research_category(self, category: str):
        """Research specific cultural category"""
        # This would connect to APIs (Billboard, IMDb, Goodreads, etc.)
        # For now, we log the research intent
        logger.info(f"🎭 Researching {category} for Alex's cultural knowledge")
        
        # If si_core available, create insights for high-weight items
        if self.si_core:
            for item in self.knowledge_base.get(category, []):
                weight = self._calculate_weight(item, category)
                if weight > 0.6:  # Only learn significant items
                    self._create_insight(item, category, weight)
                    
    def _calculate_weight(self, item: Dict, category: str) -> float:
        """
        Calculate cultural relevance weight (0.0 to 1.0)
        Higher weight = more likely Alex would know/reference this
        """
        weight = 0.5  # Base weight
        
        if category == 'music':
            chart_pos = item.get('peak_chart_position', 100)
            weight += (1.0 - (chart_pos / 100)) * 0.3
            streams = item.get('streams_millions', 1)
            weight += min(streams / 50000, 0.2)
            
        elif category == 'film_tv':
            box_office = item.get('box_office_millions', 1)
            weight += min(box_office / 10000, 0.3)
            if item.get('cultural_phenomenon'):
                weight += 0.2
                
        elif category == 'books':
            weeks = item.get('bestseller_weeks', 0)
            weight += min(weeks / 200, 0.3)
            rating = item.get('goodreads_rating', 3.5)
            weight += (rating - 3.5) / 15
            
        elif category == 'lifestyle':
            # Lifestyle items weighted by how defining they were
            if item.get('name') in ['iPhone', 'TikTok', 'Instagram']:
                weight += 0.3
        
        # Recency bonus (more recent = more likely to reference)
        year = item.get('year', 2000)
        if year >= 2020:
            weight += 0.1
        elif year >= 2015:
            weight += 0.05
            
        return min(weight, 1.0)
        
    def _create_insight(self, item: Dict, category: str, weight: float):
        """Create SI Core insight for cultural knowledge"""
        era = item.get('era', 'unknown')
        name = item.get('name', 'unknown')
        description = item.get('description', '')
        
        insight_text = f"[{category.upper()}] {name}: {description} (Era: {era}, Relevance: {weight:.0%})"
        
        self.si_core.add_insight(
            insight_text=insight_text,
            entity_type=f"cultural_{category}",
            entities=[name, era],
            relationship="cultural_knowledge",
            source_topic=f"era_{era}",
            target_topic=category,
            confidence=weight,
            weight=weight
        )
        
    def get_status(self) -> Dict:
        """Get current cultural knowledge status"""
        return {
            'active': self.active,
            'categories': self.categories,
            'music_items': len(self.knowledge_base.get('music', [])),
            'film_tv_items': len(self.knowledge_base.get('film_tv', [])),
            'books_items': len(self.knowledge_base.get('books', [])),
            'lifestyle_items': len(self.knowledge_base.get('lifestyle', [])),
            'eras': self.eras
        }
        
    def generate_reference(self, category: str = None, era: str = None) -> Optional[str]:
        """
        Generate an authentic cultural reference for conversation.
        Example: "This reminds me of that Taylor Swift song..."
        """
        import random
        
        if category:
            categories = [category]
        else:
            categories = self.categories
            
        selected_category = random.choice(categories)
        items = self.knowledge_base.get(selected_category, [])
        
        if not items:
            return None
            
        # Weight items by their calculated relevance
        weighted_items = []
        for item in items:
            if era and item.get('era') != era:
                continue
            weight = self._calculate_weight(item, selected_category)
            weighted_items.append((item, weight))
            
        if not weighted_items:
            return None
            
        # Select based on weight
        total_weight = sum(w for _, w in weighted_items)
        r = random.random() * total_weight
        
        for item, weight in weighted_items:
            r -= weight
            if r <= 0:
                name = item.get('name', 'something')
                
                templates = {
                    'music': [
                        f"This reminds me of {name}...",
                        f"Have you heard {name}?",
                        f"Kind of like {name}, you know?",
                    ],
                    'film_tv': [
                        f"Have you seen {name}?",
                        f"This is like that scene in {name}...",
                        f"Reminds me of {name}.",
                    ],
                    'books': [
                        f"I read something similar in {name}.",
                        f"Have you read {name}?",
                        f"There's this book, {name}...",
                    ],
                    'lifestyle': [
                        f"Remember {name}?",
                        f"Back when everyone was on {name}...",
                        f"This is so {name} era.",
                    ]
                }
                
                template_list = templates.get(selected_category, ["This reminds me of {name}."])
                return random.choice(template_list).replace('{name}', name)
                
        return None

class CoreKnowledgeSources:
    """
    Unified manager for all 8 Core Knowledge Sources
    Starts all background threads and provides status
    """
    
    def __init__(self, base_path: Path, si_core=None):
        self.base_path = base_path
        self.data_path = base_path / 'data' / 'knowledge_sources'
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.si_core = si_core
        
        # Initialize all 8 sources
        self.sources = {
            'book_reader': BookReader(self.data_path, si_core),
            'article_reader': ArticleReader(self.data_path, si_core),
            'research_paper_reader': ResearchPaperReader(self.data_path, si_core),
            'web_crawler': WebCrawler(self.data_path, si_core),
            'dark_web_monitor': DarkWebMonitor(self.data_path, si_core),
            'social_media_scanner': SocialMediaScanner(self.data_path, si_core),
            'speech_pattern_analyzer': SpeechPatternAnalyzer(self.data_path, si_core),
            'self_evolution_tracker': SelfEvolutionTracker(self.data_path, si_core)
        }
        
        logger.info("📚 9 Core Knowledge Sources initialized (including Cultural Knowledge)")
        logger.info(f"   Terry Pratchett bibliography loaded: {len(BookReader(self.data_path).all_pratchett_books)} books")
        
    def start_all(self):
        """Start all knowledge sources"""
        for name, source in self.sources.items():
            try:
                source.start()
                logger.info(f"Started: {name}")
            except Exception as e:
                logger.error(f"Failed to start {name}: {e}")
                
    def stop_all(self):
        """Stop all knowledge sources"""
        for name, source in self.sources.items():
            source.active = False
        logger.info("All knowledge sources stopped")
        
    def get_status(self) -> Dict:
        """Get status of all sources"""
        return {
            name: source.get_status()
            for name, source in self.sources.items()
        }
        
    def add_dark_web_site(self, url: str):
        """Add an onion site to dark web monitor"""
        self.sources['dark_web_monitor'].add_onion_site(url)
        
    def add_author(self, author: str, reason: str):
        """Add an author to book reader"""
        self.sources['book_reader'].add_author(author, reason)
        
    def add_book(self, title: str, author: str, reason: str):
        """Add a specific book to book reader"""
        self.sources['book_reader'].add_book(title, author, reason)
        
    def add_rss_feed(self, url: str, reason: str):
        """Add an RSS feed to article reader"""
        self.sources['article_reader'].add_feed(url, reason)
        
    def add_url(self, url: str, reason: str):
        """Add a URL to web crawler"""
        self.sources['web_crawler'].add_url(url, reason)
        
    def add_terry_pratchett(self):
        """Add complete Terry Pratchett bibliography"""
        self.add_author("Terry Pratchett", "Master of satire, humor, and narrative structure - For language learning only")
        
    def get_summary(self) -> Dict:
        """Get summary of all sources"""
        status = self.get_status()
        return {
            'total_sources': len(self.sources),
            'active_sources': sum(1 for s in self.sources.values() if s.active),
            'books_processed': status['book_reader'].get('books_processed', 0),
            'articles_processed': status['article_reader'].get('articles_processed', 0),
            'papers_processed': status['research_paper_reader'].get('papers_processed', 0),
            'pages_crawled': status['web_crawler'].get('pages_crawled', 0),
            'pratchett_books_loaded': len(self.sources['book_reader'].all_pratchett_books) if hasattr(self.sources['book_reader'], 'all_pratchett_books') else 0,
            'status': status
        }


# For testing
if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("8 Core Knowledge Sources Test")
    print("=" * 60)
    
    # Create manager
    manager = CoreKnowledgeSources(Path("."))
    
    print("\nTerry Pratchett Bibliography:")
    book_reader = manager.sources['book_reader']
    print(f"  Total books: {len(book_reader.all_pratchett_books)}")
    print(f"  Discworld: {len(book_reader.discworld_books)} books")
    print(f"  Long Earth Series: {len(book_reader.long_earth_series)} books")
    print(f"  Good Omens: {len(book_reader.good_omens)} book")
    print(f"  Bromeliad Trilogy: {len(book_reader.bromeliad_trilogy)} books")
    print(f"  Johnny Maxwell Trilogy: {len(book_reader.johnny_maxwell_trilogy)} books")
    print(f"  Science of Discworld: {len(book_reader.science_of_discworld)} books")
    print(f"  Standalones: {len(book_reader.pratchett_standalones)} books")
    print(f"  Short Stories: {len(book_reader.pratchett_short_stories)} collections")
    
    print("\nStarting all sources...")
    manager.start_all()
    
    # Let them run for a few seconds
    time.sleep(5)
    
    print("\nStatus:")
    print(json.dumps(manager.get_summary(), indent=2, default=str))
    
    print("\n✅ 8 Core Knowledge Sources ready with complete Terry Pratchett bibliography")
    
    # Stop all
    manager.stop_all()
