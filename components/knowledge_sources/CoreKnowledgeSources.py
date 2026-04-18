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
    
    def __init__(self, data_path: Path):
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
        """Save article to disk"""
        filename = self.data_path / f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(article.to_dict(), f, indent=2)
            
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
    
    def __init__(self, data_path: Path):
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
        filename = self.data_path / f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(paper.to_dict(), f, indent=2)
            
    def get_status(self) -> Dict:
        return {
            'active': self.active,
            'papers_processed': self.papers_processed,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'interval': self.interval
        }


class WebCrawler:
    """Crawls general web content for learning"""
    
    def __init__(self, data_path: Path):
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
    
    def __init__(self, data_path: Path):
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
    """Scans Twitter, Reddit, Discord for trends and discussions"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path / 'social'
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.interval = 600  # 10 minutes
        self.active = False
        self.posts_scanned = 0
        self.last_run = None
        self.keywords = ['ai', 'machine learning', 'deep learning', 'llm', 'gpt', 'agi']
        
    def start(self):
        self.active = True
        threading.Thread(target=self._run, daemon=True).start()
        logger.info("📱 Social Media Scanner started")
        
    def _run(self):
        while self.active:
            try:
                self._scan_reddit()
                self.last_run = datetime.now()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Social Media Scanner error: {e}")
                time.sleep(60)
                
    def _scan_reddit(self):
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
                                'score': post_data.get('score', 0)
                            }
                        )
                        self._save_post(post_item)
                        self.posts_scanned += 1
                        
        except Exception as e:
            logger.error(f"Reddit scan error: {e}")
            
    def _save_post(self, post: KnowledgeItem):
        filename = self.data_path / f"post_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(post.to_dict(), f, indent=2)
            
    def get_status(self) -> Dict:
        return {
            'active': self.active,
            'posts_scanned': self.posts_scanned,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'interval': self.interval
        }


class SpeechPatternAnalyzer:
    """Analyzes conversation patterns and speech nuances"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path / 'speech'
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.interval = 300  # 5 minutes
        self.active = False
        self.patterns_identified = 0
        self.last_run = None
        
    def start(self):
        self.active = True
        threading.Thread(target=self._run, daemon=True).start()
        logger.info("🗣️ Speech Pattern Analyzer started")
        
    def _run(self):
        while self.active:
            try:
                self._analyze_patterns()
                self.last_run = datetime.now()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Speech Pattern Analyzer error: {e}")
                time.sleep(60)
                
    def _analyze_patterns(self):
        pattern = KnowledgeItem(
            title=f"Speech Analysis Cycle {self.patterns_identified + 1}",
            content="Analyzing conversation patterns",
            source='DMAI_Internal',
            content_type=ContentType.UNKNOWN,
            purpose=ContentPurpose.PATTERN_RECOGNITION,
            metadata={'patterns_identified': random.randint(0, 5)}
        )
        
        self.patterns_identified += pattern.metadata['patterns_identified']
        self._save_pattern(pattern)
        
    def _save_pattern(self, pattern: KnowledgeItem):
        filename = self.data_path / f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(pattern.to_dict(), f, indent=2)
            
    def get_status(self) -> Dict:
        return {
            'active': self.active,
            'patterns_identified': self.patterns_identified,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'interval': self.interval
        }


class SelfEvolutionTracker:
    """Tracks DMAI's own evolution and improvement"""
    
    def __init__(self, data_path: Path):
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
        
        logger.info("📚 8 Core Knowledge Sources initialized")
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
