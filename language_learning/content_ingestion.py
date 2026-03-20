"""DMAI Content Ingestion - Reads books, news, articles, social media"""
import sys
import os
import json
import time
import requests
import random
import feedparser
import re
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
sys.path.insert(0, str(Path(__file__).parent.parent))))))

from language_learning.processor.language_learner import LanguageLearner

class ContentIngestion:
    """DMAI reads from multiple sources to learn vocabulary"""
    
    def __init__(self):
        self.learner = LanguageLearner()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DMAI-Learning-Bot/1.0; +https://github.com/Davemiles1978/dmai-system)'
        })
        
        # Sources configuration
        self.sources = {
            "books": [
                self.read_gutenberg_books,
                self.read_open_library,
                self.read_google_books
            ],
            "news": [
                self.read_rss_feeds,
                self.read_bbc_news,
                self.read_reuters,
                self.read_ap_news
            ],
            "articles": [
                self.read_medium,
                self.read_wikipedia,
                self.read_arxiv,
                self.read_wired,
                self.read_national_geographic
            ],
            "social": [
                self.read_reddit,
                self.read_twitter_trends,
                self.read_github_trending,
                self.read_stackoverflow
            ]
        }
        
        # RSS feeds for news
        self.rss_feeds = [
            "http://feeds.bbci.co.uk/news/rss.xml",
            "http://rss.cnn.com/rss/cnn_topstories.rss",
            "https://feeds.npr.org/1001/rss.xml",
            "https://www.wired.com/feed/rss",
            "https://news.ycombinator.com/rss"
        ]
        
    def read_gutenberg_books(self, count=1):
        """Read classic books from Project Gutenberg"""
        try:
            popular_books = [
                "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
                "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
                "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
                "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock Holmes
                "https://www.gutenberg.org/files/2701/2701-0.txt"   # Moby Dick
            ]
            
            for i in range(count):
                url = random.choice(popular_books)
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    text = response.text[:20000]
                    words = text.split()[:2000]
                    sample = ' '.join(words)
                    self.learner.process_text(sample, source="gutenberg")
                    print(f"📖 Learned from Gutenberg book")
            return count
        except Exception as e:
            print(f"Gutenberg error: {e}")
            return 0
    
    def read_open_library(self):
        """Read from Open Library"""
        try:
            url = "https://openlibrary.org/search.json?q=subject:fiction&limit=1&page=" + str(random.randint(1, 100))
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('docs'):
                    book = data['docs'][0]
                    title = book.get('title', '')
                    first_sentence = book.get('first_sentence', [''])[0]
                    text = f"{title}. {first_sentence}"
                    self.learner.process_text(text, source="open_library")
                    return 1
        except Exception as e:
            print(f"Open Library error: {e}")
        return 0
    
    def read_google_books(self):
        """Read from Google Books API"""
        try:
            url = f"https://www.googleapis.com/books/v1/volumes?q=subject:fiction&maxResults=1&startIndex={random.randint(1, 100)}"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    book = data['items'][0]['volumeInfo']
                    title = book.get('title', '')
                    description = book.get('description', '')
                    text = f"{title}. {description}"
                    self.learner.process_text(text[:2000], source="google_books")
                    return 1
        except Exception as e:
            print(f"Google Books error: {e}")
        return 0
    
    def read_rss_feeds(self):
        """Read from RSS news feeds"""
        total = 0
        for feed_url in self.rss_feeds[:3]:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:2]:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')[:500]
                    text = f"{title}. {summary}"
                    self.learner.process_text(text, source="news_rss")
                    total += 1
                time.sleep(2)
            except Exception as e:
                print(f"RSS error {feed_url}: {e}")
        return total
    
    def read_bbc_news(self):
        """Read BBC News"""
        try:
            url = "https://www.bbc.com/news"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                headlines = soup.find_all('h3')
                for headline in headlines[:5]:
                    self.learner.process_text(headline.text, source="bbc_news")
                return 5
        except Exception as e:
            return 0
    
    def read_reuters(self):
        """Read Reuters"""
        try:
            url = "https://www.reuters.com/"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all(['h2', 'h3', 'h4'])
                for article in articles[:5]:
                    self.learner.process_text(article.text, source="reuters")
                return 5
        except Exception as e:
            return 0
    
    def read_ap_news(self):
        """Read Associated Press"""
        try:
            url = "https://apnews.com/"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                headlines = soup.find_all('h2')
                for headline in headlines[:5]:
                    self.learner.process_text(headline.text, source="ap_news")
                return 5
        except Exception as e:
            return 0
    
    def read_medium(self):
        """Read Medium articles"""
        try:
            url = "https://medium.com/feed/tag/artificial-intelligence"
            feed = feedparser.parse(url)
            for entry in feed.entries[:2]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')[:500]
                self.learner.process_text(f"{title}. {summary}", source="medium")
            return 2
        except Exception as e:
            return 0
    
    def read_wikipedia(self):
        """Read random Wikipedia articles"""
        try:
            url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                title = data.get('title', '')
                extract = data.get('extract', '')[:1000]
                self.learner.process_text(f"{title}. {extract}", source="wikipedia_deep")
                return 1
        except Exception as e:
            return 0
    
    def read_arxiv(self):
        """Read academic papers from arXiv"""
        try:
            url = "http://export.arxiv.org/api/query?search_query=all:AI&start=0&max_results=3"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                entries = soup.find_all('entry')
                for entry in entries[:1]:
                    title = entry.find('title').text if entry.find('title') else ''
                    summary = entry.find('summary').text if entry.find('summary') else ''
                    self.learner.process_text(f"{title}. {summary[:500]}", source="arxiv")
                return 1
        except Exception as e:
            return 0
    
    def read_wired(self):
        """Read Wired articles"""
        try:
            url = "https://www.wired.com/feed/rss"
            feed = feedparser.parse(url)
            for entry in feed.entries[:2]:
                title = entry.get('title', '')
                self.learner.process_text(title, source="wired")
            return 2
        except Exception as e:
            return 0
    
    def read_national_geographic(self):
        """Read National Geographic"""
        try:
            url = "https://www.nationalgeographic.com/rss"
            feed = feedparser.parse(url)
            for entry in feed.entries[:2]:
                title = entry.get('title', '')
                self.learner.process_text(title, source="nat_geo")
            return 2
        except Exception as e:
            return 0
    
    def read_reddit(self):
        """Read Reddit discussions"""
        try:
            subreddits = ['all', 'news', 'science', 'technology', 'books']
            sub = random.choice(subreddits)
            url = f"https://www.reddit.com/r/{sub}/top.json?limit=5&t=day"
            headers = {'User-Agent': 'DMAI-Learning/1.0'}
            response = self.session.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                for post in posts[:3]:
                    title = post.get('data', {}).get('title', '')
                    self.learner.process_text(title, source="reddit")
                return 3
        except Exception as e:
            return 0
    
    def read_twitter_trends(self):
        """Read trending topics (simulated)"""
        try:
            url = "https://trends24.in/united-states"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                trends = soup.find_all('a', class_='trend-link')
                for trend in trends[:10]:
                    self.learner.process_text(trend.text, source="twitter_trends")
                return 10
        except Exception as e:
            return 0
    
    def read_github_trending(self):
        """Read GitHub trending repositories"""
        try:
            url = "https://github.com/trending"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                repos = soup.find_all('h2', class_='h3')
                for repo in repos[:5]:
                    text = repo.text.strip().replace('\n', ' ').replace('  ', ' ')
                    self.learner.process_text(text, source="github")
                return 5
        except Exception as e:
            return 0
    
    def read_stackoverflow(self):
        """Read Stack Overflow discussions"""
        try:
            url = "https://api.stackexchange.com/2.3/questions?order=desc&sort=hot&site=stackoverflow"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('items', [])[:3]:
                    title = item.get('title', '')
                    self.learner.process_text(title, source="stackoverflow")
                return 3
        except Exception as e:
            return 0
    
    def learning_cycle(self, duration_minutes=5):
        """Run a comprehensive learning cycle"""
        print("\n" + "="*60)
        print("📚 DMAI CONTENT INGESTION CYCLE")
        print("="*60)
        
        start_time = time.time()
        total_learned = 0
        source_counts = {}
        
        while time.time() - start_time < duration_minutes * 60:
            category = random.choice(list(self.sources.keys()))
            source = random.choice(self.sources[category])
            
            try:
                count = source()
                total_learned += count
                source_name = source.__name__
                source_counts[source_name] = source_counts.get(source_name, 0) + count
                
                if count > 0:
                    print(f"✅ Learned from {source_name}: +{count} items")
                
                time.sleep(random.randint(2, 5))
                
            except Exception as e:
                print(f"❌ Error in {source.__name__}: {e}")
                time.sleep(2)
        
        stats = self.learner.get_stats()
        print("\n" + "="*60)
        print(f"📊 LEARNING CYCLE COMPLETE")
        print(f"   Total items processed: {total_learned}")
        print(f"   Vocabulary now: {stats['vocabulary_size']} words")
        print("="*60)
        
        return {
            "total": total_learned,
            "vocabulary": stats['vocabulary_size'],
            "sources": source_counts
        }

if __name__ == "__main__":
    ingestion = ContentIngestion()
    print("\n🚀 Starting DMAI content ingestion test...")
    results = ingestion.learning_cycle(duration_minutes=1)
    print(f"\n📈 Final vocabulary: {results['vocabulary']} words")
