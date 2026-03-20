#!/usr/bin/env python3
"""Continuous learning system for DMAI - Runs 24/7"""
import sys
import os
import time
import json
import threading
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent)))))

from language_learning.real_book_reader import RealBookReader
from language_learning.tor_vocabulary import TorVocabularyLearner

class ContinuousLearner:
    def __init__(self):
        self.book_reader = RealBookReader()
        self.tor_learner = TorVocabularyLearner()
        self.running = True
        self.stats_file = "language_learning/data/learning_stats.json"
        
    def log_status(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
        # Append to log file
        with open("language_learning/logs/continuous_learning.log", "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def book_learning_thread(self):
        """Run book learning continuously"""
        book_index = 0
        all_books = self.book_reader.all_books
        
        while self.running:
            try:
                # Cycle through books
                if book_index >= len(all_books):
                    book_index = 0
                    self.log_status("📚 Completed all books, starting over")
                
                url, title = all_books[book_index]
                self.log_status(f"📖 Reading: {title}")
                
                # Read the book
                gained = self.book_reader.download_book(url, title)
                self.book_reader.save_vocabulary()
                
                self.log_status(f"✅ Learned {gained} new words from {title}")
                self.log_status(f"📊 Total vocabulary: {len(self.book_reader.learner.vocabulary)} words")
                
                book_index += 1
                
                # Wait between books
                for i in range(60, 0, -1):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.log_status(f"❌ Error in book learning: {e}")
                time.sleep(60)
    
    def internet_learning_thread(self):
        """Run internet/dark web learning continuously"""
        sources = [
            ("📰 News", self.learn_from_news),
            ("🌐 Wikipedia", self.learn_from_wikipedia),
            ("📚 arXiv", self.learn_from_arxiv),
            ("🌑 Dark Web", self.learn_from_darkweb),
        ]
        source_index = 0
        
        while self.running:
            try:
                name, func = sources[source_index]
                self.log_status(f"🌍 Learning from: {name}")
                
                gained = func()
                self.book_reader.save_vocabulary()
                
                self.log_status(f"✅ Learned from {name}")
                self.log_status(f"📊 Total vocabulary: {len(self.book_reader.learner.vocabulary)} words")
                
                source_index = (source_index + 1) % len(sources)
                
                # Wait between sources
                for i in range(30, 0, -1):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.log_status(f"❌ Error in internet learning: {e}")
                time.sleep(60)
    
    def learn_from_news(self):
        """Learn from news articles"""
        import requests
        from bs4 import BeautifulSoup
        
        try:
            # BBC News
            response = requests.get("http://feeds.bbci.co.uk/news/rss.xml", timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')[:3]
                for item in items:
                    title = item.find('title').text if item.find('title') else ''
                    desc = item.find('description').text if item.find('description') else ''
                    self.book_reader.learner.process_text(f"{title}. {desc}", source="news")
            return 3
        except:
            return 0
    
    def learn_from_wikipedia(self):
        """Learn from random Wikipedia articles"""
        import requests
        
        try:
            url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                title = data.get('title', '')
                extract = data.get('extract', '')[:2000]
                self.book_reader.learner.process_text(f"{title}. {extract}", source="wikipedia")
                return 1
        except:
            return 0
        return 0
    
    def learn_from_arxiv(self):
        """Learn from academic papers"""
        import requests
        from bs4 import BeautifulSoup
        
        try:
            url = "http://export.arxiv.org/api/query?search_query=all:AI&start=0&max_results=3"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                entries = soup.find_all('entry')[:1]
                for entry in entries:
                    title = entry.find('title').text
                    summary = entry.find('summary').text[:1000]
                    self.book_reader.learner.process_text(f"{title}. {summary}", source="arxiv")
            return 1
        except:
            return 0
    
    def learn_from_darkweb(self):
        """Learn from dark web if Tor is available"""
        if self.tor_learner.check_tor():
            return self.tor_learner.learning_cycle(duration_minutes=2)
        return 0
    
    def start(self):
        """Start all learning threads"""
        self.log_status("🚀 Starting Continuous Learning System")
        self.log_status(f"Initial vocabulary: {len(self.book_reader.learner.vocabulary)} words")
        
        # Create logs directory
        os.makedirs("language_learning/logs", exist_ok=True)
        
        # Start threads
        book_thread = threading.Thread(target=self.book_learning_thread, daemon=True)
        internet_thread = threading.Thread(target=self.internet_learning_thread, daemon=True)
        
        book_thread.start()
        internet_thread.start()
        
        self.log_status("✅ Learning threads started")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(60)
                # Show status every hour
                vocab = len(self.book_reader.learner.vocabulary)
                self.log_status(f"📊 Current vocabulary: {vocab} words")
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        self.log_status("🛑 Stopping Continuous Learning System")
        self.running = False
        self.book_reader.save_vocabulary()

if __name__ == "__main__":
    learner = ContinuousLearner()
    learner.start()
