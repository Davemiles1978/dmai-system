#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
import random
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from language_learning.processor.language_learner import LanguageLearner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - BOOK - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/book_reader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BOOK")

class BookReader:
    def __init__(self):
        self.learner = LanguageLearner()
        self.books_processed = 0
        self.words_learned = 0
        self.cycle_count = 0
        self.book_dir = Path("language_learning/books")
        self.book_dir.mkdir(exist_ok=True)
        self._cache = {}
        
    def get_gutenberg_book(self):
        # Add caching
        cache_key = f'get_gutenberg_book_'
        if hasattr(self, '_cache') and cache_key in self._cache:
            return self._cache[cache_key]
        try:
            import requests
            books = [
                "https://www.gutenberg.org/files/1342/1342-0.txt",
                "https://www.gutenberg.org/files/11/11-0.txt",
                "https://www.gutenberg.org/files/1661/1661-0.txt",
                "https://www.gutenberg.org/files/84/84-0.txt",
                "https://www.gutenberg.org/files/2701/2701-0.txt"
            ]
            
            url = random.choice(books)
            logger.info(f"Downloading book from {url}")
            
            for retry in range(3):
                try:
                    response = requests.get(url, timeout=30)
                    break
                except:
                    if retry == 2:
                        raise
                    time.sleep(1)
            if response.status_code == 200:
                text = response.text[:50000]
                
                result = self.learner.process_text(text, source="gutenberg")
                if result:
                    new = result.get("new_words", 0)
                    if new > 0:
                        self.words_learned += new
                        self.books_processed += 1
                        logger.info(f"+{new} words from book {self.books_processed}")
                        
        except Exception as e:
            logger.error(f"Book download error: {e}")
    
    def run(self):
        logger.info("Book Reader started")
        
        while True:
            try:
                # Inner loop runs forever
                while True:
                    self.cycle_count += 1
                    logger.info(f"Book reading cycle {self.cycle_count}")
                    
                    self.get_gutenberg_book()
                    
                    wait = random.randint(3600, 7200)
                    logger.info(f"Next book in {wait//3600} hours")
                    
                    # Sleep in chunks to be responsive to interrupts
                    for i in range(wait):
                        time.sleep(1)
                        
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Book reader error: {e}")
                logger.info("Restarting book reader cycle in 60 seconds...")
                time.sleep(60)

if __name__ == "__main__":
    reader = BookReader()
    try:
        reader.run()
    except KeyboardInterrupt:
        logger.info(f"Read {reader.books_processed} books, learned {reader.words_learned} words")
