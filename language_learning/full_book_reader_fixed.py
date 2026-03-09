#!/usr/bin/env python3
import sys
import os
import json
import time
import requests
import re
from bs4 import BeautifulSoup
sys.path.insert(0, str(Path(__file__).parent.parent))))))

from language_learning.processor.language_learner import LanguageLearner

class FullBookReader:
    def __init__(self):
        self.learner = LanguageLearner()
        self.session = requests.Session()
        self.gutenberg_books = [
            ("https://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice - Jane Austen"),
            ("https://www.gutenberg.org/files/84/84-0.txt", "Frankenstein - Mary Shelley"),
            ("https://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland - Lewis Carroll"),
        ]
    
    def save_vocabulary(self):
        """Force save with immediate disk write"""
        self.learner.save_json(self.learner.vocabulary_file, self.learner.vocabulary)
        self.learner.save_json(self.learner.stats_file, self.learner.stats)
        # Force sync to disk
        os.sync()
        print(f"💾 SAVED {len(self.learner.vocabulary)} words to {self.learner.vocabulary_file}")
    
    def download_full_book(self, url, title):
        try:
            print(f"\n📚 Downloading: {title}")
            response = self.session.get(url, timeout=60)
            if response.status_code != 200:
                return 0
            
            text = response.text
            
            # Clean headers/footers
            for marker in ["*** START OF THE PROJECT GUTENBERG EBOOK", "*** START OF THIS PROJECT GUTENBERG EBOOK"]:
                if marker in text:
                    text = text.split(marker)[1]
                    break
            for marker in ["*** END OF THE PROJECT GUTENBERG EBOOK", "*** END OF THIS PROJECT GUTENBERG EBOOK"]:
                if marker in text:
                    text = text.split(marker)[0]
                    break
            
            text = re.sub(r'[^\w\s\.\,\!\?\']', ' ', text)
            words = text.split()
            print(f"   Book has {len(words):,} words")
            
            chunk_size = 1000
            total_new = 0
            
            for i in range(0, min(len(words), 50000), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                if chunk:
                    result = self.learner.process_text(chunk, source=f"book")
                    if result:
                        total_new += result.get('new_words', 0)
                
                # Save after every 5 chunks
                if i % 5000 == 0:
                    print(f"   Progress: {i:,} words, +{total_new} new, saving...")
                    self.save_vocabulary()
            
            print(f"✅ Completed: +{total_new} new words")
            return total_new
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return 0
    
    def learn_from_books(self, num_books=3):
        print("\n" + "="*70)
        print("📚 DMAI READS FULL BOOKS")
        print("="*70)
        
        initial = len(self.learner.vocabulary)
        print(f"Starting: {initial} words\n")
        
        for i in range(min(num_books, len(self.gutenberg_books))):
            url, title = self.gutenberg_books[i]
            self.download_full_book(url, title)
            self.save_vocabulary()
            print(f"📊 After book {i+1}: {len(self.learner.vocabulary)} words")
            time.sleep(2)
        
        final = len(self.learner.vocabulary)
        print("\n" + "="*70)
        print(f"📚 FINAL: {final} words (+{final-initial})")
        print("="*70)

if __name__ == "__main__":
    reader = FullBookReader()
    reader.learn_from_books(num_books=3)
