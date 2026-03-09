#!/usr/bin/env python3
"""Full book reader - downloads COMPLETE books for massive vocabulary gain"""
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
        
        # Project Gutenberg books (full text)
        self.gutenberg_books = [
            ("https://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice - Jane Austen (Gutenberg)"),
            ("https://www.gutenberg.org/files/84/84-0.txt", "Frankenstein - Mary Shelley (Gutenberg)"),
            ("https://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland - Lewis Carroll (Gutenberg)"),
            ("https://www.gutenberg.org/files/1661/1661-0.txt", "Sherlock Holmes - Arthur Conan Doyle (Gutenberg)"),
            ("https://www.gutenberg.org/files/2701/2701-0.txt", "Moby Dick - Herman Melville (Gutenberg)"),
            ("https://www.gutenberg.org/files/98/98-0.txt", "A Tale of Two Cities - Dickens (Gutenberg)"),
            ("https://www.gutenberg.org/files/1400/1400-0.txt", "Great Expectations - Dickens (Gutenberg)"),
            ("https://www.gutenberg.org/files/43/43-0.txt", "Dr. Jekyll and Mr. Hyde (Gutenberg)"),
            ("https://www.gutenberg.org/files/345/345-0.txt", "Dracula - Bram Stoker (Gutenberg)"),
            ("https://www.gutenberg.org/files/74/74-0.txt", "Tom Sawyer - Mark Twain (Gutenberg)"),
            ("https://www.gutenberg.org/files/76/76-0.txt", "Huckleberry Finn - Twain (Gutenberg)"),
            ("https://www.gutenberg.org/files/174/174-0.txt", "Dorian Gray - Oscar Wilde (Gutenberg)"),
            ("https://www.gutenberg.org/files/161/161-0.txt", "Sense and Sensibility - Austen (Gutenberg)"),
            ("https://www.gutenberg.org/files/158/158-0.txt", "Emma - Jane Austen (Gutenberg)"),
            ("https://www.gutenberg.org/files/1260/1260-0.txt", "Anna Karenina - Tolstoy (Gutenberg)"),
            ("https://www.gutenberg.org/files/2600/2600-0.txt", "War and Peace - Tolstoy (Gutenberg)"),
            ("https://www.gutenberg.org/files/996/996-0.txt", "Don Quixote - Cervantes (Gutenberg)"),
            ("https://www.gutenberg.org/files/55/55-0.txt", "Wizard of Oz - Baum (Gutenberg)"),
        ]
    
    def save_vocabulary(self):
        self.learner.save_json(self.learner.vocabulary_file, self.learner.vocabulary)
        self.learner.save_json(self.learner.stats_file, self.learner.stats)
        print(f"💾 Saved {len(self.learner.vocabulary)} words")
    
    def download_full_book(self, url, title):
        """Download and process COMPLETE book text"""
        try:
            print(f"\n📚 Downloading FULL book: {title}")
            response = self.session.get(url, timeout=60)
            if response.status_code != 200:
                print(f"❌ Failed to download {title}")
                return 0
            
            text = response.text
            
            # Remove Gutenberg headers/footers
            start_markers = ["*** START OF THE PROJECT GUTENBERG EBOOK", 
                            "*** START OF THIS PROJECT GUTENBERG EBOOK"]
            end_markers = ["*** END OF THE PROJECT GUTENBERG EBOOK",
                          "*** END OF THIS PROJECT GUTENBERG EBOOK"]
            
            for marker in start_markers:
                if marker in text:
                    text = text.split(marker)[1]
                    break
            
            for marker in end_markers:
                if marker in text:
                    text = text.split(marker)[0]
                    break
            
            # Clean text
            text = re.sub(r'[^\w\s\.\,\!\?\']', ' ', text)
            
            # Process in large chunks
            words = text.split()
            total_words = len(words)
            print(f"   Book contains {total_words:,} words")
            
            chunk_size = 5000
            total_new = 0
            
            for i in range(0, min(total_words, 200000), chunk_size):  # First 200k words
                chunk = ' '.join(words[i:i+chunk_size])
                if chunk:
                    result = self.learner.process_text(chunk, source=f"full_book")
                    if result:
                        total_new += result.get('new_words', 0)
                
                if i % 50000 == 0 and i > 0:
                    print(f"   Processed {i:,} words, gained {total_new} new words so far")
                    self.save_vocabulary()
            
            print(f"✅ Completed '{title}': +{total_new} new words")
            return total_new
            
        except Exception as e:
            print(f"❌ Error with {title}: {e}")
            return 0
    
    def learn_from_multiple_books(self, num_books=5):
        """Download and learn from multiple complete books"""
        print("\n" + "="*70)
        print("📚 DMAI READS FULL BOOKS - MASSIVE VOCABULARY GAIN")
        print("="*70)
        
        initial = len(self.learner.vocabulary)
        print(f"Starting vocabulary: {initial} words\n")
        
        total_gained = 0
        
        for i in range(min(num_books, len(self.gutenberg_books))):
            url, title = self.gutenberg_books[i]
            gained = self.download_full_book(url, title)
            total_gained += gained
            self.save_vocabulary()
            print(f"📊 Progress: {i+1}/{num_books} books, total gained: {total_gained}")
            print(f"📊 Current vocabulary: {len(self.learner.vocabulary)} words")
            time.sleep(3)
        
        final = len(self.learner.vocabulary)
        print("\n" + "="*70)
        print(f"📚 MASSIVE LEARNING COMPLETE")
        print(f"   Initial vocabulary: {initial} words")
        print(f"   Final vocabulary: {final} words")
        print(f"   Words gained: {final - initial}")
        print("="*70)
        
        return final - initial

if __name__ == "__main__":
    reader = FullBookReader()
    # Read 3 full books to start
    reader.learn_from_multiple_books(num_books=3)
