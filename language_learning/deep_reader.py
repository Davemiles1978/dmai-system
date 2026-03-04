"""DMAI Deep Reader - Reads and learns from large text sources"""
import sys
import os
import json
import time
import requests
import random
import re
from bs4 import BeautifulSoup
import nltk
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from language_learning.processor.language_learner import LanguageLearner

class DeepReader:
    """Reads large volumes of text to build massive vocabulary"""
    
    def __init__(self):
        self.learner = LanguageLearner()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'DMAI-Learning/1.0'})
        
        try:
            nltk.data.find('tokenizers/punkt')
        except:
            nltk.download('punkt')
    
    def read_pratchett_books(self):
        """Read Terry Pratchett's works (Discworld series)"""
        print("📚 Loading Terry Pratchett's Discworld series...")
        
        # List of Pratchett books available on Project Gutenberg-like sites
        # Note: Most Pratchett books are still under copyright
        # This is a placeholder - we'll need to source them legally
        pratchett_books = [
            "The Colour of Magic",
            "The Light Fantastic", 
            "Equal Rites",
            "Mort",
            "Sourcery",
            "Wyrd Sisters",
            "Pyramids",
            "Guards! Guards!",
            "Eric",
            "Moving Pictures",
            "Reaper Man",
            "Witches Abroad",
            "Small Gods",
            "Lords and Ladies",
            "Men at Arms",
            "Soul Music",
            "Feet of Clay",
            "Hogfather",
            "Jingo",
            "The Last Continent",
            "Carpe Jugulum",
            "The Fifth Elephant",
            "The Truth",
            "Thief of Time",
            "The Last Hero",
            "The Amazing Maurice and His Educated Rodents",
            "Night Watch",
            "The Wee Free Men",
            "Monstrous Regiment",
            "A Hat Full of Sky",
            "Going Postal",
            "Thud!",
            "Wintersmith",
            "Making Money",
            "Unseen Academicals",
            "I Shall Wear Midnight",
            "Snuff",
            "Raising Steam",
            "The Shepherd's Crown"
        ]
        
        # For now, we'll use sample texts from each book
        # In production, you'd need to source full texts legally
        for book in pratchett_books[:5]:  # Start with first 5 books
            print(f"   Reading: {book}")
            # Simulate reading the book
            # In reality, you'd fetch actual text
            sample_text = f"This is a sample from {book} by Terry Pratchett. The Discworld series is known for its humor, satire, and unique vocabulary. Death speaks in SMALL CAPS. The Luggage has many feet. Granny Weatherwax is a powerful witch. Sam Vimes is the commander of the Ankh-Morpork City Watch. The Unseen University is full of wizards. Nac Mac Feegle are tiny blue men who like to steal and fight. The Patrician, Lord Vetinari, rules the city with an iron fist in a velvet glove."
            self.learner.process_text(sample_text, source=f"pratchett_{book[:20]}")
            time.sleep(1)
        
        return len(pratchett_books[:5])
    
    def read_oxford_dictionary(self):
        """Read the Oxford English Dictionary (simulated)"""
        print("📚 Starting Oxford English Dictionary download...")
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them",
            "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how", "our", "work",
            "first", "well", "way", "even", "new", "want", "because", "any", "these",
            "give", "day", "most", "us"
        ]
        
        learned = 0
        for word in common_words:
            if word not in self.learner.vocabulary:
                self.learner.vocabulary[word] = {
                    "first_heard": "oxford_dictionary",
                    "count": 1,
                    "sources": ["oxford"]
                }
                learned += 1
        
        print(f"📚 Learned {learned} words from Oxford Dictionary sample")
        return learned
    
    def read_project_gutenberg_book(self, book_url, title):
        """Read an entire book from Project Gutenberg"""
        try:
            print(f"📚 Reading book: {title}...")
            response = self.session.get(book_url, timeout=60)
            if response.status_code == 200:
                text = response.text
                start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
                end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
                
                if start_marker in text:
                    text = text.split(start_marker)[1]
                if end_marker in text:
                    text = text.split(end_marker)[0]
                
                text = re.sub(r'[^\w\s\.\,\!\?\']', ' ', text)
                words = text.split()
                total_new = 0
                
                for i in range(0, len(words), 1000):
                    chunk = ' '.join(words[i:i+1000])
                    if chunk:
                        result = self.learner.process_text(chunk, source=f"book_{title[:20]}")
                        if result:
                            total_new += result.get('new_words', 0)
                    
                    time.sleep(1)
                    
                    if i % 10000 == 0 and i > 0:
                        print(f"   Processed {i} words...")
                
                print(f"✅ Completed '{title}': +{total_new} new words")
                return total_new
                
        except Exception as e:
            print(f"Error reading book: {e}")
            return 0
    
    def massive_learning_cycle(self, duration_minutes=10):
        """Intensive learning session with modern literature"""
        print("\n" + "="*70)
        print("📚 DMAI MASSIVE LEARNING CYCLE - Modern Literature")
        print("="*70)
        
        start_time = time.time()
        initial_vocab = len(self.learner.vocabulary)
        
        # Learning phases - now with Pratchett first!
        phases = [
            ("Oxford Dictionary Sample", lambda: self.read_oxford_dictionary()),
            ("Terry Pratchett Discworld Series", self.read_pratchett_books),
            ("Classic Books", self.read_classic_books),
            ("Long-form Articles", self.read_long_articles),
            ("Wikipedia Deep Dive", self.read_wikipedia_deep),
            ("Academic Papers", self.read_arxiv_papers)
        ]
        
        for phase_name, phase_func in phases:
            if time.time() - start_time > duration_minutes * 60:
                break
                
            print(f"\n📖 Phase: {phase_name}")
            phase_func()
            time.sleep(3)
        
        final_vocab = len(self.learner.vocabulary)
        gained = final_vocab - initial_vocab
        
        print("\n" + "="*70)
        print(f"📊 MASSIVE LEARNING COMPLETE")
        print(f"   Initial vocabulary: {initial_vocab} words")
        print(f"   Final vocabulary: {final_vocab} words")
        print(f"   Words gained: {gained}")
        print("="*70)
        
        return gained
    
    def read_classic_books(self):
        """Read multiple classic books"""
        books = [
            ("https://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice"),
            ("https://www.gutenberg.org/files/84/84-0.txt", "Frankenstein"),
            ("https://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland"),
            ("https://www.gutenberg.org/files/1661/1661-0.txt", "Sherlock Holmes"),
            ("https://www.gutenberg.org/files/2701/2701-0.txt", "Moby Dick"),
        ]
        
        for url, title in books[:2]:  # First 2 books
            self.read_project_gutenberg_book(url, title)
            time.sleep(5)
    
    def read_long_articles(self):
        """Read long-form articles"""
        articles = [
            "https://www.newyorker.com/magazine/2024/01/01/the-ai-revolution",
            "https://www.theatlantic.com/technology/archive/2024/02/ai-future-prediction/",
        ]
        
        for url in articles[:1]:
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    words = text.split()
                    for i in range(0, len(words), 500):
                        chunk = ' '.join(words[i:i+500])
                        if chunk:
                            self.learner.process_text(chunk, source="long_article")
            except Exception as e:
                print(f"Error reading article: {e}")
            time.sleep(3)
    
    def read_wikipedia_deep(self):
        """Read multiple Wikipedia articles"""
        topics = ["Artificial intelligence", "Machine learning", "Linguistics"]
        
        for topic in topics[:2]:
            try:
                url = f"https://en.wikipedia.org/api/rest_v1/page/html/{topic.replace(' ', '_')}"
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text()[:10000]
                    self.learner.process_text(text, source="wikipedia_deep")
                    print(f"📚 Read Wikipedia: {topic}")
                time.sleep(2)
            except Exception as e:
                print(f"Wikipedia error: {e}")
    
    def read_arxiv_papers(self):
        """Read academic papers"""
        try:
            url = "http://export.arxiv.org/api/query?search_query=all:AI&start=0&max_results=3"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                entries = soup.find_all('entry')
                for entry in entries[:1]:
                    title = entry.find('title').text
                    summary = entry.find('summary').text
                    text = f"{title}. {summary}"
                    self.learner.process_text(text[:5000], source="arxiv_deep")
                print(f"📚 Read {len(entries[:1])} academic papers")
        except Exception as e:
            print(f"arXiv error: {e}")

if __name__ == "__main__":
    reader = DeepReader()
    print("\n🚀 Testing deep reading capabilities with Terry Pratchett...")
    reader.massive_learning_cycle(duration_minutes=2)
    print(f"\n📚 Final vocabulary: {len(reader.learner.vocabulary)} words")
