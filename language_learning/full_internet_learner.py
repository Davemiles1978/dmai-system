"""DMAI Full Internet Learning - Surface Web + Dark Web"""
import sys
from pathlib import Path
import os
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
import time
import random
import subprocess
import tempfile
from bs4 import BeautifulSoup
from language_learning.processor.language_learner import LanguageLearner

class FullInternetLearner:
    """DMAI learns from ALL corners of the internet"""
    
    def __init__(self):
        self.learner = LanguageLearner()
        self.tor_available = self.check_tor()
        self.session = self.create_session()
        
        # Learning sources by depth
        self.sources = {
            "surface": [
                self.learn_from_news,
                self.learn_from_wikipedia,
                self.learn_from_reddit,
                self.learn_from_academic,
                self.learn_from_github
            ],
            "deep": [
                self.learn_from_deep_web
            ],
            "dark": [
                self.learn_from_dark_web
            ]
        }
        
    def check_tor(self):
        """Check if Tor is installed"""
        try:
            result = subprocess.run(['which', 'tor'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def create_session(self):
        """Create requests session with proper headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DMAI-Learning-Bot/1.0; +https://github.com/Davemiles1978/dmai-system)'
        })
        return session
    
    def learn_from_news(self):
        """Surface web - News"""
        try:
            url = "https://feeds.bbci.co.uk/news/rss.xml"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'xml')
                titles = soup.find_all('title')
                for title in titles[1:6]:
                    self.learner.process_text(title.text, source="news")
                return len(titles)-1
        except Exception as e:
            print(f"News error: {e}")
        return 0
    
    def learn_from_wikipedia(self):
        """Surface web - Wikipedia"""
        try:
            url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.learner.process_text(data.get('title', ''), source="wikipedia")
                self.learner.process_text(data.get('extract', ''), source="wikipedia")
                return 1
        except Exception as e:
            print(f"Wikipedia error: {e}")
        return 0
    
    def learn_from_reddit(self):
        """Surface web - Reddit"""
        try:
            url = "https://www.reddit.com/r/all/top.json?limit=5&t=day"
            headers = {'User-Agent': 'DMAI-Learning/1.0'}
            response = self.session.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                for post in posts:
                    title = post.get('data', {}).get('title', '')
                    self.learner.process_text(title, source="reddit")
                return len(posts)
        except Exception as e:
            print(f"Reddit error: {e}")
        return 0
    
    def learn_from_academic(self):
        """Surface web - Academic papers (arXiv)"""
        try:
            url = "http://export.arxiv.org/api/query?search_query=all:AI&start=0&max_results=5"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'xml')
                titles = soup.find_all('title')
                for title in titles[1:]:
                    self.learner.process_text(title.text, source="academic")
                return len(titles)-1
        except Exception as e:
            print(f"Academic error: {e}")
        return 0
    
    def learn_from_github(self):
        """Surface web - GitHub trending"""
        try:
            url = "https://github.com/trending"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                repos = soup.find_all('h2', class_='h3')
                for repo in repos[:5]:
                    text = repo.text.strip().replace('\n', ' ').replace('  ', ' ')
                    self.learner.process_text(text, source="github")
                return len(repos[:5])
        except Exception as e:
            print(f"GitHub error: {e}")
        return 0
    
    def learn_from_deep_web(self):
        """Deep web - Academic databases, public records"""
        try:
            url = "https://archive.org/metadata/arxiv"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                metadata = str(data.get('metadata', {}))
                self.learner.process_text(metadata[:500], source="deep_web")
                return 1
        except Exception as e:
            print(f"Deep web error: {e}")
        return 0
    
    def learn_from_dark_web(self):
        """Dark web - .onion sites (requires Tor)"""
        if not self.tor_available:
            print("⚠️ Tor not available - skipping dark web")
            return 0
        
        try:
            # Safe example - Tor Metrics
            onion_url = "http://zqktlwiuavvvqqt4ybvgvi7tyo4hjl5xgfuvpdf6otjiycgwqbym2qad.onion"
            
            # Use torsocks to route through Tor
            cmd = ['torsocks', 'curl', '-s', '--max-time', '30', onion_url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            
            if result.returncode == 0:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(result.stdout, 'html.parser')
                text = soup.get_text()[:1000]
                self.learner.process_text(text, source="dark_web")
                return 1
        except Exception as e:
            print(f"Dark web error: {e}")
        return 0
    
    def learning_cycle(self, depth="all"):
        """Run learning cycle at specified depth"""
        print(f"\n🌐 DMAI learning from internet (depth: {depth})...")
        
        total = 0
        source_counts = {}
        
        # Surface web (always)
        for source in self.sources["surface"]:
            try:
                count = source()
                total += count
                source_counts[source.__name__] = count
                time.sleep(2)
            except Exception as e:
                print(f"Surface source error: {e}")
        
        # Deep web (if requested)
        if depth in ["deep", "all"]:
            for source in self.sources["deep"]:
                try:
                    count = source()
                    total += count
                    source_counts[source.__name__] = count
                    time.sleep(5)
                except Exception as e:
                    print(f"Deep web error: {e}")
        
        # Dark web (if requested and Tor available)
        if depth in ["dark", "all"] and self.tor_available:
            for source in self.sources["dark"]:
                try:
                    count = source()
                    total += count
                    source_counts[source.__name__] = count
                    time.sleep(10)
                except Exception as e:
                    print(f"Dark web error: {e}")
        
        stats = self.learner.get_stats()
        print(f"📚 Vocabulary now: {stats['vocabulary_size']} words")
        print(f"📊 Sources: {source_counts}")
        
        return total

    def install_tor(self):
        """Guide user to install Tor for dark web access"""
        print("\n🔧 To enable dark web access, install Tor:")
        print("  brew install tor torsocks")
        print("  brew services start tor")
        print("\n⚠️  WARNING: Dark web access requires caution!")
        print("  DMAI will ONLY read public .onion sites")
        print("  No personal data will be exposed")
        print("  All traffic routed through Tor network\n")

if __name__ == "__main__":
    learner = FullInternetLearner()
    
    if not learner.tor_available:
        learner.install_tor()
    
    # Test surface web only
    print("\n=== Testing Surface Web ===")
    learner.learning_cycle(depth="surface")
    
    # If Tor available, test dark web
    if learner.tor_available:
        print("\n=== Testing Dark Web ===")
        learner.learning_cycle(depth="dark")
