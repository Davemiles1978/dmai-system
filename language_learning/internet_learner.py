"""DMAI Internet Learning - Expands vocabulary from online"""
import requests
import json
import os
import random
import time
from bs4 import BeautifulSoup
from language_learning.processor.language_learner import LanguageLearner

class InternetLearner:
    """DMAI learns new words from the internet"""
    
    def __init__(self):
        self.learner = LanguageLearner()
        self.sources = [
            self.learn_from_news,
            self.learn_from_wikipedia,
            self.learn_from_reddit
        ]
        
    def learn_from_news(self):
        """Fetch latest news headlines"""
        try:
            # Simple news API (replace with actual API key)
            url = "https://newsapi.org/v2/top-headlines?country=us&apiKey=YOUR_API_KEY"
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                for article in articles[:5]:
                    title = article.get('title', '')
                    desc = article.get('description', '')
                    self.learner.process_text(title, source="news")
                    if desc:
                        self.learner.process_text(desc, source="news")
                return len(articles)
        except Exception as e:
            print(f"News error: {e}")
        return 0
    
    def learn_from_wikipedia(self):
        """Fetch random Wikipedia article"""
        try:
            # Get random article
            url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                title = data.get('title', '')
                extract = data.get('extract', '')
                
                self.learner.process_text(title, source="wikipedia")
                self.learner.process_text(extract, source="wikipedia")
                return 1
        except Exception as e:
            print(f"Wikipedia error: {e}")
        return 0
    
    def learn_from_reddit(self):
        """Fetch trending Reddit posts"""
        try:
            url = "https://www.reddit.com/r/all/top.json?limit=5&t=day"
            headers = {'User-Agent': 'DMAI-Learning/1.0'}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                posts = response.json().get('data', {}).get('children', [])
                for post in posts:
                    title = post.get('data', {}).get('title', '')
                    self.learner.process_text(title, source="reddit")
                return len(posts)
        except Exception as e:
            print(f"Reddit error: {e}")
        return 0
    
    def learning_cycle(self):
        """Run one learning cycle"""
        print("🌐 DMAI learning from internet...")
        total = 0
        for source in self.sources:
            try:
                total += source()
                time.sleep(2)  # Be polite to APIs
            except Exception as e:
                print(f"Source error: {e}")
        
        stats = self.learner.get_stats()
        print(f"📚 Vocabulary now: {stats['vocabulary_size']} words")
        return total

if __name__ == "__main__":
    learner = InternetLearner()
    learner.learning_cycle()
