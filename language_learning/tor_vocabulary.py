"""Tor vocabulary learning module"""
import requests
import time
import random
from bs4 import BeautifulSoup
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))))))

from language_learning.processor.language_learner import LanguageLearner

class TorVocabularyLearner:
    def __init__(self):
        self.learner = LanguageLearner()
        self.session = self.get_tor_session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.onion_sites = [
            'http://zqktlwiuavvvqqt4ybvgvi7tyo4hjl5xgfuvpdf6otjiycgwqbym2qad.onion/',
        ]
    
    def get_tor_session(self):
        session = requests.Session()
        session.proxies = {
            'http': 'socks5h://127.0.0.1:9150',
            'https': 'socks5h://127.0.0.1:9150'
        }
        return session
    
    def check_tor(self):
        try:
            response = self.session.get('https://check.torproject.org/', timeout=10)
            return 'Congratulations' in response.text
        except:
            return False
    
    def learn_from_onion(self, url):
        try:
            print(f'🌑 Learning from: {url[:50]}...')
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text()
                lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 20]
                for line in lines[:5]:
                    self.learner.process_text(line, source='dark_web')
                return len(lines)
        except Exception as e:
            return 0
    
    def learning_cycle(self, duration_minutes=5):
        if not self.check_tor():
            print('❌ Tor not working')
            return 0
        print('\n🌑 DARK WEB LEARNING')
        start = time.time()
        total = 0
        while time.time() - start < duration_minutes * 60:
            url = random.choice(self.onion_sites)
            total += self.learn_from_onion(url)
            time.sleep(random.randint(10, 20))
        print(f'✅ Learned from {total} sources')
        return total

if __name__ == '__main__':
    learner = TorVocabularyLearner()
    learner.learning_cycle(duration_minutes=1)
