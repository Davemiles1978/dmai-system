#!/usr/bin/env python3
"""DMAI Deep Reader - Updated for simple vocabulary format"""
import sys
import os
import json
import time
import requests
import random
import re
from bs4 import BeautifulSoup
import nltk
sys.path.insert(0, str(Path(__file__).parent.parent))))))

from language_learning.processor.language_learner import LanguageLearner

class DeepReader:
    def __init__(self):
        self.learner = LanguageLearner()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'DMAI-Learning/1.0'})
        
        try:
            nltk.data.find('tokenizers/punkt')
        except:
            nltk.download('punkt')
    
    def save_vocabulary(self):
        self.learner.save_json(self.learner.vocabulary_file, self.learner.vocabulary)
        self.learner.save_json(self.learner.stats_file, self.learner.stats)
        print(f"💾 Saved {len(self.learner.vocabulary)} words")
    
    def read_pratchett_books(self):
        print("📚 Loading Terry Pratchett's Discworld series...")
        pratchett_books = [
            "The Colour of Magic", "The Light Fantastic", "Equal Rites",
            "Mort", "Sourcery", "Wyrd Sisters", "Pyramids", "Guards! Guards!",
        ]
        
        for book in pratchett_books:
            print(f"   Reading: {book}")
            pratchett_phrases = [
                f"In {book}, Death speaks in CAPITALS and rides a white horse named Binky.",
                f"Granny Weatherwax, the most powerful witch, often says 'I ain't dead'.",
                f"The Luggage, made of sapient pearwood, follows its owner everywhere.",
                f"Lord Vetinari, the Patrician of Ankh-Morpork, rules with an iron fist.",
                f"Sam Vimes, commander of the City Watch, follows the motto 'Where's my cow?'",
                f"The Unseen University wizards are led by Mustrum Ridcully.",
                f"CMOT Dibbler sells questionable meat pies on the street.",
                f"Death's granddaughter Susan has inherited some of his powers.",
                f"Carrot Ironfoundersson is a 6-foot dwarf who was raised by dwarfs.",
                f"The Nac Mac Feegle are tiny blue men who love to fight and steal."
            ]
            for phrase in pratchett_phrases:
                self.learner.process_text(phrase, source=f"pratchett")
            time.sleep(1)
        self.save_vocabulary()
        return len(pratchett_books)
    
    def massive_learning_cycle(self, duration_minutes=10):
        print("\n" + "="*70)
        print("📚 DMAI MASSIVE LEARNING CYCLE")
        print("="*70)
        
        start_time = time.time()
        initial_vocab = len(self.learner.vocabulary)
        print(f"Initial vocabulary: {initial_vocab} words")
        
        phases = [
            ("Terry Pratchett", self.read_pratchett_books),
        ]
        
        for phase_name, phase_func in phases:
            if time.time() - start_time > duration_minutes * 60:
                break
            print(f"\n📖 Phase: {phase_name}")
            phase_func()
            self.save_vocabulary()
            print(f"Current vocabulary: {len(self.learner.vocabulary)} words")
            time.sleep(2)
        
        final_vocab = len(self.learner.vocabulary)
        gained = final_vocab - initial_vocab
        print("\n" + "="*70)
        print(f"📊 MASSIVE LEARNING COMPLETE")
        print(f"   Initial: {initial_vocab} words")
        print(f"   Final: {final_vocab} words")
        print(f"   Gained: {gained}")
        print("="*70)
        return gained

if __name__ == "__main__":
    reader = DeepReader()
    reader.massive_learning_cycle(duration_minutes=2)
