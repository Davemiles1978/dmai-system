"""DMAI Language Learning - English only"""
import json
import os
from datetime import datetime
import re
from collections import Counter
import logging
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent language detection
DetectorFactory.seed = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageLearner:
    """Learns English language from captured audio/text"""
    
    def __init__(self, target_language='en'):
        self.target_language = target_language
        self.vocabulary_file = "language_learning/data/vocabulary.json"
        self.phrases_file = "language_learning/data/phrases.json"
        self.stats_file = "language_learning/data/stats.json"
        self.rejected_file = "language_learning/data/rejected_non_english.json"
        os.makedirs("language_learning/data", exist_ok=True)
        
        self.vocabulary = self.load_json(self.vocabulary_file, {})
        self.phrases = self.load_json(self.phrases_file, [])
        self.rejected = self.load_json(self.rejected_file, [])
        self.stats = self.load_json(self.stats_file, {
            "total_phrases_heard": 0,
            "english_phrases": 0,
            "non_english_phrases": 0,
            "unique_words": 0,
            "common_words": {},
            "learning_rate": 0
        })
        
        # Common English words to bootstrap vocabulary
        self.bootstrap_english()
    
    def bootstrap_english(self):
        """Add common English words to start vocabulary"""
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them",
            "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how", "our", "work",
            "first", "well", "way", "even", "new", "want", "because", "any", "these",
            "give", "day", "most", "us", "dmai", "david", "master", "father", "sir"
        ]
        
        for word in common_words:
            if word not in self.vocabulary:
                self.vocabulary[word] = {
                    "first_heard": "bootstrap",
                    "count": 0,
                    "sources": ["initialization"]
                }
        
        logger.info(f"📚 Bootstrapped with {len(common_words)} common English words")
    
    def load_json(self, filepath, default):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except json.JSONDecodeError:
            return default
    
    def save_json(self, filepath, data):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def is_english(self, text):
        """Detect if text is English"""
        if not text or len(text) < 3:
            return False
        
        try:
            lang = detect(text)
            return lang == self.target_language
        except LangDetectException:
            # If detection fails, check if most characters are English alphabet
            english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
            total_chars = sum(1 for c in text if c.isalpha())
            if total_chars > 0:
                return (english_chars / total_chars) > 0.9
            return False
    
    def process_text(self, text, source="ambient"):
        """Process heard text if it's English"""
        if not text:
            return
        
        text = text.lower().strip()
        
        # Update total count
        self.stats["total_phrases_heard"] = self.stats.get("total_phrases_heard", 0) + 1
        
        # Check if English
        if not self.is_english(text):
            logger.info(f"🌍 Non-English detected (rejected): {text[:50]}...")
            self.rejected.append({
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "source": source
            })
            self.rejected = self.rejected[-1000:]  # Keep last 1000
            self.save_json(self.rejected_file, self.rejected)
            
            self.stats["non_english_phrases"] = self.stats.get("non_english_phrases", 0) + 1
            self.save_json(self.stats_file, self.stats)
            return {"rejected": True, "reason": "non_english"}
        
        logger.info(f"✅ English detected: {text[:50]}...")
        
        # Clean and tokenize
        words = re.findall(r'\b[a-z]+\b', text)
        
        # Update vocabulary
        new_words_count = 0
        for word in words:
            if word not in self.vocabulary:
                self.vocabulary[word] = {
                    "first_heard": datetime.now().isoformat(),
                    "count": 1,
                    "sources": [source]
                }
                new_words_count += 1
            else:
                self.vocabulary[word]["count"] += 1
                if source not in self.vocabulary[word]["sources"]:
                    self.vocabulary[word]["sources"].append(source)
        
        # Store phrase
        phrase_entry = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "word_count": len(words),
            "new_words": new_words_count
        }
        self.phrases.append(phrase_entry)
        self.phrases = self.phrases[-1000:]  # Keep last 1000
        
        # Update stats
        self.stats["english_phrases"] = self.stats.get("english_phrases", 0) + 1
        self.stats["unique_words"] = len(self.vocabulary)
        
        # Calculate common words
        word_counts = {word: data["count"] for word, data in self.vocabulary.items()}
        counter = Counter(word_counts)
        self.stats["common_words"] = dict(counter.most_common(20))
        
        english_count = self.stats.get("english_phrases", 1)
        self.stats["learning_rate"] = len(self.vocabulary) / max(1, english_count)
        
        # Save
        self.save_json(self.vocabulary_file, self.vocabulary)
        self.save_json(self.phrases_file, self.phrases)
        self.save_json(self.stats_file, self.stats)
        
        logger.info(f"📚 Learned {new_words_count} new words, total vocabulary: {len(self.vocabulary)}")
        
        return {
            "new_words": new_words_count,
            "total_vocabulary": len(self.vocabulary),
            "rejected": False
        }
    
    def get_stats(self):
        """Get learning statistics"""
        return {
            "vocabulary_size": len(self.vocabulary),
            "phrases_heard": self.stats.get("english_phrases", 0),
            "non_english_rejected": self.stats.get("non_english_phrases", 0),
            "common_words": self.stats.get("common_words", {}),
            "learning_rate": self.stats.get("learning_rate", 0)
        }
    
    def get_new_words(self, since=None):
        """Get words learned since timestamp"""
        if not since:
            # Return last 50 learned words
            words_with_time = [(word, data["first_heard"]) for word, data in self.vocabulary.items()
                             if data["first_heard"] != "bootstrap"]
            words_with_time.sort(key=lambda x: x[1], reverse=True)
            return [w[0] for w in words_with_time[:50]]
        
        new_words = []
        for word, data in self.vocabulary.items():
            if data["first_heard"] != "bootstrap" and data["first_heard"] > since:
                new_words.append(word)
        return new_words

if __name__ == "__main__":
    learner = LanguageLearner()
    
    # Test with mixed languages
    test_phrases = [
        "Hello there, how are you today?",
        "Bonjour, comment allez-vous?",  # French - should reject
        "I am learning new words every day",
        "Hola, ¿cómo estás?",  # Spanish - should reject
        "DMAI is becoming more intelligent",
        "Guten Tag, wie geht es Ihnen?"  # German - should reject
    ]
    
    print("\nTesting language detection:\n")
    for phrase in test_phrases:
        print(f"Input: {phrase}")
        result = learner.process_text(phrase)
        if result and result.get("rejected"):
            print(f"❌ REJECTED: {result['reason']}")
        else:
            print(f"✅ ACCEPTED: learned {result['new_words']} new words")
        print("-" * 50)
    
    print("\n📊 Final Stats:", learner.get_stats())
