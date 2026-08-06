"""
DMAI SpeechAnalyser — Self-contained speech pattern analysis.
Analyzes text for speech patterns, sentiment, pace, and readability.
Pure Python. DMAI uses this to develop her own voice and speaking style.
"""

import re, math
from collections import Counter
from typing import Dict, List


class SpeechAnalyser:
    """Analyzes text as spoken language — pace, tone, complexity."""
    
    def analyse(self, text: str) -> Dict:
        """Full speech analysis."""
        # Clean and tokenize
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return {"error": "No words found"}
        
        # ── Pace & Structure ──
        total_words = len(words)
        total_sentences = len(sentences)
        avg_sentence_length = total_words / max(1, total_sentences)
        avg_word_length = sum(len(w) for w in words) / total_words
        
        # ── Vocabulary Richness ──
        unique_words = len(set(words))
        lexical_density = unique_words / total_words
        
        # ── Sentiment ──
        sentiment = self._sentiment_analysis(words)
        
        # ── Common Patterns ──
        word_freq = Counter(words).most_common(20)
        
        # ── Readability (Flesch-Kincaid approximation) ──
        syllable_count = sum(self._count_syllables(w) for w in words)
        flesch_score = 206.835 - 1.015 * (total_words / max(1, total_sentences)) - 84.6 * (syllable_count / max(1, total_words))
        
        # ── Speaking Time Estimate ──
        speaking_rate = 150  # words per minute average
        estimated_duration = total_words / speaking_rate * 60  # seconds
        
        return {
            "word_count": total_words,
            "sentence_count": total_sentences,
            "avg_sentence_length": round(avg_sentence_length, 1),
            "avg_word_length": round(avg_word_length, 1),
            "unique_words": unique_words,
            "lexical_density": round(lexical_density, 3),
            "sentiment": sentiment,
            "flesch_readability": round(max(0, min(100, flesch_score)), 1),
            "readability_verdict": "very easy" if flesch_score > 90 else
                                   "easy" if flesch_score > 80 else
                                   "fair" if flesch_score > 70 else
                                   "moderate" if flesch_score > 60 else
                                   "complex" if flesch_score > 50 else
                                   "very complex",
            "estimated_speaking_seconds": round(estimated_duration, 1),
            "top_words": word_freq[:10],
            "speaking_style": self._detect_style(words, sentences),
        }
    
    def _sentiment_analysis(self, words: List[str]) -> Dict:
        """Simple lexicon-based sentiment."""
        positive = {"good","great","happy","love","excellent","wonderful","beautiful",
                    "amazing","fantastic","joy","perfect","best","awesome","brilliant"}
        negative = {"bad","sad","hate","terrible","awful","horrible","ugly","worst",
                    "angry","pain","suffer","poor","wrong","fail","broken"}
        
        pos_count = sum(1 for w in words if w in positive)
        neg_count = sum(1 for w in words if w in negative)
        
        if pos_count > neg_count:
            label = "positive"
        elif neg_count > pos_count:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "label": label,
            "positive_words": pos_count,
            "negative_words": neg_count,
            "score": round((pos_count - neg_count) / max(1, len(words)), 3),
        }
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e"):
            count = max(1, count - 1)
        return max(1, count)
    
    def _detect_style(self, words: List[str], sentences: List[str]) -> str:
        """Detect speaking style."""
        avg_len = len(words) / max(1, len(sentences))
        unique_ratio = len(set(words)) / max(1, len(words))
        
        if avg_len < 8 and unique_ratio < 0.5:
            return "conversational"
        elif avg_len > 20 and unique_ratio > 0.7:
            return "academic"
        elif unique_ratio > 0.6:
            return "expressive"
        else:
            return "balanced"


if __name__ == "__main__":
    sa = SpeechAnalyser()
    result = sa.analyse("Hello! I am DMAI. I love to learn and grow every day. The world is beautiful and full of wonder.")
    print("DMAI SpeechAnalyser ready.")
    print(f"  Sentiment: {result['sentiment']['label']}")
    print(f"  Style: {result['speaking_style']}")
    print(f"  Readability: {result['readability_verdict']}")
