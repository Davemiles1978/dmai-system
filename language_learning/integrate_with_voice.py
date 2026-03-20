#!/usr/bin/env python3
"""Connect language learning with voice interface"""
import sys
import os
import threading
import time
import json
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))))))

from language_learning.listener.ambient_listener import AmbientListener
from language_learning.processor.language_learner import LanguageLearner
from voice.speech_to_text import SpeechToText
from voice.speaker import DMAISpeaker

class DMAIWithLearning:
    """DMAI that learns from environment and tells you about it"""
    
    def __init__(self):
        self.learner = LanguageLearner()
        self.listener = AmbientListener(listen_duration=8, pause_between=15)
        self.stt = SpeechToText(model_size="base")
        self.speaker = DMAISpeaker()
        self.learning_active = False
        self.last_announcement = datetime.now()
        
        # Override the listener's process_audio to use STT and learning
        self.listener.process_audio = self.process_captured_audio
    
    def process_captured_audio(self, filename):
        """Process captured audio - transcribe and learn"""
        try:
            # Transcribe
            text = self.stt.transcribe_file(filename)
            
            if text:
                # Learn from it (will auto-filter English)
                result = self.learner.process_text(text, source="ambient")
                
                if result and not result.get("rejected"):
                    # If learned new words, maybe announce occasionally
                    if result.get("new_words", 0) > 0:
                        self.announce_learning(result)
                        
        except Exception as e:
            print(f"Error processing audio: {e}")
    
    def announce_learning(self, result):
        """Announce learning milestones"""
        # Don't announce too frequently
        now = datetime.now()
        if (now - self.last_announcement).seconds < 60:
            return
        
        new_words = result.get("new_words", 0)
        total = result.get("total_vocabulary", 0)
        
        if new_words > 0:
            message = f"I just learned {new_words} new words. My vocabulary is now {total} words."
            self.speaker.speak(message)
            self.last_announcement = now
    
    def start_learning(self):
        """Start background learning"""
        self.learning_active = True
        self.listener.start()
        self.speaker.speak("I'm now listening and learning from conversations around me.")
        print("🎧 Ambient learning activated")
    
    def stop_learning(self):
        """Stop learning"""
        self.learning_active = False
        self.listener.stop()
        self.speaker.speak("I've stopped listening.")
    
    def get_learning_summary(self):
        """Get summary of what she's learned"""
        stats = self.learner.get_stats()
        return f"I know {stats['vocabulary_size']} words from {stats['phrases_heard']} conversations. I've ignored {stats['non_english_rejected']} non-English phrases."
    
    def handle_command(self, text):
        """Handle voice commands related to learning"""
        text = text.lower()
        
        if "what did you learn" in text:
            return self.get_learning_summary()
        elif "new words" in text:
            new_words = self.learner.get_new_words(since=(datetime.now().isoformat()))
            if new_words:
                return f"Recently learned: {', '.join(new_words[:5])}"
            else:
                return "No new words recently."
        elif "stop learning" in text:
            self.stop_learning()
            return "Learning stopped."
        elif "start learning" in text:
            self.start_learning()
            return "Learning started."
        
        return None

# For testing
if __name__ == "__main__":
    dmai = DMAIWithLearning()
    
    print("Starting DMAI with learning...")
    dmai.start_learning()
    
    # Simulate some commands
    time.sleep(5)
    print("\n" + dmai.get_learning_summary())
    
    # Keep running for demo
    try:
        print("\nPress Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        dmai.stop_learning()
        print("\nStopped")
