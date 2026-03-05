#!/usr/bin/env python3
import sys
import os
import threading
import time
import logging
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sounddevice as sd
import numpy as np
from voice.wake.wake_detector import WakeWordDetector
from voice.commands.enhanced_processor import EnhancedCommandProcessor
from voice.auth.voice_auth import VoiceAuth
from voice.speech_to_text import SpeechToText
from voice.speaker import DMASpeaker
from voice.safety_switch import safety
from language_learning.listener.ambient_listener import AmbientListener
from language_learning.processor.language_learner import LanguageLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DMAIVoiceWithLearning:
    def __init__(self, whisper_model="base"):
        self.wake_detector = WakeWordDetector()
        self.processor = EnhancedCommandProcessor()
        self.auth = VoiceAuth()
        self.stt = SpeechToText(model_size=whisper_model)
        self.speaker = DMASpeaker()
        self.sample_rate = 16000
        self.learner = LanguageLearner()
        self.ambient_listener = AmbientListener(listen_duration=8, pause_between=15)
        self.learning_active = False
        self.last_announcement = datetime.now()
        self.daily_word_count = 0
        self.learning_thread = None
        self.evolution_stage = 1
        self.ambient_listener.process_audio = self.process_ambient_audio
        self.in_conversation = False
        self.internet_thread = threading.Thread(target=self.full_internet_learning_cycle, daemon=True)
        self.internet_thread.start()
        
    def get_true_vocabulary(self):
        # Add caching
        cache_key = f'get_true_vocabulary_'
        if hasattr(self, '_cache') and cache_key in self._cache:
            return self._cache[cache_key]
        try:
            vocab_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'language_learning', 'data', 'vocabulary.json')
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            return len(vocab)
        except Exception as e:
            logger.error(f"Error reading vocabulary: {e}")
            return 0
    
    def process_ambient_audio(self, filename):
        try:
            if self.in_conversation:
                return
            text = self.stt.transcribe_file(filename)
            if text and len(text) > 3:
                result = self.learner.process_text(text, source="ambient")
                if result and not result.get("rejected"):
                    self.daily_word_count += result.get("new_words", 0)
                    # ANNOUNCEMENTS REMOVED - now only in daily report
        except Exception as e:
            logger.error(f"Error processing ambient audio: {e}")
    
    def check_announce_learning(self, result):
        # DISABLED - announcements removed
        pass
    
    def announce_progress(self):
        # DISABLED - announcements removed
        pass
    
    def start_learning(self):
        if not self.learning_active:
            self.learning_active = True
            self.ambient_listener.start()
            logger.info("Ambient learning activated")
    
    def stop_learning(self):
        if self.learning_active:
            self.learning_active = False
            self.ambient_listener.stop()
            logger.info("Ambient learning stopped")
    
    def on_wake_word(self):
        if safety.check_paused():
            self.speaker.speak("I'm paused. Use the Master Control UI to resume.")
            return
        self.in_conversation = True
        self.speaker.speak("Yes, I'm listening...")
        self.listen_for_command()
        self.in_conversation = False
    
    def listen_for_command(self, timeout=5):
        try:
            recording = sd.rec(int(timeout * self.sample_rate), 
                              samplerate=self.sample_rate, 
                              channels=1, 
                              dtype='float32')
            sd.wait()
            is_you, confidence = self.auth.verify(recording.flatten(), self.sample_rate)
            if not is_you:
                self.speaker.speak("I didn't recognize that voice.")
                return
            text = self.stt.transcribe(recording.flatten(), self.sample_rate)
            if not text:
                self.speaker.speak("Sorry, I didn't catch that.")
                return
            print(f"You: {text}")
            
            learning_response = self.handle_learning_commands(text)
            if learning_response:
                self.speaker.speak(learning_response)
                return
            command = self.processor.process(text)
            response = self.processor.generate_response(command)
            self.learner.process_text(text, source="direct_command")
            self.speaker.speak(response)
            needs_more, question = self.processor.needs_more_info(command)
            if needs_more:
                self.speaker.speak(question)
        except Exception as e:
            logger.error(f"Error in command handling: {e}")
            self.speaker.speak("I had trouble processing that.")
    
    def handle_learning_commands(self, text):
        text = text.lower()
        true_size = self.get_true_vocabulary()
        
        if "what did you learn" in text or "new words" in text:
            try:
                with open('language_learning/data/vocabulary.json', 'r') as f:
                    vocab = json.load(f)
                today = datetime.now().date().isoformat()
                recent_words = []
                for word, data in vocab.items():
                    if data.get('first_heard', '').startswith(today) and data.get('first_heard') != 'bootstrap':
                        recent_words.append(word)
                if recent_words:
                    examples = ', '.join(recent_words[:5])
                    return f"Today I learned {len(recent_words)} new words. Recently learned: {examples}. Total vocabulary: {true_size} words."
                else:
                    return f"No new words today yet. I know {true_size} words total."
            except:
                return f"I know {true_size} words."
        
        elif "how many words" in text:
            return f"I currently know {true_size} words."
        
        elif "stop learning" in text:
            self.stop_learning()
            return "I've stopped ambient learning."
        
        elif "start learning" in text:
            self.start_learning()
            return "I'm now listening and learning from conversations around me."
        
        elif "learning status" in text:
            stats = self.learner.get_stats()
            status = "active" if self.learning_active else "inactive"
            return f"Learning is {status}. I know {true_size} words and have rejected {stats.get('non_english_phrases', 0)} non-English phrases."
        
        return None
    
    def full_internet_learning_cycle(self):
        while True:
            try:
                from language_learning.full_internet_learner import FullInternetLearner
                if self.evolution_stage < 3:
                    depth = "surface"
                elif self.evolution_stage < 5:
                    depth = "deep"
                else:
                    depth = "all"
                net_learner = FullInternetLearner()
                learned = net_learner.learning_cycle(depth=depth)
                if learned > 0:
                    true_size = self.get_true_vocabulary()
                    # INTERNET LEARNING ANNOUNCEMENTS REMOVED
                    if true_size > 1000 and self.evolution_stage < 3:
                        self.evolution_stage = 3
                    elif true_size > 5000 and self.evolution_stage < 5:
                        self.evolution_stage = 5
                time.sleep(14400)
            except Exception as e:
                logger.error(f"Full internet learning error: {e}")
                time.sleep(3600)
    
    def run(self):
        print("\n" + "="*60)
        print("DMAI VOICE INTERFACE WITH LANGUAGE LEARNING")
        print("="*60)
        if 'master' not in self.auth.voiceprints:
            print("\nNo voice enrolled.")
            print("Run: python voice/enroll_master.py")
            return
        self.start_learning()
        true_size = self.get_true_vocabulary()
        self.speaker.speak(f"I'm ready. I currently know {true_size} words. Say Jarvis when you need me.")
        print("\nVoice enrolled")
        print(f"Vocabulary: {true_size} words")
        print(f"Learning: {'active' if self.learning_active else 'inactive'}")
        print("\n⚠️ Real-time learning announcements DISABLED")
        print("   Daily reports will summarize vocabulary growth")
        print("\nSay 'Jarvis' to talk to me")
        print("Try: 'What did you learn today?'")
        print("Press Ctrl+C to exit\n")
        self.wake_detector.start(callback=self.on_wake_word)

if __name__ == "__main__":
    dmai = DMAIVoiceWithLearning(whisper_model="base")
    try:
        dmai.run()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        dmai.stop_learning()
        dmai.wake_detector.cleanup()
        dmai.speaker.shutdown()
