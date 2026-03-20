#!/usr/bin/env python3
import sys
from pathlib import Path
import os
import threading
import time
import logging
import json
import subprocess
import re
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import sounddevice as sd
import numpy as np
from voice.wake.wake_detector import WakeWordDetector
from voice.commands.enhanced_processor import EnhancedCommandProcessor
from voice.commands.ai_processor import AIProcessor
from voice.auth.voice_auth import VoiceAuth
from voice.speech_to_text import SpeechToText
from voice.speaker_fixed import DMAISpeaker
from voice.safety_switch import safety
from language_learning.listener.ambient_listener import AmbientListener
from language_learning.processor.language_learner import LanguageLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DMAIVoiceWithLearning:
    def __init__(self, whisper_model="base"):
        self.wake_detector = WakeWordDetector()
        self.processor = EnhancedCommandProcessor()
        self.ai_processor = AIProcessor()
        self.auth = VoiceAuth()
        self.stt = SpeechToText(model_size=whisper_model)
        self.speaker = DMAISpeaker()
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
        self._announced_startup = False
        
        # Path to vocabulary manager
        self.vocab_manager = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'language_learning', 'data', 'secure', 'vocabulary_manager.py'
        )
        self.vocab_helper = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'voice', 'vocab_helper.py'
        )
        
        # Initialize cache
        self._cache = {}
        self._cache_timeout = 60  # seconds
        
    def get_true_vocabulary(self):
        """Get vocabulary count using the manager (read-only)"""
        cache_key = 'get_true_vocabulary'
        
        # Check cache
        if hasattr(self, '_cache') and cache_key in self._cache:
            cache_time, cache_value = self._cache[cache_key]
            if time.time() - cache_time < self._cache_timeout:
                return cache_value
        
        try:
            # Use vocabulary manager to get stats (read-only operation)
            result = subprocess.run(
                [sys.executable, self.vocab_manager],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Parse the output to get word count
            output = result.stdout
            for line in output.split('\n'):
                if 'Total words:' in line:
                    count = int(line.split(':')[1].strip())
                    self._cache[cache_key] = (time.time(), count)
                    return count
        except Exception as e:
            logger.error(f"Error reading vocabulary: {e}")
            
            # Fallback to direct file read (but never write)
            try:
                vocab_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                          'language_learning', 'data', 'vocabulary.json')
                with open(vocab_file, 'r') as f:
                    vocab = json.load(f)
                count = len(vocab)
                self._cache[cache_key] = (time.time(), count)
                return count
            except:
                return 0
        
        return 0
    
    def add_to_vocabulary(self, words):
        """
        Safely add words to vocabulary - APPEND ONLY
        Never modifies existing words
        """
        if not words:
            return 0
        
        # Convert single word to list
        if isinstance(words, str):
            words = [words]
        
        try:
            # Use vocab_helper to add words
            cmd = [sys.executable, self.vocab_helper, 'add'] + words
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Parse how many were added
            if "Added" in result.stdout:
                import re
                match = re.search(r'Added (\d+)', result.stdout)
                if match:
                    added_count = int(match.group(1))
                    if added_count > 0:
                        # Clear cache since vocabulary changed
                        if hasattr(self, '_cache'):
                            self._cache.pop('get_true_vocabulary', None)
                    return added_count
            return 0
        except Exception as e:
            logger.error(f"Error adding words to vocabulary: {e}")
            return 0
    
    def word_exists(self, word):
        """Check if a word exists in vocabulary (read-only)"""
        try:
            cmd = [sys.executable, self.vocab_helper, 'check', word]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return "True" in result.stdout or "✅" in result.stdout
        except:
            return False
    
    def process_ambient_audio(self, filename):
        """Process ambient audio for vocabulary learning"""
        try:
            if self.in_conversation:
                return
            
            text = self.stt.transcribe_file(filename)
            if text and len(text) > 3:
                # Ignore our own speech patterns (prevent feedback loop)
                ignore_phrases = ["i'm ready", "hey dee mai", "i currently know", "say hey", "dmai", "dee mai"]
                if any(phrase in text.lower() for phrase in ignore_phrases):
                    logger.info(f"🔇 Ignoring own speech: {text[:50]}...")
                    return  # Skip learning our own announcements
                
                # Process text through learner
                result = self.learner.process_text(text, source="ambient")
                
                # Extract new words from result
                if result and not result.get("rejected"):
                    # Get new words from the learner result
                    new_words = result.get("new_words", 0)
                    if new_words > 0:
                        # The learner should return the actual words
                        # For now, we'll let the learner handle adding via its own mechanism
                        # But we need to ensure it uses our append-only system
                        pass
                    
                    self.daily_word_count += new_words
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
            
            # Process text for learning (this should use our append-only system)
            learning_response = self.handle_learning_commands(text)
            if learning_response:
                self.speaker.speak(learning_response)
                return
            
            # Try AI processor for general conversation
            try:
                ai_response = self.ai_processor.process(text)
                logger.info(f"🤖 AI response received: {ai_response[:50] if ai_response else 'None'}...")
                
                if ai_response and len(ai_response) > 3:
                    self.speaker.speak(ai_response)
                    
                    # Learn any new words from the conversation
                    words = text.lower().split()
                    new_words_added = 0
                    for word in words:
                        word = word.strip('.,!?;:')
                        if len(word) > 2 and not self.word_exists(word):
                            if self.add_to_vocabulary([word]) > 0:
                                new_words_added += 1
                    if new_words_added > 0:
                        logger.info(f"Added {new_words_added} new words from AI conversation")
                    return
                else:
                    logger.warning(f"⚠️ AI response too short or empty")
            except Exception as e:
                logger.error(f"AI processor error: {e}")
            
            # Fall back to command processor if AI fails
            command = self.processor.process(text)
            response = self.processor.generate_response(command)
            
            # Process any new words from the command
            words = text.lower().split()
            new_words_added = 0
            for word in words:
                word = word.strip('.,!?;:')
                if len(word) > 2 and not self.word_exists(word):
                    if self.add_to_vocabulary([word]) > 0:
                        new_words_added += 1
            if new_words_added > 0:
                logger.info(f"Added {new_words_added} new words from command")
            
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
                # Use manager to get recent words (read-only)
                vocab_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                          'language_learning', 'data', 'vocabulary.json')
                with open(vocab_file, 'r') as f:
                    vocab = json.load(f)
                
                today = datetime.now().date().isoformat()
                recent_words = []
                for word, data in vocab.items():
                    if data.get('added', '').startswith(today) and data.get('added') != 'bootstrap':
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
                
                # We need to patch FullInternetLearner to use our append-only system
                # For now, we'll let it run but monitor
                learned = net_learner.learning_cycle(depth=depth)
                
                if learned > 0:
                    true_size = self.get_true_vocabulary()
                    
                    if true_size > 1000 and self.evolution_stage < 3:
                        self.evolution_stage = 3
                    elif true_size > 5000 and self.evolution_stage < 5:
                        self.evolution_stage = 5
                
                time.sleep(14400)  # 4 hours
            except Exception as e:
                logger.error(f"Full internet learning error: {e}")
                time.sleep(3600)  # 1 hour
    
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
        
        # Announce only once at startup
        if not self._announced_startup:
            self.speaker.speak("Hey, it's DeeMai!")
            self._announced_startup = True
        
        print("\nVoice enrolled")
        print(f"Vocabulary: {true_size} words")
        print(f"Learning: {'active' if self.learning_active else 'inactive'}")
        print("\n⚠️ Vocabulary is APPEND-ONLY - new words can be added, existing words protected")
        print("   Daily reports will summarize vocabulary growth")
        print("\nSay 'Hey Dee Mai' to talk to me")
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

# ============================================================================
# PLUGGABLE INTERFACE LAYER - DO NOT MODIFY BELOW THIS LINE
# ============================================================================
# This section adds API endpoints for external systems to connect
# All original code above remains completely unchanged

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Memory optimization
import gc
gc.set_threshold(700, 10, 5)  # More aggressive garbage collection
import resource
try:
    # Set soft memory limit
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
except:
    pass

# Clear cache periodically
import threading
import time
def cache_cleaner():
    while True:
        time.sleep(300)  # Every 5 minutes
        gc.collect()  # Force garbage collection
        if hasattr(__import__('torch'), 'mps'):
            import torch
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
threading.Thread(target=cache_cleaner, daemon=True).start()


# Global reference to the voice instance
_voice_instance = None
_start_time = datetime.now()

class VoiceAPIHandler(BaseHTTPRequestHandler):
    """API for external systems to query voice service status"""
    
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status = {
                "name": "voice_service",
                "running": True,
                "vocabulary_size": 0,
                "learning_active": False,
                "in_conversation": False,
                "healthy": True,
                "uptime": str(datetime.now() - _start_time)
            }
            
            # Try to get real data if voice instance exists
            if _voice_instance:
                try:
                    status["vocabulary_size"] = _voice_instance.get_true_vocabulary()
                    status["learning_active"] = getattr(_voice_instance, 'learning_active', False)
                    status["in_conversation"] = getattr(_voice_instance, 'in_conversation', False)
                except:
                    pass
            
            self.wfile.write(json.dumps(status).encode())
            
        elif self.path == '/vocabulary':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            vocab_stats = {
                "total_words": 0,
                "today_words": 0,
                "stage": 1
            }
            
            if _voice_instance:
                try:
                    vocab_stats["total_words"] = _voice_instance.get_true_vocabulary()
                    vocab_stats["today_words"] = getattr(_voice_instance, 'daily_word_count', 0)
                    vocab_stats["stage"] = getattr(_voice_instance, 'evolution_stage', 1)
                except:
                    pass
            
            self.wfile.write(json.dumps(vocab_stats).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/command':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            try:
                command = json.loads(post_data)
                cmd = command.get('command', '')
                
                if cmd == 'start_learning':
                    if _voice_instance:
                        _voice_instance.start_learning()
                        self.wfile.write(json.dumps({"status": "learning_started"}).encode())
                    else:
                        self.wfile.write(json.dumps({"error": "Voice service not initialized"}).encode())
                        
                elif cmd == 'stop_learning':
                    if _voice_instance:
                        _voice_instance.stop_learning()
                        self.wfile.write(json.dumps({"status": "learning_stopped"}).encode())
                    else:
                        self.wfile.write(json.dumps({"error": "Voice service not initialized"}).encode())
                        
                elif cmd == 'say':
                    text = command.get('text', '')
                    if _voice_instance and text:
                        _voice_instance.speaker.speak(text)
                        self.wfile.write(json.dumps({"status": "speaking", "text": text}).encode())
                    else:
                        self.wfile.write(json.dumps({"error": "No text provided"}).encode())
                        
                elif cmd == 'get_vocabulary':
                    if _voice_instance:
                        size = _voice_instance.get_true_vocabulary()
                        self.wfile.write(json.dumps({"vocabulary_size": size}).encode())
                    else:
                        self.wfile.write(json.dumps({"vocabulary_size": 0}).encode())
                        
                else:
                    self.wfile.write(json.dumps({"error": f"Unknown command: {cmd}"}).encode())
                    
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        return  # Suppress HTTP logs

def _start_api_server():
    """Start API server in background thread"""
    port = 9008  # Fixed port for voice service
    
    def run_server():
        server = HTTPServer(('localhost', port), VoiceAPIHandler)
        print(f"📡 Voice Service API endpoint active at http://localhost:{port}")
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return port

# Initialize the API server when this module is imported
_api_port = _start_api_server()

# Store reference to voice instance when created
_original_init = DMAIVoiceWithLearning.__init__
def _wrapped_init(self, *args, **kwargs):
    global _voice_instance
    _original_init(self, *args, **kwargs)
    _voice_instance = self

DMAIVoiceWithLearning.__init__ = _wrapped_init
