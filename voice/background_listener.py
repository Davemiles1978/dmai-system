#!/usr/bin/env python3
"""DMAI Background Listener - Always ready, never sleeps"""
import sys
import os
import time
import logging
import json
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.wake.wake_detector import WakeWordDetector
from voice.auth.voice_auth import VoiceAuth
from voice.commands.enhanced_processor import EnhancedCommandProcessor
from voice.speech_to_text import SpeechToText
from voice.personality_evolution import EvolvingPersonality

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'dmai_background.log')),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger('DMAI_Background')

class DMAIBackground:
    """Runs in background, always listening for wake word"""
    
    def __init__(self):
        self.running = True
        self.wake_detector = WakeWordDetector()
        self.auth = VoiceAuth()
        self.processor = EnhancedCommandProcessor()
        self.stt = SpeechToText(model_size="tiny")  # Tiny for background, wakes full model
        self.personality = EvolvingPersonality()
        self.listening_for_command = False
        self.last_heartbeat = time.time()
        
    def on_wake_word(self):
        """Called when wake word detected"""
        logger.info("🚀 Wake word detected! Activating...")
        print("\n🔊 DMAI: Yes, I'm listening...")
        self.handle_command()
    
    def handle_command(self):
        """Handle a voice command"""
        import sounddevice as sd
        import numpy as np
        
        try:
            # Record command
            print("🎤 Listening for command...")
            recording = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
            sd.wait()
            
            # Verify it's you
            is_you, confidence = self.auth.verify(recording.flatten(), 16000)
            
            if not is_you:
                logger.warning(f"Voice verification failed: {confidence:.2%}")
                print("⚠️  Voice not recognized.")
                return
            
            logger.info(f"✅ Voice verified: {confidence:.2%}")
            
            # Transcribe
            print("📝 Transcribing...")
            text = self.stt.transcribe(recording.flatten(), 16000)
            
            if not text:
                print("🤖 DMAI: Sorry, I didn't catch that.")
                return
            
            print(f"You: {text}")
            
            # Process command
            command = self.processor.process(text)
            response = self.processor.generate_response(command)
            
            # Get current name
            name = self.processor.prefs.get_name()
            
            # Evolve personality from interaction
            self.personality.evolve(
                interaction_type=command['intent'],
                user_reaction="positive"  # Assume positive for now
            )
            
            # Speak response
            print(f"🤖 DMAI: {response}")
            
            # Handle follow-ups
            needs_more, question = self.processor.needs_more_info(command)
            if needs_more:
                print(f"🤖 DMAI: {question}")
                # Could handle follow-up here
            
        except Exception as e:
            logger.error(f"Error handling command: {e}")
    
    def heartbeat(self):
        """Log heartbeat every hour to show it's alive"""
        now = time.time()
        if now - self.last_heartbeat > 3600:  # 1 hour
            logger.info(f"❤️  DMAI heartbeat - {datetime.now().isoformat()}")
            self.last_heartbeat = now
    
    def run(self):
        """Start background listener"""
        logger.info("="*50)
        logger.info("🚀 DMAI Background Listener Starting")
        logger.info("="*50)
        
        # Check if voice enrolled
        if 'master' not in self.auth.voiceprints:
            logger.error("❌ No voice enrolled! Run voice/enroll_master.py first")
            print("\n❌ No voice enrolled! Please run:")
            print("   python voice/enroll_master.py")
            return
        
        logger.info(f"✅ Voice enrolled for: {self.processor.prefs.get_name()}")
        logger.info(f"🎭 Personality: {self.personality.get_personality_summary()['style']}")
        logger.info(f"👂 Listening for 'Hey DMAI'...")
        
        # Start wake word detection
        self.wake_detector.start(callback=self.on_wake_word)
        
        # Keep running
        try:
            while self.running:
                time.sleep(60)  # Check every minute
                self.heartbeat()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.wake_detector.cleanup()
            logger.info("DMAI background listener stopped")

if __name__ == "__main__":
    listener = DMAIBackground()
    listener.run()
