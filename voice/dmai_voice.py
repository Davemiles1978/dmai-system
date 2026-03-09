#!/usr/bin/env python3
"""Main DMAI voice interface with real speech"""
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))))))

import time
import logging
import sounddevice as sd
import numpy as np
from voice.wake.wake_detector import WakeWordDetector
from voice.commands.enhanced_processor import EnhancedCommandProcessor
from voice.auth.voice_auth import VoiceAuth
from voice.speech_to_text import SpeechToText
from voice.speaker import DMAISpeaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DMAIVoice:
    """Main voice interface for DMAI"""
    
    def __init__(self, whisper_model="base"):
        self.wake_detector = WakeWordDetector()
        self.processor = EnhancedCommandProcessor()
        self.auth = VoiceAuth()
        self.stt = SpeechToText(model_size=whisper_model)
        self.speaker = DMAISpeaker()
        self.sample_rate = 16000
        
    def on_wake_word(self):
        """Called when wake word detected"""
        self.speaker.speak("Yes, I'm listening...")
        self.listen_for_command()
    
    def listen_for_command(self, timeout=5):
        """Listen for command after wake word"""
        print("🎤 Listening...")
        
        recording = sd.rec(int(timeout * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=1, 
                          dtype='float32')
        sd.wait()
        
        print("🧠 Processing...")
        
        # Verify it's you
        is_you, confidence = self.auth.verify(recording.flatten(), self.sample_rate)
        
        if not is_you:
            self.speaker.speak("I didn't recognize that voice. Please enroll first.")
            return
        
        # Convert speech to text
        print("📝 Transcribing...")
        text = self.stt.transcribe(recording.flatten(), self.sample_rate)
        
        if not text:
            self.speaker.speak("Sorry, I didn't catch that. Could you repeat?")
            return
        
        print(f"You said: \"{text}\"")
        
        # Process command
        command = self.processor.process(text)
        response = self.processor.generate_response(command)
        
        # Speak the response
        self.speaker.speak(response)
        
        # Check if more info needed
        needs_more, question = self.processor.needs_more_info(command)
        if needs_more:
            self.speaker.speak(question)
    
    def run(self):
        """Start DMAI voice interface"""
        print("\n" + "="*50)
        print("🎙️  DMAI VOICE INTERFACE")
        print("="*50)
        
        if 'master' not in self.auth.voiceprints:
            print("\n⚠️  No voice enrolled.")
            print("Run: python voice/enroll_master.py")
            return
        
        self.speaker.speak("I'm ready. Say Hey Dee Mai when you need me.")
        print("\nSay 'Hey Dee Mai' (Ctrl+C to exit)\n")
        
        self.wake_detector.start(callback=self.on_wake_word)

if __name__ == "__main__":
    dmai = DMAIVoice(whisper_model="base")
    try:
        dmai.run()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        dmai.wake_detector.cleanup()
        dmai.speaker.shutdown()
