#!/usr/bin/env python3
"""Main DMAI voice interface with real speech recognition"""
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent)))))

import time
import logging
import sounddevice as sd
import numpy as np
from voice.wake.wake_detector import WakeWordDetector
from voice.commands.command_processor import CommandProcessor
from voice.auth.voice_auth import VoiceAuth
from voice.speech_to_text import SpeechToText

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DMAIVoice:
    """Main voice interface for DMAI"""
    
    def __init__(self, whisper_model="base"):
        self.wake_detector = WakeWordDetector()
        self.processor = CommandProcessor()
        self.auth = VoiceAuth()
        self.stt = SpeechToText(model_size=whisper_model)
        self.listening = False
        self.sample_rate = 16000
        
    def on_wake_word(self):
        """Called when wake word detected"""
        print("\n🔊 DMAI: Yes, I'm listening...")
        self.listen_for_command()
    
    def listen_for_command(self, timeout=5):
        """Listen for command after wake word using real STT"""
        print("🎤 Speak your command...")
        
        # Record command
        recording = sd.rec(int(timeout * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=1, 
                          dtype='float32')
        sd.wait()
        
        print("🧠 Processing...")
        
        # Verify it's you (voice biometrics)
        is_you, confidence = self.auth.verify(recording.flatten(), self.sample_rate)
        
        if not is_you:
            print(f"⚠️  Voice not recognized (confidence: {confidence:.1%}). Please enroll first.")
            return
        
        # Convert speech to text
        print("📝 Transcribing...")
        text = self.stt.transcribe(recording.flatten(), self.sample_rate)
        
        if not text:
            print("🤖 DMAI: Sorry, I didn't catch that. Could you repeat?")
            return
        
        print(f"You said: \"{text}\"")
        
        # Process command
        command = self.processor.process(text)
        response = self.processor.generate_response(command)
        
        print(f"\n🤖 DMAI: {response}")
        
        # Check if more info needed
        needs_more, question = self.processor.needs_more_info(command)
        if needs_more:
            print(f"🤖 DMAI: {question}")
            # Here we could listen again for clarification
    
    def run(self):
        """Start DMAI voice interface"""
        print("\n" + "="*50)
        print("🎙️  DMAI VOICE INTERFACE")
        print("="*50)
        print("\nInitializing...")
        
        # Check if master is enrolled
        if 'master' not in self.auth.voiceprints:
            print("\n⚠️  No voice enrolled yet.")
            print("Please run: python voice/enroll_master.py")
            print("(Do this when you're alone for security)")
            return
        
        print("\n✅ Voice enrolled. Ready to listen.")
        print(f"✅ Using Whisper model: {self.stt.model_size}")
        print("\nSay 'Hey DMAI' when you need me.")
        print("Press Ctrl+C to exit.\n")
        
        # Start wake word detection
        self.wake_detector.start(callback=self.on_wake_word)

if __name__ == "__main__":
    # You can change model: "tiny" (fastest) to "large" (most accurate)
    dmai = DMAIVoice(whisper_model="base")
    try:
        dmai.run()
    except KeyboardInterrupt:
        print("\n\nDMAI shutting down...")
        dmai.wake_detector.cleanup()
