#!/usr/bin/env python3
"""Enroll master user voice for DMAI"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sounddevice as sd
import numpy as np
import time
from voice.auth.voice_auth import VoiceAuth

# Configuration
SAMPLE_RATE = 16000
ENROLLMENT_PHRASES = [
    "my voice is my password",
    "dmai recognize my voice",
    "this is david speaking",
    "access granted to master",
    "I am the creator of DMAI"
]

def record_phrase(phrase, duration=3):
    """Record a single phrase"""
    print(f"\n🎤 Say: \"{phrase}\"")
    print("Get ready...", end='', flush=True)
    time.sleep(1)
    print(" SPEAK NOW!")
    
    recording = sd.rec(int(duration * SAMPLE_RATE), 
                       samplerate=SAMPLE_RATE, 
                       channels=1, 
                       dtype='float32')
    sd.wait()
    
    print("✓ Recorded!")
    return recording.flatten()

def main():
    print("\n" + "="*50)
    print("🎙️  DMAI VOICE ENROLLMENT")
    print("="*50)
    print("\nI need to learn your voice so I always know it's YOU.")
    print(f"You'll speak {len(ENROLLMENT_PHRASES)} phrases. This takes about 2 minutes.")
    print("\nMake sure you're in a quiet place.")
    
    # List available microphones
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    
    input("\nPress Enter when ready to start...")
    
    # Initialize auth
    auth = VoiceAuth()
    
    # Record all phrases
    recordings = []
    
    for i, phrase in enumerate(ENROLLMENT_PHRASES, 1):
        print(f"\n--- Phrase {i}/{len(ENROLLMENT_PHRASES)} ---")
        try:
            audio = record_phrase(phrase)
            recordings.append(audio)
            print(f"✓ Phrase {i} recorded")
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            print("Let's try that phrase again...")
            audio = record_phrase(phrase)
            recordings.append(audio)
        
        time.sleep(0.5)  # Pause between recordings
    
    print("\n" + "="*50)
    print("Creating your unique voiceprint...")
    
    # Create voiceprint
    try:
        success = auth.enroll_master(recordings, SAMPLE_RATE)
        
        if success:
            print("\n✅ SUCCESS! DMAI now knows your voice.")
            print("\nYou can now talk to me by saying:")
            print("  \"Hey DMAI\" (wake me up)")
            print("  Then speak your command")
            print("\nTry: 'Hey DMAI, what's my status?'")
        else:
            print("\n❌ Enrollment failed. Please try again.")
    except Exception as e:
        print(f"\n❌ Error during enrollment: {e}")
        print("Please check microphone permissions and try again.")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
