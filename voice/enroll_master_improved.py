#!/usr/bin/env python3
"""Improved voice enrollment - more phrases, longer pauses"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sounddevice as sd
import numpy as np
import time
from voice.auth.voice_auth import VoiceAuth

# Configuration
SAMPLE_RATE = 16000
PHRASE_DURATION = 4  # Longer recording (was 3)
PAUSE_BETWEEN = 3    # 3 second pause between phrases

ENROLLMENT_PHRASES = [
    "my voice is my password",
    "dmai recognize my voice",
    "this is david speaking",
    "access granted to master",
    "I am the creator of DMAI",
    "hello DMAI this is David",
    "my name is David",
    "you know my voice",
    "only I can command you",
    "this is my final enrollment"
]

def record_phrase(phrase, index, total):
    """Record a single phrase with countdown"""
    print(f"\n📝 Phrase {index}/{total}")
    print(f"🎤 Say: \"{phrase}\"")
    
    # Countdown
    for i in [3, 2, 1]:
        print(f"   {i}...")
        time.sleep(1)
    
    print("🔴 SPEAK NOW!")
    
    recording = sd.rec(int(PHRASE_DURATION * SAMPLE_RATE), 
                       samplerate=SAMPLE_RATE, 
                       channels=1, 
                       dtype='float32')
    sd.wait()
    
    print("✅ Recorded!")
    print(f"⏸️  Resting for {PAUSE_BETWEEN} seconds...")
    time.sleep(PAUSE_BETWEEN)
    
    return recording.flatten()

def main():
    print("\n" + "="*60)
    print("🎙️  DMAI IMPROVED VOICE ENROLLMENT")
    print("="*60)
    print(f"\nYou'll speak {len(ENROLLMENT_PHRASES)} phrases.")
    print(f"Each phrase: {PHRASE_DURATION} seconds")
    print(f"Pause between: {PAUSE_BETWEEN} seconds")
    print(f"Total time: ~{len(ENROLLMENT_PHRASES) * (PHRASE_DURATION + PAUSE_BETWEEN)} seconds")
    print("\n🎯 Tips:")
    print("  • Speak in your normal voice")
    print("  • Natural pace, don't rush")
    print("  • Same environment you'll use DMAI")
    print("  • Take your time between phrases")
    
    # List available microphones
    print("\n🎤 Available audio devices:")
    print(sd.query_devices())
    
    input("\nPress Enter when ready to start...")
    
    # Initialize auth
    auth = VoiceAuth()
    
    # Record all phrases
    recordings = []
    
    for i, phrase in enumerate(ENROLLMENT_PHRASES, 1):
        try:
            audio = record_phrase(phrase, i, len(ENROLLMENT_PHRASES))
            recordings.append(audio)
            print(f"✓ Phrase {i} stored")
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            print("Let's try that phrase again...")
            audio = record_phrase(phrase, i, len(ENROLLMENT_PHRASES))
            recordings.append(audio)
    
    print("\n" + "="*60)
    print("🧬 Creating your enhanced voiceprint...")
    print("(This may take a moment...)")
    
    # Create voiceprint
    try:
        success = auth.enroll_master(recordings, SAMPLE_RATE)
        
        if success:
            print("\n✅ SUCCESS! DMAI now knows your voice better!")
            print(f"   Used {len(recordings)} phrases for training")
            print("\nYou can now talk to me by saying:")
            print("  \"Hey Dee Mai\" (wake me up)")
            print("  Then speak your command")
            print("\nTry: 'Hey Dee Mai, what's my status?'")
        else:
            print("\n❌ Enrollment failed. Please try again.")
    except Exception as e:
        print(f"\n❌ Error during enrollment: {e}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
