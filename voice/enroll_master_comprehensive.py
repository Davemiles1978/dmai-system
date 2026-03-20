#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Comprehensive voice enrollment - captures different tones and volumes"""
import sys
import os
import sounddevice as sd
import numpy as np
import time
from voice.auth.voice_auth import VoiceAuth

SAMPLE_RATE = 16000

# Phrases in different styles
ENROLLMENT_PHRASES = [
    # Normal conversational (5 phrases)
    {"text": "my voice is my password", "style": "normal"},
    {"text": "dmai recognize my voice", "style": "normal"},
    {"text": "this is david speaking", "style": "normal"},
    {"text": "access granted to master", "style": "normal"},
    {"text": "I am the creator of DMAI", "style": "normal"},
    
    # Quiet/soft (3 phrases)
    {"text": "whisper this quietly", "style": "quiet"},
    {"text": "speaking in a low voice", "style": "quiet"},
    {"text": "soft and gentle", "style": "quiet"},
    
    # Loud/emphatic (3 phrases)
    {"text": "THIS IS IMPORTANT", "style": "loud"},
    {"text": "EMERGENCY OVERRIDE", "style": "loud"},
    {"text": "COMMAND EXECUTE NOW", "style": "loud"},
    
    # Question/inflection (3 phrases)
    {"text": "are you there DMAI?", "style": "question"},
    {"text": "can you hear me?", "style": "question"},
    {"text": "what's my status?", "style": "question"},
    
    # Command style (3 phrases)
    {"text": "create a video", "style": "command"},
    {"text": "research quantum physics", "style": "command"},
    {"text": "send to my phone", "style": "command"},
    
    # Random extra (3 phrases)
    {"text": "the quick brown fox", "style": "random"},
    {"text": "hello there general kenobi", "style": "random"},
    {"text": "to be or not to be", "style": "random"}
]

def record_phrase(phrase_data, index, total):
    """Record a single phrase with style instructions"""
    phrase = phrase_data["text"]
    style = phrase_data["style"]
    
    print(f"\n📝 Phrase {index}/{total}")
    print(f"🎤 Style: {style.upper()}")
    print(f"🗣️  Say: \"{phrase}\"")
    print("\nGet ready...")
    
    for i in [3, 2, 1]:
        print(f"   {i}...")
        time.sleep(1)
    
    print("🔴 SPEAK NOW!")
    
    recording = sd.rec(int(3 * SAMPLE_RATE), 
                       samplerate=SAMPLE_RATE, 
                       channels=1, 
                       dtype='float32')
    sd.wait()
    
    print("✅ Recorded!")
    time.sleep(2)  # Pause between phrases
    
    return recording.flatten()

def main():
    print("\n" + "="*60)
    print("🎙️  DMAI COMPREHENSIVE VOICE ENROLLMENT")
    print("="*60)
    print(f"\nYou'll speak {len(ENROLLMENT_PHRASES)} phrases in different styles:")
    print("  • Normal (5) - your everyday voice")
    print("  • Quiet (3) - soft/whisper")
    print("  • Loud (3) - emphatic/louder")
    print("  • Question (3) - rising inflection")
    print("  • Command (3) - firm/direct")
    print("  • Random (3) - varied")
    print(f"\nTotal time: ~{len(ENROLLMENT_PHRASES) * 5} seconds")
    
    input("\nPress Enter when ready to start...")
    
    auth = VoiceAuth()
    recordings = []
    
    for i, phrase_data in enumerate(ENROLLMENT_PHRASES, 1):
        try:
            audio = record_phrase(phrase_data, i, len(ENROLLMENT_PHRASES))
            recordings.append(audio)
            print(f"✓ Phrase {i} stored")
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            print("Let's try that again...")
            audio = record_phrase(phrase_data, i, len(ENROLLMENT_PHRASES))
            recordings.append(audio)
    
    print("\n" + "="*60)
    print("🧬 Creating comprehensive voiceprint...")
    
    success = auth.enroll_master(recordings, SAMPLE_RATE)
    
    if success:
        print("\n✅ SUCCESS! DMAI now knows your voice across different tones!")
        print(f"   Used {len(recordings)} samples")
    else:
        print("\n❌ Enrollment failed. Please try again.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
