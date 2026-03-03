#!/usr/bin/env python3
"""Test speech-to-text with Whisper"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sounddevice as sd
import numpy as np
from speech_to_text import SpeechToText

SAMPLE_RATE = 16000

def test_microphone():
    """Test recording and transcription"""
    print("\n🎤 Testing microphone + Whisper transcription")
    print("Speak a clear sentence when prompted")
    
    input("\nPress Enter and speak now...")
    
    # Record
    print("Recording... (3 seconds)")
    recording = sd.rec(int(3 * SAMPLE_RATE), 
                       samplerate=SAMPLE_RATE, 
                       channels=1, 
                       dtype='float32')
    sd.wait()
    
    print("Transcribing...")
    
    # Try different model sizes
    for model in ["tiny", "base"]:  # Start with fastest
        print(f"\nUsing {model} model:")
        stt = SpeechToText(model_size=model)
        text = stt.transcribe(recording.flatten(), SAMPLE_RATE)
        print(f"→ '{text}'")

def main():
    print("="*50)
    print("🎙️  DMAI SPEECH-TO-TEXT TEST")
    print("="*50)
    
    # First download will take a minute
    print("\nFirst run: Downloading Whisper model (one-time only)...")
    stt = SpeechToText(model_size="tiny")
    print("✓ Model ready")
    
    while True:
        print("\nOptions:")
        print("1. Test transcription (speak now)")
        print("2. Exit")
        
        choice = input("\nChoice (1-2): ").strip()
        
        if choice == "1":
            test_microphone()
        elif choice == "2":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
