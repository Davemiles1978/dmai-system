#!/usr/bin/env python3
"""Test voice recognition"""
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))))))

import sounddevice as sd
import numpy as np
import time
from voice.auth.voice_auth import VoiceAuth
from voice.wake.wake_detector import WakeWordDetector, on_wake

SAMPLE_RATE = 16000

def test_verification():
    """Test if DMAI recognizes your voice"""
    print("\n🔊 Testing voice verification")
    print("Say something (I'll check if it's YOU)")
    
    input("\nPress Enter and speak...")
    
    # Record 3 seconds
    print("Recording... speak now!")
    recording = sd.rec(int(3 * SAMPLE_RATE), 
                       samplerate=SAMPLE_RATE, 
                       channels=1, 
                       dtype='float32')
    sd.wait()
    
    # Verify
    auth = VoiceAuth()
    is_match, confidence = auth.verify(recording.flatten(), SAMPLE_RATE)
    
    print(f"\nConfidence: {confidence:.2%}")
    if is_match:
        print("✅ Voice recognized! It's you, David.")
    else:
        print("❌ Voice not recognized. Run enroll_master.py first.")
    
    return is_match

def test_wake_word():
    """Test wake word detection"""
    print("\n🔊 Testing wake word detection")
    print("Say 'Hey DMAI' when ready (or 'computer' as fallback)")
    
    detector = WakeWordDetector()
    if detector.initialize():
        print("Listening... (Ctrl+C to stop)")
        try:
            detector.start(callback=on_wake)
        except KeyboardInterrupt:
            detector.cleanup()
    else:
        print("Failed to initialize wake detector")

def test_listen():
    """Simple test to check microphone"""
    print("\n🎤 Testing microphone input")
    print("Speak for 3 seconds...")
    
    recording = sd.rec(int(3 * SAMPLE_RATE), 
                       samplerate=SAMPLE_RATE, 
                       channels=1, 
                       dtype='float32')
    sd.wait()
    
    print(f"Recorded {len(recording)} samples")
    print(f"Audio level: max={np.max(np.abs(recording)):.3f}, mean={np.mean(np.abs(recording)):.3f}")
    
    if np.max(np.abs(recording)) > 0.01:
        print("✅ Microphone working!")
    else:
        print("❌ Microphone may not be working - very low audio level")

def main():
    print("="*50)
    print("🎙️  DMAI VOICE TEST")
    print("="*50)
    
    while True:
        print("\nOptions:")
        print("1. Test microphone")
        print("2. Test voice verification (am I David?)")
        print("3. Test wake word detection")
        print("4. Exit")
        
        choice = input("\nChoice (1-4): ").strip()
        
        if choice == "1":
            test_listen()
        elif choice == "2":
            test_verification()
        elif choice == "3":
            test_wake_word()
        elif choice == "4":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
