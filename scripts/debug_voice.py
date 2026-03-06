#!/usr/bin/env python3
import sys
import os
import time
import sounddevice as sd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from voice.wake.wake_detector import WakeWordDetector
    from voice.speech_to_text import SpeechToText
    
    print("🔍 DMAI VOICE DEBUGGER")
    print("="*50)
    
    # Test 1: Check wake word detector
    print("\n1. Testing Wake Word Detector...")
    detector = WakeWordDetector()
    print("   ✅ Wake word detector initialized")
    
    # Test 2: Check microphone
    print("\n2. Testing Microphone Input...")
    print("   Recording for 2 seconds... Speak now!")
    recording = sd.rec(int(2 * 16000), samplerate=16000, channels=1)
    sd.wait()
    amp = np.max(np.abs(recording))
    print(f"   Max amplitude: {amp}")
    
    if amp > 0.1:
        print("   ✅ Microphone working well")
        
        # Test 3: Try speech to text
        print("\n3. Testing Speech Recognition...")
        print("   Recording for 3 seconds... Say something!")
        recording = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
        sd.wait()
        
        stt = SpeechToText()
        text = stt.transcribe(recording.flatten(), 16000)
        if text:
            print(f"   ✅ Heard: '{text}'")
        else:
            print("   ❌ No speech detected")
    else:
        print("   ❌ Microphone too quiet - check permissions")
        
except Exception as e:
    print(f"❌ Error: {e}")
