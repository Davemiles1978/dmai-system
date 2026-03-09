#!/usr/bin/env python3
"""
Fresh voice test - forces reload of modules
"""

import sys
import importlib
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))).parent))

# Force reload of voice modules
if 'voice.speech_to_text' in sys.modules:
    del sys.modules['voice.speech_to_text']
if 'voice.speaker' in sys.modules:
    del sys.modules['voice.speaker']

# Import fresh
from voice.speech_to_text import SpeechToText
from voice.speaker import DMAISpeaker

print("\n🎤 DMAI VOICE TEST - FRESH START")
print("=" * 50)

# Test if listen method exists
print("\n🔍 Checking SpeechToText methods...")
methods = [m for m in dir(SpeechToText) if not m.startswith('_')]
print(f"Available methods: {methods}")

if 'listen' in methods:
    print("✅ 'listen' method found!")
else:
    print("❌ 'listen' method NOT found!")
    print("\nUpdating speech_to_text.py with listen method...")
    
    # Update the file with the correct version
    with open('/Users/davidmiles/Desktop/dmai-system/voice/speech_to_text.py', 'r') as f:
        content = f.read()
    
    if 'def listen' not in content:
        print("Please run the previous command to update speech_to_text.py")
        sys.exit(1)

# Initialize
print("\n🔄 Initializing speech recognition...")
stt = SpeechToText()
print("✅ SpeechToText initialized")

print("\n🔄 Initializing speaker...")
speaker = DMAISpeaker()
print("✅ DMAISpeaker initialized")

speaker.speak("Voice test started. Please say something.")

print("\n🎤 Listening for 5 seconds...")
print("Say something now!")

try:
    text = stt.listen(timeout=5)
    
    if text:
        print(f"\n✅ You said: '{text}'")
        speaker.speak(f"You said: {text}")
    else:
        print("\n❌ No speech detected")
        speaker.speak("I didn't hear anything")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test complete")
