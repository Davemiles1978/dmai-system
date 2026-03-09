#!/usr/bin/env python3
"""
Direct test of DMAI voice system
"""

import sys
import time
import logging
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))).parent))

# Try to import voice modules
try:
    from voice.speech_to_text import SpeechToText
    from voice.speaker import DMAISpeaker  # Fixed: using DMAISpeaker
    VOICE_AVAILABLE = True
    print("✅ Voice modules imported successfully")
except ImportError as e:
    print(f"❌ Voice modules not available: {e}")
    VOICE_AVAILABLE = False

print("\n🎤 DMAI DIRECT VOICE TEST")
print("=" * 50)

if not VOICE_AVAILABLE:
    print("\n❌ Voice modules not found. Check that these files exist:")
    print("   - voice/speech_to_text.py")
    print("   - voice/speaker.py")
    sys.exit(1)

# Initialize
print("\n🔄 Initializing speech recognition...")
try:
    stt = SpeechToText()
    print("✅ SpeechToText initialized")
except Exception as e:
    print(f"❌ SpeechToText init failed: {e}")
    stt = None

print("\n🔄 Initializing speaker...")
try:
    speaker = DMAISpeaker()  # Using DMAISpeaker
    print("✅ DMAISpeaker initialized")
except Exception as e:
    print(f"❌ DMAISpeaker init failed: {e}")
    speaker = None

if not stt:
    print("\n❌ Cannot continue without speech recognition")
    sys.exit(1)

print("\n🎤 Testing microphone...")
print("Say something (you have 5 seconds)...")

try:
    # Listen
    text = stt.listen(timeout=5)
    
    if text:
        print(f"\n✅ You said: '{text}'")
        
        # Respond
        response = f"You said: {text}"
        if speaker:
            print(f"🔊 Speaking: {response}")
            speaker.speak(response)
        else:
            print(f"📝 Response: {response}")
    else:
        print("\n❌ No speech detected")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n📝 Next steps:")
print("1. If this test succeeds, the voice system is working")
print("2. Then we can test music commands")
