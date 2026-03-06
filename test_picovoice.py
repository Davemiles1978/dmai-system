#!/usr/bin/env python3
"""
Test DMAI voice with Picovoice
"""

import sys
import time
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from voice.wake.wake_detector import WakeWordDetector
    from voice.speaker import DMAISpeaker
    from voice.commands.music_commands import MusicCommands
    print("✅ Picovoice modules imported")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

print("\n🎤 DMAI PICOVOICE TEST")
print("=" * 50)

# Initialize speaker
print("\n🔊 Initializing speaker...")
speaker = DMAISpeaker()
speaker.speak("Picovoice test started")

# Initialize music commands
print("\n🎵 Initializing music commands...")
music = MusicCommands()

# Test wake word detector
print("\n🔊 Initializing wake word detector...")
print("Wake word is: 'Jarvis'")
print("Say 'Jarvis' then a command")

def on_wake_word():
    """Called when wake word detected"""
    print("\n✅ Wake word detected!")
    speaker.speak("Yes, I'm listening")
    
    # Now listen for command (simplified - in real system this would use Rhino)
    print("Listening for command...")
    time.sleep(2)  # Simulate listening
    speaker.speak("I heard you")

try:
    # Initialize wake word detector
    detector = WakeWordDetector(callback=on_wake_word)
    print("✅ Wake word detector initialized")
    
    print("\n🎤 Listening for 'Jarvis'...")
    print("Press Ctrl+C to exit")
    
    # Keep running
    while True:
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\n\n🛑 Test stopped")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test complete")
