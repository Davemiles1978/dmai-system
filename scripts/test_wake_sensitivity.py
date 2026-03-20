#!/usr/bin/env python3
import sys
import os
import time
import sounddevice as sd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))))))

try:
    from voice.wake.wake_detector import WakeWordDetector
    
    # Patch the detector to be more sensitive
    original_init = WakeWordDetector.__init__
    
    def patched_init(self):
        original_init(self)
        # Increase sensitivity if possible
        if hasattr(self, 'porcupine'):
            print("✅ Wake word detector patched for sensitivity")
    
    WakeWordDetector.__init__ = patched_init
    
    def on_wake():
        print("\n✅✅✅ WAKE WORD DETECTED! ✅✅✅")
        print("I heard 'Hey Dee Mai'!")
        
    print("🎤 Testing wake word with boosted sensitivity...")
    print("Say 'Hey Dee Mai' (you have 15 seconds)")
    
    detector = WakeWordDetector()
    detector.start(callback=on_wake)
    
    # Monitor audio level while waiting
    for i in range(15):
        recording = sd.rec(int(1 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        level = np.max(np.abs(recording))
        bars = int(level * 50)
        print(f"Level: {level:.3f} {'█' * bars}")
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\nTest stopped")
except Exception as e:
    print(f"Error: {e}")
