#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Simple microphone test - no dependencies except pyaudio
"""

import sys
import time
import wave
import tempfile
from pathlib import Path

try:
    import pyaudio
except ImportError:
    print("❌ PyAudio not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaudio"])
    import pyaudio

print("\n🎤 DMAI MICROPHONE TEST")
print("=" * 50)

# Test microphone
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3

p = pyaudio.PyAudio()

# List all audio devices
print("\n📋 Available audio devices:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:  # Only show input devices
        print(f"  Device {i}: {info['name']}")
        print(f"     Input channels: {int(info['maxInputChannels'])}")
        print(f"     Sample rate: {int(info['defaultSampleRate'])} Hz")

# Try to open default input device
print("\n🎤 Testing microphone...")
print(f"Recording for {RECORD_SECONDS} seconds...")
print("Say something now!")

try:
    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=None)  # Use default
    
    print("🔴 Recording...")
    
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        if i % 10 == 0:
            print(".", end="", flush=True)
    
    print("\n✅ Recording complete!")
    
    # Stop and close stream
    stream.stop_stream()
    stream.close()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wf = wave.open(f.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"📁 Saved test recording to: {f.name}")
    
    print("\n✅ Microphone is WORKING!")
    print("   Audio was captured successfully.")
    
except Exception as e:
    print(f"\n❌ Microphone test FAILED: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Check System Settings > Privacy & Security > Microphone")
    print("2. Make sure Terminal has microphone access")
    print("3. Try: sudo kill -9 `ps aux | grep coreaudiod | grep -v grep | awk '{print $2}'`")
    print("4. Restart your Mac")

finally:
    p.terminate()
