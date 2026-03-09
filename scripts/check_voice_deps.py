#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Check all voice module dependencies"""

modules = [
    'sounddevice',
    'pvporcupine',
    'pvrecorder',
    'scipy',
    'numpy',
    'requests',
    'whisper',
    'torch',
    'langdetect',
    'json',
    'threading',
    'time',
    'logging',
    'subprocess',
    'datetime'
]

print("🎤 Checking voice dependencies...")
print("=" * 40)

missing = []
for module in modules:
    try:
        __import__(module)
        print(f"✅ {module}")
    except ImportError as e:
        print(f"❌ {module} - {e}")
        missing.append(module)

print("=" * 40)
if missing:
    print(f"\nMissing modules: {missing}")
    print("Install with: pip install " + " ".join(missing))
else:
    print("\n✅ All voice dependencies satisfied!")
