#!/usr/bin/env python3
import sys
import os

print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

try:
    import harvester
    print("✅ Harvester module found!")
    print(f"Harvester path: {harvester.__file__}")
except ImportError as e:
    print(f"❌ Could not import harvester: {e}")
    
try:
    import config
    print("✅ Config module found!")
    print(f"Config path: {config.__file__}")
except ImportError as e:
    print(f"❌ Could not import config: {e}")
