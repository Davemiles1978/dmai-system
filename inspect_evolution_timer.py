# inspect_evolution_timer.py
#!/usr/bin/env python3
"""
Inspect the evolution timer module to understand its structure
"""

import os
import sys
import json
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'components'))

print("=" * 60)
print("INSPECTING EVOLUTION TIMER")
print("=" * 60)

# Try to import and inspect the timer
try:
    from components.evolution_timer import AdaptiveEvolutionTimer
    print("✅ Successfully imported AdaptiveEvolutionTimer")
    
    # Show what methods exist
    print("\n📋 Available methods:")
    for attr in dir(AdaptiveEvolutionTimer):
        if not attr.startswith('_'):
            print(f"   - {attr}")
    
    # Show what the timer expects
    print("\n🔍 The timer loads from data/evolution_timer.json")
    print("   This file should contain evolution history and stage data")
    print("   Without it, the timer can't function properly")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    
    # Look at the file directly
    timer_file = Path("./data/evolution_timer.json")
    if timer_file.exists():
        print(f"\n📂 Timer file exists at: {timer_file}")
        print(f"   Size: {timer_file.stat().st_size} bytes")
        try:
            with open(timer_file, 'r') as f:
                content = f.read()
            if content:
                print(f"   Content: {content[:200]}")
                try:
                    data = json.loads(content)
                    print(f"   Valid JSON with keys: {list(data.keys())}")
                except json.JSONDecodeError as e:
                    print(f"   ❌ Invalid JSON: {e}")
            else:
                print("   File is EMPTY")
        except Exception as e:
            print(f"   Error reading: {e}")
    else:
        print(f"\n📂 Timer file does NOT exist")
