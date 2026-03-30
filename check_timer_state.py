# check_timer_state.py
#!/usr/bin/env python3
"""
Check the evolution timer state file
"""

import os
import json
from pathlib import Path
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("CHECKING EVOLUTION TIMER STATE")
print("=" * 60)

timer_file = Path("./data/evolution_timer.json")

if timer_file.exists():
    print(f"\n📂 File exists: {timer_file}")
    print(f"   Size: {timer_file.stat().st_size} bytes")
    print(f"   Modified: {datetime.fromtimestamp(timer_file.stat().st_mtime)}")
    
    try:
        with open(timer_file, 'r') as f:
            content = f.read()
        
        print(f"\n📄 Raw content length: {len(content)} chars")
        
        if content:
            print(f"   First 200 chars: {content[:200]}")
            
            try:
                data = json.loads(content)
                print(f"\n✅ Valid JSON!")
                print(f"   Keys: {list(data.keys())}")
                
                # Show relevant state
                if 'stage' in data:
                    print(f"   Stage: {data['stage']}")
                if 'evolutions' in data:
                    print(f"   Evolutions: {data['evolutions']}")
                if 'current_interval_minutes' in data:
                    print(f"   Current interval: {data['current_interval_minutes']} minutes")
                if 'success_rate' in data:
                    print(f"   Success rate: {data['success_rate']}%")
                if 'evolution_history' in data:
                    print(f"   History entries: {len(data['evolution_history'])}")
                
                # Check if the file is valid but maybe missing required fields
                required_fields = ['evolution_history', 'stage', 'evolutions', 'success_rate', 'current_interval_minutes']
                missing = [f for f in required_fields if f not in data]
                if missing:
                    print(f"\n⚠️ Missing fields: {missing}")
                    
            except json.JSONDecodeError as e:
                print(f"\n❌ Invalid JSON: {e}")
                print(f"   Problem at position {e.pos}")
                print(f"   Around: {content[max(0, e.pos-20):e.pos+20]}")
        else:
            print("\n⚠️ File is EMPTY")
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
else:
    print(f"\n📂 File does NOT exist: {timer_file}")
    print("   This is the problem - the timer has no state file!")

print("\n" + "=" * 60)
print("RECOMMENDED ACTION")
print("=" * 60)

if not timer_file.exists() or timer_file.stat().st_size == 0:
    print("\n1. Create a valid initial state file")
    print("2. The timer will then work and dynamically evolve")
    print("3. Run: python3 create_timer_state.py")
else:
    print("\n1. The file exists but may be corrupted")
    print("2. Backup and recreate the state file")
    print("3. Run: python3 fix_timer_state.py")
