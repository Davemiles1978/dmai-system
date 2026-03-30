# create_correct_timer_state.py
#!/usr/bin/env python3
"""
Create the correct evolution timer state using the original timer's logic
This preserves the dynamic timing and stage progression
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'components'))

print("=" * 60)
print("CREATING CORRECT EVOLUTION TIMER STATE")
print("=" * 60)

# Import the adaptive timer from your original component
try:
    from components.evolution_timer import AdaptiveEvolutionTimer
    print("✅ Imported AdaptiveEvolutionTimer")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    # Fallback - create the state manually using the original structure
    print("   Creating state manually...")
    
    timer_file = Path("./data/evolution_timer.json")
    timer_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Original timer's state structure
    state = {
        "creation_date": datetime.now().isoformat(),
        "successful_evolutions": 0,
        "failed_attempts": 0,
        "current_stage": "baby",
        "evolution_history": [],
        "average_success_rate": 0.0,
        "last_adjustment": datetime.now().isoformat(),
        "current_interval": 600,  # 10 minutes
        "total_attempts": 0,
        "preferred_pairs": {},
        "learning_rate": 1.0
    }
    
    with open(timer_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"✅ Created timer state at {timer_file}")
    print(f"   Stage: baby")
    print(f"   Interval: 10 minutes")
    sys.exit(0)

# Create the timer with a fresh data path
data_path = Path("./data")
data_path.mkdir(exist_ok=True)

print("\n📝 Creating timer instance...")
try:
    timer = AdaptiveEvolutionTimer(timer_file=str(data_path / "evolution_timer.json"))
    print("   Timer created successfully")
    
    # Show the current state
    info = timer.get_stage_info()
    print(f"\n📊 Timer State:")
    print(f"   Stage: {info['name']}")
    print(f"   Description: {info['description']}")
    print(f"   Evolutions: {info['evolutions']}")
    print(f"   Success Rate: {info['success_rate']}")
    print(f"   Interval: {info['interval_minutes']} minutes")
    
    # Save to ensure file exists
    timer.save_state()
    
    timer_file = data_path / "evolution_timer.json"
    if timer_file.exists():
        print(f"\n✅ Timer file created: {timer_file}")
        print(f"   Size: {timer_file.stat().st_size} bytes")
        
        # Verify the content
        with open(timer_file, 'r') as f:
            content = json.load(f)
        print(f"   Keys: {list(content.keys())}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
