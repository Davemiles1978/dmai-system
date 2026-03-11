#!/usr/bin/env python3
"""Watch DMAI grow up in real-time"""
import time
import json
from datetime import datetime
from pathlib import Path

def watch_growth():
    """Monitor DMAI's evolution progress"""
    timer_file = Path("data/evolution/timer_state.json")
    
    stages = []
    last_stage = None
    
    print("\n" + "="*70)
    print("👶 WATCHING DMAI GROW UP")
    print("="*70)
    
    while True:
        if timer_file.exists():
            with open(timer_file) as f:
                state = json.load(f)
            
            current_stage = state['current_stage']
            
            if current_stage != last_stage:
                stages.append({
                    'stage': current_stage,
                    'time': datetime.now(),
                    'evolutions': state['successful_evolutions']
                })
                
                print(f"\n🎉 {datetime.now().strftime('%H:%M:%S')} - Reached: {current_stage.upper()}")
                print(f"   Evolutions: {state['successful_evolutions']}")
                print(f"   Success Rate: {(state['successful_evolutions']/max(1,state['total_attempts']))*100:.1f}%")
                
                last_stage = current_stage
            
            # Show progress bar
            if len(stages) > 1:
                progress = []
                for i, stage in enumerate(stages):
                    if i < len(stages) - 1:
                        next_stage = stages[i+1]
                        emoji = {
                            'baby': '👶', 'toddler': '🧒', 'child': '🧑',
                            'teen': '🧑‍🎤', 'young_adult': '👨‍💼', 'adult': '👨‍🔬',
                            'elder': '🧙'
                        }.get(stage['stage'], '🔮')
                        progress.append(emoji)
                print(f"\nGrowth: {' → '.join(progress)} ⟶ 🧠")
        
        time.sleep(10)

if __name__ == "__main__":
    try:
        watch_growth()
    except KeyboardInterrupt:
        print("\n\n📊 Growth tracking paused")
