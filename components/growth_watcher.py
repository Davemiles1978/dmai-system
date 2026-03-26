#!/usr/bin/env python3
"""Watch DMAI grow up in real-time"""
import time
import json
from datetime import datetime
from pathlib import Path
import threading
import logging

logger = logging.getLogger('dmai_growth_watcher')

class GrowthWatcher:
    """Class-based growth watcher for background monitoring"""
    def __init__(self, data_path="data/evolution", watch_interval=10):
        self.timer_file = Path(data_path) / "timer_state.json"
        self.watch_interval = watch_interval
        self.running = False
        self.thread = None
        self.last_stage = None
        self.stages = []
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()
        logger.info("🌱 Growth watcher started")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _watch_loop(self):
        while self.running:
            try:
                if self.timer_file.exists():
                    with open(self.timer_file) as f:
                        state = json.load(f)
                    current_stage = state.get('current_stage', 'baby')
                    if current_stage != self.last_stage:
                        self.stages.append({
                            'stage': current_stage,
                            'time': datetime.now(),
                            'evolutions': state.get('successful_evolutions', 0)
                        })
                        logger.info(f"🎉 DMAI reached stage: {current_stage.upper()} - Evolutions: {state.get('successful_evolutions', 0)}")
                        self.last_stage = current_stage
                time.sleep(self.watch_interval)
            except Exception as e:
                logger.debug(f"Growth watcher error: {e}")
                time.sleep(self.watch_interval)
    
    def get_growth_path(self):
        return self.stages
    
    def get_current_progress(self):
        if self.timer_file.exists():
            with open(self.timer_file) as f:
                state = json.load(f)
                return {
                    'stage': state.get('current_stage', 'baby'),
                    'evolutions': state.get('successful_evolutions', 0),
                    'success_rate': state.get('average_success_rate', 0)
                }
        return None


def watch_growth():
    """Original function-based growth watcher for command-line use"""
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
            
            if len(stages) > 1:
                progress = []
                for i, stage in enumerate(stages):
                    if i < len(stages) - 1:
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
