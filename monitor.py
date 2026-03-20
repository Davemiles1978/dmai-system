#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Monitor the evolution system
Shows real-time progress and stats
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime

class EvolutionMonitor:
    def __init__(self):
        self.checkpoints_path = Path.cwd() / "checkpoints"
        self.history_file = self.checkpoints_path / "evolution_history.json"
    
    def watch(self):
        """Continuously watch evolution progress"""
        print("\n👁️  EVOLUTION MONITOR - Watching for changes...")
        print("=" * 60)
        
        last_size = 0
        
        try:
            while True:
                os.system('clear')  # Clear screen
                
                print(f"\n📊 EVOLUTION STATUS - {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 60)
                
                # Show checkpoints
                checkpoints = list(self.checkpoints_path.glob("generation_*"))
                print(f"📁 Checkpoints: {len(checkpoints)}")
                
                if checkpoints:
                    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    print(f"🕐 Latest: {latest.name}")
                
                # Show recent evolution history
                if self.history_file.exists():
                    with open(self.history_file, 'r') as f:
                        history = json.load(f)
                    
                    if history:
                        print(f"\n🔄 Recent Evolutions (last 5):")
                        for entry in history[-5:]:
                            score = entry.get('score', 0)
                            file = Path(entry['file']).name
                            print(f"  • Gen {entry['generation']}: {file} - {score:.3f}")
                
                # Show best scores
                best_file = self.checkpoints_path / "best_versions"
                if best_file.exists():
                    print(f"\n🏆 Best Versions:")
                    for repo_dir in best_file.iterdir():
                        if repo_dir.is_dir():
                            files = list(repo_dir.glob("*.py"))
                            if files:
                                print(f"  • {repo_dir.name}: {len(files)} files")
                
                time.sleep(5)  # Update every 5 seconds
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")

if __name__ == "__main__":
    monitor = EvolutionMonitor()
    monitor.watch()
