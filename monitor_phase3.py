#!/usr/bin/env python3
"""
Monitor Phase 3: Capability Synthesis Progress
"""
import json
from pathlib import Path
from datetime import datetime

def monitor_synthesis():
    caps_dir = Path("shared_data/agi_evolution/capabilities")
    patterns_dir = Path("shared_data/agi_evolution/patterns")
    synthesis_dir = Path("shared_data/agi_evolution/synthesis")
    
    print(f"\n{'='*60}")
    print(f"🧬 PHASE 3 MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Count capabilities
    if caps_dir.exists():
        caps = list(caps_dir.glob("*.json"))
        print(f"\n📊 Total Capabilities: {len(caps)}")
        
        # Show hybrid capabilities
        hybrids = [c for c in caps if 'hybrid' in c.name]
        if hybrids:
            print(f"  ✨ Hybrids: {len(hybrids)}")
            for h in hybrids[-3:]:  # Show last 3
                print(f"    • {h.name}")
    
    # Check discovered combinations
    combos_file = patterns_dir / "discovered_combinations.json"
    if combos_file.exists():
        with open(combos_file, 'r') as f:
            combos = json.load(f)
        print(f"\n🔍 Discovered Combinations: {len(combos)}")
    
    # Check synergy optimizations
    if synthesis_dir.exists():
        synergies = list(synthesis_dir.glob("synergy_*.json"))
        if synergies:
            print(f"\n⚡ Synergy Optimizations: {len(synergies)}")
    
    # Check for cross-domain adapters
    adapters = [c for c in caps if 'adapter' in c.name] if caps_dir.exists() else []
    if adapters:
        print(f"\n🔄 Cross-Domain Adapters: {len(adapters)}")
    
    print(f"\n{'='*60}")
    print("Next synthesis cycle runs every 3 generations")
    print(f"{'='=60}")

if __name__ == "__main__":
    monitor_synthesis()
