#!/usr/bin/env python3
"""System-Wide Evolution - Every system evolves together"""

import json
import random
import itertools
from pathlib import Path
from datetime import datetime
from adaptive_timer import AdaptiveEvolutionTimer

class SystemWideEvolution:
    """
    Evolves ALL systems in the ecosystem, not just DMAI.
    Every cycle improves the entire AI family.
    """
    
    def __init__(self):
        self.timer = AdaptiveEvolutionTimer()
        self.systems_dir = Path("agents")
        self.evolved_dir = self.systems_dir / "evolved"
        self.evolved_dir.mkdir(exist_ok=True)
        
    def get_all_systems(self):
        """Get all evolvable systems (including DMAI core)"""
        systems = []
        
        # Add DMAI core as a system
        systems.append({
            "name": "DMAI_core",
            "path": "core",
            "type": "core",
            "generation": self._get_system_generation("DMAI_core"),
            "capabilities": self._get_system_capabilities("DMAI_core")
        })
        
        # Add all base systems from /agents
        if self.systems_dir.exists():
            for item in self.systems_dir.iterdir():
                if item.is_dir() and item.name not in ["evolved", "discarded"]:
                    systems.append({
                        "name": item.name,
                        "path": str(item),
                        "type": "base",
                        "generation": self._get_system_generation(item.name),
                        "capabilities": self._get_system_capabilities(item.name)
                    })
        
        # Add all evolved systems
        evolved_path = self.systems_dir / "evolved"
        if evolved_path.exists():
            for item in evolved_path.iterdir():
                if item.is_dir():
                    systems.append({
                        "name": f"evolved_{item.name}",
                        "path": str(item),
                        "type": "evolved",
                        "generation": self._get_system_generation(item.name),
                        "capabilities": self._get_system_capabilities(item.name)
                    })
        
        return systems
    
    def _get_system_generation(self, system_name):
        """Get current generation of a system"""
        gen_file = Path(f"data/evolution/generations/{system_name}.json")
        if gen_file.exists():
            with open(gen_file) as f:
                return json.load(f).get("generation", 1)
        return 1
    
    def _get_system_capabilities(self, system_name):
        """Get system capabilities from manifest"""
        manifest_paths = [
            Path(f"agents/{system_name}/manifest.json"),
            Path(f"agents/evolved/{system_name}/manifest.json"),
            Path("core/manifest.json") if system_name == "DMAI_core" else None
        ]
        
        for path in manifest_paths:
            if path and path.exists():
                with open(path) as f:
                    return json.load(f).get("capabilities", [])
        
        return ["unknown"]
    
    def select_evolution_pairs(self, systems, num_pairs=3):
        """
        Intelligently select which systems should evolve together.
        Ensures every system gets evolved over time.
        """
        if len(systems) < 2:
            return []
        
        pairs = []
        available_systems = systems.copy()
        
        # Track which systems have evolved recently
        evolution_history = self._load_evolution_history()
        
        # Prioritize systems that haven't evolved in a while
        available_systems.sort(
            key=lambda s: evolution_history.get(s["name"], 0)
        )
        
        # Create pairs ensuring system diversity
        for i in range(min(num_pairs, len(available_systems) // 2)):
            if len(available_systems) >= 2:
                # Select first system (prioritize those needing evolution)
                sys1 = available_systems.pop(0)
                
                # Select second system (prefer different type)
                compatible = [
                    s for s in available_systems 
                    if s["type"] != sys1["type"] or random.random() > 0.5
                ]
                
                if compatible:
                    sys2 = compatible[0]
                    available_systems.remove(sys2)
                else:
                    sys2 = available_systems.pop(0)
                
                pairs.append((sys1, sys2))
        
        return pairs
    
    def _load_evolution_history(self):
        """Load when each system last evolved"""
        hist_file = Path("data/evolution/system_history.json")
        if hist_file.exists():
            with open(hist_file) as f:
                return json.load(f)
        return {}
    
    def evolve_pair(self, system1, system2):
        """
        Evolve two systems together.
        Both systems can learn from each other.
        """
        print(f"\n🧬 Evolving: {system1['name']} ⟲ {system2['name']}")
        
        # Determine evolution direction based on generations
        if system1["generation"] < system2["generation"]:
            # System1 learns from more advanced System2
            donor = system2
            receiver = system1
            direction = f"{system1['name']} learns from {system2['name']}"
        elif system2["generation"] < system1["generation"]:
            # System2 learns from more advanced System1
            donor = system1
            receiver = system2
            direction = f"{system2['name']} learns from {system1['name']}"
        else:
            # Equal generations - bidirectional learning
            donor = system1
            receiver = system2
            direction = f"{system1['name']} ⟲ {system2['name']} (bidirectional)"
        
        print(f"📋 Direction: {direction}")
        
        # Generate improvements
        improvements = self._generate_improvements(donor, receiver)
        
        # Verify improvements
        verified = []
        for imp in improvements:
            if self._verify_improvement(imp, system1, system2):
                verified.append(imp)
        
        # Apply successful improvements
        if verified:
            self._apply_improvements(verified, system1, system2)
            return True
        
        return False
    
    def _generate_improvements(self, donor, receiver):
        """Generate potential improvements from donor to receiver"""
        improvements = []
        
        # Transfer capabilities
        for cap in donor["capabilities"]:
            if cap not in receiver["capabilities"]:
                improvements.append({
                    "type": "capability_transfer",
                    "capability": cap,
                    "from": donor["name"],
                    "to": receiver["name"],
                    "quality": random.uniform(0.5, 0.9)
                })
        
        # Merge approaches (if both have similar capabilities)
        common_caps = set(donor["capabilities"]) & set(receiver["capabilities"])
        for cap in common_caps:
            improvements.append({
                "type": "approach_merging",
                "capability": cap,
                "systems": [donor["name"], receiver["name"]],
                "quality": random.uniform(0.6, 0.95)
            })
        
        # Novel combinations
        if len(donor["capabilities"]) > 0 and len(receiver["capabilities"]) > 0:
            improvements.append({
                "type": "novel_combination",
                "donor_cap": random.choice(donor["capabilities"]),
                "receiver_cap": random.choice(receiver["capabilities"]),
                "quality": random.uniform(0.3, 0.8)
            })
        
        return improvements
    
    def _verify_improvement(self, improvement, system1, system2):
        """Verify if improvement is actually useful"""
        # In real implementation, this would test the improvement
        # For now, simulate with 30% success rate
        return random.random() < 0.3
    
    def _apply_improvements(self, improvements, system1, system2):
        """Apply verified improvements to both systems"""
        new_generation = max(system1["generation"], system2["generation"]) + 1
        
        # Create evolved versions
        for system in [system1, system2]:
            evolved_name = f"{system['name']}_gen{new_generation}"
            evolved_path = self.evolved_dir / evolved_name
            evolved_path.mkdir(exist_ok=True)
            
            # Save evolution record
            record = {
                "timestamp": datetime.now().isoformat(),
                "parents": [system1["name"], system2["name"]],
                "improvements": improvements,
                "new_generation": new_generation
            }
            
            with open(evolved_path / "evolution_record.json", "w") as f:
                json.dump(record, f, indent=2)
            
            print(f"✅ Created: {evolved_name}")
        
        # Update generation tracking
        self._update_generation(system1["name"], new_generation)
        self._update_generation(system2["name"], new_generation)
    
    def _update_generation(self, system_name, generation):
        """Update generation tracking for a system"""
        gen_file = Path(f"data/evolution/generations/{system_name}.json")
        gen_file.parent.mkdir(exist_ok=True)
        
        with open(gen_file, "w") as f:
            json.dump({
                "system": system_name,
                "generation": generation,
                "last_evolved": datetime.now().isoformat()
            }, f, indent=2)
    
    def run_evolution_cycle(self):
        """Run one complete evolution cycle across all systems"""
        print("\n" + "="*60)
        print("🧬 SYSTEM-WIDE EVOLUTION CYCLE STARTING")
        print("="*60)
        
        # Get all systems
        systems = self.get_all_systems()
        print(f"📊 Found {len(systems)} evolvable systems:")
        for s in systems:
            print(f"  • {s['name']} (gen {s['generation']})")
        
        # Select pairs to evolve
        pairs = self.select_evolution_pairs(systems, num_pairs=3)
        print(f"\n🔄 Selected {len(pairs)} evolution pairs")
        
        # Evolve each pair
        successes = 0
        for i, (sys1, sys2) in enumerate(pairs, 1):
            print(f"\n--- Pair {i} ---")
            if self.evolve_pair(sys1, sys2):
                successes += 1
                print(f"✅ Pair {i} evolved successfully")
            else:
                print(f"❌ Pair {i} no improvements found")
        
        # Report cycle results
        print("\n" + "="*60)
        print(f"📊 CYCLE COMPLETE: {successes}/{len(pairs)} successful evolutions")
        print("="*60)
        
        return successes

if __name__ == "__main__":
    evolution = SystemWideEvolution()
    evolution.run_evolution_cycle()
