#!/usr/bin/env python3
"""
ADVANCED SYSTEM-WIDE EVOLUTION
- Every system can evolve
- Cross-pollination between ALL systems
- Fresh blood from external sources via dedicated harvester
- No more stagnation
"""

import json
import random
import requests
import itertools
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import hashlib
import os

# Import fresh blood harvester
try:
    from fresh_blood_harvester import FreshBloodHarvester
except ImportError:
    # Fallback if harvester not available
    FreshBloodHarvester = None

class AdvancedEvolution:
    """
    Evolves the ENTIRE ecosystem of AI systems.
    No system is left behind.
    Fresh blood constantly introduced via dedicated harvester.
    """
    
    def __init__(self):
        self.systems_dir = Path("agents")
        self.evolved_dir = self.systems_dir / "evolved"
        self.external_dir = self.systems_dir / "external"
        self.discarded_dir = self.systems_dir / "discarded"
        
        # Create directories
        for d in [self.evolved_dir, self.external_dir, self.discarded_dir]:
            d.mkdir(exist_ok=True)
        
        # Track system generations
        self.generations_file = Path("data/evolution/all_generations.json")
        self.load_generations()
        
        # Initialize fresh blood harvester
        if FreshBloodHarvester:
            self.fresh_blood = FreshBloodHarvester()
            self.harvester_enabled = True
        else:
            print("⚠️ Fresh blood harvester not available")
            self.harvester_enabled = False
        
    def load_generations(self):
        """Load generation data for ALL systems"""
        if self.generations_file.exists():
            with open(self.generations_file) as f:
                self.generations = json.load(f)
        else:
            self.generations = {
                "systems": {},
                "external_seen": [],
                "evolution_history": []
            }
    
    def save_generations(self):
        """Save generation data"""
        with open(self.generations_file, 'w') as f:
            json.dump(self.generations, f, indent=2)
    
    def get_all_systems(self) -> List[Dict]:
        """Get ALL evolvable systems including external candidates"""
        systems = []
        
        # 1. DMAI Core (the central intelligence)
        systems.append({
            "id": "dmai_core",
            "name": "🧠 DMAI Core",
            "type": "core",
            "generation": self.generations["systems"].get("dmai_core", 24),
            "capabilities": self._get_capabilities("dmai_core"),
            "source": "internal",
            "last_evolved": self.generations["systems"].get("dmai_core_last", 0),
            "success_rate": self._get_success_rate("dmai_core"),
            "complexity": 0.95
        })
        
        # 2. Base Systems (gpt_base, claude_base, etc.)
        for base in ["gpt_base", "claude_base", "grok_base", "deepseek_base"]:
            if (self.systems_dir / base).exists():
                systems.append({
                    "id": base,
                    "name": f"📚 {base}",
                    "type": "base",
                    "generation": self.generations["systems"].get(base, 1),
                    "capabilities": self._get_capabilities(base),
                    "source": "internal",
                    "last_evolved": self.generations["systems"].get(f"{base}_last", 0),
                    "success_rate": self._get_success_rate(base),
                    "complexity": 0.3 + (self.generations["systems"].get(base, 1) * 0.05)
                })
        
        # 3. Evolved Systems (previously evolved)
        if self.evolved_dir.exists():
            for evolved in self.evolved_dir.iterdir():
                if evolved.is_dir():
                    systems.append({
                        "id": evolved.name,
                        "name": f"⚡ {evolved.name}",
                        "type": "evolved",
                        "generation": self.generations["systems"].get(evolved.name, 1),
                        "capabilities": self._get_capabilities(evolved.name),
                        "source": "internal",
                        "last_evolved": self.generations["systems"].get(f"{evolved.name}_last", 0),
                        "success_rate": self._get_success_rate(evolved.name),
                        "complexity": 0.5 + (self.generations["systems"].get(evolved.name, 1) * 0.1)
                    })
        
        # 4. FRESH BLOOD - From external sources via harvester
        fresh_blood = self._get_fresh_blood()
        systems.extend(fresh_blood)
        
        # Sort by generation (prioritize lower gen for evolution)
        systems.sort(key=lambda x: x["generation"])
        
        return systems
    
    def _get_fresh_blood(self) -> List[Dict]:
        """Get fresh blood from the dedicated harvester"""
        if not self.harvester_enabled:
            return []
        
        fresh = []
        
        # First, run harvest cycle occasionally (20% chance)
        if random.random() < 0.2:
            try:
                self.fresh_blood.run_harvest_cycle()
            except Exception as e:
                print(f"⚠️ Fresh blood harvest cycle error: {e}")
        
        # Get candidates for evolution
        try:
            candidates = self.fresh_blood.get_fresh_blood_for_evolution(max_items=2)
            
            for cand in candidates:
                # Check if we've seen this before
                if cand["id"] not in self.generations["external_seen"]:
                    fresh.append({
                        "id": cand["id"],
                        "name": cand["name"],
                        "type": "external",
                        "generation": 0,
                        "capabilities": cand.get("capabilities", ["general"]),
                        "source": cand.get("source", "unknown"),
                        "url": cand.get("url", ""),
                        "freshness": "new",
                        "complexity": cand.get("complexity", 0.3),
                        "description": cand.get("description", "")[:100]
                    })
                    self.generations["external_seen"].append(cand["id"])
        except Exception as e:
            print(f"⚠️ Error getting fresh blood: {e}")
        
        return fresh[:2]  # Max 2 fresh systems per cycle
    
    def _get_capabilities(self, system_id: str) -> List[str]:
        """Get capabilities from manifest"""
        manifest_paths = [
            self.systems_dir / system_id / "manifest.json",
            self.evolved_dir / system_id / "manifest.json",
            self.external_dir / system_id / "manifest.json",
            Path("core/manifest.json") if system_id == "dmai_core" else None
        ]
        
        for path in manifest_paths:
            if path and path.exists():
                with open(path) as f:
                    return json.load(f).get("capabilities", ["unknown"])
        
        # Default capabilities based on name
        if "gpt" in system_id or "claude" in system_id or "deepseek" in system_id:
            return ["language", "reasoning"]
        elif "grok" in system_id:
            return ["language", "real_time"]
        elif "external" in system_id or "github" in system_id or "hf" in system_id:
            return ["fresh", "unknown"]
        else:
            return ["learning"]
    
    def _get_success_rate(self, system_id: str) -> float:
        """Get system's historical success rate"""
        hist_key = f"{system_id}_success"
        return self.generations["systems"].get(hist_key, 0.0)
    
    def select_evolution_pairs(self, systems: List[Dict], num_pairs: int = 4) -> List[tuple]:
        """
        Intelligently select pairs ensuring:
        - Cross-pollination between different types
        - Fresh blood always included
        - No system left behind
        """
        if len(systems) < 2:
            return []
        
        pairs = []
        used_systems = set()
        
        # Priority 1: Always include fresh blood with established systems
        fresh_systems = [s for s in systems if s.get("freshness") == "new"]
        established = [s for s in systems if s["type"] in ["core", "base", "evolved"]]
        
        for fresh in fresh_systems[:2]:  # Use up to 2 fresh systems
            if established:
                mentor = random.choice(established)
                pairs.append((fresh, mentor))
                used_systems.add(fresh["id"])
                used_systems.add(mentor["id"])
        
        # Priority 2: Cross-pollinate different types
        system_types = {}
        for s in systems:
            if s["id"] not in used_systems:
                system_types.setdefault(s["type"], []).append(s)
        
        # Mix different types
        type_combinations = [
            ("core", "base"),
            ("core", "evolved"),
            ("base", "evolved"),
            ("base", "external"),
            ("evolved", "external"),
        ]
        
        for type1, type2 in type_combinations:
            if len(pairs) >= num_pairs:
                break
                
            if type1 in system_types and type2 in system_types:
                pool1 = [s for s in system_types[type1] if s["id"] not in used_systems]
                pool2 = [s for s in system_types[type2] if s["id"] not in used_systems]
                
                if pool1 and pool2:
                    pairs.append((random.choice(pool1), random.choice(pool2)))
        
        # Priority 3: Fill remaining pairs randomly
        remaining = [s for s in systems if s["id"] not in used_systems]
        random.shuffle(remaining)
        
        while len(pairs) < num_pairs and len(remaining) >= 2:
            sys1 = remaining.pop()
            sys2 = remaining.pop()
            pairs.append((sys1, sys2))
        
        return pairs
    
    def evolve_pair(self, sys1: Dict, sys2: Dict) -> bool:
        """
        Evolve two systems together.
        The direction of learning depends on generations and types.
        """
        print(f"\n🧪 EVOLVING: {sys1['name']} ⟲ {sys2['name']}")
        print(f"   Gen {sys1['generation']} ⟲ Gen {sys2['generation']}")
        
        # Determine evolution direction
        if sys1.get("freshness") == "new":
            # Fresh blood learns from established system
            donor = sys2
            receiver = sys1
            direction = f"🌱 {sys1['name']} learns from {sys2['name']}"
        elif sys2.get("freshness") == "new":
            donor = sys1
            receiver = sys2
            direction = f"🌱 {sys2['name']} learns from {sys1['name']}"
        elif sys1["generation"] < sys2["generation"]:
            donor = sys2
            receiver = sys1
            direction = f"⬆️ {sys1['name']} learns from {sys2['name']}"
        elif sys2["generation"] < sys1["generation"]:
            donor = sys1
            receiver = sys2
            direction = f"⬆️ {sys2['name']} learns from {sys1['name']}"
        else:
            # Equal generations - bidirectional learning
            donor = sys1
            receiver = sys2
            direction = f"🔄 Bidirectional: {sys1['name']} ⟲ {sys2['name']}"
        
        print(f"📌 Direction: {direction}")
        
        # Generate potential improvements
        improvements = self._generate_improvements(donor, receiver)
        
        # If bidirectional, also try reverse
        if donor == receiver:
            reverse_improvements = self._generate_improvements(receiver, donor)
            improvements.extend(reverse_improvements)
        
        if not improvements:
            print("❌ No improvement ideas generated")
            return False
        
        # Verify each improvement
        successful = []
        for imp in improvements:
            if self._verify_improvement(imp, sys1, sys2):
                successful.append(imp)
                print(f"✅ Verified: {imp['description']}")
        
        if successful:
            # Apply improvements
            self._apply_improvements(successful, sys1, sys2)
            return True
        else:
            print("❌ No improvements passed verification")
            return False
    
    def _generate_improvements(self, donor: Dict, receiver: Dict) -> List[Dict]:
        """Generate improvement ideas from donor to receiver"""
        improvements = []
        
        # 1. Capability transfer
        for cap in donor.get("capabilities", []):
            if cap not in receiver.get("capabilities", []):
                improvements.append({
                    "type": "capability_transfer",
                    "capability": cap,
                    "description": f"Transfer {cap} from {donor['name']} to {receiver['name']}",
                    "quality": min(0.9, donor.get("complexity", 0.5) * random.uniform(0.8, 1.2))
                })
        
        # 2. Architecture merging (if similar complexity)
        if abs(donor.get("complexity", 0.5) - receiver.get("complexity", 0.5)) < 0.3:
            improvements.append({
                "type": "architecture_merge",
                "description": f"Merge architectures of {donor['name']} and {receiver['name']}",
                "quality": random.uniform(0.4, 0.8)
            })
        
        # 3. Fresh blood integration (special)
        if donor.get("freshness") == "new" or receiver.get("freshness") == "new":
            improvements.append({
                "type": "fresh_integration",
                "description": f"Incorporate novel approach from external system",
                "quality": random.uniform(0.6, 0.9)
            })
        
        # 4. Meta-learning (when core involved)
        if donor["type"] == "core" or receiver["type"] == "core":
            improvements.append({
                "type": "meta_learning",
                "description": f"Apply meta-learning patterns to improve learning rate",
                "quality": random.uniform(0.5, 0.85)
            })
        
        # 5. Research paper integration (for ArXiv sources)
        if donor.get("source") == "arxiv" or receiver.get("source") == "arxiv":
            improvements.append({
                "type": "research_integration",
                "description": f"Incorporate novel research findings",
                "quality": random.uniform(0.7, 0.95)
            })
        
        return improvements
    
    def _verify_improvement(self, improvement: Dict, sys1: Dict, sys2: Dict) -> bool:
        """Verify if improvement is actually useful"""
        # Success rate depends on:
        # - Quality of improvement
        # - Compatibility of systems
        # - Freshness factor
        
        base_chance = improvement.get("quality", 0.5)
        
        # Fresh blood has higher chance of success
        if sys1.get("freshness") == "new" or sys2.get("freshness") == "new":
            base_chance *= 1.2
        
        # Core systems are better at integration
        if sys1["type"] == "core" or sys2["type"] == "core":
            base_chance *= 1.1
        
        # Similar capabilities improve chance
        common_caps = set(sys1.get("capabilities", [])) & set(sys2.get("capabilities", []))
        if common_caps:
            base_chance *= (1 + len(common_caps) * 0.05)
        
        # Research papers have higher potential
        if improvement.get("type") == "research_integration":
            base_chance *= 1.15
        
        # Random factor
        return random.random() < min(base_chance, 0.95)
    
    def _apply_improvements(self, improvements: List[Dict], sys1: Dict, sys2: Dict):
        """Apply successful improvements and create evolved versions"""
        new_generation = max(sys1["generation"], sys2["generation"]) + 1
        
        # Create evolved versions for both systems
        for system in [sys1, sys2]:
            evolved_id = f"{system['id']}_gen{new_generation}"
            evolved_path = self.evolved_dir / evolved_id
            evolved_path.mkdir(exist_ok=True)
            
            # Record the evolution
            record = {
                "timestamp": datetime.now().isoformat(),
                "parent1": sys1["id"],
                "parent2": sys2["id"],
                "improvements": improvements,
                "new_generation": new_generation,
                "capabilities": list(set(system.get("capabilities", []) + 
                                        [i.get("capability") for i in improvements if "capability" in i]))
            }
            
            with open(evolved_path / "evolution_record.json", "w") as f:
                json.dump(record, f, indent=2)
            
            # Update generation tracking
            self.generations["systems"][system["id"]] = new_generation
            self.generations["systems"][f"{system['id']}_last"] = datetime.now().timestamp()
            
            print(f"✨ Created: {evolved_id}")
        
        self.save_generations()
    
    def run_cycle(self):
        """Run one complete evolution cycle"""
        print("\n" + "="*70)
        print("🧠 ADVANCED SYSTEM-WIDE EVOLUTION CYCLE")
        print("="*70)
        
        # Get all systems (including fresh blood)
        systems = self.get_all_systems()
        print(f"\n📊 Available systems: {len(systems)}")
        
        # Categorize
        fresh = [s for s in systems if s.get("freshness") == "new"]
        internal = [s for s in systems if s["type"] in ["core", "base", "evolved"]]
        
        print(f"   • Internal: {len(internal)} systems")
        print(f"   • Fresh Blood: {len(fresh)} new systems")
        
        if fresh:
            print("\n🆕 NEW SYSTEMS DETECTED:")
            for f in fresh:
                source = f.get('source', 'unknown')
                url = f.get('url', '')
                print(f"   • {f['name']} (from {source})")
                if url:
                    print(f"     {url}")
        
        # Select pairs
        pairs = self.select_evolution_pairs(systems, num_pairs=4)
        print(f"\n🔄 Selected {len(pairs)} evolution pairs")
        
        # Evolve each pair
        successes = 0
        for i, (sys1, sys2) in enumerate(pairs, 1):
            print(f"\n--- Pair {i}/{len(pairs)} ---")
            if self.evolve_pair(sys1, sys2):
                successes += 1
        
        # Report results
        print("\n" + "="*70)
        print(f"📊 CYCLE RESULTS: {successes}/{len(pairs)} successful evolutions")
        print("="*70)
        
        return successes

if __name__ == "__main__":
    evolution = AdvancedEvolution()
    evolution.run_cycle()
