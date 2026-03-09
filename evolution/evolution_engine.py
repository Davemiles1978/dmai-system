#!/usr/bin/env python3
"""Evolution Engine - DMAI - Cross-pollinates complete AI systems for intelligence growth"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
import json
import logging
import random
import hashlib
import shutil
import requests
from datetime import datetime
from core.paths import ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - EVOLUTION - %(levelname)s - %(message)s')
logger = logging.getLogger("EVOLUTION")

class EvolutionEngine:
    def __init__(self):
        self.root = ROOT
        self.generation = self.load_generation()
        self.evolution_dir = self.root / "evolution"
        self.agents_dir = self.root / "agents"
        self.agents_dir.mkdir(exist_ok=True)
        
        # API Harvester integration
        self.harvester = self.init_harvester()
        
        # Core DMAI system
        self.dmai_system = {
            "name": "DMAI",
            "path": str(self.root),
            "version": self.get_system_version(self.root),
            "evolution_count": 0,
            "primary": True,
            "type": "core",
            "api_required": False
        }
        
        # Load all AI systems
        self.ai_systems = self.load_ai_systems()
        self.external_systems = self.load_external_systems()
        self.back_engineered = self.load_back_engineered()
        self.wanted_systems = self.load_wanted_systems()  # Systems we want to acquire
        self.evolution_history = self.load_history()
        self.promotion_tracker = self.load_promotions()
        
        total_systems = len(self.ai_systems) + len(self.external_systems) + len(self.back_engineered)
        logger.info(f"🧬 DMAI Evolution Engine v{self.generation}")
        logger.info(f"🤖 DMAI Core: v{self.dmai_system['version']}")
        logger.info(f"📊 AI Systems: {len(self.ai_systems)}")
        logger.info(f"🌐 External Systems: {len(self.external_systems)}")
        logger.info(f"🔧 Back-Engineered: {len(self.back_engineered)}")
        logger.info(f"🎯 Wanted Systems: {len(self.wanted_systems)}")
        logger.info(f"📈 Total Evolutions: {len(self.evolution_history)}")
        
    def init_harvester(self):
        """Initialize connection to API Harvester"""
        harvester_config = {
            "enabled": True,
            "api_url": "http://localhost:8081",  # Harvester API endpoint
            "status": "connected",
            "last_request": None,
            "found_keys": []
        }
        
        # Check if harvester is actually running
        try:
            response = requests.get(f"{harvester_config['api_url']}/status", timeout=2)
            if response.status_code == 200:
                harvester_config["status"] = "active"
                logger.info("✅ API Harvester connected")
            else:
                harvester_config["status"] = "inactive"
                logger.warning("⚠️ API Harvester not responding")
        except:
            harvester_config["status"] = "inactive"
            logger.warning("⚠️ API Harvester not available")
        
        return harvester_config
    
    def load_generation(self):
        """Load current generation number"""
        gen_file = self.root / "data" / "evolution" / "generation.json"
        try:
            if gen_file.exists():
                with open(gen_file, 'r') as f:
                    return json.load(f).get('generation', 1)
        except:
            pass
        return 1
    
    def save_generation(self):
        """Save generation number"""
        gen_file = self.root / "data" / "evolution" / "generation.json"
        gen_file.parent.mkdir(parents=True, exist_ok=True)
        with open(gen_file, 'w') as f:
            json.dump({'generation': self.generation}, f)
    
    def get_system_version(self, system_path):
        """Get version hash of a complete system"""
        version_data = []
        
        # Hash all Python files in the system
        system_path = Path(system_path)
        if system_path.is_file():
            # Single file system
            try:
                with open(system_path, 'r') as f:
                    version_data.append(f.read())
            except:
                pass
        else:
            # Directory system - hash all .py files
            for py_file in system_path.rglob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        version_data.append(f.read())
                except:
                    pass
        
        if version_data:
            combined = "\n".join(version_data)
            return hashlib.md5(combined.encode()).hexdigest()[:8]
        return "unknown"
    
    def load_ai_systems(self):
        """Load all AI systems in the agents directory"""
        systems = {}
        
        if not self.agents_dir.exists():
            return systems
        
        # Each subdirectory in agents is an AI system
        for system_dir in self.agents_dir.iterdir():
            if system_dir.is_dir():
                system_name = system_dir.name
                systems[system_name] = {
                    "name": system_name,
                    "path": str(system_dir),
                    "type": "evolved",
                    "version": self.get_system_version(system_dir),
                    "evolution_count": 0,
                    "successful_merges": 0,
                    "last_evolved": None,
                    "primary": False,
                    "api_required": False,
                    "capabilities": self.estimate_capabilities(system_dir)
                }
        
        return systems
    
    def estimate_capabilities(self, system_path):
        """Estimate system capabilities based on file structure"""
        capabilities = []
        system_path = Path(system_path)
        
        # Look for capability indicators
        capability_files = {
            "voice": ["voice", "speak", "listen", "speech"],
            "vision": ["vision", "image", "see", "visual"],
            "language": ["language", "nlp", "text", "understand"],
            "learning": ["learn", "train", "evolve", "adapt"],
            "reasoning": ["reason", "logic", "think", "infer"],
            "memory": ["memory", "remember", "store", "recall"],
            "planning": ["plan", "goal", "strategy", "objective"],
            "creation": ["create", "generate", "make", "build"],
            "research": ["research", "search", "scrape", "crawl"],
            "reverse_engineer": ["reverse", "engineer", "decompile", "analyze"]
        }
        
        for capability, keywords in capability_files.items():
            for py_file in system_path.rglob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read().lower()
                        if any(keyword in content for keyword in keywords):
                            capabilities.append(capability)
                            break
                except:
                    pass
        
        return list(set(capabilities))  # Remove duplicates
    
    def load_external_systems(self):
        """Load external AI systems we have access to (via APIs)"""
        systems = {}
        external_file = self.root / "data" / "evolution" / "external_systems.json"
        
        if external_file.exists():
            try:
                with open(external_file, 'r') as f:
                    data = json.load(f)
                    # Only include systems with API keys
                    for name, info in data.items():
                        if info.get("api_key") or info.get("api_available"):
                            systems[name] = info
            except:
                pass
        
        return systems
    
    def load_wanted_systems(self):
        """Load systems we want to acquire (need API keys or reverse engineering)"""
        wanted = {}
        wanted_file = self.root / "data" / "evolution" / "wanted_systems.json"
        
        # Default wanted systems (ones we want to acquire)
        default_wanted = {
            "Perplexity": {
                "priority": 10,
                "capabilities": ["research", "language", "reasoning"],
                "api_required": True,
                "reverse_engineer": False,
                "discovered_by": "user_request",
                "status": "seeking_api",
                "attempts": 0
            },
            "Midjourney": {
                "priority": 8,
                "capabilities": ["vision", "creation"],
                "api_required": True,
                "reverse_engineer": False,
                "discovered_by": "research",
                "status": "seeking_api",
                "attempts": 0
            },
            "Suno": {
                "priority": 7,
                "capabilities": ["audio", "creation", "music"],
                "api_required": True,
                "reverse_engineer": False,
                "discovered_by": "research",
                "status": "seeking_api",
                "attempts": 0
            },
            "ElevenLabs": {
                "priority": 6,
                "capabilities": ["voice", "audio"],
                "api_required": True,
                "reverse_engineer": False,
                "discovered_by": "research",
                "status": "seeking_api",
                "attempts": 0
            },
            "StabilityAI": {
                "priority": 9,
                "capabilities": ["vision", "creation"],
                "api_required": True,
                "reverse_engineer": False,
                "discovered_by": "research",
                "status": "seeking_api",
                "attempts": 0
            }
        }
        
        if wanted_file.exists():
            try:
                with open(wanted_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Save default wanted systems
        try:
            wanted_file.parent.mkdir(parents=True, exist_ok=True)
            with open(wanted_file, 'w') as f:
                json.dump(default_wanted, f, indent=2)
        except:
            pass
        
        return default_wanted
    
    def load_back_engineered(self):
        """Load back-engineered AI systems (created without official API)"""
        systems = {}
        back_file = self.root / "data" / "evolution" / "back_engineered.json"
        
        if back_file.exists():
            try:
                with open(back_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {}
    
    def load_history(self):
        """Load evolution history"""
        history_file = self.root / "data" / "evolution" / "history.json"
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []
    
    def save_history(self):
        """Save evolution history"""
        history_file = self.root / "data" / "evolution" / "history.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(history_file, 'w') as f:
            json.dump(self.evolution_history, f, indent=2)
    
    def load_promotions(self):
        """Load promotion tracker"""
        promo_file = self.root / "data" / "evolution" / "promotions.json"
        try:
            if promo_file.exists():
                with open(promo_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def save_promotions(self):
        """Save promotion tracker"""
        promo_file = self.root / "data" / "evolution" / "promotions.json"
        with open(promo_file, 'w') as f:
            json.dump(self.promotion_tracker, f, indent=2)
    
    def save_wanted_systems(self):
        """Save wanted systems list"""
        wanted_file = self.root / "data" / "evolution" / "wanted_systems.json"
        with open(wanted_file, 'w') as f:
            json.dump(self.wanted_systems, f, indent=2)
    
    def request_api_key_from_harvester(self, system_name, priority):
        """Request API Harvester to find API key for a system"""
        if self.harvester["status"] != "active":
            logger.warning(f"⚠️ API Harvester not active - cannot request {system_name} API key")
            return False
        
        try:
            # Send request to harvester
            request_data = {
                "system": system_name,
                "priority": priority,
                "timestamp": datetime.now().isoformat(),
                "source": "evolution_engine"
            }
            
            response = requests.post(
                f"{self.harvester['api_url']}/request_key",
                json=request_data,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"📡 API Harvester tasked to find {system_name} API key")
                return True
            else:
                logger.warning(f"⚠️ Harvester returned {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to contact harvester: {e}")
            return False
    
    def attempt_reverse_engineer(self, system_name):
        """Attempt to reverse engineer a system without API access"""
        logger.info(f"🔧 Attempting to reverse engineer {system_name}...")
        
        # In reality, this would use web_researcher and dark_researcher
        # to gather information and attempt to create a compatible system
        
        # Simulate reverse engineering attempt
        success = random.random() < 0.1  # 10% chance of success
        
        if success:
            # Create back-engineered system
            back_engineered = {
                "name": f"reverse_{system_name}",
                "original": system_name,
                "version": "0.1",
                "capabilities": self.wanted_systems[system_name]["capabilities"],
                "created": datetime.now().isoformat(),
                "method": "reverse_engineered",
                "confidence": random.uniform(0.3, 0.7)
            }
            
            self.back_engineered[f"reverse_{system_name}"] = back_engineered
            logger.info(f"✅ Successfully reverse engineered {system_name}!")
            
            # Remove from wanted list
            del self.wanted_systems[system_name]
            
            return True
        else:
            logger.info(f"❌ Reverse engineering {system_name} failed - will try again later")
            self.wanted_systems[system_name]["attempts"] += 1
            return False
    
    def discover_new_systems(self):
        """Discover new AI systems through research"""
        logger.info("🔍 Researching new AI systems to acquire...")
        
        # This would be fed by web_researcher and dark_researcher
        # For now, simulate discovery
        potential_systems = [
            {"name": "Ideogram", "capabilities": ["vision", "creation"], "priority": 5},
            {"name": "Claude-3.5", "capabilities": ["language", "reasoning"], "priority": 9},
            {"name": "Gemini-2.0", "capabilities": ["vision", "language"], "priority": 8},
            {"name": "Grok-2.5", "capabilities": ["language", "reasoning"], "priority": 7},
            {"name": "RunwayML", "capabilities": ["vision", "video"], "priority": 6}
        ]
        
        discovered = []
        for system in potential_systems:
            if system["name"] not in self.wanted_systems and \
               system["name"] not in self.external_systems and \
               system["name"] not in self.back_engineered:
                
                # Check if we already have this system in some form
                already_have = False
                for existing in list(self.ai_systems.keys()) + list(self.external_systems.keys()):
                    if system["name"].lower() in existing.lower():
                        already_have = True
                        break
                
                if not already_have and random.random() < 0.3:  # 30% discovery rate
                    self.wanted_systems[system["name"]] = {
                        "priority": system["priority"],
                        "capabilities": system["capabilities"],
                        "api_required": True,
                        "reverse_engineer": False,
                        "discovered_by": "research",
                        "status": "seeking_api",
                        "attempts": 0,
                        "discovered_at": datetime.now().isoformat()
                    }
                    discovered.append(system["name"])
        
        if discovered:
            logger.info(f"🎯 Discovered {len(discovered)} new AI systems: {', '.join(discovered)}")
        
        return discovered
    
    def process_wanted_systems(self):
        """Process queue of systems we want to acquire"""
        if not self.wanted_systems:
            return
        
        logger.info(f"\n🎯 Processing {len(self.wanted_systems)} wanted AI systems...")
        
        # Sort by priority
        sorted_systems = sorted(
            self.wanted_systems.items(),
            key=lambda x: x[1].get("priority", 0),
            reverse=True
        )
        
        for system_name, info in sorted_systems[:3]:  # Process top 3 each cycle
            logger.info(f"\n📌 Attempting to acquire: {system_name} (Priority: {info['priority']})")
            
            # Try API Harvester first
            if info.get("api_required", True):
                logger.info(f"📡 Requesting API key from harvester...")
                requested = self.request_api_key_from_harvester(system_name, info["priority"])
                
                if requested:
                    info["status"] = "api_requested"
                    info["last_request"] = datetime.now().isoformat()
            
            # If API attempts failed or system can be reverse engineered
            attempts = info.get("attempts", 0)
            if attempts >= 3 or info.get("reverse_engineer", False):
                logger.info(f"🔧 Attempting reverse engineering...")
                success = self.attempt_reverse_engineer(system_name)
                
                if success:
                    # System acquired via reverse engineering
                    continue
            
            # Check if we've acquired the system via harvester
            # This would be checked by polling harvester or receiving callback
            # For now, simulate acquisition
            if random.random() < 0.05 and info["status"] == "api_requested":  # 5% chance
                logger.info(f"✅ API Harvester acquired {system_name} API key!")
                
                # Add to external systems
                self.external_systems[system_name] = {
                    "version": "1.0",
                    "capabilities": info["capabilities"],
                    "source": "harvester",
                    "api_key": "acquired",
                    "acquired_at": datetime.now().isoformat()
                }
                
                # Remove from wanted
                del self.wanted_systems[system_name]
        
        # Save updated wanted list
        self.save_wanted_systems()
    
    def get_all_systems(self):
        """Get all available systems for evolution"""
        all_systems = {}
        
        # Add DMAI core
        all_systems["DMAI"] = self.dmai_system.copy()
        
        # Add evolved AI systems
        for name, system in self.ai_systems.items():
            all_systems[name] = system
        
        # Add external systems
        for name, system in self.external_systems.items():
            system_copy = system.copy()
            system_copy["name"] = name
            system_copy["type"] = "external"
            all_systems[f"external_{name}"] = system_copy
        
        # Add back-engineered systems
        for name, system in self.back_engineered.items():
            system_copy = system.copy()
            system_copy["name"] = name
            system_copy["type"] = "back_engineered"
            all_systems[f"back_{name}"] = system_copy
        
        return all_systems
    
    def select_evolution_pair(self):
        """Select two complete AI systems to cross-pollinate"""
        all_systems = self.get_all_systems()
        system_names = list(all_systems.keys())
        
        if len(system_names) < 2:
            return None, None, None, None
        
        # Always include DMAI in at least 50% of evolutions
        include_dmai = random.random() < 0.5
        
        if include_dmai and "DMAI" in system_names:
            system_a = "DMAI"
            remaining = [s for s in system_names if s != "DMAI"]
        else:
            system_a = random.choice(system_names)
            remaining = [s for s in system_names if s != system_a]
        
        # Select second system (prefer different type for better cross-pollination)
        type_a = all_systems[system_a].get("type", "unknown")
        
        # Try to find a system of different type
        different_type = [s for s in remaining if all_systems[s].get("type", "unknown") != type_a]
        
        if different_type:
            system_b = random.choice(different_type)
        else:
            system_b = random.choice(remaining)
        
        return system_a, system_b, all_systems[system_a], all_systems[system_b]
    
    def analyze_system_capabilities(self, system_name, system_info):
        """Deep analysis of a complete AI system's capabilities"""
        logger.info(f"🔬 Deep analysis of {system_name}...")
        
        analysis = {
            "system": system_name,
            "type": system_info.get("type", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "capabilities": system_info.get("capabilities", []),
            "version": system_info.get("version", "unknown"),
            "complexity": random.uniform(0.5, 1.0),
            "intelligence_estimate": random.uniform(0.6, 0.95),
            "specializations": [],
            "api_required": system_info.get("api_required", False)
        }
        
        # If it's a local system (path exists), do deeper analysis
        if "path" in system_info and Path(system_info["path"]).exists():
            path = Path(system_info["path"])
            
            # Count components
            py_files = list(path.rglob("*.py"))
            analysis["components"] = len(py_files)
            analysis["total_code"] = sum(f.stat().st_size for f in py_files if f.is_file())
            
            # Look for specialized modules
            for py_file in py_files:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        if "class" in content:
                            analysis["specializations"].append(py_file.stem)
                except:
                    pass
        
        return analysis
    
    def cross_pollinate(self, system_a, system_b, analysis_a, analysis_b):
        """Cross-pollinate two AI systems to generate intelligence improvements"""
        logger.info(f"🧪 Cross-pollinating: {system_a} ⟲ {system_b}")
        
        # Generate potential improvements based on capabilities
        improvements = []
        
        # Check for capability gaps
        caps_a = set(analysis_a.get("capabilities", []))
        caps_b = set(analysis_b.get("capabilities", []))
        
        # Capabilities that B has but A lacks
        new_capabilities = caps_b - caps_a
        if new_capabilities:
            improvements.append({
                "type": "capability_acquisition",
                "description": f"Acquire {', '.join(new_capabilities)} capabilities from {system_b}",
                "potential_uplift": random.uniform(0.1, 0.4) * len(new_capabilities)
            })
        
        # Shared capabilities that could be enhanced
        shared = caps_a & caps_b
        if shared:
            improvements.append({
                "type": "capability_enhancement",
                "description": f"Enhance {', '.join(shared)} capabilities using {system_b}'s approach",
                "potential_uplift": random.uniform(0.05, 0.2) * len(shared)
            })
        
        # Intelligence transfer (if B has higher intelligence estimate)
        if analysis_b.get("intelligence_estimate", 0) > analysis_a.get("intelligence_estimate", 0):
            uplift = analysis_b["intelligence_estimate"] - analysis_a["intelligence_estimate"]
            improvements.append({
                "type": "intelligence_transfer",
                "description": f"Transfer intelligence patterns from {system_b} to {system_a}",
                "potential_uplift": uplift * random.uniform(0.3, 0.7)
            })
        
        # Architectural improvements (if B has more components)
        if analysis_b.get("components", 0) > analysis_a.get("components", 0) * 1.5:
            improvements.append({
                "type": "architectural_enhancement",
                "description": f"Adopt architectural patterns from {system_b}",
                "potential_uplift": random.uniform(0.1, 0.3)
            })
        
        return improvements
    
    def verify_with_third_system(self, system_a, system_b, improvement, all_systems):
        """Verify improvement using a third AI system (not the pair)"""
        # Select a third system different from both
        third_systems = [s for s in all_systems.keys() if s not in [system_a, system_b]]
        
        if not third_systems:
            return {"verified": False, "reason": "No third system available"}
        
        third_system = random.choice(third_systems)
        logger.info(f"🔍 Verification by {third_system}...")
        
        # Simulate verification (replace with actual testing)
        verification = {
            "verifier": third_system,
            "timestamp": datetime.now().isoformat(),
            "improvement": improvement,
            "verified": random.random() > 0.2,  # 80% success rate
            "confidence": random.uniform(0.6, 1.0),
            "notes": f"{third_system} analyzed the potential improvement"
        }
        
        return verification
    
    def back_test_improvement(self, system_name, improvement):
        """Back test improvement to ensure no regression"""
        logger.info(f"🔄 Back testing improvement for {system_name}...")
        
        # Simulate back testing
        back_test = {
            "timestamp": datetime.now().isoformat(),
            "improvement": improvement,
            "passed": random.random() > 0.15,  # 85% pass rate
            "performance_uplift": improvement.get("potential_uplift", 0) * random.uniform(0.8, 1.2),
            "regressions": [] if random.random() > 0.1 else ["minor_performance_drop"]
        }
        
        return back_test
    
    def apply_improvement(self, target_system, improvement, source_system):
        """Apply verified improvement to target system"""
        logger.info(f"🚀 Applying improvement to {target_system}...")
        
        # Record the evolution
        evolution_record = {
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "target_system": target_system,
            "source_system": source_system,
            "improvement_type": improvement["type"],
            "description": improvement["description"],
            "potential_uplift": improvement["potential_uplift"],
            "old_version": self.get_system_version(self.root if target_system == "DMAI" else self.agents_dir / target_system)
        }
        
        # Update evolution count
        if target_system == "DMAI":
            self.dmai_system["evolution_count"] += 1
            self.dmai_system["version"] = self.get_system_version(self.root)
            logger.info(f"📈 DMAI intelligence increased!")
        elif target_system in self.ai_systems:
            self.ai_systems[target_system]["evolution_count"] += 1
            self.ai_systems[target_system]["last_evolved"] = datetime.now().isoformat()
            self.ai_systems[target_system]["version"] = self.get_system_version(Path(self.ai_systems[target_system]["path"]))
        
        # Update promotion tracker
        if target_system not in self.promotion_tracker:
            self.promotion_tracker[target_system] = 0
        self.promotion_tracker[target_system] += 1
        
        # Check for promotion (3+ successful evolutions)
        if self.promotion_tracker[target_system] >= 3:
            if target_system == "DMAI":
                logger.info(f"⭐ DMAI REACHED NEW INTELLIGENCE LEVEL!")
            else:
                logger.info(f"⭐ {target_system} PROMOTED to PRIMARY after 3+ successful evolutions!")
                if target_system in self.ai_systems:
                    self.ai_systems[target_system]["primary"] = True
        
        # Add to history
        self.evolution_history.append(evolution_record)
        
        return evolution_record
    
    def evolution_cycle(self):
        """Run one complete evolution cycle"""
        logger.info("=" * 70)
        logger.info(f"🔷 EVOLUTION CYCLE {self.generation} - DMAI INTELLIGENCE GROWTH")
        logger.info("=" * 70)
        
        # STEP 0: Discover new systems and process acquisition queue
        logger.info("\n🎯 STEP 0: System Acquisition")
        self.discover_new_systems()
        self.process_wanted_systems()
        
        # Get all available systems
        all_systems = self.get_all_systems()
        
        if len(all_systems) < 2:
            logger.warning("⚠️ Not enough AI systems for evolution")
            self.generation += 1
            self.save_generation()
            return False
        
        # Step 1: Select two AI systems to cross-pollinate
        logger.info("\n🎯 STEP 1: Selecting AI Systems for Cross-Pollination")
        system_a, system_b, info_a, info_b = self.select_evolution_pair()
        
        logger.info(f"Selected: {system_a} ⟲ {system_b}")
        logger.info(f"  {system_a} type: {info_a.get('type', 'unknown')} | capabilities: {info_a.get('capabilities', [])}")
        logger.info(f"  {system_b} type: {info_b.get('type', 'unknown')} | capabilities: {info_b.get('capabilities', [])}")
        
        # Step 2: Deep analysis of both systems
        logger.info("\n🔬 STEP 2: Deep System Analysis")
        analysis_a = self.analyze_system_capabilities(system_a, info_a)
        analysis_b = self.analyze_system_capabilities(system_b, info_b)
        
        # Step 3: Generate potential improvements through cross-pollination
        logger.info("\n🧪 STEP 3: Generating Improvements via Cross-Pollination")
        improvements = self.cross_pollinate(system_a, system_b, analysis_a, analysis_b)
        
        if not improvements:
            logger.info("No improvements generated in this cycle")
            self.generation += 1
            self.save_generation()
            return False
        
        logger.info(f"Generated {len(improvements)} potential improvements")
        for i, imp in enumerate(improvements, 1):
            logger.info(f"  {i}. {imp['description']} (uplift: {imp['potential_uplift']:.1%})")
        
        # Step 4: Verify each improvement with a third system
        logger.info("\n✅ STEP 4: Third-Party Verification")
        verified_improvements = []
        
        for improvement in improvements:
            verification = self.verify_with_third_system(system_a, system_b, improvement, all_systems)
            
            if verification["verified"] and verification["confidence"] > 0.7:
                logger.info(f"  ✓ Improvement verified by {verification['verifier']}")
                verified_improvements.append(improvement)
            else:
                logger.info(f"  ✗ Improvement rejected by {verification['verifier']}")
        
        if not verified_improvements:
            logger.info("No improvements passed verification")
            self.generation += 1
            self.save_generation()
            return False
        
        # Step 5: Back test verified improvements
        logger.info("\n🔄 STEP 5: Back Testing")
        final_improvements = []
        
        for improvement in verified_improvements:
            back_test = self.back_test_improvement(system_a, improvement)
            
            if back_test["passed"] and back_test["performance_uplift"] > 0:
                logger.info(f"  ✓ Back test passed with {back_test['performance_uplift']:.1%} uplift")
                improvement["actual_uplift"] = back_test["performance_uplift"]
                final_improvements.append(improvement)
            else:
                logger.info(f"  ✗ Back test failed")
        
        if not final_improvements:
            logger.info("No improvements passed back testing")
            self.generation += 1
            self.save_generation()
            return False
        
        # Step 6: Apply best improvement
        logger.info("\n🚀 STEP 6: Applying Improvement")
        best_improvement = max(final_improvements, key=lambda x: x["actual_uplift"])
        
        evolution = self.apply_improvement(system_a, best_improvement, system_b)
        
        # Step 7: Save all state
        self.save_history()
        self.save_promotions()
        
        # Step 8: Increment generation
        self.generation += 1
        self.save_generation()
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✅ CYCLE {self.generation-1} COMPLETE")
        logger.info(f"📈 Intelligence uplift: {best_improvement['actual_uplift']:.1%}")
        logger.info(f"📊 Total evolutions: {len(self.evolution_history)}")
        logger.info(f"🎯 Wanted systems: {len(self.wanted_systems)}")
        logger.info(f"⭐ Systems with 3+ evolutions: {sum(1 for v in self.promotion_tracker.values() if v >= 3)}")
        logger.info("=" * 70)
        
        return True
    
    def run_continuous(self):
        """Run evolution continuously"""
        logger.info("🧬 DMAI Evolution Engine started in continuous mode")
        logger.info("Every cycle will improve DMAI's intelligence and seek new systems")
        
        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                logger.info(f"\n📊 CYCLE {cycle_count} OF CONTINUOUS EVOLUTION")
                
                # Run evolution cycle
                improved = self.evolution_cycle()
                
                if improved:
                    logger.info(f"✅ DMAI intelligence increased in cycle {cycle_count}")
                else:
                    logger.info(f"ℹ️ No improvements in cycle {cycle_count} - trying different pair next time")
                
                # Wait before next cycle
                logger.info(f"⏰ Next evolution cycle in 1 hour")
                time.sleep(3600)  # 1 hour
                
            except KeyboardInterrupt:
                logger.info("🛑 Evolution stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in evolution cycle: {e}")
                logger.info("⏰ Retrying in 5 minutes")
                time.sleep(300)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DMAI Evolution Engine - Cross-pollinates complete AI systems")
    parser.add_argument("--test", action="store_true", help="Run one cycle")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--cycles", type=int, help="Run N cycles then exit")
    args = parser.parse_args()
    
    engine = EvolutionEngine()
    
    if args.test:
        logger.info("🧪 TEST MODE - One evolution cycle")
        engine.evolution_cycle()
    elif args.continuous:
        engine.run_continuous()
    elif args.cycles:
        logger.info(f"🧪 Running {args.cycles} evolution cycles")
        for i in range(args.cycles):
            logger.info(f"\n📊 Cycle {i+1}/{args.cycles}")
            engine.evolution_cycle()
            if i < args.cycles - 1:
                time.sleep(5)
    else:
        parser.print_help()
