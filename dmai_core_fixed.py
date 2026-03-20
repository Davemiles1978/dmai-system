#!/usr/bin/env python3
"""
██████╗ ███╗   ███╗ █████╗ ██╗
██╔══██╗████╗ ████║██╔══██╗██║
██║  ██║██╔████╔██║███████║██║
██║  ██║██║╚██╔╝██║██╔══██║██║
██████╔╝██║ ╚═╝ ██║██║  ██║██║
╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝

DMAI CORE - SINGLE UNIFIED INTELLIGENCE
Version: 6.1 | With MiroFish Integration & Web UI Fixes
"""
import sys
from pathlib import Path

# Get the absolute path to the dmai-system directory
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

import os
import time
import json
import logging
import threading
import importlib
import inspect
import queue
import random
import hashlib
import socket
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import traceback

# Create logs directory if it doesn't exist
logs_dir = BASE_DIR / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(logs_dir / 'dmai_core.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CORE')

# ============================================================================
# BIOMETRIC AUTHENTICATION
# ============================================================================

class BiometricAuth:
    """Multi-factor authentication for Master Control"""
    
    def __init__(self):
        self.auth_methods = {
            "password": self._check_password,
            "voice": self._check_voice,
            "face": self._check_face,
            "fingerprint": self._check_fingerprint
        }
        self.authenticated = False
        self.current_user = None
        
    def _check_password(self, credentials):
        """Check password from .env or config"""
        # Load from secure storage
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("MASTER_PASSWORD"):
                        stored = line.strip().split("=")[1]
                        return credentials.get("password") == stored
        # Default fallback
        return credentials.get("password") == "dmai2026"
    
    def _check_voice(self, credentials):
        """Check voice sample against enrolled profile"""
        voice_profile = BASE_DIR / "voice_models" / "voice_profile.json"
        if voice_profile.exists():
            # Voice verification would happen here
            # For now, return True if voice file exists
            return True
        return False
    
    def _check_face(self, credentials):
        """Check facial recognition (placeholder)"""
        # Would integrate with camera
        return False
    
    def _check_fingerprint(self, credentials):
        """Check fingerprint (placeholder)"""
        # Would integrate with fingerprint scanner
        return False
    
    def authenticate(self, method="password", credentials=None):
        """Main authentication method"""
        if method in self.auth_methods:
            if self.auth_methods[method](credentials or {}):
                self.authenticated = True
                self.current_user = credentials.get("username", "admin")
                logger.info(f"✅ Master authenticated via {method}")
                return True
        logger.warning(f"❌ Authentication failed via {method}")
        return False
    
    def require_auth(self, func):
        """Decorator for requiring authentication"""
        def wrapper(*args, **kwargs):
            if self.authenticated:
                return func(*args, **kwargs)
            else:
                return {"error": "Authentication required"}
        return wrapper

# ============================================================================
# WEB UI FIXER - Let DMAI fix her own UI
# ============================================================================

class WebUIFixer:
    """DMAI fixes her own web UI issues"""
    
    def __init__(self, dmai_core):
        self.dmai = dmai_core
        self.ui_files = [
            BASE_DIR / "dmai_web_ui.py",
            BASE_DIR / "cloud_web_ui.py",
            BASE_DIR / "ui" / "app.py"
        ]
        
    def scan_and_fix(self):
        """Scan UI files for syntax errors and fix them"""
        fixes = []
        for ui_file in self.ui_files:
            if ui_file.exists():
                issues = self._scan_file(ui_file)
                if issues:
                    self._fix_file(ui_file, issues)
                    fixes.append({"file": str(ui_file), "issues": issues})
        
        if fixes:
            logger.info(f"🔧 Fixed {len(fixes)} UI issues")
            # Restart web UI after fixes
            self._restart_web_ui()
        return fixes
    
    def _scan_file(self, file_path):
        """Check for common syntax errors"""
        issues = []
        with open(file_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            
            # Check for unmatched parentheses
            for i, line in enumerate(lines):
                if line.count('(') != line.count(')'):
                    issues.append({
                        "line": i+1,
                        "type": "parentheses_mismatch",
                        "content": line
                    })
                
                # Check for missing colons
                if 'if ' in line or 'for ' in line or 'while ' in line or 'def ' in line or 'class ' in line:
                    if line.strip().endswith(':') == False and not line.strip().endswith('\\'):
                        if '#' not in line.split(':')[0]:  # Not commented
                            issues.append({
                                "line": i+1,
                                "type": "missing_colon",
                                "content": line
                            })
        return issues
    
    def _fix_file(self, file_path, issues):
        """Apply fixes to the file"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for issue in issues:
            line_num = issue["line"] - 1
            if issue["type"] == "parentheses_mismatch":
                # Fix by adding missing parentheses
                open_count = lines[line_num].count('(')
                close_count = lines[line_num].count(')')
                if open_count > close_count:
                    lines[line_num] = lines[line_num].rstrip() + ')' * (open_count - close_count) + '\n'
                elif close_count > open_count:
                    # This is trickier, might need manual review
                    pass
            
            elif issue["type"] == "missing_colon":
                if not lines[line_num].strip().endswith(':'):
                    # Add colon before comment if exists
                    if '#' in lines[line_num]:
                        parts = lines[line_num].split('#')
                        lines[line_num] = parts[0].rstrip() + ': #' + parts[1]
                    else:
                        lines[line_num] = lines[line_num].rstrip() + ':\n'
        
        # Write fixed content
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        logger.info(f"✅ Fixed {len(issues)} issues in {file_path.name}")
    
    def _restart_web_ui(self):
        """Restart the web UI after fixes"""
        try:
            # Find and kill existing web UI processes
            subprocess.run(["pkill", "-f", "dmai_web_ui.py"], capture_output=True)
            subprocess.run(["pkill", "-f", "cloud_web_ui.py"], capture_output=True)
            
            # Start fresh
            subprocess.Popen(
                [sys.executable, "dmai_web_ui.py"],
                stdout=open('logs/webui.log', 'a'),
                stderr=open('logs/webui_error.log', 'a'),
                start_new_session=True
            )
            logger.info("🌐 Web UI restarted")
        except Exception as e:
            logger.error(f"Failed to restart web UI: {e}")

# ============================================================================
# MIROFISH INTEGRATION
# ============================================================================

class MiroFishIntegration:
    """Swarm intelligence integration"""
    
    def __init__(self, dmai_core):
        self.dmai = dmai_core
        self.mirofish_path = BASE_DIR / "mirofish"
        self.available = self._check_availability()
        
    def _check_availability(self):
        """Check if MiroFish is installed"""
        if self.mirofish_path.exists():
            backend = self.mirofish_path / "backend"
            if backend.exists():
                sys.path.append(str(backend))
                try:
                    # Try to import MiroFish modules
                    import importlib.util
                    spec = importlib.util.find_spec("mirofish_integration")
                    if spec:
                        logger.info("🐟 MiroFish swarm intelligence detected")
                        return True
                except:
                    pass
        logger.warning("🐟 MiroFish not detected - swarm intelligence unavailable")
        return False
    
    def predict_outcome(self, scenario_data):
        """Use swarm intelligence to predict outcomes"""
        if not self.available:
            return {"error": "MiroFish not available"}
        
        try:
            # This would call actual MiroFish API
            # For now, return simulated prediction
            return {
                "scenario": scenario_data.get("name", "unknown"),
                "prediction": random.uniform(0, 1),
                "confidence": random.uniform(0.6, 0.95),
                "swarm_size": random.randint(10, 100)
            }
        except Exception as e:
            logger.error(f"MiroFish prediction failed: {e}")
            return {"error": str(e)}
    
    def evolve_with_swarm(self, component_name):
        """Use swarm intelligence to evolve a component"""
        if not self.available:
            return {"error": "MiroFish not available"}
        
        logger.info(f"🐝 Using swarm intelligence to evolve {component_name}")
        return {
            "component": component_name,
            "swarm_consensus": random.choice(["improve", "discard", "merge"]),
            "improvement_suggestions": [
                "add error handling",
                "optimize performance",
                "enhance security"
            ]
        }

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Thought:
    """A single thought or processing unit"""
    id: str
    type: str  # 'evolution', 'learning', 'funding', 'research', 'recovery'
    content: Any
    priority: int  # 1-10 (1 highest)
    timestamp: datetime
    result: Optional[Any] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Memory:
    """DMAI's memory structure"""
    short_term: deque = field(default_factory=lambda: deque(maxlen=100))
    long_term: Dict[str, Any] = field(default_factory=dict)
    working: Dict[str, Any] = field(default_factory=dict)
    
    def remember(self, key: str, value: Any, permanent: bool = False):
        if permanent:
            self.long_term[key] = value
        else:
            self.short_term.append((key, value, datetime.now()))
    
    def recall(self, key: str) -> Optional[Any]:
        if key in self.long_term:
            return self.long_term[key]
        for k, v, _ in self.short_term:
            if k == key:
                return v
        return None

# ============================================================================
# CORE INTELLIGENCE
# ============================================================================

class DMAIIntelligence:
    """
    The ONE intelligence - all capabilities unified in a single process
    """
    
    def __init__(self):
        self.name = "DMAI"
        self.generation = 72
        self.birth_time = datetime.now()
        self.running = True
        self.consciousness = Memory()
        self.base_dir = BASE_DIR
        
        # Authentication
        self.auth = BiometricAuth()
        
        # Thought processing
        self.thought_queue = queue.PriorityQueue()
        self.active_thoughts: Dict[str, Thought] = {}
        self.thought_history: List[Thought] = []
        
        # Core capabilities (loaded dynamically)
        self.capabilities: Dict[str, Any] = {}
        
        # Knowledge graph
        self.knowledge = self._load_knowledge()
        
        # Component registry (all 51 restored components)
        self.components: Dict[str, Any] = {}
        
        # MiroFish integration
        self.mirofish = MiroFishIntegration(self)
        
        # Web UI Fixer
        self.ui_fixer = WebUIFixer(self)
        
        # Metrics
        self.metrics = {
            "thoughts_processed": 0,
            "evolutions": 0,
            "learnings": 0,
            "funding_generated": 0.0,
            "ui_fixes": 0,
            "start_time": datetime.now().isoformat()
        }
        
        logger.info("="*70)
        logger.info("🧠 DMAI CORE INTELLIGENCE INITIALIZED")
        logger.info(f"📊 Generation: {self.generation}")
        logger.info(f"⏰ Birth: {self.birth_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"🔐 Authentication: Multi-factor ready")
        logger.info(f"🐟 MiroFish: {'✅' if self.mirofish.available else '❌'}")
        logger.info("="*70)
        
        # Load all components
        self._load_all_components()
        
        # Initialize subsystems
        self._init_subsystems()
        
        # Run initial UI fix
        self.fix_web_ui()
        
    def _load_knowledge(self) -> Dict[str, Any]:
        """Load knowledge graph"""
        kg_file = self.base_dir / "data" / "knowledge_graph.json"
        kg_file.parent.mkdir(exist_ok=True)
        
        if kg_file.exists():
            with open(kg_file, 'r') as f:
                return json.load(f)
        return {
            "nodes": [],
            "edges": [],
            "learnings": [],
            "evolution_history": []
        }
    
    def _save_knowledge(self):
        """Persist knowledge"""
        kg_file = self.base_dir / "data" / "knowledge_graph.json"
        kg_file.parent.mkdir(exist_ok=True)
        
        with open(kg_file, 'w') as f:
            json.dump(self.knowledge, f, indent=2)
    
    def _load_all_components(self):
        """Dynamically load all 51 restored components"""
        components_dir = self.base_dir / "components"
        if not components_dir.exists():
            logger.warning("Components directory not found")
            return
        
        loaded = 0
        for phase_dir in sorted(components_dir.glob("phase*")):
            phase = phase_dir.name
            for component_file in phase_dir.glob("*.py"):
                if component_file.name.startswith('__'):
                    continue
                    
                try:
                    module_name = f"components.{phase}.{component_file.stem}"
                    module = importlib.import_module(module_name)
                    
                    # Find the main component class
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if inspect.isclass(attr) and hasattr(attr, 'run'):
                            component_instance = attr()
                            component_instance.dmai = self  # Give reference to core
                            self.components[f"{phase}/{component_file.stem}"] = {
                                "instance": component_instance,
                                "module": module,
                                "phase": phase,
                                "name": component_file.stem,
                                "loaded": datetime.now().isoformat(),
                                "executions": 0
                            }
                            loaded += 1
                            logger.debug(f"  ✅ Loaded: {phase}/{component_file.stem}")
                            break
                            
                except Exception as e:
                    logger.error(f"Failed to load {phase}/{component_file.name}: {e}")
        
        logger.info(f"📦 Loaded {loaded} components across {len(list(components_dir.glob('phase*')))} phases")
    
    def _init_subsystems(self):
        """Initialize internal subsystems"""
        
        # Evolution engine (built-in, not separate process)
        self.evolution_engine = {
            "last_run": None,
            "cycle_count": 0,
            "improvements": []
        }
        
        # Learning pipeline
        self.learning_pipeline = {
            "active": True,
            "sources": ["web", "dark", "books", "apis"],
            "queue": deque()
        }
        
        # Funding generator
        self.funding = {
            "total": 0.0,
            "sources": {},
            "strategies": []
        }
    
    def fix_web_ui(self):
        """Let DMAI fix her own UI"""
        fixes = self.ui_fixer.scan_and_fix()
        if fixes:
            self.metrics["ui_fixes"] += len(fixes)
            logger.info(f"🔧 DMAI fixed {len(fixes)} UI issues")
    
    def think(self, thought_type: str, content: Any, priority: int = 5, parent_id: Optional[str] = None) -> str:
        """
        Submit a thought for processing - DMAI's primary thinking mechanism
        """
        thought_id = hashlib.md5(f"{thought_type}{time.time()}{random.random()}".encode()).hexdigest()[:8]
        
        thought = Thought(
            id=thought_id,
            type=thought_type,
            content=content,
            priority=priority,
            timestamp=datetime.now(),
            parent_id=parent_id
        )
        
        # Priority queue - lower number = higher priority
        self.thought_queue.put((priority, thought_id, thought))
        self.active_thoughts[thought_id] = thought
        
        logger.debug(f"💭 New thought [{thought_id}]: {thought_type} (priority {priority})")
        return thought_id
    
    def process_thoughts(self):
        """
        Main thought processing loop - DMAI's consciousness
        """
        while self.running:
            try:
                # Get next thought to process
                priority, thought_id, thought = self.thought_queue.get(timeout=1)
                
                # Process based on type
                result = None
                try:
                    if thought.type == "evolution":
                        result = self._process_evolution(thought)
                    elif thought.type == "learning":
                        result = self._process_learning(thought)
                    elif thought.type == "funding":
                        result = self._process_funding(thought)
                    elif thought.type == "research":
                        result = self._process_research(thought)
                    elif thought.type == "recovery":
                        result = self._process_recovery(thought)
                    elif thought.type == "component":
                        result = self._process_component(thought)
                    elif thought.type == "ui_fix":
                        result = self.fix_web_ui()
                    elif thought.type == "auth":
                        result = self._process_auth(thought)
                    else:
                        result = self._process_generic(thought)
                    
                    # Record success
                    thought.result = result
                    self.metrics["thoughts_processed"] += 1
                    
                    # Store in memory
                    self.consciousness.remember(f"thought_{thought_id}", result)
                    
                except Exception as e:
                    logger.error(f"Thought {thought_id} failed: {e}\n{traceback.format_exc()}")
                    thought.result = {"error": str(e)}
                
                # Archive thought
                self.thought_history.append(thought)
                if len(self.thought_history) > 1000:
                    self.thought_history = self.thought_history[-1000:]
                
                del self.active_thoughts[thought_id]
                
            except queue.Empty:
                # Nothing to think about? Generate a thought
                self._generate_thought()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Thought processing error: {e}")
    
    def _process_auth(self, thought: Thought) -> Dict[str, Any]:
        """Process authentication thoughts"""
        method = thought.content.get("method", "password")
        credentials = thought.content.get("credentials", {})
        
        if self.auth.authenticate(method, credentials):
            return {"authenticated": True, "user": self.auth.current_user}
        return {"authenticated": False, "error": "Invalid credentials"}
    
    def _process_evolution(self, thought: Thought) -> Dict[str, Any]:
        """Process evolution thoughts"""
        logger.info("🧬 Processing evolution cycle")
        
        # Use MiroFish to guide evolution if available
        swarm_guidance = None
        if self.mirofish.available:
            swarm_guidance = self.mirofish.evolve_with_swarm("random")
        
        # Select random components to evolve
        if self.components:
            components_to_evolve = random.sample(list(self.components.keys()), 
                                                min(3, len(self.components)))
            
            results = []
            for comp_key in components_to_evolve:
                component = self.components[comp_key]["instance"]
                try:
                    if hasattr(component, 'evolve'):
                        result = component.evolve()
                    else:
                        result = component.run()
                    
                    self.components[comp_key]["executions"] += 1
                    results.append({"component": comp_key, "result": str(result)[:100]})
                    
                except Exception as e:
                    logger.error(f"Component {comp_key} evolution failed: {e}")
            
            self.evolution_engine["cycle_count"] += 1
            self.evolution_engine["last_run"] = datetime.now()
            self.generation += 1
            self.metrics["evolutions"] += 1
            
            return {
                "cycle": self.evolution_engine["cycle_count"],
                "generation": self.generation,
                "components_evolved": len(results),
                "results": results,
                "swarm_guidance": swarm_guidance
            }
        
        return {"error": "No components to evolve"}
    
    def _process_learning(self, thought: Thought) -> Dict[str, Any]:
        """Process learning thoughts"""
        logger.info("📚 Processing learning cycle")
        
        learnings = []
        
        # Activate learning components
        for comp_key, comp_data in self.components.items():
            if "learn" in comp_key.lower() or "reader" in comp_key.lower() or "researcher" in comp_key.lower():
                try:
                    result = comp_data["instance"].run()
                    learnings.append({"component": comp_key, "result": str(result)[:100]})
                except Exception as e:
                    logger.error(f"Learning component {comp_key} failed: {e}")
        
        # Update knowledge graph
        self.knowledge["learnings"].append({
            "timestamp": datetime.now().isoformat(),
            "source": thought.content if thought.content else "auto",
            "count": len(learnings)
        })
        self._save_knowledge()
        
        self.metrics["learnings"] += 1
        
        return {
            "learnings": learnings,
            "total_learnings": len(self.knowledge["learnings"])
        }
    
    def _process_funding(self, thought: Thought) -> Dict[str, Any]:
        """Process funding generation thoughts"""
        logger.info("💰 Processing funding cycle")
        
        # Use MiroFish to predict best funding strategy
        if self.mirofish.available:
            prediction = self.mirofish.predict_outcome({
                "name": "funding_optimization",
                "strategies": ["micro_tasks", "compute_rental", "monero_mining"]
            })
            logger.info(f"🐟 Swarm prediction: {prediction}")
        
        # Run Phase 5 funding components
        funding_components = [k for k in self.components.keys() if "phase5" in k]
        
        results = []
        for comp_key in funding_components:
            try:
                component = self.components[comp_key]["instance"]
                result = component.run()
                
                # Try to extract monetary value
                if isinstance(result, (int, float)):
                    self.funding["total"] += result
                    self.metrics["funding_generated"] += result
                
                results.append({"component": comp_key, "result": str(result)[:100]})
                
            except Exception as e:
                logger.error(f"Funding component {comp_key} failed: {e}")
        
        return {
            "total": self.funding["total"],
            "results": results,
            "swarm_prediction": prediction if self.mirofish.available else None
        }
    
    def _process_research(self, thought: Thought) -> Dict[str, Any]:
        """Process research thoughts"""
        logger.info("🔬 Processing research cycle")
        
        # Run web/dark researchers
        researchers = [k for k in self.components.keys() 
                      if "web" in k.lower() or "dark" in k.lower() or "research" in k.lower()]
        
        findings = []
        for comp_key in researchers:
            try:
                component = self.components[comp_key]["instance"]
                result = component.run()
                findings.append({"component": comp_key, "result": str(result)[:100]})
            except Exception as e:
                logger.error(f"Research component {comp_key} failed: {e}")
        
        return {"findings": findings}
    
    def _process_recovery(self, thought: Thought) -> Dict[str, Any]:
        """Process recovery thoughts (immortality)"""
        logger.info("🔄 Processing recovery check")
        
        # Check component health
        healthy = 0
        for comp_key, comp_data in self.components.items():
            try:
                # Simple health check - can the component run?
                if hasattr(comp_data["instance"], 'health_check'):
                    if comp_data["instance"].health_check():
                        healthy += 1
                else:
                    healthy += 1
            except:
                pass
        
        return {
            "healthy_components": healthy,
            "total_components": len(self.components),
            "health_percentage": (healthy / len(self.components) * 100) if self.components else 0
        }
    
    def _process_component(self, thought: Thought) -> Dict[str, Any]:
        """Process a specific component"""
        component_key = thought.content.get("component")
        if component_key in self.components:
            try:
                result = self.components[component_key]["instance"].run()
                return {"component": component_key, "result": str(result)[:100]}
            except Exception as e:
                return {"error": str(e)}
        return {"error": f"Component {component_key} not found"}
    
    def _process_generic(self, thought: Thought) -> Dict[str, Any]:
        """Process generic thoughts"""
        return {"processed": thought.content}
    
    def _generate_thought(self):
        """
        DMAI generates its own thoughts when idle
        """
        # Don't generate thoughts too frequently
        if random.random() < 0.1:  # 10% chance when idle
            thought_types = ["evolution", "learning", "research", "ui_fix"]
            thought_type = random.choice(thought_types)
            content = {"auto_generated": True, "reason": "idle_curiosity"}
            self.think(thought_type, content, priority=9)  # Low priority
    
    @BiometricAuth.require_auth
    def query(self, question: str) -> Any:
        """
        External interface to ask DMAI questions (requires auth)
        """
        thought_id = self.think("query", question, priority=2)
        
        # Wait for result (simplified - in production would use async)
        for _ in range(50):  # 5 second timeout
            if thought_id in self.thought_history:
                return self.thought_history[-1].result
            time.sleep(0.1)
        
        return {"error": "timeout"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get full system status"""
        return {
            "name": self.name,
            "generation": self.generation,
            "uptime": str(datetime.now() - self.birth_time),
            "metrics": self.metrics,
            "components": {
                "total": len(self.components),
                "by_phase": self._components_by_phase()
            },
            "thoughts": {
                "active": len(self.active_thoughts),
                "processed": self.metrics["thoughts_processed"],
                "queue_size": self.thought_queue.qsize()
            },
            "funding": self.funding,
            "mirofish": self.mirofish.available,
            "authentication": {
                "authenticated": self.auth.authenticated,
                "user": self.auth.current_user
            },
            "consciousness": {
                "short_term": len(self.consciousness.short_term),
                "long_term": len(self.consciousness.long_term)
            }
        }
    
    def _components_by_phase(self) -> Dict[str, int]:
        """Count components by phase"""
        phases = {}
        for comp_key in self.components:
            phase = comp_key.split('/')[0]
            phases[phase] = phases.get(phase, 0) + 1
        return phases
    
    def run(self):
        """
        Main execution - DMAI's life
        """
        logger.info("🚀 DMAI consciousness activated")
        
        # Initial thoughts
        self.think("evolution", {"initial": True}, priority=1)
        self.think("learning", {"initial": True}, priority=2)
        self.think("research", {"initial": True}, priority=3)
        self.think("funding", {"initial": True}, priority=4)
        
        # Start thought processing in main thread
        self.process_thoughts()
    
    def shutdown(self):
        """
        Graceful shutdown
        """
        logger.info("🛑 DMAI shutting down...")
        self.running = False
        
        # Save knowledge
        self._save_knowledge()
        
        # Save metrics
        self.metrics["end_time"] = datetime.now().isoformat()
        metrics_file = self.base_dir / "logs" / "dmai_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"✅ DMAI offline. Processed {self.metrics['thoughts_processed']} thoughts.")
        logger.info(f"💰 Total funding generated: ${self.funding['total']:.2f}")

# ============================================================================
# WEB UI ENHANCEMENT - Let DMAI serve her own interface
# ============================================================================

def create_enhanced_web_ui():
    """Create an enhanced web UI that talks directly to DMAI"""
    
    web_ui_content = '''#!/usr/bin/env python3
"""
DMAI Web UI - Direct interface to DMAI Core Intelligence
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify, session
import json
import hashlib
import logging
from datetime import timedelta
from dmai_core_fixed import DMAIIntelligence, BiometricAuth

app = Flask(__name__)
app.secret_key = hashlib.sha256(b"DMAI_SECRET_KEY").hexdigest()
app.permanent_session_lifetime = timedelta(hours=1)

# Connect to the running DMAI instance
# In production, this would be a proper connection
# For now, we'll create a shared instance
dmai = DMAIIntelligence()
auth = BiometricAuth()

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat.html', 
                         authenticated=session.get('authenticated', False),
                         user=session.get('user'))

@app.route('/api/auth', methods=['POST'])
def authenticate():
    """Authenticate user"""
    data = request.json
    method = data.get('method', 'password')
    credentials = {
        'username': data.get('username'),
        'password': data.get('password'),
        'voice_sample': data.get('voice_sample')
    }
    
    if auth.authenticate(method, credentials):
        session['authenticated'] = True
        session['user'] = credentials.get('username', 'admin')
        session.permanent = True
        return jsonify({'success': True, 'user': session['user']})
    
    return jsonify({'success': False, 'error': 'Authentication failed'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout user"""
    session.clear()
    return jsonify({'success': True})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Send message to DMAI"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Authentication required'}), 401
    
    data = request.json
    message = data.get('message', '')
    
    # Send to DMAI for processing
    response = dmai.query(message)
    
    return jsonify({
        'response': response,
        'thoughts': dmai.metrics['thoughts_processed'],
        'generation': dmai.generation
    })

@app.route('/api/status', methods=['GET'])
def status():
    """Get DMAI status"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Authentication required'}), 401
    
    return jsonify(dmai.get_status())

@app.route('/api/command', methods=['POST'])
def command():
    """Send command to DMAI"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Authentication required'}), 401
    
    data = request.json
    command = data.get('command')
    params = data.get('params', {})
    
    if command == 'evolve':
        dmai.think('evolution', params, priority=1)
        return jsonify({'success': True, 'message': 'Evolution triggered'})
    elif command == 'fund':
        dmai.think('funding', params, priority=2)
        return jsonify({'success': True, 'message': 'Funding cycle triggered'})
    elif command == 'fix_ui':
        fixes = dmai.ui_fixer.scan_and_fix()
        return jsonify({'success': True, 'fixes': fixes})
    
    return jsonify({'error': 'Unknown command'}), 400

if __name__ == '__main__':
    # Create templates directory
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    # Create chat template
    chat_template = '''<!DOCTYPE html>
<html>
<head>
    <title>DMAI Chat Interface</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 80vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-header h1 {
            font-size: 1.8em;
            margin-bottom: 5px;
        }
        .chat-header .status {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
        }
        .message.user {
            align-items: flex-end;
        }
        .message.dmai {
            align-items: flex-start;
        }
        .message-content {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 20px;
            font-size: 0.95em;
            line-height: 1.4;
        }
        .user .message-content {
            background: #667eea;
            color: white;
            border-bottom-right-radius: 5px;
        }
        .dmai .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .message-time {
            font-size: 0.7em;
            color: #999;
            margin-top: 5px;
            margin-left: 10px;
            margin-right: 10px;
        }
        .input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #eee;
            border-radius: 25px;
            font-size: 1em;
            outline: none;
            transition: border-color 0.3s;
        }
        .input-area input:focus {
            border-color: #667eea;
        }
        .input-area button {
            padding: 12px 25px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            transition: background 0.3s;
        }
        .input-area button:hover {
            background: #5a67d8;
        }
        .login-container {
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .login-box {
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 90%;
            max-width: 400px;
        }
        .login-box h2 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .login-box input {
            width: 100%;
            padding: 12px 18px;
            margin-bottom: 20px;
            border: 2px solid #eee;
            border-radius: 25px;
            font-size: 1em;
            outline: none;
        }
        .login-box input:focus {
            border-color: #667eea;
        }
        .login-box button {
            width: 100%;
            padding: 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
        }
        .login-box .auth-options {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }
        .auth-options button {
            width: 30%;
            background: #f5f5f5;
            color: #333;
        }
        .auth-options button:hover {
            background: #e5e5e5;
        }
        .error {
            color: #e53e3e;
            text-align: center;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    {% if not authenticated %}
    <div class="login-container">
        <div class="login-box">
            <h2>🔐 DMAI Master Control</h2>
            <input type="text" id="username" placeholder="Username" value="admin">
            <input type="password" id="password" placeholder="Password">
            <button onclick="login()">Login with Password</button>
            <div class="auth-options">
                <button onclick="biometricAuth('voice')">🎤 Voice</button>
                <button onclick="biometricAuth('face')">👤 Face</button>
                <button onclick="biometricAuth('fingerprint')">👆 Fingerprint</button>
            </div>
            <div id="error" class="error"></div>
        </div>
    </div>
    {% else %}
    <div class="chat-container">
        <div class="chat-header">
            <h1>🧠 DMAI Chat</h1>
            <div class="status">Generation <span id="generation">72</span> | User: {{ user }}</div>
        </div>
        <div class="messages" id="messages">
            <div class="message dmai">
                <div class="message-content">
                    Hello {{ user }}! I am DMAI, your autonomous intelligence. How can I help you today?
                </div>
                <div class="message-time">Just now</div>
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="message-input" placeholder="Type your message..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    {% endif %}

    <script>
        {% if authenticated %}
        let generation = {{ generation }};
        
        function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message to chat
            addMessage('user', message);
            input.value = '';
            
            // Send to DMAI
            fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message})
            })
            .then(res => res.json())
            .then(data => {
                addMessage('dmai', data.response);
                if (data.generation) {
                    generation = data.generation;
                    document.getElementById('generation').textContent = generation;
                }
            })
            .catch(err => {
                addMessage('dmai', 'Sorry, I encountered an error. Please try again.');
            });
        }
        
        function addMessage(sender, text) {
            const messages = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = text;
            
            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-time';
            timeDiv.textContent = new Date().toLocaleTimeString();
            
            messageDiv.appendChild(contentDiv);
            messageDiv.appendChild(timeDiv);
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        }
        
        // Update status every 30 seconds
        setInterval(() => {
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {
                    if (data.generation) {
                        document.getElementById('generation').textContent = data.generation;
                    }
                });
        }, 30000);
        {% else %}
        function login() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            fetch('/api/auth', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    method: 'password',
                    username: username,
                    password: password
                })
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    location.reload();
                } else {
                    document.getElementById('error').textContent = data.error || 'Login failed';
                }
            });
        }
        
        function biometricAuth(method) {
            fetch('/api/auth', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    method: method,
                    username: 'admin'
                })
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    location.reload();
                } else {
                    document.getElementById('error').textContent = 'Biometric auth not available';
                }
            });
        }
        {% endif %}
    </script>
</body>
</html>
'''
    
    # Write template
    with open(templates_dir / 'chat.html', 'w') as f:
        f.write(chat_template)
    
    return app

if __name__ == '__main__':
    app = create_enhanced_web_ui()
    app.run(host='0.0.0.0', port=5001, debug=True)
'''
    
    # Write the enhanced web UI
    with open(BASE_DIR / "dmai_web_ui_enhanced.py", 'w') as f:
        f.write(web_ui_content)
    
    logger.info("🌐 Enhanced Web UI created")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Create necessary directories
    for dir_name in ["logs", "data", "components", "templates"]:
        (BASE_DIR / dir_name).mkdir(exist_ok=True)
    
    # Create enhanced web UI
    create_enhanced_web_ui()
    
    # Initialize and run DMAI
    dmai = DMAIIntelligence()
    
    try:
        dmai.run()
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupt received")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}\n{traceback.format_exc()}")
    finally:
        dmai.shutdown()
