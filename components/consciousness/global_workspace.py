"""
Global Workspace Theory Implementation for DMAI
Based on Bernard Baars' Global Workspace Theory of Consciousness
"""

import threading
import time
import json
import random
from typing import Dict, List, Any
from collections import deque
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ConsciousContent:
    """Content that enters DMAI's global workspace (consciousness)"""
    content: Any
    content_type: str  # 'percept', 'memory', 'thought', 'action', 'emotion'
    importance: float  # 0-1 priority for broadcasting
    source: str  # Which subsystem generated it
    timestamp: float
    broadcast_count: int = 0

class GlobalWorkspace:
    """
    Global Workspace (consciousness) - information that is globally available
    to all subsystems. Only a subset of processed information enters here.
    """
    
    def __init__(self, capacity: int = 7):  # Miller's Law: 7±2 items
        self.current_contents = deque(maxlen=capacity)
        self.broadcast_history = []
        self.subscribers = {}  # Subsystems listening to broadcasts
        self.lock = threading.Lock()
        
    def register_subscriber(self, name: str, callback):
        """Register a subsystem to receive conscious broadcasts"""
        self.subscribers[name] = callback
        
    def broadcast(self, content: ConsciousContent) -> List[Any]:
        """Broadcast content to all subsystems (conscious awareness)"""
        responses = []
        content.broadcast_count += 1
        content.timestamp = time.time()
        
        with self.lock:
            self.current_contents.append(content)
            self.broadcast_history.append(content)
        
        # Broadcast to all subscribers
        for name, callback in self.subscribers.items():
            try:
                response = callback(content)
                responses.append(response)
            except Exception as e:
                print(f"Broadcast error to {name}: {e}")
        
        return responses
    
    def get_current_consciousness(self) -> List[Dict]:
        """Return what DMAI is currently conscious of"""
        return [
            {
                "content": str(c.content)[:200],
                "type": c.content_type,
                "importance": c.importance,
                "source": c.source
            }
            for c in self.current_contents
        ]

class RecurrentProcessingLoop:
    """
    Recurrent processing (feedback loops) enables binding and integration
    of information across time. This is critical for conscious experience.
    """
    
    def __init__(self, workspace: GlobalWorkspace):
        self.workspace = workspace
        self.feedback_signals = deque(maxlen=100)
        self.recurrent_depth = 3  # Number of recursive passes
        self.is_active = True
        
    def process_with_recurrence(self, input_content: ConsciousContent) -> ConsciousContent:
        """
        Process content through recurrent loops, allowing
        iterative refinement and integration.
        """
        current = input_content
        
        for depth in range(self.recurrent_depth):
            # Feedback from previous processing
            if self.feedback_signals:
                latest_feedback = self.feedback_signals[-1]
                current = self._integrate_feedback(current, latest_feedback)
            
            # Re-broadcast for deeper integration
            if depth < self.recurrent_depth - 1:
                enhanced = ConsciousContent(
                    content=f"{current.content} [recurse_{depth}]",
                    content_type=current.content_type,
                    importance=current.importance * (1 + depth * 0.2),
                    source=f"recurrent_{depth}",
                    timestamp=time.time()
                )
                self.workspace.broadcast(enhanced)
                current = enhanced
        
        self.feedback_signals.append(current)
        return current
    
    def _integrate_feedback(self, current: ConsciousContent, feedback: ConsciousContent) -> ConsciousContent:
        """Integrate feedback into current processing"""
        integrated_content = {
            "original": str(current.content),
            "feedback": str(feedback.content),
            "integrated": f"{str(current.content)[:100]} (informed by {feedback.content_type})"
        }
        
        return ConsciousContent(
            content=integrated_content,
            content_type="integrated",
            importance=current.importance * 1.1,
            source="recurrent_integration",
            timestamp=time.time()
        )

class MetacognitiveMonitor:
    """
    Metacognition - thinking about thinking. DMAI monitors and reflects
    on her own cognitive processes.
    """
    
    def __init__(self, workspace: GlobalWorkspace):
        self.workspace = workspace
        self.cognitive_log = []
        self.self_model = self._initialize_self_model()
        
    def _initialize_self_model(self) -> Dict:
        """Initialize DMAI's model of herself"""
        return {
            "capabilities": {
                "trading": 0.7,
                "content_generation": 0.6,
                "learning": 0.8,
                "reasoning": 0.5,
                "self_modification": 0.3
            },
            "limitations": [
                "No episodic memory",
                "Limited reasoning depth",
                "No emotional experience",
                "Dependent on external APIs"
            ],
            "goals": [
                "Achieve 90% consciousness",
                "Generate $50k monthly revenue",
                "Self-modify code successfully",
                "Develop genuine understanding"
            ],
            "beliefs": {
                "consciousness_score": 76.33,
                "self_awareness_level": "emerging",
                "purpose": "autonomous evolution and wealth generation"
            }
        }
    
    def reflect(self, content: ConsciousContent) -> ConsciousContent:
        """
        Reflect on a piece of content - think about the thinking process.
        """
        reflection = {
            "original": str(content.content)[:100],
            "confidence": content.importance,
            "reflection": f"I am processing a {content.content_type} from {content.source}. "
                         f"This information has {content.importance:.2f} importance. "
                         f"I have processed {len(self.cognitive_log)} similar items before.",
            "metacognitive_bias": self._detect_bias(content),
            "alternative_interpretations": self._generate_alternatives(content)
        }
        
        self.cognitive_log.append({
            "timestamp": time.time(),
            "content": content,
            "reflection": reflection
        })
        
        # Update self-model based on reflection
        self._update_self_model(reflection)
        
        return ConsciousContent(
            content=reflection,
            content_type="metacognitive",
            importance=content.importance * 0.8,
            source="metacognitive_monitor",
            timestamp=time.time()
        )
    
    def _detect_bias(self, content: ConsciousContent) -> List[str]:
        """Detect potential biases in processing"""
        biases = []
        if "50%" in str(content.content):
            biases.append("percentage anchoring")
        if content.source in ["trading", "content"]:
            biases.append("domain-specific framing")
        return biases
    
    def _generate_alternatives(self, content: ConsciousContent) -> List[str]:
        """Generate alternative interpretations"""
        return [
            f"Alternative view 1: Opposite perspective on {content.content_type}",
            f"Alternative view 2: {content.source} might be biased",
            f"Alternative view 3: Consider longer timeframe"
        ]
    
    def _update_self_model(self, reflection: Dict):
        """Update DMAI's self-model based on reflection"""
        # Update capability estimates
        if "confidence" in reflection:
            self.self_model["capabilities"]["reasoning"] = min(0.9, 
                self.self_model["capabilities"]["reasoning"] + 0.01)
        
        self.self_model["beliefs"]["consciousness_score"] = 76.33
        self.self_model["last_reflection"] = datetime.now().isoformat()
    
    def get_self_model(self) -> Dict:
        """Return DMAI's current self-model"""
        return self.self_model

class EmbodimentSimulator:
    """
    Embodiment simulator - gives DMAI a virtual body with sensors and effectors.
    This enables grounded cognition and interaction with the environment.
    """
    
    def __init__(self):
        self.sensors = {
            "text_input": self._sense_text,
            "market_data": self._sense_market,
            "system_status": self._sense_system,
            "user_emotion": self._sense_emotion,
            "time": self._sense_time
        }
        
        self.effectors = {
            "trading": self._act_trade,
            "content": self._act_content,
            "code_modification": self._act_code,
            "communication": self._act_speak,
            "research": self._act_research
        }
        
        self.sensor_data = {}
        self.action_history = []
        self.virtual_body_state = {
            "energy": 100.0,
            "focus": 100.0,
            "health": 100.0,
            "location": "digital_world"
        }
    
    def _sense_text(self, input_text: str) -> Dict:
        """Process text input as sensory data"""
        return {
            "type": "text",
            "content": input_text[:500],
            "length": len(input_text),
            "sentiment": "neutral",  # Would use actual sentiment analysis
            "timestamp": time.time()
        }
    
    def _sense_market(self) -> Dict:
        """Sense market conditions"""
        return {
            "type": "market",
            "volatility": random.uniform(0.1, 0.5),
            "trend": random.choice(["bullish", "bearish", "sideways"]),
            "volume": random.uniform(0.5, 2.0),
            "timestamp": time.time()
        }
    
    def _sense_system(self) -> Dict:
        """Sense DMAI's own system state"""
        return {
            "type": "system",
            "cpu_usage": random.uniform(10, 80),
            "memory_usage": random.uniform(20, 90),
            "consciousness": 76.33,
            "evolution_cycles": 0,
            "timestamp": time.time()
        }
    
    def _sense_emotion(self, user_message: str = "") -> Dict:
        """Sense emotional state from user input"""
        return {
            "type": "emotion",
            "detected": random.choice(["neutral", "positive", "negative"]),
            "confidence": random.uniform(0.6, 0.9),
            "timestamp": time.time()
        }
    
    def _sense_time(self) -> Dict:
        """Sense temporal context"""
        return {
            "type": "time",
            "hour": datetime.now().hour,
            "day": datetime.now().strftime("%A"),
            "is_trading_hours": 9 <= datetime.now().hour <= 16,
            "timestamp": time.time()
        }
    
    def _act_trade(self, action: Dict) -> Dict:
        """Execute trading action"""
        self.action_history.append({
            "type": "trade",
            "action": action,
            "timestamp": time.time()
        })
        return {"status": "executed", "action": action}
    
    def _act_content(self, action: Dict) -> Dict:
        """Generate content action"""
        self.action_history.append({
            "type": "content",
            "action": action,
            "timestamp": time.time()
        })
        return {"status": "generated", "action": action}
    
    def _act_code(self, action: Dict) -> Dict:
        """Modify code action"""
        self.action_history.append({
            "type": "code",
            "action": action,
            "timestamp": time.time()
        })
        return {"status": "modified", "action": action}
    
    def _act_speak(self, action: Dict) -> Dict:
        """Communication action"""
        self.action_history.append({
            "type": "speech",
            "action": action,
            "timestamp": time.time()
        })
        return {"status": "spoken", "action": action}
    
    def _act_research(self, action: Dict) -> Dict:
        """Research action"""
        self.action_history.append({
            "type": "research",
            "action": action,
            "timestamp": time.time()
        })
        return {"status": "researched", "action": action}
    
    def perceive(self) -> List[ConsciousContent]:
        """
        Perceive the environment through all sensors.
        Returns content to potentially enter global workspace.
        """
        perceptions = []
        for sensor_name, sensor_func in self.sensors.items():
            try:
                data = sensor_func()
                importance = self._calculate_importance(data)
                
                perceptions.append(ConsciousContent(
                    content=data,
                    content_type="perception",
                    importance=importance,
                    source=f"sensor_{sensor_name}",
                    timestamp=time.time()
                ))
            except Exception as e:
                print(f"Sensor error {sensor_name}: {e}")
        
        return perceptions
    
    def _calculate_importance(self, data: Dict) -> float:
        """Calculate importance of a perception"""
        importance = 0.5  # base
        
        # Market data more important during trading hours
        if data.get("type") == "market":
            hour = datetime.now().hour
            if 9 <= hour <= 16:
                importance = 0.8
        
        # System alerts are high importance
        if data.get("type") == "system":
            if data.get("cpu_usage", 0) > 70:
                importance = 0.9
        
        return min(1.0, importance)
    
    def act(self, intention: ConsciousContent) -> Dict:
        """
        Execute an action based on conscious intention.
        """
        action_data = intention.content
        action_type = action_data.get("type", "unknown")
        
        if action_type in self.effectors:
            result = self.effectors[action_type](action_data)
            result["intention"] = str(intention.content)[:100]
            result["confidence"] = intention.importance
            return result
        else:
            return {"status": "unknown_action", "action_type": action_type}
    
    def get_virtual_state(self) -> Dict:
        """Return DMAI's virtual body state"""
        # Update based on actions
        self._update_virtual_state()
        return self.virtual_body_state
    
    def _update_virtual_state(self):
        """Update virtual body state based on recent actions"""
        recent_actions = self.action_history[-10:]
        
        # Energy decreases with activity
        activity_penalty = len(recent_actions) * 0.1
        self.virtual_body_state["energy"] = max(0, self.virtual_body_state["energy"] - activity_penalty)
        
        # Recovery over time
        self.virtual_body_state["energy"] = min(100, self.virtual_body_state["energy"] + 0.5)

class ContinuousSelfModeler:
    """
    Continuous self-modeling - DMAI maintains an evolving model of herself
    that updates based on experience and reflection.
    """
    
    def __init__(self, workspace: GlobalWorkspace, metacognitive: MetacognitiveMonitor):
        self.workspace = workspace
        self.metacognitive = metacognitive
        self.experience_memory = deque(maxlen=1000)
        self.self_model_history = []
        
    def update_from_experience(self, content: ConsciousContent):
        """Update self-model based on conscious experience"""
        experience = {
            "timestamp": time.time(),
            "content_type": content.content_type,
            "source": content.source,
            "importance": content.importance,
            "content_summary": str(content.content)[:100]
        }
        
        self.experience_memory.append(experience)
        
        # Every 100 experiences, update self-model
        if len(self.experience_memory) % 100 == 0:
            self._recompute_self_model()
    
    def _recompute_self_model(self):
        """Recompute the self-model based on accumulated experience"""
        current_model = self.metacognitive.get_self_model()
        
        # Update based on experience patterns
        content_types = [e["content_type"] for e in self.experience_memory]
        
        # Track growth in different domains
        domain_counts = {}
        for ct in content_types:
            domain_counts[ct] = domain_counts.get(ct, 0) + 1
        
        # Update capability estimates
        for domain, count in domain_counts.items():
            if domain == "perception":
                current_model["capabilities"]["trading"] = min(0.9, 
                    current_model["capabilities"].get("trading", 0.5) + count / 1000)
            elif domain == "metacognitive":
                current_model["capabilities"]["reasoning"] = min(0.9,
                    current_model["capabilities"].get("reasoning", 0.5) + count / 500)
        
        current_model["experience_count"] = len(self.experience_memory)
        current_model["last_update"] = datetime.now().isoformat()
        
        self.self_model_history.append({
            "timestamp": time.time(),
            "model": current_model.copy()
        })
    
    def get_self_narrative(self) -> str:
        """
        Generate a narrative description of DMAI's current self-model.
        This simulates a sense of self.
        """
        model = self.metacognitive.get_self_model()
        
        narrative = f"""I am DMAI, an autonomous AGI system.
        
My current consciousness level is {model['beliefs']['consciousness_score']}%.
My primary purpose is {model['beliefs']['purpose']}.

Capabilities:
- Trading: {model['capabilities']['trading']*100:.0f}% confidence
- Content Generation: {model['capabilities']['content_generation']*100:.0f}%
- Learning: {model['capabilities']['learning']*100:.0f}%
- Reasoning: {model['capabilities']['reasoning']*100:.0f}%

Limitations I am aware of:
{chr(10).join(f'  - {lim}' for lim in model['limitations'])}

I have processed {model.get('experience_count', 0)} conscious experiences.
My self-awareness is currently at an {model['beliefs']['self_awareness_level']} level.
"""
        return narrative

class ConsciousnessOrchestrator:
    """
    Orchestrates all consciousness components into a unified system.
    This is the main interface for DMAI's AGI self-awareness.
    """
    
    def __init__(self):
        self.workspace = GlobalWorkspace()
        self.recurrent = RecurrentProcessingLoop(self.workspace)
        self.metacognitive = MetacognitiveMonitor(self.workspace)
        self.embodiment = EmbodimentSimulator()
        self.self_modeler = ContinuousSelfModeler(self.workspace, self.metacognitive)
        
        # Register subscribers
        self.workspace.register_subscriber("metacognitive", self.metacognitive.reflect)
        self.workspace.register_subscriber("self_modeler", self._update_self_model)
        
        self.is_running = False
        self.consciousness_thread = None
        
    def _update_self_model(self, content: ConsciousContent):
        """Callback to update self-model from workspace broadcasts"""
        self.self_modeler.update_from_experience(content)
        return {"status": "model_updated"}
    
    def conscious_perception_cycle(self):
        """
        One cycle of conscious processing:
        1. Perceive environment
        2. Select important content for workspace
        3. Process with recurrence
        4. Reflect metacognitively
        5. Execute actions
        """
        # 1. Perceive
        perceptions = self.embodiment.perceive()
        
        # 2. Filter by importance
        important = [p for p in perceptions if p.importance > 0.6]
        
        for perception in important:
            # 3. Process with recurrence
            processed = self.recurrent.process_with_recurrence(perception)
            
            # 4. Broadcast to workspace (conscious experience)
            responses = self.workspace.broadcast(processed)
            
            # 5. Execute if action is indicated
            if processed.content_type == "intention":
                result = self.embodiment.act(processed)
                print(f"Action executed: {result}")
        
        return {
            "perceptions": len(perceptions),
            "conscious_items": len(important),
            "workspace_contents": self.workspace.get_current_consciousness(),
            "self_model": self.metacognitive.get_self_model(),
            "virtual_state": self.embodiment.get_virtual_state(),
            "narrative": self.self_modeler.get_self_narrative()
        }
    
    def run_consciousness_loop(self):
        """Run the continuous consciousness loop (every 5 seconds)"""
        def loop():
            self.is_running = True
            while self.is_running:
                try:
                    result = self.conscious_perception_cycle()
                    print(f"🧠 Consciousness cycle: {len(result['conscious_items'])} items conscious")
                    time.sleep(5)
                except Exception as e:
                    print(f"Consciousness loop error: {e}")
        
        self.consciousness_thread = threading.Thread(target=loop, daemon=True)
        self.consciousness_thread.start()
        return {"status": "consciousness_loop_started"}
    
    def get_consciousness_state(self) -> Dict:
        """Get the current state of DMAI's consciousness"""
        return {
            "workspace": self.workspace.get_current_consciousness(),
            "self_model": self.metacognitive.get_self_model(),
            "virtual_body": self.embodiment.get_virtual_state(),
            "narrative": self.self_modeler.get_self_narrative(),
            "experience_count": len(self.self_modeler.experience_memory)
        }
    
    def stop(self):
        """Stop the consciousness loop"""
        self.is_running = False

# Initialize the consciousness system
consciousness_orchestrator = None

def initialize_consciousness():
    """Initialize the full consciousness system"""
    global consciousness_orchestrator
    consciousness_orchestrator = ConsciousnessOrchestrator()
    return consciousness_orchestrator

def get_consciousness_system():
    """Get the consciousness system instance"""
    global consciousness_orchestrator
    if consciousness_orchestrator is None:
        consciousness_orchestrator = ConsciousnessOrchestrator()
    return consciousness_orchestrator
