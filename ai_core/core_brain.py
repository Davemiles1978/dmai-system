import os
import json
import logging
from datetime import datetime
import importlib


# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DMAIBrain:
    """The actual intelligence behind DMAI"""
    
    def __init__(self, brain_dir='ai_core'):
        self.brain_dir = brain_dir
        self.models = self.load_models()
        self.knowledge_base = self.load_knowledge()
        self.capabilities = self.load_capabilities()
        self.active = False
        self.current_task = None
        
    def load_models(self):
        """Load available AI models"""
        models = {
            "tiny": None,
            "small": None,
            "large": None,
            "vision": None,
            "audio": None,
        }
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("🤗 Transformers available")
            models["transformers"] = True
        except ImportError:
            logger.warning("Transformers not installed")
            
        try:
            import torch
            logger.info(f"🔥 PyTorch available: {torch.__version__}")
            models["pytorch"] = True
        except ImportError:
            logger.warning("PyTorch not installed")
            
        return models
    
    def load_knowledge(self):
        """Load knowledge base"""
        kb_file = os.path.join(self.brain_dir, 'knowledge', 'knowledge_base.json')
        if os.path.exists(kb_file):
            with open(kb_file, 'r') as f:
                return json.load(f)
        else:
            os.makedirs(os.path.join(self.brain_dir, 'knowledge'), exist_ok=True)
            default_kb = {
                "facts": {},
                "learned": [],
                "connections": [],
                "last_updated": datetime.now().isoformat()
            }
            with open(kb_file, 'w') as f:
                json.dump(default_kb, f, indent=2)
            return default_kb
    
    def load_capabilities(self):
        """Load capability modules"""
        capabilities = {}
        caps_dir = os.path.join(self.brain_dir, 'capabilities')
        os.makedirs(caps_dir, exist_ok=True)
        
        capability_list = [
            "reasoning", "planning", "memory", "learning",
            "creation", "analysis", "communication", "self_improvement"
        ]
        
        for cap in capability_list:
            cap_file = os.path.join(caps_dir, f"{cap}.json")
            if os.path.exists(cap_file):
                with open(cap_file, 'r') as f:
                    capabilities[cap] = json.load(f)
            else:
                capabilities[cap] = {
                    "level": 0.1,
                    "experience": 0,
                    "last_used": None,
                    "enabled": True
                }
                with open(cap_file, 'w') as f:
                    json.dump(capabilities[cap], f, indent=2)
                
        return capabilities
    
    def think(self, input_text, context=None):
        """Process input and generate response"""
        logger.info(f"🤔 DMAI thinking about: {input_text[:50]}...")
        
        analysis = self.analyze_input(input_text)
        
        # For now, simple response - will be replaced with actual model inference
        response_text = self.generate_response(input_text, analysis)
        
        response = {
            "understanding": analysis,
            "needs_evolution": self.should_evolve(),
            "confidence": 0.7,
            "response_text": response_text,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update capability usage
        if analysis.get("requires_creation", False):
            self.update_capability("creation", 0.01)
        if analysis.get("requires_research", False):
            self.update_capability("reasoning", 0.01)
            
        self.save_state()
        return response
    
    def generate_response(self, input_text, analysis):
        """Generate actual response based on input"""
        if analysis.get("requires_creation", False):
            return f"I'll create that for you. Processing your request for: {input_text}"
        elif analysis.get("requires_research", False):
            return f"Let me research that. Looking into: {input_text}"
        elif analysis.get("requires_reasoning", False):
            return f"Let me think about that. Analyzing: {input_text}"
        else:
            return f"I understand. Processing: {input_text}"
    
    def analyze_input(self, text):
        """Understand what's being asked"""
        analysis = {
            "complexity": len(text.split()) / 10,
            "requires_knowledge": "?" in text or "what" in text.lower(),
            "requires_creation": any(word in text.lower() for word in ["create", "make", "generate"]),
            "requires_research": any(word in text.lower() for word in ["research", "find", "search"]),
            "requires_reasoning": any(word in text.lower() for word in ["why", "how", "explain"])
        }
        return analysis
    
    def update_capability(self, capability_name, increment):
        """Update capability level based on usage"""
        if capability_name in self.capabilities:
            self.capabilities[capability_name]["level"] = min(1.0, 
                self.capabilities[capability_name]["level"] + increment)
            self.capabilities[capability_name]["experience"] += 1
            self.capabilities[capability_name]["last_used"] = datetime.now().isoformat()
    
    def should_evolve(self):
        """Determine if DMAI should evolve"""
        # Evolve if:
        # 1. It's been a while since last evolution
        # 2. Capabilities are plateauing
        # 3. New patterns detected
        
        evolution_file = os.path.join(self.brain_dir, 'evolution', 'last_evolution.txt')
        if os.path.exists(evolution_file):
            with open(evolution_file, 'r') as f:
                last_evo = datetime.fromisoformat(f.read().strip())
            days_since = (datetime.now() - last_evo).days
            return days_since >= 7  # Evolve weekly
        return True
    
    def evolve(self):
        """Self-improvement cycle"""
        logger.info("🧬 DMAI beginning evolution cycle...")
        
        from ai_core.evolution_engine import EvolutionEngine
        engine = EvolutionEngine()
        
        # Get performance metrics
        metrics = {
            "accuracy": self.calculate_accuracy(),
            "speed": self.calculate_speed(),
            "knowledge_gaps": len(self.knowledge_base.get("learned", []))
        }
        
        # Run evolution
        brain_state = {
            "capabilities": self.capabilities,
            "knowledge": self.knowledge_base
        }
        
        evolution_result = engine.evolve(brain_state, metrics)
        
        # Record evolution
        evo_file = os.path.join(self.brain_dir, 'evolution', 'last_evolution.txt')
        with open(evo_file, 'w') as f:
            f.write(datetime.now().isoformat())
        
        return evolution_result
    
    def calculate_accuracy(self):
        """Calculate how accurate DMAI has been"""
        # Would track success rate of responses
        return 0.85
    
    def calculate_speed(self):
        """Calculate response speed"""
        # Would track average response time
        return 0.7
    
    def get_next_generation(self):
        """Track DMAI's version"""
        gen_file = os.path.join(self.brain_dir, 'evolution', 'generation.txt')
        if os.path.exists(gen_file):
            with open(gen_file, 'r') as f:
                return int(f.read().strip())
        return 1
    
    def create(self, request):
        """Create something based on request"""
        creation_types = {
            "video": self.create_video,
            "code": self.create_code,
            "image": self.create_image,
            "music": self.create_music,
            "document": self.create_document
        }
        
        for ctype, func in creation_types.items():
            if ctype in request.lower():
                return func(request)
        
        return {"error": "Creation type not supported", "request": request}
    
    def create_video(self, request):
        """Video creation capability"""
        logger.info(f"🎬 Creating video: {request}")
        return {
            "status": "processing", 
            "type": "video",
            "estimated_time": "5 minutes",
            "request": request
        }
    
    def create_code(self, request):
        """Code generation capability"""
        logger.info(f"💻 Generating code: {request}")
        return {
            "status": "processing", 
            "type": "code",
            "estimated_time": "30 seconds",
            "request": request
        }
    
    def create_image(self, request):
        """Image generation capability"""
        logger.info(f"🖼️ Generating image: {request}")
        return {
            "status": "processing", 
            "type": "image",
            "estimated_time": "2 minutes",
            "request": request
        }
    
    def create_music(self, request):
        """Music generation capability"""
        logger.info(f"🎵 Generating music: {request}")
        return {
            "status": "processing", 
            "type": "music",
            "estimated_time": "3 minutes",
            "request": request
        }
    
    def create_document(self, request):
        """Document creation capability"""
        logger.info(f"📄 Creating document: {request}")
        return {
            "status": "processing", 
            "type": "document",
            "estimated_time": "1 minute",
            "request": request
        }
    
    def research(self, topic):
        """Research capability"""
        logger.info(f"🔍 Researching: {topic}")
        return {
            "status": "researching", 
            "topic": topic,
            "estimated_time": "2 minutes"
        }
    
    def analyze(self, target):
        """Analysis capability"""
        logger.info(f"📊 Analyzing: {target}")
        return {
            "status": "analyzing", 
            "target": target,
            "estimated_time": "1 minute"
        }
    
    def save_state(self):
        """Save brain state"""
        state_file = os.path.join(self.brain_dir, 'brain_state.json')
        state = {
            "capabilities": self.capabilities,
            "knowledge": self.knowledge_base,
            "last_active": datetime.now().isoformat(),
            "evolution_count": self.get_next_generation() - 1
        }
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info("💾 Brain state saved")

if __name__ == "__main__":
    brain = DMAIBrain()
    print("🧠 DMAI Brain initialized")
    print(f"Models: {brain.models}")
    print(f"Capabilities: {list(brain.capabilities.keys())}")
    
    # Test thinking
    response = brain.think("create a video about artificial intelligence")
    print(f"\nTest response: {response['response_text']}")
