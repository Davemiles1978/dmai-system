import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))))))

import logging
from ai_core.core_brain import DMAIBrain
from voice.commands.enhanced_processor import EnhancedCommandProcessor

logger = logging.getLogger(__name__)

class DMAIIntegrator:
    """Connects voice interface to actual AI brain"""
    
    def __init__(self):
        self.brain = DMAIBrain()
        self.processor = EnhancedCommandProcessor()
        
    def process_with_brain(self, voice_command):
        """Take voice command and actually do something"""
        
        # Get the processed command
        command = self.processor.process(voice_command)
        intent = command['intent']
        items = command['items']
        
        # Use actual brain to handle it
        if intent == "create":
            result = self.brain.create(voice_command)
            return f"Creating {items.get('type', 'content')} about {items.get('topic', 'your request')}"
            
        elif intent == "research":
            topic = items.get('topic', voice_command)
            result = self.brain.research(topic)
            return f"Researching {topic}"
            
        elif intent == "analyze":
            result = self.brain.analyze(voice_command)
            return f"Analyzing your request"
            
        elif intent == "status":
            return f"Generation {self.brain.get_next_generation() - 1}. All systems evolving."
            
        else:
            # Let brain think about it
            thought = self.brain.think(voice_command)
            return thought.get('response_text', "Processing your request")
    
    def evolve_if_needed(self):
        """Check if DMAI should evolve"""
        if self.brain.should_evolve():
            return self.brain.evolve()
        return None

if __name__ == "__main__":
    integrator = DMAIIntegrator()
    print("🧠 DMAI Brain integrated with Voice")
    test = integrator.process_with_brain("create a video about quantum computing")
    print(f"Response: {test}")
