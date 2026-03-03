"""Process voice commands and determine intent"""
import re
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommandProcessor:
    """Understands what you want DMAI to do"""
    
    def __init__(self):
        self.commands = {
            "create": {
                "patterns": ["create", "make", "generate", "build", "write"],
                "response": "I'll create that for you."
            },
            "research": {
                "patterns": ["research", "find", "search", "look up", "what is", "who is"],
                "response": "Let me research that."
            },
            "analyze": {
                "patterns": ["analyze", "check", "audit", "review", "examine"],
                "response": "Analyzing now."
            },
            "status": {
                "patterns": ["status", "how are you", "what's happening", "progress"],
                "response": "All systems are evolving normally."
            },
            "protect": {
                "patterns": ["protect", "secure", "defend", "monitor"],
                "response": "I'll enhance security."
            },
            "automate": {
                "patterns": ["automate", "schedule", "set up", "workflow"],
                "response": "Setting up automation."
            },
            "help": {
                "patterns": ["help", "what can you do", "capabilities"],
                "response": "I can create, research, analyze, protect, and automate. What do you need?"
            }
        }
        
        # Track conversation context
        self.context = {
            "last_command": None,
            "awaiting_details": False,
            "pending_action": None
        }
    
    def process(self, text):
        """Convert spoken text to actionable command"""
        text = text.lower().strip()
        
        # Check for delivery preferences
        delivery = self.extract_delivery(text)
        if delivery:
            text = text.replace(delivery['phrase'], '').strip()
        
        # Determine intent
        intent = self.classify_intent(text)
        
        # Extract any specific items (like filenames, topics)
        items = self.extract_items(text, intent)
        
        command = {
            "original": text,
            "intent": intent,
            "items": items,
            "delivery": delivery,
            "timestamp": datetime.now().isoformat(),
            "needs_clarification": False
        }
        
        # Store context
        self.context['last_command'] = command
        
        logger.info(f"Processed command: {intent} - {items}")
        return command
    
    def classify_intent(self, text):
        """Determine what action the user wants"""
        for intent, data in self.commands.items():
            for pattern in data['patterns']:
                if pattern in text:
                    return intent
        return "unknown"
    
    def extract_items(self, text, intent):
        """Extract what the command applies to"""
        items = {}
        
        # Remove common words
        words = text.split()
        stop_words = ['the', 'a', 'an', 'for', 'me', 'please', 'can', 'you']
        filtered = [w for w in words if w not in stop_words]
        
        if intent == "create":
            # Look for what to create
            if "video" in text:
                items["type"] = "video"
                # Extract topic after "about" or "on"
                about_match = re.search(r'about\s+(.+?)(?:\s+and|\s*$)', text)
                if about_match:
                    items["topic"] = about_match.group(1)
            
            elif "script" in text or "code" in text or "program" in text:
                items["type"] = "code"
                
            elif "email" in text or "message" in text:
                items["type"] = "message"
        
        elif intent == "research":
            # Extract research topic
            for word in ['about', 'on', 'what is', 'who is']:
                if word in text:
                    parts = text.split(word, 1)
                    if len(parts) > 1:
                        items["topic"] = parts[1].strip()
                        break
        
        return items
    
    def extract_delivery(self, text):
        """Check if user specified where to send results"""
        delivery_patterns = {
            "phone": ["send to my phone", "on my phone"],
            "laptop": ["send to my laptop", "on my laptop", "to my computer"],
            "email": ["email me", "send to my email"],
            "speaker": ["tell me", "just tell me", "say it"]
        }
        
        for device, patterns in delivery_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    return {
                        "method": device,
                        "phrase": pattern
                    }
        return None
    
    def generate_response(self, command):
        """Create appropriate spoken response"""
        intent = command['intent']
        
        if intent == "unknown":
            return "I'm not sure what you need. Can you rephrase that?"
        
        base_response = self.commands[intent]['response']
        
        # Add specifics if available
        if command['items']:
            items = command['items']
            if 'topic' in items:
                base_response = f"{base_response} About {items['topic']}."
            if 'type' in items:
                base_response = f"{base_response} Creating {items['type']}."
        
        # Add delivery confirmation
        if command['delivery']:
            device = command['delivery']['method']
            base_response = f"{base_response} I'll send it to your {device} when ready."
        else:
            # If no delivery specified, ask
            base_response = f"{base_response} Where should I send the result?"
            self.context['awaiting_details'] = True
            self.context['pending_action'] = intent
        
        return base_response
    
    def needs_more_info(self, command):
        """Check if we need clarification"""
        if command['intent'] == "create" and not command['items']:
            return True, "What would you like me to create?"
        
        if command['intent'] == "research" and not command['items'].get('topic'):
            return True, "What topic should I research?"
        
        return False, None

# Example usage
if __name__ == "__main__":
    processor = CommandProcessor()
    
    test_commands = [
        "Hey DMAI, create a video about black holes",
        "send to my phone",
        "what's my server status",
        "research quantum computing",
        "protect my network"
    ]
    
    print("Testing command processor:\n")
    for cmd in test_commands:
        print(f"User: {cmd}")
        processed = processor.process(cmd)
        response = processor.generate_response(processed)
        print(f"DMAI: {response}")
        needs_more, question = processor.needs_more_info(processed)
        if needs_more:
            print(f"DMAI: {question}")
        print("-" * 40)
