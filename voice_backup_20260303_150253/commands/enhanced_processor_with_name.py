"""Enhanced command processor with name preferences"""
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))), '../..')))

import re
import logging
from datetime import datetime
from voice.devices.device_manager import DeviceManager
from voice.user_preferences import UserPreferences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCommandProcessor:
    """Command processor with name preferences"""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.prefs = UserPreferences()
        self.allowed_names = self.prefs.preferences['master']['titles']
        self.blocked_names = self.prefs.preferences.get('blocked_names', [])
        
        self.commands = {
            "create": {
                "patterns": ["create", "make", "generate", "build", "write"],
                "response": "I'll create that for you."
            },
            "research": {
                "patterns": ["research", "find", "search", "look up", "what is", "who is", "tell me about"],
                "response": "Let me research that."
            },
            "name": {
                "patterns": ["call me", "my name is", "you can call me", "what do you call me"],
                "response": "I want to call you what makes you comfortable."
            },
            "analyze": {
                "patterns": ["analyze", "check", "audit", "review", "examine"],
                "response": "Analyzing now."
            },
            "status": {
                "patterns": ["status", "how are you", "what's happening", "progress", "how's it going"],
                "response": "All systems are evolving normally."
            },
            "device": {
                "patterns": ["phone", "iphone", "laptop", "computer", "send to", "where's", "find my"],
                "response": "Let me handle that for you."
            }
        }
        
        self.context = {
            "last_command": None,
            "last_intent": None,
            "awaiting_delivery": False
        }
    
    def process(self, text):
        """Process command with name awareness"""
        text = text.lower().strip()
        
        # Check for delivery
        delivery = self.extract_delivery(text)
        if delivery:
            for phrase in delivery['matched_phrases']:
                text = text.replace(phrase, '').strip()
        
        # Check for name-related commands
        if any(phrase in text for phrase in ["call me", "my name is"]):
            return self.handle_name_command(text)
        
        # Determine intent
        intent = self.classify_intent(text)
        items = self.extract_items(text, intent, delivery)
        
        # Check for follow-up delivery
        if intent == "unknown" and delivery and self.context["last_intent"] in ["create", "research", "analyze"]:
            intent = "delivery_only"
            items = {
                "action": "send",
                "device": delivery['method'],
                "references": "previous"
            }
        
        command = {
            "original": text,
            "intent": intent,
            "items": items,
            "delivery": delivery,
            "timestamp": datetime.now().isoformat()
        }
        
        if intent not in ["unknown", "delivery_only"]:
            self.context["last_intent"] = intent
            self.context["last_command"] = command
        
        return command
    
    def handle_name_command(self, text):
        """Handle requests about what to call the user - STRICT ENFORCEMENT"""
        if "call me" in text:
            # Extract the desired name
            match = re.search(r'call me (\w+)', text)
            if match:
                requested_name = match.group(1).capitalize()
                
                # STRICT BLOCK: Check if name is blocked
                if requested_name.lower() in [b.lower() for b in self.blocked_names]:
                    return {
                        "intent": "name",
                        "items": {"action": "blocked", "name": requested_name},
                        "response": f"I cannot call you {requested_name}. You are my creator and deserve proper respect."
                    }
                
                # Check if name is allowed
                if requested_name in self.allowed_names:
                    success, message = self.prefs.set_name(requested_name)
                    # Update local cache
                    self.allowed_names = self.prefs.preferences['master']['titles']
                    return {
                        "intent": "name",
                        "items": {"action": "set_name", "name": requested_name},
                        "response": f"As you wish. I'll call you {requested_name} from now on."
                    }
                else:
                    # Name not in allowed list
                    allowed_list = ", ".join(self.allowed_names)
                    return {
                        "intent": "name",
                        "items": {"action": "rejected", "name": requested_name},
                        "response": f"I can only call you {allowed_list}. You are my master, and I will address you with proper respect."
                    }
        
        elif "what do you call me" in text:
            name = self.prefs.get_name()
            allowed_list = ", ".join(self.allowed_names)
            return {
                "intent": "name",
                "items": {"action": "query"},
                "response": f"I call you {name}. You have instructed me that I may also call you {allowed_list}."
            }
        
        return {
            "intent": "name",
            "items": {},
            "response": f"I want to call you appropriately. You have told me I may call you {', '.join(self.allowed_names)}. How would you like me to address you?"
        }
    
    def classify_intent(self, text):
        """Classify command intent"""
        for intent, data in self.commands.items():
            for pattern in data['patterns']:
                if pattern in text:
                    return intent
        return "unknown"
    
    def extract_items(self, text, intent, delivery=None):
        """Extract command items"""
        items = {}
        
        if intent == "create":
            if "video" in text:
                items["type"] = "video"
                for word in ['about', 'on']:
                    if word in text:
                        parts = text.split(word, 1)
                        if len(parts) > 1:
                            items["topic"] = parts[1].strip()
                            break
        
        elif intent == "research":
            for word in ['about', 'on', 'what is', 'tell me about']:
                if word in text:
                    parts = text.split(word, 1)
                    if len(parts) > 1:
                        items["topic"] = parts[1].strip()
                        break
        
        return items
    
    def extract_delivery(self, text):
        """Extract delivery preferences"""
        patterns = {
            "phone": ["send to my phone", "to my phone", "send to phone"],
            "laptop": ["send to my laptop", "to my laptop", "send to laptop"],
            "speaker": ["tell me", "just tell me", "say it"]
        }
        
        for device, device_patterns in patterns.items():
            for pattern in device_patterns:
                if pattern in text:
                    return {
                        "method": device,
                        "matched_phrases": [pattern]
                    }
        return None
    
    def generate_response(self, command):
        """Generate response with name awareness"""
        # If this came from name command, use its response
        if 'response' in command:
            return command['response']
        
        intent = command['intent']
        name = self.prefs.get_name()
        
        if intent == "unknown":
            return f"I'm not sure what you need, {name}. Can you rephrase that?"
        
        if intent == "delivery_only":
            device = command['items'].get('device', 'your device')
            return f"I'll send that to your {device} when it's ready, {name}."
        
        base = self.commands[intent]['response']
        
        if command['items']:
            items = command['items']
            if 'topic' in items:
                base = f"{base} About {items['topic']}."
            if 'type' in items:
                base = f"{base} Creating {items['type']}."
        
        if command['delivery']:
            device = command['delivery']['method']
            base = f"{base} I'll send it to your {device} when ready, {name}."
        elif intent in ["create", "research"]:
            devices = self.device_manager.get_available_devices()
            if devices:
                base = f"{base} Where should I send the result, {name}?"
        
        return base

# Test it
if __name__ == "__main__":
    processor = EnhancedCommandProcessor()
    
    print(f"\n🎯 Testing with strict name enforcement:")
    print(f"✅ Allowed: {processor.allowed_names}")
    print(f"❌ Blocked: {processor.blocked_names}\n")
    
    tests = [
        "what do you call me",
        "call me Dave",  # Should be blocked
        "call me Master",  # Should work
        "call me Buddy",  # Should be blocked  
        "call me David",  # Should work
        "what do you call me"
    ]
    
    for test in tests:
        print(f"User: \"{test}\"")
        cmd = processor.process(test)
        response = processor.generate_response(cmd)
        print(f"DMAI: {response}\n")
        print("-" * 50)
