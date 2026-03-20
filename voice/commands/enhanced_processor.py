"""Enhanced command processor with device awareness"""
import sys
from pathlib import Path
import os
# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import json
import logging
from datetime import datetime
from voice.devices.device_manager import DeviceManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCommandProcessor:
    """Command processor that knows about your devices"""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.commands = {
            "create": {
                "patterns": ["create", "make", "generate", "build", "write"],
                "response": "I'll create that for you."
            },
            "research": {
                "patterns": ["research", "find", "search", "look up", "what is", "who is", "tell me about"],
                "response": "Let me research that."
            },
            "analyze": {
                "patterns": ["analyze", "check", "audit", "review", "examine"],
                "response": "Analyzing now."
            },
            "status": {
                "patterns": ["status", "how are you", "what's happening", "progress", "how's it going"],
                "response": "All systems are evolving normally."
            },
            "protect": {
                "patterns": ["protect", "secure", "defend", "monitor", "watch"],
                "response": "I'll enhance security."
            },
            "automate": {
                "patterns": ["automate", "schedule", "set up", "workflow", "create a task"],
                "response": "Setting up automation."
            },
            "device": {
                "patterns": [
                    "phone", "iphone", "laptop", "computer",  # Device mentions
                    "send to", "send it", "where's", "find", "locate"  # Device actions
                ],
                "response": "Let me handle that for you."
            },
            "help": {
                "patterns": ["help", "what can you do", "capabilities", "what do you do"],
                "response": "I can create, research, analyze, protect, automate, and manage your devices. What do you need?"
            }
        }
        
        # Track conversation context
        self.context = {
            "last_command": None,
            "last_intent": None,
            "last_items": {},
            "awaiting_delivery": False,
            "pending_result": None
        }
    
    def process(self, text):
        """Convert spoken text to actionable command"""
        text = text.lower().strip()
        
        # Check for delivery preferences
        delivery = self.extract_delivery(text)
        if delivery:
            # Remove the delivery phrase for intent detection
            for phrase in delivery['matched_phrases']:
                text = text.replace(phrase, '').strip()
        
        # Determine intent
        intent = self.classify_intent(text, delivery)
        
        # Extract any specific items
        items = self.extract_items(text, intent, delivery)
        
        # Check if this is a follow-up command (like "send it to my phone")
        if intent == "unknown" and delivery and self.context["last_intent"] in ["create", "research", "analyze"]:
            # This is a delivery instruction for the previous command
            intent = "delivery_only"
            items = {
                "action": "send",
                "device": delivery['method'],
                "references": "previous"
            }
            logger.info(f"Interpreted as delivery for previous {self.context['last_intent']} command")
        
        command = {
            "original": text,
            "intent": intent,
            "items": items,
            "delivery": delivery,
            "timestamp": datetime.now().isoformat(),
            "needs_clarification": False
        }
        
        # Store context for future commands
        if intent not in ["unknown", "delivery_only"]:
            self.context["last_intent"] = intent
            self.context["last_items"] = items
            self.context["last_command"] = command
        
        logger.info(f"Processed command: {intent} - {items}")
        return command
    
    def classify_intent(self, text, delivery=None):
        """Determine what action the user wants"""
        # First check for device-related commands
        device_keywords = ['phone', 'iphone', 'laptop', 'computer', 'where is', 'find my']
        if any(keyword in text for keyword in device_keywords):
            return "device"
        
        # Then check other intents
        for intent, data in self.commands.items():
            if intent == "device":  # Already checked
                continue
            for pattern in data['patterns']:
                if pattern in text:
                    return intent
        return "unknown"
    
    def extract_items(self, text, intent, delivery=None):
        """Extract what the command applies to"""
        items = {}
        
        if intent == "create":
            if "video" in text:
                items["type"] = "video"
                # Look for topic after "about" or "on"
                about_match = re.search(r'about\s+(.+?)(?:\s+and|\s*$)', text)
                if about_match:
                    items["topic"] = about_match.group(1)
                else:
                    on_match = re.search(r'on\s+(.+?)(?:\s+and|\s*$)', text)
                    if on_match:
                        items["topic"] = on_match.group(1)
            elif "script" in text or "code" in text:
                items["type"] = "code"
            elif "email" in text:
                items["type"] = "message"
            elif "file" in text:
                items["type"] = "file"
        
        elif intent == "research":
            # Look for topic after common phrases
            for word in ['about', 'on', 'what is', 'who is', 'tell me about']:
                if word in text:
                    parts = text.split(word, 1)
                    if len(parts) > 1:
                        topic = parts[1].strip()
                        # Remove trailing words
                        for stop in ['and', 'please', 'for me']:
                            if stop in topic:
                                topic = topic.split(stop)[0].strip()
                        items["topic"] = topic
                        break
        
        elif intent == "device":
            # Determine which device
            if "phone" in text or "iphone" in text:
                items["device"] = "phone"
            elif "laptop" in text or "computer" in text:
                items["device"] = "laptop"
            
            # Determine action
            if "where" in text or "find" in text or "locate" in text:
                items["action"] = "locate"
            elif "send" in text:
                items["action"] = "send"
            
            # If delivery was extracted, use that device
            if delivery and not items.get('device'):
                items["device"] = delivery['method']
                items["action"] = "send"
        
        elif intent == "delivery_only":
            items = {
                "action": "send",
                "device": delivery['method'] if delivery else "unknown",
                "references": "previous"
            }
        
        return items
    
    def extract_delivery(self, text):
        """Check if user specified where to send results"""
        delivery_patterns = {
            "phone": [
                "send to my phone", "on my phone", "to my phone",
                "send to iphone", "to my iphone", "send it to my phone",
                "send to phone", "to phone", "send it to phone"
            ],
            "laptop": [
                "send to my laptop", "on my laptop", "to my laptop",
                "send to my computer", "to my computer", "send it to my laptop",
                "send to laptop", "to laptop", "send it to laptop"
            ],
            "email": [
                "email me", "send to my email", "to my email",
                "send it via email"
            ],
            "speaker": [
                "tell me", "just tell me", "say it", "speak it",
                "just say it", "tell me now"
            ]
        }
        
        for device, patterns in delivery_patterns.items():
            matched = []
            for pattern in patterns:
                if pattern in text:
                    matched.append(pattern)
            if matched:
                return {
                    "method": device,
                    "matched_phrases": matched
                }
        return None
    
    def generate_response(self, command):
        """Create appropriate spoken response with device awareness"""
        intent = command['intent']
        items = command['items']
        delivery = command['delivery']
        
        if intent == "unknown":
            return "I'm not sure what you need. Can you rephrase that?"
        
        # Handle delivery-only commands (like "send it to my phone")
        if intent == "delivery_only":
            device = items.get('device', 'your device')
            return f"I'll send that to your {device} when it's ready. Is there anything else?"
        
        # Handle device commands first
        if intent == "device":
            if items.get('action') == 'locate':
                device = items.get('device', 'device')
                return f"I'll help you locate your {device}. Check your {device} for a notification."
            elif items.get('action') == 'send':
                device = items.get('device', 'your device')
                return f"I'll send that to your {device} when it's ready."
            else:
                return "I can help you with your devices. What would you like me to do?"
        
        # Handle other intents
        base_response = self.commands[intent]['response']
        
        # Add specifics if available
        if items:
            if 'topic' in items:
                base_response = f"{base_response} About {items['topic']}."
            if 'type' in items:
                base_response = f"{base_response} Creating {items['type']}."
        
        # Add delivery confirmation
        if delivery:
            device = delivery['method']
            if device == "speaker":
                base_response = f"{base_response} I'll tell you when ready."
            else:
                base_response = f"{base_response} I'll send it to your {device} when ready."
        else:
            # If no delivery specified and it's a creation/research command, ask
            if intent in ["create", "research", "analyze"]:
                devices = self.device_manager.get_available_devices()
                if devices:
                    device_names = [d['name'].replace('.local', '') for d in devices[:2]]
                    base_response = f"{base_response} Where should I send the result? "
                    base_response += f"Your {', '.join(device_names)} are available."
                    self.context['awaiting_delivery'] = True
                    self.context['pending_result'] = intent
        
        return base_response
    
    def needs_more_info(self, command):
        """Check if we need clarification"""
        if command['intent'] == "create" and not command['items']:
            return True, "What would you like me to create?"
        
        if command['intent'] == "create" and command['items'].get('type') == 'video' and not command['items'].get('topic'):
            return True, "What topic should the video be about?"
        
        if command['intent'] == "research" and not command['items'].get('topic'):
            return True, "What topic should I research?"
        
        if command['intent'] == "device" and not command['items'].get('device'):
            return True, "Which device are you looking for? Your iPhone or laptop?"
        
        return False, None

# Test the enhanced processor
if __name__ == "__main__":
    processor = EnhancedCommandProcessor()
    
    # Register current device
    processor.device_manager.register_device()
    
    print("\n🎯 Testing context-aware command processor:\n")
    
    # Simulate a conversation
    conversation = [
        "create a video about DMAI",
        "send it to my phone",
        "find my iPhone",
        "research quantum computing",
        "send to laptop"
    ]
    
    for cmd in conversation:
        print(f"User: \"{cmd}\"")
        processed = processor.process(cmd)
        response = processor.generate_response(processed)
        print(f"DMAI: {response}\n")
        needs_more, question = processor.needs_more_info(processed)
        if needs_more:
            print(f"DMAI: {question}")
        print("-" * 50)
