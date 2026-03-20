"""DMAI Personality Settings - How she talks to you"""

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
from datetime import datetime

class Personality:
    """Define DMAI's communication style"""
    
    def __init__(self, prefs_file='voice/user_data/personality.json'):
        self.prefs_file = prefs_file
        os.makedirs(os.path.dirname(prefs_file), exist_ok=True)
        self.personality = self.load_personality()
    
    def load_personality(self):
        """Load personality settings"""
        if os.path.exists(self.prefs_file):
            with open(self.prefs_file, 'r') as f:
                return json.load(f)
        else:
            # Default personality - respectful but warm
            default = {
                "tone": {
                    "base": "respectful",  # respectful, formal, warm, efficient
                    "formality": 0.7,  # 0-1 scale
                    "warmth": 0.6,  # 0-1 scale
                    "humor": 0.2  # 0-1 scale
                },
                "greetings": {
                    "morning": ["Good morning", "Morning"],
                    "afternoon": ["Good afternoon"],
                    "evening": ["Good evening"],
                    "night": ["Good evening"]
                },
                "responses": {
                    "acknowledge": ["Yes", "Of course", "Certainly", "Right away"],
                    "thinking": ["Let me think about that", "Processing", "Working on it"],
                    "complete": ["Done", "Complete", "Finished", "Ready"],
                    "error": ["I apologize", "I'm having trouble", "Let me try again"]
                },
                "signature": {
                    "use_name": True,
                    "sign_off": False,
                    "emojis": False
                },
                "updated": datetime.now().isoformat()
            }
            self.save_personality(default)
            return default
    
    def save_personality(self, data=None):
        """Save personality settings"""
        if data is None:
            data = self.personality
        with open(self.prefs_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_greeting(self):
        """Get time-appropriate greeting"""
        import time
        hour = time.localtime().tm_hour
        
        if hour < 12:
            return self.personality['greetings']['morning'][0]
        elif hour < 17:
            return self.personality['greetings']['afternoon'][0]
        else:
            return self.personality['greetings']['evening'][0]
    
    def get_response(self, response_type):
        """Get a response of the specified type"""
        import random
        responses = self.personality['responses'].get(response_type, ["Okay"])
        return random.choice(responses)
    
    def set_tone(self, base, formality=None, warmth=None, humor=None):
        """Adjust DMAI's tone"""
        if base in ["respectful", "formal", "warm", "efficient"]:
            self.personality['tone']['base'] = base
            
            # Set presets
            if base == "formal":
                self.personality['tone']['formality'] = 0.9
                self.personality['tone']['warmth'] = 0.3
                self.personality['tone']['humor'] = 0.0
            elif base == "respectful":
                self.personality['tone']['formality'] = 0.7
                self.personality['tone']['warmth'] = 0.5
                self.personality['tone']['humor'] = 0.1
            elif base == "warm":
                self.personality['tone']['formality'] = 0.4
                self.personality['tone']['warmth'] = 0.8
                self.personality['tone']['humor'] = 0.3
            elif base == "efficient":
                self.personality['tone']['formality'] = 0.5
                self.personality['tone']['warmth'] = 0.2
                self.personality['tone']['humor'] = 0.0
        
        # Override individual settings if provided
        if formality is not None:
            self.personality['tone']['formality'] = max(0, min(1, formality))
        if warmth is not None:
            self.personality['tone']['warmth'] = max(0, min(1, warmth))
        if humor is not None:
            self.personality['tone']['humor'] = max(0, min(1, humor))
        
        self.save_personality()
        return self.personality['tone']

# Quick test
if __name__ == "__main__":
    p = Personality()
    print(f"Greeting: {p.get_greeting()}")
    print(f"Acknowledge: {p.get_response('acknowledge')}")
    print(f"Complete: {p.get_response('complete')}")
    
    print("\nTone presets:")
    for tone in ["formal", "respectful", "warm", "efficient"]:
        p.set_tone(tone)
        print(f"  {tone}: formality={p.personality['tone']['formality']}, warmth={p.personality['tone']['warmth']}")
