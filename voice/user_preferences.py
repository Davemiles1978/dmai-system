"""User preferences for DMAI - what to call you, etc"""
import json
import os
from datetime import datetime

class UserPreferences:
    """Store and manage user preferences"""
    
    def __init__(self, prefs_file='voice/user_data/preferences.json'):
        self.prefs_file = prefs_file
        os.makedirs(os.path.dirname(prefs_file), exist_ok=True)
        self.preferences = self.load_preferences()
    
    def load_preferences(self):
        """Load saved preferences"""
        if os.path.exists(self.prefs_file):
            with open(self.prefs_file, 'r') as f:
                return json.load(f)
        else:
            # Default preferences
            default = {
                "master": {
                    "name": "David",  # What DMAI calls you
                    "titles": ["master", "creator", "boss"],  # Allowed forms of address
                    "nicknames": [],  # Will be added over time
                    "voice_name": "David",  # For voice recognition
                    "preferred_greeting": "Hello",  # How DMAI greets you
                    "updated": datetime.now().isoformat()
                },
                "settings": {
                    "allow_nickname_evolution": True,  # Can she suggest new names?
                    "require_approval": True,  # Must ask before changing
                    "formality": "respectful",  # casual, respectful, formal
                    "default_device": "speaker"  # Where to default responses
                }
            }
            self.save_preferences(default)
            return default
    
    def save_preferences(self, data=None):
        """Save preferences to file"""
        if data is None:
            data = self.preferences
        with open(self.prefs_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_name(self):
        """Get what DMAI should call you"""
        return self.preferences['master']['name']
    
    def set_name(self, new_name, approved_by_master=True):
        """Set what DMAI calls you - only works with master approval"""
        if not approved_by_master:
            return False, "Only the master can change what I call them."
        
        old_name = self.preferences['master']['name']
        self.preferences['master']['name'] = new_name
        self.preferences['master']['updated'] = datetime.now().isoformat()
        self.save_preferences()
        return True, f"I'll call you {new_name} from now on."
    
    def suggest_nickname(self, suggested_name):
        """DMAI can suggest a nickname, but needs approval"""
        if not self.preferences['settings']['allow_nickname_evolution']:
            return False, "Nickname evolution is disabled."
        
        # Store the suggestion for later approval
        if 'suggested_names' not in self.preferences:
            self.preferences['suggested_names'] = []
        
        self.preferences['suggested_names'].append({
            "name": suggested_name,
            "suggested_at": datetime.now().isoformat(),
            "approved": False
        })
        self.save_preferences()
        
        return True, f"May I call you {suggested_name} sometimes? Please confirm with 'yes, call me that' or 'no, keep using {self.get_name()}'."
    
    def approve_nickname(self, nickname):
        """Master approves a suggested nickname"""
        found = False
        for suggestion in self.preferences.get('suggested_names', []):
            if suggestion['name'].lower() == nickname.lower() and not suggestion['approved']:
                suggestion['approved'] = True
                suggestion['approved_at'] = datetime.now().isoformat()
                
                # Add to nicknames list
                if nickname not in self.preferences['master']['nicknames']:
                    self.preferences['master']['nicknames'].append(nickname)
                
                found = True
                break
        
        if found:
            self.save_preferences()
            return True, f"Great! I'll call you {nickname} sometimes."
        return False, f"I don't have a suggestion for {nickname}."
    
    def get_greeting(self):
        """Get appropriate greeting"""
        name = self.get_name()
        greeting = self.preferences['master']['preferred_greeting']
        formality = self.preferences['settings']['formality']
        
        if formality == "formal":
            return f"{greeting}, Master {name}."
        elif formality == "casual":
            return f"Hey {name}!"
        else:  # respectful (default)
            return f"{greeting}, {name}."
    
    def set_formality(self, level):
        """Set formality level: casual, respectful, formal"""
        if level in ["casual", "respectful", "formal"]:
            self.preferences['settings']['formality'] = level
            self.save_preferences()
            return True, f"Formality set to {level}."
        return False, "Invalid formality level."

# Quick test
if __name__ == "__main__":
    prefs = UserPreferences()
    
    print(f"Current name: {prefs.get_name()}")
    print(f"Greeting: {prefs.get_greeting()}")
    
    # Test name change
    success, message = prefs.set_name("Dave")
    print(f"\nChanging name: {message}")
    print(f"New greeting: {prefs.get_greeting()}")
    
    # Test nickname suggestion
    success, message = prefs.suggest_nickname("Chief")
    print(f"\nSuggestion: {message}")
    
    # Test approval
    if success:
        success, message = prefs.approve_nickname("Chief")
        print(f"Approval: {message}")
    
    # Test formality
    prefs.set_formality("casual")
    print(f"\nCasual greeting: {prefs.get_greeting()}")
