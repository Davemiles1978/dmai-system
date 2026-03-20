"""Expanded command types for DMAI"""
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))), '../..')))

class CommandExpander:
    """Add more command categories"""
    
    def __init__(self):
        self.new_commands = {
            "email": {
                "patterns": ["email", "send email", "check email", "compose email"],
                "response": "I'll handle your email.",
                "requires": ["email_account"]
            },
            "calendar": {
                "patterns": ["schedule", "calendar", "appointment", "meeting", "remind me"],
                "response": "Let me check your calendar.",
                "requires": ["calendar_access"]
            },
            "weather": {
                "patterns": ["weather", "temperature", "forecast", "rain"],
                "response": "Checking the weather for you.",
                "requires": ["location"]
            },
            "news": {
                "patterns": ["news", "headlines", "what's happening", "latest"],
                "response": "Fetching the latest news.",
                "requires": ["news_api"]
            },
            "memory": {
                "patterns": ["remember", "remind", "don't forget", "note"],
                "response": "I'll remember that for you.",
                "requires": []
            },
            "math": {
                "patterns": ["calculate", "math", "equation", "solve"],
                "response": "Let me calculate that.",
                "requires": []
            },
            "translate": {
                "patterns": ["translate", "in spanish", "in french", "in german"],
                "response": "I'll translate that for you.",
                "requires": ["translation_api"]
            },
            "timer": {
                "patterns": ["timer", "countdown", "remind me in", "alert me in"],
                "response": "Setting a timer.",
                "requires": []
            }
        }
    
    def integrate_with_processor(self, processor):
        """Add these commands to the existing processor"""
        for cmd_name, cmd_data in self.new_commands.items():
            processor.commands[cmd_name] = cmd_data
        return processor

# Quick test
if __name__ == "__main__":
    expander = CommandExpander()
    print("📋 New commands available:")
    for cmd, data in expander.new_commands.items():
        print(f"  • {cmd}: {data['patterns'][0]}")
