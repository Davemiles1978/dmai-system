"""DMAI's voice - makes her speak with correct pronunciation"""

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyttsx3
import logging
import re

logger = logging.getLogger(__name__)

class DMAISpeaker:
    """Makes DMAI talk back to you - with female voice & correct pronunciation"""
    
    def __init__(self):
        self.engine = None
        self.initialize()
    
    def fix_pronunciation(self, text):
        """Fix how DMAI pronounces certain words"""
        
        # Replace "DMAI" with phonetic pronunciation
        # "Dee-Mai" - two syllables as you said
        text = re.sub(r'\bDMAI\b', 'Dee-Mai', text)
        text = re.sub(r'\bDmai\b', 'Dee-Mai', text)
        text = re.sub(r'\bdmai\b', 'Dee-Mai', text)
        
        # Add any other pronunciation fixes here
        # For example, if she mispronounces your name:
        # text = re.sub(r'\bDavid\b', 'Day-vid', text)
        
        return text
    
    def initialize(self):
        """Initialize text-to-speech engine with female voice"""
        try:
            self.engine = pyttsx3.init()
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            
            # Print available voices for debugging
            print("\n🎤 Available voices:")
            female_voices = []
            
            for i, voice in enumerate(voices):
                voice_name = voice.name
                is_female = any(name in voice_name.lower() for name in 
                               ['samantha', 'victoria', 'kate', 'female', 'girl', 'fiona', 'moira', 'tessa'])
                gender = "FEMALE" if is_female else "MALE"
                print(f"  {i}: {voice_name} ({gender})")
                if is_female:
                    female_voices.append(voice)
            
            # Try to find Samantha first (best female voice on Mac)
            samantha_voice = None
            for voice in voices:
                if 'samantha' in voice.name.lower():
                    samantha_voice = voice.id
                    print(f"\n✅ Selected Samantha (best female voice)")
                    self.engine.setProperty('voice', samantha_voice)
                    break
            
            # If no Samantha, try any female voice
            if not samantha_voice and female_voices:
                self.engine.setProperty('voice', female_voices[0].id)
                print(f"\n✅ Selected female voice: {female_voices[0].name}")
            
            # Adjust for natural female voice
            self.engine.setProperty('rate', 175)  # Conversational speed
            self.engine.setProperty('volume', 0.9)
            
            logger.info("✅ Speaker initialized with female voice")
            
            # Test pronunciation
            test_text = self.fix_pronunciation("Hello, I am DMAI")
            print(f"\n🔊 Testing pronunciation: '{test_text}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize speaker: {e}")
    
    def speak(self, text):
        """Speak the given text with corrected pronunciation"""
        if not self.engine:
            self.initialize()
        
        # Fix pronunciation before speaking
        fixed_text = self.fix_pronunciation(text)
        
        if self.engine:
            print(f"🤖 DMAI: {text}")  # Original for logging
            print(f"🔊 Speaking: {fixed_text}")  # What she actually says
            self.engine.say(fixed_text)
            self.engine.runAndWait()
        else:
            print(f"🤖 DMAI: {text} (text only)")
    
    def shutdown(self):
        """Clean up"""
        if self.engine:
            self.engine.stop()
