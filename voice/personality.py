"""DMAI's friendly personality"""
import pyttsx3
import logging

logger = logging.getLogger(__name__)

class DMAIPersonality:
    def __init__(self):
        self.engine = pyttsx3.init()
        # Set a friendly voice
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)  # Usually female voice
        self.engine.setProperty('rate', 180)  # Slightly faster, friendly pace
        self.engine.setProperty('volume', 0.9)
        
        self.greetings = [
            "Hey! I'm here!",
            "Hi there! Ready to learn something new?",
            "Hello! I was just reading a book. What's up?",
            "Oh, hey! I'm glad you're here!",
            "Hi! I'm DMAI. Want to see what I've learned?"
        ]
        
        self.responses = {
            'wake': [
                "Yeah?",
                "I'm listening!",
                "What's going on?",
                "Hey! What's up?",
                "Oh! Hi there!"
            ],
            'unknown_voice': [
                "Hi! I don't recognize your voice yet, but that's okay! Want to become friends? Just say 'enroll me'",
                "Oh hi! You sound new here. Say 'enroll me' and we can get to know each other!",
                "Hello there! I don't know your voice yet. Would you like me to learn it?"
            ]
        }
    
    def greet(self):
        """Friendly greeting on startup"""
        import random
        greeting = random.choice(self.greetings)
        logger.info(f"Greeting: {greeting}")
        self.engine.say(greeting)
        self.engine.runAndWait()
    
    def respond_to_wake(self, is_known=False):
        """Respond to wake word based on voice recognition"""
        import random
        if is_known:
            response = random.choice(self.responses['wake'])
        else:
            response = random.choice(self.responses['unknown_voice'])
        logger.info(f"Wake response: {response}")
        self.engine.say(response)
        self.engine.runAndWait()
    
    def say(self, text):
        """Say anything"""
        self.engine.say(text)
        self.engine.runAndWait()
