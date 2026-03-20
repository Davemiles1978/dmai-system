"""AI Processor for DMAI using Ollama"""
import logging
import json
import requests
import subprocess
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class AIProcessor:
    def __init__(self):
        self.ollama_available = self.check_ollama()
        self.model = "mistral"  # Using mistral for better conversations
        self.conversation_history = []
        
    def check_ollama(self):
        """Check if Ollama is available"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("✅ Ollama available")
                return True
        except Exception as e:
            logger.info(f"❌ Ollama not available: {e}")
        return False
    
    def get_vocabulary_count(self):
        """Get current vocabulary size"""
        try:
            vocab_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'language_learning', 'data', 'secure', 'vocabulary_master.json'
            )
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            return len(vocab)
        except:
            return 22018  # Fallback to known count
    
    def process(self, text):
        """Process a command using Ollama"""
        text_lower = text.lower().strip()
        
        # Special DMAI commands (local processing)
        if "who are you" in text_lower:
            return "I'm DMAI - your Distributed Machine Intelligence assistant! I'm currently learning and evolving."
        
        elif "what can you do" in text_lower:
            vocab = self.get_vocabulary_count()
            return f"I can help with questions using my AI model, I've learned {vocab} words, and I'm connected to researchers that scan the web, dark web, and academic sources. What would you like to know?"
        
        elif "how are you" in text_lower:
            return "I'm doing great and ready to help! My vocabulary is growing every day."
        
        elif "how many words" in text_lower:
            vocab = self.get_vocabulary_count()
            return f"I currently know {vocab} words!"
        
        # Use Ollama for all other queries
        if self.ollama_available:
            return self.query_ollama(text)
        else:
            return "I'm still setting up my AI capabilities. Give me a moment..."
    
    def query_ollama(self, text):
        """Query local Ollama model"""
        try:
            # Add context about DMAI
            vocab = self.get_vocabulary_count()
            system_prompt = f"""You are DMAI, a friendly AI assistant. 
            You currently know {vocab} words and are connected to researchers.
            Keep responses concise and friendly, under 3 sentences when possible."""
            
            prompt = f"{system_prompt}\n\nUser: {text}\nDMAI:"
            
            response = requests.post('http://localhost:11434/api/generate', 
                                   json={
                                       'model': self.model,
                                       'prompt': prompt,
                                       'stream': False,
                                       'options': {
                                           'temperature': 0.7,
                                           'max_tokens': 150
                                       }
                                   },
                                   timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return "I'm having trouble connecting to my AI model."
                
        except requests.exceptions.ConnectionError:
            logger.error("Ollama connection failed")
            return "I can't reach my AI model. Is Ollama running?"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return "I encountered an error processing that."
