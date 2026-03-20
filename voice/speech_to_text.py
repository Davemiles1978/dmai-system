"""Convert speech to text using Whisper"""

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import whisper
import numpy as np
import tempfile
import soundfile as sf
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToText:
    """Convert spoken words to text"""
    
    def __init__(self, model_size="base"):
        """
        model_size: "tiny", "base", "small", "medium", "large"
        - tiny: fastest, least accurate (39M params)
        - base: good balance (74M params)
        - small: better accuracy (244M params)
        - medium: even better (769M params)
        - large: most accurate, slowest (1550M params)
        """
        self.model_size = model_size
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load Whisper model (downloaded once, cached forever)"""
        logger.info(f"Loading Whisper {self.model_size} model...")
        try:
            self.model = whisper.load_model(self.model_size)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def transcribe(self, audio_data, sample_rate=16000):
        """Convert audio to text"""
        if self.model is None:
            logger.error("Model not loaded")
            return ""
        
        try:
            # Save audio temporarily
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as f:
                sf.write(f.name, audio_data, sample_rate)
                
                # Transcribe
                result = self.model.transcribe(
                    f.name,
                    language='en',
                    task='transcribe',
                    fp16=False  # Use FP32 for CPU
                )
                
                text = result['text'].strip()
                logger.info(f"Transcribed: '{text}'")
                
                return text
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
    
    def transcribe_file(self, audio_path):
        """Transcribe an audio file"""
        if self.model is None:
            logger.error("Model not loaded")
            return ""
        
        try:
            result = self.model.transcribe(audio_path, language='en')
            return result['text'].strip()
        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return ""
    
    def get_available_models(self):
        """List available model sizes"""
        return ["tiny", "base", "small", "medium", "large"]

# Quick test
if __name__ == "__main__":
    stt = SpeechToText(model_size="base")
    print("Speech-to-text module ready!")
    print(f"Available models: {stt.get_available_models()}")
