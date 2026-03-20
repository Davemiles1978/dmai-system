"""Wake word detection for DMAI"""

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pvporcupine
import pvrecorder
import numpy as np
import struct
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WakeWordDetector:
    """Detects 'Hey DMAI' wake word"""
    
    def __init__(self, wake_word="hey dmai", sensitivity=0.7):
        self.wake_word = wake_word
        self.sensitivity = sensitivity
        self.porcupine = None
        self.recorder = None
        self.running = False
        self.callback = None
        
        # Get Picovoice access key from environment or prompt
        self.access_key = os.environ.get('PICOVOICE_ACCESS_KEY')
        if not self.access_key:
            logger.warning("PICOVOICE_ACCESS_KEY not set. Using built-in keywords.")
            # Will use built-in "computer" as fallback
    
    def initialize(self):
        """Initialize Porcupine wake word engine"""
        try:
            if self.access_key:
                # Use custom wake word with access key
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keywords=[self.wake_word],
                    sensitivities=[self.sensitivity]
                )
            else:
                # Use built-in keyword as fallback
                logger.info("Using built-in 'computer' wake word")
                self.porcupine = pvporcupine.create(
                    keywords=["computer"],
                    sensitivities=[self.sensitivity]
                )
            
            # Initialize recorder
            self.recorder = pvrecorder.PvRecorder(
                frame_length=self.porcupine.frame_length,
                device_index=-1  # default microphone
            )
            
            logger.info(f"Wake word detector initialized with frame length {self.porcupine.frame_length}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize wake word detector: {e}")
            return False
    
    def start(self, callback=None):
        """Start listening for wake word"""
        if not self.porcupine:
            if not self.initialize():
                return False
        
        self.callback = callback
        self.running = True
        self.recorder.start()
        
        logger.info(f"Listening for '{self.wake_word}'...")
        
        try:
            while self.running:
                pcm = self.recorder.read()
                
                # Process for wake word
                result = self.porcupine.process(pcm)
                
                if result >= 0:
                    logger.info(f"Wake word detected! (keyword index: {result})")
                    if self.callback:
                        self.callback()
                    
                    # Small pause to prevent multiple triggers
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            self.stop()
    
    def stop(self):
        """Stop listening"""
        self.running = False
        if self.recorder:
            self.recorder.stop()
        logger.info("Wake word detector stopped")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        if self.porcupine:
            self.porcupine.delete()
        if self.recorder:
            self.recorder.delete()
        logger.info("Wake word detector cleaned up")

def on_wake():
    """Callback when wake word detected"""
    print("\n🔊 DMAI: Yes, I'm listening...")
    # Here we'll later add command listening
