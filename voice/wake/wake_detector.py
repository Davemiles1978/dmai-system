"""Wake word detection for DMAI - Now with 'Hey Dee Mai'"""

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pvporcupine
import pvrecorder
import numpy as np
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WakeWordDetector:
    """Detects wake word - 'Hey Dee Mai'"""
    
    def __init__(self, sensitivity=0.7):
        self.sensitivity = sensitivity
        self.porcupine = None
        self.recorder = None
        self.running = False
        self.callback = None
        
        # Get Picovoice access key
        self.access_key = os.environ.get('PICOVOICE_ACCESS_KEY')
        if not self.access_key:
            logger.error("PICOVOICE_ACCESS_KEY not set")
    
    def initialize(self):
        """Initialize Porcupine with Hey Dee Mai wake word"""
        try:
            # Path to your custom wake word file
            keyword_path = os.path.join(
                os.path.dirname(__file__), 
                'keywords', 
                'Hey-Dee-Mai_en_mac_v4_0_0.ppn'
            )
            
            # Check if file exists
            if not os.path.exists(keyword_path):
                logger.error(f"Keyword file not found at: {keyword_path}")
                logger.info("Available keywords:")
                keywords_dir = os.path.join(os.path.dirname(__file__), 'keywords')
                if os.path.exists(keywords_dir):
                    for f in os.listdir(keywords_dir):
                        if f.endswith('.ppn'):
                            logger.info(f"  - {f}")
                return False
            
            logger.info(f"Using keyword file: {os.path.basename(keyword_path)}")
            
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[keyword_path],
                sensitivities=[self.sensitivity]
            )
            
            # Initialize recorder
            self.recorder = pvrecorder.PvRecorder(
                frame_length=self.porcupine.frame_length,
                device_index=-1  # Default microphone
            )
            
            logger.info("✅ Wake word detector initialized with 'Hey Dee Mai'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    def start(self, callback=None):
        """Start listening for wake word"""
        if not self.porcupine:
            if not self.initialize():
                return False
        
        self.callback = callback
        self.running = True
        self.recorder.start()
        
        logger.info("🎤 Listening for 'Hey Dee Mai'...")
        print("\n🎤 DMAI is listening for 'Hey Dee Mai'...")
        
        try:
            while self.running:
                pcm = self.recorder.read()
                result = self.porcupine.process(pcm)
                
                if result >= 0:
                    logger.info("✅ 'Hey Dee Mai' detected!")
                    print("\n✅ Wake word detected!")
                    if self.callback:
                        self.callback()
                    time.sleep(1.5)  # Prevent multiple triggers
                    
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.error(f"Error: {e}")
            self.stop()
    
    def stop(self):
        self.running = False
        if self.recorder:
            self.recorder.stop()
    
    def cleanup(self):
        self.stop()
        if self.porcupine:
            self.porcupine.delete()
        if self.recorder:
            self.recorder.delete()
        logger.info("Wake word detector cleaned up")
