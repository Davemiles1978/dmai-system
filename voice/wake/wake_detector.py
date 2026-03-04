"""Wake word detection for DMAI"""
import pvporcupine
import pvrecorder
import numpy as np
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WakeWordDetector:
    """Detects wake word - currently Jarvis, but DMAI will evolve this"""
    
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
        """Initialize Porcupine with Jarvis wake word"""
        try:
            # Use the Jarvis keyword file
            keyword_path = os.path.join(
                os.path.dirname(__file__), 
                'keywords', 
                'jarvis_en_mac_v4_0_0.ppn'
            )
            
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[keyword_path],
                sensitivities=[self.sensitivity]
            )
            
            # Initialize recorder
            self.recorder = pvrecorder.PvRecorder(
                frame_length=self.porcupine.frame_length,
                device_index=-1
            )
            
            logger.info("Wake word detector initialized with Jarvis")
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
        
        logger.info("Listening for 'Jarvis'...")
        
        try:
            while self.running:
                pcm = self.recorder.read()
                result = self.porcupine.process(pcm)
                
                if result >= 0:
                    logger.info("Wake word detected!")
                    if self.callback:
                        self.callback()
                    time.sleep(1)
                    
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
