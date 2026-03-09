"""Fixed speaker module for DMAI with voice selection"""
import subprocess
import logging
import platform

logger = logging.getLogger(__name__)

class DMAISpeaker:
    """Speaker using 'say' command on macOS with configurable voice"""
    
    def __init__(self, voice='Samantha'):
        self.system = platform.system()
        self.voice = voice
        
        # Test if voice exists, fallback to default
        if self.system == 'Darwin':
            try:
                # Check if voice exists
                result = subprocess.run(['say', '-v', '?'], capture_output=True, text=True)
                if voice not in result.stdout:
                    # Try alternative female voices
                    alternatives = ['Victoria', 'Karen', 'Moira', 'Tessa', 'Veena']
                    for alt in alternatives:
                        if alt in result.stdout:
                            self.voice = alt
                            logger.info(f"Using alternative voice: {self.voice}")
                            break
            except:
                logger.warning("Could not check voices")
    
    def speak(self, text):
        """Speak text using system TTS with selected voice"""
        try:
            if self.system == 'Darwin':
                # Use macOS 'say' command with specific voice
                subprocess.run(['say', '-v', self.voice, text], check=False)
                logger.info(f"Speaking ({self.voice}): {text}")
            else:
                # Fallback to print
                print(f"[DMAI would say]: {text}")
        except Exception as e:
            logger.error(f"Speech error: {e}")
            print(f"[DMAI]: {text}")
    
    def shutdown(self):
        """Nothing to shutdown"""
        pass
