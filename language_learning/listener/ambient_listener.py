"""Ambient listener with proper resource cleanup (using sounddevice)"""
import threading
import time
import logging
import sounddevice as sd
import numpy as np
import os
import tempfile
import wave
import atexit
from datetime import datetime
import json
import subprocess

# Import speech-to-text for language learning
from voice.speech_to_text import SpeechToText
from language_learning.processor.language_learner import LanguageLearner

logger = logging.getLogger(__name__)

class AmbientListener:
    """Listens to environment to learn language and music naturally - with proper cleanup"""
    
    def __init__(self, listen_duration=8, pause_between=15):
        self.listen_duration = listen_duration
        self.pause_between = pause_between
        self.sample_rate = 16000
        self.is_running = False
        self.thread = None
        self.process_audio_callback = None  # Will be set by parent
        self.temp_files = []  # Track temp files for cleanup
        self._cleanup_registered = False
        
        # Learning systems
        self.stt = SpeechToText(model_size="base")
        self.language_learner = LanguageLearner()
        self.music_prefs_file = "language_learning/data/music_preferences.json"
        
        # Create necessary directories
        os.makedirs("language_learning/logs", exist_ok=True)
        os.makedirs("language_learning/data", exist_ok=True)
        os.makedirs("language_learning/captures", exist_ok=True)
        
    def start(self):
        """Start the ambient listener thread"""
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        logger.info("🚀 Ambient listening started (learning speech AND music)")
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
    
    def stop(self):
        """Stop the ambient listener"""
        logger.info("🛑 Stopping ambient listener...")
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.cleanup()
    
    def cleanup(self):
        """Clean up all resources"""
        logger.info("🧹 Cleaning up ambient listener resources...")
        
        # Clean up temp files
        for temp_file in self.temp_files[:]:  # Copy list to avoid modification during iteration
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Removed temp file: {temp_file}")
            except Exception as e:
                logger.error(f"Error removing temp file {temp_file}: {e}")
            finally:
                if temp_file in self.temp_files:
                    self.temp_files.remove(temp_file)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("✅ Cleanup complete")
    
    def _listen_loop(self):
        """Main listening loop"""
        logger.info(f"🎧 Ambient listener activated (learning speech AND music)")
        
        while self.is_running:
            try:
                logger.info(f"🎧 Listening for {self.listen_duration} seconds...")
                
                # Record audio using sounddevice (no PyAudio)
                recording = sd.rec(
                    int(self.listen_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32'
                )
                sd.wait()
                
                # Save to temp file
                temp_file = self._save_to_temp(recording.flatten())
                
                # Process the audio
                if temp_file:
                    self.process_audio(temp_file)
                
                # Pause between recordings (check is_running during pause)
                if self.is_running and self.pause_between > 0:
                    logger.info(f"💤 Pausing for {self.pause_between} seconds...")
                    for _ in range(self.pause_between):
                        if not self.is_running:
                            break
                        time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                time.sleep(1)
        
        logger.info("👂 Ambient listener stopped")
    
    def _save_to_temp(self, audio_data):
        """Save audio data to a temporary file with automatic cleanup"""
        try:
            # Create temp file
            fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='dmai_ambient_')
            os.close(fd)
            
            # Save as WAV
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                
                # Convert float32 to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            # Track for cleanup
            self.temp_files.append(temp_path)
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving temp file: {e}")
            return None
    
    def is_music(self, audio_file):
        """Simple check if audio is likely music vs speech"""
        try:
            # Use ffmpeg to analyze audio properties if available
            result = subprocess.run([
                'ffprobe', '-i', audio_file, '-show_streams', 
                '-select_streams', 'a', '-print_format', 'json'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if data.get('streams'):
                    return True
        except:
            pass
        return False
    
    def extract_music_metadata(self, audio_file):
        """Try to identify music using available tools"""
        try:
            result = subprocess.run([
                'ffprobe', '-i', audio_file, '-show_format', '-print_format', 'json'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                format_data = data.get('format', {})
                tags = format_data.get('tags', {})
                
                artist = tags.get('artist') or tags.get('ARTIST') or tags.get('Artist')
                title = tags.get('title') or tags.get('TITLE') or tags.get('Title')
                album = tags.get('album') or tags.get('ALBUM')
                
                if artist or title:
                    return {
                        'artist': artist,
                        'title': title,
                        'album': album,
                        'source': 'metadata'
                    }
        except:
            pass
        return None
    
    def learn_music(self, music_info):
        """Update music preferences"""
        try:
            with open(self.music_prefs_file, 'r') as f:
                prefs = json.load(f)
        except:
            prefs = {"artists": {}, "songs": {}, "listening_history": []}
        
        if music_info.get('artist'):
            artist = music_info['artist']
            prefs["artists"][artist] = prefs["artists"].get(artist, 0) + 1
            logger.info(f"🎵 DMAI learned you like: {artist}")
        
        if music_info.get('title'):
            song = f"{music_info.get('artist', 'Unknown')} - {music_info['title']}"
            prefs["songs"][song] = prefs["songs"].get(song, 0) + 1
        
        prefs["listening_history"].append({
            "timestamp": datetime.now().isoformat(),
            "info": music_info
        })
        
        with open(self.music_prefs_file, 'w') as f:
            json.dump(prefs, f, indent=2)
    
    def learn_speech(self, audio_file):
        """Process speech for language learning"""
        try:
            # Transcribe the audio
            text = self.stt.transcribe_file(audio_file)
            
            if text and len(text) > 3:
                # Learn from the speech
                result = self.language_learner.process_text(text, source="ambient")
                
                if result and not result.get("rejected"):
                    logger.info(f"🗣️ Learned {result.get('new_words', 0)} new words from speech")
                else:
                    logger.info("🗣️ Speech processed (no new words)")
            else:
                logger.info("🗣️ No speech detected in audio")
                
        except Exception as e:
            logger.error(f"Error processing speech: {e}")
    
    def process_audio(self, filename):
        """Process captured audio - detect if music or speech, learn from both"""
        logger.info(f"🔍 Processing {filename}")
        
        try:
            # Check if this is music
            if self.is_music(filename):
                logger.info("🎵 Music detected! Learning preferences...")
                music_info = self.extract_music_metadata(filename)
                if music_info:
                    self.learn_music(music_info)
                else:
                    # Couldn't identify, but note that music was heard
                    self.learn_music({"artist": "Unknown", "note": "Unidentified music"})
            else:
                logger.info("🗣️ Speech detected - learning language...")
                self.learn_speech(filename)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
        
        # Call parent callback if set (for DMAI voice integration)
        if self.process_audio_callback:
            try:
                self.process_audio_callback(filename)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
        
        # Clean up temp file
        try:
            if os.path.exists(filename) and filename in self.temp_files:
                os.remove(filename)
                self.temp_files.remove(filename)
        except:
            pass
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        self.cleanup()
