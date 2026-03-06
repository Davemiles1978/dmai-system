"""DMAI Ambient Listening - Learns from environment (speech + music)"""
import pyaudio
import wave
import threading
import queue
import os
from datetime import datetime
import json
import logging
import tempfile
import subprocess
import time

# Import speech-to-text for language learning
from voice.speech_to_text import SpeechToText
from language_learning.processor.language_learner import LanguageLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmbientListener:
    """Listens to environment to learn language and music naturally"""
    
    def __init__(self, listen_duration=10, pause_between=30):
        self.listen_duration = listen_duration
        self.pause_between = pause_between
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.stop_event = threading.Event()
        self.learning_log = "language_learning/logs/heard_phrases.json"
        self.music_prefs_file = "language_learning/data/music_preferences.json"
        
        # Initialize both learning systems
        self.stt = SpeechToText(model_size="base")
        self.language_learner = LanguageLearner()
        
        os.makedirs("language_learning/logs", exist_ok=True)
        os.makedirs("language_learning/data", exist_ok=True)
        os.makedirs("language_learning/captures", exist_ok=True)
        
    def listen_chunk(self):
        """Listen for a short chunk of audio"""
        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        rate = 16000
        
        p = pyaudio.PyAudio()
        stream = p.open(format=format,
                       channels=channels,
                       rate=rate,
                       input=True,
                       frames_per_buffer=chunk)
        
        logger.info(f"🎧 Listening for {self.listen_duration} seconds...")
        frames = []
        
        for i in range(0, int(rate / chunk * self.listen_duration)):
            if self.stop_event.is_set():
                break
            data = stream.read(chunk)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save audio for later processing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"language_learning/captures/audio_{timestamp}.wav"
        
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return filename
    
    def is_music(self, audio_file):
        """Simple check if audio is likely music vs speech"""
        try:
            # Use ffmpeg to analyze audio properties
            # Music typically has more consistent volume and frequency range
            result = subprocess.run([
                'ffprobe', '-i', audio_file, '-show_streams', 
                '-select_streams', 'a', '-print_format', 'json'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if data.get('streams'):
                    # This is a simple heuristic - can be improved
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
                
                # Look for music metadata
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
        
        # Clean up the audio file to save space
        try:
            os.remove(filename)
        except:
            pass
        
        return {"file": filename, "processed": True}
    
    def learning_loop(self):
        """Main listening loop"""
        logger.info("🚀 Ambient listening started (learning speech AND music)")
        
        while self.is_listening and not self.stop_event.is_set():
            try:
                # Listen
                audio_file = self.listen_chunk()
                
                if audio_file and not self.stop_event.is_set():
                    # Queue for processing
                    self.audio_queue.put(audio_file)
                    
                    # Process in background
                    self.process_queue()
                    
                    # Pause before next listen (check stop_event during pause)
                    logger.info(f"💤 Pausing for {self.pause_between} seconds...")
                    for _ in range(self.pause_between):
                        if self.stop_event.is_set():
                            break
                        time.sleep(1)
                
            except Exception as e:
                logger.error(f"Listening error: {e}")
                if self.stop_event.is_set():
                    break
                time.sleep(1)
    
    def process_queue(self):
        """Process audio files in background"""
        while not self.audio_queue.empty() and not self.stop_event.is_set():
            audio_file = self.audio_queue.get()
            # Process in thread to not block listening
            thread = threading.Thread(target=self.process_audio, args=(audio_file,))
            thread.daemon = True
            thread.start()
    
    def start(self):
        """Start ambient listening"""
        self.is_listening = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.learning_loop, daemon=True)
        self.thread.start()
        logger.info("🎧 Ambient listener activated (learning speech AND music)")
    
    def stop(self):
        """Stop listening and clean up"""
        self.is_listening = False
        self.stop_event.set()
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                pass
        
        # Wait for thread to finish
        if hasattr(self, 'thread') and self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        logger.info("🛑 Ambient listener stopped")
    
    def __del__(self):
        """Destructor for clean shutdown"""
        self.stop()

if __name__ == "__main__":
    # Test
    listener = AmbientListener(listen_duration=5, pause_between=10)
    listener.start()
    
    try:
        # Run for 1 minute with proper cleanup
        time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        listener.stop()
