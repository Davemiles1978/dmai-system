#!/usr/bin/env python3
"""
Voice Integration Module - Real TTS and STT for DMAI
Supports multiple backends with graceful fallback when dependencies missing.
- TTS: pyttsx3 (offline), ElevenLabs (cloud), OpenAI TTS, system fallback
- STT: Whisper (local), OpenAI Whisper API, Google Speech-to-Text
"""

import os
import sys
import json
import time
import threading
import logging
import queue
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from enum import Enum
import tempfile
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
logger = logging.getLogger(__name__)


class TTSBackend(Enum):
    """Text-to-Speech backends"""
    PYTTSX3 = "pyttsx3"      # Offline, system voice
    ELEVENLABS = "elevenlabs" # Cloud, high quality
    OPENAI_TTS = "openai_tts" # OpenAI TTS API
    SYSTEM = "system"         # System fallback


class STTBackend(Enum):
    """Speech-to-Text backends"""
    WHISPER_LOCAL = "whisper_local"     # Local Whisper model
    WHISPER_API = "whisper_api"         # OpenAI Whisper API
    GOOGLE_STT = "google_stt"           # Google Cloud Speech-to-Text
    SPEECH_RECOGNITION = "speech_recognition"  # Web API fallback


class VoiceIntegration:
    """
    Complete voice integration for DMAI
    Handles listening (STT) and speaking (TTS) with graceful fallbacks
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path / 'voice'
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.tts_backend = None
        self.stt_backend = None
        
        # Load API keys
        self.elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.google_stt_key = os.getenv('GOOGLE_STT_KEY')
        
        # Voice profile
        self.voice_profile = {
            'pitch': 1.0,
            'speed': 1.0,
            'voice_id': 'default',
            'language': 'en'
        }
        
        # Listening state
        self.is_listening = False
        self.listen_thread = None
        self.audio_queue = queue.Queue()
        self.wake_word = "hey dma"
        self.wake_word_active = False
        self.listen_callback = None
        
        # Initialize components (with fallbacks)
        self._init_tts()
        self._init_stt()
        
        logger.info(f"🎤 Voice Integration initialized")
        logger.info(f"   TTS Backend: {self.tts_backend.value if self.tts_backend else 'None'}")
        logger.info(f"   STT Backend: {self.stt_backend.value if self.stt_backend else 'None'}")
        
        if not self.stt_backend:
            logger.info("   Note: No STT backend available. Voice commands disabled.")
        if self.tts_backend == TTSBackend.SYSTEM:
            logger.info("   Note: Using system TTS fallback")
        
    def _init_tts(self):
        """Initialize Text-to-Speech backend with fallbacks"""
        # Try pyttsx3 (offline)
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_backend = TTSBackend.PYTTSX3
            logger.info("   ✅ pyttsx3 TTS initialized")
            return
        except ImportError:
            logger.debug("pyttsx3 not installed")
        except Exception as e:
            logger.debug(f"pyttsx3 init error: {e}")
            
        # Try ElevenLabs (cloud)
        if self.elevenlabs_key and self.elevenlabs_key != "pending":
            try:
                from elevenlabs import generate, play
                self.elevenlabs_available = True
                self.tts_backend = TTSBackend.ELEVENLABS
                logger.info("   ✅ ElevenLabs TTS initialized")
                return
            except ImportError:
                logger.debug("elevenlabs not installed")
            except Exception as e:
                logger.debug(f"ElevenLabs init error: {e}")
                
        # Try OpenAI TTS
        if self.openai_key and self.openai_key != "pending":
            try:
                import openai
                self.openai_client = openai
                self.openai_client.api_key = self.openai_key
                self.tts_backend = TTSBackend.OPENAI_TTS
                logger.info("   ✅ OpenAI TTS initialized")
                return
            except ImportError:
                logger.debug("openai not installed")
            except Exception as e:
                logger.debug(f"OpenAI TTS init error: {e}")
                
        # Fallback to system
        self.tts_backend = TTSBackend.SYSTEM
        logger.info("   ⚠️ Using system TTS fallback (say/espeak)")
        
    def _init_stt(self):
        """Initialize Speech-to-Text backend with fallbacks"""
        # Try Whisper local
        try:
            import whisper
            self.whisper_model = whisper.load_model("base")
            self.stt_backend = STTBackend.WHISPER_LOCAL
            logger.info("   ✅ Whisper local STT initialized")
            return
        except ImportError:
            logger.debug("whisper not installed")
        except Exception as e:
            logger.debug(f"Whisper local init error: {e}")
            
        # Try Whisper API
        if self.openai_key and self.openai_key != "pending":
            try:
                import openai
                self.openai_client = openai
                self.openai_client.api_key = self.openai_key
                self.stt_backend = STTBackend.WHISPER_API
                logger.info("   ✅ Whisper API STT ready")
                return
            except ImportError:
                logger.debug("openai not installed")
                
        # Try Google STT
        if self.google_stt_key and self.google_stt_key != "pending":
            try:
                from google.cloud import speech
                self.stt_backend = STTBackend.GOOGLE_STT
                logger.info("   ✅ Google STT ready")
                return
            except ImportError:
                logger.debug("google-cloud-speech not installed")
                
        # Try SpeechRecognition (web API) - this is the fallback that works without pyaudio?
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.stt_backend = STTBackend.SPEECH_RECOGNITION
            logger.info("   ✅ SpeechRecognition (Google Web API) ready")
            return
        except ImportError:
            logger.debug("speech_recognition not installed")
            
        # No STT available
        self.stt_backend = None
        logger.warning("   ⚠️ No STT backend available - voice commands disabled")
        
    def set_voice_profile(self, pitch: float = None, speed: float = None, voice_id: str = None):
        """Update voice profile"""
        if pitch is not None:
            self.voice_profile['pitch'] = pitch
        if speed is not None:
            self.voice_profile['speed'] = speed
        if voice_id is not None:
            self.voice_profile['voice_id'] = voice_id
            
        # Update TTS engine if available
        if hasattr(self, 'tts_engine') and self.tts_backend == TTSBackend.PYTTSX3:
            self.tts_engine.setProperty('rate', int(150 * self.voice_profile['speed']))
            self.tts_engine.setProperty('volume', 1.0)
            
        logger.info(f"🎤 Voice profile updated: {self.voice_profile}")
        
    def speak(self, text: str, wait: bool = True):
        """
        Speak text using configured TTS backend
        """
        if not text:
            return
            
        logger.info(f"🗣️ Speaking: {text[:100]}...")
        
        if self.tts_backend == TTSBackend.PYTTSX3:
            self._speak_pyttsx3(text, wait)
        elif self.tts_backend == TTSBackend.ELEVENLABS:
            self._speak_elevenlabs(text, wait)
        elif self.tts_backend == TTSBackend.OPENAI_TTS:
            self._speak_openai_tts(text, wait)
        else:
            self._speak_system(text, wait)
            
    def _speak_pyttsx3(self, text: str, wait: bool):
        """Speak using pyttsx3"""
        try:
            self.tts_engine.say(text)
            if wait:
                self.tts_engine.runAndWait()
            else:
                threading.Thread(target=self.tts_engine.runAndWait, daemon=True).start()
        except Exception as e:
            logger.error(f"pyttsx3 speech error: {e}")
            self._speak_system(text, wait)
            
    def _speak_elevenlabs(self, text: str, wait: bool):
        """Speak using ElevenLabs API"""
        try:
            from elevenlabs import generate, play
            audio = generate(
                text=text,
                voice=self.voice_profile['voice_id'] if self.voice_profile['voice_id'] != 'default' else "Rachel",
                model="eleven_monolingual_v1"
            )
            if wait:
                play(audio)
            else:
                threading.Thread(target=play, args=(audio,), daemon=True).start()
        except Exception as e:
            logger.error(f"ElevenLabs speech error: {e}")
            self._speak_pyttsx3(text, wait)
            
    def _speak_openai_tts(self, text: str, wait: bool):
        """Speak using OpenAI TTS API"""
        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            
            # Save to temp file and play
            temp_file = self.data_path / "temp_speech.mp3"
            response.stream_to_file(str(temp_file))
            
            # Play with system player
            self._play_audio_file(str(temp_file), wait)
            
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            self._speak_system(text, wait)
            
    def _play_audio_file(self, filepath: str, wait: bool):
        """Play audio file using system player"""
        try:
            if sys.platform == "darwin":  # macOS
                cmd = ["afplay", filepath]
                if wait:
                    subprocess.run(cmd, check=False)
                else:
                    subprocess.Popen(cmd)
            elif sys.platform == "linux":
                cmd = ["mpg123", filepath]
                if wait:
                    subprocess.run(cmd, check=False)
                else:
                    subprocess.Popen(cmd)
            elif sys.platform == "win32":
                import winsound
                winsound.PlaySound(filepath, winsound.SND_FILENAME)
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            
    def _speak_system(self, text: str, wait: bool):
        """Fallback to system say command"""
        try:
            if sys.platform == "darwin":  # macOS
                cmd = f'say "{text[:200]}"'
                if wait:
                    subprocess.run(cmd, shell=True, check=False)
                else:
                    subprocess.Popen(cmd, shell=True)
            elif sys.platform == "linux":
                cmd = f'spd-say "{text[:200]}"'
                subprocess.Popen(cmd, shell=True)
            elif sys.platform == "win32":
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                speaker.Speak(text)
            else:
                logger.warning(f"Cannot speak: {text[:50]}... (no TTS on this platform)")
        except Exception as e:
            logger.error(f"System speech error: {e}")
            
    def start_listening(self, callback: Callable[[str], None] = None):
        """
        Start continuous listening for speech (STT must be available)
        """
        if self.stt_backend is None:
            logger.warning("No STT backend available - cannot listen")
            return
            
        if self.is_listening:
            logger.warning("Already listening")
            return
            
        self.is_listening = True
        self.listen_callback = callback
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        logger.info("🎤 Started listening for speech")
        
    def stop_listening(self):
        """Stop listening"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2)
        logger.info("🎤 Stopped listening")
        
    def _listen_loop(self):
        """Main listening loop"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                logger.info("   Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                
            while self.is_listening:
                try:
                    with self.microphone as source:
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                        
                    text = self._transcribe_audio(audio)
                    
                    if text and self.listen_callback:
                        self.listen_callback(text)
                        
                except sr.WaitTimeoutError:
                    pass
                except Exception as e:
                    logger.error(f"Listening error: {e}")
                    time.sleep(1)
                    
        except ImportError:
            logger.error("SpeechRecognition not installed. Run: pip install SpeechRecognition")
            self._listen_loop_fallback()
        except Exception as e:
            logger.error(f"Listen loop error: {e}")
            self._listen_loop_fallback()
            
    def _listen_loop_fallback(self):
        """Fallback listening using system microphone (basic) - PRESERVED from original"""
        try:
            import pyaudio
            import wave
            import audioop
            
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                          input=True, frames_per_buffer=CHUNK)
            
            SILENCE_THRESHOLD = 500
            SILENCE_DURATION = 2
            silence_frames = 0
            recording = []
            is_recording = False
            
            while self.is_listening:
                data = stream.read(CHUNK)
                audio_level = max(audioop.rms(data, 2), 1)
                
                if audio_level > SILENCE_THRESHOLD:
                    if not is_recording:
                        is_recording = True
                        recording = []
                    recording.append(data)
                    silence_frames = 0
                else:
                    if is_recording:
                        silence_frames += 1
                        if silence_frames > (SILENCE_DURATION * RATE / CHUNK):
                            is_recording = False
                            audio_data = b''.join(recording)
                            text = self._transcribe_audio_bytes(audio_data)
                            if text and self.listen_callback:
                                self.listen_callback(text)
                            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except ImportError:
            logger.error("pyaudio not installed - cannot listen")
            self.is_listening = False
        except Exception as e:
            logger.error(f"Fallback listening error: {e}")
            self.is_listening = False
            
    def _transcribe_audio(self, audio) -> Optional[str]:
        """Transcribe audio to text using configured STT backend"""
        if self.stt_backend == STTBackend.WHISPER_LOCAL:
            return self._transcribe_whisper_local(audio)
        elif self.stt_backend == STTBackend.WHISPER_API:
            return self._transcribe_whisper_api(audio)
        elif self.stt_backend == STTBackend.GOOGLE_STT:
            return self._transcribe_google_stt(audio)
        else:
            return self._transcribe_speech_recognition(audio)
            
    def _transcribe_whisper_local(self, audio) -> Optional[str]:
        """Transcribe using local Whisper model"""
        try:
            temp_file = self.data_path / "temp_audio.wav"
            with open(temp_file, 'wb') as f:
                f.write(audio.get_wav_data())
                
            result = self.whisper_model.transcribe(str(temp_file))
            text = result["text"].strip()
            
            if text:
                logger.info(f"📝 Whisper (local): {text}")
                return text
        except Exception as e:
            logger.error(f"Whisper local error: {e}")
        return None
        
    def _transcribe_whisper_api(self, audio) -> Optional[str]:
        """Transcribe using OpenAI Whisper API"""
        try:
            temp_file = self.data_path / "temp_audio.wav"
            with open(temp_file, 'wb') as f:
                f.write(audio.get_wav_data())
                
            with open(temp_file, 'rb') as f:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
                
            text = transcript.text.strip()
            if text:
                logger.info(f"📝 Whisper API: {text}")
                return text
        except Exception as e:
            logger.error(f"Whisper API error: {e}")
        return None
        
    def _transcribe_google_stt(self, audio) -> Optional[str]:
        """Transcribe using Google Cloud Speech-to-Text"""
        try:
            from google.cloud import speech
            client = speech.SpeechClient()
            
            audio_data = audio.get_wav_data()
            audio_obj = speech.RecognitionAudio(content=audio_data)
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
            )
            
            response = client.recognize(config=config, audio=audio_obj)
            
            if response.results:
                text = response.results[0].alternatives[0].transcript
                logger.info(f"📝 Google STT: {text}")
                return text
        except Exception as e:
            logger.error(f"Google STT error: {e}")
        return None
        
    def _transcribe_speech_recognition(self, audio) -> Optional[str]:
        """Transcribe using speech_recognition library (Google Web API)"""
        try:
            text = self.recognizer.recognize_google(audio)
            logger.info(f"📝 Google Web STT: {text}")
            return text
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            logger.error(f"STT request error: {e}")
        return None
        
    def _transcribe_audio_bytes(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe raw audio bytes - PRESERVED from original"""
        try:
            import io
            import wave
            
            # Create WAV from bytes
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)
                
            wav_io.seek(0)
            
            # Use whisper local for transcription if available
            if self.stt_backend == STTBackend.WHISPER_LOCAL:
                temp_file = self.data_path / "temp_audio.wav"
                with open(temp_file, 'wb') as f:
                    f.write(wav_io.getvalue())
                result = self.whisper_model.transcribe(str(temp_file))
                text = result["text"].strip()
                if text:
                    logger.info(f"📝 Whisper: {text}")
                    return text
                    
        except Exception as e:
            logger.error(f"Audio bytes transcription error: {e}")
        return None
        
    def wake_word_detect(self, text: str) -> bool:
        """Check if text contains wake word"""
        return self.wake_word in text.lower()
        
    def set_wake_word(self, word: str):
        """Set the wake word for activation"""
        self.wake_word = word.lower()
        logger.info(f"Wake word set to: '{self.wake_word}'")
        
    def get_status(self) -> Dict:
        """Get voice system status"""
        return {
            'tts_backend': self.tts_backend.value if self.tts_backend else None,
            'stt_backend': self.stt_backend.value if self.stt_backend else None,
            'is_listening': self.is_listening,
            'voice_profile': self.voice_profile,
            'wake_word': self.wake_word,
            'wake_word_active': self.wake_word_active
        }


# For testing
if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("Voice Integration Test")
    print("=" * 60)
    
    voice = VoiceIntegration(Path("."))
    
    print("\nStatus:")
    print(json.dumps(voice.get_status(), indent=2))
    
    print("\nTesting TTS:")
    voice.speak("Hey, I am DeeMai. My voice is now active.")
    
    if voice.stt_backend:
        print("\nTesting STT (listening for 5 seconds)...")
        
        def on_speech(text):
            print(f"\n📝 Heard: {text}")
            if voice.wake_word_detect(text):
                print("🔊 Wake word detected!")
                voice.speak("Yes, I'm here. How can I help?")
                
        voice.start_listening(on_speech)
        
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            voice.stop_listening()
    else:
        print("\n⚠️ STT not available - skipping listening test")
        
    print("\n✅ Voice Integration test complete")
