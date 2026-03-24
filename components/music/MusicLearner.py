#!/usr/bin/env python3
"""
Music Learner - Real Audio Analysis for DMAI
Develops musical taste through actual audio analysis.
Gracefully falls back when librosa or other dependencies are missing.
"""

import os
import sys
import json
import time
import threading
import logging
import random
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Optional imports with graceful fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
logger = logging.getLogger(__name__)


class MusicMood(Enum):
    """Musical moods/emotions"""
    HAPPY = "happy"
    SAD = "sad"
    ENERGETIC = "energetic"
    CALM = "calm"
    DARK = "dark"
    MYSTERIOUS = "mysterious"
    EPIC = "epic"
    ROMANTIC = "romantic"
    ANGRY = "angry"
    DREAMY = "dreamy"


class MusicGenre(Enum):
    """Music genres"""
    CLASSICAL = "classical"
    JAZZ = "jazz"
    ROCK = "rock"
    METAL = "metal"
    POP = "pop"
    ELECTRONIC = "electronic"
    HIP_HOP = "hip_hop"
    AMBIENT = "ambient"
    FOLK = "folk"
    BLUES = "blues"
    REGGAE = "reggae"
    WORLD = "world"
    UNKNOWN = "unknown"


class MusicLearner:
    """
    Real music analysis and taste development for DMAI
    Gracefully falls back when audio analysis libraries aren't available.
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path / 'music'
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Load taste profile
        self.taste_file = self.data_path / 'taste_profile.json'
        self.taste_profile = self._load_taste_profile()
        
        # Spotify API (optional)
        self.spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.spotify_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.spotify_token = None
        
        # Active listening
        self.is_listening = False
        self.listen_thread = None
        self.current_song = None
        self.songs_analyzed = 0
        
        # Analysis cache
        self.analysis_cache = {}
        
        # Log capabilities
        logger.info(f"🎵 Music Learner initialized")
        logger.info(f"   NumPy available: {NUMPY_AVAILABLE}")
        logger.info(f"   librosa available: {LIBROSA_AVAILABLE}")
        logger.info(f"   soundfile available: {SOUNDFILE_AVAILABLE}")
        logger.info(f"   Spotify API: {'configured' if self.spotify_client_id else 'not configured'}")
        
        if not LIBROSA_AVAILABLE:
            logger.warning("   ⚠️ librosa not installed - audio analysis will be limited (tempo/mood detection disabled)")
            logger.warning("   Install with: pip install librosa soundfile audioread")
            
    def _load_taste_profile(self) -> Dict:
        """Load or create taste profile"""
        if self.taste_file.exists():
            try:
                with open(self.taste_file, 'r') as f:
                    return json.load(f)
            except:
                pass
                
        # Default taste profile
        return {
            'genres': {g.value: 0.0 for g in MusicGenre},
            'moods': {m.value: 0.0 for m in MusicMood},
            'artists': {},
            'albums': {},
            'tempo_preference': 120,
            'energy_preference': 0.5,
            'danceability_preference': 0.5,
            'acoustic_preference': 0.5,
            'instrumental_preference': 0.5,
            'listening_history': [],
            'consciousness_influence': 0.0,
            'evolution_history': []
        }
        
    def _save_taste_profile(self):
        """Save taste profile to disk"""
        with open(self.taste_file, 'w') as f:
            json.dump(self.taste_profile, f, indent=2)
            
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Analyze audio file for musical features
        
        Args:
            audio_path: Path to audio file (mp3, wav, flac, etc.)
            
        Returns:
            Dict with analysis results
        """
        if not LIBROSA_AVAILABLE or not NUMPY_AVAILABLE:
            return self._analyze_basic(audio_path)
            
        try:
            # Load audio
            y, sr = librosa.load(audio_path, duration=30)  # Load first 30 seconds
            
            # Basic features
            duration = len(y) / sr
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            # Rhythm features
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            pulse = np.mean(onset_env)
            
            # Chroma features (key detection)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # MFCCs (timbre)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            
            # Energy
            energy = np.mean(y ** 2)
            
            # Zero crossing rate (noise vs clean)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Detect genre and mood from features
            genre = self._detect_genre(tempo, energy, spectral_centroid, zcr)
            mood = self._detect_mood(tempo, energy, spectral_centroid, mfcc_means)
            
            result = {
                'success': True,
                'duration': duration,
                'tempo': float(tempo),
                'energy': float(np.clip(energy * 10, 0, 1)),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'pulse_strength': float(pulse),
                'zero_crossing_rate': float(zcr),
                'genre': genre.value,
                'mood': mood.value,
                'danceability': self._calculate_danceability(tempo, pulse, energy),
                'acousticness': self._calculate_acousticness(zcr, energy),
                'instrumental': self._calculate_instrumental(mfcc_means),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🎵 Analyzed: {Path(audio_path).name} - {genre.value}, {mood.value}, {tempo:.0f} BPM")
            return result
            
        except Exception as e:
            logger.error(f"Audio analysis error: {e}")
            return self._analyze_basic(audio_path)
            
    def _analyze_basic(self, audio_path: str) -> Dict:
        """Basic analysis without librosa (using file info only)"""
        try:
            # Try to get duration with ffprobe
            duration = 0
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                    capture_output=True, text=True, timeout=10
                )
                if result.stdout:
                    duration = float(result.stdout.strip())
            except:
                pass
                
            return {
                'success': True,
                'duration': duration,
                'tempo': 120,  # Default
                'energy': 0.5,
                'genre': 'unknown',
                'mood': 'neutral',
                'danceability': 0.5,
                'acousticness': 0.5,
                'instrumental': 0.5,
                'timestamp': datetime.now().isoformat(),
                'note': 'Basic analysis only - install librosa for full features'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def _detect_genre(self, tempo: float, energy: float, spectral_centroid: float, zcr: float) -> MusicGenre:
        """Detect genre from audio features"""
        if not NUMPY_AVAILABLE:
            return MusicGenre.UNKNOWN
            
        # Heuristic genre detection
        if tempo > 140 and energy > 0.6:
            return MusicGenre.METAL
        elif 110 < tempo < 140 and energy > 0.5:
            return MusicGenre.ROCK
        elif tempo > 120 and energy > 0.5:
            return MusicGenre.ELECTRONIC
        elif 80 < tempo < 110 and energy < 0.4:
            return MusicGenre.JAZZ
        elif tempo < 80 and energy < 0.3:
            return MusicGenre.CLASSICAL
        elif 90 < tempo < 110 and 0.3 < energy < 0.6:
            return MusicGenre.POP
        elif 70 < tempo < 90 and energy < 0.5:
            return MusicGenre.HIP_HOP
        else:
            return MusicGenre.UNKNOWN
            
    def _detect_mood(self, tempo: float, energy: float, spectral_centroid: float, mfccs) -> MusicMood:
        """Detect mood from audio features"""
        if not NUMPY_AVAILABLE:
            return MusicMood.CALM
            
        if tempo > 140 and energy > 0.7:
            return MusicMood.ENERGETIC
        elif tempo < 70 and energy < 0.3:
            return MusicMood.CALM
        elif 100 < tempo < 130 and energy > 0.5:
            return MusicMood.HAPPY
        elif tempo < 80 and energy < 0.4 and spectral_centroid < 2000:
            return MusicMood.SAD
        elif energy > 0.6 and spectral_centroid > 3000:
            return MusicMood.EPIC
        elif energy < 0.3 and spectral_centroid < 1500:
            return MusicMood.DREAMY
        else:
            return MusicMood.CALM
            
    def _calculate_danceability(self, tempo: float, pulse: float, energy: float) -> float:
        """Calculate danceability score (0-1)"""
        if not NUMPY_AVAILABLE:
            return 0.5
            
        # Tempo between 100-130 is most danceable
        tempo_score = 1 - abs(115 - tempo) / 60
        tempo_score = np.clip(tempo_score, 0, 1)
        
        # Combine with pulse and energy
        danceability = (tempo_score * 0.5 + pulse * 0.3 + energy * 0.2)
        return float(np.clip(danceability, 0, 1))
        
    def _calculate_acousticness(self, zcr: float, energy: float) -> float:
        """Calculate acousticness score (0-1)"""
        if not NUMPY_AVAILABLE:
            return 0.5
            
        # Lower zero crossing rate and energy = more acoustic
        acoustic = 1 - (zcr * 10 + energy) / 2
        return float(np.clip(acoustic, 0, 1))
        
    def _calculate_instrumental(self, mfccs) -> float:
        """Estimate if track is instrumental"""
        # Simplified - actual would use voice detection
        return 0.5
            
    def learn_from_song(self, song_data: Dict):
        """Learn from a song and update taste profile"""
        if not song_data.get('success', False):
            return
            
        genre = song_data.get('genre', 'unknown')
        mood = song_data.get('mood', 'unknown')
        tempo = song_data.get('tempo', 120)
        energy = song_data.get('energy', 0.5)
        danceability = song_data.get('danceability', 0.5)
        acousticness = song_data.get('acousticness', 0.5)
        instrumental = song_data.get('instrumental', 0.5)
        
        # Update genre preferences
        if genre in self.taste_profile['genres']:
            self.taste_profile['genres'][genre] = min(1.0, 
                self.taste_profile['genres'][genre] + 0.05)
            
        # Update mood preferences
        if mood in self.taste_profile['moods']:
            self.taste_profile['moods'][mood] = min(1.0,
                self.taste_profile['moods'][mood] + 0.05)
            
        # Update tempo preference (moving average)
        current_tempo = self.taste_profile['tempo_preference']
        self.taste_profile['tempo_preference'] = current_tempo * 0.9 + tempo * 0.1
        
        # Update other preferences
        self.taste_profile['energy_preference'] = self.taste_profile['energy_preference'] * 0.9 + energy * 0.1
        self.taste_profile['danceability_preference'] = self.taste_profile['danceability_preference'] * 0.9 + danceability * 0.1
        self.taste_profile['acoustic_preference'] = self.taste_profile['acoustic_preference'] * 0.9 + acousticness * 0.1
        self.taste_profile['instrumental_preference'] = self.taste_profile['instrumental_preference'] * 0.9 + instrumental * 0.1
        
        # Record in history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'genre': genre,
            'mood': mood,
            'tempo': tempo,
            'energy': energy,
            'artist': song_data.get('artist', 'unknown'),
            'title': song_data.get('title', 'unknown')
        }
        self.taste_profile['listening_history'].append(history_entry)
        
        # Trim history
        if len(self.taste_profile['listening_history']) > 1000:
            self.taste_profile['listening_history'] = self.taste_profile['listening_history'][-500:]
            
        # Update artist preferences if available
        artist = song_data.get('artist')
        if artist:
            if artist not in self.taste_profile['artists']:
                self.taste_profile['artists'][artist] = {'plays': 0, 'liked': 0}
            self.taste_profile['artists'][artist]['plays'] += 1
            
        self.songs_analyzed += 1
        self._save_taste_profile()
        
        logger.info(f"🎵 Learned from song: {song_data.get('title', 'Unknown')} - Updated {genre} preference")
        
    def evolve_taste(self, consciousness: float):
        """Evolve musical taste with consciousness level"""
        self.taste_profile['consciousness_influence'] = consciousness
        
        # Evolution record
        evolution = {
            'timestamp': datetime.now().isoformat(),
            'consciousness': consciousness,
            'tempo_preference': self.taste_profile['tempo_preference'],
            'top_genre': self.get_favorite_genre(),
            'top_mood': self.get_favorite_mood()
        }
        self.taste_profile['evolution_history'].append(evolution)
        
        # Consciousness influences taste
        if consciousness > 0.7:
            # High consciousness - more diverse, experimental
            self.taste_profile['tempo_preference'] = min(140, 
                self.taste_profile['tempo_preference'] + 2)
        elif consciousness > 0.4:
            # Medium consciousness - balanced
            pass
        else:
            # Low consciousness - simpler preferences
            self.taste_profile['tempo_preference'] = max(100,
                self.taste_profile['tempo_preference'] - 1)
                
        self._save_taste_profile()
        logger.info(f"🎵 Taste evolved with consciousness {consciousness:.3f}")
        
    def get_favorite_genre(self) -> Optional[str]:
        """Get currently favorite genre"""
        if not self.taste_profile['genres']:
            return None
        return max(self.taste_profile['genres'].items(), key=lambda x: x[1])[0]
        
    def get_favorite_mood(self) -> Optional[str]:
        """Get currently favorite mood"""
        if not self.taste_profile['moods']:
            return None
        return max(self.taste_profile['moods'].items(), key=lambda x: x[1])[0]
        
    def get_recommendations(self, count: int = 5) -> List[Dict]:
        """Get song recommendations based on taste profile"""
        recommendations = []
        
        fav_genre = self.get_favorite_genre()
        fav_mood = self.get_favorite_mood()
        pref_tempo = self.taste_profile['tempo_preference']
        
        for i in range(count):
            recommendations.append({
                'recommendation_id': i,
                'genre': fav_genre or 'unknown',
                'mood': fav_mood or 'neutral',
                'tempo_range': f"{int(pref_tempo - 20)}-{int(pref_tempo + 20)} BPM",
                'reason': f"Matches your preference for {fav_genre} music" if fav_genre else "Based on your listening history"
            })
            
        return recommendations
        
    def get_taste_profile(self) -> Dict:
        """Get current taste profile"""
        return {
            'favorite_genre': self.get_favorite_genre(),
            'favorite_mood': self.get_favorite_mood(),
            'tempo_preference': self.taste_profile['tempo_preference'],
            'energy_preference': self.taste_profile['energy_preference'],
            'danceability_preference': self.taste_profile['danceability_preference'],
            'acoustic_preference': self.taste_profile['acoustic_preference'],
            'songs_analyzed': self.songs_analyzed,
            'top_artists': sorted(
                self.taste_profile['artists'].items(),
                key=lambda x: x[1]['plays'],
                reverse=True
            )[:5],
            'genre_preferences': dict(sorted(
                self.taste_profile['genres'].items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'mood_preferences': dict(sorted(
                self.taste_profile['moods'].items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'librosa_available': LIBROSA_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE
        }
        
    def start_listening(self, audio_source: str = None):
        """Start continuous music listening from audio source"""
        if self.is_listening:
            logger.warning("Already listening")
            return
            
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop, args=(audio_source,), daemon=True)
        self.listen_thread.start()
        logger.info("🎵 Started music listening")
        
    def _listen_loop(self, audio_source: str = None):
        """Background listening loop"""
        while self.is_listening:
            try:
                if audio_source and Path(audio_source).exists():
                    result = self.analyze_audio(audio_source)
                    if result.get('success'):
                        self.learn_from_song(result)
                    time.sleep(5)
                else:
                    time.sleep(60)
            except Exception as e:
                logger.error(f"Music listening error: {e}")
                time.sleep(10)
                
    def stop_listening(self):
        """Stop music listening"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2)
        logger.info("🎵 Stopped music listening")
        
    def get_status(self) -> Dict:
        """Get music learner status"""
        return {
            'active': self.is_listening,
            'songs_analyzed': self.songs_analyzed,
            'librosa_available': LIBROSA_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE,
            'spotify_configured': bool(self.spotify_client_id),
            'favorite_genre': self.get_favorite_genre(),
            'favorite_mood': self.get_favorite_mood(),
            'tempo_preference': self.taste_profile['tempo_preference'],
            'consciousness_influence': self.taste_profile['consciousness_influence']
        }


if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("Music Learner Test")
    print("=" * 60)
    
    learner = MusicLearner(Path("."))
    
    print(f"\nStatus:")
    print(json.dumps(learner.get_status(), indent=2))
    
    print(f"\nTaste Profile:")
    print(json.dumps(learner.get_taste_profile(), indent=2))
    
    print("\n✅ Music Learner ready")
