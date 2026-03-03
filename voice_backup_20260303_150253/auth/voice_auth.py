"""Voice Biometric Authentication"""
import numpy as np
import pickle
import os
import json
from datetime import datetime
import soundfile as sf
from scipy import signal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceAuth:
    """Authenticate users by their unique voice characteristics"""
    
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.users_file = os.path.join(self.config.get('data_dir', 'voice/enrollment_data'), 'users.json')
        self.voiceprints = {}
        self.load_users()
    
    def load_config(self, config_path):
        """Load or create default config"""
        default_config = {
            'data_dir': 'voice/enrollment_data',
            'min_confidence': 0.85,
            'sample_rate': 16000,
            'freq_range': (300, 3400)  # Human voice range
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_users(self):
        """Load registered users"""
        os.makedirs(self.config['data_dir'], exist_ok=True)
        
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                users = json.load(f)
                for user_id, user_data in users.items():
                    if 'voiceprint_path' in user_data:
                        vp_path = user_data['voiceprint_path']
                        if os.path.exists(vp_path):
                            with open(vp_path, 'rb') as vpf:
                                self.voiceprints[user_id] = pickle.load(vpf)
        else:
            # Initialize empty users dict
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
    
    def extract_features(self, audio_data, sample_rate):
        """Extract voice features for biometric comparison"""
        # Ensure audio is float in [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data / 32768.0
        
        # Compute spectrogram
        frequencies, times, spectrogram = signal.spectrogram(
            audio_data, 
            fs=sample_rate,
            nperseg=512,
            noverlap=256
        )
        
        # Focus on voice frequency range
        voice_freq_mask = (frequencies >= self.config['freq_range'][0]) & \
                          (frequencies <= self.config['freq_range'][1])
        
        voice_spectrogram = spectrogram[voice_freq_mask, :]
        
        # Compute statistical features
        if voice_spectrogram.size > 0:
            mean_spectral = np.mean(voice_spectrogram, axis=1)
            std_spectral = np.std(voice_spectrogram, axis=1)
            
            # Spectral centroid
            if np.sum(voice_spectrogram, axis=0).size > 0:
                spectral_centroid = np.sum(
                    frequencies[voice_freq_mask, None] * voice_spectrogram, axis=0
                ) / (np.sum(voice_spectrogram, axis=0) + 1e-10)
            else:
                spectral_centroid = np.array([0])
        else:
            mean_spectral = np.array([0])
            std_spectral = np.array([0])
            spectral_centroid = np.array([0])
        
        features = {
            'mean_spectral': mean_spectral,
            'std_spectral': std_spectral,
            'spectral_centroid': spectral_centroid,
            'rms_energy': np.sqrt(np.mean(audio_data**2)),
            'zero_crossing_rate': np.mean(np.abs(np.diff(np.sign(audio_data)))) / 2
        }
        
        return features
    
    def create_voiceprint(self, audio_samples, sample_rate, user_id):
        """Create a voiceprint from multiple audio samples"""
        all_features = []
        
        for audio in audio_samples:
            features = self.extract_features(audio, sample_rate)
            all_features.append(features)
        
        # Average the features to create voiceprint
        voiceprint = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'samples_used': len(audio_samples),
            'mean_spectral': np.mean([f['mean_spectral'] for f in all_features], axis=0),
            'std_spectral': np.mean([f['std_spectral'] for f in all_features], axis=0),
            'mean_centroid': np.mean([f['spectral_centroid'].mean() for f in all_features]),
            'std_centroid': np.std([f['spectral_centroid'].mean() for f in all_features]),
            'mean_rms': np.mean([f['rms_energy'] for f in all_features]),
            'std_rms': np.std([f['rms_energy'] for f in all_features]),
            'mean_zcr': np.mean([f['zero_crossing_rate'] for f in all_features]),
            'std_zcr': np.std([f['zero_crossing_rate'] for f in all_features])
        }
        
        return voiceprint
    
    def enroll_master(self, audio_samples, sample_rate):
        """Enroll the master user (you)"""
        logger.info("Enrolling master user...")
        
        voiceprint = self.create_voiceprint(audio_samples, sample_rate, "master")
        
        # Save voiceprint
        vp_path = os.path.join(self.config['data_dir'], 'master_voiceprint.pkl')
        with open(vp_path, 'wb') as f:
            pickle.dump(voiceprint, f)
        
        # Update users.json
        with open(self.users_file, 'r') as f:
            users = json.load(f)
        
        users['master'] = {
            'role': 'master_admin',
            'name': 'David',
            'enrolled_at': datetime.now().isoformat(),
            'voiceprint_path': vp_path,
            'biometrics_enrolled': ['voice']
        }
        
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
        
        self.voiceprints['master'] = voiceprint
        logger.info("Master enrollment complete")
        
        return True
    
    def verify(self, audio_data, sample_rate, user_id="master"):
        """Verify if audio matches enrolled user"""
        if user_id not in self.voiceprints:
            logger.warning(f"No voiceprint for user {user_id}")
            return False, 0.0
        
        enrolled = self.voiceprints[user_id]
        current = self.extract_features(audio_data, sample_rate)
        
        try:
            # Compare features
            if len(enrolled['mean_spectral']) > 0 and len(current['mean_spectral']) > 0:
                # Reshape for correlation if needed
                min_len = min(len(enrolled['mean_spectral']), len(current['mean_spectral']))
                spectral_similarity = np.corrcoef(
                    enrolled['mean_spectral'][:min_len], 
                    current['mean_spectral'][:min_len]
                )[0, 1] if min_len > 1 else 0.5
            else:
                spectral_similarity = 0.5
            
            centroid_diff = abs(current['spectral_centroid'].mean() - enrolled['mean_centroid'])
            centroid_similarity = max(0, 1 - centroid_diff / 500)  # Normalize
            
            rms_similarity = 1 - min(1, abs(current['rms_energy'] - enrolled['mean_rms']) / 
                                      (enrolled['mean_rms'] + 1e-10))
            
            # Weighted combination
            confidence = (
                0.5 * max(0, spectral_similarity) +
                0.3 * centroid_similarity +
                0.2 * max(0, rms_similarity)
            )
            
            is_match = confidence >= self.config['min_confidence']
            
            if is_match:
                logger.info(f"Voice match: {confidence:.2f}")
            else:
                logger.warning(f"Voice mismatch: {confidence:.2f}")
            
            return is_match, confidence
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return False, 0.0
