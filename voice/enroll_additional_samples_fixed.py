#!/usr/bin/env python3
"""Add more voice samples to existing enrollment - fixed version"""
import pickle
import sounddevice as sd
import numpy as np
import os
import logging
from datetime import datetime
from scipy import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_features(audio, sample_rate=16000):
    """Extract voice features"""
    # Compute spectrogram
    frequencies, times, spectrogram = signal.spectrogram(
        audio, fs=sample_rate, nperseg=512, noverlap=256
    )
    
    # Extract features
    features = {
        'mean_spectral': np.mean(spectrogram, axis=1),
        'spectral_centroid': np.sum(frequencies[:, None] * spectrogram, axis=0) / (np.sum(spectrogram, axis=0) + 1e-10),
        'rms_energy': np.sqrt(np.mean(audio**2)),
        'zero_crossing_rate': np.mean(np.abs(np.diff(np.sign(audio))) / 2)
    }
    
    # Aggregate for comparison
    return {
        'mean_spectral': features['mean_spectral'],
        'mean_centroid': np.mean(features['spectral_centroid']),
        'mean_rms': features['rms_energy'],
        'mean_zcr': features['zero_crossing_rate']
    }

def update_statistics(existing_stats, new_features):
    """Update running statistics with new sample"""
    # For mean: new_mean = (old_mean * n + new_value) / (n + 1)
    # For std: more complex, but we'll recompute from scratch for simplicity
    
    # We don't have individual samples, so we'll just update means
    # This is a simplified approach - in production you'd store samples
    
    n = existing_stats.get('samples_used', 20)  # Assuming 20 from comprehensive enrollment
    
    # Update means
    existing_stats['mean_centroid'] = (existing_stats['mean_centroid'] * n + new_features['mean_centroid']) / (n + 1)
    existing_stats['mean_rms'] = (existing_stats['mean_rms'] * n + new_features['mean_rms']) / (n + 1)
    existing_stats['mean_zcr'] = (existing_stats['mean_zcr'] * n + new_features['mean_zcr']) / (n + 1)
    
    # Update spectral mean (more complex - approximate)
    existing_mean = existing_stats['mean_spectral']
    new_mean = new_features['mean_spectral']
    
    # Ensure same length
    min_len = min(len(existing_mean), len(new_mean))
    existing_stats['mean_spectral'] = (existing_mean[:min_len] * n + new_mean[:min_len]) / (n + 1)
    
    # Update sample count
    existing_stats['samples_used'] = n + 1
    
    return existing_stats

def main():
    voiceprint_file = 'voice/enrollment_data/master_voiceprint.pkl'
    
    # Load existing voiceprint
    if not os.path.exists(voiceprint_file):
        print("❌ No existing voiceprint found. Run enroll_master_comprehensive.py first.")
        return
    
    with open(voiceprint_file, 'rb') as f:
        voiceprint = pickle.load(f)
    
    current_samples = voiceprint.get('samples_used', 20)
    print(f"📊 Current voiceprint based on {current_samples} samples")
    print("\n🎤 Let's add 10 more samples to improve recognition.")
    print("Speak naturally each time - vary your tone slightly.")
    
    sample_rate = 16000
    duration = 3
    added = 0
    
    for i in range(10):
        input(f"\nPress Enter and say phrase {i+1}/10 (any phrase)...")
        
        print("🎤 Recording...")
        recording = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='float32')
        sd.wait()
        
        # Extract features
        features = extract_features(recording.flatten(), sample_rate)
        
        # Update statistics
        voiceprint = update_statistics(voiceprint, features)
        added += 1
        
        print(f"✅ Sample {i+1}/10 added and statistics updated")
    
    # Save updated voiceprint
    with open(voiceprint_file, 'wb') as f:
        pickle.dump(voiceprint, f)
    
    print(f"\n✅ SUCCESS! Voiceprint now based on {voiceprint['samples_used']} total samples")
    print("Try saying 'Hey Dee Mai' again - recognition should be improved!")

if __name__ == "__main__":
    main()
