#!/usr/bin/env python3
"""Add more voice samples to existing enrollment"""
import pickle
import sounddevice as sd
import numpy as np
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_features(audio, sample_rate=16000):
    """Extract voice features"""
    # Simple spectral features
    from scipy import signal
    import numpy as np
    
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

def main():
    voiceprint_file = 'voice/enrollment_data/master_voiceprint.pkl'
    
    # Load existing voiceprint
    if not os.path.exists(voiceprint_file):
        print("❌ No existing voiceprint found. Run enroll_master_comprehensive.py first.")
        return
    
    with open(voiceprint_file, 'rb') as f:
        voiceprint = pickle.load(f)
    
    print(f"📊 Current voiceprint has {len(voiceprint['samples'])} samples")
    print("\n🎤 Let's add 10 more samples to improve recognition.")
    print("Speak naturally each time - vary your tone slightly.")
    
    sample_rate = 16000
    duration = 3
    
    for i in range(10):
        input(f"\nPress Enter and say phrase {i+1}/10 (any phrase)...")
        
        print("🎤 Recording...")
        recording = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='float32')
        sd.wait()
        
        # Extract and store features
        features = extract_features(recording.flatten(), sample_rate)
        
        # Add to samples
        voiceprint['samples'].append({
            'features': features,
            'timestamp': datetime.now().isoformat(),
            'sample_num': len(voiceprint['samples']) + i + 1
        })
        
        # Update means
        samples = voiceprint['samples']
        voiceprint['mean_spectral'] = np.mean([s['features']['mean_spectral'] for s in samples], axis=0)
        voiceprint['mean_centroid'] = np.mean([s['features']['mean_centroid'] for s in samples])
        voiceprint['mean_rms'] = np.mean([s['features']['mean_rms'] for s in samples])
        voiceprint['mean_zcr'] = np.mean([s['features']['mean_zcr'] for s in samples])
        
        print(f"✅ Sample {i+1}/10 recorded and added")
    
    # Save updated voiceprint
    with open(voiceprint_file, 'wb') as f:
        pickle.dump(voiceprint, f)
    
    print(f"\n✅ SUCCESS! Voiceprint now has {len(voiceprint['samples'])} total samples")
    print("Try saying 'Hey Dee Mai' again - recognition should be improved!")

if __name__ == "__main__":
    main()
