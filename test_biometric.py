#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
import sounddevice as sd
import time
from scipy import signal
import json
from pathlib import Path

SAMPLE_RATE = 16000

def extract_features(audio):
    """Extract features for comparison"""
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    
    # Basic features
    rms = np.sqrt(np.mean(audio**2))
    zcr = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)
    
    # Spectral features
    freqs, psd = signal.welch(audio, SAMPLE_RATE)
    spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
    spectral_rolloff = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]
    
    # Energy in frequency bands
    low_band = np.sum(psd[(freqs >= 0) & (freqs < 300)])
    mid_band = np.sum(psd[(freqs >= 300) & (freqs < 2000)])
    high_band = np.sum(psd[(freqs >= 2000)])
    
    return {
        'rms': float(rms),
        'zcr': float(zcr),
        'spectral_centroid': float(spectral_centroid),
        'spectral_rolloff': float(spectral_rolloff),
        'low_energy': float(low_band),
        'mid_energy': float(mid_band),
        'high_energy': float(high_band)
    }

def compare_features(f1, f2):
    """Compare two feature sets and return similarity score"""
    weights = {
        'rms': 0.25,
        'zcr': 0.15,
        'spectral_centroid': 0.20,
        'spectral_rolloff': 0.15,
        'low_energy': 0.08,
        'mid_energy': 0.09,
        'high_energy': 0.08
    }
    
    total_score = 0
    details = {}
    
    for feature, weight in weights.items():
        if f1[feature] == 0:
            diff = 1.0
        else:
            diff = min(abs(f1[feature] - f2[feature]) / f1[feature], 1.0)
        
        similarity = (1 - diff) * 100
        details[feature] = similarity
        total_score += similarity * weight
    
    return total_score, details

def main():
    print("\n🎤 DMAI Biometric Voice Test")
    print("="*50)
    
    # Load voiceprint
    model_path = Path("voice_models/master_voice.pkl")
    if not model_path.exists():
        print("❌ Voice model not found. Run enrollment first.")
        return
    
    with open(model_path, 'rb') as f:
        voiceprint = pickle.load(f)
    
    print(f"✅ Loaded voiceprint with {voiceprint['sample_count']} samples")
    print(f"🎭 Styles: {', '.join(voiceprint['style_profiles'].keys())}")
    
    print("\n📋 Say one of these phrases:")
    print("  1. 'DMAI, it's me David'")
    print("  2. 'Hey DMAI, what's going on?'")
    print("  3. 'My voice is my password'")
    print("  4. 'Access granted'")
    print("  5. 'Random test phrase'")
    
    choice = input("\nChoose phrase (1-5) or press Enter for any: ").strip()
    
    input("\nPress Enter when ready to speak...")
    for i in [3,2,1]:
        print(f"{i}...")
        time.sleep(1)
    
    print("🔴 SPEAK NOW!")
    
    # Record
    recording = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, 
                       channels=1, dtype='float32')
    sd.wait()
    
    audio = recording.flatten()
    
    # Extract features from current speech
    current_features = extract_features(audio)
    
    # Compare with all enrolled samples
    best_score = 0
    best_match = None
    all_scores = []
    
    for i, saved in enumerate(voiceprint['features']):
        # Create feature dict from saved data
        saved_features = {
            'rms': saved['rms'],
            'zcr': saved['zero_crossing_rate'],
            'spectral_centroid': saved['spectral_centroid'],
            'spectral_rolloff': saved['spectral_centroid'] * 0.8,  # Approx
            'low_energy': saved['rms'] * 0.3,  # Approx
            'mid_energy': saved['rms'] * 0.5,  # Approx
            'high_energy': saved['rms'] * 0.2   # Approx
        }
        
        score, details = compare_features(current_features, saved_features)
        all_scores.append(score)
        
        if score > best_score:
            best_score = score
            best_match = {
                'index': i,
                'style': saved.get('style', 'unknown'),
                'score': score,
                'details': details
            }
    
    avg_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    
    print("\n📊 Recognition Results:")
    print(f"   Best match: {best_score:.1f}%")
    print(f"   Average: {avg_score:.1f}%")
    print(f"   Consistency: {100 - std_score:.1f}%")
    
    if best_match:
        print(f"\n🎭 Best match style: {best_match['style']}")
        print("\n📈 Feature breakdown:")
        for feature, score in best_match['details'].items():
            print(f"   {feature:18}: {score:.1f}%")
    
    # Decision logic
    if best_score >= 75:
        print("\n✅✅✅ VOICE VERIFIED - HIGH CONFIDENCE")
        print("   Access granted to all systems")
    elif best_score >= 65:
        print("\n✅ VOICE VERIFIED - MEDIUM CONFIDENCE")
        print("   Access granted to basic functions")
    elif best_score >= 55:
        print("\n⚠️ PARTIAL MATCH - LOW CONFIDENCE")
        print("   Additional verification required")
    else:
        print("\n❌ VOICE NOT RECOGNIZED")
        print("\nTips for better recognition:")
        print("• Speak at same volume as during enrollment")
        print("• Hold microphone at consistent distance")
        print("• Reduce background noise")
        print("• Use similar phrasing to enrollment")
    
    # Save test result
    result = {
        'timestamp': time.time(),
        'best_score': best_score,
        'avg_score': avg_score,
        'matched_style': best_match['style'] if best_match else None,
        'threshold_met': best_score >= 65
    }
    
    with open('voice/auth/last_test.json', 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest cancelled")
    except Exception as e:
        print(f"\nError: {e}")
