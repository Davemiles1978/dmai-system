#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
import sounddevice as sd
import time
from pathlib import Path

SAMPLE_RATE = 16000

def extract_features(audio):
    """Extract features that match what's in your voiceprint"""
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    
    # Basic features that ARE in your voiceprint
    rms = float(np.sqrt(np.mean(audio**2)))
    zcr = float(np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio))
    
    return {
        'rms': rms,
        'zero_crossing_rate': zcr,
        'audio': audio  # Keep for additional processing if needed
    }

def compare_with_voiceprint(current_features, voiceprint):
    """Compare current speech with enrolled voiceprint"""
    best_score = 0
    best_match = None
    all_scores = []
    
    current_rms = current_features['rms']
    current_zcr = current_features['zero_crossing_rate']
    
    for i, saved in enumerate(voiceprint['features']):
        # Calculate RMS similarity (0-100%)
        rms_diff = abs(current_rms - saved['rms']) / (saved['rms'] + 1e-10)
        rms_sim = max(0, 100 - (rms_diff * 100))
        
        # Calculate ZCR similarity (0-100%)
        zcr_diff = abs(current_zcr - saved['zero_crossing_rate']) / (saved['zero_crossing_rate'] + 1e-10)
        zcr_sim = max(0, 100 - (zcr_diff * 100))
        
        # Weighted score (RMS 60%, ZCR 40%)
        score = (rms_sim * 0.6) + (zcr_sim * 0.4)
        all_scores.append(score)
        
        if score > best_score:
            best_score = score
            best_match = {
                'index': i,
                'style': saved.get('style', 'unknown'),
                'rms_sim': rms_sim,
                'zcr_sim': zcr_sim,
                'score': score
            }
    
    avg_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    
    return best_score, best_match, avg_score, std_score

def main():
    print("\n🎤 DMAI Biometric Voice Test (Fixed)")
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
    
    print("\n📋 Say one of these phrases from enrollment:")
    print("  1. 'my voice is my password' (normal)")
    print("  2. 'dmai recognize my voice' (normal)")
    print("  3. 'this is david speaking' (normal)")
    print("  4. 'whisper this quietly' (quiet)")
    print("  5. 'THIS IS IMPORTANT' (loud)")
    print("  6. 'are you there DMAI?' (question)")
    
    choice = input("\nChoose phrase (1-6) or press Enter for any: ").strip()
    
    # Map choice to phrase text for reference
    phrase_map = {
        '1': "my voice is my password",
        '2': "dmai recognize my voice", 
        '3': "this is david speaking",
        '4': "whisper this quietly",
        '5': "THIS IS IMPORTANT",
        '6': "are you there DMAI?"
    }
    
    if choice in phrase_map:
        print(f"\n💬 Say exactly: \"{phrase_map[choice]}\"")
    
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
    
    # Check recording quality
    max_amp = np.max(np.abs(audio))
    if max_amp < 0.01:
        print("\n⚠️ Recording is very quiet - check microphone and speak louder")
    elif max_amp > 0.9:
        print("\n⚠️ Recording is clipping - move microphone further away")
    else:
        print(f"\n✅ Good recording level (peak: {max_amp:.3f})")
    
    # Extract features
    current_features = extract_features(audio)
    
    print(f"\n📊 Your voice metrics:")
    print(f"   Volume (RMS): {current_features['rms']:.4f}")
    print(f"   Zero Crossing Rate: {current_features['zero_crossing_rate']:.4f}")
    
    # Compare with voiceprint
    best_score, best_match, avg_score, std_score = compare_with_voiceprint(current_features, voiceprint)
    
    print("\n📊 Recognition Results:")
    print(f"   Best match: {best_score:.1f}%")
    print(f"   Average: {avg_score:.1f}%")
    print(f"   Consistency: {100 - std_score:.1f}%")
    
    if best_match:
        print(f"\n🎭 Matched style: {best_match['style']}")
        print(f"   RMS similarity: {best_match['rms_sim']:.1f}%")
        print(f"   ZCR similarity: {best_match['zcr_sim']:.1f}%")
    
    # Decision logic adjusted for simpler features
    if best_score >= 70:
        print("\n✅✅✅ VOICE VERIFIED - HIGH CONFIDENCE")
        print("   Access granted to all systems")
    elif best_score >= 60:
        print("\n✅ VOICE VERIFIED - MEDIUM CONFIDENCE")
        print("   Access granted to basic functions")
    elif best_score >= 50:
        print("\n⚠️ PARTIAL MATCH - LOW CONFIDENCE")
        print("   Additional verification required")
    else:
        print("\n❌ VOICE NOT RECOGNIZED")
        print("\nTips for better recognition:")
        print("• Speak at the same volume as during enrollment")
        print(f"  (Your enrolled volumes range from 0.08 to 0.11)")
        print(f"  (Current volume: {current_features['rms']:.4f})")
        print("• Use exact phrases from enrollment")
        print("• Wait for the SPEAK NOW prompt")
        print("• Reduce background noise")
    
    # Show enrolled volume ranges for reference
    print("\n📊 Enrolled volume ranges by style:")
    for style, data in voiceprint['style_profiles'].items():
        print(f"   {style:8}: {data['avg_rms']:.4f} ± {data['std_rms']:.4f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest cancelled")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
