#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Comprehensive voice enrollment - captures different tones and volumes"""
import sys
import os
import sounddevice as sd
import numpy as np
import time
import json
import pickle
from pathlib import Path
import hashlib
from scipy import signal
from scipy.io import wavfile

SAMPLE_RATE = 16000

# Phrases in different styles
ENROLLMENT_PHRASES = [
    # Normal conversational (5 phrases)
    {"text": "my voice is my password", "style": "normal"},
    {"text": "dmai recognize my voice", "style": "normal"},
    {"text": "this is david speaking", "style": "normal"},
    {"text": "access granted to master", "style": "normal"},
    {"text": "I am the creator of DMAI", "style": "normal"},
    
    # Quiet/soft (3 phrases)
    {"text": "whisper this quietly", "style": "quiet"},
    {"text": "speaking in a low voice", "style": "quiet"},
    {"text": "soft and gentle", "style": "quiet"},
    
    # Loud/emphatic (3 phrases)
    {"text": "THIS IS IMPORTANT", "style": "loud"},
    {"text": "EMERGENCY OVERRIDE", "style": "loud"},
    {"text": "COMMAND EXECUTE NOW", "style": "loud"},
    
    # Question/inflection (3 phrases)
    {"text": "are you there DMAI?", "style": "question"},
    {"text": "can you hear me?", "style": "question"},
    {"text": "what's my status?", "style": "question"},
    
    # Command style (3 phrases)
    {"text": "create a video", "style": "command"},
    {"text": "research quantum physics", "style": "command"},
    {"text": "send to my phone", "style": "command"},
    
    # Random extra (3 phrases)
    {"text": "the quick brown fox", "style": "random"},
    {"text": "hello there general kenobi", "style": "random"},
    {"text": "to be or not to be", "style": "random"}
]

class SimpleVoiceAuth:
    """Simplified voice authentication for enrollment"""
    
    def __init__(self, model_dir="voice_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.user_model_path = self.model_dir / "master_voice.pkl"
        self.user_profile_path = self.model_dir / "voice_profile.json"
        
    def extract_features(self, audio):
        """Extract simple voice features"""
        if len(audio) == 0:
            return None
            
        # Normalize audio
        audio = audio / np.max(np.abs(audio) + 1e-10)
        
        # Basic acoustic features
        features = {
            'mean': float(np.mean(audio)),
            'std': float(np.std(audio)),
            'rms': float(np.sqrt(np.mean(audio**2))),
            'zero_crossing_rate': float(np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)),
            'spectral_centroid': float(self._spectral_centroid(audio)),
            'mfcc_like': self._simple_mfcc(audio, n_coeffs=13),
            'duration': len(audio) / SAMPLE_RATE,
            'energy': float(np.sum(audio**2))
        }
        
        return features
    
    def _spectral_centroid(self, audio):
        """Calculate spectral centroid"""
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/SAMPLE_RATE)
        if np.sum(spectrum) == 0:
            return 0
        return float(np.sum(freqs * spectrum) / np.sum(spectrum))
    
    def _simple_mfcc(self, audio, n_coeffs=13):
        """Simplified MFCC-like features"""
        # Segment into frames
        frame_length = int(0.025 * SAMPLE_RATE)  # 25ms
        hop_length = int(0.010 * SAMPLE_RATE)    # 10ms
        
        frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            # Apply hamming window and compute spectrum
            windowed = frame * np.hamming(len(frame))
            spectrum = np.abs(np.fft.rfft(windowed))
            # Take log spectrum (simplified)
            log_spectrum = np.log(spectrum + 1e-10)
            # Downsample to n_coeffs
            if len(log_spectrum) >= n_coeffs:
                indices = np.linspace(0, len(log_spectrum)-1, n_coeffs, dtype=int)
                frames.append(log_spectrum[indices])
        
        if not frames:
            return [0.0] * n_coeffs
        
        # Average across frames
        return np.mean(frames, axis=0).tolist()
    
    def enroll_master(self, recordings, sample_rate):
        """Enroll user with multiple recordings"""
        print("Processing voice samples...")
        
        all_features = []
        style_stats = {}
        
        for i, audio in enumerate(recordings):
            # Extract features
            features = self.extract_features(audio)
            if features:
                # Add style info from index
                style = ENROLLMENT_PHRASES[i % len(ENROLLMENT_PHRASES)]['style']
                features['style'] = style
                features['phrase_index'] = i
                all_features.append(features)
                
                # Track style statistics
                if style not in style_stats:
                    style_stats[style] = []
                style_stats[style].append(features['rms'])
        
        if len(all_features) < 5:  # Need minimum samples
            print(f"❌ Only got {len(all_features)} valid samples, need at least 5")
            return False
        
        # Create master voiceprint
        voiceprint = {
            'features': all_features,
            'style_profiles': {},
            'timestamp': time.time(),
            'user_id': 'primary_user',
            'sample_count': len(all_features)
        }
        
        # Create style-specific profiles
        for style, rms_values in style_stats.items():
            if rms_values:
                voiceprint['style_profiles'][style] = {
                    'avg_rms': float(np.mean(rms_values)),
                    'std_rms': float(np.std(rms_values)),
                    'samples': len(rms_values)
                }
        
        # Create average feature vector for quick matching
        feature_names = ['mean', 'std', 'rms', 'zero_crossing_rate', 'spectral_centroid']
        avg_features = {}
        for name in feature_names:
            values = [f[name] for f in all_features]
            avg_features[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        voiceprint['avg_features'] = avg_features
        
        # Calculate MFCC averages
        mfcc_values = np.array([f['mfcc_like'] for f in all_features])
        voiceprint['avg_mfcc'] = np.mean(mfcc_values, axis=0).tolist()
        voiceprint['std_mfcc'] = np.std(mfcc_values, axis=0).tolist()
        
        # Save voiceprint
        try:
            with open(self.user_model_path, 'wb') as f:
                pickle.dump(voiceprint, f)
            
            # Also save human-readable profile
            profile = {
                'enrolled': time.ctime(),
                'samples': len(all_features),
                'styles_enrolled': list(style_stats.keys()),
                'quality': 'high' if len(all_features) > 15 else 'medium',
                'status': 'active'
            }
            
            with open(self.user_profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            # Also save raw audio for backup
            audio_dir = self.model_dir / "recordings"
            audio_dir.mkdir(exist_ok=True)
            
            for i, audio in enumerate(recordings):
                # Convert to int16 for wav file
                audio_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(audio_dir / f"sample_{i}.wav", SAMPLE_RATE, audio_int16)
            
            print(f"✅ Voiceprint saved to {self.user_model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving voiceprint: {e}")
            return False
    
    def verify(self, audio):
        """Simple verification (for testing)"""
        if not self.user_model_path.exists():
            return False, 0
        
        with open(self.user_model_path, 'rb') as f:
            voiceprint = pickle.load(f)
        
        features = self.extract_features(audio)
        if not features:
            return False, 0
        
        # Simple similarity score (you'd want something more sophisticated)
        scores = []
        for saved in voiceprint['features']:
            # Compare RMS and spectral centroid
            rms_diff = abs(features['rms'] - saved['rms']) / (saved['rms'] + 1e-10)
            spec_diff = abs(features['spectral_centroid'] - saved['spectral_centroid']) / (saved['spectral_centroid'] + 1e-10)
            
            # Simple score (0-100)
            score = max(0, 100 - (rms_diff * 50 + spec_diff * 50))
            scores.append(score)
        
        avg_score = np.mean(scores)
        return avg_score > 70, avg_score

def record_phrase(phrase_data, index, total):
    """Record a single phrase with style instructions"""
    phrase = phrase_data["text"]
    style = phrase_data["style"]
    
    print(f"\n📝 Phrase {index}/{total}")
    print(f"🎤 Style: {style.upper()}")
    print(f"🗣️  Say: \"{phrase}\"")
    
    # Style-specific instructions
    if style == "quiet":
        print("   🤫 Speak softly, almost whispering")
    elif style == "loud":
        print("   🔊 Speak with emphasis, louder than normal")
    elif style == "question":
        print("   ❓ Use a rising inflection at the end")
    elif style == "command":
        print("   ⚡ Firm, direct tone")
    else:
        print("   🗣️  Your natural speaking voice")
    
    print("\nGet ready...")
    
    for i in [3, 2, 1]:
        print(f"   {i}...")
        time.sleep(1)
    
    print("🔴 SPEAK NOW!")
    
    # Record for 3 seconds
    recording = sd.rec(int(3 * SAMPLE_RATE), 
                       samplerate=SAMPLE_RATE, 
                       channels=1, 
                       dtype='float32')
    sd.wait()
    
    print("✅ Recorded!")
    
    # Check if recording was too quiet
    if np.max(np.abs(recording)) < 0.01:
        print("⚠️  Recording was very quiet - check microphone")
    elif np.max(np.abs(recording)) > 0.9:
        print("⚠️  Recording was very loud - possible clipping")
    else:
        print("✓ Good volume level")
    
    time.sleep(1)  # Pause between phrases
    
    return recording.flatten()

def check_microphone():
    """Check if microphone is working"""
    print("\n🎤 Testing microphone...")
    
    try:
        # List available devices
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        
        if default_input is None or default_input < 0:
            print("❌ No default input device found")
            print("\nAvailable devices:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"  [{i}] {device['name']}")
            
            # Try to set first available input device
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    sd.default.device = i
                    print(f"✅ Selected device: {device['name']}")
                    break
        else:
            device_info = devices[default_input]
            print(f"✅ Using microphone: {device_info['name']}")
        
        # Quick test recording
        print("\nTesting recording for 2 seconds...")
        test_rec = sd.rec(int(2 * SAMPLE_RATE), 
                         samplerate=SAMPLE_RATE, 
                         channels=1, 
                         dtype='float32')
        sd.wait()
        
        max_level = np.max(np.abs(test_rec))
        if max_level < 0.001:
            print("❌ Microphone test failed - very low signal")
            return False
        else:
            print(f"✅ Microphone working (peak level: {max_level:.3f})")
            return True
            
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("🎙️  DMAI COMPREHENSIVE VOICE ENROLLMENT")
    print("="*60)
    
    # Check microphone first
    if not check_microphone():
        print("\n❌ Microphone not working properly")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    print(f"\nYou'll speak {len(ENROLLMENT_PHRASES)} phrases in different styles:")
    print("  • Normal (5) - your everyday voice")
    print("  • Quiet (3) - soft/whisper")
    print("  • Loud (3) - emphatic/louder")
    print("  • Question (3) - rising inflection")
    print("  • Command (3) - firm/direct")
    print("  • Random (3) - varied")
    print(f"\nTotal time: ~{len(ENROLLMENT_PHRASES) * 5} seconds of speaking")
    
    input("\nPress Enter when ready to start...")
    
    # Initialize auth
    auth = SimpleVoiceAuth()
    recordings = []
    
    # Record each phrase
    for i, phrase_data in enumerate(ENROLLMENT_PHRASES, 1):
        try:
            audio = record_phrase(phrase_data, i, len(ENROLLMENT_PHRASES))
            recordings.append(audio)
            print(f"✓ Phrase {i} stored")
        except KeyboardInterrupt:
            print("\n\n⚠️ Enrollment interrupted by user")
            break
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            print("Let's try that again...")
            try:
                audio = record_phrase(phrase_data, i, len(ENROLLMENT_PHRASES))
                recordings.append(audio)
            except:
                print("Failed again, skipping this phrase")
    
    if len(recordings) < 5:
        print(f"\n❌ Not enough recordings ({len(recordings)}). Need at least 5.")
        return
    
    print("\n" + "="*60)
    print("🧬 Creating comprehensive voiceprint...")
    
    # Create voiceprint
    success = auth.enroll_master(recordings, SAMPLE_RATE)
    
    if success:
        print("\n✅ SUCCESS! DMAI now knows your voice across different tones!")
        print(f"   Used {len(recordings)} samples")
        print(f"   Models saved in: {auth.model_dir.absolute()}")
        
        # Test verification
        print("\n🔍 Testing verification with last sample...")
        score, match = auth.verify(recordings[-1])
        print(f"   Match confidence: {score:.1f}%")
        
    else:
        print("\n❌ Enrollment failed. Please try again.")
    
    print("\n" + "="*60)
    print("To use this voice model with DMAI, run:")
    print(f"  cp -r {auth.model_dir.absolute()}/* /path/to/dmai/voice/models/")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Enrollment cancelled")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
