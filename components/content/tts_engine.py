"""
DMAI TTS Engine — Self-contained text-to-speech.
Generates speech WAV files from text using formant synthesis.
Pure Python (wave + math). No external APIs, no pyttsx3, no gTTS.
DMAI speaks with her own voice, generated entirely within her system.

Formant synthesis creates vowel sounds by combining specific frequencies
that mimic the human vocal tract. Consonants are approximated with
noise bursts and transitions.
"""

import wave, math, struct, random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict


# ── Phonetic Mappings ──
VOWELS = {
    'a': [(730, 1.0), (1090, 0.6), (2440, 0.3)],   # "ah"
    'e': [(440, 0.8), (1800, 0.7), (2600, 0.4)],    # "eh"
    'i': [(280, 0.9), (2250, 0.8), (2890, 0.3)],    # "ee"
    'o': [(480, 0.9), (850, 0.5), (2400, 0.2)],     # "oh"
    'u': [(310, 0.9), (870, 0.4), (2250, 0.2)],     # "oo"
    'y': [(280, 0.8), (2100, 0.7), (2800, 0.3)],    # "ih"
}

VOICED_CONSONANTS = {
    'b': [(200, 0.5)], 'd': [(300, 0.6)], 'g': [(250, 0.5)],
    'j': [(300, 0.6)], 'l': [(350, 0.7)], 'm': [(250, 0.9)],
    'n': [(280, 0.9)], 'r': [(400, 0.7)], 'v': [(300, 0.5)],
    'w': [(350, 0.7)], 'z': [(400, 0.4)],
}

UNVOICED_CONSONANTS = {
    'f': 0.4, 'h': 0.2, 'k': 0.5, 'p': 0.3,
    's': 0.6, 't': 0.4, 'th': 0.5, 'sh': 0.6, 'ch': 0.5,
}


class DMAIVoice:
    """DMAI's voice — formant-based speech synthesis."""
    
    SAMPLE_RATE = 22050
    BASE_PITCH = 180  # Hz — feminine voice range
    
    def __init__(self):
        self.speed = 1.0      # Speaking rate multiplier
        self.pitch = 1.0      # Pitch multiplier
        self.volume = 0.8
    
    def _formant_tone(self, freq: float, duration: float, amplitude: float) -> list:
        """Generate a tone at a specific formant frequency."""
        samples = int(self.SAMPLE_RATE * duration)
        result = []
        for i in range(samples):
            t = i / self.SAMPLE_RATE
            # Glottal pulse approximation (buzzy sound source)
            pulse = math.sin(2 * math.pi * self.BASE_PITCH * self.pitch * t)
            pulse = 0.5 if pulse > 0 else -0.5  # Square-ish for richness
            # Apply formant filtering
            val = pulse * math.sin(2 * math.pi * freq * t)
            # Envelope
            env = 1.0
            if i < samples * 0.1:
                env = i / (samples * 0.1)  # Attack
            elif i > samples * 0.8:
                env = (samples - i) / (samples * 0.2)  # Release
            result.append(val * amplitude * env * self.volume)
        return result
    
    def _noise_burst(self, duration: float, amplitude: float, freq_center: float = 2000) -> list:
        """Generate noise for unvoiced consonants."""
        samples = int(self.SAMPLE_RATE * duration)
        result = []
        for i in range(samples):
            t = i / self.SAMPLE_RATE
            noise = random.uniform(-1, 1)
            # Band-pass approximation via AM
            carrier = math.sin(2 * math.pi * freq_center * t)
            val = noise * carrier
            env = 1.0
            if i < samples * 0.2:
                env = i / (samples * 0.2)
            elif i > samples * 0.7:
                env = (samples - i) / (samples * 0.3)
            result.append(val * amplitude * env * self.volume * 0.6)
        return result
    
    def _phoneme_to_audio(self, phoneme: str, duration: float) -> list:
        """Convert a phoneme character to audio samples."""
        result = []
        
        if phoneme in VOWELS:
            formants = VOWELS[phoneme]
            # Mix all formants
            for freq, amp in formants:
                tone = self._formant_tone(freq, duration, amp * 0.35)
                if not result:
                    result = tone
                else:
                    for j in range(min(len(result), len(tone))):
                        result[j] += tone[j]
        
        elif phoneme in VOICED_CONSONANTS:
            formants = VOICED_CONSONANTS[phoneme]
            for freq, amp in formants:
                tone = self._formant_tone(freq, duration * 0.6, amp * 0.3)
                if not result:
                    result = tone
                else:
                    for j in range(min(len(result), len(tone))):
                        result[j] += tone[j]
        
        elif phoneme in UNVOICED_CONSONANTS:
            result = self._noise_burst(duration * 0.7, UNVOICED_CONSONANTS[phoneme])
        
        elif phoneme == ' ':  # Space = pause
            result = [0.0] * int(self.SAMPLE_RATE * duration)
        
        else:
            # Unknown phoneme — short schwa
            result = self._formant_tone(500, duration * 0.3, 0.2)
        
        # Normalize
        if result:
            max_val = max(abs(v) for v in result)
            if max_val > 0.9:
                result = [v * 0.9 / max_val for v in result]
        
        return result if result else [0.0] * int(self.SAMPLE_RATE * 0.05)
    
    def _text_to_phonemes(self, text: str) -> list:
        """Convert text to simplified phoneme sequence."""
        text = text.lower().strip()
        # Simple mapping — each character becomes a phoneme
        phonemes = []
        i = 0
        while i < len(text):
            c = text[i]
            if c in 'aeiou':
                phonemes.append(c)
            elif c in 'bcdfghjklmnpqrstvwxyz':
                # Check for digraphs
                if i + 1 < len(text) and text[i:i+2] in ('th', 'sh', 'ch'):
                    phonemes.append(text[i:i+2])
                    i += 1
                else:
                    phonemes.append(c)
            elif c in '.,!?;:':
                phonemes.append(' ')  # Punctuation = pause
            elif c == ' ':
                phonemes.append(' ')
            i += 1
        return phonemes
    
    def speak(self, text: str) -> list:
        """Convert text to audio samples."""
        phonemes = self._text_to_phonemes(text)
        all_samples = []
        
        # Duration per phoneme (seconds)
        base_dur = 0.08 / self.speed
        
        for ph in phonemes:
            if ph == ' ':
                dur = base_dur * 1.5
            elif ph in VOWELS:
                dur = base_dur * 2.0
            elif ph in VOICED_CONSONANTS:
                dur = base_dur * 1.2
            else:
                dur = base_dur * 0.8
            
            samples = self._phoneme_to_audio(ph, dur)
            all_samples.extend(samples)
        
        return all_samples
    
    def generate_speech(self, text: str, output_dir: str = "data/generated_content") -> Dict:
        """Generate a WAV file of DMAI speaking the given text."""
        samples = self.speak(text)
        
        # Write WAV
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"dma_speech_{ts}.wav"
        filepath = Path(output_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with wave.open(str(filepath), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            int_samples = [int(max(-32767, min(32767, s * 32767))) for s in samples]
            wf.writeframes(struct.pack('<' + 'h' * len(int_samples), *int_samples))
        
        duration = len(samples) / self.SAMPLE_RATE
        
        return {
            "ok": True,
            "file": str(filepath),
            "filename": filename,
            "view_url": f"/api/content/view/{filename}",
            "duration_seconds": round(duration, 1),
            "text": text[:200],
            "word_count": len(text.split()),
            "generator": "DMAI Voice Engine (Formant Synthesis)",
        }


if __name__ == "__main__":
    voice = DMAIVoice()
    print("DMAI Voice Engine ready.")
    result = voice.generate_speech("Hello. I am DMAI. I am learning to speak with my own voice.")
    print(f"  Generated: {result['filename']} ({result['duration_seconds']}s)")
