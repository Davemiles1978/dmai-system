"""
DMAI MusicGenerator — Pure Python music/audio generation.
No external dependencies beyond wave/struct (stdlib).
Generates: melodies, chord progressions, drum patterns, song structures.

DMAI's first step toward becoming a musician.
"""

import wave
import struct
import math
import random
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional


# ── Music Theory Constants ────────────────────────────────────────────────
NOTES = {
    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
    'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
    'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
}

SCALES = {
    'major':        ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
    'minor':        ['C', 'D', 'D#', 'F', 'G', 'G#', 'A#'],
    'pentatonic':   ['C', 'D', 'E', 'G', 'A'],
    'blues':        ['C', 'D#', 'F', 'F#', 'G', 'A#'],
    'dorian':       ['C', 'D', 'D#', 'F', 'G', 'A', 'A#'],
    'phrygian':     ['C', 'C#', 'D#', 'F', 'G', 'G#', 'A#'],
    'lydian':       ['C', 'D', 'E', 'F#', 'G', 'A', 'B'],
}

CHORD_TEMPLATES = {
    'major':    [0, 4, 7],
    'minor':    [0, 3, 7],
    'dim':      [0, 3, 6],
    'aug':      [0, 4, 8],
    'sus2':     [0, 2, 7],
    'sus4':     [0, 5, 7],
    'maj7':     [0, 4, 7, 11],
    'min7':     [0, 3, 7, 10],
    'dom7':     [0, 4, 7, 10],
}

COMMON_PROGRESSIONS = {
    'pop':      [[0, 'major'], [5, 'major'], [3, 'minor'], [4, 'major']],       # I-V-vi-IV
    'jazz':     [[2, 'min7'], [5, 'dom7'], [0, 'maj7'], [0, 'maj7']],           # ii-V-I-I
    'blues':    [[0, 'dom7'], [3, 'dom7'], [0, 'dom7'], [0, 'dom7']],           # I7-IV7-I7-I7
    'rock':     [[0, 'major'], [4, 'major'], [0, 'major'], [5, 'major']],        # I-IV-I-V
    'ambient':  [[0, 'maj7'], [3, 'min7'], [5, 'maj7'], [4, 'maj7']],            # Imaj7-iiim7-Vmaj7-IVmaj7
    'electronic': [[0, 'minor'], [5, 'minor'], [3, 'minor'], [4, 'minor']],      # i-v-iii-iv
}

DRUM_PATTERNS = {
    'basic_rock':   [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],  # Kick on every beat
    'four_on_floor':[1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],  # Every 16th
    'hiphop':       [1,0,0,0, 0,1,0,0, 1,0,0,0, 0,0,1,0],
    'halftime':     [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,1,0],
    'electronic':   [1,0,1,1, 0,1,1,0, 1,0,1,1, 0,1,1,0],
}

DRUM_SAMPLES = {
    'kick':   {'freq': 60,  'duration': 0.08, 'shape': 'sine_decay'},
    'snare':  {'freq': 200, 'duration': 0.06, 'shape': 'noise_decay'},
    'hihat':  {'freq': 8000,'duration': 0.03, 'shape': 'noise_short'},
    'clap':   {'freq': 500, 'duration': 0.04, 'shape': 'noise_decay'},
}


class Synth:
    """Basic synthesizer — generates waveforms from scratch."""
    
    SAMPLE_RATE = 44100
    
    @staticmethod
    def sine(t: float, freq: float) -> float:
        return math.sin(2 * math.pi * freq * t)
    
    @staticmethod
    def square(t: float, freq: float) -> float:
        return 1.0 if math.sin(2 * math.pi * freq * t) >= 0 else -1.0
    
    @staticmethod
    def sawtooth(t: float, freq: float) -> float:
        return 2.0 * (t * freq - math.floor(t * freq + 0.5))
    
    @staticmethod
    def triangle(t: float, freq: float) -> float:
        return 2.0 * abs(2.0 * (t * freq - math.floor(t * freq + 0.5))) - 1.0
    
    @staticmethod
    def noise() -> float:
        return random.uniform(-0.3, 0.3)


class DrumSynth:
    """Synthesize drum sounds."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
    
    def kick(self) -> List[float]:
        """Synthesized kick drum — sine sweep down."""
        dur = 0.15
        samples = int(self.sr * dur)
        result = []
        for i in range(samples):
            t = i / self.sr
            freq = 150 - (t / dur) * 130  # Sweep 150Hz → 20Hz
            amp = 1.0 - (t / dur) * 0.8   # Decay
            result.append(math.sin(2 * math.pi * freq * t) * amp * 0.8)
        return result
    
    def snare(self) -> List[float]:
        """Synthesized snare — noise burst + tone."""
        dur = 0.1
        samples = int(self.sr * dur)
        result = []
        for i in range(samples):
            t = i / self.sr
            amp = 1.0 - (t / dur)
            noise = random.uniform(-1, 1) * amp * 0.5
            tone = math.sin(2 * math.pi * 200 * t) * amp * 0.3
            result.append(noise + tone)
        return result
    
    def hihat(self) -> List[float]:
        """Synthesized hi-hat — high-frequency noise."""
        dur = 0.05
        samples = int(self.sr * dur)
        result = []
        for i in range(samples):
            t = i / self.sr
            amp = (1.0 - (t / dur)) * 0.4
            result.append(random.uniform(-1, 1) * amp)
        return result


class MusicGenerator:
    """
    Generate complete songs: melody + chords + drums.
    Pure Python, no external audio libraries.
    Outputs WAV files.
    """
    
    SAMPLE_RATE = 44100
    BPM = 120
    
    def __init__(self, output_dir: str = "data/generated_content"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.synth = Synth()
        self.drums = DrumSynth(self.SAMPLE_RATE)
        self.generation_count = 0
    
    def _seconds_to_samples(self, seconds: float) -> int:
        return int(self.SAMPLE_RATE * seconds)
    
    def _beat_duration(self) -> float:
        return 60.0 / self.BPM
    
    def _note_freq(self, note_name: str, octave: int = 4) -> float:
        """Convert note name + octave to frequency."""
        base = NOTES.get(note_name, 440.0)
        return base * (2 ** (octave - 4))
    
    def _render_note(self, freq: float, duration: float, waveform: str = 'sine',
                     velocity: float = 0.7) -> List[float]:
        """Render a single note as samples."""
        samples = self._seconds_to_samples(duration)
        result = []
        for i in range(samples):
            t = i / self.SAMPLE_RATE
            amp = velocity * (1.0 - (i / samples) * 0.3)  # Gentle decay
            amp *= 0.5 + 0.5 * math.sin(math.pi * i / samples)  # ADSR-like envelope
            
            if waveform == 'sine':
                val = self.synth.sine(t, freq)
            elif waveform == 'square':
                val = self.synth.square(t, freq)
            elif waveform == 'sawtooth':
                val = self.synth.sawtooth(t, freq)
            elif waveform == 'triangle':
                val = self.synth.triangle(t, freq)
            else:
                val = self.synth.sine(t, freq)
            
            result.append(val * amp)
        return result
    
    def _silence(self, duration: float) -> List[float]:
        return [0.0] * self._seconds_to_samples(duration)
    
    def _mix(self, *tracks: List[float]) -> List[float]:
        """Mix multiple tracks together (additive mixing with limiting)."""
        if not tracks:
            return []
        max_len = max(len(t) for t in tracks)
        result = [0.0] * max_len
        for track in tracks:
            for i, val in enumerate(track):
                if i < max_len:
                    result[i] += val
        # Soft limiter
        return [max(-0.95, min(0.95, v / max(1, len(tracks) * 0.7))) for v in result]
    
    def generate_melody(self, scale_name: str = 'major', octave: int = 4,
                        num_notes: int = 16, waveform: str = 'sine') -> List[float]:
        """Generate a melody using notes from the given scale."""
        scale = SCALES.get(scale_name, SCALES['major'])
        beat_dur = self._beat_duration()
        note_dur = beat_dur * 0.9  # Slight gap between notes
        
        result = []
        for i in range(num_notes):
            # Pick a note from the scale with melodic preference
            if i == 0:
                note_idx = 0  # Start on root
            else:
                # Prefer small intervals
                prev_idx = note_idx
                note_idx = prev_idx + random.choice([-2, -1, 1, 2, 0, 3, -3])
                note_idx = max(0, min(len(scale) - 1, note_idx))
            
            note_name = scale[note_idx]
            note_octave = octave + (1 if note_idx >= len(scale) - 3 else 0)
            freq = self._note_freq(note_name, note_octave)
            
            result.extend(self._render_note(freq, note_dur, waveform, velocity=0.6))
            result.extend(self._silence(beat_dur * 0.1))  # Gap
        
        return result
    
    def generate_chords(self, progression_style: str = 'pop', octave: int = 3,
                        bars: int = 4) -> List[float]:
        """Generate chord progression."""
        scale = SCALES['major']
        progression = COMMON_PROGRESSIONS.get(progression_style, COMMON_PROGRESSIONS['pop'])
        beat_dur = self._beat_duration()
        chord_dur = beat_dur * 4  # One chord per bar
        
        result = []
        for bar in range(bars):
            chord_info = progression[bar % len(progression)]
            root_idx, chord_type = chord_info
            root_note = scale[root_idx % len(scale)]
            template = CHORD_TEMPLATES.get(chord_type, CHORD_TEMPLATES['major'])
            
            # Render chord as overlapping notes
            chord_samples = []
            for interval in template:
                note_idx = (root_idx + (interval // 2)) % len(scale)
                note_name = scale[note_idx]
                freq = self._note_freq(note_name, octave + (interval // 7))
                note_samples = self._render_note(freq, chord_dur * 0.95, 'sine', velocity=0.35)
                chord_samples.append(note_samples)
            
            result.extend(self._mix(*chord_samples))
            result.extend(self._silence(beat_dur * 0.1))
        
        return result
    
    def generate_drums(self, pattern_name: str = 'basic_rock', bars: int = 4) -> List[float]:
        """Generate drum track."""
        pattern = DRUM_PATTERNS.get(pattern_name, DRUM_PATTERNS['basic_rock'])
        beat_dur = self._beat_duration()
        sixteenth = beat_dur / 4
        
        result = []
        for bar in range(bars):
            for i, hit in enumerate(pattern):
                if hit:
                    if i % 4 == 0:  # Downbeat
                        drum = self.drums.kick()
                    elif i % 4 == 2:  # Backbeat
                        drum = self.drums.snare()
                    else:
                        drum = self.drums.hihat()
                    
                    # Trim or pad to sixteenth duration
                    target_len = self._seconds_to_samples(sixteenth)
                    if len(drum) < target_len:
                        drum.extend([0.0] * (target_len - len(drum)))
                    result.extend(drum[:target_len])
                else:
                    result.extend([0.0] * self._seconds_to_samples(sixteenth))
        
        return result
    
    def generate_bassline(self, progression_style: str = 'pop', octave: int = 2,
                          bars: int = 4) -> List[float]:
        """Generate a simple bassline following the chord progression."""
        scale = SCALES['major']
        progression = COMMON_PROGRESSIONS.get(progression_style, COMMON_PROGRESSIONS['pop'])
        beat_dur = self._beat_duration()
        
        result = []
        for bar in range(bars):
            chord_info = progression[bar % len(progression)]
            root_idx = chord_info[0]
            root_note = scale[root_idx % len(scale)]
            freq = self._note_freq(root_note, octave)
            
            # Simple bass pattern: root on beats 1 and 3, fifth on beats 2 and 4
            for beat in range(4):
                if beat % 2 == 0:
                    note_freq = freq
                else:
                    # Fifth above root
                    fifth_idx = (root_idx + 4) % len(scale)
                    note_freq = self._note_freq(scale[fifth_idx], octave)
                
                note_samples = self._render_note(note_freq, beat_dur * 0.7, 'sawtooth', velocity=0.5)
                result.extend(note_samples)
                result.extend(self._silence(beat_dur * 0.3))
        
        return result
    
    def generate_song(self, prompt: str = "", style: str = "pop",
                      duration_bars: int = 16, bpm: int = 120) -> Dict:
        """
        Generate a complete song with melody, chords, bass, and drums.
        Returns {ok, file, filename, view_url, duration_seconds}.
        """
        self.BPM = bpm
        
        # Determine musical style from prompt or parameter
        pl = prompt.lower()
        if 'jazz' in pl: prog_style, drum_style, scale = 'jazz', 'halftime', 'dorian'
        elif 'rock' in pl: prog_style, drum_style, scale = 'rock', 'basic_rock', 'minor'
        elif 'electronic' in pl or 'dance' in pl: prog_style, drum_style, scale = 'electronic', 'electronic', 'minor'
        elif 'ambient' in pl: prog_style, drum_style, scale = 'ambient', 'halftime', 'lydian'
        elif 'hiphop' in pl or 'hip hop' in pl: prog_style, drum_style, scale = 'pop', 'hiphop', 'blues'
        elif 'blues' in pl: prog_style, drum_style, scale = 'blues', 'halftime', 'blues'
        else:
            style_map = {
                'pop': ('pop', 'basic_rock', 'major'),
                'rock': ('rock', 'basic_rock', 'minor'),
                'jazz': ('jazz', 'halftime', 'dorian'),
                'classical': ('ambient', 'halftime', 'major'),
                'electronic': ('electronic', 'electronic', 'minor'),
                'lo_fi': ('ambient', 'hiphop', 'major'),
            }
            prog_style, drum_style, scale = style_map.get(style, ('pop', 'basic_rock', 'major'))
        
        # Generate tracks
        melody = self.generate_melody(scale_name=scale, num_notes=duration_bars * 4)
        chords = self.generate_chords(progression_style=prog_style, bars=duration_bars)
        bass = self.generate_bassline(progression_style=prog_style, bars=duration_bars)
        drums = self.generate_drums(pattern_name=drum_style, bars=duration_bars)
        
        # Mix all tracks
        final = self._mix(melody, chords, bass, drums)
        
        # Add fade out
        fade_samples = self._seconds_to_samples(2.0)
        fade_start = max(0, len(final) - fade_samples)
        for i in range(fade_start, len(final)):
            alpha = (len(final) - i) / fade_samples
            final[i] *= alpha
        
        # Write WAV
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"dma_song_{timestamp}_{self.generation_count}.wav"
        filepath = self.output_dir / filename
        
        with wave.open(str(filepath), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.SAMPLE_RATE)
            
            # Convert float to 16-bit int
            int_samples = [int(max(-32767, min(32767, s * 32767))) for s in final]
            wf.writeframes(struct.pack('<' + 'h' * len(int_samples), *int_samples))
        
        duration = len(final) / self.SAMPLE_RATE
        self.generation_count += 1
        
        return {
            "ok": True,
            "file": str(filepath),
            "filename": filename,
            "view_url": f"/api/content/view/{filename}",
            "duration_seconds": round(duration, 1),
            "bpm": self.BPM,
            "style": style,
            "scale": scale,
            "progression": prog_style,
            "drum_style": drum_style,
            "prompt": prompt,
            "generator": "DMAI MusicGenerator",
        }


# ── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("DMAI MusicGenerator — Self Test")
    print("=" * 50)
    
    mg = MusicGenerator()
    
    styles = ['pop', 'rock', 'jazz', 'electronic', 'lo_fi']
    for style in styles:
        result = mg.generate_song(
            prompt=f"A {style} song",
            style=style,
            duration_bars=8,
            bpm=100 if style == 'lo_fi' else 120
        )
        print(f"  {style:12s}: {result['filename']} — {result['duration_seconds']}s, {result['bpm']} BPM")
    
    print("=" * 50)
    print("All songs generated. DMAI can now compose original music.")
