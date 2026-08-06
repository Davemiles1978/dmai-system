"""
DMAI AudioAnalyser — Self-contained audio analysis.
Analyzes WAV files for BPM, frequency content, and quality metrics.
Pure Python (wave + math). DMAI uses this to evaluate her own music.
"""

import wave, math, struct
from pathlib import Path
from typing import Dict, List


class AudioAnalyser:
    """Analyzes WAV audio files for musical properties."""
    
    def analyse(self, wav_path: str) -> Dict:
        """Full analysis of a WAV file."""
        with wave.open(wav_path, 'r') as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / sample_rate
            
            # Read all samples
            raw = wf.readframes(min(n_frames, sample_rate * 60))  # Max 60s
            fmt = '<' + 'h' * (len(raw) // 2)
            samples = struct.unpack(fmt, raw)
            
            # Convert to mono if stereo
            if n_channels == 2:
                mono = [(samples[i] + samples[i+1]) / 2 for i in range(0, len(samples), 2)]
            else:
                mono = list(samples)
        
        # ── BPM Detection (simple onset detection) ──
        bpm = self._detect_bpm(mono, sample_rate)
        
        # ── Frequency Analysis (simplified FFT approximation) ──
        freq_data = self._frequency_analysis(mono, sample_rate)
        
        # ── Dynamic Range ──
        abs_samples = [abs(s) for s in mono]
        peak = max(abs_samples)
        rms = math.sqrt(sum(s*s for s in mono) / len(mono))
        dynamic_range = 20 * math.log10(peak / rms) if rms > 0 else 0
        
        # ── Silence Ratio ──
        silence_threshold = peak * 0.02
        silence_count = sum(1 for s in abs_samples if s < silence_threshold)
        silence_ratio = silence_count / len(abs_samples)
        
        return {
            "file": wav_path,
            "duration_seconds": round(duration, 1),
            "sample_rate": sample_rate,
            "channels": n_channels,
            "bpm": round(bpm, 1),
            "peak_amplitude": peak,
            "rms_amplitude": round(rms, 1),
            "dynamic_range_db": round(dynamic_range, 1),
            "silence_ratio": round(silence_ratio, 3),
            "dominant_freqs": freq_data.get("dominant_frequencies", [])[:5],
            "spectral_centroid": freq_data.get("spectral_centroid", 0),
        }
    
    def _detect_bpm(self, samples: List[float], sr: int) -> float:
        """Simple onset-detection BPM estimator."""
        if len(samples) < sr:
            return 0
        
        # Compute energy envelope
        window = int(sr * 0.01)  # 10ms windows
        energy = []
        for i in range(0, len(samples) - window, window):
            e = sum(abs(samples[j]) for j in range(i, i + window))
            energy.append(e)
        
        # Detect onsets (sudden energy increases)
        onsets = []
        for i in range(1, len(energy)):
            if energy[i] > energy[i-1] * 1.5 and energy[i] > 1000:
                onsets.append(i)
        
        if len(onsets) < 2:
            return 0
        
        # Calculate intervals
        intervals = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]
        avg_interval = sum(intervals) / len(intervals)
        bpm = 60 / (avg_interval * 0.01) if avg_interval > 0 else 0
        
        return bpm
    
    def _frequency_analysis(self, samples: List[float], sr: int) -> Dict:
        """Simplified frequency analysis using zero-crossing and autocorrelation."""
        # Zero-crossing rate (rough frequency estimate)
        zcr = 0
        for i in range(1, len(samples)):
            if (samples[i] >= 0) != (samples[i-1] >= 0):
                zcr += 1
        zcr_rate = zcr / (len(samples) / sr)
        
        # Simple autocorrelation for pitch detection
        corr = self._autocorrelate(samples[:min(len(samples), sr * 2)], sr)
        
        return {
            "zero_crossing_rate": round(zcr_rate, 1),
            "dominant_frequencies": corr[:5],
            "spectral_centroid": round(zcr_rate / 2, 1),
        }
    
    def _autocorrelate(self, samples: List[float], sr: int) -> List[float]:
        """Find dominant frequencies via autocorrelation."""
        n = len(samples)
        if n < 100:
            return []
        
        results = []
        for lag in range(10, min(n//2, sr//20)):
            corr = sum(samples[i] * samples[i + lag] for i in range(n - lag))
            results.append((sr / lag, corr))
        
        results.sort(key=lambda x: -x[1])
        return [round(f, 1) for f, c in results[:5] if c > 0]


if __name__ == "__main__":
    a = AudioAnalyser()
    print("DMAI AudioAnalyser ready.")
    print("  Methods: analyse(wav_path)")
