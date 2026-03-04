"""Advanced Music Identification for DMAI"""
import json
import os
import subprocess
import tempfile
from datetime import datetime

class MusicIdentifier:
    def __init__(self):
        self.music_file = "language_learning/data/music_preferences.json"
        self.init_file()
    
    def init_file(self):
        """Ensure music preferences file exists with proper structure"""
        if not os.path.exists(self.music_file):
            with open(self.music_file, 'w') as f:
                json.dump({
                    "artists": {},
                    "songs": {},
                    "listening_history": [],
                    "identified_tracks": []
                }, f, indent=2)
    
    def identify_with_ffmpeg(self, audio_file):
        """Extract metadata using ffprobe"""
        try:
            result = subprocess.run([
                'ffprobe', '-i', audio_file, '-show_format', '-print_format', 'json'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                format_data = data.get('format', {})
                tags = format_data.get('tags', {})
                
                # Look for music metadata in various tag formats
                artist = (tags.get('artist') or tags.get('ARTIST') or 
                         tags.get('Artist') or tags.get('TPE1'))
                title = (tags.get('title') or tags.get('TITLE') or 
                        tags.get('Title') or tags.get('TIT2'))
                album = (tags.get('album') or tags.get('ALBUM') or 
                        tags.get('Album'))
                
                if artist or title:
                    return {
                        'artist': artist,
                        'title': title,
                        'album': album,
                        'method': 'ffmpeg'
                    }
        except:
            pass
        return None
    
    def identify_with_acoustid(self, audio_file):
        """Use AcoustID fingerprinting (requires chromaprint)"""
        try:
            # Generate fingerprint
            fp_result = subprocess.run([
                'fpcalc', '-raw', '-length', '30', audio_file
            ], capture_output=True, text=True, timeout=30)
            
            if fp_result.returncode == 0:
                # Parse fingerprint
                lines = fp_result.stdout.strip().split('\n')
                fingerprint = None
                duration = None
                
                for line in lines:
                    if line.startswith('FINGERPRINT='):
                        fingerprint = line[12:]
                    elif line.startswith('DURATION='):
                        duration = float(line[9:])
                
                if fingerprint and duration:
                    # Note: This would need an AcoustID API key for lookup
                    # For now, store fingerprint for future matching
                    return {
                        'fingerprint': fingerprint[:50] + '...',
                        'duration': duration,
                        'method': 'acoustid_partial'
                    }
        except:
            pass
        return None
    
    def identify_audio(self, audio_file):
        """Try multiple methods to identify music"""
        
        # Method 1: Metadata
        info = self.identify_with_ffmpeg(audio_file)
        if info:
            return info
        
        # Method 2: Fingerprinting (partial)
        info = self.identify_with_acoustid(audio_file)
        if info:
            return info
        
        return None
    
    def learn_from_audio(self, audio_file, source="ambient"):
        """Process audio file and update music preferences"""
        
        # Try to identify
        info = self.identify_audio(audio_file)
        
        # Load current preferences
        with open(self.music_file, 'r') as f:
            prefs = json.load(f)
        
        if info and info.get('artist'):
            artist = info['artist']
            prefs["artists"][artist] = prefs["artists"].get(artist, 0) + 1
            print(f"🎵 Identified: {artist}")
            
            if info.get('title'):
                song = f"{artist} - {info['title']}"
                prefs["songs"][song] = prefs["songs"].get(song, 0) + 1
                print(f"   Song: {info['title']}")
            
            prefs["identified_tracks"].append({
                "timestamp": datetime.now().isoformat(),
                "artist": artist,
                "title": info.get('title'),
                "album": info.get('album'),
                "method": info.get('method')
            })
        else:
            # Unknown music - just count it
            prefs["artists"]["Unknown"] = prefs["artists"].get("Unknown", 0) + 1
            print("🎵 Music detected (unknown artist)")
        
        # Always log listening history
        prefs["listening_history"].append({
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "identified": info is not None
        })
        
        # Save
        with open(self.music_file, 'w') as f:
            json.dump(prefs, f, indent=2)
        
        return info

if __name__ == "__main__":
    # Test with a sample audio file
    import sys
    if len(sys.argv) > 1:
        identifier = MusicIdentifier()
        result = identifier.learn_from_audio(sys.argv[1])
        print(f"Result: {result}")
    else:
        print("Usage: python music_identifier.py <audio_file>")
