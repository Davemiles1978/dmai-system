#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
DMAI Music Fetcher - Searches and plays free music from working sources
"""

import json
import requests
import subprocess
import random
import os
import tempfile
import time
from pathlib import Path

class MusicFetcher:
    def __init__(self):
        self.music_dir = "/Users/davidmiles/Desktop/dmai-system/music"
        self.prefs_file = f"{self.music_dir}/preferences.json"
        self.cache_dir = f"{self.music_dir}/cache"
        Path(self.cache_dir).mkdir(exist_ok=True)
        
        # Working free music APIs
        self.apis = {
            "fma": {
                "search": "https://freemusicarchive.org/api/trackSearch?q={}",
                "details": "https://freemusicarchive.org/api/track/{}/details"
            },
            "jamendo": {
                "search": "https://api.jamendo.com/v3.0/tracks/?client_id=2c9f96c1&format=json&limit=10&namesearch={}",
                "stream": "https://mp3l.jamendo.com/?trackid={}&format=mp31"
            },
            "audius": {
                "search": "https://discoveryprovider.audius.co/v1/full/search/tracks?query={}",
                "stream": "https://audius.co/tracks/{}/stream"
            }
        }
    
    def search_fma(self, query):
        """Search Free Music Archive"""
        try:
            import urllib.parse
            encoded = urllib.parse.quote(query)
            url = self.apis["fma"]["search"].format(encoded)
            
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('aTracks'):
                    tracks = data['aTracks']
                    if tracks:
                        track = tracks[0]
                        return {
                            'title': track.get('track_title', query),
                            'artist': track.get('artist_name', 'Unknown'),
                            'audio_url': track.get('track_url', ''),
                            'source': 'fma'
                        }
            return None
        except Exception as e:
            print(f"FMA search error: {e}")
            return None
    
    def search_jamendo(self, query):
        """Search Jamendo (free music)"""
        try:
            import urllib.parse
            encoded = urllib.parse.quote(query)
            url = self.apis["jamendo"]["search"].format(encoded)
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    track = data['results'][0]
                    track_id = track.get('id')
                    return {
                        'title': track.get('name', query),
                        'artist': track.get('artist_name', 'Unknown'),
                        'audio_url': self.apis["jamendo"]["stream"].format(track_id),
                        'source': 'jamendo'
                    }
            return None
        except Exception as e:
            print(f"Jamendo search error: {e}")
            return None
    
    def search_audius(self, query):
        """Search Audius"""
        try:
            import urllib.parse
            encoded = urllib.parse.quote(query)
            url = self.apis["audius"]["search"].format(encoded)
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    track = data['data'][0]
                    track_id = track.get('id')
                    return {
                        'title': track.get('title', query),
                        'artist': track.get('user', {}).get('name', 'Unknown'),
                        'audio_url': self.apis["audius"]["stream"].format(track_id),
                        'source': 'audius'
                    }
            return None
        except Exception as e:
            print(f"Audius search error: {e}")
            return None
    
    def search_song(self, query):
        """Try all APIs until one works"""
        print(f"🔍 Searching for: {query}")
        
        # Try Jamendo first
        result = self.search_jamendo(query)
        if result:
            print(f"✅ Found on Jamendo: {result['title']} - {result['artist']}")
            return result
        
        # Try Free Music Archive
        result = self.search_fma(query)
        if result:
            print(f"✅ Found on FMA: {result['title']} - {result['artist']}")
            return result
        
        # Try Audius
        result = self.search_audius(query)
        if result:
            print(f"✅ Found on Audius: {result['title']} - {result['artist']}")
            return result
        
        print("❌ Could not find on any free music service")
        return None
    
    def play_song(self, song_info):
        """Play a song by streaming from URL"""
        if not song_info or 'audio_url' not in song_info:
            print("❌ No audio URL available")
            return False
        
        print(f"\n🎵 Now playing: {song_info['title']} - {song_info['artist']}")
        print(f"📡 Source: {song_info.get('source', 'unknown')}")
        
        try:
            # Download to temp file
            import tempfile
            import requests
            
            print("📥 Downloading audio...")
            
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            response = requests.get(song_info['audio_url'], stream=True, timeout=30, headers=headers)
            
            if response.status_code == 200:
                # Save to temp file
                temp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp.write(chunk)
                        downloaded += len(chunk)
                
                temp.close()
                print(f"✅ Downloaded {downloaded/1024/1024:.1f} MB")
                print("▶️ Playing... (Press Ctrl+C to stop)")
                
                # Play with afplay (macOS built-in)
                process = subprocess.run(['afplay', temp.name])
                
                # Clean up
                os.unlink(temp.name)
                return True
            else:
                print(f"❌ Failed to download: {response.status_code}")
                return False
                
        except KeyboardInterrupt:
            print("\n⏹️ Playback stopped")
            if 'temp' in locals():
                os.unlink(temp.name)
            return True
        except Exception as e:
            print(f"❌ Playback error: {e}")
            if 'temp' in locals():
                try:
                    os.unlink(temp.name)
                except:
                    pass
            return False
    
    def recommend_from_preferences(self):
        """Recommend a song based on your preferences"""
        try:
            with open(self.prefs_file, 'r') as f:
                prefs = json.load(f)
            
            all_songs = prefs.get('all_songs', [])
            fav_artists = prefs.get('favorite_artists', [])
            
            if not all_songs:
                print("❌ No songs in preferences")
                return None
            
            # Try to pick from favorite artists first
            if fav_artists:
                # Get top 3 favorite artists
                top_artists = [a['artist'] for a in fav_artists[:3]]
                print(f"🎯 Top artists: {', '.join(top_artists)}")
                
                # Find songs by these artists
                artist_songs = [s for s in all_songs if s['artist'] in top_artists]
                if artist_songs:
                    return random.choice(artist_songs)
            
            # Otherwise pick random
            return random.choice(all_songs)
            
        except Exception as e:
            print(f"❌ Error reading preferences: {e}")
            return None
    
    def install_dependencies(self):
        """Install required packages"""
        print("📦 Checking dependencies...")
        # Python requests already installed
        print("✅ Python dependencies ready")
        print("ℹ️ Using afplay for playback (macOS built-in)")

if __name__ == "__main__":
    fetcher = MusicFetcher()
    fetcher.install_dependencies()
    
    print("\n🎵 DMAI MUSIC FETCHER")
    print("="*50)
    
    # Try a recommendation from your preferences
    print("\n📋 Your music preferences loaded")
    rec = fetcher.recommend_from_preferences()
    
    if rec:
        print(f"\n🎯 Recommended: {rec['title']} - {rec['artist']}")
        choice = input("\nSearch for this online? (y/n): ")
        
        if choice.lower() == 'y':
            query = f"{rec['title']} {rec['artist']}"
            song = fetcher.search_song(query)
            
            if song:
                fetcher.play_song(song)
            else:
                print("❌ Could not find that song online")
    
    # Or search manually
    print("\n🔍 Or search manually:")
    manual = input("Enter song name: ")
    if manual:
        song = fetcher.search_song(manual)
        if song:
            fetcher.play_song(song)
