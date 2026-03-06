#!/usr/bin/env python3
"""
DMAI Music Fetcher - Searches and plays free music from online sources
"""

import json
import requests
import subprocess
import random
import os
import tempfile
from pathlib import Path

class MusicFetcher:
    def __init__(self):
        self.music_dir = "/Users/davidmiles/Desktop/AI-Evolution-System/music"
        self.prefs_file = f"{self.music_dir}/preferences.json"
        self.cache_dir = f"{self.music_dir}/cache"
        Path(self.cache_dir).mkdir(exist_ok=True)
        
        # API endpoints
        self.apis = {
            "musicapi": {
                "prepare": "https://bhindi1.ddns.net/music/api/prepare/{}",
                "fetch": "https://bhindi1.ddns.net/music/api/fetch/{}",
                "audio": "https://bhindi1.ddns.net/music/api/audio/{}"
            }
        }
    
    def search_song(self, query):
        """Search for a song using MusicAPI"""
        try:
            # URL encode the query
            import urllib.parse
            encoded = urllib.parse.quote(query)
            
            # Step 1: Get song ID
            prepare_url = self.apis["musicapi"]["prepare"].format(encoded)
            response = requests.get(prepare_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('song_id'):
                    song_id = data['song_id']
                    print(f"✅ Found song ID: {song_id}")
                    
                    # Step 2: Fetch song details
                    fetch_url = self.apis["musicapi"]["fetch"].format(song_id)
                    fetch_response = requests.get(fetch_url, timeout=10)
                    
                    if fetch_response.status_code == 200:
                        song_data = fetch_response.json()
                        
                        # Step 3: Get audio URL
                        audio_url = self.apis["musicapi"]["audio"].format(song_id)
                        
                        return {
                            'id': song_id,
                            'title': song_data.get('title', query),
                            'artist': song_data.get('artist', 'Unknown'),
                            'audio_url': audio_url,
                            'thumbnail': song_data.get('thumbnail', ''),
                            'duration': song_data.get('duration', 0)
                        }
        except Exception as e:
            print(f"❌ Search error: {e}")
        
        return None
    
    def play_song(self, song_info):
        """Play a song by streaming from URL"""
        if not song_info or 'audio_url' not in song_info:
            print("❌ No audio URL available")
            return False
        
        print(f"\n🎵 Now playing: {song_info['title']} - {song_info['artist']}")
        print(f"📡 Streaming from: {song_info['audio_url']}")
        
        try:
            # Use ffmpeg or mpg123 to stream audio (install first if needed)
            # Option 1: Use mpg123 (install: brew install mpg123)
            # cmd = ['mpg123', song_info['audio_url']]
            
            # Option 2: Use ffmpeg and afplay (macOS built-in)
            # Download to temp file first (more reliable)
            import tempfile
            import requests
            
            print("📥 Buffering audio...")
            response = requests.get(song_info['audio_url'], stream=True, timeout=30)
            
            if response.status_code == 200:
                # Save to temp file
                temp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                for chunk in response.iter_content(chunk_size=8192):
                    temp.write(chunk)
                temp.close()
                
                print("▶️ Playing... (Press Ctrl+C to stop)")
                
                # Play with afplay (macOS built-in)
                subprocess.run(['afplay', temp.name])
                
                # Clean up
                os.unlink(temp.name)
                return True
            else:
                print(f"❌ Failed to download: {response.status_code}")
                return False
                
        except KeyboardInterrupt:
            print("\n⏹️ Playback stopped")
            return True
        except Exception as e:
            print(f"❌ Playback error: {e}")
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
                top_artist = fav_artists[0]['artist']
                artist_songs = [s for s in all_songs if s['artist'] == top_artist]
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
        
        # Check for mpg123 (optional but better)
        try:
            subprocess.run(['which', 'mpg123'], check=True, capture_output=True)
            print("✅ mpg123 found")
        except:
            print("⚠️ mpg123 not found (optional, using afplay instead)")
            print("   To install: brew install mpg123")
        
        # Install Python packages
        subprocess.run(['pip', 'install', 'requests'], capture_output=True)
        print("✅ Python dependencies ready")

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
            print(f"\n🔍 Searching: {query}")
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
