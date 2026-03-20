"""DMAI Music Learning - Understands your music taste and develops her own"""

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import time
import random
from datetime import datetime
import subprocess
import requests
from bs4 import BeautifulSoup

class MusicLearner:
    def __init__(self):
        self.music_file = "language_learning/data/music_preferences.json"
        self.load_preferences()
    
    def load_preferences(self):
        if os.path.exists(self.music_file):
            with open(self.music_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "master_taste": {
                    "genres": {},
                    "artists": {},
                    "decades": {},
                    "moods": {},
                    "listening_patterns": []
                },
                "dmai_taste": {
                    "genres": {},
                    "artists": {},
                    "moods": {},
                    "discoveries": []
                },
                "recommendations": [],
                "last_scan": None
            }
            self.save()
    
    def save(self):
        os.makedirs(os.path.dirname(self.music_file), exist_ok=True)
        with open(self.music_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def scan_local_music(self):
        """Scan local music files (MacOS)"""
        music_dirs = [
            os.path.expanduser("~/Music"),
            os.path.expanduser("~/Downloads"),
            "/System/Volumes/Data/Users/davidmiles/Music"
        ]
        
        found = 0
        for music_dir in music_dirs:
            if os.path.exists(music_dir):
                for root, dirs, files in os.walk(music_dir):
                    for file in files:
                        if file.endswith(('.mp3', '.m4a', '.flac', '.wav', '.aiff')):
                            # Simple filename parsing
                            name = file.replace('.mp3', '').replace('.m4a', '').replace('.flac', '')
                            if ' - ' in name:
                                artist, song = name.split(' - ', 1)
                            else:
                                artist = "Unknown"
                                song = name
                            
                            # Update artist preference
                            self.data["master_taste"]["artists"][artist] = self.data["master_taste"]["artists"].get(artist, 0) + 1
                            found += 1
        
        self.data["last_scan"] = datetime.now().isoformat()
        self.save()
        return found
    
    def get_artist_info(self, artist):
        """Fetch artist info from MusicBrainz"""
        try:
            url = f"https://musicbrainz.org/ws/2/artist/?query={artist}&fmt=json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('artists'):
                    artist_data = data['artists'][0]
                    genres = artist_data.get('genres', [])
                    tags = artist_data.get('tags', [])
                    return {
                        'genres': [g['name'] for g in genres],
                        'tags': [t['name'] for t in tags],
                        'country': artist_data.get('country', ''),
                        'type': artist_data.get('type', '')
                    }
        except:
            pass
        return None
    
    def learn_from_spotify(self, playlist_url=None):
        """Learn from Spotify (would need API key)"""
        # Placeholder - would use Spotify API
        pass
    
    def analyze_your_taste(self):
        """Analyze patterns in your music"""
        artists = self.data["master_taste"]["artists"]
        if not artists:
            return {"top_artists": [], "top_genres": []}
        
        # Find top artists
        top_artists = sorted(artists.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Try to get genres for top artists
        genres = {}
        for artist, count in top_artists:
            info = self.get_artist_info(artist)
            if info and info.get('genres'):
                for genre in info['genres']:
                    genres[genre] = genres.get(genre, 0) + count
        
        return {
            'top_artists': top_artists,
            'top_genres': sorted(genres.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def develop_dmai_taste(self):
        """DMAI develops her own music preferences"""
        # Start with your taste as baseline
        your_taste = self.analyze_your_taste()
        
        if your_taste and your_taste.get('top_genres'):
            # Explore related genres
            dmai_genres = []
            for genre, score in your_taste['top_genres']:
                # Add some variation - DMAI might prefer slightly different genres
                variations = {
                    'rock': ['alternative rock', 'indie rock', 'classic rock', 'progressive rock'],
                    'jazz': ['fusion', 'smooth jazz', 'bebop', 'cool jazz'],
                    'classical': ['modern classical', 'minimalism', 'soundtrack', 'baroque'],
                    'electronic': ['ambient', 'downtempo', 'techno', 'house'],
                    'pop': ['indie pop', 'synth pop', 'dream pop', 'art pop'],
                    'hip hop': ['alternative hip hop', 'trip hop', 'instrumental hip hop']
                }
                
                for key, options in variations.items():
                    if key in genre.lower():
                        dmai_genres.extend(options)
            
            # Update DMAI's taste
            for genre in dmai_genres[:5]:
                self.data["dmai_taste"]["genres"][genre] = self.data["dmai_taste"]["genres"].get(genre, 0) + 1
        
        self.save()
    
    def discover_new_music(self):
        """Find new music DMAI thinks you might like"""
        recommendations = []
        
        # Based on your top artists
        top_artists = sorted(self.data["master_taste"]["artists"].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        
        for artist, count in top_artists:
            # Get similar artists (would need API)
            similar = [
                f"Artists similar to {artist} - Check out their related artists",
                f"Fans of {artist} also enjoy similar genres"
            ]
            recommendations.extend(similar)
        
        # Based on DMAI's taste
        dmai_artists = list(self.data["dmai_taste"]["artists"].keys())
        if dmai_artists:
            recommendations.append(f"DMAI recommends exploring: {random.choice(dmai_artists)}")
        
        self.data["recommendations"] = recommendations[-10:]
        self.save()
        return recommendations
    
    def get_report(self):
        """Get music learning summary"""
        your_taste = self.analyze_your_taste()
        
        print("\n🎵 MUSIC LEARNING REPORT")
        print("="*50)
        
        print("\n🎧 Your Top Artists:")
        if your_taste and your_taste.get('top_artists'):
            for artist, count in your_taste['top_artists'][:5]:
                print(f"   • {artist}: {count} songs")
        else:
            print("   No music data yet - scan your library first")
        
        print("\n🎼 Your Preferred Genres:")
        if your_taste and your_taste.get('top_genres'):
            for genre, score in your_taste['top_genres']:
                print(f"   • {genre}")
        else:
            print("   Genre analysis pending...")
        
        print("\n🤖 DMAI's Developing Taste:")
        if self.data["dmai_taste"]["genres"]:
            for genre, count in sorted(self.data["dmai_taste"]["genres"].items(), 
                                      key=lambda x: x[1], reverse=True)[:5]:
                print(f"   • {genre} (exploring)")
        else:
            print("   DMAI is still developing her musical identity")
        
        print("\n💿 Recent Discoveries:")
        if self.data["recommendations"]:
            for rec in self.data["recommendations"][-3:]:
                print(f"   • {rec}")
        else:
            print("   No recommendations yet - keep listening!")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    music = MusicLearner()
    print("Scanning your music...")
    found = music.scan_local_music()
    print(f"Found {found} music files")
    if found > 0:
        music.develop_dmai_taste()
        music.discover_new_music()
    music.get_report()
