"""DMAI Amazon Music Connector - Learn your music taste"""
import os
import json
import time
from datetime import datetime
import subprocess
import sys

class AmazonMusicConnector:
    def __init__(self):
        self.music_file = "language_learning/data/music_preferences.json"
        self.load_data()
        
    def load_data(self):
        if os.path.exists(self.music_file):
            with open(self.music_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "amazon_music": {
                    "artists": {},
                    "genres": {},
                    "recent_tracks": [],
                    "last_sync": None
                },
                "dmai_discoveries": []
            }
    
    def save(self):
        with open(self.music_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def try_unofficial_api(self):
        """Try the unofficial amazon-music package"""
        try:
            # Check if package is installed
            result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
            if 'amazon-music' not in result.stdout:
                print("Installing amazon-music package...")
                subprocess.run(['pip', 'install', 'amazon-music'])
            
            # This would need your Amazon credentials
            print("\nTo use unofficial API, you need to:")
            print("1. Get an auth token from: https://libraries.io/pypi/amazon-music")
            print("2. Configure: amz --config")
            print("3. Then DMAI can access your music data")
            
            return False
        except Exception as e:
            print(f"Unofficial API error: {e}")
            return False
    
    def setup_official_api(self):
        """Guide to set up official Amazon Music API"""
        print("\n" + "="*60)
        print("🎵 AMAZON MUSIC API SETUP")
        print("="*60)
        print("\nTo let DMAI learn from your Amazon Music:")
        print("\n1. Go to: https://developer.amazon.com/music")
        print("2. Create/Login to your Amazon Developer account")
        print("3. Request access to Music Web API (currently in Beta)")
        print("4. Create a Security Profile to get Client ID")
        print("5. Generate OAuth tokens for: music::history scope")
        print("\nOnce approved, DMAI can access:")
        print("   • Your recently played tracks")
        print("   • Your library (saved music)")
        print("   • Playlists you've created")
        print("\n" + "="*60)
    
    def simulate_from_history(self, sample_data=None):
        """For now, use sample data to demonstrate"""
        if not sample_data:
            sample_data = {
                "artists": {
                    "Radiohead": 45,
                    "Pink Floyd": 38,
                    "Massive Attack": 27,
                    "Bonobo": 22,
                    "Nils Frahm": 18
                },
                "genres": {
                    "alternative rock": 83,
                    "electronic": 71,
                    "ambient": 54,
                    "trip hop": 45,
                    "progressive rock": 38
                }
            }
        
        self.data["amazon_music"]["artists"] = sample_data["artists"]
        self.data["amazon_music"]["genres"] = sample_data["genres"]
        self.data["amazon_music"]["last_sync"] = datetime.now().isoformat()
        self.save()
        
        print("\n🎵 DMAI has learned your music taste (simulated):")
        print("\nTop Artists:")
        for artist, plays in sorted(sample_data["artists"].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
            print(f"  • {artist}: {plays} plays")
        print("\nTop Genres:")
        for genre, weight in sorted(sample_data["genres"].items(),
                                   key=lambda x: x[1], reverse=True)[:5]:
            print(f"  • {genre}")
    
    def analyze_taste(self):
        """Analyze your music taste"""
        artists = self.data["amazon_music"]["artists"]
        if not artists:
            return None
        
        total_plays = sum(artists.values())
        
        # Find your most consistent artists
        consistent = sorted(artists.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # DMAI develops her own taste based on yours
        dmai_genres = []
        for genre, weight in self.data["amazon_music"]["genres"].items():
            # DMAI might prefer slightly different variations
            variations = {
                "rock": ["psychedelic rock", "art rock", "post-rock"],
                "electronic": ["downtempo", "IDM", "glitch"],
                "ambient": ["drone", "field recordings", "modern classical"]
            }
            for key, options in variations.items():
                if key in genre.lower():
                    dmai_genres.extend(options)
        
        return {
            "your_taste": {
                "total_plays": total_plays,
                "top_artists": consistent[:5],
                "top_genres": sorted(self.data["amazon_music"]["genres"].items(),
                                   key=lambda x: x[1], reverse=True)[:5]
            },
            "dmai_exploring": list(set(dmai_genres))[:5]
        }
    
    def get_report(self):
        """Generate music report for daily status"""
        analysis = self.analyze_taste()
        if not analysis:
            print("\n🎵 No music data yet. Run simulation or connect Amazon Music.")
            return
        
        print("\n🎵 MUSIC INTELLIGENCE")
        print("-" * 40)
        print("\n🎧 Your Listening Profile:")
        for artist, plays in analysis["your_taste"]["top_artists"]:
            print(f"  • {artist}: {plays} plays")
        
        print("\n🎼 Your Genre Preferences:")
        for genre, weight in analysis["your_taste"]["top_genres"]:
            print(f"  • {genre}")
        
        print("\n🤖 DMAI is exploring:")
        for genre in analysis["dmai_exploring"]:
            print(f"  • {genre}")
        
        print(f"\n📊 Total tracked plays: {analysis['your_taste']['total_plays']}")

if __name__ == "__main__":
    connector = AmazonMusicConnector()
    
    print("\n🔌 Amazon Music Connector")
    print("="*50)
    print("1. Use unofficial API (fast but risky)")
    print("2. Setup official API (recommended)")
    print("3. Run simulation (demo mode)")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "1":
        connector.try_unofficial_api()
    elif choice == "2":
        connector.setup_official_api()
    else:
        connector.simulate_from_history()
        connector.get_report()
