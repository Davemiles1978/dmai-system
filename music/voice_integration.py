#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
DMAI Voice Integration for Music
Connects with the music database for voice commands
"""

import os
import sys
import sqlite3
import logging
import random
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("MUSIC_VOICE")

class MusicVoiceIntegration:
    """Connects voice commands to music system"""
    
    def __init__(self):
        self.db_path = Path(__file__).parent / 'music_library.db'
        self.prefs_path = Path(__file__).parent / 'preferences.json'
        
        # Check if database exists
        if not self.db_path.exists():
            print("⚠️  Music database not found. Please run import_csv.py first.")
        
        self.commands = {
            "play my music": self.play_music,
            "what's in my library": self.show_library,
            "recommend something": self.recommend,
            "music stats": self.show_stats,
            "songs by": self.songs_by_artist,
            "random song": self.random_song
        }
    
    def get_stats(self):
        """Get library statistics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM songs")
        total_songs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT artist) FROM songs")
        total_artists = cursor.fetchone()[0]
        
        conn.close()
        return total_songs, total_artists
    
    def play_music(self, artist=None):
        """Play music based on preferences"""
        total_songs, _ = self.get_stats()
        if total_songs > 0:
            if artist:
                return f"Playing songs by {artist} from your library."
            return f"Playing music from your library. You have {total_songs} songs."
        return "Your music library is empty. Please add some songs first."
    
    def show_library(self):
        """Show library stats"""
        total_songs, total_artists = self.get_stats()
        return f"Your music library has {total_songs} songs by {total_artists} artists."
    
    def recommend(self, count=3):
        """Get random recommendations from your library"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, artist FROM songs 
            ORDER BY RANDOM() 
            LIMIT ?
        ''', (count,))
        
        recs = cursor.fetchall()
        conn.close()
        
        if recs:
            songs_list = [f"{title} by {artist}" for title, artist in recs]
            if len(songs_list) == 1:
                return f"I recommend: {songs_list[0]}"
            elif len(songs_list) == 2:
                return f"I recommend: {songs_list[0]} and {songs_list[1]}"
            else:
                last = songs_list.pop()
                return f"I recommend: {', '.join(songs_list)}, and {last}"
        return "Not enough songs for recommendations yet."
    
    def show_stats(self):
        """Show detailed stats"""
        total_songs, total_artists = self.get_stats()
        
        # Get genre stats
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT genre, COUNT(*) as count 
            FROM songs 
            WHERE genre IS NOT NULL AND genre != '' 
            GROUP BY genre 
            ORDER BY count DESC 
            LIMIT 3
        ''')
        top_genres = cursor.fetchall()
        conn.close()
        
        stats = f"Library: {total_songs} songs, {total_artists} artists"
        if top_genres:
            genre_text = ", ".join([f"{g[0]} ({g[1]})" for g in top_genres])
            stats += f". Top genres: {genre_text}"
        
        return stats
    
    def songs_by_artist(self, artist_name=None):
        """Get songs by a specific artist"""
        if not artist_name:
            return "Please specify an artist. Example: 'songs by David Kushner'"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title FROM songs 
            WHERE artist LIKE ? 
            ORDER BY title
        ''', (f'%{artist_name}%',))
        
        songs = cursor.fetchall()
        conn.close()
        
        if songs:
            if len(songs) == 1:
                return f"Found 1 song by {artist_name}: {songs[0][0]}"
            else:
                song_list = [s[0] for s in songs]
                return f"Found {len(songs)} songs by {artist_name}: {', '.join(song_list)}"
        return f"No songs found by {artist_name}"
    
    def random_song(self):
        """Get a random song from your library"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, artist FROM songs 
            ORDER BY RANDOM() 
            LIMIT 1
        ''')
        
        song = cursor.fetchone()
        conn.close()
        
        if song:
            return f"Random song: {song[0]} by {song[1]}"
        return "No songs in library"
    
    def process_command(self, command_text: str) -> str:
        """Process a voice command"""
        command_text = command_text.lower().strip()
        
        # Check for artist-specific command
        if "songs by" in command_text:
            artist = command_text.replace("songs by", "").strip()
            return self.songs_by_artist(artist)
        
        # Check other commands
        for cmd, func in self.commands.items():
            if cmd in command_text:
                return func()
        
        return ("Music command not recognized. Try: "
                "'play my music', "
                "'what's in my library', "
                "'recommend something', "
                "'music stats', "
                "'songs by [artist]', "
                "'random song'")

def main():
    """Test the voice integration"""
    music = MusicVoiceIntegration()
    
    print("🎵 Testing Music Voice Integration")
    print("=" * 40)
    
    test_commands = [
        "play my music",
        "what's in my library",
        "recommend something",
        "music stats",
        "songs by David Kushner",
        "random song"
    ]
    
    for cmd in test_commands:
        response = music.process_command(cmd)
        print(f"\nCommand: '{cmd}'")
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
