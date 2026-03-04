#!/usr/bin/env python3
"""Scan local Amazon Music data folder for your music taste"""
import os
import json
import glob
from pathlib import Path

print("🎵 Scanning local Amazon Music data...")
print("="*50)

# Amazon Music data locations
amazon_paths = [
    os.path.expanduser("~/Library/Application Support/Amazon Music/Data"),
    os.path.expanduser("~/Music/Amazon Music"),
    os.path.expanduser("~/Music/Amazon Music/Downloads")
]

music_data = {
    "artists": {},
    "albums": {},
    "tracks": [],
    "source": "local_amazon_scan"
}

found_files = 0

for base_path in amazon_paths:
    if os.path.exists(base_path):
        print(f"\n📁 Checking: {base_path}")
        
        # Look for music files
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(('.mp3', '.m4a', '.flac', '.wav', '.json')):
                    found_files += 1
                    
                    # Try to extract artist from filename
                    name = file.replace('.mp3', '').replace('.m4a', '').replace('.flac', '').replace('.wav', '')
                    
                    # Common patterns: "Artist - Song" or "Artist_Song"
                    if ' - ' in name:
                        artist, song = name.split(' - ', 1)
                    elif '_' in name:
                        parts = name.split('_')
                        if len(parts) >= 2:
                            artist, song = parts[0], '_'.join(parts[1:])
                        else:
                            artist = "Unknown"
                            song = name
                    else:
                        artist = "Unknown"
                        song = name
                    
                    # Clean up artist name
                    artist = artist.strip()
                    if artist and artist != "Unknown":
                        music_data["artists"][artist] = music_data["artists"].get(artist, 0) + 1
                        print(f"  Found: {artist}")

print(f"\n📊 Scan complete!")
print(f"   Files found: {found_files}")
print(f"   Unique artists: {len(music_data['artists'])}")

# Save to DMAI's music preferences
music_file = "language_learning/data/music_preferences.json"

# Load existing or create new
if os.path.exists(music_file):
    with open(music_file, 'r') as f:
        prefs = json.load(f)
else:
    prefs = {"amazon_music": {}, "dmai_taste": {}}

prefs["amazon_music"] = music_data

with open(music_file, 'w') as f:
    json.dump(prefs, f, indent=2)

print(f"\n✅ Music taste saved to DMAI!")
print("\nTop artists DMAI learned:")
top_artists = sorted(music_data["artists"].items(), key=lambda x: x[1], reverse=True)[:10]
for artist, count in top_artists:
    print(f"  • {artist}: {count} files")
