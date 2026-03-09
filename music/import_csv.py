#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Import script for your exact CSV format
Handles the "my_music_library" title line at the top
"""

import csv
import sqlite3
import json
from pathlib import Path

# Paths
csv_path = Path(__file__).parent / 'my_music_library.csv'
db_path = Path(__file__).parent / 'music_library.db'
prefs_path = Path(__file__).parent / 'preferences.json'

print("🎵 Importing your music library...")
print("=" * 40)

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Clear existing manual entries
cursor.execute("DELETE FROM songs WHERE source_file = 'Manual Entry'")
deleted = cursor.rowcount
print(f"🗑️  Removed {deleted} existing manual entries")

# Read the CSV file
with open(csv_path, 'r') as f:
    # Skip the first line (it's "my_music_library")
    f.readline()
    
    # Now read the CSV with headers on the second line
    reader = csv.DictReader(f)
    
    count = 0
    skipped = 0
    
    for row in reader:
        if row.get('Title') and row.get('Artist'):  # Skip empty rows
            try:
                # Clean up the data
                title = row['Title'].strip()
                artist = row['Artist'].strip()
                album = row.get('Album', '').strip() or None
                genre = row.get('Genre', '').strip() or None
                year = row.get('Year', '').strip()
                year = int(year) if year and year.isdigit() else None
                
                # Insert the song
                cursor.execute('''
                    INSERT OR IGNORE INTO songs 
                    (title, artist, album, genre, year, source_file, play_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    title,
                    artist,
                    album,
                    genre,
                    year,
                    'Manual Entry',
                    0
                ))
                
                if cursor.rowcount > 0:
                    count += 1
                    print(f"  ✅ Added: {title} - {artist}")
                else:
                    skipped += 1
                    print(f"  ⚠️  Already exists: {title} - {artist}")
                    
            except Exception as e:
                print(f"  ❌ Error adding {row.get('Title', 'Unknown')}: {e}")
                print(f"     Row data: {row}")

# Commit the changes
conn.commit()

print(f"\n📊 Import summary:")
print(f"   - Added: {count} new songs")
print(f"   - Skipped (duplicates): {skipped} songs")

# Update preferences
cursor.execute("SELECT artist, COUNT(*) as count FROM songs GROUP BY artist ORDER BY count DESC")
artists = [{'artist': row[0], 'count': row[1]} for row in cursor.fetchall()]

cursor.execute("SELECT title, artist FROM songs ORDER BY artist, title")
songs = [{'title': row[0], 'artist': row[1]} for row in cursor.fetchall()]

# Get total count
cursor.execute("SELECT COUNT(*) FROM songs")
total = cursor.fetchone()[0]

# Get stats by genre
cursor.execute("SELECT genre, COUNT(*) as count FROM songs WHERE genre IS NOT NULL AND genre != '' GROUP BY genre ORDER BY count DESC")
genres = [{'genre': row[0], 'count': row[1]} for row in cursor.fetchall()]

conn.close()

# Create preferences
preferences = {
    'favorite_artists': artists,
    'all_songs': songs,
    'genres': genres,
    'total_songs': total,
    'total_artists': len(artists),
    'last_updated': str(Path(csv_path).stat().st_mtime),
    'source': 'Manual Import CSV'
}

# Save preferences
with open(prefs_path, 'w') as f:
    json.dump(preferences, f, indent=2)

print(f"\n✅ Import complete!")
print(f"📊 Final library stats:")
print(f"   - Total songs: {total}")
print(f"   - Unique artists: {len(artists)}")
print(f"   - Genres: {len(genres)}")

# Show top artists
if artists:
    print("\n🎤 Top artists in your library:")
    for artist in artists[:10]:
        print(f"   • {artist['artist']}: {artist['count']} songs")

# Show genres
if genres:
    print("\n🎵 Genres in your library:")
    for genre in genres[:5]:
        print(f"   • {genre['genre']}: {genre['count']} songs")

print(f"\n📁 Database: {db_path}")
print(f"⚙️  Preferences: {prefs_path}")
