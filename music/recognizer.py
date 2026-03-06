#!/usr/bin/env python3
"""
DMAI Music Recognition System
Fixed version - matches your exact screenshot format
"""

import os
import sys
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
import re
from typing import List, Dict, Optional

# Try to import OCR
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  pytesseract not installed. Run: pip install pytesseract pillow")

# Setup logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MUSIC - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'music_recognizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MUSIC")

class MusicRecognizer:
    """
    Recognizes music from screenshots using OCR
    Matches your specific format: "Title © Artist"
    """
    
    def __init__(self):
        self.screenshots_dir = Path(__file__).parent / 'screenshots'
        self.db_path = Path(__file__).parent / 'music_library.db'
        self.preferences_path = Path(__file__).parent / 'preferences.json'
        
        self._init_database()
        self._load_preferences()
        
        if OCR_AVAILABLE:
            logger.info("✅ OCR initialized - will extract ALL text from images")
        else:
            logger.error("❌ OCR not available - install pytesseract and tesseract")
        
        logger.info("🎵 Music Recognizer initialized")
    
    def _init_database(self):
        """Initialize SQLite database for music library"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Songs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                artist TEXT,
                album TEXT,
                genre TEXT,
                year INTEGER,
                source_file TEXT,
                play_count INTEGER DEFAULT 0,
                last_played TIMESTAMP,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(title, artist)
            )
        ''')
        
        # Raw OCR text storage
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ocr_text (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                raw_text TEXT,
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Screenshots processed tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_screenshots (
                filename TEXT PRIMARY KEY,
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                text_length INTEGER,
                songs_found INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("📀 Music database initialized")
    
    def _load_preferences(self):
        """Load user preferences from JSON"""
        if self.preferences_path.exists():
            with open(self.preferences_path, 'r') as f:
                self.preferences = json.load(f)
        else:
            self.preferences = {
                'favorite_artists': [],
                'favorite_genres': [],
                'favorite_songs': [],
                'recently_played': [],
                'playlists': {},
                'total_screenshots': 0,
                'total_songs_found': 0
            }
            self._save_preferences()
    
    def _save_preferences(self):
        """Save preferences to JSON"""
        with open(self.preferences_path, 'w') as f:
            json.dump(self.preferences, f, indent=2)
    
    def extract_text_with_ocr(self, image_path: Path) -> str:
        """Extract ALL text from image using OCR"""
        if not OCR_AVAILABLE:
            return "OCR not available"
        
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            
            words = len(text.split())
            lines = len([line for line in text.split('\n') if line.strip()])
            
            logger.info(f"📝 OCR extracted {words} words, {lines} lines from {image_path.name}")
            
            return text
            
        except Exception as e:
            logger.error(f"OCR error on {image_path.name}: {e}")
            return ""
    
    def parse_music_info(self, text: str) -> List[Dict]:
        """
        Parse music information from your specific screenshot format
        Matches: "Title © Artist" where © is char 169
        """
        songs = []
        
        if not text:
            return songs
        
        # Split into lines and process each
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            # Look for the copyright symbol (char 169) which is the separator
            if '©' in line or chr(169) in line:
                # Split on the copyright symbol
                parts = line.split('©')
                if len(parts) >= 2:
                    title_part = parts[0].strip()
                    artist_part = parts[1].strip()
                    
                    # Clean up title - remove leading symbols like '|', 'P|', '}', etc.
                    title = re.sub(r'^[|P{}>\s]+', '', title_part)
                    title = re.sub(r'[|\[\]{}()]$', '', title)
                    title = title.strip()
                    
                    # Clean up artist - remove trailing symbols like '@', 'QO', etc.
                    artist = re.sub(r'\s+[@QO].*$', '', artist_part)
                    artist = re.sub(r'[|\[\]{}()]$', '', artist)
                    artist = artist.strip()
                    
                    # Additional cleanup for specific patterns
                    artist = re.sub(r'\s+ee+$', '', artist)  # Remove trailing 'ee' or 'eee'
                    artist = re.sub(r'\s+@\s+\d+$', '', artist)  # Remove '@ 000'
                    
                    if title and artist and len(title) > 2 and len(artist) > 2:
                        # Don't add if it looks like garbage
                        if not re.match(r'^[©\s]+$', title) and not re.match(r'^[©\s]+$', artist):
                            songs.append({
                                'title': title,
                                'artist': artist
                            })
                            logger.debug(f"Found: '{title}' by '{artist}'")
        
        # Remove duplicates
        unique_songs = []
        seen = set()
        for song in songs:
            key = f"{song['title']}_{song['artist']}"
            if key not in seen:
                seen.add(key)
                unique_songs.append(song)
        
        return unique_songs
    
    def process_screenshots(self) -> int:
        """Process all unprocessed screenshots with OCR"""
        if not self.screenshots_dir.exists():
            logger.error(f"Screenshots directory not found: {self.screenshots_dir}")
            return 0
        
        screenshots = list(self.screenshots_dir.glob("*.PNG")) + \
                     list(self.screenshots_dir.glob("*.png")) + \
                     list(self.screenshots_dir.glob("*.jpg")) + \
                     list(self.screenshots_dir.glob("*.jpeg"))
        
        logger.info(f"Found {len(screenshots)} screenshots to process with OCR")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        processed_count = 0
        total_songs_found = 0
        all_songs = []
        
        for screenshot in screenshots:
            logger.info(f"🔍 Processing: {screenshot.name}")
            
            # Extract ALL text with OCR
            text = self.extract_text_with_ocr(screenshot)
            
            # Store raw OCR text
            cursor.execute('''
                INSERT INTO ocr_text (filename, raw_text)
                VALUES (?, ?)
            ''', (screenshot.name, text))
            
            # Parse ALL music info
            songs = self.parse_music_info(text)
            all_songs.extend(songs)
            
            # Store songs in database
            songs_added = 0
            for song in songs:
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO songs 
                        (title, artist, source_file)
                        VALUES (?, ?, ?)
                    ''', (
                        song['title'],
                        song['artist'],
                        screenshot.name
                    ))
                    if cursor.rowcount > 0:
                        songs_added += 1
                except Exception as e:
                    logger.error(f"Error adding song: {e}")
            
            # Mark as processed
            cursor.execute('''
                INSERT INTO processed_screenshots (filename, text_length, songs_found)
                VALUES (?, ?, ?)
            ''', (screenshot.name, len(text), songs_added))
            
            conn.commit()
            processed_count += 1
            total_songs_found += songs_added
            
            logger.info(f"✅ Processed {screenshot.name} - found {songs_added} new songs")
        
        conn.close()
        
        # Update preferences
        self.preferences['total_screenshots'] = self.preferences.get('total_screenshots', 0) + processed_count
        self.preferences['total_songs_found'] = self.preferences.get('total_songs_found', 0) + total_songs_found
        
        self.update_preferences()
        
        logger.info(f"📊 Total: {processed_count} screenshots processed, {total_songs_found} new songs found")
        
        return processed_count
    
    def update_preferences(self):
        """Update user preferences based on database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get favorite artists
        cursor.execute('''
            SELECT artist, COUNT(*) as count 
            FROM songs 
            WHERE artist IS NOT NULL AND artist != ''
            GROUP BY artist 
            ORDER BY count DESC 
            LIMIT 20
        ''')
        self.preferences['favorite_artists'] = [
            {'artist': row[0], 'count': row[1]} 
            for row in cursor.fetchall()
        ]
        
        # Get all songs
        cursor.execute('''
            SELECT title, artist 
            FROM songs 
            WHERE title IS NOT NULL AND artist IS NOT NULL
            ORDER BY artist, title
        ''')
        self.preferences['all_songs'] = [
            {'title': row[0], 'artist': row[1]}
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        self.preferences['last_updated'] = datetime.now().isoformat()
        self._save_preferences()
    
    def get_library_stats(self) -> Dict:
        """Get detailed music library statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM songs")
        total_songs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT artist) FROM songs WHERE artist IS NOT NULL AND artist != ''")
        total_artists = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM processed_screenshots")
        total_screenshots = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(songs_found) FROM processed_screenshots")
        total_songs_found = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_songs': total_songs,
            'total_artists': total_artists,
            'total_screenshots': total_screenshots,
            'total_songs_found': total_songs_found
        }

def main():
    """Process all music screenshots with real OCR"""
    
    if not OCR_AVAILABLE:
        print("\n❌ OCR NOT AVAILABLE")
        print("Please run these commands first:")
        print("  brew install tesseract")
        print("  pip install pytesseract pillow")
        return
    
    # Reset database to reprocess with new parser
    db_path = Path(__file__).parent / 'music_library.db'
    if db_path.exists():
        print("\n🔄 Resetting database to reprocess with improved parser...")
        db_path.unlink()
    
    recognizer = MusicRecognizer()
    
    print("\n🎵 DMAI MUSIC RECOGNITION SYSTEM - FIXED PARSER")
    print("=" * 50)
    
    # Process screenshots
    processed = recognizer.process_screenshots()
    
    if processed > 0:
        print(f"\n✅ Processed {processed} new screenshots")
        
        # Show stats
        stats = recognizer.get_library_stats()
        print(f"\n📊 Music Library Stats:")
        print(f"   - Songs in database: {stats['total_songs']}")
        print(f"   - Unique artists: {stats['total_artists']}")
        print(f"   - Screenshots processed: {stats['total_screenshots']}")
        print(f"   - Total songs found: {stats['total_songs_found']}")
        
        # Show all songs found
        if recognizer.preferences.get('all_songs'):
            print(f"\n📝 Songs in your library:")
            current_artist = None
            for song in recognizer.preferences['all_songs']:
                if song['artist'] != current_artist:
                    print(f"\n   🎤 {song['artist']}:")
                    current_artist = song['artist']
                print(f"      • {song['title']}")
    else:
        print("\n📝 No new screenshots to process")
    
    print(f"\n📁 Screenshots: {recognizer.screenshots_dir}")
    print(f"💾 Database: {recognizer.db_path}")
    print(f"⚙️  Preferences: {recognizer.preferences_path}")

if __name__ == "__main__":
    main()
