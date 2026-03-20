#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""DMAI Screenshot Reader - Extract music artists from your screenshots"""
import os
import json
import glob
from PIL import Image
import pytesseract
import re

class ScreenshotReader:
    def __init__(self):
        self.music_file = "language_learning/data/music_preferences.json"
        self.screenshot_dir = os.path.expanduser("~/Desktop")
        self.load_preferences()
    
    def load_preferences(self):
        if os.path.exists(self.music_file):
            with open(self.music_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "artists": {},
                "screenshots_processed": [],
                "source": "screenshot_ocr"
            }
    
    def save(self):
        with open(self.music_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def extract_text_from_image(self, image_path):
        """Extract all text from an image"""
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            return ""
    
    def find_artist_names(self, text):
        """Extract potential artist names from text"""
        # Look for common music app patterns
        artists = set()
        
        # Split into lines and clean
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) < 2:
                continue
            
            # Pattern 1: "Artist - Song" or "Artist – Song"
            if ' - ' in line or ' – ' in line:
                parts = re.split(r' [–-] ', line)
                if len(parts) >= 1:
                    artists.add(parts[0].strip())
            
            # Pattern 2: Lines that are just artist names (all caps, etc.)
            elif line.isupper() and len(line.split()) <= 4:
                artists.add(line.title())
            
            # Pattern 3: Look for "by Artist" pattern
            elif ' by ' in line.lower():
                parts = line.lower().split(' by ')
                if len(parts) > 1:
                    artists.add(parts[1].strip().title())
        
        return list(artists)
    
    def scan_screenshots(self):
        """Scan all screenshots on your Desktop"""
        screenshots = glob.glob(os.path.join(self.screenshot_dir, "*.png"))
        screenshots += glob.glob(os.path.join(self.screenshot_dir, "*.jpg"))
        screenshots += glob.glob(os.path.join(self.screenshot_dir, "*.jpeg"))
        
        print(f"📸 Found {len(screenshots)} screenshots")
        
        for screenshot in screenshots:
            if screenshot in self.data["screenshots_processed"]:
                continue
            
            print(f"\n🔍 Scanning: {os.path.basename(screenshot)}")
            text = self.extract_text_from_image(screenshot)
            
            if text:
                artists = self.find_artist_names(text)
                for artist in artists:
                    self.data["artists"][artist] = self.data["artists"].get(artist, 0) + 1
                    print(f"  🎵 Found artist: {artist}")
                
                self.data["screenshots_processed"].append(screenshot)
        
        self.save()
        
        print("\n📊 Results:")
        top = sorted(self.data["artists"].items(), key=lambda x: x[1], reverse=True)[:15]
        for artist, count in top:
            print(f"  • {artist}: found in {count} screenshots")
    
    def get_report(self):
        """Generate report for daily status"""
        print("\n🎵 MUSIC ARTISTS LEARNED FROM SCREENSHOTS")
        print("="*50)
        
        if not self.data["artists"]:
            print("No artists learned yet. Run scan_screenshots() first.")
            return
        
        top = sorted(self.data["artists"].items(), key=lambda x: x[1], reverse=True)[:20]
        print(f"\nTop artists DMAI has learned:")
        for artist, count in top:
            print(f"  • {artist}: {count} mentions")
        
        print(f"\n📊 Total unique artists: {len(self.data['artists'])}")
        print(f"📸 Screenshots processed: {len(self.data['screenshots_processed'])}")

if __name__ == "__main__":
    reader = ScreenshotReader()
    print("\n1. Scan all screenshots")
    print("2. Show current results")
    choice = input("\nChoice (1-2): ").strip()
    
    if choice == "1":
        reader.scan_screenshots()
    else:
        reader.get_report()
