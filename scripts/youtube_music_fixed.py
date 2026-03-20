#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
DMAI YouTube Music Player - Fixed version with SSL certificate handling
"""

import subprocess
import sys
import json
import os
import tempfile
import certifi
import ssl
import urllib.request

# Set SSL certificate path
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

def search_youtube(query):
    """Search YouTube and get first result"""
    try:
        import yt_dlp
        
        print(f"🔍 Searching YouTube for: {query}")
        
        # Configure yt-dlp with SSL certificates
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'force_generic_extractor': True,
            'ssl_verify': True,  # Enable SSL verification
            'ssl_cert_file': certifi.where(),  # Use certifi certificates
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_query = f"ytsearch3:{query} audio"  # Get top 3 results
            result = ydl.extract_info(search_query, download=False)
            
            if result and 'entries' in result and result['entries']:
                # Show top results
                print(f"\n📋 Top results:")
                for i, video in enumerate(result['entries'][:3], 1):
                    title = video.get('title', 'Unknown')
                    duration = video.get('duration', 0)
                    duration_str = f"{duration//60}:{duration%60:02d}" if duration else "Unknown"
                    print(f"  {i}. {title} [{duration_str}]")
                
                # Auto-select first result
                video = result['entries'][0]
                video_url = f"https://youtube.com/watch?v={video['id']}"
                
                return {
                    'title': video.get('title', query),
                    'url': video_url,
                    'duration': video.get('duration', 0),
                    'uploader': video.get('uploader', 'Unknown')
                }
    except Exception as e:
        print(f"❌ YouTube search error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check your internet connection")
        print("   2. Try using a VPN if you're in a restricted region")
        print("   3. Update yt-dlp: pip install --upgrade yt-dlp")
    return None

def play_from_youtube(video_info):
    """Extract audio and play from YouTube"""
    if not video_info:
        return False
    
    print(f"\n🎵 Now playing: {video_info['title']}")
    print(f"📺 Channel: {video_info.get('uploader', 'Unknown')}")
    if video_info.get('duration'):
        duration = video_info['duration']
        print(f"⏱️ Duration: {duration//60}:{duration%60:02d}")
    
    temp_dir = None
    try:
        import yt_dlp
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, '%(title)s.%(ext)s')
        
        # Configure yt-dlp with SSL certificates
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_template,
            'quiet': False,  # Show progress
            'no_warnings': False,
            'ssl_verify': True,
            'ssl_cert_file': certifi.where(),
            'verbose': True,  # Show detailed output for debugging
        }
        
        print("\n📥 Downloading audio...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_info['url']])
        
        # Find the downloaded file
        for file in os.listdir(temp_dir):
            if file.endswith('.mp3'):
                audio_file = os.path.join(temp_dir, file)
                file_size = os.path.getsize(audio_file) / (1024*1024)
                print(f"✅ Downloaded: {file} ({file_size:.1f} MB)")
                print("▶️ Playing... (Press Ctrl+C to stop)")
                
                # Play with afplay
                subprocess.run(['afplay', audio_file])
                
                # Clean up
                import shutil
                shutil.rmtree(temp_dir)
                return True
        
    except KeyboardInterrupt:
        print("\n⏹️ Playback stopped")
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        print(f"❌ Playback error: {e}")
        if temp_dir:
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
    
    return False

def check_dependencies():
    """Check if all dependencies are installed"""
    print("🔧 Checking dependencies...")
    
    # Check Python version
    import platform
    py_version = platform.python_version()
    print(f"   Python version: {py_version}")
    if tuple(map(int, py_version.split('.'))) < (3, 10):
        print("   ⚠️ Warning: Python 3.10+ recommended")
    
    # Check certifi
    try:
        import certifi
        print(f"   ✅ certifi: {certifi.__version__}")
    except:
        print("   ❌ certifi not installed")
    
    # Check yt-dlp
    try:
        import yt_dlp
        print(f"   ✅ yt-dlp: {yt_dlp.version.__version__}")
    except:
        print("   ❌ yt-dlp not installed")
    
    # Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"   ✅ FFmpeg: {version_line[:50]}...")
        else:
            print("   ❌ FFmpeg not working")
    except:
        print("   ❌ FFmpeg not found")
    
    print("")

if __name__ == "__main__":
    # Check dependencies first
    check_dependencies()
    
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = input("Enter song name: ")
    
    video = search_youtube(query)
    if video:
        play_from_youtube(video)
    else:
        print("\n❌ Could not find the song.")
        print("\nAlternative options:")
        print("1. Try a different song name")
        print("2. Use the music fetcher with other sources:")
        print("   /Users/davidmiles/Desktop/dmai-system/scripts/music_fetcher_fixed.py")
