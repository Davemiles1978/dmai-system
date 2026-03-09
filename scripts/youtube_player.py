#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
DMAI YouTube Player - With multiple fallback methods
"""

import subprocess
import sys
import os
import tempfile
import time

def check_dependencies():
    """Check if all dependencies are available"""
    print("🔧 Checking dependencies...")
    
    # Check Python version
    import platform
    py_version = platform.python_version()
    print(f"   Python: {py_version}")
    
    # Check ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"   ✅ FFmpeg: {version_line[:50]}...")
        else:
            print("   ❌ FFmpeg not working")
            return False
    except:
        print("   ❌ FFmpeg not found")
        return False
    
    # Check yt-dlp
    try:
        result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ yt-dlp: {result.stdout.strip()}")
        else:
            print("   ❌ yt-dlp not working")
            return False
    except:
        print("   ❌ yt-dlp not found")
        return False
    
    return True

def search_with_cookies(query):
    """Search using browser cookies (most reliable)"""
    print(f"\n🔍 Method 1: Using browser cookies...")
    
    try:
        # Try to get cookies from Chrome/Safari
        cmd = [
            'yt-dlp',
            '--cookies-from-browser', 'safari',  # Try Safari first (macOS)
            '--no-playlist',
            '--print', '%(title)s|%(webpage_url)s',
            f'ytsearch3:{query}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split('\n')
            videos = []
            for line in lines:
                if '|' in line:
                    title, url = line.split('|', 1)
                    videos.append({'title': title, 'url': url})
            
            if videos:
                print(f"\n📋 Found {len(videos)} videos:")
                for i, v in enumerate(videos, 1):
                    print(f"  {i}. {v['title']}")
                
                # Return first video
                return videos[0]
    except Exception as e:
        print(f"   Cookie method failed: {e}")
    
    return None

def search_with_user_agent(query):
    """Search with custom user agent"""
    print(f"\n🔍 Method 2: Using custom user agent...")
    
    try:
        cmd = [
            'yt-dlp',
            '--user-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '--no-playlist',
            '--print', '%(title)s|%(webpage_url)s',
            f'ytsearch1:{query}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and result.stdout:
            line = result.stdout.strip()
            if '|' in line:
                title, url = line.split('|', 1)
                return {'title': title, 'url': url}
    except Exception as e:
        print(f"   User agent method failed: {e}")
    
    return None

def search_with_proxy(query):
    """Try with a public proxy (if available)"""
    print(f"\n🔍 Method 3: Trying with proxy...")
    
    # List of public proxies to try (these are examples - you may need fresh ones)
    proxies = [
        'http://203.145.0.1:8080',
        'http://103.155.54.22:80',
        'http://45.87.21.33:3128'
    ]
    
    for proxy in proxies:
        try:
            cmd = [
                'yt-dlp',
                '--proxy', proxy,
                '--no-playlist',
                '--print', '%(title)s|%(webpage_url)s',
                f'ytsearch1:{query}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and result.stdout:
                line = result.stdout.strip()
                if '|' in line:
                    title, url = line.split('|', 1)
                    print(f"   ✅ Success with proxy: {proxy}")
                    return {'title': title, 'url': url}
        except:
            continue
    
    return None

def search_without_extractor_args(query):
    """Try with minimal options"""
    print(f"\n🔍 Method 4: Using minimal options...")
    
    try:
        cmd = [
            'yt-dlp',
            '--no-check-certificate',  # Skip SSL verification (temporary)
            '--extractor-args', 'youtube:skip=dash',
            '--no-playlist',
            '--print', '%(title)s|%(webpage_url)s',
            f'ytsearch1:{query}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and result.stdout:
            line = result.stdout.strip()
            if '|' in line:
                title, url = line.split('|', 1)
                return {'title': title, 'url': url}
    except Exception as e:
        print(f"   Minimal method failed: {e}")
    
    return None

def search_youtube(query):
    """Try multiple search methods"""
    print(f"\n🎵 Searching for: {query}")
    
    # Try methods in order of reliability
    methods = [
        search_with_cookies,
        search_with_user_agent,
        search_without_extractor_args,
        search_with_proxy
    ]
    
    for method in methods:
        result = method(query)
        if result:
            return result
        time.sleep(1)  # Small delay between methods
    
    print("\n❌ All search methods failed.")
    print("\n💡 Troubleshooting tips:")
    print("   1. Open YouTube in your browser and make sure you're logged in")
    print("   2. Try a VPN if you're in a restricted region")
    print("   3. Check your internet connection")
    print("   4. Try again later - YouTube sometimes temporarily blocks requests")
    
    return None

def play_video(video_info):
    """Play the video/audio"""
    if not video_info:
        return False
    
    print(f"\n🎵 Now playing: {video_info['title']}")
    print(f"📺 URL: {video_info['url']}")
    
    temp_file = None
    try:
        # Create temp file
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, '%(title)s.%(ext)s')
        
        print("\n📥 Downloading audio...")
        
        # Download with multiple fallback options
        cmd = [
            'yt-dlp',
            '--no-check-certificate',
            '--extractor-args', 'youtube:skip=dash',
            '-f', 'bestaudio[ext=m4a]/bestaudio/best',
            '--no-playlist',
            '--quiet',
            '--output', output_template,
            video_info['url']
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Download failed: {result.stderr}")
            return False
        
        # Find downloaded file
        for file in os.listdir(temp_dir):
            if file.endswith(('.mp3', '.m4a', '.webm')):
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
        
        print("❌ No audio file found")
        return False
        
    except KeyboardInterrupt:
        print("\n⏹️ Playback stopped")
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        return False

def main():
    """Main function"""
    print("\n🎵 DMAI YOUTUBE PLAYER")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install ffmpeg:")
        print("   brew install ffmpeg")
        return
    
    # Get search query
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = input("\nEnter song name: ")
    
    if not query:
        return
    
    # Search for video
    video = search_youtube(query)
    
    # Play if found
    if video:
        play_video(video)
    else:
        print("\n❌ Could not find the song.")
        print("\nAlternative options:")
        print("1. Try a different song name")
        print("2. Open YouTube in browser and search manually")
        print("3. Try the music fetcher with other sources")

if __name__ == "__main__":
    main()
