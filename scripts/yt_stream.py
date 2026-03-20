#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Stream YouTube audio directly - no download
"""

import subprocess
import sys
import signal
import os

def stream_youtube_audio(query):
    """Stream YouTube audio directly to audio player"""
    
    print(f"\n🎵 Streaming: {query}")
    print("📡 Connecting...")
    
    # Use the proxy that worked
    cmd = [
        'yt-dlp',
        '--proxy', 'socks5://127.0.0.1:9050',
        '-f', 'bestaudio',
        '--no-playlist',
        '--quiet',
        '--no-warnings',
        '-o', '-',
        f'ytsearch1:{query}'
    ]
    
    try:
        # Start yt-dlp and pipe directly to afplay
        print("▶️ Playing... (Press Ctrl+C to stop)")
        print("-" * 40)
        
        # Create processes
        yt_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        afplay_process = subprocess.Popen(['afplay'], stdin=yt_process.stdout)
        
        # Wait for playback to finish or be interrupted
        afplay_process.wait()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Playback stopped")
        # Kill processes
        yt_process.terminate()
        afplay_process.terminate()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = input("Enter song name: ")
    
    stream_youtube_audio(query)
