#!/usr/bin/env python3
"""
Stream YouTube audio directly - FIXED VERSION
"""

import subprocess
import sys
import os
import signal

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
        print("▶️ Playing... (Press Ctrl+C to stop)")
        print("-" * 40)
        
        # Create a single pipeline using shell
        pipeline = ' | '.join([
            ' '.join(cmd),
            'afplay'
        ])
        
        # Run the pipeline
        process = subprocess.Popen(pipeline, shell=True)
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Playback stopped")
        process.terminate()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = input("Enter song name: ")
    
    stream_youtube_audio(query)
