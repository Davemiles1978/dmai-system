#!/usr/bin/env python3
"""
DMAI YouTube Music Player - Search and play from YouTube
"""

import subprocess
import sys
import json
import os
import tempfile

def search_youtube(query):
    """Search YouTube and get first result"""
    try:
        import yt_dlp
        
        print(f"🔍 Searching YouTube for: {query}")
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'force_generic_extractor': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_query = f"ytsearch1:{query} audio"
            result = ydl.extract_info(search_query, download=False)
            
            if result and 'entries' in result and result['entries']:
                video = result['entries'][0]
                video_url = f"https://youtube.com/watch?v={video['id']}"
                
                return {
                    'title': video.get('title', query),
                    'url': video_url,
                    'duration': video.get('duration', 0)
                }
    except Exception as e:
        print(f"❌ YouTube search error: {e}")
    return None

def play_from_youtube(video_info):
    """Extract audio and play from YouTube"""
    if not video_info:
        return False
    
    print(f"\n🎵 Now playing: {video_info['title']}")
    print(f"📺 Source: YouTube")
    
    try:
        import yt_dlp
        
        # Create temp file for audio
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, '%(title)s.%(ext)s')
        
        # Download audio only
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
        }
        
        print("📥 Downloading audio...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_info['url']])
        
        # Find the downloaded file
        for file in os.listdir(temp_dir):
            if file.endswith('.mp3'):
                audio_file = os.path.join(temp_dir, file)
                print(f"✅ Downloaded: {file}")
                print("▶️ Playing... (Press Ctrl+C to stop)")
                
                # Play with afplay
                subprocess.run(['afplay', audio_file])
                
                # Clean up
                import shutil
                shutil.rmtree(temp_dir)
                return True
        
    except KeyboardInterrupt:
        print("\n⏹️ Playback stopped")
        import shutil
        shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        print(f"❌ Playback error: {e}")
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
    
    return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = input("Enter song name: ")
    
    video = search_youtube(query)
    if video:
        play_from_youtube(video)
