#!/bin/bash
# Stream YouTube audio directly - no download, no temp files

if [ $# -eq 0 ]; then
    echo "Usage: $0 'song name'"
    exit 1
fi

QUERY="$*"
echo "🎵 Streaming: $QUERY"
echo "📡 Using proxy connection..."
echo "▶️ Playing... (Press Ctrl+C to stop)"
echo ""

# Stream directly to afplay - no files saved
yt-dlp --proxy socks5://127.0.0.1:9050 \
       -f bestaudio \
       --no-playlist \
       --quiet \
       --no-warnings \
       -o - "ytsearch1:$QUERY" | afplay

echo "⏹️ Playback ended"
