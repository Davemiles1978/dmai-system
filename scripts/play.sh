#!/bin/bash
# Simple streaming player

if [ $# -eq 0 ]; then
    echo "Usage: $0 'song name'"
    exit 1
fi

QUERY="$*"
echo "🎵 Playing: $QUERY"
echo "▶️ Press Ctrl+C to stop"
echo ""

# Direct pipeline - this always works
yt-dlp --proxy socks5://127.0.0.1:9050 -f bestaudio --no-playlist --quiet -o - "ytsearch1:$QUERY" | afplay
