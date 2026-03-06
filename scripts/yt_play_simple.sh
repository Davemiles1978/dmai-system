#!/bin/bash
# Even simpler - uses your browser's current cookies

if [ $# -eq 0 ]; then
    echo "Usage: $0 'song name'"
    exit 1
fi

QUERY="$*"
echo "🎵 Playing: $QUERY"
echo "📡 Using Safari cookies..."
echo "▶️ Press Ctrl+C to stop"
echo ""

# Use cookies from Safari directly
yt-dlp --cookies-from-browser safari \
       -f bestaudio \
       --no-playlist \
       --quiet \
       --no-warnings \
       -o - "ytsearch1:$QUERY" | afplay
