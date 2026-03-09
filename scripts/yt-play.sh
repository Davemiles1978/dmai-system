#!/bin/bash
# Simple YouTube audio player using yt-dlp binary

SCRIPT_DIR="/Users/davidmiles/Desktop/dmai-system/scripts"

# Download yt-dlp binary if not present
if [ ! -f "$SCRIPT_DIR/yt-dlp" ]; then
    echo "📥 Downloading yt-dlp binary..."
    curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos -o "$SCRIPT_DIR/yt-dlp"
    chmod +x "$SCRIPT_DIR/yt-dlp"
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg not found. Installing..."
    brew install ffmpeg
fi

# Get search query
if [ $# -eq 0 ]; then
    echo -n "Enter song name: "
    read QUERY
else
    QUERY="$*"
fi

echo "🔍 Searching for: $QUERY"

# Try multiple methods
echo "📡 Attempting to connect..."

# Method 1: With cookies from Safari
echo "   Method 1: Using Safari cookies..."
URL=$("$SCRIPT_DIR/yt-dlp" --cookies-from-browser safari --no-playlist --get-url "ytsearch1:$QUERY" 2>/dev/null)

# Method 2: If that fails, try with user agent
if [ -z "$URL" ]; then
    echo "   Method 2: Using custom user agent..."
    URL=$("$SCRIPT_DIR/yt-dlp" --user-agent "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" --no-playlist --get-url "ytsearch1:$QUERY" 2>/dev/null)
fi

# Method 3: Last resort - no certificate check
if [ -z "$URL" ]; then
    echo "   Method 3: Using no certificate check..."
    URL=$("$SCRIPT_DIR/yt-dlp" --no-check-certificate --no-playlist --get-url "ytsearch1:$QUERY" 2>/dev/null)
fi

if [ -n "$URL" ]; then
    echo "✅ Found! Playing audio..."
    echo "   (Press Ctrl+C to stop)"
    
    # Download and play
    "$SCRIPT_DIR/yt-dlp" --no-check-certificate -f bestaudio -o - "$URL" 2>/dev/null | afplay
else
    echo "❌ Could not find or access the video."
    echo ""
    echo "💡 Suggestions:"
    echo "1. Open YouTube in Safari and make sure you're logged in"
    echo "2. Try a VPN if you're in a restricted region"
    echo "3. Try again later"
fi
