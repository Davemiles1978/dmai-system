#!/bin/bash
# Play a random song from your DMAI music preferences

PREFERENCES="/Users/davidmiles/Desktop/AI-Evolution-System/music/preferences.json"

if [ ! -f "$PREFERENCES" ]; then
    echo "❌ No music preferences found"
    exit 1
fi

# Get a random favorite artist
ARTIST=$(python3 -c "
import json, random
with open('$PREFERENCES') as f:
    prefs = json.load(f)
artists = [a['artist'] for a in prefs.get('favorite_artists', [])]
if artists:
    print(random.choice(artists))
else:
    print('Faithless')
")

echo "🎵 Playing something from $ARTIST..."

# Search for a song by this artist
/Users/davidmiles/Desktop/AI-Evolution-System/scripts/yt_stream.sh "$ARTIST"
