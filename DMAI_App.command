#!/bin/bash
# DMAI - Most Evolved AI Launcher
# This opens the full UI with projects, history, and archive

cd ~/Desktop/AI-Evolution-System
source venv/bin/activate

# Function to get the best evolved version
get_best_version() {
    python3 -c "
import json
from pathlib import Path
best_path = Path('checkpoints/best_versions')
if best_path.exists():
    for repo in best_path.iterdir():
        if repo.is_dir():
            files = list(repo.glob('*.py'))
            if files:
                print(f'Loading best version from {repo.name}')
                # This would load the actual evolved model
                # For now, just start the UI
print('DM_BEST_READY')
"
}

echo "🧬 DMAI - Loading Best Evolved Version..."
echo "========================================"

# Show evolution stats
python3 -c "
from pathlib import Path
import json
hist_file = Path('checkpoints/evolution_history.json')
if hist_file.exists():
    with open(hist_file) as f:
        hist = json.load(f)
    if hist:
        best = max(hist[-100:], key=lambda x: x.get('score', 0))
        print(f'📊 Best Score: {best.get(\"score\", 0):.3f}')
        print(f'🔄 Generation: {best.get(\"generation\", 0)}')
        print(f'📁 Repo: {best.get(\"repo\", \"unknown\")}')
"

# Launch the UI
python3 -m http.server 8888 --bind 127.0.0.1 &
SERVER_PID=$!

sleep 2
open http://localhost:8888/ai_ui.html

echo ""
echo "✅ DMAI is running!"
echo "📍 UI: http://localhost:8888"
echo "📌 Projects and chats are saved locally"
echo ""
echo "Press Ctrl+C to stop"

wait $SERVER_PID
