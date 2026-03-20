import json
import os
import shutil
from datetime import datetime


# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🧬 Importing old DMAI evolution history...")

# Load old history
with open('checkpoints/evolution_history.json', 'r') as f:
    old_history = json.load(f)

print(f"Found {len(old_history)} historical evolution entries")

# Load old best scores
with open('checkpoints/best_scores.json', 'r') as f:
    old_scores = json.load(f)

# Load current generation
with open('checkpoints/current_generation.txt', 'r') as f:
    old_gen = int(f.read().strip())
print(f"Old system reached generation {old_gen}")

# Create new evolution structure
new_history = []
for i, entry in enumerate(old_history[-20:]):  # Import last 20 evolutions
    new_entry = {
        "generation": i + 1,
        "timestamp": entry.get('timestamp', datetime.now().isoformat()),
        "improvements": entry.get('improvements', []),
        "score": entry.get('score', 0.8),
        "success": entry.get('score', 0) > 0.7,
        "best_score": max([e.get('score', 0) for e in new_history + [entry]])
    }
    new_history.append(new_entry)

# Save to new system
os.makedirs('ai_core/evolution', exist_ok=True)

with open('ai_core/evolution/evolution_history.json', 'w') as f:
    json.dump(new_history, f, indent=2)

# Update generation
with open('ai_core/evolution/current_generation.txt', 'w') as f:
    f.write(str(len(new_history) + 1))  # Next generation

# Update best score
best = max([e.get('score', 0) for e in new_history])
with open('ai_core/evolution/best_score.txt', 'w') as f:
    f.write(str(best))

print(f"✅ Imported {len(new_history)} generations")
print(f"🏆 Best score: {best}")
print(f"📊 Next generation will be: {len(new_history) + 1}")

# Also copy any successful improvements
improvements_dir = 'ai_core/evolution/imported_improvements'
os.makedirs(improvements_dir, exist_ok=True)

for i, entry in enumerate(new_history):
    if entry.get('improvements'):
        with open(f"{improvements_dir}/gen_{i+1}_improvements.json", 'w') as f:
            json.dump(entry['improvements'], f, indent=2)

print(f"💾 Saved improvement files to {improvements_dir}")
