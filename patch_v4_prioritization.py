#!/usr/bin/env python3
"""
Modify StageAwareLearningOrchestrator.get_next_topic to check the V4 progress file.
"""

from pathlib import Path

FILE = Path("components/evolution/StageAwareLearningOrchestrator.py")

with open(FILE, "r") as f:
    lines = f.readlines()

# Find the get_next_topic method and add a call to _get_next_v4_module at the end
# We'll insert a new method and modify the return.

# First, find where to insert the new method (after get_next_topic)
insert_idx = None
for i, line in enumerate(lines):
    if "def get_next_topic(self, consciousness: float, prioritize_accelerators: bool = True) -> Optional[Dict]:" in line:
        # Find the end of the method (the return None)
        for j in range(i, len(lines)):
            if "return None  # truly exhausted" in lines[j]:
                insert_idx = j
                break
        break

if insert_idx is None:
    print("Could not find get_next_topic method")
    exit(1)

# Insert a new method before the return None
new_method = """
    def _get_next_v4_module(self):
        \"\"\"Check V4 progress file for the next unmastered module.\"\"\"
        import json
        from pathlib import Path
        v4_file = Path(\"data/v4_progress.json\")
        if not v4_file.exists():
            return None
        try:
            with open(v4_file) as f:
                progress = json.load(f)
            for mod_id, data in progress.items():
                if data.get(\"status\") in (\"not_started\", \"in_progress\") and data.get(\"pct\", 0) < 100:
                    return {
                        \"topic\": mod_id,
                        \"category\": \"v4_self_evolution\",
                        \"is_accelerator\": False,
                        \"mastery_threshold\": 3,
                    }
        except Exception:
            pass
        return None
"""

# Insert the new method before the return None
lines.insert(insert_idx, new_method)

# Now modify the get_next_topic method to call _get_next_v4_module if no other topics found
# We need to find where it returns None and change it to call the V4 check first.
# We'll find the line that returns None and add the check before it.
for i, line in enumerate(lines):
    if "return None  # truly exhausted" in line:
        # Insert a check before this line
        lines.insert(i, "        v4_topic = self._get_next_v4_module()\n")
        lines.insert(i+1, "        if v4_topic:\n")
        lines.insert(i+2, "            return v4_topic\n")
        break

with open(FILE, "w") as f:
    f.writelines(lines)

print("✅ Patched StageAwareLearningOrchestrator to prioritize V4 modules")
