#!/usr/bin/env python3
"""
Add debug logging to _get_next_v4_module to see what it's returning.
"""

from pathlib import Path

FILE = Path("components/evolution/StageAwareLearningOrchestrator.py")

with open(FILE, "r") as f:
    lines = f.readlines()

# Find the _get_next_v4_module method
start_idx = None
for i, line in enumerate(lines):
    if "def _get_next_v4_module(self):" in line:
        start_idx = i
        break

if start_idx is None:
    print("Could not find _get_next_v4_module method")
    exit(1)

# Find the end of the method (the next method definition)
end_idx = None
for i in range(start_idx + 1, len(lines)):
    if lines[i].strip().startswith("def ") and "def _get_next_v4_module" not in lines[i]:
        end_idx = i
        break

if end_idx is None:
    end_idx = len(lines)

# Insert debug logging after the method start
debug_line = '        logger.info(f"V4: Checking progress file: {v4_file}")\n'
debug_line2 = '        logger.info(f"V4: Found {len(progress)} modules, checking for next")\n'

# Remove the old method and rebuild with debug logging
new_method = [
    "    def _get_next_v4_module(self):\n",
    '        """Check V4 progress file for the next unmastered module."""\n',
    "        import json\n",
    "        from pathlib import Path\n",
    "        v4_file = Path(\"data/v4_progress.json\")\n",
    "        if not v4_file.exists():\n",
    "            logger.info(\"V4: progress file not found\")\n",
    "            return None\n",
    "        try:\n",
    "            with open(v4_file) as f:\n",
    "                progress = json.load(f)\n",
    "            logger.info(f\"V4: Progress file loaded, keys: {list(progress.keys())}\")\n",
    "            for mod_id, data in progress.items():\n",
    '                if data.get("status") in ("not_started", "in_progress") and data.get("pct", 0) < 100:\n',
    f'                    logger.info(f"V4: Returning module: {{mod_id}} with status {{data.get(\"status\")}} and pct {{data.get(\"pct\", 0)}}")\n',
    "                    return {\n",
    '                        "topic": mod_id,\n',
    '                        "category": "v4_self_evolution",\n',
    '                        "is_accelerator": False,\n',
    '                        "mastery_threshold": 3,\n',
    "                    }\n",
    "            logger.info(\"V4: No unmastered modules found\")\n",
    "        except Exception as e:\n",
    "            logger.warning(f\"V4: Error reading progress file: {e}\")\n",
    "        return None\n",
]

# Replace the method
lines[start_idx:end_idx] = new_method

with open(FILE, "w") as f:
    f.writelines(lines)

print("✅ Added debug logging to _get_next_v4_module")
