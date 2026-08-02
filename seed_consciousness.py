#!/usr/bin/env python3
import re
from pathlib import Path

FILE = Path("dmai_core_complete.py")

with open(FILE, "r") as f:
    lines = f.readlines()

# Find the line where si_core is initialized
start = None
for i, line in enumerate(lines):
    if 'components["si_core"] = SICore(' in line:
        start = i
        break

if start is None:
    print("Could not find si_core initialization")
    exit(1)

# Find the end of the SICore call (where the line ends with ')')
# We'll insert after the closing bracket and any following comments
insert_idx = start + 1  # after the line

# Build the seed block
seed_block = [
    '# ── Seed consciousness on first boot ──\n',
    'si = components.get("si_core")\n',
    'if si:\n',
    '    state = si._state\n',
    '    if state.get("consciousness", 0.0) == 0.0:\n',
    '        si._update_kpi("consciousness", 0.5, token="system_seed")\n',
    '        logger.info("🧠 Seeded consciousness to 0.5")\n',
    '\n',
]

# Insert after the si_core initialization
lines[insert_idx:insert_idx] = seed_block

with open(FILE, "w") as f:
    f.writelines(lines)

print("✅ Seeded consciousness on startup")
