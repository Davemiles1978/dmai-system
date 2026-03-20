#!/usr/bin/env python3
"""
Fix for dual_launcher.py - Properly import evolution engine
"""

import re
from pathlib import Path

dual_path = Path("/Users/davidmiles/Desktop/dmai-system/evolution/dual_launcher.py")

if not dual_path.exists():
    print(f"❌ dual_launcher.py not found at {dual_path}")
    exit(1)

# Read current content
with open(dual_path, 'r') as f:
    content = f.read()

# Fix the import and run function
old_import = "from continuous_advanced_evolution import main as evolution_main"
new_import = """# Import evolution engine properly
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from continuous_advanced_evolution import EvolutionEngine"""

old_run = """def run_evolution():
    \"\"\"Run the evolution system in a separate process\"\"\"
    try:
        logger.info("🚀 Starting Evolution System...")
        evolution_main()  # This will block
    except Exception as e:
        logger.error(f"❌ Evolution error: {e}")
        logger.error(traceback.format_exc())"""

new_run = """def run_evolution():
    \"\"\"Run the evolution system in a separate process\"\"\"
    try:
        logger.info("🚀 Starting Evolution System...")
        # Create and run evolution engine
        engine = EvolutionEngine()
        
        # Run in server mode
        import sys
        if '--server' not in sys.argv:
            sys.argv.append('--server')
        
        # Import and run the server function
        from continuous_advanced_evolution import run_server
        run_server()
        
    except Exception as e:
        logger.error(f"❌ Evolution error: {e}")
        logger.error(traceback.format_exc())"""

# Replace the content
content = content.replace(old_import, new_import)
content = content.replace(old_run, new_run)

# Also add traceback import if missing
if 'import traceback' not in content:
    content = content.replace(
        'import logging',
        'import logging\nimport traceback'
    )

# Write back
with open(dual_path, 'w') as f:
    f.write(content)

print("✅ Fixed dual_launcher.py import and run function")
