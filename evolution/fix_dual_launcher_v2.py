#!/usr/bin/env python3
"""
Fix for dual_launcher.py - Properly import evolution engine with correct indentation
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

# Let's see what's around line 23
lines = content.split('\n')
print(f"Line 23 is: {lines[22] if len(lines) > 22 else 'N/A'}")

# Fix the entire run_evolution function
old_run = """def run_evolution():
    \"\"\"Run the evolution system in a separate process\"\"\"
    try:
        logger.info("🚀 Starting Evolution System...")
        evolution_main()  # This will block
    except Exception as e:
        logger.error(f"❌ Evolution error: {e}")
        logger.error(traceback.format_exc())"""

new_run = '''def run_evolution():
    """Run the evolution system in a separate process"""
    try:
        logger.info("🚀 Starting Evolution System...")
        # Import evolution engine
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Create and run evolution engine
        from continuous_advanced_evolution import EvolutionEngine
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
        logger.error(traceback.format_exc())'''

# Replace the function
if old_run in content:
    content = content.replace(old_run, new_run)
    print("✅ Replaced run_evolution function")
else:
    print("⚠️ Could not find exact match, trying pattern match...")
    # Try regex as fallback
    import re
    pattern = r'def run_evolution\(\):.*?logger\.error\(traceback\.format_exc\(\)\)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        content = content.replace(match.group(0), new_run)
        print("✅ Replaced run_evolution function using regex")
    else:
        print("❌ Could not find run_evolution function")

# Add traceback import if missing
if 'import traceback' not in content:
    content = content.replace(
        'import logging',
        'import logging\nimport traceback'
    )
    print("✅ Added traceback import")

# Remove any stray imports at wrong indentation
lines = content.split('\n')
fixed_lines = []
in_function = False
function_indent = 0

for line in lines:
    # Track if we're inside a function
    if line.strip().startswith('def ') and line.strip().endswith(':'):
        in_function = True
        function_indent = len(line) - len(line.lstrip())
        fixed_lines.append(line)
    elif in_function:
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= function_indent and line.strip():
            in_function = False
            fixed_lines.append(line)
        else:
            # Inside function, check for misplaced imports
            if 'import' in line and current_indent == function_indent + 4:
                # This is a proper import inside function
                fixed_lines.append(line)
            elif 'import' in line and current_indent == 0:
                # This import is at wrong level, indent it
                fixed_lines.append('    ' + line.lstrip())
            else:
                fixed_lines.append(line)
    else:
        fixed_lines.append(line)

content = '\n'.join(fixed_lines)

# Write back
with open(dual_path, 'w') as f:
    f.write(content)

print("\n✅ Fixed dual_launcher.py. You can now test it with:")
print("python3 evolution/dual_launcher.py")
