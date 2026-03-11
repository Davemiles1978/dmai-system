#!/usr/bin/env python3
"""
Fix for evolution_engine.py - Ensure it runs in server mode with health endpoint
"""

import os
import sys
from pathlib import Path

evolution_path = Path("/Users/davidmiles/Desktop/dmai-system/evolution/continuous_advanced_evolution.py")

if not evolution_path.exists():
    print(f"❌ Evolution engine not found at {evolution_path}")
    sys.exit(1)

# Read current content
with open(evolution_path, 'r') as f:
    content = f.read()

# Check if it's starting in server mode
if '--server' not in content and 'run_server' in content:
    # Modify the main block to always run in server mode
    new_content = content.replace(
        'if __name__ == "__main__":',
        '''if __name__ == "__main__":
    # Always run in server mode for daemon management
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--server':
        # Already in server mode
        pass
    else:
        # Add --server argument
        sys.argv.append('--server')'''
    )
    
    with open(evolution_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Fixed evolution engine to always run in server mode")
else:
    print("✅ Evolution engine already has server mode")

# Add better error handling
if 'try:' not in content.split('if __name__')[1]:
    # Add try-except around main execution
    new_content = content.replace(
        'if __name__ == "__main__":',
        '''if __name__ == "__main__":
    try:'''
    ).replace(
        'run_server()',
        '''        run_server()
    except Exception as e:
        import traceback
        import sys
        print(f"❌ Fatal error in evolution_engine: {e}")
        traceback.print_exc()
        sys.exit(1)'''
    )
    
    with open(evolution_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Added error handling to evolution_engine.py")

print("\n🧬 Evolution engine fixes applied.")
