#!/usr/bin/env python3
"""
Fix for dual_launcher.py - Ensure it doesn't conflict with evolution engine
"""

import os
import sys
from pathlib import Path

dual_path = Path("/Users/davidmiles/Desktop/dmai-system/evolution/dual_launcher.py")

if not dual_path.exists():
    print(f"❌ Dual launcher not found at {dual_path}")
    sys.exit(1)

# Read current content
with open(dual_path, 'r') as f:
    content = f.read()

# Check if it's trying to run evolution engine on the same port
if '9003' in content:
    # Change port to avoid conflict (dual launcher should use 9009)
    new_content = content.replace('9003', '9009')
    
    with open(dual_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Fixed port conflict in dual_launcher.py")
else:
    print("✅ Dual launcher port seems correct")

# Add health endpoint
if '"/health"' not in content:
    # Add health endpoint if Flask is used
    if 'from flask import Flask' in content:
        new_content = content.replace(
            'if __name__ == "__main__":',
            '''
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for daemon monitoring"""
    return jsonify({'status': 'healthy', 'service': 'dual_launcher'})

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint for daemon monitoring"""
    return jsonify({
        'status': 'running',
        'service': 'dual_launcher',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == "__main__":
'''
        )
        
        # Add imports if needed
        if 'from datetime import datetime' not in new_content:
            new_content = new_content.replace(
                'from flask import Flask',
                'from flask import Flask, jsonify\nfrom datetime import datetime'
            )
        
        with open(dual_path, 'w') as f:
            f.write(new_content)
        
        print("✅ Added health endpoints to dual_launcher.py")

print("\n🔄 Dual launcher fixes applied.")
