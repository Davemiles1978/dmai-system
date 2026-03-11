#!/usr/bin/env python3
"""
Fix for music_learner.py - Add proper health check endpoint and error handling
"""

import os
import sys
from pathlib import Path

music_learner_path = Path("/Users/davidmiles/Desktop/dmai-system/music/music_learner.py")

if not music_learner_path.exists():
    print(f"❌ Music learner not found at {music_learner_path}")
    sys.exit(1)

# Read current content
with open(music_learner_path, 'r') as f:
    content = f.read()

# Check if health endpoint already exists
if '"/health"' in content or '"/status"' in content:
    print("✅ Health endpoint already exists")
else:
    # Add Flask health endpoint if Flask is used
    if 'from flask import Flask' in content:
        # Add health endpoint before the main block
        new_content = content.replace(
            'if __name__ == "__main__":',
            '''
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for daemon monitoring"""
    return jsonify({'status': 'healthy', 'service': 'music_learner'})

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint with detailed info"""
    return jsonify({
        'status': 'running',
        'service': 'music_learner',
        'timestamp': str(datetime.now())
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
        
        with open(music_learner_path, 'w') as f:
            f.write(new_content)
        
        print("✅ Added health check endpoints to music_learner.py")
    else:
        print("⚠️ Music learner doesn't use Flask, manual check needed")

# Also add better error handling
if 'try:' not in content:
    # Wrap main execution in try-except
    new_content = content.replace(
        'if __name__ == "__main__":',
        '''if __name__ == "__main__":
    try:'''
    ).replace(
        'main()',
        '''    main()
    except Exception as e:
        import traceback
        import sys
        print(f"❌ Fatal error in music_learner: {e}")
        traceback.print_exc()
        sys.exit(1)'''
    )
    
    with open(music_learner_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Added error handling to music_learner.py")

print("\n🎵 Music learner fixes applied. Now restart the daemon.")
