#!/usr/bin/env python3
"""Minimal startup wrapper for DMAI - ensures Flask starts first"""
import os
import sys
import threading
import time

# Disable problematic features
os.environ['DISABLE_NEO4J'] = 'true'
os.environ['DISABLE_VOICE'] = 'true'

# Import the main app from the fixed file
sys.path.insert(0, os.getcwd())
from dmai_core_complete_fixed import get_dmai_app

print("=" * 60)
print("Starting DMAI Full System")
print("=" * 60)

# Get the Flask app
dmai_app = get_dmai_app()
app = dmai_app.app

# Start Flask in a background thread
def run_flask():
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting Flask on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

print(f"Flask server starting on http://localhost:{os.environ.get('PORT', 5001)}")
print("Press Ctrl+C to stop")
print("-" * 60)

# Keep main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down DMAI...")
    sys.exit(0)
