#!/usr/bin/env python3
"""
DMAI Cloud UI - Flask application serving static files and health checks
"""
import os
import json
from datetime import datetime
from flask import Flask, send_from_directory, jsonify

# Track start time
start_time = datetime.now()

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Flask app
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')

@app.route('/')
def serve_index():
    """Serve the main index.html"""
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(BASE_DIR, path)

@app.route('/health')
def health_check():
    """Health check endpoint for cron-job.org"""
    return jsonify({
        "status": "healthy",
        "service": "dmai-cloud-ui",
        "timestamp": datetime.now().isoformat(),
        "uptime": str(datetime.now() - start_time)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"🚀 Starting DMAI Cloud UI on port {port}")
    print(f"   Health endpoint: http://localhost:{port}/health")
    app.run(host='0.0.0.0', port=port, debug=False)
