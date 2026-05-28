"""DMAI Minimal Core - Stable version for Render deployment"""
import os
import logging
import sys
from flask import Flask, jsonify
from flask_cors import CORS

# Disable all problematic features
os.environ['DISABLE_NEO4J'] = 'true'
os.environ['DISABLE_AUTO_THREADS'] = 'true'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Simple status endpoint
@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "running",
        "version": "minimal-stable",
        "neo4j": "disabled",
        "threads": "controlled",
        "message": "DMAI is running in stable mode"
    })

# Health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

# Import smart endpoints after app is created
try:
    from dmai_smart_endpoint_stable import smart_bp
    app.register_blueprint(smart_bp, url_prefix="/v2")
    logger.info("✅ Smart endpoint registered")
except Exception as e:
    logger.error(f"Failed to load smart endpoint: {e}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
