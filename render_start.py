#!/usr/bin/env python3
"""
Render entry point - Starts DMAI as ONE intelligence
"""
import os
import sys
import time
import logging
from pathlib import Path

# Configure logging for Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - RENDER[DMAI] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('render')

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("🚀 Starting DMAI on Render")
    logger.info("="*60)
    
    # Import the clean core
    from dmai_core_clean import DMAIIntelligence, create_web_server
    
    # Initialize DMAI
    dmai = DMAIIntelligence()
    
    # Start DMAI in background thread
    import threading
    def run_dmai():
        dmai.run()
    
    dmai_thread = threading.Thread(target=run_dmai, daemon=True)
    dmai_thread.start()
    
    # Start web server
    app = create_web_server()
    if app:
        port = int(os.environ.get('PORT', 5001))
        logger.info(f"🌐 Web interface starting on port {port}")
        app.run(host='0.0.0.0', port=port)
    else:
        logger.error("Failed to create web server")
        sys.exit(1)
