#!/usr/bin/env python3
"""
DMAI Unified Launcher - Runs web UI and Telegram bot together
"""

import os
import sys
import threading
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - DMAI - %(levelname)s - %(message)s')
logger = logging.getLogger('DMAI')

def run_web():
    """Run web interface"""
    try:
        from dmai_web import app
        port = int(os.environ.get('PORT', 5001))
        logger.info(f"🌐 Web interface starting on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Web error: {e}")

def run_telegram():
    """Run Telegram bot"""
    try:
        from telegram_master_control import DMAIMasterControl
        logger.info("📱 Telegram bot starting")
        controller = DMAIMasterControl()
        controller.run()
    except Exception as e:
        logger.error(f"Telegram error: {e}")

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("🧬 DMAI UNIFIED SYSTEM")
    logger.info("=" * 50)
    
    # Start both in threads
    threads = []
    
    web_thread = threading.Thread(target=run_web, daemon=True)
    web_thread.start()
    threads.append(web_thread)
    logger.info("✅ Web thread started")
    
    tele_thread = threading.Thread(target=run_telegram, daemon=True)
    tele_thread.start()
    threads.append(tele_thread)
    logger.info("✅ Telegram thread started")
    
    logger.info("🎉 DMAI is running - Web + Telegram together")
    
    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
