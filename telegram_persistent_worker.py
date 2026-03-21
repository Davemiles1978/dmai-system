#!/usr/bin/env python3
"""
DMAI Persistent Telegram Worker
Runs 24/7 with auto-restart capability
Sends status updates and receives commands
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - DMAI-TELEGRAM - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('telegram_persistent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DMAI-TELEGRAM')

class PersistentTelegramWorker:
    """Self-healing Telegram worker that runs forever"""
    
    def __init__(self):
        self.running = True
        self.bot_process = None
        self.restart_count = 0
        self.last_restart = None
        self.heartbeat_file = Path("data/telegram_heartbeat.json")
        
    def start_bot(self):
        """Start the telegram bot as a subprocess"""
        try:
            logger.info("🤖 Starting Telegram bot...")
            
            # Send startup notification
            self._send_startup_message()
            
            # Start the bot
            self.bot_process = subprocess.Popen(
                [sys.executable, "telegram_worker.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Update heartbeat
            self._update_heartbeat("started")
            self.restart_count += 1
            self.last_restart = datetime.now().isoformat()
            
            logger.info(f"✅ Telegram bot started (PID: {self.bot_process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start bot: {e}")
            return False
    
    def monitor_bot(self):
        """Monitor bot and restart if needed"""
        while self.running:
            if self.bot_process:
                # Check if bot is still running
                poll = self.bot_process.poll()
                
                if poll is not None:
                    # Bot died, restart it
                    logger.warning(f"⚠️ Bot died with code {poll}. Restarting...")
                    self._send_crash_message(poll)
                    self.start_bot()
                else:
                    # Bot is running, update heartbeat
                    self._update_heartbeat("running")
                    
                    # Check bot responsiveness
                    self._check_bot_health()
            
            # Wait before next check
            time.sleep(30)
    
    def _check_bot_health(self):
        """Check if bot is responding"""
        try:
            import requests
            import os
            
            token = os.environ.get('TELEGRAM_BOT_TOKEN')
            if token:
                url = f"https://api.telegram.org/bot{token}/getMe"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    logger.debug("✅ Bot is responsive")
                    self._update_heartbeat("healthy")
                else:
                    logger.warning("⚠️ Bot health check failed")
                    
        except Exception as e:
            logger.warning(f"Health check error: {e}")
    
    def _update_heartbeat(self, status: str):
        """Update heartbeat file for other processes"""
        try:
            self.heartbeat_file.parent.mkdir(exist_ok=True)
            with open(self.heartbeat_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "status": status,
                    "pid": self.bot_process.pid if self.bot_process else None,
                    "restart_count": self.restart_count,
                    "last_restart": self.last_restart
                }, f, indent=2)
        except:
            pass
    
    def _send_startup_message(self):
        """Send startup notification via Telegram"""
        try:
            import requests
            import os
            
            token = os.environ.get('TELEGRAM_BOT_TOKEN')
            chat_id = os.environ.get('TELEGRAM_CHAT_ID')
            
            if token and chat_id:
                message = """
🧬 DMAI TELEGRAM BOT STARTED

✅ Bot is now running 24/7
📱 You can send commands:
   /status - Check DMAI status
   /repair_ui - Fix web interface
   /evolve - Trigger evolution
   /health - Health check

DMAI will now self-heal and monitor all systems.
"""
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
                logger.info("📨 Startup message sent")
        except Exception as e:
            logger.error(f"Failed to send startup message: {e}")
    
    def _send_crash_message(self, exit_code):
        """Send crash notification"""
        try:
            import requests
            import os
            
            token = os.environ.get('TELEGRAM_BOT_TOKEN')
            chat_id = os.environ.get('TELEGRAM_CHAT_ID')
            
            if token and chat_id:
                message = f"""
⚠️ DMAI TELEGRAM BOT CRASHED

Exit code: {exit_code}
Restart count: {self.restart_count + 1}
Time: {datetime.now().isoformat()}

Auto-restarting in 10 seconds...
"""
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
        except:
            pass
    
    def run(self):
        """Main run loop"""
        logger.info("=" * 60)
        logger.info("DMAI Persistent Telegram Worker Starting")
        logger.info("=" * 60)
        
        # Start the bot
        if self.start_bot():
            # Monitor and keep alive
            self.monitor_bot()
    
    def stop(self):
        """Stop the worker"""
        self.running = False
        if self.bot_process:
            self.bot_process.terminate()
            self.bot_process.wait(timeout=10)
        logger.info("🛑 Telegram worker stopped")

if __name__ == "__main__":
    worker = PersistentTelegramWorker()
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.stop()
        logger.info("Exited by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
