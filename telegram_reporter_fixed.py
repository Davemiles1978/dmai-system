#!/usr/bin/env python3
"""
Telegram Reporter for DMAI Evolution System
Reports evolution status and system health to Telegram
"""

import os
import sys
import json
import time
import logging
import requests
import threading
from datetime import datetime
from pathlib import Path

# Configure logging
log_dir = Path.home() / "Library/Logs/dmai"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - telegram - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'telegram.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('telegram_reporter')

class DMAITelegramReporter:
    """Telegram bot for DMAI evolution reporting"""
    
    def __init__(self, token=None, chat_id="6273188922"):
        """Initialize the Telegram reporter"""
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "7835051489:AAHwCg3sB3-8eqa9MBkEEkP32zblQy7FJ1Y")
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0
        self.running = False
        
        logger.info(f"🤖 Telegram bot initialized for chat_id: {chat_id}")
        
    def send_message(self, text, parse_mode='HTML'):
        """Send a message to the configured chat"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                logger.info(f"📤 Message sent: {text[:50]}...")
                return True
            else:
                logger.error(f"❌ Failed to send message: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")
            return False
    
    def send_evolution_report(self, evolution_data):
        """Send an evolution cycle report"""
        report = f"""
🧬 <b>Evolution Cycle Report</b>
📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Generation: {evolution_data.get('generation', 'N/A')}
🔄 Cycle: {evolution_data.get('cycle', 'N/A')}

📈 Statistics:
• Opportunities: {evolution_data.get('opportunities', 0)}
• Mutations tested: {evolution_data.get('mutations_tested', 0)}
• Mutations applied: {evolution_data.get('mutations_applied', 0)}

🏆 Best Score: {evolution_data.get('best_score', 0):.3f}
        """
        return self.send_message(report)
    
    def send_health_report(self, health_data):
        """Send a system health report"""
        report = f"""
🏥 <b>System Health Report</b>
📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔍 Status:
• Total weaknesses: {health_data.get('total_weaknesses', 0)}
• Fixed: {health_data.get('fixes_applied', 0)}
• Pending: {health_data.get('service_issues', 0)}

🩺 Daemon: {'✅ RUNNING' if health_data.get('daemon_healthy') else '❌ STOPPED'}
        """
        return self.send_message(report)
    
    def handle_command(self, command):
        """Handle incoming commands"""
        command = command.lower().strip()
        
        if command == '/start':
            return self.send_message("👋 Hello! I'm the DMAI Evolution Bot. Use /status to check system health.")
        
        elif command == '/status':
            # Get system status
            try:
                # Try to get evolution status
                sys.path.append(str(Path(__file__).parent))
                from core_connector import get_evolution_status
                status = get_evolution_status()
                
                msg = f"""
📊 <b>DMAI System Status</b>

🧬 Evolution:
• Generation: {status.get('generation', 'N/A')}
• Best Score: {status.get('best_score', 0):.3f}

🔄 Services:
• Evolution Engine: ✅ Running
• Telegram Bot: ✅ Running

📅 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                return self.send_message(msg)
            except:
                return self.send_message("⚠️ System status temporarily unavailable")
        
        elif command == '/help':
            help_text = """
🤖 <b>Available Commands:</b>
/start - Welcome message
/status - System status
/health - Detailed health check
/evolution - Last evolution report
/help - This message
            """
            return self.send_message(help_text)
        
        elif command == '/health':
            # Run health scanner
            try:
                sys.path.append(str(Path(__file__).parent))
                from evolution.system_weakness_scanner import SystemWeaknessScanner
                scanner = SystemWeaknessScanner()
                result = scanner.scan_and_heal()
                return self.send_health_report(result)
            except:
                return self.send_message("⚠️ Health check temporarily unavailable")
        
        elif command == '/evolution':
            # Get last evolution report
            try:
                from core_connector import get_evolution_status
                status = get_evolution_status()
                return self.send_evolution_report({
                    'generation': status.get('generation', 0),
                    'cycle': status.get('cycle', 0),
                    'opportunities': status.get('opportunities', 0),
                    'mutations_tested': status.get('mutations_tested', 0),
                    'mutations_applied': status.get('mutations_applied', 0),
                    'best_score': status.get('best_score', 0)
                })
            except:
                return self.send_message("⚠️ Evolution data temporarily unavailable")
        
        else:
            return self.send_message(f"❓ Unknown command: {command}\nUse /help for available commands.")
    
    def get_updates(self):
        """Get new messages from Telegram"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {
                'offset': self.last_update_id + 1,
                'timeout': 30
            }
            response = requests.get(url, params=params, timeout=35)
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    return data.get('result', [])
            return []
        except Exception as e:
            logger.error(f"❌ Error getting updates: {e}")
            return []
    
    def process_updates(self):
        """Process incoming messages"""
        updates = self.get_updates()
        for update in updates:
            self.last_update_id = update['update_id']
            
            if 'message' in update and 'text' in update['message']:
                text = update['message']['text']
                chat_id = update['message']['chat']['id']
                
                # Only respond to messages from our configured chat
                if str(chat_id) == self.chat_id:
                    response = self.handle_command(text)
                else:
                    logger.info(f"Ignoring message from unauthorized chat: {chat_id}")
    
    def run(self):
        """Main bot loop"""
        self.running = True
        logger.info("🤖 Telegram bot polling for commands...")
        
        while self.running:
            try:
                self.process_updates()
                time.sleep(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Error in bot loop: {e}")
                time.sleep(5)
        
        logger.info("👋 Telegram bot stopped")

if __name__ == "__main__":
    # Test the bot
    bot = DMAITelegramReporter()
    print("🤖 Testing Telegram bot...")
    bot.send_message("🧪 Test message from DMAI Telegram Reporter")
    print("✅ Test message sent. Starting bot loop...")
    bot.run()
