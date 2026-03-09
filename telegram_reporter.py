#!/usr/bin/env python3
"""Telegram Bot for DMAI Evolution Reports"""
import requests
import json
import time
import os
from datetime import datetime
from pathlib import Path

class DMAITelegramBot:
    """Sends DMAI's evolution progress to Telegram"""
    
    def __init__(self, token_file="data/telegram_token.json"):
        self.token_file = Path(token_file)
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.config = None
        self.base_url = None
        self.last_update_id = 0  # Track last processed message
        self.load_config()
        
    def load_config(self):
        """Load or create config"""
        if self.token_file.exists():
            with open(self.token_file) as f:
                self.config = json.load(f)
                self.base_url = f"https://api.telegram.org/bot{self.config['token']}"
                # Load last update ID if exists
                self.last_update_id = self.config.get('last_update_id', 0)
                print(f"✅ Loaded existing config for chat_id: {self.config['chat_id']}")
        else:
            # First run - prompt for token
            print("\n🤖 First time setup - Enter your Telegram Bot Token:")
            token = input("Token: ").strip()
            print("Enter your Telegram Chat ID (from @userinfobot):")
            chat_id = input("Chat ID: ").strip()
            
            self.config = {
                "token": token,
                "chat_id": chat_id,
                "daily_report_time": "09:00",
                "notify_on_success": True,
                "bot_username": "dmai_evolution_bot",
                "last_update_id": 0
            }
            
            # Save config first
            with open(self.token_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Then set base_url
            self.base_url = f"https://api.telegram.org/bot{token}"
            
            # Test connection
            print("📤 Sending test message...")
            result = self.send_message("🎉 DMAI Telegram Bot Connected!\nI'll report my evolution progress here.")
            
            if result and result.get('ok'):
                print("✅ Test message sent successfully!")
            else:
                print("❌ Failed to send test message. Check your token and chat ID.")
    
    def save_last_update_id(self):
        """Save last processed update ID to config"""
        self.config['last_update_id'] = self.last_update_id
        with open(self.token_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def send_message(self, text, parse_mode=None):
        """Send message to Telegram"""
        if not self.base_url or not self.config:
            print("❌ Telegram not configured properly")
            return None
            
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": self.config['chat_id'],
            "text": text
        }
        if parse_mode:
            data["parse_mode"] = parse_mode
        
        try:
            response = requests.post(url, data=data)
            if response.ok:
                print(f"✅ Message sent: {text[:50]}...")
            else:
                print(f"❌ Telegram error: {response.text}")
            return response.json()
        except Exception as e:
            print(f"❌ Telegram send failed: {e}")
            return None
    
    def send_photo(self, caption, photo_path):
        """Send a chart or image"""
        if not self.base_url or not self.config:
            return None
            
        url = f"{self.base_url}/sendPhoto"
        with open(photo_path, 'rb') as photo:
            files = {'photo': photo}
            data = {
                "chat_id": self.config['chat_id'],
                "caption": caption
            }
            try:
                response = requests.post(url, data=data, files=files)
                return response.json()
            except Exception as e:
                print(f"❌ Photo send failed: {e}")
                return None
    
    def report_evolution(self, success, stage_info):
        """Report an evolution attempt"""
        if not self.config:
            return
            
        if success:
            emoji = "🎉"
            message = f"{emoji} *DMAI EVOLVED!*\n\n"
        else:
            return  # Only report successes to reduce spam
        
        message += f"Stage: {stage_info['name']}\n"
        message += f"Evolutions: {stage_info['evolutions']}\n"
        message += f"Success Rate: {stage_info['success_rate']}\n"
        
        if success and stage_info.get('next_stage'):
            message += f"\nNext stage: {stage_info['next_stage']['name']}\n"
            message += f"Need {stage_info['next_stage']['evolutions_needed']} more evolutions"
        
        self.send_message(message, parse_mode="Markdown")
    
    def daily_report(self):
        """Send daily summary"""
        if not self.config:
            return
            
        try:
            from evolution.adaptive_timer import AdaptiveEvolutionTimer
            
            timer = AdaptiveEvolutionTimer()
            info = timer.get_stage_info()
            
            message = f"📊 *DMAI DAILY REPORT*\n"
            message += f"📅 {datetime.now().strftime('%Y-%m-%d')}\n\n"
            message += f"Stage: {info['name']}\n"
            message += f"Total Evolutions: {info['evolutions']}\n"
            message += f"Success Rate: {info['success_rate']}\n"
            message += f"Total Attempts: {timer.state['total_attempts']}\n"
            
            if info.get('preferred_pairs') and info['preferred_pairs']:
                message += f"\n🎯 Best Parent Pair:\n"
                message += f"   {info['preferred_pairs'][0]['pair']}\n"
                message += f"   ({info['preferred_pairs'][0]['success_rate']} success)"
            
            self.send_message(message, parse_mode="Markdown")
        except Exception as e:
            print(f"❌ Daily report failed: {e}")
    
    def check_for_commands(self):
        """Check for new commands only"""
        if not self.base_url:
            return
        
        # Only get updates after our last processed one
        url = f"{self.base_url}/getUpdates"
        params = {
            'offset': self.last_update_id + 1,  # Don't reprocess old messages
            'timeout': 30  # Long polling
        }
        
        try:
            response = requests.get(url, params=params)
            updates = response.json()
            
            if not updates.get('ok'):
                return
            
            for update in updates.get('result', []):
                # Update last processed ID
                self.last_update_id = max(self.last_update_id, update['update_id'])
                
                if 'message' in update and 'text' in update['message']:
                    text = update['message']['text']
                    chat_id = update['message']['chat']['id']
                    
                    # Only respond to messages from our configured chat
                    if str(chat_id) == str(self.config['chat_id']):
                        self.handle_command(text, chat_id)
            
            # Save last update ID
            if updates.get('result'):
                self.save_last_update_id()
                
        except Exception as e:
            print(f"❌ Command check failed: {e}")
    
    def handle_command(self, command, chat_id):
        """Handle user commands"""
        command = command.lower().strip()
        print(f"📨 Received command: {command}")
        
        if command == '/status':
            try:
                from evolution.adaptive_timer import AdaptiveEvolutionTimer
                timer = AdaptiveEvolutionTimer()
                info = timer.get_stage_info()
                
                message = f"🧬 *DMAI STATUS*\n\n"
                message += f"Stage: {info['name']}\n"
                message += f"Evolutions: {info['evolutions']}\n"
                message += f"Success Rate: {info['success_rate']}\n"
                message += f"Current Interval: {info['interval_minutes']:.0f} minutes"
                
                self.send_message(message, parse_mode="Markdown")
            except Exception as e:
                self.send_message(f"❌ Error getting status: {e}")
            
        elif command == '/growth':
            self.daily_report()
            
        elif command == '/help':
            message = "🤖 *DMAI Bot Commands*\n\n"
            message += "/status - Current evolution status\n"
            message += "/growth - Daily growth report\n"
            message += "/help - Show this help"
            self.send_message(message, parse_mode="Markdown")
        
        elif command == '/start':
            message = "👋 Welcome to DMAI Evolution Bot!\n\n"
            message += "I'll notify you when I evolve and send daily reports.\n"
            message += "Use /help to see all commands."
            self.send_message(message)
    
    def run_polling(self):
        """Run continuous polling for commands"""
        if not self.base_url:
            print("❌ Cannot poll - Telegram not configured")
            return
            
        print("🤖 Telegram bot polling for commands...")
        print(f"📝 Last processed update ID: {self.last_update_id}")
        
        while True:
            try:
                self.check_for_commands()
                time.sleep(1)  # Small delay to prevent CPU spinning
            except Exception as e:
                print(f"❌ Polling error: {e}")
                time.sleep(5)  # Wait longer on error

# Test the bot
if __name__ == "__main__":
    print("🚀 Starting DMAI Telegram Bot...")
    bot = DMAITelegramBot()
    
    if bot.config:
        print("✅ Telegram bot configured successfully")
        print(f"📝 Starting from update ID: {bot.last_update_id}")
        
        # Start polling for commands in background
        import threading
        polling_thread = threading.Thread(target=bot.run_polling, daemon=True)
        polling_thread.start()
        print("🤖 Polling for commands started")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down Telegram bot")
            bot.save_last_update_id()
            print(f"✅ Saved last update ID: {bot.last_update_id}")
    else:
        print("❌ Failed to configure Telegram bot")
