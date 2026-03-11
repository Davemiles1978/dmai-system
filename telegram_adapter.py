#!/usr/bin/env python3
"""
Adapter for DMAI Telegram Bot
Makes the existing DMAITelegramBot class available as DMAITelegramReporter
for compatibility with dual_launcher
"""

import sys
import time
import threading
from pathlib import Path

# Import the original bot
sys.path.insert(0, str(Path(__file__).parent))
from telegram_reporter import DMAITelegramBot

class DMAITelegramReporter:
    """Adapter class that wraps DMAITelegramBot for dual_launcher compatibility"""
    
    def __init__(self, token=None, chat_id=None):
        """Initialize using the original bot"""
        # If token and chat_id are provided, set them as env vars for the original bot
        if token:
            import os
            os.environ['TELEGRAM_TOKEN'] = token
        if chat_id:
            import os
            os.environ['TELEGRAM_CHAT_ID'] = chat_id
        
        # Initialize the original bot
        self.bot = DMAITelegramBot()
        print(f"✅ Telegram adapter initialized with bot class: {self.bot.__class__.__name__}")
        
        # Check what methods are available
        print(f"📋 Available methods: {[m for m in dir(self.bot) if not m.startswith('_')][:10]}...")
    
    def run(self):
        """Start the bot (compatibility method for dual_launcher)"""
        print("🤖 Starting Telegram bot via adapter...")
        
        # The original bot uses run_polling() method
        if hasattr(self.bot, 'run_polling'):
            print("✅ Using bot.run_polling()")
            self.bot.run_polling()
        elif hasattr(self.bot, 'check_for_commands'):
            # Fallback: create a simple polling loop
            print("✅ Using custom polling loop with bot.check_for_commands()")
            try:
                while True:
                    self.bot.check_for_commands()
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 Telegram bot stopped")
        else:
            print("⚠️ No polling method found, bot will be passive")
            # Keep thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    
    def send_message(self, text, parse_mode=None):
        """Forward send_message to original bot"""
        if hasattr(self.bot, 'send_message'):
            return self.bot.send_message(text, parse_mode)
        return None
    
    def __getattr__(self, name):
        """Forward any other attributes to the original bot"""
        return getattr(self.bot, name)

if __name__ == "__main__":
    # Test the adapter
    print("🧪 Testing Telegram adapter...")
    adapter = DMAITelegramReporter()
    print(f"✅ Adapter created successfully")
    print(f"✅ Bot instance: {adapter.bot}")
    
    # Try to send a test message
    if hasattr(adapter.bot, 'send_message'):
        result = adapter.bot.send_message("🧪 Test message from DMAI adapter")
        print(f"✅ Test message sent: {result}")
    else:
        print("⚠️ send_message method not found")
