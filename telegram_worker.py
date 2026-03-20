#!/usr/bin/env python3
"""
Telegram Worker - Runs alongside DMAI
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    from telegram_bot import DMAITelegramBot
    bot = DMAITelegramBot()
    bot.run()
