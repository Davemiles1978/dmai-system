#!/usr/bin/env python3
"""Launcher that runs both evolution and telegram in parallel"""
import threading
import time
import subprocess
import sys
from pathlib import Path

def run_evolution():
    """Run the memory-safe evolution"""
    print("🚀 Starting Evolution System...")
    import evolution.continuous_advanced_evolution
    # This will block, which is fine - it runs forever

def run_telegram():
    """Run Telegram bot"""
    print("📱 Starting Telegram Bot...")
    import telegram_reporter
    # This will block, which is fine - it runs forever

if __name__ == "__main__":
    print("="*60)
    print("🔄 DMAI DUAL LAUNCHER")
    print("Running both Evolution and Telegram in parallel")
    print("="*60)
    
    # Start Telegram in a separate thread
    telegram_thread = threading.Thread(target=run_telegram, daemon=True)
    telegram_thread.start()
    print("✅ Telegram bot started in background thread")
    
    # Give Telegram a moment to initialize
    time.sleep(2)
    
    # Run evolution in main thread (will block)
    print("✅ Evolution system starting in main thread\n")
    run_evolution()
