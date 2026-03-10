#!/usr/bin/env python3
"""Launcher that runs both evolution and telegram in parallel"""
import threading
import time
import sys
import traceback
import os
from pathlib import Path

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evolution"))

print(f"📂 Project root: {PROJECT_ROOT}")
print(f"📂 Python path: {sys.path}")

def run_evolution():
    """Run the memory-safe evolution"""
    print("🚀 Starting Evolution System...")
    try:
        from continuous_advanced_evolution import main as evolution_main
        evolution_main()
    except Exception as e:
        print(f"❌ Evolution error: {e}")
        traceback.print_exc()

def run_telegram():
    """Run Telegram bot with error handling and restart"""
    print("📱 Starting Telegram Bot...")
    while True:  # Add restart loop
        try:
            import telegram_reporter
            print("✅ Telegram module imported successfully")
            # The module runs its own infinite loop
            # If it exits, we'll catch and restart
            break
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            traceback.print_exc()
            print("🔄 Restarting Telegram in 10 seconds...")
            time.sleep(10)

def monitor_threads(telegram_thread):
    """Monitor threads and restart if needed"""
    while True:
        if not telegram_thread.is_alive():
            print("❌ Telegram thread died - restarting...")
            new_thread = threading.Thread(target=run_telegram, daemon=True)
            new_thread.start()
            telegram_thread = new_thread
        time.sleep(30)

if __name__ == "__main__":
    print("="*60)
    print("🔄 DMAI DUAL LAUNCHER")
    print("Running both Evolution and Telegram in parallel")
    print("="*60)
    
    # Check if files exist
    telegram_file = PROJECT_ROOT / "telegram_reporter.py"
    evolution_dir = PROJECT_ROOT / "evolution"
    
    print(f"📄 telegram_reporter.py exists: {telegram_file.exists()}")
    print(f"📁 evolution directory exists: {evolution_dir.exists()}")
    if evolution_dir.exists():
        print(f"📄 continuous_advanced_evolution.py exists: {(evolution_dir / 'continuous_advanced_evolution.py').exists()}")
    
    # Start Telegram in a separate thread
    telegram_thread = threading.Thread(target=run_telegram, daemon=True)
    telegram_thread.start()
    print("✅ Telegram bot started in background thread")
    
    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_threads, args=(telegram_thread,), daemon=True)
    monitor_thread.start()
    
    # Give Telegram a moment to initialize
    time.sleep(2)
    
    # Run evolution in main thread (will block)
    print("✅ Evolution system starting in main thread\n")
    run_evolution()
