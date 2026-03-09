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
        # Try different import approaches
        try:
            from continuous_advanced_evolution import main as evolution_main
        except ImportError:
            from evolution.continuous_advanced_evolution import main as evolution_main
        evolution_main()
    except Exception as e:
        print(f"❌ Evolution error: {e}")
        traceback.print_exc()

def run_telegram():
    """Run Telegram bot with error handling"""
    print("📱 Starting Telegram Bot...")
    try:
        # Check if file exists first
        telegram_path = PROJECT_ROOT / "telegram_reporter.py"
        if not telegram_path.exists():
            print(f"❌ telegram_reporter.py not found at {telegram_path}")
            return
            
        # Try different import approaches
        try:
            import telegram_reporter
        except ImportError:
            sys.path.insert(0, str(PROJECT_ROOT))
            import telegram_reporter
            
        print("✅ Telegram module imported successfully")
        
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("="*60)
    print("🔄 DMAI DUAL LAUNCHER")
    print("Running both Evolution and Telegram in parallel")
    print("="*60)
    print(f"📂 Project root: {PROJECT_ROOT}")
    
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
    
    # Give Telegram a moment to initialize
    time.sleep(2)
    
    # Check if thread is still alive
    if telegram_thread.is_alive():
        print("✅ Telegram thread is running")
    else:
        print("❌ Telegram thread died - check errors above")
    
    # Run evolution in main thread (will block)
    print("✅ Evolution system starting in main thread\n")
    run_evolution()
