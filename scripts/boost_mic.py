#!/usr/bin/env python3
"""
Microphone booster for DMAI voice
"""
import subprocess
import sys

def boost_mic():
    """Set macOS microphone to max"""
    try:
        subprocess.run(['osascript', '-e', 'set volume input volume 100'], check=True)
        print("✅ Microphone boosted to 100%")
    except:
        print("⚠️ Could not set mic volume automatically")

def check_mic_level():
    """Check current mic level"""
    try:
        result = subprocess.run(['osascript', '-e', 'input volume of (get volume settings)'], 
                              capture_output=True, text=True)
        level = result.stdout.strip()
        print(f"📊 Current microphone level: {level}%")
        return int(level)
    except:
        return None

if __name__ == "__main__":
    check_mic_level()
    boost_mic()
    check_mic_level()
