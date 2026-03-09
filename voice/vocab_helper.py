#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Simple helper for voice system to add vocabulary
"""
import sys
import subprocess
from pathlib import Path

MANAGER_PATH = "/Users/davidmiles/Desktop/dmai-system/language_learning/data/secure/vocabulary_manager.py"

def add_words(words):
    """Add words via the manager"""
    cmd = [sys.executable, MANAGER_PATH, "add"] + words
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def check_word(word):
    """Check if word exists"""
    cmd = [sys.executable, MANAGER_PATH, "check", word]
    result = subprocess.run(cmd, capture-output=True, text=True)
    return "✅ Exists" in result.stdout

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "add":
            if len(sys.argv) > 2:
                print(add_words(sys.argv[2:]))
        elif sys.argv[1] == "check" and len(sys.argv) > 2:
            exists = check_word(sys.argv[2])
            print(exists)
