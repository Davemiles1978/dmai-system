#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Test DMAI learning announcement"""
import time
from language_learning.integrate_with_voice import DMAIWithLearning

print("🎯 Testing DMAI Learning Announcement")
print("="*50)

dmai = DMAIWithLearning()
dmai.start_learning()

print("\nDMAI will now listen for 30 seconds...")
print("Try speaking some English phrases naturally")
print("(She'll announce when she learns new words)\n")

try:
    time.sleep(30)
finally:
    print("\n" + "="*50)
    print(dmai.get_learning_summary())
    dmai.stop_learning()
