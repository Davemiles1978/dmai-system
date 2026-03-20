#!/usr/bin/env python3
"""Check memory usage of evolution system"""
import psutil
import os
import gc

process = psutil.Process(os.getpid())
print(f"Current memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Check number of systems
from pathlib import Path
evolved = list(Path('agents/evolved').iterdir())
print(f"Evolved systems: {len(evolved)}")

# Suggest cleanup if needed
if len(evolved) > 40:
    print("⚠️ High system count - cleanup recommended")
