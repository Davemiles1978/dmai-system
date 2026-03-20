#!/usr/bin/env python3
"""Auto-cleanup based on memory usage"""
import psutil
import os
import subprocess

def cleanup_if_needed(threshold_mb=600):
    """Run cleanup if memory is high"""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    
    if mem_mb > threshold_mb:
        print(f"🧹 Memory high ({mem_mb:.1f} MB), running cleanup...")
        subprocess.run(["python3", "evolution/evolution_cleanup.py", "--execute"])
        return True
    return False

if __name__ == "__main__":
    cleanup_if_needed()
