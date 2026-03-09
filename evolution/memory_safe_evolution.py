#!/usr/bin/env python3
"""Memory-safe evolution that won't crash"""
import gc
import psutil
import os
from advanced_evolution import AdvancedEvolution

class MemorySafeEvolution(AdvancedEvolution):
    """Evolution with memory protection"""
    
    def __init__(self):
        super().__init__()
        self.memory_limit_mb = 800  # Leave 200MB for system
        self.last_gc = 0
        
    def check_memory(self):
        """Check current memory usage"""
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        
        if mem_mb > self.memory_limit_mb:
            print(f"⚠️ Memory high ({mem_mb:.1f} MB), cleaning...")
            gc.collect()
            
            # If still high, reduce systems processed
            if mem_mb > self.memory_limit_mb:
                print("⚡ Reducing system count for next cycle")
                self.max_systems_per_cycle = 30
                
        return mem_mb
    
    def run_cycle(self):
        """Run cycle with memory protection"""
        mem_before = self.check_memory()
        print(f"📊 Memory before cycle: {mem_before:.1f} MB")
        
        # Run cycle
        result = super().run_cycle()
        
        # Force cleanup
        gc.collect()
        
        mem_after = self.check_memory()
        print(f"📊 Memory after cycle: {mem_after:.1f} MB")
        
        return result

if __name__ == "__main__":
    evolution = MemorySafeEvolution()
    evolution.run_cycle()
