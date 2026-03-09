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

def get_all_systems_debug(self):
    """Get systems with memory tracking"""
    import tracemalloc
    tracemalloc.start()
    
    systems = self.get_all_systems_debug()
    current, peak = tracemalloc.get_traced_memory()
    
    print(f"📊 Systems loaded: {len(systems)}")
    print(f"   Memory for systems: {current / 1024 / 1024:.1f} MB")
    print(f"   Peak memory: {peak / 1024 / 1024:.1f} MB")
    
    # If too many, log which ones are biggest
    if len(systems) > 30:
        print("⚠️ High system count - top 5 by name:")
        for s in sorted(systems, key=lambda x: len(x.get('name', '')))[:5]:
            print(f"   • {s.get('name', 'unknown')}")
    
    return systems

def get_all_systems_debug(self):
    """Get systems with memory tracking"""
    import tracemalloc
    tracemalloc.start()
    
    systems = self.get_all_systems_debug()
    current, peak = tracemalloc.get_traced_memory()
    
    print(f"📊 Systems loaded: {len(systems)}")
    print(f"   Memory for systems: {current / 1024 / 1024:.1f} MB")
    print(f"   Peak memory: {peak / 1024 / 1024:.1f} MB")
    
    # If too many, log which ones are biggest
    if len(systems) > 30:
        print("⚠️ High system count - top 5 by name:")
        for s in sorted(systems, key=lambda x: len(x.get('name', '')))[:5]:
            print(f"   • {s.get('name', 'unknown')}")
    
    return systems

# Replace the original get_all_systems with debug version
# Add this line in run_cycle:
# systems = self.get_all_systems_debug()
