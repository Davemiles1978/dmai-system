#!/usr/bin/env python3
"""
DMAI AUTOPILOT BUILDER - Builds everything automatically without stopping
"""

from test_aware_builder_fixed import TestAwareBuilder
import time

class AutopilotBuilder(TestAwareBuilder):
    def run_autopilot(self):
        """Keep building until everything is done"""
        print("\n" + "="*80)
        print("🚀 DMAI AUTOPILOT MODE - Building continuously")
        print("="*80)
        
        self.load_roadmap()
        
        while True:
            self.print_status()
            
            # Get next component
            next_comp = self.get_next_component()
            if not next_comp:
                print("\n🎉 ALL COMPONENTS BUILT AND TESTED!")
                break
            
            comp = self.components[next_comp]
            print(f"\n🤖 Autopilot building: {comp['name']}")
            
            # Build automatically (no confirmation)
            success = self.build_component(next_comp)
            
            if not success:
                print(f"\n⚠️  Build failed for {comp['name']}")
                print("   Switching to manual mode for troubleshooting")
                break
            
            time.sleep(2)  # Brief pause between builds
        
        self.print_status()
        print(f"\n💾 Final state saved to {self.state_file}")

if __name__ == "__main__":
    builder = AutopilotBuilder()
    
    # Ask which mode
    print("\nSelect mode:")
    print("  [m] Manual (approve each component)")
    print("  [a] Autopilot (build everything automatically)")
    
    choice = input("\nChoice: ").lower()
    
    if choice == 'a':
        builder.run_autopilot()
    else:
        builder.run()  # Original manual mode
