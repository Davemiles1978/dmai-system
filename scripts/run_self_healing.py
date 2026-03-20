#!/usr/bin/env python3
"""
Run self-healing cycle manually
This script triggers the system weakness scanner to detect and fix issues
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from evolution.system_weakness_scanner import SystemWeaknessScanner
    from core_connector import voice_say
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Make sure you're running from the DMAI system directory")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run DMAI self-healing cycle')
    parser.add_argument('--watch', '-w', action='store_true', 
                       help='Watch mode - continuously monitor and heal')
    parser.add_argument('--interval', '-i', type=int, default=300,
                       help='Watch mode interval in seconds (default: 300)')
    parser.add_argument('--severity', '-s', type=int, default=0,
                       help='Minimum severity to attempt fix (default: 0 = all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--no-voice', action='store_true',
                       help='Disable voice announcements')
    return parser.parse_args()

def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    DMAI SELF-HEALING SYSTEM                  ║
║                    Automatic Health Scanner                  ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_result(result, verbose=False):
    """Print scan results in a nice format"""
    print("\n" + "="*60)
    print("📊 SELF-HEALING SCAN RESULTS")
    print("="*60)
    print(f"📅 Timestamp: {result.get('timestamp', 'Unknown')}")
    print(f"🔍 Total weaknesses detected: {result.get('total_weaknesses', 0)}")
    print(f"🛠️  Fixes attempted: {result.get('fixes_attempted', 0)}")
    print(f"✅ Fixes applied successfully: {result.get('fixes_applied', 0)}")
    
    if result.get('weaknesses') and verbose:
        print("\n📋 Detected Weaknesses:")
        for w in sorted(result['weaknesses'], key=lambda x: x.get('severity', 0), reverse=True):
            severity = w.get('severity', 0)
            severity_char = "🔴" if severity >= 8 else "🟠" if severity >= 5 else "🟡"
            print(f"  {severity_char} [{severity}] {w.get('type', 'unknown')} in {w.get('module', '?')}")
            print(f"     └─ {w.get('description', 'No description')}")
    
    print("\n" + "="*60)
    
    # Overall health assessment
    if result.get('total_weaknesses', 0) == 0:
        print("✅ SYSTEM HEALTH: PERFECT - No issues detected")
    elif result.get('fixes_applied', 0) == result.get('total_weaknesses', 0):
        print(f"✅ SYSTEM HEALTH: GOOD - All {result['total_weaknesses']} issues healed")
    elif result.get('fixes_applied', 0) > 0:
        print(f"⚠️ SYSTEM HEALTH: PARTIAL - Healed {result['fixes_applied']}/{result['total_weaknesses']} issues")
    else:
        print(f"❌ SYSTEM HEALTH: POOR - {result['total_weaknesses']} issues remain")
    
    print("="*60)

def run_healing_cycle(args):
    """Run a single healing cycle"""
    print("\n🔄 Running self-healing cycle...")
    
    scanner = SystemWeaknessScanner()
    
    # Override severity threshold if specified
    if args.severity > 0:
        # This would require modifying the scanner to respect severity thresholds
        print(f"⚠️ Severity filtering not yet implemented, scanning all issues")
    
    result = scanner.scan_and_heal()
    
    if not args.no_voice:
        if result['total_weaknesses'] == 0:
            voice_say("Self-healing scan complete. No issues found.")
        elif result['fixes_applied'] == result['total_weaknesses']:
            voice_say(f"Self-healing complete. All {result['total_weaknesses']} issues resolved.")
        else:
            voice_say(f"Self-healing complete. Resolved {result['fixes_applied']} out of {result['total_weaknesses']} issues.")
    
    return result

def watch_mode(args):
    """Run in watch mode - continuously monitor and heal"""
    print(f"\n👀 Watch mode enabled - scanning every {args.interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    cycle_count = 0
    
    try:
        while True:
            cycle_count += 1
            print(f"\n{'='*60}")
            print(f"📡 Watch Cycle #{cycle_count} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            result = run_healing_cycle(args)
            
            if args.verbose:
                print_result(result, verbose=True)
            else:
                # Brief summary for watch mode
                status = "✅" if result['total_weaknesses'] == 0 else "⚠️" if result['fixes_applied'] > 0 else "❌"
                print(f"{status} Cycle {cycle_count}: {result['fixes_applied']}/{result['total_weaknesses']} issues healed")
            
            print(f"\n💤 Sleeping for {args.interval} seconds...")
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Watch mode stopped by user")
        print(f"Total cycles completed: {cycle_count}")
        
        if not args.no_voice:
            voice_say(f"Self-healing watch mode stopped after {cycle_count} cycles")

def main():
    """Main entry point"""
    args = parse_arguments()
    print_banner()
    
    # Check if we're in the right directory
    if not Path("evolution/system_weakness_scanner.py").exists():
        print("❌ Error: Must be run from DMAI system root directory")
        print("   cd /Users/davidmiles/Desktop/dmai-system")
        sys.exit(1)
    
    if args.watch:
        watch_mode(args)
    else:
        # Single run mode
        result = run_healing_cycle(args)
        print_result(result, verbose=args.verbose)
        
        # Exit with code based on findings
        if result['total_weaknesses'] == 0:
            sys.exit(0)
        elif result['fixes_applied'] == result['total_weaknesses']:
            sys.exit(0)
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()
