#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_enrollment_readiness():
    issues = []
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default = sd.default.device
        if default is None:
            issues.append("❌ No default microphone found")
        else:
            print("✅ Microphone found")
    except Exception as e:
        issues.append(f"❌ Microphone error: {e}")
    
    import tempfile
    try:
        test_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=True)
        print("✅ File permissions OK")
    except:
        issues.append("❌ Cannot write temp files")
    
    import shutil
    usage = shutil.disk_usage("/")
    free_gb = usage.free / (1024**3)
    if free_gb < 1:
        issues.append(f"❌ Low disk space: {free_gb:.1f}GB free")
    else:
        print(f"✅ Disk space: {free_gb:.1f}GB free")
    
    print("\n🎤 Enrollment Tips:")
    print("  • Find a quiet room")
    print("  • Close windows/doors")
    print("  • Speak in your normal voice")
    print("  • Take your time - no rush")
    print("  • 5 phrases, about 2 minutes total")
    
    if issues:
        print("\n⚠️  Issues to fix before enrollment:")
        for issue in issues:
            print(f"  {issue}")
        return False
    
    print("\n✅ Ready for enrollment at 20:00!")
    return True

if __name__ == "__main__":
    print("="*50)
    print("🎤 DMAI VOICE ENROLLMENT READINESS CHECK")
    print("="*50)
    check_enrollment_readiness()
