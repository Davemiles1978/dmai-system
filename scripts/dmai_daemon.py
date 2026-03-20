#!/usr/bin/env python3
"""DMAI Unified Daemon - Manages all services with auto-restart"""
import subprocess
import time
import os
import sys
import signal
from pathlib import Path

# Get absolute paths
DMAI_ROOT = Path(__file__).parent.parent
HARVESTER_ROOT = Path("/Users/davidmiles/Desktop/api-harvester")
VENV_PYTHON = DMAI_ROOT / "venv" / "bin" / "python3"
HARVESTER_VENV = HARVESTER_ROOT / "venv" / "bin" / "python3"
LOG_DIR = DMAI_ROOT / "logs"

# Create log directory
LOG_DIR.mkdir(exist_ok=True)

SERVICES = [
    # HARVESTER SERVICES (Critical for API keys)
    {
        "name": "harvester_daemon",
        "path": HARVESTER_ROOT / "harvester.py",
        "args": ["--daemon"],
        "pid": None,
        "log": LOG_DIR / "harvester.log",
        "venv": HARVESTER_VENV,
        "cwd": HARVESTER_ROOT
    },
    {
        "name": "harvester_api",
        "path": HARVESTER_ROOT / "api_server.py",
        "args": [],
        "pid": None,
        "log": LOG_DIR / "harvester_api.log",
        "venv": HARVESTER_VENV,
        "cwd": HARVESTER_ROOT
    },
    
    # DMAI CORE SERVICES
    {
        "name": "evolution_engine",
        "path": DMAI_ROOT / "evolution" / "evolution_engine.py",
        "args": ["--continuous"],
        "pid": None,
        "log": LOG_DIR / "evolution.log",
        "venv": VENV_PYTHON,
        "cwd": DMAI_ROOT
    },
    {
        "name": "book_reader",
        "path": DMAI_ROOT / "services" / "book_reader.py",
        "args": ["--continuous"],
        "pid": None,
        "log": LOG_DIR / "book_reader.log",
        "venv": VENV_PYTHON,
        "cwd": DMAI_ROOT
    },
    {
        "name": "web_researcher",
        "path": DMAI_ROOT / "services" / "web_researcher.py",
        "args": ["--continuous"],
        "pid": None,
        "log": LOG_DIR / "web_researcher.log",
        "venv": VENV_PYTHON,
        "cwd": DMAI_ROOT
    },
    {
        "name": "dark_researcher",
        "path": DMAI_ROOT / "services" / "dark_researcher.py",
        "args": ["--continuous"],
        "pid": None,
        "log": LOG_DIR / "dark_researcher.log",
        "venv": VENV_PYTHON,
        "cwd": DMAI_ROOT
    },
    {
        "name": "music_learner",
        "path": DMAI_ROOT / "services" / "music_learner.py",
        "args": [],
        "pid": None,
        "log": LOG_DIR / "music_learner.log",
        "venv": VENV_PYTHON,
        "cwd": DMAI_ROOT
    },
    {
        "name": "voice_service",
        "path": DMAI_ROOT / "voice" / "dmai_voice_with_learning.py",
        "args": ["--continuous"],
        "pid": None,
        "log": LOG_DIR / "voice.log",
        "venv": VENV_PYTHON,
        "cwd": DMAI_ROOT
    }
]

def start_service(service):
    """Start a service and return PID"""
    cmd = [str(service["venv"]), str(service["path"])] + service["args"]
    
    # Open log file
    log_file = open(service["log"], "a")
    
    # Start process in its correct working directory
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
        cwd=str(service["cwd"])
    )
    
    service["pid"] = process.pid
    service["log_file"] = log_file
    print(f"✅ Started {service['name']} (PID: {process.pid})")
    return process.pid

def check_service(service):
    """Check if service is running"""
    if not service.get("pid"):
        return False
    try:
        os.kill(service["pid"], 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def stop_all_services():
    """Stop all services gracefully"""
    print("\n🛑 Shutting down all services...")
    for service in SERVICES:
        if service.get("pid"):
            try:
                os.kill(service["pid"], signal.SIGTERM)
                print(f"  Stopped {service['name']}")
            except:
                pass
            time.sleep(1)

def main():
    print("🚀 DMAI Unified Daemon Starting...")
    print(f"📂 DMAI Root: {DMAI_ROOT}")
    print(f"📂 Harvester Root: {HARVESTER_ROOT}")
    print(f"🐍 DMAI Python: {VENV_PYTHON}")
    print(f"🐍 Harvester Python: {HARVESTER_VENV}")
    
    # Verify harvester paths exist
    if not HARVESTER_ROOT.exists():
        print(f"⚠️ Harvester directory not found at {HARVESTER_ROOT}")
    
    # Start all services
    for service in SERVICES:
        if service["path"].exists():
            start_service(service)
            time.sleep(2)  # Stagger starts
        else:
            print(f"⚠️ Service not found: {service['path']}")
    
    # Print initial status
    print("\n📊 Initial Status:")
    for service in SERVICES:
        if service.get("pid"):
            status = check_service(service)
            print(f"  {service['name']}: {'✅ RUNNING' if status else '❌ STOPPED'}")
    
    # Monitor and restart as needed
    print("\n🔄 Monitoring services (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(30)
            for service in SERVICES:
                if service.get("pid") and not check_service(service):
                    print(f"⚠️ {service['name']} died, restarting...")
                    start_service(service)
    except KeyboardInterrupt:
        stop_all_services()

if __name__ == "__main__":
    main()
