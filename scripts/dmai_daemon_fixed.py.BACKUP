#!/usr/bin/env python3
"""DMAI Unified Daemon - Fixed version with proper zombie detection"""
import subprocess
import time
import os
import sys
import signal
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - DAEMON - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get absolute paths
DMAI_ROOT = Path("/Users/davidmiles/Desktop/dmai-system")
HARVESTER_ROOT = Path("/Users/davidmiles/Desktop/dmai-system/api-harvester")  
HARVESTER_VENV = HARVESTER_ROOT / "venv" / "bin" / "python3"
MAIN_VENV = DMAI_ROOT / "venv" / "bin" / "python3"
LOG_DIR = DMAI_ROOT / "logs"

# Create log directory
LOG_DIR.mkdir(exist_ok=True)

SERVICES = [
    {
        "name": "harvester_daemon",
        "path": HARVESTER_ROOT / "harvester.py",
        "venv": HARVESTER_VENV,
        "log": LOG_DIR / "harvester_daemon.log",
        "args": ["--daemon"]
    },
    {
        "name": "harvester_api",
        "path": HARVESTER_ROOT / "api_server.py",
        "venv": HARVESTER_VENV,
        "log": LOG_DIR / "harvester_api.log",
        "args": ["--daemon"]
    },
    {
        "name": "book_reader",
        "path": DMAI_ROOT / "services" / "book_reader.py",
        "venv": MAIN_VENV,
        "log": LOG_DIR / "book_reader.log",
        "args": ["--continuous"]
    },
    {
        "name": "web_researcher",
        "path": DMAI_ROOT / "services" / "web_researcher.py",
        "venv": MAIN_VENV,
        "log": LOG_DIR / "web_researcher.log",
        "args": ["--continuous"]
    },
    {
        "name": "dark_researcher",
        "path": DMAI_ROOT / "services" / "dark_researcher.py",
        "venv": MAIN_VENV,
        "log": LOG_DIR / "dark_researcher.log",
        "args": ["--continuous"]
    },
    {
        "name": "music_learner",
        "path": DMAI_ROOT / "music" / "music_learner.py",
        "venv": MAIN_VENV,
        "log": LOG_DIR / "music_learner.log",
        "args": ["--continuous"]
    },
    {
        "name": "voice_service",
        "path": DMAI_ROOT / "voice" / "dmai_voice_with_learning.py",
        "venv": MAIN_VENV,
        "log": LOG_DIR / "voice.log",
        "args": []
    },
    {
        "name": "dual_launcher",
        "path": DMAI_ROOT / "evolution" / "dual_launcher.py",
        "venv": MAIN_VENV,
        "log": LOG_DIR / "dual_launcher.log",
        "args": []
    },
]

def is_process_alive(pid):
    """Check if process is alive and not a zombie"""
    try:
        # Check if process exists
        os.kill(pid, 0)
        
        # Check if it's a zombie
        result = subprocess.run(['ps', '-o', 'state=', '-p', str(pid)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            state = result.stdout.strip()
            if state == 'Z':
                return False
        return True
    except (OSError, ProcessLookupError):
        return False

def start_service(service):
    """Start a service and return PID"""
    cmd = [str(service["venv"]), str(service["path"])]
    if "args" in service and service["args"]:
        cmd += service["args"]
    
    try:
        # Open log file
        log_file = open(service["log"], "a")
        log_file.write(f"\n--- Starting at {datetime.now()} ---\n")
        log_file.flush()
        
        # Set resource limits before starting (Unix only)
        import resource
        def set_limits():
            # Set memory limit to 512MB per service
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        
        # Start process with preexec function to set limits
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
            preexec_fn=set_limits if hasattr(os, 'fork') else None
        )
        
        service["pid"] = process.pid
        service["log_file"] = log_file
        service["last_restart"] = time.time()
        
        logger.info(f"✅ Started {service['name']} (PID: {process.pid})")
        return process.pid
    except Exception as e:
        logger.error(f"❌ Failed to start {service['name']}: {e}")
        return None

def check_service(service):
    """Check if service is running and not a zombie"""
    if not service.get("pid"):
        return False
    return is_process_alive(service["pid"])

def stop_service(service):
    """Stop a service gracefully"""
    if service.get("pid"):
        try:
            os.kill(service["pid"], signal.SIGTERM)
            logger.info(f"Stopped {service['name']} (PID: {service['pid']})")
            time.sleep(2)
        except:
            pass
        service["pid"] = None

def stop_all_services():
    """Stop all services gracefully"""
    logger.info("\n🛑 Shutting down all services...")
    for service in SERVICES:
        stop_service(service)

def main():
    logger.info("="*60)
    logger.info("🚀 DMAI Unified Daemon Starting...")
    logger.info("="*60)
    logger.info(f"📂 DMAI Root: {DMAI_ROOT}")
    logger.info(f"📂 Harvester Root: {HARVESTER_ROOT}")
    
    # Kill any existing service processes
    logger.info("Cleaning up old processes...")
    os.system("pkill -f 'dmai_voice_with_learning.py' 2>/dev/null")
    os.system("pkill -f 'book_reader.py' 2>/dev/null")
    os.system("pkill -f 'web_researcher.py' 2>/dev/null")
    os.system("pkill -f 'dark_researcher.py' 2>/dev/null")
    os.system("pkill -f 'music_learner.py' 2>/dev/null")
    os.system("pkill -f 'harvester.py' 2>/dev/null")
    os.system("pkill -f 'api_server.py' 2>/dev/null")
    os.system("pkill -f 'dual_launcher.py' 2>/dev/null")
    
    time.sleep(2)
    
    # Start all services
    for service in SERVICES:
        if service["path"].exists():
            start_service(service)
            time.sleep(2)
        else:
            logger.warning(f"⚠️ Service not found: {service['path']}")
    
    # Print initial status
    logger.info("\n📊 Initial Status:")
    for service in SERVICES:
        if service.get("pid"):
            status = check_service(service)
            logger.info(f"  {service['name']}: {'✅ RUNNING' if status else '❌ STOPPED'}")
    
    # Monitor and restart as needed
    logger.info("\n🔄 Monitoring services (checking every 15 seconds)...")
    
    try:
        while True:
            time.sleep(15)
            
            for service in SERVICES:
                if not check_service(service):
                    logger.warning(f"⚠️ {service['name']} is NOT RUNNING (PID: {service.get('pid')})")
                    
                    # Don't restart too frequently (minimum 30 seconds between restarts)
                    if time.time() - service.get("last_restart", 0) > 30:
                        logger.info(f"🔄 Restarting {service['name']}...")
                        # Clean up zombie if it exists
                        if service.get("pid"):
                            try:
                                os.kill(service["pid"], signal.SIGKILL)
                            except:
                                pass
                        start_service(service)
                    else:
                        logger.info(f"⏳ Waiting before restarting {service['name']}")
            
            # Log status every 2 minutes
            if int(time.time()) % 120 < 15:
                running = [s['name'] for s in SERVICES if check_service(s)]
                logger.info(f"📊 Running: {len(running)}/{len(SERVICES)} services: {', '.join(running)}")
                
    except KeyboardInterrupt:
        stop_all_services()
        logger.info("\n👋 Daemon stopped")

if __name__ == "__main__":
    main()
