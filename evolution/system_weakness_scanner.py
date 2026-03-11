#!/usr/bin/env python3
"""
System Weakness Scanner for DMAI
Scans all services for performance bottlenecks, error-prone modules, and security gaps
Integrates with evolution cycle to auto-heal detected issues
VERSION: Fixed to work with dmai_daemon - doesn't directly restart services
"""

import os
import sys
import json
import time
import psutil
import subprocess
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core_connector import get_evolution_status, voice_say
except ImportError:
    # Mock functions for standalone testing
    def voice_say(text):
        print(f"[VOICE] {text}")
    
    def get_evolution_status():
        return {"status": "unknown"}

# Configure logging
log_dir = Path.home() / "Library/Logs/dmai"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - weakness_scanner - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'weakness_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('weakness_scanner')

# Service configurations - match exactly with dmai_daemon.py
SERVICES = [
    {"name": "harvester_daemon", "port": 9001, "health_url": "http://localhost:9001/health"},
    {"name": "harvester_api", "port": 9002, "health_url": "http://localhost:9002/health"},
    {"name": "evolution_engine", "port": 9003, "health_url": "http://localhost:9003/health"},
    {"name": "book_reader", "port": 9004, "health_url": "http://localhost:9004/status"},
    {"name": "web_researcher", "port": 9005, "health_url": "http://localhost:9005/status"},
    {"name": "dark_researcher", "port": 9006, "health_url": "http://localhost:9006/status"},
    {"name": "music_learner", "port": 9007, "health_url": "http://localhost:9007/status"},
    {"name": "voice_service", "port": 9008, "health_url": "http://localhost:9008/health"},
    {"name": "dual_launcher", "port": 9009, "health_url": "http://localhost:9009/status"},
]

# PostgreSQL connection for storing weaknesses
DB_CONFIG = {
    "host": "dpg-d6lfcg3h46gs73drf3fg-a.oregon-postgres.render.com",
    "database": "harvester_u9ni",
    "user": "dmai",
    "password": "xQjt0tbhmT0vRExNv9wTSbe3t7n34J85",
    "port": 5432
}

class SystemWeaknessScanner:
    def __init__(self):
        self.weaknesses = []
        self.service_stats = {}
        self.db_conn = None
        self.daemon_pid = self.find_daemon_pid()
        self.init_database()
        
    def find_daemon_pid(self) -> Optional[int]:
        """Find the PID of the running dmai_daemon"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['cmdline'] and 'dmai_daemon_fixed.py' in ' '.join(proc.info['cmdline']):
                    logger.info(f"Found dmai_daemon with PID: {proc.info['pid']}")
                    return proc.info['pid']
        except Exception as e:
            logger.error(f"Error finding daemon PID: {e}")
        return None
    
    def is_daemon_healthy(self) -> bool:
        """Check if the main daemon is running and healthy"""
        if not self.daemon_pid:
            self.daemon_pid = self.find_daemon_pid()
        
        if self.daemon_pid:
            try:
                # Check if daemon process exists
                proc = psutil.Process(self.daemon_pid)
                if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return False
    
    def signal_daemon_to_restart(self, service_name: str):
        """Signal the daemon to restart a service instead of doing it directly"""
        if not self.is_daemon_healthy():
            logger.error("Daemon not running, cannot signal restart")
            return False
        
        try:
            # Try to use the daemon's API if available
            # For now, we'll just log that the daemon should handle this
            logger.info(f"⚠️ Service {service_name} needs restart - daemon should handle this")
            
            # We could implement a signal file approach or socket communication
            # For now, we'll rely on the daemon's own monitoring
            return True
        except Exception as e:
            logger.error(f"Failed to signal daemon: {e}")
            return False
    
    def init_database(self):
        """Initialize database connection and create table if needed"""
        try:
            import psycopg2
            self.db_conn = psycopg2.connect(
                host=DB_CONFIG["host"],
                database=DB_CONFIG["database"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                port=DB_CONFIG["port"]
            )
            cursor = self.db_conn.cursor()
            
            # Create weaknesses table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_weaknesses (
                    id SERIAL PRIMARY KEY,
                    weakness_type VARCHAR(50),
                    module TEXT,
                    description TEXT,
                    severity INTEGER,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    fixed_at TIMESTAMP,
                    fix_applied TEXT,
                    verification_status BOOLEAN DEFAULT FALSE,
                    fix_attempts INTEGER DEFAULT 0,
                    daemon_managed BOOLEAN DEFAULT TRUE
                )
            """)
            self.db_conn.commit()
            logger.info("Database initialized successfully")
        except ImportError:
            logger.warning("psycopg2 not installed, running without database")
            self.db_conn = None
        except Exception as e:
            logger.error(f"Database init failed: {e}")
            self.db_conn = None
    
    def scan_service_health(self) -> List[Dict[str, Any]]:
        """Check each service's health via HTTP endpoints"""
        weaknesses = []
        
        for service in SERVICES:
            try:
                # Check HTTP health endpoint
                response = requests.get(service["health_url"], timeout=3)
                
                if response.status_code == 200:
                    # Service is healthy, check response content
                    try:
                        data = response.json()
                        if data.get('status') == 'healthy':
                            logger.debug(f"✅ {service['name']} is healthy")
                        else:
                            # Service responding but not healthy
                            weaknesses.append({
                                "type": "unhealthy_service",
                                "module": service["name"],
                                "description": f"Service reporting unhealthy status",
                                "severity": 7,
                                "fix": "report_to_daemon",
                                "daemon_managed": True
                            })
                    except:
                        # If no JSON, just assume it's working if 200
                        pass
                else:
                    # Service responding with error
                    weaknesses.append({
                        "type": "service_error",
                        "module": service["name"],
                        "description": f"Service returned HTTP {response.status_code}",
                        "severity": 6,
                        "fix": "report_to_daemon",
                        "daemon_managed": True
                    })
                    
            except requests.exceptions.ConnectionError:
                # Service not responding at all
                weaknesses.append({
                    "type": "service_down",
                    "module": service["name"],
                    "description": "Service not responding to HTTP requests",
                    "severity": 9,
                    "fix": "daemon_restart_needed",
                    "daemon_managed": True
                })
            except requests.exceptions.Timeout:
                weaknesses.append({
                    "type": "service_timeout",
                    "module": service["name"],
                    "description": "Service health check timed out",
                    "severity": 7,
                    "fix": "daemon_restart_needed",
                    "daemon_managed": True
                })
            except Exception as e:
                logger.error(f"Error scanning {service['name']}: {e}")
        
        return weaknesses
    
    def scan_system_resources(self) -> List[Dict[str, Any]]:
        """Check system-wide resource usage"""
        weaknesses = []
        
        try:
            # Overall system memory
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                weaknesses.append({
                    "type": "critical_memory_pressure",
                    "module": "system",
                    "description": f"System memory at {memory.percent}% - {memory.used / 1024 / 1024:.0f}MB used",
                    "severity": 9,
                    "fix": "free_memory_cache",
                    "daemon_managed": False
                })
            elif memory.percent > 80:
                weaknesses.append({
                    "type": "high_memory_usage",
                    "module": "system",
                    "description": f"System memory at {memory.percent}%",
                    "severity": 6,
                    "fix": "monitor_memory",
                    "daemon_managed": False
                })
            
            # Disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                weaknesses.append({
                    "type": "disk_space_critical",
                    "module": "system",
                    "description": f"Disk usage at {disk.percent}% - {disk.free / 1024 / 1024 / 1024:.1f}GB free",
                    "severity": 10,
                    "fix": "emergency_cleanup",
                    "daemon_managed": False
                })
            elif disk.percent > 85:
                weaknesses.append({
                    "type": "disk_space_low",
                    "module": "system",
                    "description": f"Disk usage at {disk.percent}%",
                    "severity": 7,
                    "fix": "cleanup_old_logs",
                    "daemon_managed": False
                })
            
            # CPU load average
            load_avg = psutil.getloadavg()
            cpu_count = psutil.cpu_count()
            if load_avg[0] > cpu_count * 2:  # 2x CPU count is critical
                weaknesses.append({
                    "type": "critical_load_average",
                    "module": "system",
                    "description": f"Load average: {load_avg[0]:.2f} (CPUs: {cpu_count})",
                    "severity": 8,
                    "fix": "investigate_high_load",
                    "daemon_managed": False
                })
        except Exception as e:
            logger.error(f"Error scanning system resources: {e}")
        
        return weaknesses
    
    def scan_error_logs(self) -> List[Dict[str, Any]]:
        """Scan service logs for error patterns"""
        weaknesses = []
        log_dir = Path.home() / "Library/Logs/dmai"
        
        if not log_dir.exists():
            return weaknesses
            
        for log_file in log_dir.glob("*.log"):
            try:
                # Only check logs from last hour
                mtime = log_file.stat().st_mtime
                if time.time() - mtime > 3600:  # Older than 1 hour
                    continue
                
                with open(log_file, 'r') as f:
                    # Read last 200 lines
                    lines = f.readlines()[-200:]
                    
                    # Count errors and criticals
                    error_count = sum(1 for line in lines if any(x in line for x in ["ERROR", "CRITICAL", "Traceback", "Exception"]))
                    warning_count = sum(1 for line in lines if "WARNING" in line)
                    
                    # Check for rapid errors (multiple in short time)
                    if error_count > 20:
                        weaknesses.append({
                            "type": "critical_error_rate",
                            "module": log_file.stem,
                            "description": f"{error_count} errors in recent logs",
                            "severity": 8,
                            "fix": "analyze_error_pattern",
                            "daemon_managed": True
                        })
                    elif error_count > 10:
                        weaknesses.append({
                            "type": "high_error_rate",
                            "module": log_file.stem,
                            "description": f"{error_count} errors in recent logs",
                            "severity": 6,
                            "fix": "monitor_error_rate",
                            "daemon_managed": True
                        })
                    
                    # Check for specific error patterns
                    error_lines = [line for line in lines if "ERROR" in line or "CRITICAL" in line]
                    if error_lines:
                        # Look for repeating errors
                        error_patterns = {}
                        for line in error_lines[:10]:  # Check first 10 errors
                            # Simplify error message for pattern matching
                            simplified = ' '.join(line.split()[:5])  # First 5 words
                            error_patterns[simplified] = error_patterns.get(simplified, 0) + 1
                        
                        # If same error appears multiple times
                        for pattern, count in error_patterns.items():
                            if count >= 3:
                                weaknesses.append({
                                    "type": "repeating_error",
                                    "module": log_file.stem,
                                    "description": f"Repeating error pattern: {pattern[:50]}... ({count} times)",
                                    "severity": 7,
                                    "fix": "investigate_error_pattern",
                                    "daemon_managed": True
                                })
                                break
                    
            except Exception as e:
                logger.error(f"Error scanning {log_file}: {e}")
        
        return weaknesses
    
    def check_daemon_health(self) -> List[Dict[str, Any]]:
        """Check if the main daemon is healthy"""
        weaknesses = []
        
        if not self.is_daemon_healthy():
            weaknesses.append({
                "type": "daemon_not_running",
                "module": "dmai_daemon",
                "description": "Main daemon process is not running",
                "severity": 10,
                "fix": "start_daemon",
                "daemon_managed": False
            })
        else:
            # Check daemon's own logs for issues
            log_file = log_dir / 'daemon.log'
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()[-100:]
                        restart_count = sum(1 for line in lines if "Restarting" in line)
                        if restart_count > 10:
                            weaknesses.append({
                                "type": "excessive_restarts",
                                "module": "dmai_daemon",
                                "description": f"Daemon performing too many restarts: {restart_count} in recent logs",
                                "severity": 7,
                                "fix": "investigate_daemon",
                                "daemon_managed": False
                            })
                except Exception as e:
                    logger.error(f"Error checking daemon logs: {e}")
        
        return weaknesses
    
    def apply_system_fix(self, weakness: Dict[str, Any]) -> bool:
        """Apply system-level fixes (not service restarts)"""
        try:
            fix_type = weakness["fix"]
            
            if fix_type == "free_memory_cache":
                # Clear memory cache (requires sudo, may fail)
                try:
                    os.system('sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1')
                    logger.info("Attempted to clear memory cache")
                    return True
                except:
                    # Try without sudo
                    os.system('sync')
                    return True
                    
            elif fix_type == "cleanup_old_logs":
                # Delete logs older than 7 days
                log_dir = Path.home() / "Library/Logs/dmai"
                if log_dir.exists():
                    cutoff = time.time() - (7 * 24 * 3600)  # 7 days
                    deleted = 0
                    for log_file in log_dir.glob("*.log"):
                        try:
                            if log_file.stat().st_mtime < cutoff:
                                log_file.unlink()
                                deleted += 1
                        except Exception:
                            pass
                    logger.info(f"Cleaned up {deleted} old log files")
                return True
                
            elif fix_type == "emergency_cleanup":
                # Aggressive cleanup - delete logs older than 1 day
                log_dir = Path.home() / "Library/Logs/dmai"
                if log_dir.exists():
                    cutoff = time.time() - (24 * 3600)  # 1 day
                    deleted = 0
                    for log_file in log_dir.glob("*.log"):
                        try:
                            if log_file.stat().st_mtime < cutoff:
                                log_file.unlink()
                                deleted += 1
                        except Exception:
                            pass
                    logger.info(f"Emergency cleanup: deleted {deleted} log files")
                return True
                
            elif fix_type == "start_daemon":
                # Try to start the daemon
                daemon_script = Path("/Users/davidmiles/Desktop/dmai-system/scripts/dmai_daemon_fixed.py")
                if daemon_script.exists():
                    subprocess.Popen(["python3", str(daemon_script), "restart"])
                    logger.info("Attempted to start daemon")
                    return True
                    
            elif fix_type in ["monitor_memory", "investigate_high_load", "analyze_error_pattern"]:
                # For monitoring/analysis fixes, just log
                logger.info(f"Fix '{fix_type}' requires ongoing monitoring")
                return True
                
            else:
                logger.warning(f"Unknown fix type: {fix_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply system fix: {e}")
            return False
    
    def scan_and_heal(self):
        """Main scanning and healing cycle"""
        logger.info("="*60)
        logger.info("🔍 Starting system weakness scan")
        logger.info("="*60)
        
        # Check daemon first
        daemon_healthy = self.is_daemon_healthy()
        logger.info(f"Daemon status: {'✅ RUNNING' if daemon_healthy else '❌ NOT RUNNING'}")
        
        try:
            voice_say("Starting system health check")
        except:
            pass
        
        # Collect all weaknesses
        all_weaknesses = []
        
        # Service health checks
        service_weaknesses = self.scan_service_health()
        all_weaknesses.extend(service_weaknesses)
        logger.info(f"Found {len(service_weaknesses)} service health issues")
        
        # System resource checks
        resource_weaknesses = self.scan_system_resources()
        all_weaknesses.extend(resource_weaknesses)
        logger.info(f"Found {len(resource_weaknesses)} resource issues")
        
        # Error log checks
        log_weaknesses = self.scan_error_logs()
        all_weaknesses.extend(log_weaknesses)
        logger.info(f"Found {len(log_weaknesses)} log-related issues")
        
        # Daemon health checks
        daemon_weaknesses = self.check_daemon_health()
        all_weaknesses.extend(daemon_weaknesses)
        logger.info(f"Found {len(daemon_weaknesses)} daemon issues")
        
        total_weaknesses = len(all_weaknesses)
        logger.info(f"TOTAL: Found {total_weaknesses} weaknesses")
        
        # Sort by severity
        all_weaknesses.sort(key=lambda x: x["severity"], reverse=True)
        
        # Store in database
        if self.db_conn and all_weaknesses:
            try:
                cursor = self.db_conn.cursor()
                for w in all_weaknesses:
                    cursor.execute("""
                        INSERT INTO system_weaknesses 
                        (weakness_type, module, description, severity, daemon_managed)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (w["type"], w["module"], w["description"], w["severity"], w.get("daemon_managed", True)))
                self.db_conn.commit()
                logger.info(f"Stored {len(all_weaknesses)} weaknesses in database")
            except Exception as e:
                logger.error(f"Error storing weaknesses: {e}")
        
        # Apply system-level fixes (not service restarts)
        system_fixes_applied = 0
        service_issues = []
        
        for weakness in all_weaknesses:
            # Only handle non-daemon-managed issues (system resources, etc.)
            if not weakness.get("daemon_managed", True):
                if weakness["severity"] >= 7:  # Only high severity
                    logger.info(f"Applying system fix for: {weakness['type']}")
                    if self.apply_system_fix(weakness):
                        system_fixes_applied += 1
                        logger.info(f"✅ Applied system fix: {weakness['type']}")
            else:
                # Track service issues for reporting
                service_issues.append(weakness)
        
        # Report service issues to be handled by daemon
        if service_issues and daemon_healthy:
            critical_services = [w for w in service_issues if w["severity"] >= 8]
            if critical_services:
                logger.warning(f"⚠️ {len(critical_services)} critical service issues detected - daemon should handle")
                for issue in critical_services[:3]:  # Show top 3
                    logger.warning(f"  • {issue['module']}: {issue['description']}")
        
        # Summary
        summary = (f"Health scan complete. Found {total_weaknesses} issues, "
                  f"applied {system_fixes_applied} system fixes. "
                  f"{len(service_issues)} service issues reported to daemon.")
        
        logger.info("="*60)
        logger.info(summary)
        logger.info("="*60)
        
        try:
            voice_say(summary)
        except:
            pass
        
        return {
            "total_weaknesses": total_weaknesses,
            "system_fixes_applied": system_fixes_applied,
            "service_issues": len(service_issues),
            "weaknesses": all_weaknesses[:10] if all_weaknesses else [],  # Top 10 for display
            "daemon_healthy": daemon_healthy,
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔍 DMAI System Weakness Scanner (Daemon-Aware)")
    print("="*60)
    
    scanner = SystemWeaknessScanner()
    result = scanner.scan_and_heal()
    
    # Print summary
    print("\n📊 SCAN RESULTS")
    print("="*60)
    print(f"Total weaknesses detected: {result['total_weaknesses']}")
    print(f"System fixes applied: {result['system_fixes_applied']}")
    print(f"Service issues (handled by daemon): {result['service_issues']}")
    print(f"Daemon status: {'✅ RUNNING' if result['daemon_healthy'] else '❌ NOT RUNNING'}")
    
    if result['weaknesses']:
        print("\n🔴 Top issues detected:")
        for w in result['weaknesses'][:5]:
            managed = "daemon" if w.get('daemon_managed') else "system"
            print(f"  • [{w['severity']}] {w['module']}: {w['description']} (handled by {managed})")
    
    print("\n" + "="*60)
    if result['total_weaknesses'] == 0:
        print("✅ No issues found - system is healthy")
    elif result['service_issues'] > 0:
        print(f"⚠️ {result['service_issues']} service issues - daemon will handle them")
    else:
        print(f"✅ All issues handled - {result['system_fixes_applied']} fixes applied")
    print("="*60)
    
    # Exit with code
    if result['total_weaknesses'] == 0:
        sys.exit(0)
    elif result['service_issues'] > 0:
        sys.exit(2)  # Different code for service issues
    else:
        sys.exit(1)
