#!/usr/bin/env python3
"""
DMAI System Health Check
Verifies all components are working at 100%
"""

import os
import json
import requests
import subprocess
from pathlib import Path
from datetime import datetime

class SystemHealthCheck:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "status": "PENDING",
            "checks": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }
        self.colors = {
            "PASS": "✅",
            "FAIL": "❌",
            "WARN": "⚠️"
        }
    
    def add_check(self, name, status, message, details=None):
        """Add a check result"""
        self.results["checks"].append({
            "name": name,
            "status": status,
            "message": message,
            "details": details or {}
        })
        self.results["summary"]["total"] += 1
        if status == "PASS":
            self.results["summary"]["passed"] += 1
        elif status == "FAIL":
            self.results["summary"]["failed"] += 1
        else:
            self.results["summary"]["warnings"] += 1
    
    def check_file_exists(self, path, name):
        """Check if a file exists"""
        exists = Path(path).exists()
        status = "PASS" if exists else "FAIL"
        message = f"{path} {'found' if exists else 'not found'}"
        self.add_check(f"File: {name}", status, message, {"path": path})
        return exists
    
    def check_directory_exists(self, path, name):
        """Check if a directory exists"""
        exists = Path(path).exists() and Path(path).is_dir()
        status = "PASS" if exists else "FAIL"
        message = f"{name} directory {'found' if exists else 'not found'}"
        self.add_check(f"Directory: {name}", status, message, {"path": path})
        return exists
    
    def check_process_running(self, process_name, port=None):
        """Check if a process is running"""
        try:
            # Check by process name
            result = subprocess.run(
                f"pgrep -f '{process_name}'",
                shell=True,
                capture_output=True,
                text=True
            )
            running = result.returncode == 0
            
            # If port specified, check if listening
            port_open = False
            if port and running:
                port_result = subprocess.run(
                    f"lsof -i :{port} | grep LISTEN",
                    shell=True,
                    capture_output=True,
                    text=True
                )
                port_open = port_result.returncode == 0
            
            status = "PASS" if running else "FAIL"
            message = f"{process_name} is {'running' if running else 'not running'}"
            if port and running:
                message += f" on port {port} {'(listening)' if port_open else '(not listening)'}"
            
            self.add_check(f"Process: {process_name}", status, message, {
                "running": running,
                "port": port,
                "port_open": port_open if port else None
            })
            return running
        except Exception as e:
            self.add_check(f"Process: {process_name}", "FAIL", f"Error checking: {e}")
            return False
    
    def check_api_endpoint(self, url, name):
        """Check if API endpoint is responding"""
        try:
            response = requests.get(url, timeout=5)
            status = "PASS" if response.status_code == 200 else "FAIL"
            message = f"{name} API returned {response.status_code}"
            
            data = None
            if response.status_code == 200:
                try:
                    data = response.json()
                except:
                    data = {"note": "non-JSON response"}
            
            self.add_check(f"API: {name}", status, message, {
                "url": url,
                "status_code": response.status_code,
                "response": data
            })
            return response.status_code == 200
        except Exception as e:
            self.add_check(f"API: {name}", "FAIL", f"Connection failed: {e}", {"url": url})
            return False
    
    def check_git_status(self):
        """Check git repository status"""
        try:
            # Check if git repository
            result = subprocess.run(
                "git status --porcelain",
                shell=True,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            modified_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            modified_count = len([f for f in modified_files if f])
            
            if modified_count == 0:
                status = "PASS"
                message = "Git working directory clean"
            else:
                status = "WARN"
                message = f"{modified_count} uncommitted changes"
            
            self.add_check("Git Status", status, message, {
                "modified_files": modified_count,
                "details": modified_files[:5]  # First 5 files
            })
        except Exception as e:
            self.add_check("Git Status", "FAIL", f"Error checking git: {e}")
    
    def check_disk_usage(self):
        """Check disk usage of critical directories"""
        critical_paths = [
            ("checkpoints", Path("checkpoints")),
            ("repos", Path("repos")),
            ("logs", Path("logs"))
        ]
        
        for name, path in critical_paths:
            if path.exists():
                try:
                    # Get size in MB
                    size_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    size_mb = size_bytes / (1024 * 1024)
                    
                    status = "PASS" if size_mb < 5000 else "WARN"  # Warn if >5GB
                    message = f"{name} size: {size_mb:.2f} MB"
                    
                    self.add_check(f"Disk: {name}", status, message, {
                        "size_mb": round(size_mb, 2)
                    })
                except Exception as e:
                    self.add_check(f"Disk: {name}", "WARN", f"Could not calculate size: {e}")
    
    def check_cron_job(self):
        """Check if cron job is installed"""
        try:
            result = subprocess.run(
                "crontab -l",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and "cleanup_versions" in result.stdout:
                status = "PASS"
                message = "Cleanup cron job installed"
            else:
                status = "FAIL"
                message = "Cleanup cron job not found"
            
            self.add_check("Cron Job", status, message, {
                "crontab": result.stdout.strip()
            })
        except Exception as e:
            self.add_check("Cron Job", "FAIL", f"Error checking cron: {e}")
    
    def check_evolution_progress(self):
        """Check evolution engine progress"""
        gen_file = Path("checkpoints/current_generation.txt")
        if gen_file.exists():
            try:
                with open(gen_file, 'r') as f:
                    gen = int(f.read().strip())
                
                status = "PASS" if gen > 0 else "WARN"
                message = f"Evolution at generation {gen}"
                
                self.add_check("Evolution Progress", status, message, {
                    "generation": gen
                })
            except Exception as e:
                self.add_check("Evolution Progress", "FAIL", f"Error reading generation: {e}")
        else:
            self.add_check("Evolution Progress", "WARN", "No generation file found")
    
    def run_all_checks(self):
        """Run all system checks"""
        print("\n" + "="*60)
        print("🚀 DMAI SYSTEM HEALTH CHECK")
        print("="*60)
        
        # Critical Files
        print("\n📁 Checking Critical Files...")
        self.check_file_exists("ai_ui.html", "Main UI")
        self.check_file_exists("evolution_engine.py", "Evolution Engine")
        self.check_file_exists("api_server.py", "API Server")
        self.check_file_exists("app_with_api.py", "Flask App")
        self.check_file_exists("requirements.txt", "Requirements")
        self.check_file_exists("cleanup_versions.py", "Cleanup Script")
        
        # Directories
        print("\n📂 Checking Directories...")
        self.check_directory_exists("repos", "Repositories")
        self.check_directory_exists("checkpoints", "Checkpoints")
        self.check_directory_exists("logs", "Logs")
        self.check_directory_exists("ui", "UI Files")
        
        # Processes (if running locally)
        print("\n⚙️ Checking Processes...")
        self.check_process_running("evolution_engine.py", None)
        self.check_process_running("api_server.py", 8889)
        
        # API Endpoints
        print("\n🌐 Checking API Endpoints...")
        self.check_api_endpoint(
            "https://dmai-final.onrender.com/api/evolution-stats",
            "Render Evolution Stats"
        )
        self.check_api_endpoint(
            "https://dmai-final.onrender.com/",
            "Render Main UI"
        )
        
        # Local API if running
        self.check_api_endpoint(
            "http://localhost:8889/api/evolution-stats",
            "Local Evolution Stats"
        )
        
        # System Status
        print("\n📊 Checking System Status...")
        self.check_git_status()
        self.check_disk_usage()
        self.check_cron_job()
        self.check_evolution_progress()
        
        # Summary
        print("\n" + "="*60)
        print("📋 HEALTH CHECK SUMMARY")
        print("="*60)
        
        total = self.results["summary"]["total"]
        passed = self.results["summary"]["passed"]
        failed = self.results["summary"]["failed"]
        warnings = self.results["summary"]["warnings"]
        
        # Calculate percentage
        score = (passed / total * 100) if total > 0 else 0
        
        print(f"\nTotal Checks: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️ Warnings: {warnings}")
        print(f"\n📈 Health Score: {score:.1f}%")
        
        # Status bar
        bar_length = 40
        filled = int(bar_length * score / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\n{bar} {score:.1f}%")
        
        # Overall status
        if failed == 0:
            if warnings == 0:
                self.results["status"] = "EXCELLENT"
                print("\n✅ OVERALL STATUS: EXCELLENT - All systems operational")
            else:
                self.results["status"] = "GOOD"
                print("\n⚠️ OVERALL STATUS: GOOD - Some warnings, but core systems working")
        else:
            self.results["status"] = "ISSUES DETECTED"
            print(f"\n❌ OVERALL STATUS: ISSUES DETECTED - {failed} checks failed")
        
        # Save results
        report_file = f"health_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📝 Full report saved to: {report_file}")
        
        return self.results

if __name__ == "__main__":
    check = SystemHealthCheck()
    results = check.run_all_checks()
