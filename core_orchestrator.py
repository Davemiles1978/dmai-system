#!/usr/bin/env python3
"""
DMAI Core Services Orchestrator
Manages the 8 strategic services that must run 24/7
"""
import os
import sys
import time
import json
import signal
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/core_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('core_orchestrator')

# Core services that must run 24/7
CORE_SERVICES = {
    'evolution_suite': {
        'components': [
            'agi_orchestrator.py',
            'evolution/evolution_engine.py',
            'knowledge_graph.py'
        ],
        'cmd': ['python3', 'evolution/evolution_engine.py', '--continuous'],
        'description': 'Evolution Engine Suite',
        'critical': True
    },
    'dual_recovery': {
        'components': [
            'phase1/P1T4_Implement_engine.py',
            'phase1/P1T5_Implement_validator.py'
        ],
        'cmd': ['python3', 'phase1/P1T4_Implement_engine.py', '--daemon'],
        'description': 'Dual Recovery Engine',
        'critical': True
    },
    'fund_generator': {
        'components': [
            'phase5/P5T2_Implement_monero_miner.py',
            'phase5/P5T35_Implement_micro_tasks.py'
        ],
        'cmd': ['python3', 'phase5/P5T2_Implement_monero_miner.py', '--daemon'],
        'description': 'Fund Generation',
        'critical': True
    },
    'self_healer': {
        'components': [
            'evolution/system_weakness_scanner.py'
        ],
        'cmd': ['python3', 'evolution/system_weakness_scanner.py', '--continuous'],
        'description': 'Self-Healing System',
        'critical': True
    },
    'continuous_learner': {
        'components': [
            'api-harvester/harvester.py',
            'services/web_researcher.py',
            'services/dark_researcher.py'
        ],
        'cmd': ['python3', 'api-harvester/harvester.py', '--daemon'],
        'description': 'Continuous Learner',
        'critical': True
    },
    'master_control': {
        'components': [
            'dmai_web_ui.py'
        ],
        'cmd': ['python3', 'dmai_web_ui.py', '--port', '5001'],
        'description': 'Master Control UI',
        'critical': True
    },
    'cloud_coordinator': {
        'components': [
            'render_start.py'
        ],
        'cmd': ['python3', 'render_start.py'],
        'description': 'Cloud Coordinator',
        'critical': False
    },
    'intelligence_suite': {
        'components': [
            'phase4/P4T1_Implement_traffic_masquerade.py'
        ],
        'cmd': ['python3', 'phase4/P4T1_Implement_traffic_masquerade.py', '--daemon'],
        'description': 'Intelligence & Defense',
        'critical': True
    }
}

PID_DIR = 'logs/core'
os.makedirs(PID_DIR, exist_ok=True)

class CoreOrchestrator:
    def __init__(self):
        self.running = False
        self.processes = {}
        
    def start_service(self, service_name, service_info):
        """Start a core service"""
        pid_file = os.path.join(PID_DIR, f'{service_name}.pid')
        
        # Check if already running
        if os.path.exists(pid_file):
            with open(pid_file, 'r') as f:
                try:
                    pid = int(f.read().strip())
                    os.kill(pid, 0)
                    logger.info(f"{service_name} already running (PID: {pid})")
                    return True
                except:
                    pass
        
        # Start the service
        try:
            log_file = open(f'logs/core/{service_name}.log', 'a')
            process = subprocess.Popen(
                service_info['cmd'],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
            
            with open(pid_file, 'w') as f:
                f.write(str(process.pid))
            
            logger.info(f"Started {service_name} (PID: {process.pid})")
            self.processes[service_name] = process
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {service_name}: {e}")
            return False
    
    def stop_service(self, service_name):
        """Stop a core service"""
        pid_file = os.path.join(PID_DIR, f'{service_name}.pid')
        if os.path.exists(pid_file):
            with open(pid_file, 'r') as f:
                try:
                    pid = int(f.read().strip())
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    logger.info(f"Stopped {service_name} (PID: {pid})")
                except:
                    pass
            os.remove(pid_file)
    
    def start_all(self):
        """Start all core services"""
        logger.info("🚀 Starting all core services...")
        for name, info in CORE_SERVICES.items():
            self.start_service(name, info)
            time.sleep(2)
    
    def stop_all(self):
        """Stop all core services"""
        logger.info("🛑 Stopping all core services...")
        for name in CORE_SERVICES.keys():
            self.stop_service(name)
    
    def status(self):
        """Show status of all core services"""
        print("\n" + "="*60)
        print("🔷 DMAI CORE SERVICES STATUS")
        print("="*60)
        print(f"{'SERVICE':<20} {'STATUS':<10} {'PID':<8} {'CRITICAL':<10}")
        print("-"*60)
        
        all_running = True
        for name, info in CORE_SERVICES.items():
            pid_file = os.path.join(PID_DIR, f'{name}.pid')
            running = False
            pid = None
            
            if os.path.exists(pid_file):
                with open(pid_file, 'r') as f:
                    try:
                        pid = int(f.read().strip())
                        os.kill(pid, 0)
                        running = True
                    except:
                        pass
            
            status = "✅" if running else "❌"
            pid_str = str(pid) if pid else "-"
            critical = "🔴" if info.get('critical', False) else "🟢"
            
            print(f"{info['description']:<20} {status:<10} {pid_str:<8} {critical:<10}")
            
            if not running and info.get('critical', False):
                all_running = False
        
        print("="*60)
        if all_running:
            print("✅ ALL CRITICAL SERVICES RUNNING")
        else:
            print("⚠️  SOME CRITICAL SERVICES ARE DOWN")
        print("="*60)
        
        return all_running
    
    def monitor(self):
        """Continuous monitoring loop"""
        logger.info("Starting core services monitor...")
        self.running = True
        self.start_all()
        
        while self.running:
            try:
                time.sleep(60)  # Check every minute
                for name, info in CORE_SERVICES.items():
                    if info.get('critical', False):
                        pid_file = os.path.join(PID_DIR, f'{name}.pid')
                        if os.path.exists(pid_file):
                            with open(pid_file, 'r') as f:
                                try:
                                    pid = int(f.read().strip())
                                    os.kill(pid, 0)
                                except:
                                    logger.warning(f"{name} is down, restarting...")
                                    self.start_service(name, info)
                        else:
                            self.start_service(name, info)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
        
        self.stop_all()

def main():
    parser = argparse.ArgumentParser(description='DMAI Core Services Orchestrator')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status', 'monitor'])
    parser.add_argument('--service', help='Specific service to manage')
    
    args = parser.parse_args()
    orchestrator = CoreOrchestrator()
    
    if args.service:
        if args.action == 'start':
            orchestrator.start_service(args.service, CORE_SERVICES[args.service])
        elif args.action == 'stop':
            orchestrator.stop_service(args.service)
        elif args.action == 'restart':
            orchestrator.stop_service(args.service)
            time.sleep(2)
            orchestrator.start_service(args.service, CORE_SERVICES[args.service])
        elif args.action == 'status':
            # Show specific service status
            pass
    else:
        if args.action == 'start':
            orchestrator.start_all()
        elif args.action == 'stop':
            orchestrator.stop_all()
        elif args.action == 'restart':
            orchestrator.stop_all()
            time.sleep(3)
            orchestrator.start_all()
        elif args.action == 'status':
            orchestrator.status()
        elif args.action == 'monitor':
            orchestrator.monitor()

if __name__ == '__main__':
    main()
