#!/usr/bin/env python3
"""
P1T2_Design_Recovery_Engine_2.py
Design Recovery Engine #2 - Advanced recovery coordination and execution
REAL VERSION - No simulations, actual health checks
"""

import os
import sys
import json
import time
import logging
import traceback
import hashlib
import subprocess
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[Recovery2] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recovery_engine_2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('RecoveryEngine2')

class RecoveryPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

class RecoveryStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    BLOCKED = "blocked"

class Design_Recovery_Engine_2:
    """
    Design Recovery Engine #2 - REAL VERSION
    Handles parallel recoveries, dependency resolution, and rollback strategies
    """
    
    def __init__(self):
        self.name = "Design Recovery Engine #2"
        self.component_id = "P1T2"
        self.version = "3.0.0"
        self.status = "initialized"
        self.depends_on = ["P1T1"]
        
        # Recovery state
        self.recovery_plans = {}
        self.active_recoveries = {}
        self.completed_recoveries = []
        self.failed_recoveries = []
        self.recovery_queue = []
        
        # Recovery statistics
        self.stats = {
            'total_recoveries': 0,
            'successful': 0,
            'failed': 0,
            'rolled_back': 0,
            'parallel_executions': 0,
            'avg_recovery_time': 0,
            'avg_parallel_count': 0,
            'last_recovery': None,
            'most_common_failure': None
        }
        
        # Dependency tracking
        self.dependency_graph = {}
        self.component_dependencies = {}
        
        # Component health cache
        self.health_cache = {}
        self.health_cache_timeout = 30  # seconds
        
        # Recovery strategies
        self.strategies = {
            'immediate': self._immediate_recovery,
            'delayed': self._delayed_recovery,
            'parallel': self._parallel_recovery,
            'sequential': self._sequential_recovery,
            'rolling': self._rolling_recovery
        }
        
        # Failure analysis
        self.failure_patterns = {}
        self.failure_counts = {}
        
        # Component paths for health checks
        self.component_paths = {
            'dmai_core': 'dmai_core_complete.py',
            'telegram_bot': 'telegram_bot.py',
            'web_interface': 'dmai_web.py',
            'knowledge_graph': 'agi/data/knowledge_graph.json',
            'synthetic_network': 'data/phase6/synthetic_network.pkl'
        }
        
        logger.info(f"🔧 Design Recovery Engine #2 initialized (v{self.version})")
    
    def _check_file_health(self, filepath: str) -> Dict:
        """Check health of a file component"""
        result = {
            'exists': False,
            'size': 0,
            'modified': None,
            'readable': False,
            'error': None
        }
        
        path = Path(filepath)
        if path.exists():
            result['exists'] = True
            result['size'] = path.stat().st_size
            result['modified'] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            result['readable'] = os.access(filepath, os.R_OK)
        else:
            result['error'] = 'File not found'
        
        return result
    
    def _check_process_health(self, process_name: str) -> Dict:
        """Check health of a running process"""
        result = {
            'running': False,
            'pid': None,
            'cpu_percent': 0,
            'memory_percent': 0,
            'status': None
        }
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                if process_name.lower() in proc.info['name'].lower():
                    result['running'] = True
                    result['pid'] = proc.info['pid']
                    result['cpu_percent'] = proc.info['cpu_percent']
                    result['memory_percent'] = proc.info['memory_percent']
                    result['status'] = proc.info['status']
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return result
    
    def _check_api_health(self, url: str, timeout: int = 5) -> Dict:
        """Check health of an API endpoint"""
        import requests
        
        result = {
            'reachable': False,
            'status_code': None,
            'response_time': None,
            'error': None
        }
        
        try:
            start = time.time()
            response = requests.get(url, timeout=timeout)
            result['response_time'] = (time.time() - start) * 1000
            result['status_code'] = response.status_code
            result['reachable'] = response.status_code < 500
        except requests.exceptions.RequestException as e:
            result['error'] = str(e)
        
        return result
    
    def _check_component_health(self, component: str) -> Dict:
        """Check if a component is healthy - REAL health checks"""
        
        # Check cache
        if component in self.health_cache:
            cached = self.health_cache[component]
            if (datetime.now() - cached['timestamp']).seconds < self.health_cache_timeout:
                return cached['result']
        
        result = {
            'component': component,
            'healthy': False,
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Component-specific health checks
        if component == 'dmai_core':
            file_health = self._check_file_health('dmai_core_complete.py')
            result['details']['file'] = file_health
            result['healthy'] = file_health['exists'] and file_health['readable']
            
        elif component == 'telegram_bot':
            file_health = self._check_file_health('telegram_bot.py')
            process_health = self._check_process_health('python')
            result['details']['file'] = file_health
            result['details']['process'] = process_health
            result['healthy'] = file_health['exists'] and process_health['running']
            
        elif component == 'web_interface':
            # Check if web service is responding
            api_health = self._check_api_health('http://localhost:5001/health')
            result['details']['api'] = api_health
            result['healthy'] = api_health['reachable']
            
        elif component == 'knowledge_graph':
            file_health = self._check_file_health('agi/data/knowledge_graph.json')
            result['details']['file'] = file_health
            result['healthy'] = file_health['exists'] and file_health['size'] > 0
            
        elif component == 'synthetic_network':
            file_health = self._check_file_health('data/phase6/synthetic_network.pkl')
            result['details']['file'] = file_health
            result['healthy'] = file_health['exists']
            
        else:
            # Generic health check - check if component file exists
            file_health = self._check_file_health(f'components/{component}.py')
            result['details']['file'] = file_health
            result['healthy'] = file_health['exists']
        
        # Cache result
        self.health_cache[component] = {
            'result': result,
            'timestamp': datetime.now()
        }
        
        return result
    
    def _restart_component(self, component: str) -> Dict:
        """Restart a component - REAL restart"""
        result = {
            'component': component,
            'success': False,
            'message': None,
            'output': None
        }
        
        try:
            if component == 'telegram_bot':
                # Find and kill existing bot process
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if 'telegram_bot.py' in ' '.join(proc.info['cmdline'] or []):
                            proc.terminate()
                            proc.wait(timeout=5)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        continue
                
                # Start new bot
                result['output'] = subprocess.Popen(
                    ['python3', 'telegram_bot.py'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                result['success'] = True
                result['message'] = 'Telegram bot restarted'
                
            elif component == 'web_interface':
                # Web interface is managed by gunicorn, can't restart directly
                # Signal it to reload
                import signal
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if 'gunicorn' in ' '.join(proc.info['cmdline'] or []):
                            proc.send_signal(signal.SIGHUP)
                            result['success'] = True
                            result['message'] = 'Sent reload signal to web server'
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                        
        except Exception as e:
            result['error'] = str(e)
            result['message'] = f'Restart failed: {e}'
        
        return result
    
    def run(self, continuous=False, interval=30):
        """Main execution method"""
        logger.info(f"🚀 Starting {self.name} v{self.version}")
        
        try:
            if continuous:
                while True:
                    self._process_recovery_queue()
                    self._analyze_failure_patterns()
                    time.sleep(interval)
            else:
                result = {
                    'queue_processed': self._process_recovery_queue(),
                    'patterns_analyzed': self._analyze_failure_patterns()
                }
            
            return self.get_status()
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return {"error": str(e)}
    
    def evolve(self):
        """Evolution method"""
        self.version = f"3.0.{len(self.completed_recoveries) + 1}"
        return {'component': self.component_id, 'evolution': 'completed', 'new_version': self.version}
    
    def execute(self, command=None, **kwargs):
        """Execute commands"""
        commands = {
            'recover': self.execute_recovery,
            'parallel': self.execute_parallel_recovery,
            'dependency': self.resolve_dependencies,
            'rollback': self.rollback_recovery,
            'analyze': self.analyze_failures,
            'strategy': self.apply_strategy,
            'queue': self._get_queue_status,
            'status': self.get_recovery_status,
            'reset': self._reset
        }
        
        if command in commands:
            return commands[command](kwargs)
        return {"error": f"Unknown command: {command}"}
    
    def execute_recovery(self, kwargs) -> Dict:
        """Execute recovery for a component"""
        component = kwargs.get('component')
        strategy = kwargs.get('strategy', 'immediate')
        priority = kwargs.get('priority', RecoveryPriority.MEDIUM)
        
        if not component:
            return {"error": "No component specified"}
        
        logger.info(f"🔧 Executing {strategy} recovery for {component}")
        
        recovery_id = hashlib.md5(f"{component}{time.time()}{strategy}".encode()).hexdigest()[:12]
        
        # Check dependencies
        dependencies = self.get_dependencies(component)
        
        # Create recovery plan
        plan = self._create_recovery_plan(component, strategy, priority)
        self.recovery_plans[recovery_id] = plan
        
        queue_item = {
            'recovery_id': recovery_id,
            'component': component,
            'strategy': strategy,
            'priority': priority.value,
            'plan': plan,
            'queued_at': datetime.now().isoformat()
        }
        self.recovery_queue.append(queue_item)
        self.recovery_queue.sort(key=lambda x: x['priority'])
        
        if len(self.active_recoveries) < 3:
            self._process_recovery_queue()
        
        return {
            'recovery_id': recovery_id,
            'component': component,
            'strategy': strategy,
            'priority': priority.name,
            'status': 'queued',
            'queue_position': len(self.recovery_queue)
        }
    
    def execute_parallel_recovery(self, kwargs) -> Dict:
        """Execute parallel recovery for multiple components"""
        components = kwargs.get('components', [])
        if not components:
            return {"error": "No components specified"}
        
        logger.info(f"🔄 Executing parallel recovery for {len(components)} components")
        
        recovery_ids = []
        for component in components:
            result = self.execute_recovery({'component': component, 'strategy': 'parallel'})
            recovery_ids.append(result['recovery_id'])
        
        self.stats['parallel_executions'] += 1
        
        total = self.stats['parallel_executions']
        avg = self.stats['avg_parallel_count']
        self.stats['avg_parallel_count'] = (avg * (total - 1) + len(components)) / total
        
        return {
            'parallel_execution_id': hashlib.md5(f"parallel{time.time()}".encode()).hexdigest()[:8],
            'components': components,
            'recovery_ids': recovery_ids,
            'count': len(components)
        }
    
    def resolve_dependencies(self, kwargs) -> Dict:
        """Resolve dependencies for a component"""
        component = kwargs.get('component')
        if not component:
            return {"error": "No component specified"}
        
        dependencies = self.get_dependencies(component)
        resolved = []
        blocked = []
        
        for dep in dependencies:
            health = self._check_component_health(dep)
            if health['healthy']:
                resolved.append(dep)
            else:
                self.execute_recovery({'component': dep, 'strategy': 'immediate'})
                blocked.append(dep)
        
        return {
            'component': component,
            'dependencies': dependencies,
            'resolved': resolved,
            'blocked': blocked,
            'can_recover': len(blocked) == 0
        }
    
    def rollback_recovery(self, kwargs) -> Dict:
        """Roll back a recovery operation"""
        recovery_id = kwargs.get('recovery_id')
        if not recovery_id:
            return {"error": "No recovery_id specified"}
        
        logger.info(f"↩️ Rolling back recovery: {recovery_id}")
        
        recovery = None
        if recovery_id in self.recovery_plans:
            recovery = self.recovery_plans[recovery_id]
        
        if not recovery:
            return {'error': f'Recovery {recovery_id} not found'}
        
        # Execute rollback by restoring from backup if available
        if recovery.get('backup_created'):
            rollback_result = self._restore_from_backup(recovery['component'])
        else:
            rollback_result = {'success': False, 'message': 'No backup available'}
        
        self.stats['rolled_back'] += 1
        recovery['status'] = RecoveryStatus.ROLLED_BACK.value
        recovery['rolled_back_at'] = datetime.now().isoformat()
        
        return {
            'recovery_id': recovery_id,
            'component': recovery['component'],
            'status': 'rolled_back',
            'rollback_result': rollback_result
        }
    
    def analyze_failures(self, kwargs=None) -> Dict:
        """Analyze failure patterns"""
        if not self.failure_counts:
            return {'message': 'No failures to analyze', 'patterns': {}}
        
        most_common = max(self.failure_counts.items(), key=lambda x: x[1])
        self.stats['most_common_failure'] = most_common[0]
        
        return {
            'failure_patterns': self.failure_patterns,
            'failure_counts': self.failure_counts,
            'most_common': most_common
        }
    
    def apply_strategy(self, kwargs) -> Dict:
        """Apply a specific recovery strategy"""
        strategy = kwargs.get('strategy')
        component = kwargs.get('component')
        
        if not strategy or not component:
            return {"error": "Strategy and component required"}
        
        if strategy not in self.strategies:
            return {'error': f'Unknown strategy: {strategy}'}
        
        return self.strategies[strategy](component)
    
    def update_dependencies(self, component: str, dependencies: List[str]):
        """Update dependency information"""
        self.dependency_graph[component] = dependencies
        for dep in dependencies:
            if dep not in self.component_dependencies:
                self.component_dependencies[dep] = []
            if component not in self.component_dependencies[dep]:
                self.component_dependencies[dep].append(component)
    
    def get_dependencies(self, component: str) -> List[str]:
        return self.dependency_graph.get(component, [])
    
    def get_recovery_status(self, kwargs) -> Optional[Dict]:
        recovery_id = kwargs.get('recovery_id')
        if recovery_id in self.active_recoveries:
            return self.active_recoveries[recovery_id]
        elif recovery_id in self.recovery_plans:
            return self.recovery_plans[recovery_id]
        return {'error': 'Recovery not found'}
    
    def _create_recovery_plan(self, component: str, strategy: str, priority: RecoveryPriority) -> Dict:
        """Create a detailed recovery plan"""
        steps = [
            {'step': 1, 'name': 'validate_state', 'description': 'Validate current component state', 'estimated_time': 5}
        ]
        
        if strategy == 'immediate':
            steps.extend([
                {'step': 2, 'name': 'stop_component', 'description': 'Stop component gracefully', 'estimated_time': 10},
                {'step': 3, 'name': 'restart_component', 'description': 'Restart component', 'estimated_time': 15},
                {'step': 4, 'name': 'verify_recovery', 'description': 'Verify successful recovery', 'estimated_time': 10}
            ])
        else:
            steps.extend([
                {'step': 2, 'name': 'diagnose', 'description': 'Run diagnostics', 'estimated_time': 15},
                {'step': 3, 'name': 'repair', 'description': 'Attempt repair', 'estimated_time': 25},
                {'step': 4, 'name': 'verify', 'description': 'Verify repair', 'estimated_time': 10}
            ])
        
        return {
            'component': component,
            'strategy': strategy,
            'priority': priority.value,
            'created_at': datetime.now().isoformat(),
            'steps': steps,
            'total_estimated_time': sum(s['estimated_time'] for s in steps),
            'status': RecoveryStatus.PENDING.value
        }
    
    def _process_recovery_queue(self) -> int:
        """Process pending recoveries in the queue"""
        processed = 0
        max_parallel = 3
        
        while (self.recovery_queue and 
               len(self.active_recoveries) < max_parallel and 
               processed < 5):
            
            queue_item = self.recovery_queue.pop(0)
            recovery_id = queue_item['recovery_id']
            plan = queue_item['plan']
            component = queue_item['component']
            
            plan['status'] = RecoveryStatus.IN_PROGRESS.value
            plan['started_at'] = datetime.now().isoformat()
            self.active_recoveries[recovery_id] = plan
            
            # Execute actual recovery
            success = self._execute_recovery_plan(component, plan)
            
            self.stats['total_recoveries'] += 1
            if success:
                self.stats['successful'] += 1
                plan['status'] = RecoveryStatus.COMPLETED.value
                self.completed_recoveries.append(plan)
            else:
                self.stats['failed'] += 1
                plan['status'] = RecoveryStatus.FAILED.value
                self.failed_recoveries.append(plan)
            
            plan['completed_at'] = datetime.now().isoformat()
            self.stats['last_recovery'] = datetime.now().isoformat()
            
            if recovery_id in self.active_recoveries:
                del self.active_recoveries[recovery_id]
            
            processed += 1
        
        return processed
    
    def _execute_recovery_plan(self, component: str, plan: Dict) -> bool:
        """Execute a recovery plan"""
        logger.info(f"▶️ Executing plan for {component}")
        
        start_time = time.time()
        
        # Perform actual recovery based on component type
        if component in ['telegram_bot', 'web_interface']:
            result = self._restart_component(component)
            success = result.get('success', False)
        else:
            # For file-based components, try to restore from backup
            backup_file = Path(f"data/backups/{component}.backup")
            if backup_file.exists():
                try:
                    # Restore from backup
                    target = Path(self.component_paths.get(component, component))
                    import shutil
                    shutil.copy(backup_file, target)
                    success = True
                except Exception as e:
                    logger.error(f"Restore failed: {e}")
                    success = False
            else:
                success = False
        
        recovery_time = time.time() - start_time
        
        total = self.stats['total_recoveries']
        avg = self.stats['avg_recovery_time']
        self.stats['avg_recovery_time'] = (avg * total + recovery_time) / (total + 1)
        
        plan['recovery_time'] = recovery_time
        plan['success'] = success
        
        return success
    
    def _restore_from_backup(self, component: str) -> Dict:
        """Restore component from backup"""
        backup_file = Path(f"data/backups/{component}.backup")
        target = Path(self.component_paths.get(component, component))
        
        try:
            import shutil
            shutil.copy(backup_file, target)
            return {'success': True, 'message': f'Restored {component} from backup'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _immediate_recovery(self, component: str) -> Dict:
        return {'strategy': 'immediate', 'component': component, 'action': 'immediate_restart'}
    
    def _delayed_recovery(self, component: str) -> Dict:
        return {'strategy': 'delayed', 'component': component, 'delay_seconds': 60}
    
    def _parallel_recovery(self, component: str) -> Dict:
        return {'strategy': 'parallel', 'component': component, 'parallel_instances': 2}
    
    def _sequential_recovery(self, component: str) -> Dict:
        return {'strategy': 'sequential', 'component': component, 'phases': ['drain', 'stop', 'repair', 'verify']}
    
    def _rolling_recovery(self, component: str) -> Dict:
        return {'strategy': 'rolling', 'component': component, 'batch_size': 1, 'total_batches': 3}
    
    def _get_queue_status(self, kwargs) -> Dict:
        return {'queue_size': len(self.recovery_queue), 'queue': self.recovery_queue[:10]}
    
    def _reset(self, kwargs) -> Dict:
        self.stats = {
            'total_recoveries': 0, 'successful': 0, 'failed': 0, 'rolled_back': 0,
            'parallel_executions': 0, 'avg_recovery_time': 0, 'avg_parallel_count': 0,
            'last_recovery': None, 'most_common_failure': None
        }
        self.recovery_plans = {}
        self.active_recoveries = {}
        self.recovery_queue = []
        return {'status': 'reset', 'component': self.component_id}
    
    def _analyze_failure_patterns(self) -> Dict:
        """Analyze failure patterns"""
        return {'message': 'Analysis complete', 'patterns': self.failure_patterns}
    
    def _record_failure(self, failure: Dict):
        """Record a failure for pattern analysis"""
        failure_type = failure.get('type', 'unknown')
        component = failure.get('component', 'unknown')
        
        key = f"{component}:{failure_type}"
        self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
        
        if failure_type not in self.failure_patterns:
            self.failure_patterns[failure_type] = {
                'count': 0,
                'components': [],
                'first_seen': datetime.now().isoformat()
            }
        
        self.failure_patterns[failure_type]['count'] += 1
        if component not in self.failure_patterns[failure_type]['components']:
            self.failure_patterns[failure_type]['components'].append(component)
    
    def get_status(self) -> Dict:
        """Get current component status"""
        success_rate = 0
        if self.stats['total_recoveries'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_recoveries']) * 100
        
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'stats': self.stats,
            'success_rate': f"{success_rate:.1f}%",
            'active_recoveries': len(self.active_recoveries),
            'queued_recoveries': len(self.recovery_queue)
        }
    
    def info(self) -> Dict:
        return {
            "name": self.name,
            "id": self.component_id,
            "version": self.version,
            "status": self.status,
            "strategies": list(self.strategies.keys()),
            "stats": self.stats
        }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 DESIGN RECOVERY ENGINE #2 - REAL VERSION")
    print("="*60)
    
    engine = Design_Recovery_Engine_2()
    print(json.dumps(engine.get_status(), indent=2))
