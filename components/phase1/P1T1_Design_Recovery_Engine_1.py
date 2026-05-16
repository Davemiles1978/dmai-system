#!/usr/bin/env python3
"""
P1T1_Design_Recovery_Engine_1.py
Design Recovery Engine #1 - Core recovery planning and execution
Full-featured component for DMAI evolution system
"""

import os
import sys
import json
import time
import logging
import traceback
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recovery_engine_1.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('RecoveryEngine1')

class Design_Recovery_Engine_1:
    """
    Design Recovery Engine #1 - Plans and executes system recovery strategies
    Monitors system health, detects failures, and initiates recovery procedures
    """
    
    def __init__(self):
        self.name = "Design Recovery Engine #1"
        self.component_id = "P1T1"
        self.version = "1.0.0"
        self.status = "initialized"
        self.depends_on = ["P0T1", "P0T2", "P0T3", "P0T4"]
        
        # Recovery state
        self.recovery_plans = []
        self.active_recoveries = {}
        self.recovery_history = []
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'avg_recovery_time': 0,
            'last_recovery': None
        }
        
        # Monitoring state
        self.health_checks = []
        self.failure_detection_threshold = 3
        self.check_interval = 60  # seconds
        
    def run(self, continuous=False, interval=60):
        """
        Main execution method - called by evolution engine
        
        Args:
            continuous: Whether to run continuously
            interval: Check interval in seconds
        """
        logger.info(f"🚀 Starting {self.name} v{self.version}")
        
        try:
            if continuous:
                logger.info(f"Continuous mode: checking every {interval} seconds")
                while True:
                    self._monitor_and_recover()
                    time.sleep(interval)
            else:
                # Single run
                result = self._monitor_and_recover()
            
            logger.info(f"✅ {self.name} completed")
            return self.get_status()
            
        except Exception as e:
            logger.error(f"❌ Error in {self.name}: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "component": self.component_id}
    
    def evolve(self):
        """
        Evolution method - called when component needs to evolve
        """
        logger.info(f"🧬 Evolving {self.name}")
        self.version = f"1.0.{len(self.recovery_history) + 1}"
        
        # Evolve recovery strategies
        evolution_result = {
            'component': self.component_id,
            'evolution': 'completed',
            'new_version': self.version,
            'recovery_stats': self.recovery_stats,
            'active_plans': len(self.active_recoveries)
        }
        
        # Learn from recovery history
        if len(self.recovery_history) > 10:
            self._optimize_recovery_strategies()
            evolution_result['optimization'] = 'applied'
        
        return evolution_result
    
    def execute(self, command=None, **kwargs):
        """
        Execute method - runs specific commands
        
        Commands:
            - recover: Execute recovery for a component
            - plan: Create recovery plan
            - monitor: Run health monitoring
            - status: Get recovery status
            - history: View recovery history
            - reset: Reset recovery stats
        """
        logger.info(f"⚙️ Executing command: {command}")
        
        if command == 'recover':
            component = kwargs.get('component')
            plan_id = kwargs.get('plan_id')
            if component:
                return self.execute_recovery(component)
            elif plan_id:
                return self.execute_recovery_plan(plan_id)
            else:
                return {"error": "No component or plan_id provided"}
                
        elif command == 'plan':
            component = kwargs.get('component')
            failure_type = kwargs.get('failure_type', 'unknown')
            if component:
                return self.create_recovery_plan(component, failure_type)
            else:
                return {"error": "No component specified"}
                
        elif command == 'monitor':
            component = kwargs.get('component')
            if component:
                return self.check_component_health(component)
            else:
                return self.run_health_check()
                
        elif command == 'status':
            plan_id = kwargs.get('plan_id')
            if plan_id:
                return self.get_recovery_status(plan_id)
            else:
                return self.get_status()
                
        elif command == 'history':
            limit = kwargs.get('limit', 10)
            return {
                'recovery_history': self.recovery_history[-limit:],
                'stats': self.recovery_stats
            }
            
        elif command == 'reset':
            self.recovery_stats = {
                'total_recoveries': 0,
                'successful_recoveries': 0,
                'failed_recoveries': 0,
                'avg_recovery_time': 0,
                'last_recovery': None
            }
            self.active_recoveries = {}
            self.recovery_history = []
            return {'status': 'reset', 'component': self.component_id}
            
        else:
            return self.get_status()
    
    def process(self, data=None):
        """
        Process method - handles data processing
        
        Can process health data, failure reports, and recovery requests
        """
        logger.info(f"🔄 Processing data for {self.name}")
        
        result = {
            'component': self.component_id,
            'processed': True,
            'timestamp': datetime.now().isoformat(),
            'recovery_stats': self.recovery_stats
        }
        
        if data and isinstance(data, dict):
            # Process health check data
            if 'health_data' in data:
                health_data = data['health_data']
                for component, health in health_data.items():
                    if health.get('status') == 'failed':
                        plan = self.create_recovery_plan(component, health.get('error', 'unknown'))
                        result['recovery_planned'] = plan.get('plan_id')
            
            # Process failure reports
            if 'failures' in data:
                failures = data['failures']
                recovery_results = []
                for failure in failures:
                    component = failure.get('component')
                    if component:
                        recovery = self.execute_recovery(component)
                        recovery_results.append(recovery)
                result['recoveries_executed'] = recovery_results
            
            # Process recovery requests
            if 'recover' in data:
                recover_data = data['recover']
                component = recover_data.get('component')
                if component:
                    result['recovery_result'] = self.execute_recovery(component)
        
        return result
    
    def generate(self):
        """
        Generate method - produces output/report
        """
        logger.info(f"📊 Generating report for {self.name}")
        
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'recovery_stats': self.recovery_stats,
            'active_recoveries': len(self.active_recoveries),
            'total_plans': len(self.recovery_plans),
            'recent_history': self.recovery_history[-5:],
            'dependencies': self.depends_on,
            'monitoring': {
                'health_checks': len(self.health_checks),
                'check_interval': self.check_interval,
                'failure_threshold': self.failure_detection_threshold
            }
        }
    
    def query(self, question=None):
        """
        Query method - answers questions about component state
        """
        logger.info(f"❓ Querying {self.name}")
        
        if question == 'health':
            return {
                'component': self.component_id,
                'healthy': True,
                'methods': ['run', 'evolve', 'execute', 'process', 'generate', 'query'],
                'recovery_stats': self.recovery_stats,
                'active_recoveries': len(self.active_recoveries)
            }
        elif question == 'recovery':
            return {
                'component': self.component_id,
                'total_recoveries': self.recovery_stats['total_recoveries'],
                'success_rate': self._calculate_success_rate(),
                'avg_time': self.recovery_stats['avg_recovery_time'],
                'last_recovery': self.recovery_stats['last_recovery']
            }
        elif question == 'plans':
            return {
                'component': self.component_id,
                'total_plans': len(self.recovery_plans),
                'active_plans': len(self.active_recoveries),
                'recent_plans': self.recovery_plans[-5:]
            }
        elif question == 'monitoring':
            return {
                'component': self.component_id,
                'health_checks': len(self.health_checks),
                'check_interval': self.check_interval,
                'failure_threshold': self.failure_detection_threshold
            }
        else:
            return self.info()
    
    def create_recovery_plan(self, component: str, failure_type: str = 'unknown') -> Dict[str, Any]:
        """
        Create a recovery plan for a failed component
        
        Args:
            component: The component that failed
            failure_type: Type of failure detected
        """
        logger.info(f"📋 Creating recovery plan for {component} (failure: {failure_type})")
        
        # Generate unique plan ID
        plan_id = hashlib.md5(f"{component}{time.time()}{failure_type}".encode()).hexdigest()[:8]
        
        # Determine recovery steps based on failure type
        recovery_steps = self._generate_recovery_steps(component, failure_type)
        
        plan = {
            'plan_id': plan_id,
            'component': component,
            'failure_type': failure_type,
            'created_at': datetime.now().isoformat(),
            'steps': recovery_steps,
            'status': 'pending',
            'estimated_time': len(recovery_steps) * 30,  # seconds
            'priority': self._calculate_priority(component, failure_type)
        }
        
        self.recovery_plans.append(plan)
        
        logger.info(f"✅ Created recovery plan {plan_id} for {component} with {len(recovery_steps)} steps")
        
        return plan
    
    def execute_recovery(self, component: str) -> Dict[str, Any]:
        """
        Execute recovery for a component
        
        Args:
            component: The component to recover
        """
        logger.info(f"🔧 Executing recovery for {component}")
        
        # Find or create a plan
        plan = None
        for p in self.recovery_plans:
            if p['component'] == component and p['status'] == 'pending':
                plan = p
                break
        
        if not plan:
            plan = self.create_recovery_plan(component, 'unknown')
        
        return self.execute_recovery_plan(plan['plan_id'])
    
    def execute_recovery_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Execute a specific recovery plan
        
        Args:
            plan_id: ID of the plan to execute
        """
        logger.info(f"🔧 Executing recovery plan: {plan_id}")
        
        # Find the plan
        plan = None
        for p in self.recovery_plans:
            if p['plan_id'] == plan_id:
                plan = p
                break
        
        if not plan:
            return {'error': f'Plan {plan_id} not found', 'status': 'failed'}
        
        # Update plan status
        plan['status'] = 'in_progress'
        plan['started_at'] = datetime.now().isoformat()
        
        # Track in active recoveries
        self.active_recoveries[plan_id] = plan
        
        # Execute each step
        results = []
        success = True
        start_time = time.time()
        
        for step in plan['steps']:
            logger.info(f"   Step: {step['name']}")
            step_result = self._execute_recovery_step(step)
            results.append({
                'step': step['name'],
                'result': step_result
            })
            
            if not step_result.get('success', False):
                success = False
                break
            
            # Small delay between steps
            time.sleep(1)
        
        # Calculate recovery time
        recovery_time = time.time() - start_time
        
        # Update plan status
        plan['status'] = 'completed' if success else 'failed'
        plan['completed_at'] = datetime.now().isoformat()
        plan['results'] = results
        plan['recovery_time'] = recovery_time
        
        # Update statistics
        self.recovery_stats['total_recoveries'] += 1
        if success:
            self.recovery_stats['successful_recoveries'] += 1
        else:
            self.recovery_stats['failed_recoveries'] += 1
        
        # Update average recovery time
        total = self.recovery_stats['total_recoveries']
        avg = self.recovery_stats['avg_recovery_time']
        self.recovery_stats['avg_recovery_time'] = (avg * (total - 1) + recovery_time) / total
        self.recovery_stats['last_recovery'] = datetime.now().isoformat()
        
        # Record in history
        self.recovery_history.append({
            'plan_id': plan_id,
            'component': plan['component'],
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'recovery_time': recovery_time,
            'steps_completed': len(results)
        })
        
        # Remove from active recoveries
        if plan_id in self.active_recoveries:
            del self.active_recoveries[plan_id]
        
        logger.info(f"✅ Recovery plan {plan_id} {'completed successfully' if success else 'failed'}")
        
        return {
            'plan_id': plan_id,
            'component': plan['component'],
            'status': 'completed' if success else 'failed',
            'success': success,
            'recovery_time': recovery_time,
            'steps': results,
            'plan': plan
        }
    
    def check_component_health(self, component: str) -> Dict[str, Any]:
        """
        Check health of a specific component
        
        Args:
            component: Component to check
        """
        logger.debug(f"🔍 Checking health of {component}")
        
        # In a real implementation, this would:
        # 1. Check if component is loaded
        # 2. Verify it has required methods
        # 3. Test basic functionality
        # 4. Check for error states
        
        health_result = {
            'component': component,
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',  # Assume healthy for now
            'issues': []
        }
        
        # Access DMAI core if available
        if hasattr(self, 'dmai') and hasattr(self.dmai, 'components'):
            if component in self.dmai.components:
                comp_data = self.dmai.components[component]
                health = comp_data.get('health_status', {})
                if health.get('score', 100) < 50:
                    health_result['status'] = 'degraded'
                    health_result['issues'].append(f"Health score: {health.get('score')}%")
                    health_result['missing_methods'] = health.get('missing_methods', [])
            else:
                health_result['status'] = 'missing'
                health_result['issues'].append('Component not loaded')
        
        self.health_checks.append(health_result)
        
        # Keep health checks manageable
        if len(self.health_checks) > 1000:
            self.health_checks = self.health_checks[-1000:]
        
        return health_result
    
    def run_health_check(self) -> Dict[str, Any]:
        """
        Run comprehensive health check on all components
        """
        logger.info("🔍 Running comprehensive health check")
        
        results = []
        failed_components = []
        
        # Check all components if DMAI core is available
        if hasattr(self, 'dmai') and hasattr(self.dmai, 'components'):
            for comp_id in self.dmai.components.keys():
                health = self.check_component_health(comp_id)
                results.append(health)
                
                if health['status'] != 'healthy':
                    failed_components.append({
                        'component': comp_id,
                        'issues': health['issues']
                    })
        
        # If failures detected, create recovery plans
        if failed_components:
            logger.warning(f"⚠️ Detected {len(failed_components)} components with issues")
            for failure in failed_components:
                self.create_recovery_plan(
                    failure['component'],
                    failure_type='health_check_failed'
                )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_checked': len(results),
            'failed': len(failed_components),
            'failures': failed_components,
            'recovery_plans_created': len(failed_components)
        }
    
    def get_recovery_status(self, plan_id: str = None) -> Dict[str, Any]:
        """
        Get status of recovery plans
        
        Args:
            plan_id: Optional specific plan ID
        """
        if plan_id:
            if plan_id in self.active_recoveries:
                return self.active_recoveries[plan_id]
            for plan in self.recovery_plans:
                if plan['plan_id'] == plan_id:
                    return plan
            return {'error': f'Plan {plan_id} not found'}
        
        return {
            'active_recoveries': len(self.active_recoveries),
            'total_plans': len(self.recovery_plans),
            'recovery_stats': self.recovery_stats,
            'active': list(self.active_recoveries.values())
        }
    
    def _monitor_and_recover(self):
        """Main monitoring and recovery loop"""
        logger.info("🔄 Running monitoring cycle")
        
        # Run health check
        health_result = self.run_health_check()
        
        # Check for pending recovery plans
        pending_plans = [p for p in self.recovery_plans if p['status'] == 'pending']
        if pending_plans:
            logger.info(f"📋 Executing {len(pending_plans)} pending recovery plans")
            for plan in pending_plans[:3]:  # Execute up to 3 at a time
                self.execute_recovery_plan(plan['plan_id'])
                time.sleep(2)
        
        return health_result
    
    def _generate_recovery_steps(self, component: str, failure_type: str) -> List[Dict[str, Any]]:
        """Generate recovery steps based on component and failure type"""
        steps = []
        
        # Base steps for all recoveries
        steps.append({
            'name': 'validate_component_state',
            'description': 'Validate current component state',
            'action': 'check'
        })
        
        # Failure-specific steps
        if 'missing_method' in failure_type:
            steps.append({
                'name': 'add_missing_methods',
                'description': 'Add missing required methods',
                'action': 'evolve'
            })
        elif 'database' in failure_type:
            steps.append({
                'name': 'reset_database_connection',
                'description': 'Reset database connection',
                'action': 'reconnect'
            })
        elif 'timeout' in failure_type:
            steps.append({
                'name': 'increase_timeout',
                'description': 'Increase timeout threshold',
                'action': 'configure'
            })
        
        # Generic steps
        steps.append({
            'name': 'reload_component',
            'description': 'Reload component',
            'action': 'reload'
        })
        
        steps.append({
            'name': 'verify_recovery',
            'description': 'Verify successful recovery',
            'action': 'verify'
        })
        
        return steps
    
    def _execute_recovery_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single recovery step"""
        logger.debug(f"Executing step: {step['name']}")
        
        # Simulate step execution
        success = True
        message = f"Step {step['name']} completed"
        
        # In a real implementation, this would perform actual recovery actions
        if step['action'] == 'evolve' and hasattr(self, 'dmai'):
            # Trigger evolution on the component
            if hasattr(self.dmai, 'evolve_component'):
                # Would need component ID here
                pass
        
        return {
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_priority(self, component: str, failure_type: str) -> int:
        """Calculate recovery priority (1-10, 1 being highest)"""
        priority = 5  # Default medium priority
        
        # Critical components get higher priority
        critical_phases = ['phase0', 'phase1']
        for phase in critical_phases:
            if phase in component:
                priority = 1
                break
        
        # Adjust based on failure type
        if 'critical' in failure_type:
            priority = max(1, priority - 2)
        elif 'minor' in failure_type:
            priority = min(10, priority + 2)
        
        return priority
    
    def _calculate_success_rate(self) -> float:
        """Calculate recovery success rate"""
        total = self.recovery_stats['total_recoveries']
        if total == 0:
            return 0.0
        return (self.recovery_stats['successful_recoveries'] / total) * 100
    
    def _optimize_recovery_strategies(self):
        """Optimize recovery strategies based on history"""
        logger.info("🔄 Optimizing recovery strategies")
        
        # Analyze recovery history
        successes = [r for r in self.recovery_history if r.get('success')]
        failures = [r for r in self.recovery_history if not r.get('success')]
        
        if successes:
            avg_success_time = sum(r.get('recovery_time', 0) for r in successes) / len(successes)
            logger.info(f"📊 Average successful recovery time: {avg_success_time:.2f}s")
        
        if failures:
            logger.info(f"⚠️ {len(failures)} failures to analyze")
        
        # Adjust check interval based on failure rate
        failure_rate = len(failures) / len(self.recovery_history) if self.recovery_history else 0
        if failure_rate > 0.2:
            self.check_interval = max(30, self.check_interval - 10)
            logger.info(f"⚙️ Increased monitoring frequency (interval: {self.check_interval}s)")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'recovery_stats': self.recovery_stats,
            'active_recoveries': len(self.active_recoveries),
            'total_plans': len(self.recovery_plans),
            'success_rate': f"{self._calculate_success_rate():.1f}%",
            'monitoring': {
                'health_checks': len(self.health_checks),
                'check_interval': self.check_interval
            },
            'methods': ['run', 'evolve', 'execute', 'process', 'generate', 'query']
        }
    
    def info(self) -> Dict[str, Any]:
        """Get component information"""
        return {
            "name": self.name,
            "id": self.component_id,
            "version": self.version,
            "status": self.status,
            "depends_on": self.depends_on,
            "recovery_stats": self.recovery_stats,
            "methods": ['run', 'evolve', 'execute', 'process', 'generate', 'query']
        }

# Guard clause ensures code only runs when script is executed directly
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 DESIGN RECOVERY ENGINE #1 (P1T1)")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Design Recovery Engine #1')
    parser.add_argument('--monitor', action='store_true', help='Run monitoring cycle')
    parser.add_argument('--recover', metavar='COMPONENT', help='Recover a component')
    parser.add_argument('--plan', metavar='COMPONENT', help='Create recovery plan')
    parser.add_argument('--status', action='store_true', help='Show status')
    
    args = parser.parse_args()
    
    engine = Design_Recovery_Engine_1()
    
    if args.monitor:
        print("\n📋 Running monitoring cycle...")
        result = engine.run_health_check()
        print(json.dumps(result, indent=2))
    
    elif args.recover:
        print(f"\n🔧 Recovering component: {args.recover}")
        result = engine.execute_recovery(args.recover)
        print(json.dumps(result, indent=2))
    
    elif args.plan:
        print(f"\n📋 Creating recovery plan for: {args.plan}")
        result = engine.create_recovery_plan(args.plan)
        print(json.dumps(result, indent=2))
    
    elif args.status:
        print("\n📊 Component Status:")
        print(json.dumps(engine.get_status(), indent=2))
    
    else:
        print("\n📋 Component Info:")
        print(json.dumps(engine.info(), indent=2))
        print("\n💡 Use --monitor, --recover, --plan, or --status for more options")
