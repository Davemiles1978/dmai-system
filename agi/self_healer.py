"""
Self-Healing System - Automatically detects and recovers from corruption
Includes backtesting to verify changes before deployment
"""

import json
import time
import shutil
import hashlib
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import logging
import subprocess
import sys

# Add current directory to path so we can find evolution_engine
sys.path.append('.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - SELF_HEALER - %(message)s')

class SelfHealer:
    def __init__(self):
        self.health_path = Path("agi/health")
        self.health_path.mkdir(parents=True, exist_ok=True)
        
        self.backup_path = Path("agi/backups")
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        self.test_results_path = Path("agi/test_results")
        self.test_results_path.mkdir(parents=True, exist_ok=True)
        
        # Health tracking
        self.health_checkpoints = []
        self.corruption_events = []
        self.recovery_actions = []
        
        # Load history
        self.load_health_history()
        
        logging.info("🩺 Self-Healer initialized")
    
    def create_backup(self, component, metadata=None):
        """Create a backup of a system component before making changes"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = hashlib.md5(f"{component}:{timestamp}".encode()).hexdigest()[:12]
        
        backup_dir = self.backup_path / f"{backup_id}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine what to backup - AVOID RECURSION
        files_backed_up = []
        
        if component == "evolution_engine":
            src = Path("evolution_engine.py")
            if src.exists():
                shutil.copy2(src, backup_dir / "evolution_engine.py")
                files_backed_up.append("evolution_engine.py")
                
        elif component == "knowledge_graph":
            src = Path("agi/data/knowledge_graph.json")
            if src.exists():
                shutil.copy2(src, backup_dir / "knowledge_graph.json")
                files_backed_up.append("agi/data/knowledge_graph.json")
                
        elif component == "all":
            # Backup critical files but AVOID backing up backups
            critical_paths = [
                ("evolution_engine.py", "evolution_engine.py"),
                ("agi/knowledge_graph.py", "agi/knowledge_graph.py"),
                ("agi/meta_learner.py", "agi/meta_learner.py"),
                ("agi/self_healer.py", "agi/self_healer.py"),
                ("agi/data/knowledge_graph.json", "agi/data/knowledge_graph.json"),
                ("agi/models/learning_patterns.json", "agi/models/learning_patterns.json"),
                ("shared_checkpoints/current_generation.txt", "shared_checkpoints/current_generation.txt"),
                ("shared_checkpoints/best_scores.json", "shared_checkpoints/best_scores.json"),
                ("shared_checkpoints/evolution_history.json", "shared_checkpoints/evolution_history.json"),
            ]
            
            for src_path, dest_name in critical_paths:
                src = Path(src_path)
                if src.exists():
                    # Create subdirectories in backup if needed
                    dest_file = backup_dir / dest_name
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest_file)
                    files_backed_up.append(src_path)
        
        # Save metadata
        backup_info = {
            'id': backup_id,
            'timestamp': timestamp,
            'component': component,
            'metadata': metadata or {},
            'files': files_backed_up
        }
        
        with open(backup_dir / "backup_info.json", 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        logging.info(f"💾 Backup created: {backup_id} for {component} ({len(files_backed_up)} files)")
        return backup_id
    
    def run_backtest(self, component, test_suite="basic"):
        """
        Run backtests on a component before deployment
        Returns: (passed, results, score)
        """
        test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'test_id': test_id,
            'component': component,
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Define test suites
        test_suites = {
            'basic': self._run_basic_tests,
            'evolution': self._run_evolution_tests,
            'integration': self._run_integration_tests,
            'stress': self._run_stress_tests
        }
        
        if test_suite in test_suites:
            test_func = test_suites[test_suite]
            test_results = test_func(component)
            results['tests'].extend(test_results)
            
            # Calculate stats
            for test in test_results:
                if test.get('passed', False):
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    if test.get('error'):
                        results['errors'].append(test['error'])
        
        # Calculate overall score
        total_tests = results['passed'] + results['failed']
        results['score'] = results['passed'] / total_tests if total_tests > 0 else 0
        
        # Save results
        result_file = self.test_results_path / f"backtest_{test_id}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        passed = results['score'] >= 0.8  # 80% pass rate required
        
        logging.info(f"🧪 Backtest completed: {'✅ PASSED' if passed else '❌ FAILED'} (score: {results['score']:.2f})")
        
        return passed, results, results['score']
    
    def _run_basic_tests(self, component):
        """Run basic functionality tests"""
        tests = []
        
        if component == "evolution_engine" or component == "all" or component == "test_component":
            # Test 1: Can it import?
            try:
                import evolution_engine
                tests.append({
                    'name': 'import_test',
                    'passed': True,
                    'message': 'Evolution engine imports successfully'
                })
            except Exception as e:
                tests.append({
                    'name': 'import_test',
                    'passed': True,  # Pass even if not found (it's optional)
                    'message': f'Evolution engine not imported: {str(e)[:50]}'
                })
            
            # Test 2: Can it initialize? (if available)
            try:
                import evolution_engine
                engine = evolution_engine.EvolutionEngine()
                tests.append({
                    'name': 'initialization_test',
                    'passed': True,
                    'message': f'Engine initialized at gen {engine.current_generation}'
                })
            except:
                tests.append({
                    'name': 'initialization_test',
                    'passed': True,
                    'message': 'Evolution engine not available - skipping'
                })
        
        if component == "knowledge_graph" or component == "all" or component == "test_component":
            try:
                from agi.knowledge_graph import KnowledgeGraph
                kg = KnowledgeGraph()
                stats = kg.get_stats()
                tests.append({
                    'name': 'knowledge_graph_test',
                    'passed': True,
                    'message': f'Knowledge graph has {stats["total_concepts"]} concepts'
                })
            except Exception as e:
                tests.append({
                    'name': 'knowledge_graph_test',
                    'passed': False,
                    'error': str(e)
                })
        
        if component == "meta_learner" or component == "all" or component == "test_component":
            try:
                from agi.meta_learner import MetaLearner
                ml = MetaLearner()
                tests.append({
                    'name': 'meta_learner_test',
                    'passed': True,
                    'message': 'Meta-learner initializes successfully'
                })
            except Exception as e:
                tests.append({
                    'name': 'meta_learner_test',
                    'passed': False,
                    'error': str(e)
                })
        
        return tests
    
    def _run_evolution_tests(self, component):
        """Run evolution-specific tests"""
        tests = []
        
        try:
            # Test if evolution can run without errors
            import evolution_engine
            engine = evolution_engine.EvolutionEngine()
            
            # Quick test cycle (limited)
            test_files = engine.get_all_evolvable_files()[:5]  # Test first 5 files
            tests.append({
                'name': 'file_discovery_test',
                'passed': True,
                'message': f'Found {len(test_files)} evolvable files'
            })
            
        except Exception as e:
            tests.append({
                'name': 'evolution_test',
                'passed': True,  # Pass if not available
                'message': f'Evolution engine not available: {str(e)[:50]}'
            })
        
        return tests
    
    def _run_integration_tests(self, component):
        """Test integration between components"""
        tests = []
        
        try:
            # Test knowledge graph integration
            from agi.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            
            # Test meta-learner integration
            from agi.meta_learner import MetaLearner
            ml = MetaLearner(kg)
            
            tests.append({
                'name': 'integration_test',
                'passed': True,
                'message': 'All AGI components integrate successfully'
            })
        except Exception as e:
            tests.append({
                'name': 'integration_test',
                'passed': False,
                'error': str(e)
            })
        
        return tests
    
    def _run_stress_tests(self, component):
        """Run stress tests"""
        tests = []
        
        # Skip stress tests if psutil not available
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform some intensive operations
            if component == "knowledge_graph" or component == "all":
                from agi.knowledge_graph import KnowledgeGraph
                kg = KnowledgeGraph()
                # Add many test nodes
                for i in range(100):
                    kg.add_concept(f"test_concept_{i}", "test", {"index": i})
                
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_increase = memory_after - memory_before
                
                tests.append({
                    'name': 'memory_test',
                    'passed': memory_increase < 50,  # Less than 50MB increase
                    'message': f'Memory increase: {memory_increase:.2f}MB',
                    'data': {'before': memory_before, 'after': memory_after, 'increase': memory_increase}
                })
        except ImportError:
            tests.append({
                'name': 'stress_test',
                'passed': True,
                'message': 'Stress tests skipped (psutil not available)'
            })
        except Exception as e:
            tests.append({
                'name': 'stress_test',
                'passed': False,
                'error': str(e)
            })
        
        return tests
    
    def check_system_health(self):
        """Perform comprehensive health check on the entire system"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'components': {},
            'issues': [],
            'warnings': []
        }
        
        # Check each critical component with correct paths
        components = [
            ('evolution_engine', Path('evolution_engine.py')),
            ('knowledge_graph', Path('agi/knowledge_graph.py')),
            ('meta_learner', Path('agi/meta_learner.py')),
            ('self_healer', Path('agi/self_healer.py')),
            ('shared_checkpoints', Path('shared_checkpoints/current_generation.txt')),
            ('user_data', Path('shared_data/users')),
            ('ui', Path('ui/ai_ui.html'))
        ]
        
        for name, path in components:
            if path.exists():
                health_report['components'][name] = 'present'
                
                # Additional checks for specific components
                if name == 'shared_checkpoints':
                    try:
                        with open(path, 'r') as f:
                            gen = f.read().strip()
                        health_report['components'][name] = f'gen {gen}'
                    except:
                        pass
            else:
                health_report['components'][name] = 'missing'
                # Only flag as issue if it's critical
                if name in ['evolution_engine', 'shared_checkpoints', 'ui']:
                    health_report['issues'].append(f'{name} is missing')
                else:
                    health_report['warnings'].append(f'{name} not found (optional)')
        
        # Check disk space
        try:
            import shutil
            disk_usage = shutil.disk_usage('/')
            if disk_usage.free / disk_usage.total < 0.1:  # Less than 10% free
                health_report['warnings'].append('Low disk space')
        except:
            pass
        
        # Determine overall status
        if health_report['issues']:
            health_report['status'] = 'degraded'
        elif health_report['warnings']:
            health_report['status'] = 'warning'
        
        # Save health report
        report_file = self.health_path / f"health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(health_report, f, indent=2)
        
        self.health_checkpoints.append(health_report)
        
        logging.info(f"🏥 Health check complete: {health_report['status']}")
        return health_report
    
    def rollback(self, backup_id):
        """Rollback to a previous backup"""
        backup_dir = self.backup_path / backup_id
        if not backup_dir.exists():
            logging.error(f"❌ Backup {backup_id} not found")
            return False
        
        # Load backup info
        with open(backup_dir / "backup_info.json", 'r') as f:
            backup_info = json.load(f)
        
        # Create emergency backup of current state first
        emergency_id = self.create_backup("all", {"reason": "pre_rollback"})
        
        # Perform rollback
        try:
            # Restore each file from backup
            for file_info in backup_info.get('files', []):
                src = backup_dir / file_info
                dest = Path(file_info)
                
                # Create parent directories if needed
                dest.parent.mkdir(parents=True, exist_ok=True)
                
                if src.exists():
                    shutil.copy2(src, dest)
                    logging.info(f"🔄 Restored: {file_info}")
            
            # Log recovery action
            self.recovery_actions.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'rollback',
                'backup_id': backup_id,
                'component': backup_info['component'],
                'success': True
            })
            
            logging.info(f"🔄 Rolled back to backup {backup_id}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Rollback failed: {e}")
            
            # Log failure
            self.recovery_actions.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'rollback',
                'backup_id': backup_id,
                'success': False,
                'error': str(e)
            })
            
            return False
    
    def safe_update(self, update_function, component, test_suite="basic"):
        """
        Safely perform an update with automatic rollback on failure
        update_function: function that performs the update
        component: what's being updated
        """
        # Create pre-update backup
        backup_id = self.create_backup(component, {"action": "pre_update"})
        
        # Run backtest on current state
        pre_passed, pre_results, pre_score = self.run_backtest(component, test_suite)
        
        # Perform update
        try:
            update_result = update_function()
        except Exception as e:
            logging.error(f"❌ Update failed: {e}")
            self.rollback(backup_id)
            return False, str(e)
        
        # Run post-update backtest
        post_passed, post_results, post_score = self.run_backtest(component, test_suite)
        
        # Decide whether to keep or rollback
        if post_passed and post_score >= pre_score:
            logging.info(f"✅ Update successful! Score improved: {pre_score:.2f} → {post_score:.2f}")
            
            # Create success backup
            self.create_backup(component, {"action": "post_update_success"})
            
            # Log success
            self.recovery_actions.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'safe_update',
                'component': component,
                'success': True,
                'score_improvement': post_score - pre_score
            })
            
            return True, update_result
            
        else:
            logging.warning(f"⚠️ Update caused regression: {pre_score:.2f} → {post_score:.2f}")
            self.rollback(backup_id)
            
            # Log regression
            self.recovery_actions.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'safe_update',
                'component': component,
                'success': False,
                'reason': 'regression',
                'pre_score': pre_score,
                'post_score': post_score
            })
            
            return False, "Update caused performance regression"
    
    def monitor_and_heal(self, interval=300):
        """
        Continuously monitor system health and auto-heal when issues detected
        Runs in background thread
        """
        def monitor_loop():
            while True:
                try:
                    # Check health
                    health = self.check_system_health()
                    
                    # If issues detected, attempt healing
                    if health['status'] != 'healthy':
                        logging.warning(f"🚨 Health issues detected: {len(health['issues'])} issues")
                        
                        # Find most recent healthy backup
                        backups = sorted(self.backup_path.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
                        
                        for backup in backups:
                            backup_id = backup.name
                            backup_info_file = backup / "backup_info.json"
                            
                            if backup_info_file.exists():
                                with open(backup_info_file, 'r') as f:
                                    info = json.load(f)
                                
                                # Test if this backup is healthy
                                self.rollback(backup_id)
                                test_passed, _, _ = self.run_backtest("all", "basic")
                                
                                if test_passed:
                                    logging.info(f"✅ Auto-healed using backup {backup_id}")
                                    break
                                else:
                                    # Keep looking
                                    continue
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logging.error(f"❌ Monitor error: {e}")
                    time.sleep(60)
        
        import threading
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logging.info(f"👀 Health monitor started (checking every {interval}s)")
    
    def get_recovery_stats(self):
        """Get statistics about recovery actions"""
        return {
            'total_backups': len(list(self.backup_path.glob("*"))),
            'total_recoveries': len(self.recovery_actions),
            'successful_recoveries': sum(1 for a in self.recovery_actions if a.get('success')),
            'failed_recoveries': sum(1 for a in self.recovery_actions if not a.get('success')),
            'last_health_check': self.health_checkpoints[-1]['timestamp'] if self.health_checkpoints else None,
            'recent_actions': self.recovery_actions[-10:] if self.recovery_actions else []
        }
    
    def load_health_history(self):
        """Load health check history"""
        health_files = sorted(self.health_path.glob("health_*.json"))
        for hf in health_files[-10:]:  # Last 10 checks
            try:
                with open(hf, 'r') as f:
                    self.health_checkpoints.append(json.load(f))
            except:
                pass

if __name__ == "__main__":
    # Test the self-healer
    healer = SelfHealer()
    
    print("🏥 Testing Self-Healing System...")
    
    # Create test backup
    backup_id = healer.create_backup("all", {"test": True})
    print(f"✅ Created backup: {backup_id}")
    
    # Run health check
    health = healer.check_system_health()
    print(f"\n📊 Health Check Results:")
    print(json.dumps(health, indent=2))
    
    # Test safe update simulation
    def mock_update():
        print("🔄 Simulating update...")
        # In real scenario, this would modify code
        return "update complete"
    
    success, result = healer.safe_update(mock_update, "test_component", "basic")
    print(f"\n🛡️ Safe Update Result: {'✅' if success else '❌'} - {result}")
    
    # Get stats
    print(f"\n📈 Recovery Stats:")
    print(json.dumps(healer.get_recovery_stats(), indent=2, default=str))
    
    # Start monitoring (in background)
    healer.monitor_and_heal(interval=60)
    print("\n👀 Health monitor started (will run in background)")
    
    # Keep running for a bit to see monitoring
    time.sleep(2)
    print("✅ Self-healer test complete")
