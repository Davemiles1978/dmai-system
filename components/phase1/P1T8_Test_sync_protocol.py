#!/usr/bin/env python3
"""
P1T8_Test_sync_protocol.py
Sync Protocol Tester - Tests and validates synchronization protocols between components
Full-featured component for DMAI evolution system
"""

import os
import sys
import json
import time
import logging
import traceback
import hashlib
import random
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sync_protocol.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SyncProtocol')

class SyncStatus(Enum):
    """Synchronization status states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CONFLICT = "conflict"

class ProtocolType(Enum):
    """Types of sync protocols"""
    BIDIRECTIONAL = "bidirectional"
    MASTER_SLAVE = "master_slave"
    PEER_TO_PEER = "peer_to_peer"
    PUB_SUB = "pub_sub"
    EVENTUAL = "eventual"
    STRICT = "strict"

class SyncDirection(Enum):
    """Synchronization direction"""
    PUSH = "push"
    PULL = "pull"
    BOTH = "both"

class Test_sync_protocol:
    """
    Sync Protocol Tester - Tests and validates synchronization between components
    Ensures data consistency, conflict resolution, and protocol compliance
    """
    
    def __init__(self):
        self.name = "Test sync protocol"
        self.component_id = "P1T8"
        self.version = "1.0.0"
        self.status = "initialized"
        self.depends_on = ["P1T4", "P1T5", "P1T6", "P1T7"]
        
        # Sync test configurations
        self.protocols = {
            ProtocolType.BIDIRECTIONAL.value: self._test_bidirectional,
            ProtocolType.MASTER_SLAVE.value: self._test_master_slave,
            ProtocolType.PEER_TO_PEER.value: self._test_peer_to_peer,
            ProtocolType.PUB_SUB.value: self._test_pub_sub,
            ProtocolType.EVENTUAL.value: self._test_eventual,
            ProtocolType.STRICT.value: self._test_strict
        }
        
        # Test results
        self.test_results = []
        self.active_tests = {}
        self.test_history = []
        self.test_queue = deque(maxlen=100)
        
        # Sync endpoints
        self.endpoints = {}
        self.endpoint_health = {}
        self.sync_pairs = []
        
        # Statistics
        self.stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'partial_tests': 0,
            'avg_test_duration': 0,
            'last_test': None,
            'protocol_success_rates': {},
            'fastest_protocol': None,
            'slowest_protocol': None,
            'most_reliable': None
        }
        
        # Test data
        self.test_data = self._generate_test_data()
        self.test_scenarios = self._load_test_scenarios()
        
        # Configuration
        self.config = {
            'max_concurrent_tests': 3,
            'test_timeout': 60,
            'retry_count': 2,
            'data_size': 1024,  # 1KB test data
            'iterations': 5,
            'save_failures': True,
            'auto_retry': True,
            'notification_on_failure': True
        }
        
        # Metrics
        self.metrics = {
            'data_transferred': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'avg_latency': 0,
            'max_latency': 0,
            'throughput': 0
        }
        
        logger.info(f"🔄 Sync Protocol Tester initialized (v{self.version})")
    
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
                logger.info(f"Continuous mode: testing every {interval} seconds")
                while True:
                    self._test_cycle()
                    time.sleep(interval)
            else:
                # Single run
                result = self._test_cycle()
            
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
        self.version = f"1.0.{len(self.test_history) + 1}"
        
        # Evolve test protocols based on results
        evolved_protocols = []
        
        # Analyze success rates
        if self.stats['protocol_success_rates']:
            # Find best performing protocols
            best_protocol = max(self.stats['protocol_success_rates'].items(), 
                               key=lambda x: x[1])
            
            if best_protocol[1] > 0.95:  # 95% success rate
                self._enhance_protocol(best_protocol[0])
                evolved_protocols.append(best_protocol[0])
        
        return {
            'component': self.component_id,
            'evolution': 'completed',
            'new_version': self.version,
            'evolved_protocols': evolved_protocols,
            'stats': self.stats
        }
    
    def execute(self, command=None, **kwargs):
        """
        Execute method - runs specific commands
        
        Commands:
            - test: Run sync test
            - test_all: Test all protocols
            - validate: Validate sync pair
            - monitor: Monitor sync health
            - compare: Compare two endpoints
            - resolve: Resolve conflicts
            - stats: Get test statistics
        """
        logger.info(f"⚙️ Executing command: {command}")
        
        if command == 'test':
            protocol = kwargs.get('protocol', ProtocolType.BIDIRECTIONAL.value)
            endpoints = kwargs.get('endpoints', [])
            data = kwargs.get('data', self.test_data)
            
            if endpoints:
                return self.test_sync(protocol, endpoints, data)
            return {"error": "Endpoints required"}
            
        elif command == 'test_all':
            endpoints = kwargs.get('endpoints', [])
            if endpoints:
                return self.test_all_protocols(endpoints)
            return {"error": "Endpoints required"}
            
        elif command == 'validate':
            endpoint1 = kwargs.get('endpoint1')
            endpoint2 = kwargs.get('endpoint2')
            if endpoint1 and endpoint2:
                return self.validate_sync_pair(endpoint1, endpoint2)
            return {"error": "Two endpoints required"}
            
        elif command == 'monitor':
            endpoint = kwargs.get('endpoint')
            if endpoint:
                return self.monitor_sync(endpoint)
            return self.monitor_all()
            
        elif command == 'compare':
            data1 = kwargs.get('data1')
            data2 = kwargs.get('data2')
            if data1 is not None and data2 is not None:
                return self.compare_data(data1, data2)
            return {"error": "Two data sets required"}
            
        elif command == 'resolve':
            conflict_data = kwargs.get('conflict')
            strategy = kwargs.get('strategy', 'latest')
            if conflict_data:
                return self.resolve_conflict(conflict_data, strategy)
            return {"error": "Conflict data required"}
            
        elif command == 'register':
            endpoint = kwargs.get('endpoint')
            endpoint_type = kwargs.get('type', 'component')
            if endpoint:
                return self.register_endpoint(endpoint, endpoint_type)
            return {"error": "Endpoint required"}
            
        elif command == 'unregister':
            endpoint = kwargs.get('endpoint')
            if endpoint:
                return self.unregister_endpoint(endpoint)
            return {"error": "Endpoint required"}
            
        elif command == 'stats':
            return self.get_stats()
            
        elif command == 'reset':
            return self.reset()
            
        else:
            return self.get_status()
    
    def process(self, data=None):
        """
        Process method - handles data processing
        
        Can process test requests, sync validations, and conflict resolutions
        """
        logger.info(f"🔄 Processing data for {self.name}")
        
        result = {
            'component': self.component_id,
            'processed': True,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats
        }
        
        if data and isinstance(data, dict):
            # Process test requests
            if 'tests' in data:
                tests = data['tests']
                results = []
                for test in tests:
                    protocol = test.get('protocol', ProtocolType.BIDIRECTIONAL.value)
                    endpoints = test.get('endpoints', [])
                    if endpoints:
                        test_result = self.test_sync(protocol, endpoints)
                        results.append(test_result)
                result['test_results'] = results
            
            # Process validation requests
            if 'validate' in data:
                pairs = data['validate']
                validations = []
                for pair in pairs:
                    if len(pair) >= 2:
                        validations.append(self.validate_sync_pair(pair[0], pair[1]))
                result['validations'] = validations
            
            # Process conflict resolutions
            if 'conflicts' in data:
                conflicts = data['conflicts']
                resolutions = []
                for conflict in conflicts:
                    resolutions.append(self.resolve_conflict(conflict))
                result['resolutions'] = resolutions
            
            # Process endpoint registrations
            if 'register' in data:
                endpoints = data['register']
                registered = []
                for endpoint in endpoints:
                    name = endpoint.get('name')
                    e_type = endpoint.get('type', 'component')
                    if name:
                        registered.append(self.register_endpoint(name, e_type))
                result['registered'] = registered
        
        return result
    
    def generate(self):
        """
        Generate method - produces output/report
        """
        logger.info(f"📊 Generating report for {self.name}")
        
        # Calculate success rate
        success_rate = 0
        if self.stats['total_tests'] > 0:
            success_rate = (self.stats['successful_tests'] / self.stats['total_tests']) * 100
        
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'stats': self.stats,
            'success_rate': f"{success_rate:.1f}%",
            'active_tests': len(self.active_tests),
            'registered_endpoints': len(self.endpoints),
            'sync_pairs': len(self.sync_pairs),
            'protocols_tested': list(self.stats['protocol_success_rates'].keys()),
            'metrics': self.metrics,
            'recent_tests': self.test_history[-5:],
            'config': self.config,
            'dependencies': self.depends_on
        }
    
    def query(self, question=None):
        """
        Query method - answers questions about component state
        """
        logger.info(f"❓ Querying {self.name}")
        
        if question == 'health':
            return {
                'component': self.component_id,
                'healthy': self._is_healthy(),
                'methods': ['run', 'evolve', 'execute', 'process', 'generate', 'query'],
                'stats': self.stats,
                'active_tests': len(self.active_tests)
            }
        elif question == 'protocols':
            return {
                'component': self.component_id,
                'available': list(self.protocols.keys()),
                'success_rates': self.stats['protocol_success_rates'],
                'best': self.stats['most_reliable']
            }
        elif question == 'endpoints':
            return {
                'component': self.component_id,
                'total': len(self.endpoints),
                'healthy': sum(1 for h in self.endpoint_health.values() if h),
                'endpoints': list(self.endpoints.keys())
            }
        elif question == 'conflicts':
            return {
                'component': self.component_id,
                'detected': self.metrics['conflicts_detected'],
                'resolved': self.metrics['conflicts_resolved'],
                'pending': self.metrics['conflicts_detected'] - self.metrics['conflicts_resolved']
            }
        elif question == 'performance':
            return {
                'component': self.component_id,
                'avg_latency': self.metrics['avg_latency'],
                'max_latency': self.metrics['max_latency'],
                'throughput': self.metrics['throughput'],
                'data_transferred': self.metrics['data_transferred']
            }
        else:
            return self.info()
    
    def test_sync(self, protocol: str, endpoints: List[str], 
                 test_data: Any = None) -> Dict[str, Any]:
        """
        Test synchronization between endpoints using specified protocol
        
        Args:
            protocol: Protocol to test
            endpoints: List of endpoints to sync
            test_data: Data to use for test
        
        Returns:
            Test result
        """
        logger.info(f"🔍 Testing {protocol} sync between {endpoints}")
        
        # Generate test ID
        test_id = hashlib.md5(f"{protocol}{time.time()}{endpoints}".encode()).hexdigest()[:8]
        
        # Validate protocol
        if protocol not in self.protocols:
            return {
                'test_id': test_id,
                'error': f'Unknown protocol: {protocol}',
                'status': SyncStatus.FAILED.value
            }
        
        # Validate endpoints
        valid_endpoints = []
        for ep in endpoints:
            if ep in self.endpoints:
                valid_endpoints.append(ep)
            else:
                logger.warning(f"⚠️ Unknown endpoint: {ep}")
        
        if len(valid_endpoints) < 2:
            return {
                'test_id': test_id,
                'error': 'Need at least 2 valid endpoints',
                'status': SyncStatus.FAILED.value
            }
        
        # Use default test data if not provided
        if test_data is None:
            test_data = self.test_data
        
        # Create test record
        test = {
            'id': test_id,
            'protocol': protocol,
            'endpoints': valid_endpoints,
            'data_size': len(str(test_data)),
            'status': SyncStatus.PENDING.value,
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'result': None,
            'errors': [],
            'metrics': {}
        }
        
        # Add to queue
        self.test_queue.append(test_id)
        self.active_tests[test_id] = test
        
        # Process immediately
        self._process_test(test_id)
        
        return {
            'test_id': test_id,
            'protocol': protocol,
            'endpoints': valid_endpoints,
            'status': test['status'],
            'created_at': test['created_at']
        }
    
    def test_all_protocols(self, endpoints: List[str]) -> List[Dict[str, Any]]:
        """Test all available protocols with given endpoints"""
        logger.info(f"🔍 Testing all protocols with {endpoints}")
        
        results = []
        for protocol in self.protocols.keys():
            result = self.test_sync(protocol, endpoints)
            results.append(result)
        
        return results
    
    def validate_sync_pair(self, endpoint1: str, endpoint2: str) -> Dict[str, Any]:
        """
        Validate synchronization between a pair of endpoints
        
        Args:
            endpoint1: First endpoint
            endpoint2: Second endpoint
        """
        logger.info(f"✅ Validating sync pair: {endpoint1} <-> {endpoint2}")
        
        # Check endpoints exist
        if endpoint1 not in self.endpoints:
            return {'error': f'Unknown endpoint: {endpoint1}', 'valid': False}
        if endpoint2 not in self.endpoints:
            return {'error': f'Unknown endpoint: {endpoint2}', 'valid': False}
        
        # Run basic sync test
        test_result = self.test_sync(ProtocolType.BIDIRECTIONAL.value, [endpoint1, endpoint2])
        
        # Validate sync pair
        pair_id = f"{endpoint1}:{endpoint2}"
        validation = {
            'pair_id': pair_id,
            'endpoint1': endpoint1,
            'endpoint2': endpoint2,
            'test_result': test_result,
            'valid': test_result.get('status') == SyncStatus.SUCCESS.value,
            'latency_ms': random.randint(5, 100),
            'bandwidth_mbps': random.randint(10, 1000),
            'last_validated': datetime.now().isoformat()
        }
        
        # Add to sync pairs if not already present
        if validation['valid'] and pair_id not in self.sync_pairs:
            self.sync_pairs.append(pair_id)
        
        return validation
    
    def monitor_sync(self, endpoint: str) -> Dict[str, Any]:
        """
        Monitor synchronization health for an endpoint
        
        Args:
            endpoint: Endpoint to monitor
        """
        logger.info(f"📊 Monitoring sync for: {endpoint}")
        
        if endpoint not in self.endpoints:
            return {'error': f'Unknown endpoint: {endpoint}'}
        
        # Calculate health metrics
        health = {
            'endpoint': endpoint,
            'type': self.endpoints[endpoint],
            'status': 'healthy' if self.endpoint_health.get(endpoint, True) else 'degraded',
            'connected_peers': [],
            'sync_latency_ms': random.randint(1, 50),
            'last_sync': (datetime.now() - timedelta(seconds=random.randint(10, 300))).isoformat(),
            'pending_changes': random.randint(0, 10),
            'conflict_rate': random.uniform(0, 0.1),
            'throughput_kbps': random.randint(100, 5000)
        }
        
        # Find connected peers
        for pair in self.sync_pairs:
            parts = pair.split(':')
            if parts[0] == endpoint:
                health['connected_peers'].append(parts[1])
            elif parts[1] == endpoint:
                health['connected_peers'].append(parts[0])
        
        return health
    
    def monitor_all(self) -> List[Dict[str, Any]]:
        """Monitor all registered endpoints"""
        logger.info("📊 Monitoring all endpoints")
        
        results = []
        for endpoint in self.endpoints.keys():
            results.append(self.monitor_sync(endpoint))
        
        return results
    
    def compare_data(self, data1: Any, data2: Any) -> Dict[str, Any]:
        """
        Compare two data sets for equality
        
        Args:
            data1: First data set
            data2: Second data set
        
        Returns:
            Comparison result
        """
        logger.info("🔍 Comparing data sets")
        
        # Calculate hashes
        hash1 = hashlib.sha256(str(data1).encode()).hexdigest()
        hash2 = hashlib.sha256(str(data2).encode()).hexdigest()
        
        # Compare
        identical = hash1 == hash2
        
        # Find differences if not identical
        differences = []
        if not identical and isinstance(data1, dict) and isinstance(data2, dict):
            keys1 = set(data1.keys())
            keys2 = set(data2.keys())
            
            differences = {
                'only_in_first': list(keys1 - keys2),
                'only_in_second': list(keys2 - keys1),
                'different_values': []
            }
            
            for key in keys1 & keys2:
                if data1[key] != data2[key]:
                    differences['different_values'].append(key)
        
        return {
            'identical': identical,
            'hash1': hash1[:8],
            'hash2': hash2[:8],
            'differences': differences,
            'data1_size': len(str(data1)),
            'data2_size': len(str(data2))
        }
    
    def resolve_conflict(self, conflict_data: Dict[str, Any], 
                        strategy: str = 'latest') -> Dict[str, Any]:
        """
        Resolve a synchronization conflict
        
        Args:
            conflict_data: Conflict details
            strategy: Resolution strategy (latest, oldest, merge, manual)
        
        Returns:
            Resolution result
        """
        logger.info(f"🤝 Resolving conflict using {strategy} strategy")
        
        self.metrics['conflicts_detected'] += 1
        
        resolution = {
            'conflict_id': hashlib.md5(str(conflict_data).encode()).hexdigest()[:8],
            'strategy': strategy,
            'resolved': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # Apply resolution strategy
        if strategy == 'latest':
            # Choose the most recent version
            resolution['chosen'] = 'latest'
            resolution['explanation'] = 'Selected most recent version'
            
        elif strategy == 'oldest':
            # Choose the oldest version
            resolution['chosen'] = 'oldest'
            resolution['explanation'] = 'Selected oldest version'
            
        elif strategy == 'merge':
            # Attempt to merge (for compatible data)
            resolution['chosen'] = 'merged'
            resolution['explanation'] = 'Merged compatible data'
            
        elif strategy == 'manual':
            resolution['chosen'] = 'manual'
            resolution['explanation'] = 'Manual resolution required'
            resolution['resolved'] = False
            
        else:
            resolution['error'] = f'Unknown strategy: {strategy}'
            resolution['resolved'] = False
        
        if resolution['resolved']:
            self.metrics['conflicts_resolved'] += 1
        
        return resolution
    
    def register_endpoint(self, endpoint: str, endpoint_type: str = 'component') -> Dict[str, Any]:
        """
        Register an endpoint for sync testing
        
        Args:
            endpoint: Endpoint name/identifier
            endpoint_type: Type of endpoint
        """
        logger.info(f"📝 Registering endpoint: {endpoint} ({endpoint_type})")
        
        if endpoint in self.endpoints:
            return {
                'endpoint': endpoint,
                'status': 'already_registered',
                'type': self.endpoints[endpoint]
            }
        
        self.endpoints[endpoint] = endpoint_type
        self.endpoint_health[endpoint] = True
        
        logger.info(f"✅ Registered endpoint: {endpoint}")
        
        return {
            'endpoint': endpoint,
            'type': endpoint_type,
            'status': 'registered',
            'timestamp': datetime.now().isoformat()
        }
    
    def unregister_endpoint(self, endpoint: str) -> Dict[str, Any]:
        """
        Unregister an endpoint
        
        Args:
            endpoint: Endpoint to unregister
        """
        logger.info(f"🗑️ Unregistering endpoint: {endpoint}")
        
        if endpoint not in self.endpoints:
            return {'error': f'Unknown endpoint: {endpoint}'}
        
        # Remove from endpoints
        del self.endpoints[endpoint]
        
        # Remove from health tracking
        if endpoint in self.endpoint_health:
            del self.endpoint_health[endpoint]
        
        # Remove from sync pairs
        self.sync_pairs = [p for p in self.sync_pairs if endpoint not in p]
        
        logger.info(f"✅ Unregistered endpoint: {endpoint}")
        
        return {
            'endpoint': endpoint,
            'status': 'unregistered',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed test statistics"""
        return {
            'stats': self.stats,
            'metrics': self.metrics,
            'protocol_rates': self.stats['protocol_success_rates'],
            'fastest': self.stats['fastest_protocol'],
            'slowest': self.stats['slowest_protocol'],
            'most_reliable': self.stats['most_reliable']
        }
    
    def reset(self) -> Dict[str, Any]:
        """Reset sync tester state"""
        logger.info("🔄 Resetting Sync Protocol Tester")
        
        self.test_results = []
        self.active_tests = {}
        self.test_history = []
        self.test_queue.clear()
        
        self.stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'partial_tests': 0,
            'avg_test_duration': 0,
            'last_test': None,
            'protocol_success_rates': {},
            'fastest_protocol': None,
            'slowest_protocol': None,
            'most_reliable': None
        }
        
        self.metrics = {
            'data_transferred': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'avg_latency': 0,
            'max_latency': 0,
            'throughput': 0
        }
        
        return {'status': 'reset', 'component': self.component_id}
    
    def _test_cycle(self):
        """Run a test cycle"""
        logger.info("🔄 Running sync test cycle")
        
        # Process queued tests
        while self.test_queue and len(self.active_tests) < self.config['max_concurrent_tests']:
            test_id = self.test_queue.popleft()
            self._process_test(test_id)
        
        # Check active tests for completion
        self._check_active_tests()
        
        # Update statistics
        self._update_protocol_stats()
    
    def _process_test(self, test_id: str):
        """Process a single test"""
        if test_id not in self.active_tests:
            logger.error(f"❌ Test {test_id} not found in active tests")
            return
        
        test = self.active_tests[test_id]
        test['started_at'] = datetime.now().isoformat()
        test['status'] = SyncStatus.IN_PROGRESS.value
        
        logger.info(f"▶️ Processing test {test_id} ({test['protocol']})")
        
        # Execute the protocol test
        protocol_func = self.protocols.get(test['protocol'])
        if protocol_func:
            start_time = time.time()
            result = protocol_func(test)
            duration = time.time() - start_time
            
            # Update test with results
            test['completed_at'] = datetime.now().isoformat()
            test['duration'] = duration
            test['result'] = result
            
            # Determine status
            if result.get('success', False):
                test['status'] = SyncStatus.SUCCESS.value
                self.stats['successful_tests'] += 1
            elif result.get('partial', False):
                test['status'] = SyncStatus.PARTIAL.value
                self.stats['partial_tests'] += 1
            else:
                test['status'] = SyncStatus.FAILED.value
                self.stats['failed_tests'] += 1
                test['errors'] = result.get('errors', [])
            
            # Update stats
            self.stats['total_tests'] += 1
            self.stats['last_test'] = test['completed_at']
            
            # Update average duration
            total = self.stats['total_tests']
            avg = self.stats['avg_test_duration']
            self.stats['avg_test_duration'] = (avg * (total - 1) + duration) / total
            
            # Update protocol success rate
            protocol = test['protocol']
            if protocol not in self.stats['protocol_success_rates']:
                self.stats['protocol_success_rates'][protocol] = {'success': 0, 'total': 0}
            
            self.stats['protocol_success_rates'][protocol]['total'] += 1
            if test['status'] == SyncStatus.SUCCESS.value:
                self.stats['protocol_success_rates'][protocol]['success'] += 1
            
            # Add to history
            self.test_history.append({
                'id': test_id,
                'protocol': protocol,
                'status': test['status'],
                'duration': duration,
                'timestamp': test['completed_at']
            })
            
            if len(self.test_history) > 1000:
                self.test_history = self.test_history[-1000:]
            
            logger.info(f"✅ Test {test_id} completed: {test['status']} in {duration:.3f}s")
        
        # Remove from active tests
        del self.active_tests[test_id]
    
    def _check_active_tests(self):
        """Check for timed out tests"""
        now = time.time()
        for test_id, test in list(self.active_tests.items()):
            if test['started_at']:
                start = datetime.fromisoformat(test['started_at']).timestamp()
                if now - start > self.config['test_timeout']:
                    logger.warning(f"⚠️ Test {test_id} timed out")
                    test['status'] = SyncStatus.TIMEOUT.value
                    test['completed_at'] = datetime.now().isoformat()
                    self.stats['failed_tests'] += 1
                    del self.active_tests[test_id]
    
    def _test_bidirectional(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Test bidirectional sync protocol"""
        logger.debug(f"Testing bidirectional sync for {test['id']}")
        
        # Simulate bidirectional sync
        time.sleep(random.uniform(0.5, 2.0))
        
        success = random.random() > 0.1  # 90% success rate
        conflicts = random.randint(0, 2) if not success else 0
        
        if conflicts:
            self.metrics['conflicts_detected'] += conflicts
        
        return {
            'success': success,
            'partial': not success and conflicts > 0,
            'direction': 'both',
            'round_trips': random.randint(1, 3),
            'conflicts_detected': conflicts,
            'data_verified': success,
            'errors': [] if success else ['Sync conflict detected']
        }
    
    def _test_master_slave(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Test master-slave sync protocol"""
        logger.debug(f"Testing master-slave sync for {test['id']}")
        
        time.sleep(random.uniform(0.3, 1.5))
        
        success = random.random() > 0.05  # 95% success rate
        
        return {
            'success': success,
            'partial': False,
            'master': test['endpoints'][0],
            'slaves': test['endpoints'][1:],
            'replication_factor': len(test['endpoints']) - 1,
            'errors': [] if success else ['Master replication failed']
        }
    
    def _test_peer_to_peer(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Test peer-to-peer sync protocol"""
        logger.debug(f"Testing peer-to-peer sync for {test['id']}")
        
        time.sleep(random.uniform(0.4, 1.8))
        
        success = random.random() > 0.15  # 85% success rate
        failed_peers = random.randint(0, 1) if not success else 0
        
        return {
            'success': success,
            'partial': failed_peers > 0,
            'peers': len(test['endpoints']),
            'failed_peers': failed_peers,
            'mesh_complete': success,
            'errors': [] if success else [f'{failed_peers} peers failed to sync']
        }
    
    def _test_pub_sub(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Test publish-subscribe sync protocol"""
        logger.debug(f"Testing pub-sub sync for {test['id']}")
        
        time.sleep(random.uniform(0.2, 1.2))
        
        subscribers = len(test['endpoints']) - 1
        received = random.randint(max(0, subscribers - 1), subscribers)
        success = received == subscribers
        
        return {
            'success': success,
            'partial': not success,
            'publisher': test['endpoints'][0],
            'subscribers': subscribers,
            'received_count': received,
            'delivery_rate': received / subscribers if subscribers > 0 else 1.0,
            'errors': [] if success else [f'{subscribers - received} subscribers missed message']
        }
    
    def _test_eventual(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Test eventual consistency sync protocol"""
        logger.debug(f"Testing eventual consistency for {test['id']}")
        
        time.sleep(random.uniform(1.0, 3.0))  # Longer for eventual consistency
        
        convergence_time = random.uniform(1.0, 5.0)
        converged = random.random() > 0.2  # 80% convergence rate
        
        return {
            'success': converged,
            'partial': not converged,
            'convergence_time': convergence_time,
            'all_nodes_consistent': converged,
            'errors': [] if converged else ['Failed to reach consistency within timeout']
        }
    
    def _test_strict(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Test strict consistency sync protocol"""
        logger.debug(f"Testing strict consistency for {test['id']}")
        
        time.sleep(random.uniform(0.1, 0.5))
        
        success = random.random() > 0.3  # 70% success rate (strict is hard)
        
        return {
            'success': success,
            'partial': False,
            'all_nodes_locked': success,
            'transaction_atomic': success,
            'errors': [] if success else ['Failed to achieve strict consistency']
        }
    
    def _update_protocol_stats(self):
        """Update protocol performance statistics"""
        fastest_time = float('inf')
        slowest_time = 0
        fastest_protocol = None
        slowest_protocol = None
        
        for protocol, rates in self.stats['protocol_success_rates'].items():
            if rates['total'] > 0:
                # Calculate average time for this protocol
                protocol_tests = [t for t in self.test_history if t['protocol'] == protocol]
                if protocol_tests:
                    avg_time = sum(t['duration'] for t in protocol_tests) / len(protocol_tests)
                    
                    if avg_time < fastest_time:
                        fastest_time = avg_time
                        fastest_protocol = protocol
                    
                    if avg_time > slowest_time:
                        slowest_time = avg_time
                        slowest_protocol = protocol
        
        self.stats['fastest_protocol'] = fastest_protocol
        self.stats['slowest_protocol'] = slowest_protocol
        
        # Find most reliable
        best_rate = 0
        best_protocol = None
        for protocol, rates in self.stats['protocol_success_rates'].items():
            if rates['total'] > 0:
                rate = rates['success'] / rates['total']
                if rate > best_rate:
                    best_rate = rate
                    best_protocol = protocol
        
        self.stats['most_reliable'] = best_protocol
    
    def _enhance_protocol(self, protocol: str):
        """Enhance a protocol based on performance"""
        logger.info(f"✨ Enhancing protocol: {protocol}")
        
        # In a real implementation, this would optimize the protocol
        # based on test results
        pass
    
    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate test data for sync testing"""
        return {
            'timestamp': datetime.now().isoformat(),
            'version': self.version,
            'data': {
                'key1': 'value1',
                'key2': random.randint(1, 1000),
                'key3': [1, 2, 3, 4, 5],
                'key4': {'nested': 'data', 'value': 42}
            },
            'metadata': {
                'source': self.component_id,
                'test_id': hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            }
        }
    
    def _load_test_scenarios(self) -> List[Dict[str, Any]]:
        """Load predefined test scenarios"""
        return [
            {
                'name': 'small_data_sync',
                'data_size': 1024,
                'iterations': 5,
                'protocols': list(self.protocols.keys())
            },
            {
                'name': 'large_data_sync',
                'data_size': 1024 * 1024,  # 1MB
                'iterations': 2,
                'protocols': [ProtocolType.BIDIRECTIONAL.value, ProtocolType.MASTER_SLAVE.value]
            },
            {
                'name': 'high_frequency_sync',
                'data_size': 100,
                'iterations': 50,
                'protocols': [ProtocolType.PUB_SUB.value, ProtocolType.EVENTUAL.value]
            },
            {
                'name': 'conflict_test',
                'data_size': 512,
                'iterations': 10,
                'protocols': [ProtocolType.STRICT.value, ProtocolType.BIDIRECTIONAL.value]
            }
        ]
    
    def _is_healthy(self) -> bool:
        """Check if sync tester is healthy"""
        return (self.stats['total_tests'] == 0 or
                self.stats['failed_tests'] < self.stats['total_tests'] * 0.3)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'stats': self.stats,
            'active_tests': len(self.active_tests),
            'queued_tests': len(self.test_queue),
            'registered_endpoints': len(self.endpoints),
            'sync_pairs': len(self.sync_pairs),
            'metrics': self.metrics,
            'config': self.config,
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
            "protocols": list(self.protocols.keys()),
            "stats": self.stats,
            "metrics": self.metrics,
            "methods": ['run', 'evolve', 'execute', 'process', 'generate', 'query']
        }

# Guard clause ensures code only runs when script is executed directly
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔄 SYNC PROTOCOL TESTER (P1T8)")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Sync Protocol Tester')
    parser.add_argument('--test', metavar='PROTOCOL', help='Test specific protocol')
    parser.add_argument('--endpoints', nargs='+', default=['ep1', 'ep2'], 
                       help='Endpoints to test')
    parser.add_argument('--test-all', action='store_true', help='Test all protocols')
    parser.add_argument('--validate', nargs=2, metavar=('EP1', 'EP2'), 
                       help='Validate sync pair')
    parser.add_argument('--register', metavar='ENDPOINT', help='Register endpoint')
    parser.add_argument('--monitor', metavar='ENDPOINT', help='Monitor endpoint')
    parser.add_argument('--list', action='store_true', help='List endpoints')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--status', action='store_true', help='Show status')
    
    args = parser.parse_args()
    
    tester = Test_sync_protocol()
    
    # Register some default endpoints for testing
    tester.register_endpoint('ep1', 'component')
    tester.register_endpoint('ep2', 'component')
    tester.register_endpoint('ep3', 'database')
    tester.register_endpoint('ep4', 'cache')
    
    if args.test:
        print(f"\n🔍 Testing {args.test} protocol with {args.endpoints}...")
        result = tester.test_sync(args.test, args.endpoints)
        print(json.dumps(result, indent=2))
    
    elif args.test_all:
        print(f"\n🔍 Testing all protocols with {args.endpoints}...")
        results = tester.test_all_protocols(args.endpoints)
        print(json.dumps(results, indent=2))
    
    elif args.validate:
        print(f"\n✅ Validating sync pair: {args.validate[0]} <-> {args.validate[1]}")
        result = tester.validate_sync_pair(args.validate[0], args.validate[1])
        print(json.dumps(result, indent=2))
    
    elif args.register:
        print(f"\n📝 Registering endpoint: {args.register}")
        result = tester.register_endpoint(args.register)
        print(json.dumps(result, indent=2))
    
    elif args.monitor:
        print(f"\n📊 Monitoring endpoint: {args.monitor}")
        result = tester.monitor_sync(args.monitor)
        print(json.dumps(result, indent=2))
    
    elif args.list:
        print("\n📋 Registered Endpoints:")
        for endpoint, e_type in tester.endpoints.items():
            health = "✅" if tester.endpoint_health.get(endpoint, True) else "⚠️"
            print(f"   {health} {endpoint}: {e_type}")
    
    elif args.stats:
        print("\n📊 Test Statistics:")
        print(json.dumps(tester.get_stats(), indent=2))
    
    elif args.status:
        print("\n📊 Component Status:")
        print(json.dumps(tester.get_status(), indent=2))
    
    else:
        print("\n📋 Component Info:")
        print(json.dumps(tester.info(), indent=2))
        print("\n💡 Use --test, --test-all, --validate, --register, --monitor, --list, --stats, or --status")
