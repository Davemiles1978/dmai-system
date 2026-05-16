#!/usr/bin/env python3
"""
P1T5_Implement_validator.py
Validator Implementation - Validates components, data, and system state
Full-featured component for DMAI evolution system
"""

import os
import sys
import json
import time
import logging
import traceback
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Validator')

class ValidationLevel(Enum):
    """Validation strictness levels"""
    CRITICAL = 1
    STRICT = 2
    NORMAL = 3
    PERMISSIVE = 4
    DEBUG = 5

class ValidationStatus(Enum):
    """Validation result status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"

class ValidationType(Enum):
    """Types of validation"""
    COMPONENT = "component"
    DATA = "data"
    SCHEMA = "schema"
    CONFIG = "config"
    HEALTH = "health"
    SECURITY = "security"
    PERFORMANCE = "performance"
    INTEGRITY = "integrity"

class Validator:
    """
    Validator Implementation - Validates components, data, and system state
    Provides comprehensive validation framework for DMAI
    """
    
    def __init__(self):
        self.name = "Validator"
        self.component_id = "P1T5"
        self.version = "1.0.0"
        self.status = "initialized"
        self.depends_on = ["P1T4"]
        
        # Validation rules
        self.rules = self._load_default_rules()
        self.custom_rules = {}
        
        # Validation history
        self.validation_history = []
        self.validation_stats = {
            'total_validations': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'errors': 0,
            'avg_duration': 0,
            'last_validation': None
        }
        
        # Component-specific validators
        self.component_validators = {}
        self.data_validators = {}
        self.schema_validators = {}
        
        # Validation cache
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Results storage
        self.results = []
        self.failures = []
        self.warnings = []
        
        # Configuration
        self.config = {
            'default_level': ValidationLevel.NORMAL.value,
            'cache_enabled': True,
            'strict_mode': False,
            'fail_fast': False,
            'max_validations_per_cycle': 100,
            'timeout_seconds': 30
        }
        
        # Metrics
        self.metrics = {
            'fastest_validation': float('inf'),
            'slowest_validation': 0,
            'most_common_failure': None,
            'reliability_score': 1.0
        }
        
        logger.info(f"✅ Validator component initialized (v{self.version})")
    
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
                logger.info(f"Continuous mode: validating every {interval} seconds")
                while True:
                    self._validation_cycle()
                    time.sleep(interval)
            else:
                # Single run
                result = self._validation_cycle()
            
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
        self.version = f"1.0.{len(self.validation_history) + 1}"
        
        # Evolve validation rules based on history
        evolved_rules = []
        
        # Analyze failure patterns
        if self.failures:
            failure_types = {}
            for failure in self.failures[-100:]:  # Last 100 failures
                f_type = failure.get('type', 'unknown')
                failure_types[f_type] = failure_types.get(f_type, 0) + 1
            
            # Add new rules for common failures
            for f_type, count in failure_types.items():
                if count > 10 and f_type not in self.rules:
                    self._create_rule_from_failure(f_type)
                    evolved_rules.append(f_type)
        
        return {
            'component': self.component_id,
            'evolution': 'completed',
            'new_version': self.version,
            'evolved_rules': evolved_rules,
            'stats': self.validation_stats
        }
    
    def execute(self, command=None, **kwargs):
        """
        Execute method - runs specific commands
        
        Commands:
            - validate: Run validation
            - check: Quick health check
            - verify: Verify specific item
            - add_rule: Add validation rule
            - remove_rule: Remove validation rule
            - list_rules: List validation rules
            - clear_cache: Clear validation cache
            - stats: Get validation statistics
        """
        logger.info(f"⚙️ Executing command: {command}")
        
        if command == 'validate':
            target = kwargs.get('target')
            v_type = kwargs.get('type', ValidationType.COMPONENT.value)
            level = kwargs.get('level', self.config['default_level'])
            
            if target:
                return self.validate(target, v_type, level)
            return self.validate_all(level)
            
        elif command == 'check':
            target = kwargs.get('target')
            if target:
                return self.quick_check(target)
            return self.health_check()
            
        elif command == 'verify':
            target = kwargs.get('target')
            expected = kwargs.get('expected')
            if target and expected is not None:
                return self.verify(target, expected)
            return {"error": "Target and expected required"}
            
        elif command == 'add_rule':
            rule_name = kwargs.get('name')
            rule_config = kwargs.get('config')
            if rule_name and rule_config:
                return self.add_validation_rule(rule_name, rule_config)
            return {"error": "Rule name and config required"}
            
        elif command == 'remove_rule':
            rule_name = kwargs.get('name')
            if rule_name:
                return self.remove_validation_rule(rule_name)
            return {"error": "Rule name required"}
            
        elif command == 'list_rules':
            v_type = kwargs.get('type')
            return self.list_rules(v_type)
            
        elif command == 'clear_cache':
            return self.clear_cache()
            
        elif command == 'stats':
            return self.get_stats()
            
        elif command == 'reset':
            return self.reset()
            
        else:
            return self.get_status()
    
    def process(self, data=None):
        """
        Process method - handles data processing
        
        Can process validation requests, batch validations, and rule updates
        """
        logger.info(f"🔄 Processing data for {self.name}")
        
        result = {
            'component': self.component_id,
            'processed': True,
            'timestamp': datetime.now().isoformat(),
            'stats': self.validation_stats
        }
        
        if data and isinstance(data, dict):
            # Process validation requests
            if 'validations' in data:
                validations = data['validations']
                results = []
                for v in validations:
                    target = v.get('target')
                    v_type = v.get('type', ValidationType.COMPONENT.value)
                    level = v.get('level', self.config['default_level'])
                    if target:
                        v_result = self.validate(target, v_type, level)
                        results.append(v_result)
                result['validation_results'] = results
            
            # Process batch validation
            if 'batch' in data:
                batch = data['batch']
                targets = batch.get('targets', [])
                v_type = batch.get('type', ValidationType.COMPONENT.value)
                results = self.validate_batch(targets, v_type)
                result['batch_results'] = results
            
            # Process rule updates
            if 'rules' in data:
                rules = data['rules']
                for rule_name, rule_config in rules.items():
                    self.add_validation_rule(rule_name, rule_config)
                result['rules_updated'] = len(rules)
            
            # Process verification requests
            if 'verify' in data:
                verify_data = data['verify']
                targets = verify_data.get('targets', [])
                expectations = verify_data.get('expectations', {})
                results = {}
                for target in targets:
                    if target in expectations:
                        results[target] = self.verify(target, expectations[target])
                result['verifications'] = results
        
        return result
    
    def generate(self):
        """
        Generate method - produces output/report
        """
        logger.info(f"📊 Generating report for {self.name}")
        
        # Calculate success rate
        success_rate = 0
        if self.validation_stats['total_validations'] > 0:
            success_rate = (self.validation_stats['passed'] / self.validation_stats['total_validations']) * 100
        
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'stats': self.validation_stats,
            'success_rate': f"{success_rate:.1f}%",
            'metrics': self.metrics,
            'cache_size': len(self.cache),
            'rules_count': len(self.rules) + len(self.custom_rules),
            'recent_validations': self.validation_history[-10:],
            'recent_failures': self.failures[-5:],
            'recent_warnings': self.warnings[-5:],
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
                'stats': self.validation_stats,
                'status': self.status
            }
        elif question == 'rules':
            return {
                'component': self.component_id,
                'rules': list(self.rules.keys()),
                'custom_rules': list(self.custom_rules.keys()),
                'total_rules': len(self.rules) + len(self.custom_rules)
            }
        elif question == 'failures':
            return {
                'component': self.component_id,
                'total_failures': len(self.failures),
                'recent_failures': self.failures[-10:],
                'most_common': self.metrics['most_common_failure']
            }
        elif question == 'performance':
            return {
                'component': self.component_id,
                'fastest': self.metrics['fastest_validation'],
                'slowest': self.metrics['slowest_validation'],
                'average': self.validation_stats['avg_duration'],
                'reliability': self.metrics['reliability_score']
            }
        elif question == 'cache':
            return {
                'component': self.component_id,
                'enabled': self.config['cache_enabled'],
                'size': len(self.cache),
                'keys': list(self.cache.keys())[:10]
            }
        else:
            return self.info()
    
    def validate(self, target: Any, validation_type: str = ValidationType.COMPONENT.value, 
                level: int = None) -> Dict[str, Any]:
        """
        Validate a target
        
        Args:
            target: Target to validate
            validation_type: Type of validation
            level: Validation strictness level
        
        Returns:
            Validation result
        """
        start_time = time.time()
        
        # Use default level if not specified
        if level is None:
            level = self.config['default_level']
        
        # Check cache
        cache_key = self._get_cache_key(target, validation_type, level)
        if self.config['cache_enabled'] and cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                logger.debug(f"Using cached validation for {cache_key}")
                return cache_entry['result']
        
        logger.info(f"🔍 Validating {target} ({validation_type}, level={level})")
        
        # Perform validation based on type
        if validation_type == ValidationType.COMPONENT.value:
            result = self._validate_component(target, level)
        elif validation_type == ValidationType.DATA.value:
            result = self._validate_data(target, level)
        elif validation_type == ValidationType.SCHEMA.value:
            result = self._validate_schema(target, level)
        elif validation_type == ValidationType.CONFIG.value:
            result = self._validate_config(target, level)
        elif validation_type == ValidationType.HEALTH.value:
            result = self._validate_health(target, level)
        elif validation_type == ValidationType.SECURITY.value:
            result = self._validate_security(target, level)
        elif validation_type == ValidationType.PERFORMANCE.value:
            result = self._validate_performance(target, level)
        elif validation_type == ValidationType.INTEGRITY.value:
            result = self._validate_integrity(target, level)
        else:
            result = {
                'status': ValidationStatus.ERROR.value,
                'message': f"Unknown validation type: {validation_type}",
                'errors': [f"Unknown validation type: {validation_type}"]
            }
        
        # Add metadata
        duration = time.time() - start_time
        result.update({
            'target': str(target),
            'type': validation_type,
            'level': level,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update metrics
        self._update_metrics(result, duration)
        
        # Store in history
        self.validation_history.append(result)
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        # Update stats
        self.validation_stats['total_validations'] += 1
        self.validation_stats[result['status']] = self.validation_stats.get(result['status'], 0) + 1
        self.validation_stats['last_validation'] = result['timestamp']
        
        # Store failures separately
        if result['status'] == ValidationStatus.FAILED.value:
            self.failures.append(result)
        elif result['status'] == ValidationStatus.WARNING.value:
            self.warnings.append(result)
        
        # Cache result
        if self.config['cache_enabled']:
            self.cache[cache_key] = {
                'timestamp': time.time(),
                'result': result
            }
        
        logger.info(f"✅ Validation complete: {result['status']} in {duration:.3f}s")
        
        return result
    
    def validate_all(self, level: int = None) -> List[Dict[str, Any]]:
        """
        Validate all available targets
        
        Args:
            level: Validation strictness level
        
        Returns:
            List of validation results
        """
        logger.info("🔍 Running comprehensive validation")
        
        results = []
        targets = []
        
        # Get components if DMAI core is available
        if hasattr(self, 'dmai') and hasattr(self.dmai, 'components'):
            for comp_id in self.dmai.components.keys():
                targets.append(('component', comp_id))
        
        # Add other targets
        targets.extend([
            ('config', 'system'),
            ('data', 'knowledge_graph'),
            ('health', 'core'),
            ('integrity', 'database')
        ])
        
        # Validate each target
        for v_type, target in targets[:self.config['max_validations_per_cycle']]:
            result = self.validate(target, v_type, level)
            results.append(result)
            
            if self.config['fail_fast'] and result['status'] == ValidationStatus.FAILED.value:
                logger.warning(f"⚠️ Fail-fast triggered by {target}")
                break
        
        return results
    
    def validate_batch(self, targets: List[str], validation_type: str = ValidationType.COMPONENT.value,
                      level: int = None) -> List[Dict[str, Any]]:
        """
        Validate a batch of targets
        
        Args:
            targets: List of targets to validate
            validation_type: Type of validation
            level: Validation strictness level
        
        Returns:
            List of validation results
        """
        logger.info(f"🔍 Validating batch of {len(targets)} targets")
        
        results = []
        for target in targets:
            result = self.validate(target, validation_type, level)
            results.append(result)
        
        return results
    
    def quick_check(self, target: Any) -> Dict[str, Any]:
        """
        Perform a quick health check on a target
        
        Args:
            target: Target to check
        
        Returns:
            Quick check result
        """
        return self.validate(target, ValidationType.HEALTH.value, ValidationLevel.PERMISSIVE.value)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform overall system health check"""
        logger.info("🏥 Running system health check")
        
        results = self.validate_all(ValidationLevel.NORMAL.value)
        
        # Calculate overall health
        passed = sum(1 for r in results if r['status'] == ValidationStatus.PASSED.value)
        total = len(results)
        
        health_score = (passed / total * 100) if total > 0 else 100
        
        return {
            'status': ValidationStatus.PASSED.value if health_score >= 80 else ValidationStatus.WARNING.value,
            'health_score': health_score,
            'passed': passed,
            'total': total,
            'failed': total - passed,
            'results': results[:5],  # First 5 results
            'timestamp': datetime.now().isoformat()
        }
    
    def verify(self, target: Any, expected: Any) -> Dict[str, Any]:
        """
        Verify that a target matches expected value
        
        Args:
            target: Target to verify
            expected: Expected value
        
        Returns:
            Verification result
        """
        logger.info(f"✅ Verifying {target} against expected")
        
        result = {
            'target': str(target),
            'expected': expected,
            'actual': None,
            'matches': False,
            'status': ValidationStatus.FAILED.value
        }
        
        # Try to get actual value
        if hasattr(self, 'dmai'):
            if hasattr(self.dmai, 'components') and target in self.dmai.components:
                result['actual'] = self.dmai.components[target].get('status')
            elif hasattr(self.dmai, 'db'):
                # Query database
                pass
        
        # Compare
        if result['actual'] == expected:
            result['matches'] = True
            result['status'] = ValidationStatus.PASSED.value
        
        return result
    
    def add_validation_rule(self, name: str, rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a custom validation rule
        
        Args:
            name: Rule name
            rule_config: Rule configuration
        """
        logger.info(f"➕ Adding validation rule: {name}")
        
        self.custom_rules[name] = {
            'name': name,
            'config': rule_config,
            'created_at': datetime.now().isoformat(),
            'usage_count': 0,
            'success_rate': 1.0
        }
        
        return {
            'status': 'added',
            'name': name,
            'rule': self.custom_rules[name]
        }
    
    def remove_validation_rule(self, name: str) -> Dict[str, Any]:
        """
        Remove a custom validation rule
        
        Args:
            name: Rule name to remove
        """
        if name in self.custom_rules:
            del self.custom_rules[name]
            logger.info(f"➖ Removed validation rule: {name}")
            return {'status': 'removed', 'name': name}
        
        return {'error': f'Rule {name} not found', 'status': 'failed'}
    
    def list_rules(self, validation_type: str = None) -> Dict[str, Any]:
        """
        List all validation rules
        
        Args:
            validation_type: Optional type filter
        """
        rules = {}
        
        # Add default rules
        for r_name, r_config in self.rules.items():
            if not validation_type or r_config.get('type') == validation_type:
                rules[r_name] = r_config
        
        # Add custom rules
        for r_name, r_config in self.custom_rules.items():
            if not validation_type or r_config['config'].get('type') == validation_type:
                rules[r_name] = r_config['config']
        
        return {
            'total': len(rules),
            'rules': rules
        }
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear validation cache"""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"🧹 Cleared validation cache ({cache_size} entries)")
        
        return {
            'status': 'cleared',
            'entries_removed': cache_size
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return self.validation_stats.copy()
    
    def reset(self) -> Dict[str, Any]:
        """Reset validator state"""
        logger.info("🔄 Resetting validator")
        
        self.validation_history = []
        self.validation_stats = {
            'total_validations': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'errors': 0,
            'avg_duration': 0,
            'last_validation': None
        }
        
        self.failures = []
        self.warnings = []
        self.cache.clear()
        
        self.metrics = {
            'fastest_validation': float('inf'),
            'slowest_validation': 0,
            'most_common_failure': None,
            'reliability_score': 1.0
        }
        
        return {'status': 'reset', 'component': self.component_id}
    
    def _validate_component(self, component_id: str, level: int) -> Dict[str, Any]:
        """Validate a component"""
        errors = []
        warnings = []
        
        # Check if component exists
        if not hasattr(self, 'dmai') or not hasattr(self.dmai, 'components'):
            return {
                'status': ValidationStatus.ERROR.value,
                'message': 'Cannot access DMAI core',
                'errors': ['DMAI core not available']
            }
        
        if component_id not in self.dmai.components:
            return {
                'status': ValidationStatus.FAILED.value,
                'message': f'Component {component_id} not found',
                'errors': [f'Component {component_id} not found']
            }
        
        component = self.dmai.components[component_id]
        instance = component.get('instance')
        
        if not instance:
            return {
                'status': ValidationStatus.FAILED.value,
                'message': f'Component {component_id} has no instance',
                'errors': ['No instance found']
            }
        
        # Check required methods
        required_methods = ['run', 'evolve', 'execute', 'process', 'generate', 'query']
        present_methods = []
        missing_methods = []
        
        for method in required_methods:
            if hasattr(instance, method):
                present_methods.append(method)
            else:
                missing_methods.append(method)
        
        # Calculate health score
        health_score = (len(present_methods) / len(required_methods)) * 100
        
        if missing_methods:
            if level <= ValidationLevel.NORMAL.value:
                errors.append(f"Missing methods: {missing_methods}")
            else:
                warnings.append(f"Missing methods: {missing_methods}")
        
        # Check evolution attempts
        evolution_attempts = component.get('evolution_attempts', 0)
        if evolution_attempts > 10 and level <= ValidationLevel.STRICT.value:
            warnings.append(f"High evolution attempts: {evolution_attempts}")
        
        # Determine status
        if errors:
            status = ValidationStatus.FAILED.value
        elif warnings:
            status = ValidationStatus.WARNING.value
        else:
            status = ValidationStatus.PASSED.value
        
        return {
            'status': status,
            'message': f"Component {component_id} validation complete",
            'errors': errors,
            'warnings': warnings,
            'health_score': health_score,
            'present_methods': present_methods,
            'missing_methods': missing_methods,
            'evolution_attempts': evolution_attempts
        }
    
    def _validate_data(self, data: Any, level: int) -> Dict[str, Any]:
        """Validate data"""
        errors = []
        warnings = []
        
        # Check if data exists
        if data is None:
            return {
                'status': ValidationStatus.FAILED.value,
                'message': 'Data is None',
                'errors': ['Data is None']
            }
        
        # Type-specific validation
        if isinstance(data, dict):
            # Check for required keys based on level
            if level <= ValidationLevel.STRICT.value:
                if not data:
                    warnings.append("Empty dictionary")
            
            # Check for common issues
            if len(data) > 1000 and level <= ValidationLevel.PERMISSIVE.value:
                warnings.append(f"Large dictionary size: {len(data)}")
        
        elif isinstance(data, list):
            if not data and level <= ValidationLevel.STRICT.value:
                warnings.append("Empty list")
            
            if len(data) > 10000 and level <= ValidationLevel.PERMISSIVE.value:
                warnings.append(f"Large list size: {len(data)}")
        
        elif isinstance(data, str):
            if not data and level <= ValidationLevel.NORMAL.value:
                errors.append("Empty string")
            elif len(data) > 1000000 and level <= ValidationLevel.PERMISSIVE.value:
                warnings.append(f"Large string length: {len(data)}")
        
        # Determine status
        if errors:
            status = ValidationStatus.FAILED.value
        elif warnings:
            status = ValidationStatus.WARNING.value
        else:
            status = ValidationStatus.PASSED.value
        
        return {
            'status': status,
            'message': 'Data validation complete',
            'errors': errors,
            'warnings': warnings,
            'data_type': type(data).__name__,
            'size': self._get_size(data)
        }
    
    def _validate_schema(self, schema: Any, level: int) -> Dict[str, Any]:
        """Validate a schema"""
        errors = []
        warnings = []
        
        # Basic schema validation
        if not isinstance(schema, dict):
            return {
                'status': ValidationStatus.FAILED.value,
                'message': 'Schema must be a dictionary',
                'errors': ['Schema must be a dictionary']
            }
        
        # Check for required schema fields
        required_fields = ['type', 'properties'] if level <= ValidationLevel.NORMAL.value else []
        for field in required_fields:
            if field not in schema:
                errors.append(f"Missing required field: {field}")
        
        # Validate properties if present
        if 'properties' in schema:
            properties = schema['properties']
            if not isinstance(properties, dict):
                errors.append("Properties must be a dictionary")
            elif not properties and level <= ValidationLevel.STRICT.value:
                warnings.append("Empty properties")
        
        # Determine status
        if errors:
            status = ValidationStatus.FAILED.value
        elif warnings:
            status = ValidationStatus.WARNING.value
        else:
            status = ValidationStatus.PASSED.value
        
        return {
            'status': status,
            'message': 'Schema validation complete',
            'errors': errors,
            'warnings': warnings,
            'fields': list(schema.keys())
        }
    
    def _validate_config(self, config: Any, level: int) -> Dict[str, Any]:
        """Validate configuration"""
        errors = []
        warnings = []
        
        if not isinstance(config, dict):
            return {
                'status': ValidationStatus.FAILED.value,
                'message': 'Config must be a dictionary',
                'errors': ['Config must be a dictionary']
            }
        
        # Check for common config issues
        for key, value in config.items():
            if value is None and level <= ValidationLevel.NORMAL.value:
                warnings.append(f"Config key '{key}' is None")
            
            if isinstance(value, (int, float)) and value < 0 and level <= ValidationLevel.STRICT.value:
                warnings.append(f"Negative value for '{key}': {value}")
        
        # Determine status
        if errors:
            status = ValidationStatus.FAILED.value
        elif warnings:
            status = ValidationStatus.WARNING.value
        else:
            status = ValidationStatus.PASSED.value
        
        return {
            'status': status,
            'message': 'Config validation complete',
            'errors': errors,
            'warnings': warnings,
            'config_keys': list(config.keys())
        }
    
    def _validate_health(self, target: Any, level: int) -> Dict[str, Any]:
        """Validate health of a target"""
        errors = []
        warnings = []
        
        # Quick health check
        if hasattr(self, 'dmai'):
            if target == 'core':
                # Check core health
                if hasattr(self.dmai, 'running'):
                    if not self.dmai.running:
                        errors.append("DMAI core not running")
                
                if hasattr(self.dmai, 'components'):
                    component_count = len(self.dmai.components)
                    if component_count == 0:
                        errors.append("No components loaded")
                    elif component_count < 10 and level <= ValidationLevel.STRICT.value:
                        warnings.append(f"Low component count: {component_count}")
        
        # Determine status
        if errors:
            status = ValidationStatus.FAILED.value
        elif warnings:
            status = ValidationStatus.WARNING.value
        else:
            status = ValidationStatus.PASSED.value
        
        return {
            'status': status,
            'message': f'Health check for {target} complete',
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_security(self, target: Any, level: int) -> Dict[str, Any]:
        """Validate security of a target"""
        errors = []
        warnings = []
        
        # Security checks
        if isinstance(target, str):
            # Check for sensitive patterns
            sensitive_patterns = [
                r'password\s*=\s*[\'"].+[\'"]',
                r'api_key\s*=\s*[\'"].+[\'"]',
                r'token\s*=\s*[\'"].+[\'"]',
                r'secret\s*=\s*[\'"].+[\'"]'
            ]
            
            for pattern in sensitive_patterns:
                if re.search(pattern, target, re.IGNORECASE):
                    if level <= ValidationLevel.NORMAL.value:
                        errors.append(f"Sensitive data pattern found: {pattern}")
                    else:
                        warnings.append(f"Possible sensitive data: {pattern}")
        
        # Determine status
        if errors:
            status = ValidationStatus.FAILED.value
        elif warnings:
            status = ValidationStatus.WARNING.value
        else:
            status = ValidationStatus.PASSED.value
        
        return {
            'status': status,
            'message': 'Security validation complete',
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_performance(self, target: Any, level: int) -> Dict[str, Any]:
        """Validate performance of a target"""
        errors = []
        warnings = []
        
        # Performance metrics
        if isinstance(target, (int, float)):
            if target < 0:
                errors.append("Negative performance metric")
            elif target > 1000 and level <= ValidationLevel.STRICT.value:
                warnings.append(f"High performance metric: {target}")
        
        # Determine status
        if errors:
            status = ValidationStatus.FAILED.value
        elif warnings:
            status = ValidationStatus.WARNING.value
        else:
            status = ValidationStatus.PASSED.value
        
        return {
            'status': status,
            'message': 'Performance validation complete',
            'errors': errors,
            'warnings': warnings,
            'value': target
        }
    
    def _validate_integrity(self, target: Any, level: int) -> Dict[str, Any]:
        """Validate integrity of a target"""
        errors = []
        warnings = []
        
        # Check integrity using hash if target is string
        if isinstance(target, str):
            # Simple integrity check - just check if string is not empty
            if not target:
                errors.append("Empty target")
            elif len(target) < 10 and level <= ValidationLevel.STRICT.value:
                warnings.append(f"Target too short for integrity check: {len(target)}")
        
        # Determine status
        if errors:
            status = ValidationStatus.FAILED.value
        elif warnings:
            status = ValidationStatus.WARNING.value
        else:
            status = ValidationStatus.PASSED.value
        
        return {
            'status': status,
            'message': 'Integrity validation complete',
            'errors': errors,
            'warnings': warnings
        }
    
    def _load_default_rules(self) -> Dict[str, Any]:
        """Load default validation rules"""
        return {
            'component_exists': {
                'type': ValidationType.COMPONENT.value,
                'description': 'Component must exist',
                'severity': 'error'
            },
            'has_required_methods': {
                'type': ValidationType.COMPONENT.value,
                'description': 'Component must have required methods',
                'required_methods': ['run', 'evolve', 'execute'],
                'severity': 'error'
            },
            'data_not_empty': {
                'type': ValidationType.DATA.value,
                'description': 'Data should not be empty',
                'severity': 'warning'
            },
            'config_has_required_keys': {
                'type': ValidationType.CONFIG.value,
                'description': 'Config must have required keys',
                'severity': 'error'
            },
            'health_check_passed': {
                'type': ValidationType.HEALTH.value,
                'description': 'Health check must pass',
                'severity': 'error'
            }
        }
    
    def _create_rule_from_failure(self, failure_type: str):
        """Create a new validation rule from a failure pattern"""
        rule_name = f"auto_rule_{failure_type}_{int(time.time())}"
        self.custom_rules[rule_name] = {
            'name': rule_name,
            'config': {
                'type': failure_type,
                'description': f'Auto-generated rule from failure pattern',
                'severity': 'error',
                'auto_generated': True,
                'created_at': datetime.now().isoformat()
            },
            'created_at': datetime.now().isoformat(),
            'usage_count': 0,
            'success_rate': 0.5
        }
        logger.info(f"🤖 Auto-generated rule: {rule_name}")
    
    def _validation_cycle(self):
        """Run a validation cycle"""
        logger.info("🔄 Running validation cycle")
        
        # Validate all components
        if hasattr(self, 'dmai') and hasattr(self.dmai, 'components'):
            for comp_id in self.dmai.components.keys():
                self.validate(comp_id, ValidationType.COMPONENT.value)
        
        # Validate system health
        self.validate('system', ValidationType.HEALTH.value)
        
        # Clean old cache entries
        self._clean_cache()
    
    def _clean_cache(self):
        """Remove expired cache entries"""
        now = time.time()
        expired = []
        
        for key, entry in self.cache.items():
            if now - entry['timestamp'] > self.cache_ttl:
                expired.append(key)
        
        for key in expired:
            del self.cache[key]
        
        if expired:
            logger.debug(f"Cleaned {len(expired)} expired cache entries")
    
    def _get_cache_key(self, target: Any, v_type: str, level: int) -> str:
        """Generate cache key for validation"""
        return hashlib.md5(f"{target}{v_type}{level}".encode()).hexdigest()
    
    def _get_size(self, data: Any) -> int:
        """Get size of data in bytes (approximate)"""
        try:
            return len(str(data))
        except:
            return 0
    
    def _update_metrics(self, result: Dict[str, Any], duration: float):
        """Update performance metrics"""
        # Update fastest/slowest
        if duration < self.metrics['fastest_validation']:
            self.metrics['fastest_validation'] = duration
        if duration > self.metrics['slowest_validation']:
            self.metrics['slowest_validation'] = duration
        
        # Update average duration
        total = self.validation_stats['total_validations']
        avg = self.validation_stats['avg_duration']
        self.validation_stats['avg_duration'] = (avg * total + duration) / (total + 1)
        
        # Update most common failure
        if result['status'] == ValidationStatus.FAILED.value:
            failure_key = f"{result.get('type', 'unknown')}:{result.get('message', 'unknown')}"
            if not self.metrics['most_common_failure']:
                self.metrics['most_common_failure'] = failure_key
        
        # Update reliability score
        total_valid = self.validation_stats['total_validations']
        if total_valid > 0:
            self.metrics['reliability_score'] = self.validation_stats['passed'] / total_valid
    
    def _is_healthy(self) -> bool:
        """Check if validator itself is healthy"""
        return (self.validation_stats['total_validations'] == 0 or
                self.validation_stats['failed'] < self.validation_stats['total_validations'] * 0.1)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'stats': self.validation_stats,
            'cache_size': len(self.cache),
            'rules_count': len(self.rules) + len(self.custom_rules),
            'failures_count': len(self.failures),
            'warnings_count': len(self.warnings),
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
            "validation_types": [t.value for t in ValidationType],
            "stats": self.validation_stats,
            "methods": ['run', 'evolve', 'execute', 'process', 'generate', 'query']
        }

# Guard clause ensures code only runs when script is executed directly
if __name__ == "__main__":
    print("\n" + "="*60)
    print("✅ VALIDATOR IMPLEMENTATION (P1T5)")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Validator Implementation')
    parser.add_argument('--validate', metavar='TARGET', help='Validate a target')
    parser.add_argument('--type', default='component', help='Validation type')
    parser.add_argument('--level', type=int, default=3, help='Validation level (1-5)')
    parser.add_argument('--check', action='store_true', help='Run health check')
    parser.add_argument('--verify', nargs=2, metavar=('TARGET', 'EXPECTED'), help='Verify target')
    parser.add_argument('--list-rules', action='store_true', help='List validation rules')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache')
    
    args = parser.parse_args()
    
    validator = Validator()
    
    if args.validate:
        print(f"\n🔍 Validating {args.validate} ({args.type}, level={args.level})...")
        result = validator.validate(args.validate, args.type, args.level)
        print(json.dumps(result, indent=2))
    
    elif args.check:
        print("\n🏥 Running health check...")
        result = validator.health_check()
        print(json.dumps(result, indent=2))
    
    elif args.verify:
        print(f"\n✅ Verifying {args.verify[0]} against {args.verify[1]}...")
        result = validator.verify(args.verify[0], args.verify[1])
        print(json.dumps(result, indent=2))
    
    elif args.list_rules:
        print("\n📋 Validation Rules:")
        rules = validator.list_rules()
        print(json.dumps(rules, indent=2))
    
    elif args.stats:
        print("\n📊 Validation Statistics:")
        print(json.dumps(validator.get_stats(), indent=2))
    
    elif args.clear_cache:
        print("\n🧹 Clearing cache...")
        result = validator.clear_cache()
        print(json.dumps(result, indent=2))
    
    else:
        print("\n📋 Component Info:")
        print(json.dumps(validator.info(), indent=2))
        print("\n💡 Use --validate, --check, --verify, --list-rules, --stats, or --clear-cache")
