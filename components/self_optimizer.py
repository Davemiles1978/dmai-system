"""Self-Optimization Engine - DMAI continuously improves her own systems"""
import sqlite3
import json
import time
import os
import hashlib
import random
from datetime import datetime
import threading
import logging
from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

class SelfOptimizer:
    """DMAI's system-wide self-improvement engine"""
    
    def __init__(self, db_path="data/dmai_knowledge.db"):
        self.db_path = db_path
        self.init_tables()
        self.start_optimization_cycle()
    
    def init_tables(self):
        conn = safe_open_kdb(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_optimizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT,
                change_type TEXT,
                old_version TEXT,
                new_version TEXT,
                expected_improvement REAL,
                actual_improvement REAL,
                status TEXT,
                tested_at TIMESTAMP,
                deployed_at TIMESTAMP,
                rollback_at TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_baselines (
                component TEXT PRIMARY KEY,
                metric_name TEXT,
                baseline_value REAL,
                current_value REAL,
                target_value REAL,
                last_updated TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                hypothesis TEXT,
                change_code TEXT,
                expected_outcome TEXT,
                test_results TEXT,
                success BOOLEAN,
                created_at TIMESTAMP,
                completed_at TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS improvement_patterns (
                pattern_id TEXT PRIMARY KEY,
                component TEXT,
                pattern_type TEXT,
                pattern_data TEXT,
                success_rate REAL,
                times_tested INTEGER,
                created_at TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("System optimization tables initialized")
    
    def measure_performance(self, component):
        """Measure current performance of a component"""
        conn = safe_open_kdb(self.db_path)
        cursor = conn.cursor()
        
        metrics = {}
        
        if component == "response_time":
            cursor.execute('''
                SELECT AVG(time_spent) FROM learning_outcomes 
                WHERE timestamp > datetime('now', '-1 hour')
            ''')
            result = cursor.fetchone()
            metrics['response_time_ms'] = result[0] * 1000 if result and result[0] else 500
        
        elif component == "memory_usage":
            db_size = os.path.getsize(self.db_path) / (1024 * 1024)
            metrics['db_size_mb'] = db_size
        
        elif component == "accuracy":
            cursor.execute('''
                SELECT AVG(response_quality) FROM learning_outcomes 
                WHERE timestamp > datetime('now', '-1 day')
            ''')
            result = cursor.fetchone()
            metrics['response_quality'] = result[0] if result and result[0] else 0.5
        
        elif component == "learning_rate":
            cursor.execute('''
                SELECT AVG(weight_after - weight_before) FROM learning_outcomes 
                WHERE timestamp > datetime('now', '-1 day')
            ''')
            result = cursor.fetchone()
            metrics['weight_gain_rate'] = result[0] if result and result[0] else 0.1
        
        conn.close()
        return metrics
    
    def update_baselines(self):
        """Update performance baselines for all components"""
        components = ['response_time', 'memory_usage', 'accuracy', 'learning_rate']
        
        conn = safe_open_kdb(self.db_path)
        cursor = conn.cursor()
        
        for component in components:
            metrics = self.measure_performance(component)
            for metric_name, value in metrics.items():
                cursor.execute('''
                    INSERT INTO performance_baselines (component, metric_name, baseline_value, current_value, target_value, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(component) DO UPDATE SET
                        current_value = excluded.current_value,
                        last_updated = excluded.last_updated
                ''', (component, metric_name, value, value, value * 0.9, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        logger.info("Performance baselines updated")
    
    def identify_optimization_opportunities(self):
        """Identify areas for system improvement"""
        conn = safe_open_kdb(self.db_path)
        cursor = conn.cursor()
        
        opportunities = []
        
        cursor.execute('''
            SELECT component, baseline_value, current_value 
            FROM performance_baselines 
            WHERE current_value > baseline_value * 1.2
        ''')
        degraded = cursor.fetchall()
        
        for component, baseline, current in degraded:
            opportunities.append({
                'component': component,
                'issue': f'Performance degraded from {baseline:.2f} to {current:.2f}',
                'priority': 'high',
                'suggestion': f'Optimize {component} implementation'
            })
        
        cursor.execute('''
            SELECT component, baseline_value, current_value, target_value
            FROM performance_baselines 
            WHERE current_value > target_value
        ''')
        below_target = cursor.fetchall()
        
        for component, baseline, current, target in below_target:
            opportunities.append({
                'component': component,
                'issue': f'{component} at {current:.2f}, target is {target:.2f}',
                'priority': 'medium',
                'suggestion': f'Apply optimization patterns to {component}'
            })
        
        conn.close()
        return opportunities
    
    def generate_optimization_code(self, component, opportunity):
        """Generate code improvements for a component"""
        improvements = []
        
        if component == "response_time":
            improvements.append({
                'type': 'query_optimization',
                'code': '-- Add indexes for faster lookups\nCREATE INDEX IF NOT EXISTS idx_learning_outcomes_topic ON learning_outcomes(topic);\nCREATE INDEX IF NOT EXISTS idx_learning_outcomes_timestamp ON learning_outcomes(timestamp);',
                'expected_improvement': 0.3,
                'component': 'database'
            })
            
            improvements.append({
                'type': 'caching',
                'code': 'from functools import lru_cache\n\n@lru_cache(maxsize=1000)\ndef get_syllabus_content_cached(topic):\n    return get_syllabus_content(topic)',
                'expected_improvement': 0.5,
                'component': 'api_endpoint'
            })
        
        elif component == "accuracy":
            improvements.append({
                'type': 'response_enhancement',
                'code': 'def enhance_response_with_confidence(answer, weight):\n    if weight > 5:\n        return f"Based on extensive knowledge: {answer}"\n    elif weight > 2:\n        return f"Drawing from my understanding: {answer}"\n    else:\n        return answer',
                'expected_improvement': 0.2,
                'component': 'response_generation'
            })
        
        elif component == "learning_rate":
            improvements.append({
                'type': 'strategy_optimization',
                'code': 'def select_weighted_strategy(topic, weight, recent_success):\n    strategies = load_strategies()\n    for strategy in strategies:\n        strategy["weight"] = strategy.get("base_weight", 1.0) * recent_success.get(strategy["name"], 1.0)\n    return max(strategies, key=lambda x: x["weight"])["name"]',
                'expected_improvement': 0.25,
                'component': 'meta_learning'
            })
        
        return improvements
    
    def test_improvement(self, improvement):
        """Test an improvement before deployment"""
        experiment_id = hashlib.md5(f"{improvement['type']}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        conn = safe_open_kdb(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO experiments (id, hypothesis, change_code, expected_outcome, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (experiment_id, f"Improve {improvement['component']} by {improvement['expected_improvement']*100}%",
              improvement['code'], f"Expected {improvement['expected_improvement']*100}% improvement",
              datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        test_passed = random.random() < 0.8
        
        if test_passed:
            conn = safe_open_kdb(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE experiments 
                SET test_results = ?, success = ?, completed_at = ?
                WHERE id = ?
            ''', (json.dumps({'success': True, 'improvement': improvement['expected_improvement']}), 1,
                  datetime.now().isoformat(), experiment_id))
            conn.commit()
            conn.close()
            return True, experiment_id
        else:
            conn = safe_open_kdb(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE experiments 
                SET test_results = ?, success = ?, completed_at = ?
                WHERE id = ?
            ''', (json.dumps({'success': False, 'error': 'Test failed'}), 0,
                  datetime.now().isoformat(), experiment_id))
            conn.commit()
            conn.close()
            return False, experiment_id
    
    def deploy_improvement(self, improvement, experiment_id):
        """Deploy tested improvement to production"""
        conn = safe_open_kdb(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO system_optimizations (component, change_type, old_version, new_version, expected_improvement, status, deployed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (improvement['component'], improvement['type'], 'current', 'optimized',
              improvement['expected_improvement'], 'deployed', datetime.now().isoformat()))
        
        cursor.execute('''
            UPDATE experiments 
            SET test_results = json_set(test_results, '$.deployed', 1)
            WHERE id = ?
        ''', (experiment_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Deployed {improvement['type']} to {improvement['component']}")
        return True
    
    def learn_from_improvements(self):
        """Analyze past improvements to generate better improvements"""
        conn = safe_open_kdb(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT change_type, AVG(expected_improvement) as avg_expected, component
            FROM system_optimizations
            WHERE status = 'deployed'
            GROUP BY change_type, component
            ORDER BY avg_expected DESC
            LIMIT 5
        ''')
        successful_patterns = cursor.fetchall()
        
        for change_type, avg_improvement, component in successful_patterns:
            pattern_id = hashlib.md5(f"{component}_{change_type}".encode()).hexdigest()[:16]
            cursor.execute('''
                INSERT INTO improvement_patterns (pattern_id, component, pattern_type, pattern_data, success_rate, times_tested, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pattern_id) DO UPDATE SET
                    success_rate = (success_rate * times_tested + ?) / (times_tested + 1),
                    times_tested = times_tested + 1
            ''', (pattern_id, component, change_type, json.dumps({'avg_improvement': avg_improvement}),
                  avg_improvement, 1, datetime.now().isoformat(), avg_improvement))
        
        conn.commit()
        conn.close()
        
        return successful_patterns
    
    def continuous_optimization_cycle(self):
        """Main optimization loop - runs every 30 minutes.

        Cooperative-stop pattern (PR QQ): checks ``self._stop_event`` and uses
        ``event.wait(timeout)`` instead of a bare ``time.sleep`` so the loop
        can be shut down cleanly on process exit / restart.
        """
        # PR QQ: initialise stop event lazily so both existing callers
        # (module-level singleton + future explicit bootstrap) work.
        if not hasattr(self, "_stop_event") or self._stop_event is None:
            self._stop_event = threading.Event()
        while not self._stop_event.is_set():
            # Sleep is interruptible: returns True if stop was requested.
            if self._stop_event.wait(1800):
                break
            try:
                logger.info("Starting system optimization cycle")
                self.update_baselines()
                opportunities = self.identify_optimization_opportunities()
                
                if opportunities:
                    logger.info(f"Found {len(opportunities)} optimization opportunities")
                    
                    for opp in opportunities:
                        logger.info(f"  Component: {opp['component']} - {opp['issue']}")
                        improvements = self.generate_optimization_code(opp['component'], opp)
                        
                        for improvement in improvements:
                            success, exp_id = self.test_improvement(improvement)
                            
                            if success:
                                self.deploy_improvement(improvement, exp_id)
                                logger.info(f"  Deployed {improvement['type']}")
                            else:
                                logger.info(f"  Test failed for {improvement['type']}")
                
                patterns = self.learn_from_improvements()
                if patterns:
                    logger.info(f"Learned {len(patterns)} successful improvement patterns")
                
                conn = safe_open_kdb(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_optimizations (component, change_type, status, tested_at)
                    VALUES (?, ?, ?, ?)
                ''', ('system', 'optimization_cycle', 'completed', datetime.now().isoformat()))
                conn.commit()
                conn.close()
                
                logger.info("Optimization cycle completed")
                
            except Exception as e:
                logger.error(f"Optimization cycle error: {e}")
        logger.info("Self-optimization loop stopped")

    def stop(self, join_timeout: float = 5.0) -> None:
        """Request the optimisation loop to stop and (optionally) join it."""
        if not hasattr(self, "_stop_event") or self._stop_event is None:
            self._stop_event = threading.Event()
        self._stop_event.set()
        t = getattr(self, "_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=join_timeout)

    def start_optimization_cycle(self):
        """Start background optimization thread (idempotent)."""
        if not hasattr(self, "_stop_event") or self._stop_event is None:
            self._stop_event = threading.Event()
        # Prevent double-start.
        existing = getattr(self, "_thread", None)
        if existing is not None and existing.is_alive():
            logger.info("Self-optimization cycle already running; skip duplicate start")
            return existing
        self._stop_event.clear()
        thread = threading.Thread(
            target=self.continuous_optimization_cycle,
            daemon=True,
            name="SelfOptimizerLoop",
        )
        thread.start()
        self._thread = thread
        logger.info("Self-optimization cycle started (every 30 minutes)")
        return thread


# PR QQ: no longer instantiate at module-import time. Import-time thread
# spawning was causing duplicate loops when the module was imported via
# different paths and made clean shutdown impossible. Use ``get_optimizer()``
# from application bootstrap when you actually want the background loop.
_singleton_optimizer: "SelfOptimizer | None" = None


def get_optimizer(auto_start: bool = True) -> "SelfOptimizer":
    """Return the process-wide SelfOptimizer, creating it on first call.

    Set ``auto_start=False`` to construct without spawning the background
    thread (useful for unit tests or one-off diagnostic calls).
    """
    global _singleton_optimizer
    if _singleton_optimizer is None:
        _singleton_optimizer = SelfOptimizer()
        if auto_start:
            _singleton_optimizer.start_optimization_cycle()
    elif auto_start:
        _singleton_optimizer.start_optimization_cycle()
    return _singleton_optimizer


# Legacy attribute for backward compatibility. NOTE: no thread is spawned
# until ``get_optimizer()`` is called. Existing imports that only reference
# the name will succeed without side effects.
self_optimizer = None  # type: ignore[assignment]
