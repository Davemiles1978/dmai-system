"""
Self-Assessment Metrics for AGI Evolution System
Measures learning effectiveness, evolution quality, and system health
Phase 2 - Self-Awareness
"""

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from collections import defaultdict
import psutil
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - ASSESSMENT - %(message)s')
logger = logging.getLogger('self_assessment')

class SelfAssessment:
    def __init__(self, data_path="shared_data/agi_evolution/assessment"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics_history = []
        self.learning_curves = defaultdict(list)
        self.quality_scores = {}
        
        # Load history
        self.load_history()
        
        logger.info("📊 Self-Assessment system initialized")
    
    def load_history(self):
        """Load assessment history"""
        history_file = self.data_path / "assessment_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                data = json.load(f)
                self.metrics_history = data.get('history', [])
                self.learning_curves = defaultdict(list, data.get('curves', {}))
    
    def save_history(self):
        """Save assessment history"""
        history_file = self.data_path / "assessment_history.json"
        with open(history_file, 'w') as f:
            json.dump({
                'history': self.metrics_history[-100:],  # Keep last 100
                'curves': dict(self.learning_curves)
            }, f, indent=2, default=str)
    
    def assess_learning_effectiveness(self, evolution_data):
        """
        Measure how effective learning has been
        Returns metrics on learning quality
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'generation': evolution_data.get('generation', 0),
            'learning_rate': 0,
            'improvement_rate': 0,
            'stability_score': 0,
            'convergence_speed': 0,
            'exploration_efficiency': 0
        }
        
        # Calculate learning rate (how fast improvements happen)
        if len(self.learning_curves['scores']) >= 2:
            recent = self.learning_curves['scores'][-5:]
            if len(recent) >= 2:
                improvements = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
                metrics['learning_rate'] = np.mean(improvements) if improvements else 0
        
        # Calculate improvement rate (percentage of positive changes)
        mutations = evolution_data.get('mutations', [])
        if mutations:
            successes = sum(1 for m in mutations if m.get('success', False))
            metrics['improvement_rate'] = successes / len(mutations)
        
        # Calculate stability score (variance in performance)
        if len(self.learning_curves['scores']) >= 10:
            recent_scores = self.learning_curves['scores'][-10:]
            metrics['stability_score'] = 1.0 - min(1.0, np.std(recent_scores) / np.mean(recent_scores))
        
        # Calculate convergence speed (how quickly we reach targets)
        if 'target_scores' in evolution_data:
            target = evolution_data['target_scores']
            current = evolution_data.get('best_score', 0)
            if target > 0:
                metrics['convergence_speed'] = current / target
        
        # Calculate exploration efficiency
        if 'files_processed' in evolution_data and 'improvements' in evolution_data:
            files = evolution_data['files_processed']
            improvements = evolution_data['improvements']
            metrics['exploration_efficiency'] = improvements / files if files > 0 else 0
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def assess_knowledge_quality(self, knowledge_graph):
        """
        Assess the quality of the knowledge graph
        Measures connectivity, depth, and usefulness
        """
        quality = {
            'timestamp': datetime.now().isoformat(),
            'node_count': 0,
            'edge_count': 0,
            'connectivity': 0,
            'avg_clustering': 0,
            'knowledge_depth': 0,
            'usefulness_score': 0
        }
        
        try:
            # Get graph stats
            quality['node_count'] = len(knowledge_graph.graph.nodes)
            quality['edge_count'] = len(knowledge_graph.graph.edges)
            
            # Calculate connectivity (edges per node)
            if quality['node_count'] > 0:
                quality['connectivity'] = quality['edge_count'] / quality['node_count']
            
            # Calculate knowledge depth (longest path)
            if quality['node_count'] > 1:
                try:
                    # Find longest path (simplified)
                    max_depth = 0
                    for node in list(knowledge_graph.graph.nodes)[:10]:  # Limit for performance
                        try:
                            paths = nx.single_source_shortest_path_length(knowledge_graph.graph, node)
                            max_depth = max(max_depth, max(paths.values()))
                        except:
                            pass
                    quality['knowledge_depth'] = max_depth
                except:
                    pass
            
            # Calculate usefulness (based on access patterns)
            total_access = sum(
                knowledge_graph.graph.nodes[n].get('access_count', 0) 
                for n in knowledge_graph.graph.nodes
            )
            if quality['node_count'] > 0:
                quality['usefulness_score'] = min(1.0, total_access / (quality['node_count'] * 100))
                
        except Exception as e:
            logger.error(f"Error assessing knowledge graph: {e}")
        
        return quality
    
    def assess_system_health(self):
        """
        Assess overall system health
        Returns comprehensive health metrics
        """
        health = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'response_time': 0,
            'error_rate': 0,
            'uptime': 0,
            'overall_score': 0
        }
        
        try:
            # System resources
            health['cpu_usage'] = psutil.cpu_percent(interval=1)
            health['memory_usage'] = psutil.virtual_memory().percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            health['disk_usage'] = (disk.used / disk.total) * 100
            
            # Calculate error rate from recent history
            recent_errors = [m for m in self.metrics_history[-50:] if m.get('error_rate', 0) > 0]
            health['error_rate'] = len(recent_errors) / 50 if self.metrics_history else 0
            
            # Overall health score (weighted average)
            health['overall_score'] = (
                (100 - health['cpu_usage']) * 0.3 +
                (100 - health['memory_usage']) * 0.3 +
                (100 - health['disk_usage']) * 0.2 +
                (1 - health['error_rate']) * 0.2
            ) / 100
            
        except Exception as e:
            logger.error(f"Error assessing system health: {e}")
        
        return health
    
    def generate_report(self, knowledge_graph=None, evolution_data=None):
        """
        Generate comprehensive self-assessment report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'learning_metrics': {},
            'knowledge_metrics': {},
            'health_metrics': {},
            'recommendations': []
        }
        
        # Get learning metrics
        if evolution_data:
            report['learning_metrics'] = self.assess_learning_effectiveness(evolution_data)
        
        # Get knowledge metrics
        if knowledge_graph:
            report['knowledge_metrics'] = self.assess_knowledge_quality(knowledge_graph)
        
        # Get health metrics
        report['health_metrics'] = self.assess_system_health()
        
        # Generate summary
        lm = report['learning_metrics']
        km = report['knowledge_metrics']
        hm = report['health_metrics']
        
        report['summary'] = {
            'learning_quality': 'HIGH' if lm.get('improvement_rate', 0) > 0.7 else 'MEDIUM' if lm.get('improvement_rate', 0) > 0.3 else 'LOW',
            'knowledge_maturity': 'HIGH' if km.get('node_count', 0) > 100 else 'MEDIUM' if km.get('node_count', 0) > 20 else 'LOW',
            'system_health': 'GOOD' if hm.get('overall_score', 0) > 0.7 else 'FAIR' if hm.get('overall_score', 0) > 0.4 else 'POOR',
            'overall_progress': (lm.get('improvement_rate', 0) + km.get('connectivity', 0) + hm.get('overall_score', 0)) / 3
        }
        
        # Generate recommendations
        if lm.get('improvement_rate', 0) < 0.3:
            report['recommendations'].append("🔧 Low improvement rate - Consider adjusting mutation strategies")
        
        if km.get('node_count', 0) < 20:
            report['recommendations'].append("📚 Knowledge graph too small - Focus on adding more concepts")
        
        if hm.get('overall_score', 0) < 0.5:
            report['recommendations'].append("⚕️ System health declining - Check resources and error logs")
        
        if lm.get('exploration_efficiency', 0) < 0.1:
            report['recommendations'].append("🎯 Inefficient exploration - Reduce search space or improve targeting")
        
        # Save report
        report_file = self.data_path / f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Assessment report generated - Learning: {report['summary']['learning_quality']}, Health: {report['summary']['system_health']}")
        
        return report
    
    def get_learning_curve(self, metric='improvement_rate', window=10):
        """Get learning curve data for visualization"""
        if not self.metrics_history:
            return []
        
        values = [m.get(metric, 0) for m in self.metrics_history]
        
        # Calculate moving average
        moving_avg = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            moving_avg.append(np.mean(values[start:i+1]))
        
        return {
            'raw': values,
            'moving_average': moving_avg,
            'generations': [m.get('generation', i) for i, m in enumerate(self.metrics_history)]
        }
    
    def get_status(self):
        """Get current assessment status"""
        return {
            'total_assessments': len(self.metrics_history),
            'last_assessment': self.metrics_history[-1] if self.metrics_history else None,
            'learning_curve': self.get_learning_curve(),
            'recommendations': self.generate_recommendations()
        }
    
    def generate_recommendations(self):
        """Generate actionable recommendations based on metrics"""
        recs = []
        
        if self.metrics_history:
            recent = self.metrics_history[-5:]
            avg_improvement = np.mean([m.get('improvement_rate', 0) for m in recent])
            
            if avg_improvement < 0.2:
                recs.append({
                    'priority': 'HIGH',
                    'area': 'learning',
                    'message': 'Learning rate is very low - increase mutation diversity'
                })
            elif avg_improvement > 0.8:
                recs.append({
                    'priority': 'LOW',
                    'area': 'learning',
                    'message': 'Learning rate is excellent - maintain current strategy'
                })
            
            # Check stability
            if len(recent) > 1:
                variance = np.var([m.get('stability_score', 0) for m in recent])
                if variance > 0.3:
                    recs.append({
                        'priority': 'MEDIUM',
                        'area': 'stability',
                        'message': 'High variance in performance - consider stabilizing'
                    })
        
        return recs

if __name__ == "__main__":
    # Test the self-assessment system
    sa = SelfAssessment()
    
    # Simulate some evolution data
    test_data = {
        'generation': 5,
        'files_processed': 150,
        'improvements': 45,
        'success_rate': 0.45,
        'mutations': [
            {'type': 'add_comment', 'success': True},
            {'type': 'optimize_loop', 'success': False},
            {'type': 'add_error_handling', 'success': True},
        ],
        'best_score': 1.26,
        'target_scores': 10.0
    }
    
    # Generate report
    report = sa.generate_report(evolution_data=test_data)
    print("\n📊 Self-Assessment Report:")
    print(json.dumps(report, indent=2))
    
    # Show learning curve
    print("\n📈 Learning Curve:")
    curve = sa.get_learning_curve()
    print(f"Recent values: {curve['raw'][-5:] if curve['raw'] else 'None'}")
    
    print("\n💡 Recommendations:")
    for rec in sa.generate_recommendations():
        print(f"  [{rec['priority']}] {rec['message']}")
