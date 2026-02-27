"""
Meta-Learner - Analyzes evolution patterns to learn how to learn better
This is the core AGI component that enables recursive self-improvement
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from collections import defaultdict, Counter
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - META_LEARNER - %(message)s')

class MetaLearner:
    def __init__(self, knowledge_graph=None):
        self.kg = knowledge_graph
        self.models_path = Path("agi/models")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Learning patterns database
        self.learning_patterns = self.load_patterns()
        self.pattern_success_rates = defaultdict(list)
        
        # Evolution strategies
        self.strategies = {
            'mutation': self._analyze_mutation_patterns,
            'combination': self._analyze_combination_patterns,
            'pruning': self._analyze_pruning_patterns,
            'expansion': self._analyze_expansion_patterns
        }
        
        logging.info("🧠 Meta-Learner initialized")
    
    def is_healthy(self):
        """Check if meta-learner is healthy"""
        return True
    
    async def learn_from_evolution(self, evolution_record):
        """Learn from evolution records - wrapper for analyze_evolution_cycle"""
        logging.info(f"📚 Learning from evolution generation {evolution_record.get('generation', 'unknown')}")
        return self.analyze_evolution_cycle(evolution_record)
    
    def load_patterns(self):
        """Load learned patterns from disk"""
        pattern_file = self.models_path / "learning_patterns.json"
        if pattern_file.exists():
            with open(pattern_file, 'r') as f:
                return json.load(f)
        return {
            'successful_mutations': [],
            'failed_mutations': [],
            'effective_combinations': [],
            'learning_rates': [],
            'pattern_clusters': {}
        }
    
    def save_patterns(self):
        """Save learned patterns to disk"""
        pattern_file = self.models_path / "learning_patterns.json"
        with open(pattern_file, 'w') as f:
            json.dump(self.learning_patterns, f, indent=2)
        logging.info("💾 Learning patterns saved")
    
    def analyze_evolution_cycle(self, cycle_data):
        """
        Analyze a complete evolution cycle to extract learning patterns
        cycle_data: {
            'generation': int,
            'files_processed': int,
            'improvements': int,
            'best_scores': dict,
            'mutations': list of applied mutations,
            'success_rate': float
        }
        """
        patterns = {}
        
        # Run all strategy analyses
        for strategy_name, strategy_func in self.strategies.items():
            result = strategy_func(cycle_data)
            if result:
                patterns[strategy_name] = result
        
        # Store success rate
        self.pattern_success_rates[cycle_data['generation']] = cycle_data.get('success_rate', 0)
        
        # Update learning patterns
        self._update_patterns(patterns, cycle_data)
        
        return patterns
    
    def _analyze_mutation_patterns(self, cycle_data):
        """Analyze which mutation types were most successful"""
        mutations = cycle_data.get('mutations', [])
        if not mutations:
            return None
        
        # Group by mutation type
        type_success = defaultdict(list)
        for mutation in mutations:
            m_type = mutation.get('type', 'unknown')
            success = mutation.get('success', False)
            score_improvement = mutation.get('score_improvement', 0)
            
            type_success[m_type].append({
                'success': success,
                'improvement': score_improvement
            })
        
        # Calculate success rates per type
        results = {}
        for m_type, outcomes in type_success.items():
            successes = sum(1 for o in outcomes if o['success'])
            total = len(outcomes)
            avg_improvement = np.mean([o['improvement'] for o in outcomes]) if outcomes else 0
            
            results[m_type] = {
                'success_rate': successes / total if total > 0 else 0,
                'avg_improvement': avg_improvement,
                'total_applications': total
            }
        
        return results
    
    def _analyze_combination_patterns(self, cycle_data):
        """Analyze which function combinations work well together"""
        combinations = cycle_data.get('combinations', [])
        if not combinations:
            return None
        
        successful_combs = []
        for comb in combinations:
            if comb.get('success', False):
                successful_combs.append({
                    'functions': comb['functions'],
                    'result_score': comb['score'],
                    'context': comb.get('context', {})
                })
        
        return {
            'successful_combinations': successful_combs,
            'total_attempted': len(combinations),
            'success_rate': len(successful_combs) / len(combinations) if combinations else 0
        }
    
    def _analyze_pruning_patterns(self, cycle_data):
        """Analyze what code patterns are being removed/optimized"""
        pruned = cycle_data.get('pruned', [])
        if not pruned:
            return None
        
        # What patterns are commonly removed?
        removed_patterns = Counter()
        for item in pruned:
            pattern = item.get('pattern', 'unknown')
            removed_patterns[pattern] += 1
        
        return {
            'most_removed': removed_patterns.most_common(5),
            'total_pruned': len(pruned)
        }
    
    def _analyze_expansion_patterns(self, cycle_data):
        """Analyze what new capabilities are being added"""
        expanded = cycle_data.get('expanded', [])
        if not expanded:
            return None
        
        new_capabilities = []
        for exp in expanded:
            new_capabilities.append({
                'capability': exp['name'],
                'based_on': exp.get('based_on', []),
                'success': exp.get('success', False)
            })
        
        return {
            'new_capabilities': new_capabilities,
            'total_added': len(expanded)
        }
    
    def _update_patterns(self, new_patterns, cycle_data):
        """Update the learning patterns database"""
        generation = cycle_data['generation']
        
        # Store patterns by generation
        self.learning_patterns[f'gen_{generation}'] = {
            'patterns': new_patterns,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'files_processed': cycle_data.get('files_processed', 0),
                'improvements': cycle_data.get('improvements', 0),
                'success_rate': cycle_data.get('success_rate', 0)
            }
        }
        
        # Keep only last 100 generations
        if len(self.learning_patterns) > 100:
            oldest = min([k for k in self.learning_patterns.keys() if k.startswith('gen_')])
            del self.learning_patterns[oldest]
    
    def suggest_optimal_strategies(self, context):
        """
        Suggest the best evolution strategies for the current context
        context: {
            'code_type': str,
            'complexity': float,
            'previous_success': float,
            'available_mutations': list
        }
        """
        suggestions = []
        
        # Analyze historical success rates
        if self.pattern_success_rates:
            recent_rates = list(self.pattern_success_rates.values())[-10:]
            avg_success = np.mean(recent_rates) if recent_rates else 0
        else:
            avg_success = 0.5
        
        # Suggest based on context
        if context.get('complexity', 0) > 0.8:
            # Complex code - use conservative mutations
            suggestions.append({
                'strategy': 'conservative_mutation',
                'confidence': 0.8,
                'reason': 'High complexity code needs careful mutation'
            })
        elif context.get('previous_success', 0) < 0.3:
            # Low success rate - try exploratory mutations
            suggestions.append({
                'strategy': 'exploratory_mutation',
                'confidence': 0.7,
                'reason': 'Low success rate indicates need for exploration'
            })
        else:
            # Normal case - use balanced approach
            suggestions.append({
                'strategy': 'balanced_mutation',
                'confidence': 0.9,
                'reason': 'Standard evolution path'
            })
        
        # Add combination suggestions
        if context.get('available_mutations', []):
            suggestions.append({
                'strategy': 'combine_successful',
                'confidence': 0.75,
                'reason': 'Combine previously successful mutations',
                'mutations': context['available_mutations'][:3]
            })
        
        return sorted(suggestions, key=lambda x: x['confidence'], reverse=True)
    
    def predict_improvement(self, code_features):
        """
        Predict how much improvement a given code change might yield
        Uses learned patterns to estimate impact
        """
        # Simple model - will be enhanced with ML later
        base_prediction = 1.05  # 5% improvement baseline
        
        # Adjust based on historical patterns
        if self.learning_patterns.get('successful_mutations'):
            avg_success = np.mean([
                m.get('improvement', 1.0) 
                for m in self.learning_patterns['successful_mutations'][-20:]
            ])
            base_prediction = avg_success
        
        # Adjust for code complexity
        complexity_factor = 1.0 - (code_features.get('complexity', 0) * 0.1)
        
        # Adjust for novelty (new patterns might work better)
        novelty = code_features.get('novelty', 0)
        novelty_factor = 1.0 + (novelty * 0.05)
        
        prediction = base_prediction * complexity_factor * novelty_factor
        
        return {
            'predicted_improvement': prediction,
            'confidence': 0.6 + (novelty * 0.2),
            'factors': {
                'base': base_prediction,
                'complexity': complexity_factor,
                'novelty': novelty_factor
            }
        }
    
    def get_learning_stats(self):
        """Get statistics about the meta-learning process"""
        return {
            'total_patterns': len(self.learning_patterns),
            'generations_analyzed': len([k for k in self.learning_patterns.keys() if k.startswith('gen_')]),
            'avg_success_rate': np.mean(list(self.pattern_success_rates.values())) if self.pattern_success_rates else 0,
            'best_generation': max(self.pattern_success_rates.items(), key=lambda x: x[1])[0] if self.pattern_success_rates else None,
            'learning_curves': self._calculate_learning_curves()
        }
    
    def _calculate_learning_curves(self):
        """Calculate learning curves over time"""
        if not self.pattern_success_rates:
            return {}
        
        gens = sorted(self.pattern_success_rates.keys())
        rates = [self.pattern_success_rates[g] for g in gens]
        
        # Calculate moving average
        window = min(5, len(rates))
        moving_avg = []
        for i in range(len(rates) - window + 1):
            moving_avg.append(np.mean(rates[i:i+window]))
        
        return {
            'generations': gens,
            'success_rates': rates,
            'moving_average': moving_avg,
            'trend': 'up' if rates[-1] > rates[0] else 'down' if rates[-1] < rates[0] else 'stable'
        }

if __name__ == "__main__":
    # Test the meta-learner
    ml = MetaLearner()
    
    # Simulate some evolution cycles
    test_cycles = [
        {
            'generation': 1,
            'files_processed': 100,
            'improvements': 45,
            'success_rate': 0.45,
            'mutations': [
                {'type': 'add_comment', 'success': True, 'score_improvement': 0.1},
                {'type': 'optimize_loop', 'success': False, 'score_improvement': -0.05},
                {'type': 'add_error_handling', 'success': True, 'score_improvement': 0.15},
            ],
            'combinations': [
                {'functions': ['add_comment', 'add_error_handling'], 'success': True, 'score': 1.2},
            ],
            'pruned': [{'pattern': 'redundant_code'}, {'pattern': 'dead_code'}],
            'expanded': [{'name': 'error_recovery', 'based_on': ['error_handling'], 'success': True}]
        },
        {
            'generation': 2,
            'files_processed': 120,
            'improvements': 65,
            'success_rate': 0.54,
            'mutations': [
                {'type': 'add_comment', 'success': True, 'score_improvement': 0.12},
                {'type': 'optimize_loop', 'success': True, 'score_improvement': 0.08},
                {'type': 'refactor_names', 'success': True, 'score_improvement': 0.05},
            ],
            'combinations': [
                {'functions': ['optimize_loop', 'refactor_names'], 'success': True, 'score': 1.15},
            ],
            'pruned': [{'pattern': 'duplicate_code'}],
            'expanded': [{'name': 'auto_optimizer', 'based_on': ['optimize_loop'], 'success': True}]
        }
    ]
    
    print("📊 Analyzing evolution cycles...")
    for cycle in test_cycles:
        patterns = ml.analyze_evolution_cycle(cycle)
        print(f"\nGeneration {cycle['generation']} patterns:")
        print(json.dumps(patterns, indent=2))
    
    print("\n🎯 Strategy suggestions for new code:")
    context = {
        'code_type': 'evolution_engine',
        'complexity': 0.7,
        'previous_success': 0.5,
        'available_mutations': ['add_comment', 'optimize_loop', 'add_error_handling']
    }
    suggestions = ml.suggest_optimal_strategies(context)
    print(json.dumps(suggestions, indent=2))
    
    print("\n📈 Learning stats:")
    print(json.dumps(ml.get_learning_stats(), indent=2, default=str))
    
    # Save patterns
    ml.save_patterns()
