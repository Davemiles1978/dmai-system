#!/usr/bin/env python3

import random
import json
from itertools import permutations

class EvolutionRandomizer:
    def __init__(self):
        self.evaluators = [
            'gemini',
            'grok',
            'gpt',
            'claude',
            'deepseek_v4',
            'qwen3',
            'kimi_k2',
            'kimi_dev',
            'llama4_scout',
            'alpamayo',
            'nova',
            'mistral',
            'dmai_core'
        ]
        self.targets = self.evaluators.copy()
    
    def generate_pairs(self, ensure_full_coverage=True):
        evaluators = self.evaluators.copy()
        targets = self.targets.copy()
        
        random.shuffle(evaluators)
        random.shuffle(targets)
        
        pairs = []
        
        if ensure_full_coverage:
            for i, target in enumerate(targets):
                available = [e for e in evaluators if e != target]
                if not available:
                    continue
                
                evaluator = random.choice(available)
                pairs.append({
                    'evaluator': evaluator,
                    'target': target,
                    'timestamp': None
                })
                
                if evaluator in evaluators:
                    evaluators.remove(evaluator)
        else:
            num_pairs = random.randint(5, 8)
            used_pairs = set()
            
            for _ in range(num_pairs):
                attempts = 0
                while attempts < 20:
                    evaluator = random.choice(evaluators)
                    target = random.choice([t for t in targets if t != evaluator])
                    pair_key = f"{evaluator}-{target}"
                    
                    if pair_key not in used_pairs:
                        used_pairs.add(pair_key)
                        pairs.append({
                            'evaluator': evaluator,
                            'target': target
                        })
                        break
                    attempts += 1
        
        return pairs
    
    def get_evaluation_pool(self):
        pool = []
        for name in self.evaluators:
            if name == 'dmai_core':
                path = "/Users/davidmiles/Desktop/AI-Evolution-System/core/dmai_core.py"
            else:
                path = f"/Users/davidmiles/Desktop/AI-Evolution-System/evolution/evaluators/{name}_evaluator.py"
            
            pool.append({
                'name': name,
                'path': path,
                'enabled': True
            })
        return pool
    
    def should_run_external(self, external_weight=0.6):
        return random.random() < external_weight
    
    def get_random_evaluator_for_target(self, target_name):
        available = [e for e in self.evaluators if e != target_name]
        if not available:
            return None
        return random.choice(available)
    
    def generate_full_cycle_plan(self):
        plan = {
            'evaluations': self.generate_pairs(ensure_full_coverage=True),
            'run_external': self.should_run_external(),
            'timestamp': None,
            'cycle_type': 'evolution'
        }
        return plan
    
    def get_evolution_stats(self):
        return {
            'total_evaluators': len(self.evaluators),
            'total_targets': len(self.targets),
            'unique_perspectives': len(set(self.evaluators)),
            'can_self_evaluate': False,
            'external_research_weight': 0.6
        }

randomizer = EvolutionRandomizer()

if __name__ == "__main__":
    print("🎲 DMAI EVOLUTION RANDOMIZER")
    print("============================")
    print(f"\nTotal AI systems: {len(randomizer.evaluators)}")
    
    tier1 = ['gemini', 'grok', 'gpt']
    tier2 = ['claude', 'deepseek_v4', 'qwen3', 'kimi_k2', 'kimi_dev']
    tier3 = ['llama4_scout', 'alpamayo', 'nova', 'mistral']
    
    print("\nTier 1 - Core Evaluators:")
    for e in tier1:
        print(f"  • {e}")
    
    print("\nTier 2 - Essential Additions:")
    for e in tier2:
        print(f"  • {e}")
    
    print("\nTier 3 - Specialized:")
    for e in tier3:
        print(f"  • {e}")
    
    print("\nDMAI Core:")
    print(f"  • dmai_core")
    
    print("\nSAMPLE EVOLUTION CYCLE:")
    pairs = randomizer.generate_pairs()
    for p in pairs:
        print(f"  {p['evaluator']} → {p['target']}")
    
    print(f"\nRun external research this cycle: {randomizer.should_run_external()}")
    print(f"\nEvolution Stats: {randomizer.get_evolution_stats()}")

    def run_provider_check_with_merge(self):
        """Check providers and merge updates with internal versions"""
        print("\n🔍 Checking for provider updates to merge...")
        
        try:
            from evolution.provider_checker import provider_checker
            
            # Check and process all updates through merger
            processed = provider_checker.check_and_process_all_updates()
            
            if processed:
                print(f"\n✅ Processed {len(processed)} provider updates:")
                for p in processed:
                    print(f"  • {p['provider']}: v{p['from']} → v{p['to']}")
                    print(f"    Merged: {p['merged_path']}")
                
                # Check for versions ready for promotion
                from evolution.version_merger import version_merger
                for p in processed:
                    if version_merger.should_replace_internal(p['provider'], p['merged_path']):
                        print(f"\n   ⚠️  {p['provider']} ready for promotion!")
                        print(f"      Run: scripts/promote_merged_version.sh")
            else:
                print("   No new provider updates to merge")
            
            return processed
        except Exception as e:
            print(f"   Error checking provider updates: {e}")
            return []
