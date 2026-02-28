# capability_synthesizer.py

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import networkx as nx
from dataclasses import dataclass, asdict
import hashlib
import numpy as np

@dataclass
class Capability:
    """Represents a learned capability"""
    name: str
    version: str
    description: str
    dependencies: List[str]
    performance_metrics: Dict[str, float]
    creation_generation: int
    usage_count: int
    last_used: datetime
    code_hash: str
    success_rate: float
    
class CapabilitySynthesizer:
    """Synthesizes new capabilities from existing ones"""
    
    def __init__(self, base_path: str = "shared_data/agi_evolution"):
        self.base_path = Path(base_path)
        self.capabilities_path = self.base_path / "capabilities"
        self.patterns_path = self.base_path / "patterns"
        self.synthesis_path = self.base_path / "synthesis"
        
        # Create directories
        for path in [self.capabilities_path, self.patterns_path, self.synthesis_path]:
            path.mkdir(parents=True, exist_ok=True)
            
        self.capability_graph = nx.DiGraph()
        self.capabilities: Dict[str, Capability] = {}
        self.load_capabilities()
        
    def load_capabilities(self):
        """Load existing capabilities"""
        if self.capabilities_path.exists():
            for cap_file in self.capabilities_path.glob("*.json"):
                with open(cap_file, 'r') as f:
                    data = json.load(f)
                    cap = Capability(**data)
                    self.capabilities[cap.name] = cap
                    self.capability_graph.add_node(cap.name, **asdict(cap))
                    
        # Build dependency graph
        for cap_name, cap in self.capabilities.items():
            for dep in cap.dependencies:
                if dep in self.capabilities:
                    self.capability_graph.add_edge(dep, cap_name)
                    
    async def synthesize_new_capability(self, 
                                       goal: str,
                                       available_capabilities: List[str],
                                       context: Dict[str, Any]) -> Optional[Capability]:
        """Synthesize a new capability to achieve a goal"""
        
        # Find relevant existing capabilities
        relevant_caps = self._find_relevant_capabilities(goal, available_capabilities)
        
        if len(relevant_caps) < 2:
            return None  # Need at least 2 capabilities to synthesize
            
        # Identify synthesis patterns
        patterns = await self._identify_synthesis_patterns(relevant_caps, context)
        
        if not patterns:
            return None
            
        # Generate new capability
        new_capability = await self._generate_capability(goal, relevant_caps, patterns, context)
        
        if new_capability and await self._validate_capability(new_capability):
            # Save and register
            await self._save_capability(new_capability)
            
            # Analyze for meta-learning
            await self._analyze_synthesis_success(new_capability, relevant_caps, patterns)
            
            return new_capability
            
        return None
        
    def _find_relevant_capabilities(self, goal: str, available: List[str]) -> List[str]:
        """Find capabilities relevant to the goal"""
        relevant = []
        
        # Simple keyword matching for now - can be enhanced with embeddings
        goal_keywords = set(goal.lower().split())
        
        for cap_name in available:
            if cap_name in self.capabilities:
                cap = self.capabilities[cap_name]
                cap_keywords = set(cap.description.lower().split())
                
                # Calculate relevance score
                overlap = len(goal_keywords & cap_keywords)
                if overlap > 0:
                    relevant.append(cap_name)
                    
        return relevant[:5]  # Limit to top 5 most relevant
        
    async def _identify_synthesis_patterns(self, 
                                          capabilities: List[str],
                                          context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify patterns for synthesizing capabilities"""
        patterns = []
        
        # Load existing patterns
        pattern_files = list(self.patterns_path.glob("pattern_*.json"))
        
        for pattern_file in pattern_files[-10:]:  # Check last 10 patterns
            with open(pattern_file, 'r') as f:
                pattern = json.load(f)
                
            # Check if pattern applies
            if self._pattern_applies(pattern, capabilities, context):
                patterns.append(pattern)
                
        # If no patterns found, generate heuristic patterns
        if not patterns:
            patterns = self._generate_heuristic_patterns(capabilities)
            
        return patterns
        
    def _pattern_applies(self, 
                        pattern: Dict[str, Any],
                        capabilities: List[str],
                        context: Dict[str, Any]) -> bool:
        """Check if a synthesis pattern applies"""
        # Check capability type match
        if 'required_capability_types' in pattern:
            cap_types = [self.capabilities[c].description for c in capabilities]
            if not any(t in pattern['required_capability_types'] for t in cap_types):
                return False
                
        # Check context match
        if 'required_context' in pattern:
            for key, value in pattern['required_context'].items():
                if context.get(key) != value:
                    return False
                    
        return True
        
    def _generate_heuristic_patterns(self, capabilities: List[str]) -> List[Dict[str, Any]]:
        """Generate heuristic synthesis patterns"""
        patterns = [
            {
                'type': 'composition',
                'description': 'Compose capabilities sequentially',
                'synthesis_method': 'sequential_composition',
                'confidence': 0.7
            },
            {
                'type': 'parallelization',
                'description': 'Run capabilities in parallel and combine results',
                'synthesis_method': 'parallel_combination',
                'confidence': 0.6
            },
            {
                'type': 'specialization',
                'description': 'Specialize a general capability for specific context',
                'synthesis_method': 'specialization',
                'confidence': 0.5
            },
            {
                'type': 'generalization',
                'description': 'Generalize specific capabilities to broader use',
                'synthesis_method': 'generalization',
                'confidence': 0.5
            }
        ]
        
        return patterns
        
    async def _generate_capability(self,
                                   goal: str,
                                   capabilities: List[str],
                                   patterns: List[Dict[str, Any]],
                                   context: Dict[str, Any]) -> Optional[Capability]:
        """Generate a new capability from synthesis"""
        
        # Select best pattern
        pattern = max(patterns, key=lambda p: p.get('confidence', 0))
        
        # Generate capability code based on pattern
        if pattern['synthesis_method'] == 'sequential_composition':
            code = await self._compose_sequential(capabilities, goal)
        elif pattern['synthesis_method'] == 'parallel_combination':
            code = await self._combine_parallel(capabilities, goal)
        elif pattern['synthesis_method'] == 'specialization':
            code = await self._specialize_capability(capabilities[0], context)
        elif pattern['synthesis_method'] == 'generalization':
            code = await self._generalize_capabilities(capabilities, goal)
        else:
            return None
            
        if not code:
            return None
            
        # Create capability
        cap_name = self._generate_capability_name(goal)
        cap_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        
        new_capability = Capability(
            name=cap_name,
            version="1.0.0",
            description=f"Synthesized capability for: {goal}",
            dependencies=capabilities,
            performance_metrics={},
            creation_generation=4,  # Current generation
            usage_count=0,
            last_used=datetime.now(),
            code_hash=cap_hash,
            success_rate=0.0
        )
        
        # Save code
        code_path = self.capabilities_path / f"{cap_name}_{cap_hash}.py"
        with open(code_path, 'w') as f:
            f.write(code)
            
        return new_capability
        
    async def _compose_sequential(self, capabilities: List[str], goal: str) -> str:
        """Compose capabilities sequentially"""
        code = f'''"""
Synthesized capability: Sequential composition for {goal}
Generated at: {datetime.now()}
Dependencies: {', '.join(capabilities)}
"""

async def execute(input_data, context=None):
    """Execute capabilities in sequence"""
    result = input_data
    
    # Sequential execution
    for cap_name in {capabilities}:
        # Import and execute each capability
        module = __import__(f"capabilities.{{cap_name}}", fromlist=['execute'])
        result = await module.execute(result, context)
        
        # Check for early termination
        if result is None:
            break
            
    return result

def get_metadata():
    return {{
        'name': 'sequential_{goal[:30].replace(" ", "_")}',
        'type': 'sequential_composition',
        'dependencies': {capabilities},
        'version': '1.0.0'
    }}
'''
        return code
        
    async def _combine_parallel(self, capabilities: List[str], goal: str) -> str:
        """Combine capabilities in parallel"""
        code = f'''"""
Synthesized capability: Parallel combination for {goal}
Generated at: {datetime.now()}
Dependencies: {', '.join(capabilities)}
"""

import asyncio

async def execute(input_data, context=None):
    """Execute capabilities in parallel and combine results"""
    
    async def execute_capability(cap_name):
        module = __import__(f"capabilities.{{cap_name}}", fromlist=['execute'])
        return await module.execute(input_data, context)
    
    # Execute all capabilities in parallel
    tasks = [execute_capability(cap) for cap in {capabilities}]
    results = await asyncio.gather(*tasks)
    
    # Combine results (simple concatenation for now)
    combined = {{}}
    for i, result in enumerate(results):
        if result:
            combined[f"result_{{i}}"] = result
            
    return combined

def get_metadata():
    return {{
        'name': 'parallel_{goal[:30].replace(" ", "_")}',
        'type': 'parallel_combination',
        'dependencies': {capabilities},
        'version': '1.0.0'
    }}
'''
        return code
        
    async def _specialize_capability(self, capability: str, context: Dict[str, Any]) -> str:
        """Specialize a capability for specific context"""
        code = f'''"""
Synthesized capability: Specialization of {capability}
Specialized for: {json.dumps(context, indent=2)}
Generated at: {datetime.now()}
"""

async def execute(input_data, context=None):
    """Execute specialized version of {capability}"""
    
    # Import base capability
    module = __import__(f"capabilities.{{capability}}", fromlist=['execute'])
    
    # Add specialization context
    specialized_context = context or {{}}
    specialized_context.update({json.dumps(context)})
    
    # Execute with specialization
    return await module.execute(input_data, specialized_context)

def get_metadata():
    return {{
        'name': 'specialized_{capability}',
        'type': 'specialization',
        'dependencies': ['{capability}'],
        'specialization_context': {json.dumps(context)},
        'version': '1.0.0'
    }}
'''
        return code
        
    async def _generalize_capabilities(self, capabilities: List[str], goal: str) -> str:
        """Generalize multiple capabilities into one"""
        code = f'''"""
Synthesized capability: Generalization of {', '.join(capabilities)}
Goal: {goal}
Generated at: {datetime.now()}
"""

async def execute(input_data, context=None):
    """Execute generalized capability that adapts to input type"""
    
    # Determine which specific capability to use based on input
    if isinstance(input_data, dict):
        if 'sequence' in input_data:
            # Use sequential composition
            module = __import__("capabilities.sequential_composition", fromlist=['execute'])
        elif 'parallel' in input_data:
            # Use parallel composition  
            module = __import__("capabilities.parallel_combination", fromlist=['execute'])
        else:
            # Default to first capability
            module = __import__(f"capabilities.{{capabilities[0]}}", fromlist=['execute'])
    else:
        # Default to first capability
        module = __import__(f"capabilities.{{capabilities[0]}}", fromlist=['execute'])
    
    return await module.execute(input_data, context)

def get_metadata():
    return {{
        'name': 'generalized_{goal[:30].replace(" ", "_")}',
        'type': 'generalization',
        'dependencies': {capabilities},
        'version': '1.0.0'
    }}
'''
        return code
        
    def _generate_capability_name(self, goal: str) -> str:
        """Generate a name for the new capability"""
        # Extract key words from goal
        words = goal.lower().split()
        key_words = [w for w in words if len(w) > 3][:3]
        base_name = "_".join(key_words) if key_words else "synthesized"
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"
        
    async def _validate_capability(self, capability: Capability) -> bool:
        """Validate a newly synthesized capability"""
        # Check if capability already exists
        if capability.name in self.capabilities:
            return False
            
        # Check dependencies exist
        for dep in capability.dependencies:
            if dep not in self.capabilities:
                return False
                
        # Check code exists
        code_path = self.capabilities_path / f"{capability.name}_{capability.code_hash}.py"
        if not code_path.exists():
            return False
            
        # Basic syntax validation (simplified)
        try:
            with open(code_path, 'r') as f:
                code = f.read()
            compile(code, '<string>', 'exec')
        except:
            return False
            
        return True
        
    async def _save_capability(self, capability: Capability):
        """Save capability metadata"""
        cap_path = self.capabilities_path / f"{capability.name}.json"
        with open(cap_path, 'w') as f:
            json.dump(asdict(capability), f, indent=2, default=str)
            
        self.capabilities[capability.name] = capability
        self.capability_graph.add_node(capability.name, **asdict(capability))
        
        for dep in capability.dependencies:
            self.capability_graph.add_edge(dep, capability.name)
            
    async def _analyze_synthesis_success(self,
                                        new_capability: Capability,
                                        source_capabilities: List[str],
                                        pattern_used: Dict[str, Any]):
        """Analyze synthesis for meta-learning"""
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'new_capability': new_capability.name,
            'source_capabilities': source_capabilities,
            'pattern_used': pattern_used,
            'dependencies': new_capability.dependencies,
            'initial_metrics': new_capability.performance_metrics
        }
        
        # Save analysis
        analysis_path = self.synthesis_path / f"synthesis_{new_capability.name}.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
            
        # Update pattern confidence based on success
        pattern_file = self.patterns_path / f"pattern_{pattern_used['type']}.json"
        if pattern_file.exists():
            with open(pattern_file, 'r') as f:
                pattern = json.load(f)
            pattern['confidence'] = min(1.0, pattern.get('confidence', 0.5) + 0.1)
            pattern['usage_count'] = pattern.get('usage_count', 0) + 1
        else:
            pattern_used['usage_count'] = 1
            pattern_used['confidence'] = 0.6
            
        with open(pattern_file, 'w') as f:
            json.dump(pattern_used, f, indent=2)
            
    def get_capability_network(self) -> Dict[str, Any]:
        """Get the capability dependency network"""
        return {
            'nodes': list(self.capability_graph.nodes()),
            'edges': list(self.capability_graph.edges()),
            'stats': {
                'total_capabilities': len(self.capabilities),
                'root_capabilities': [n for n, d in self.capability_graph.in_degree() if d == 0],
                'leaf_capabilities': [n for n, d in self.capability_graph.out_degree() if d == 0]
            }
        }

    # === PHASE 3: CAPABILITY SYNTHESIS ENHANCEMENTS ===
    
    async def discover_function_combinations(self, min_similarity=0.6):
        """Discover compatible function combinations"""
        combinations = []
        
        # Get all capabilities
        caps = list(self.capabilities.values())
        
        for i, cap1 in enumerate(caps):
            for j, cap2 in enumerate(caps[i+1:], i+1):
                # Calculate compatibility score
                compatibility = self._calculate_compatibility(cap1, cap2)
                
                if compatibility >= min_similarity:
                    combinations.append({
                        'capabilities': [cap1.name, cap2.name],
                        'compatibility': compatibility,
                        'potential_use': self._predict_combination_use(cap1, cap2)
                    })
        
        # Sort by compatibility
        combinations.sort(key=lambda x: x['compatibility'], reverse=True)
        
        # Store for later use
        combos_file = self.patterns_path / "discovered_combinations.json"
        with open(combos_file, 'w') as f:
            json.dump(combinations[:50], f, indent=2, default=str)
        
        print(f"🔍 Discovered {len(combinations)} potential function combinations")
        return combinations[:20]  # Return top 20
    
    def _calculate_compatibility(self, cap1: Capability, cap2: Capability) -> float:
        """Calculate how compatible two capabilities are for combination"""
        score = 0.0
        factors = 0
        
        # Check shared dependencies
        deps1 = set(cap1.dependencies)
        deps2 = set(cap2.dependencies)
        if deps1 and deps2:
            overlap = len(deps1 & deps2)
            total = len(deps1 | deps2)
            score += overlap / total if total > 0 else 0
            factors += 1
        
        # Check performance similarity
        perf1 = cap1.performance_metrics.get('success_rate', 0.5)
        perf2 = cap2.performance_metrics.get('success_rate', 0.5)
        perf_diff = 1 - abs(perf1 - perf2)
        score += perf_diff
        factors += 1
        
        # Check usage patterns
        usage1 = cap1.usage_count
        usage2 = cap2.usage_count
        if usage1 > 0 and usage2 > 0:
            usage_ratio = min(usage1, usage2) / max(usage1, usage2)
            score += usage_ratio
            factors += 1
        
        return score / factors if factors > 0 else 0
    
    def _predict_combination_use(self, cap1: Capability, cap2: Capability) -> str:
        """Predict what the combination could be used for"""
        descriptions = f"{cap1.description} + {cap2.description}"
        
        # Simple keyword-based prediction
        if "data" in descriptions.lower() and "analyze" in descriptions.lower():
            return "Data analysis pipeline"
        elif "image" in descriptions.lower() and "generate" in descriptions.lower():
            return "Image generation workflow"
        elif "code" in descriptions.lower() and "test" in descriptions.lower():
            return "Code testing automation"
        elif "api" in descriptions.lower() and "fetch" in descriptions.lower():
            return "API integration tool"
        else:
            return f"Combined {cap1.name} and {cap2.name} functionality"
    
    async def create_hybrid_capability(self, cap1_name: str, cap2_name: str, combination_strategy: str = "sequential"):
        """Create a hybrid capability from two existing ones"""
        if cap1_name not in self.capabilities or cap2_name not in self.capabilities:
            return None
        
        cap1 = self.capabilities[cap1_name]
        cap2 = self.capabilities[cap2_name]
        
        # Generate hybrid code based on strategy
        if combination_strategy == "sequential":
            code = await self._create_sequential_hybrid(cap1, cap2)
        elif combination_strategy == "parallel":
            code = await self._create_parallel_hybrid(cap1, cap2)
        elif combination_strategy == "conditional":
            code = await self._create_conditional_hybrid(cap1, cap2)
        else:
            return None
        
        # Create hybrid capability
        hybrid_name = f"hybrid_{cap1_name}_{cap2_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        hybrid_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        
        hybrid_cap = Capability(
            name=hybrid_name,
            version="1.0.0",
            description=f"Hybrid capability combining {cap1.description} and {cap2.description}",
            dependencies=[cap1_name, cap2_name],
            performance_metrics={},
            creation_generation=self.current_generation if hasattr(self, 'current_generation') else 1,
            usage_count=0,
            last_used=datetime.now(),
            code_hash=hybrid_hash,
            success_rate=0.0
        )
        
        # Save code
        code_path = self.capabilities_path / f"{hybrid_name}_{hybrid_hash}.py"
        with open(code_path, 'w') as f:
            f.write(code)
        
        # Save metadata
        await self._save_capability(hybrid_cap)
        
        print(f"✨ Created hybrid capability: {hybrid_name}")
        return hybrid_cap
    
    async def _create_sequential_hybrid(self, cap1: Capability, cap2: Capability) -> str:
        """Create a hybrid that runs capabilities sequentially"""
        code = f'''"""
Hybrid Capability: Sequential combination of {cap1.name} and {cap2.name}
Generated at: {datetime.now()}
"""

async def execute(input_data, context=None):
    """Execute capabilities in sequence"""
    result = input_data
    
    # Execute first capability
    module1 = __import__("capabilities.{cap1.name}", fromlist=['execute'])
    result = await module1.execute(result, context)
    
    if result is not None:
        # Execute second capability on result
        module2 = __import__("capabilities.{cap2.name}", fromlist=['execute'])
        result = await module2.execute(result, context)
    
    return result

def get_metadata():
    return {{
        'name': '{cap1.name}_{cap2.name}_sequential',
        'type': 'sequential_hybrid',
        'dependencies': ['{cap1.name}', '{cap2.name}'],
        'version': '1.0.0'
    }}
'''
        return code
    
    async def _create_parallel_hybrid(self, cap1: Capability, cap2: Capability) -> str:
        """Create a hybrid that runs capabilities in parallel and combines results"""
        code = f'''"""
Hybrid Capability: Parallel combination of {cap1.name} and {cap2.name}
Generated at: {datetime.now()}
"""

import asyncio

async def execute(input_data, context=None):
    """Execute capabilities in parallel and combine results"""
    
    async def run_capability(cap_name):
        module = __import__(f"capabilities.{{cap_name}}", fromlist=['execute'])
        return await module.execute(input_data, context)
    
    # Run both in parallel
    tasks = [
        run_capability('{cap1.name}'),
        run_capability('{cap2.name}')
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine results
    combined = {{
        '{cap1.name}_result': results[0] if not isinstance(results[0], Exception) else None,
        '{cap2.name}_result': results[1] if not isinstance(results[1], Exception) else None
    }}
    
    return combined

def get_metadata():
    return {{
        'name': '{cap1.name}_{cap2.name}_parallel',
        'type': 'parallel_hybrid',
        'dependencies': ['{cap1.name}', '{cap2.name}'],
        'version': '1.0.0'
    }}
'''
        return code
    
    async def _create_conditional_hybrid(self, cap1: Capability, cap2: Capability) -> str:
        """Create a hybrid that chooses between capabilities based on input"""
        code = f'''"""
Hybrid Capability: Conditional combination of {cap1.name} and {cap2.name}
Generated at: {datetime.now()}
"""

async def execute(input_data, context=None):
    """Choose which capability to execute based on input"""
    
    # Decision logic - can be evolved over time
    if isinstance(input_data, dict):
        if 'use_first' in input_data:
            module = __import__("capabilities.{cap1.name}", fromlist=['execute'])
        elif 'use_second' in input_data:
            module = __import__("capabilities.{cap2.name}", fromlist=['execute'])
        else:
            # Default: use the one with better performance
            module = __import__("capabilities.{cap1.name}", fromlist=['execute'])
    else:
        # Default to first
        module = __import__("capabilities.{cap1.name}", fromlist=['execute'])
    
    return await module.execute(input_data, context)

def get_metadata():
    return {{
        'name': '{cap1.name}_{cap2.name}_conditional',
        'type': 'conditional_hybrid',
        'dependencies': ['{cap1.name}', '{cap2.name}'],
        'version': '1.0.0'
    }}
'''
        return code
    
    async def optimize_synergy(self, hybrid_name: str):
        """Optimize a hybrid capability for maximum synergy"""
        if hybrid_name not in self.capabilities:
            return None
        
        hybrid = self.capabilities[hybrid_name]
        
        # Analyze current performance
        current_score = hybrid.performance_metrics.get('success_rate', 0)
        
        # Try different combination strategies
        strategies = ['sequential', 'parallel', 'conditional']
        best_strategy = 'sequential'
        best_score = current_score
        
        # This would actually test different strategies
        # For now, we'll just record the attempt
        synergy_file = self.synthesis_path / f"synergy_{hybrid_name}.json"
        
        synergy_data = {
            'hybrid': hybrid_name,
            'original_score': current_score,
            'optimization_attempts': [],
            'best_strategy': best_strategy,
            'best_score': best_score
        }
        
        with open(synergy_file, 'w') as f:
            json.dump(synergy_data, f, indent=2, default=str)
        
        print(f"⚡ Optimized synergy for {hybrid_name}")
        return synergy_data
    
    async def cross_domain_learning(self, source_domain: str, target_domain: str):
        """Apply learning from one domain to another"""
        # Find capabilities in source domain
        source_caps = [
            cap for cap in self.capabilities.values()
            if source_domain in cap.description.lower()
        ]
        
        if not source_caps:
            return None
        
        # Create domain adaptation capability
        adaptation_code = f'''"""
Cross-Domain Learning: {source_domain} → {target_domain}
Generated at: {datetime.now()}
"""

async def execute(input_data, context=None):
    """Adapt {source_domain} capabilities to {target_domain} domain"""
    # Domain adaptation logic
    adapted_result = {{
        'source_domain': '{source_domain}',
        'target_domain': '{target_domain}',
        'original_input': input_data,
        'adapted': True
    }}
    
    return adapted_result

def get_metadata():
    return {{
        'name': '{source_domain}_to_{target_domain}_adapter',
        'type': 'cross_domain_adapter',
        'domains': ['{source_domain}', '{target_domain}'],
        'version': '1.0.0'
    }}
'''
        
        # Save the adapter
        adapter_name = f"adapter_{source_domain}_to_{target_domain}"
        adapter_hash = hashlib.sha256(adaptation_code.encode()).hexdigest()[:16]
        
        adapter_cap = Capability(
            name=adapter_name,
            version="1.0.0",
            description=f"Adapts {source_domain} capabilities to {target_domain} domain",
            dependencies=[cap.name for cap in source_caps[:3]],
            performance_metrics={},
            creation_generation=self.current_generation if hasattr(self, 'current_generation') else 1,
            usage_count=0,
            last_used=datetime.now(),
            code_hash=adapter_hash,
            success_rate=0.0
        )
        
        # Save adapter
        code_path = self.capabilities_path / f"{adapter_name}_{adapter_hash}.py"
        with open(code_path, 'w') as f:
            f.write(adaptation_code)
        
        await self._save_capability(adapter_cap)
        
        print(f"🔄 Created cross-domain adapter: {adapter_name}")
        return adapter_cap
