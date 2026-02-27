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
