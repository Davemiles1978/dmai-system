import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import importlib.util
import sys
from dataclasses import dataclass, asdict
import traceback
import hashlib

from knowledge_graph import KnowledgeGraph
from meta_learner import MetaLearner
from self_healer import SelfHealer
from data_validator import DataValidator
from capability_synthesizer import CapabilitySynthesizer

@dataclass
class AGIState:
    """Current state of the AGI system"""
    generation: int
    active_capabilities: List[str]
    current_goals: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    learning_rate: float
    exploration_rate: float
    last_evolution: str  # Changed to str for JSON serialization
    health_status: str

class AGIOrchestrator:
    """Orchestrates all AGI components for recursive self-improvement"""
    
    def __init__(self, base_path: str = "shared_data/agi_evolution"):
        self.base_path = Path(base_path)
        self.state_path = self.base_path / "orchestrator_state"
        self.evolution_path = self.base_path / "evolution_history"
        self.checkpoint_path = Path("shared_checkpoints")
        
        # Create directories
        for path in [self.state_path, self.evolution_path]:
            path.mkdir(parents=True, exist_ok=True)
            
        # Initialize all AGI components
        self.knowledge_graph = KnowledgeGraph()
        self.meta_learner = MetaLearner()
        self.self_healer = SelfHealer()
        self.data_validator = DataValidator()
        self.capability_synthesizer = CapabilitySynthesizer()
        
        # State management
        self.state = self._load_state()
        self.evolution_queue = asyncio.Queue()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.capability_cache: Dict[str, Any] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'capability_synthesized': [],
            'evolution_step': [],
            'state_changed': [],
            'error_occurred': [],
            'checkpoint_created': []
        }
        
    def _load_state(self) -> AGIState:
        """Load orchestrator state"""
        state_file = self.state_path / "current_state.json"
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                data = json.load(f)
                return AGIState(**data)
        else:
            # Initialize new state
            return AGIState(
                generation=4,  # Current generation from shared_checkpoints
                active_capabilities=[],
                current_goals=[],
                performance_metrics={},
                learning_rate=0.1,
                exploration_rate=0.3,
                last_evolution=datetime.now().isoformat(),
                health_status="initializing"
            )
            
    async def start(self):
        """Start the AGI orchestrator"""
        print("🚀 Starting AGI Orchestrator...")
        
        # Register with self-healer
        self.self_healer.register_component("agi_orchestrator", self)
        
        # Load capabilities
        await self._load_active_capabilities()
        
        # Start evolution loop
        asyncio.create_task(self._evolution_loop())
        
        # Start health monitoring
        asyncio.create_task(self._health_monitor())
        
        # Start goal processing
        asyncio.create_task(self._goal_processor())
        
        self.state.health_status = "running"
        self._save_state()
        
        print(f"✅ AGI Orchestrator started at Generation {self.state.generation}")
        
    async def _load_active_capabilities(self):
        """Load and cache active capabilities"""
        caps_dir = Path("shared_data/agi_evolution/capabilities")
        if caps_dir.exists():
            for cap_file in caps_dir.glob("*.json"):
                with open(cap_file, 'r') as f:
                    cap_data = json.load(f)
                    if cap_data.get('name'):
                        self.state.active_capabilities.append(cap_data['name'])
                        
    async def _evolution_loop(self):
        """Main evolution loop for recursive self-improvement"""
        while True:
            try:
                # Check if evolution is needed
                if await self._should_evolve():
                    await self._perform_evolution_step()
                    
                # Process evolution queue
                while not self.evolution_queue.empty():
                    evolution_task = await self.evolution_queue.get()
                    result = await self._process_evolution_task(evolution_task)
                    self.evolution_queue.task_done()
                    await self._process_evolution_task(evolution_task)
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                await self._handle_error("evolution_loop", e)
                await asyncio.sleep(300)  # Wait longer on error
                
    async def _should_evolve(self) -> bool:
        """Determine if evolution should occur"""
        # Check time since last evolution
        if isinstance(self.state.last_evolution, str):
            last_evo = datetime.fromisoformat(self.state.last_evolution)
        else:
            last_evo = self.state.last_evolution
            
        time_since_evolution = (datetime.now() - last_evo).seconds
        
        if time_since_evolution < 3600:  # Minimum 1 hour between evolutions
            return False
            
        # Check performance metrics
        if self.state.performance_metrics:
            avg_performance = sum(self.state.performance_metrics.values()) / len(self.state.performance_metrics)
            if avg_performance < 0.7:  # Low performance triggers evolution
                return True
                
        # Check exploration rate
        if self.state.exploration_rate > 0.2:
            return True
            
        return False
        
    async def _perform_evolution_step(self):
        """Perform a single evolution step"""
        print(f"\n🔄 Starting Evolution Step (Generation {self.state.generation})...")
        
        evolution_record = {
            'timestamp': datetime.now().isoformat(),
            'generation': self.state.generation,
            'actions': []
        }
        
        try:
            # 1. Analyze current capabilities
            capability_analysis = await self._analyze_capabilities()
            evolution_record['actions'].append({'type': 'analysis', 'result': capability_analysis})
            
            # 2. Identify improvement opportunities
            opportunities = await self._identify_improvements(capability_analysis)
            evolution_record['actions'].append({'type': 'opportunities', 'result': opportunities})
            
            # 3. Synthesize new capabilities
            for opportunity in opportunities[:3]:  # Limit to top 3
                new_capability = await self.capability_synthesizer.synthesize_new_capability(
                    goal=opportunity['goal'],
                    available_capabilities=self.state.active_capabilities,
                    context=opportunity.get('context', {})
                )
                
                if new_capability:
                    evolution_record['actions'].append({
                        'type': 'synthesis',
                        'capability': new_capability.name
                    })
                    
                    # Trigger event
                    await self._trigger_event('capability_synthesized', {
                        'capability': new_capability.name,
                        'generation': self.state.generation
                    })
                    
            # 4. Update meta-learner
            meta_updates = await self.meta_learner.learn_from_evolution(evolution_record)
            evolution_record['actions'].append({'type': 'meta_learning', 'result': meta_updates})
            
            # 5. Update state
            self.state.generation += 1
            self.state.last_evolution = datetime.now().isoformat()
            self.state.learning_rate = self._adjust_learning_rate()
            self.state.exploration_rate = self._adjust_exploration_rate()
            
            # 6. Create checkpoint
            checkpoint_id = await self._create_evolution_checkpoint(evolution_record)
            evolution_record['checkpoint'] = checkpoint_id
            
            # 7. Save evolution record
            await self._save_evolution_record(evolution_record)
            
            # 8. Trigger event
            await self._trigger_event('evolution_step', {
                'generation': self.state.generation,
                'record': evolution_record
            })
            
            print(f"✅ Evolution Step Complete - Now at Generation {self.state.generation}")
            
        except Exception as e:
            await self._handle_error("evolution_step", e)
            
    async def _analyze_capabilities(self) -> Dict[str, Any]:
        """Analyze current capability set"""
        analysis = {
            'total_capabilities': len(self.state.active_capabilities),
            'capability_network': self.capability_synthesizer.get_capability_network(),
            'performance_by_capability': {},
            'gaps': []
        }
        
        # Get performance metrics from knowledge graph
        for capability in self.state.active_capabilities:
            perf = await self._get_capability_performance(capability)
            if perf:
                analysis['performance_by_capability'][capability] = perf
                
        # Identify capability gaps
        analysis['gaps'] = await self._identify_capability_gaps()
        
        return analysis
        
    async def _identify_improvements(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify improvement opportunities"""
        opportunities = []
        
        # Look for poorly performing capabilities
        for cap_name, perf in analysis.get('performance_by_capability', {}).items():
            if perf.get('success_rate', 1.0) < 0.6:
                opportunities.append({
                    'type': 'improve_existing',
                    'goal': f'Improve {cap_name} performance',
                    'capability': cap_name,
                    'context': {'current_performance': perf}
                })
                
        # Look for capability gaps
        for gap in analysis.get('gaps', []):
            opportunities.append({
                'type': 'fill_gap',
                'goal': gap['description'],
                'context': gap.get('context', {})
            })
            
        # Look for composition opportunities
        if len(self.state.active_capabilities) >= 3:
            opportunities.append({
                'type': 'compose',
                'goal': 'Create composite capability from existing ones',
                'context': {'capabilities': self.state.active_capabilities[:3]}
            })
            
        return opportunities
        
    async def _identify_capability_gaps(self) -> List[Dict[str, Any]]:
        """Identify missing capabilities"""
        gaps = []
        
        # Get capability network
        network = self.capability_synthesizer.get_capability_network()
        
        # Check for disconnected components
        if len(network.get('root_capabilities', [])) > 1:
            gaps.append({
                'type': 'integration',
                'description': 'Integrate disconnected capability clusters',
                'context': {'roots': network['root_capabilities']}
            })
            
        # Check for missing common patterns
        common_patterns = ['error_handling', 'validation', 'logging']
        for pattern in common_patterns:
            if not any(pattern in cap for cap in self.state.active_capabilities):
                gaps.append({
                    'type': 'pattern',
                    'description': f'Add {pattern} capability',
                    'context': {'pattern': pattern}
                })
                
        return gaps
        
    async def _process_evolution_task(self, task: Dict[str, Any]):
        """Process a specific evolution task"""
        task_type = task.get('type')
        
        if task_type == 'synthesize':
            result = await self.capability_synthesizer.synthesize_new_capability(
                goal=task['goal'],
                available_capabilities=self.state.active_capabilities,
                context=task.get('context', {})
            )
            if result:
                self.state.active_capabilities.append(result.name)
                
        elif task_type == 'optimize':
            result = await self.meta_learner.optimize_capability(
                capability=task['capability'],
                metrics=task.get('metrics', {})
            )
            
        elif task_type == 'validate':
            result = await self.data_validator.validate_capability_data(
                capability=task['capability'],
                data=task.get('data', {})
            )
            
        # Update task result
        task['result'] = result
        task['completed'] = datetime.now().isoformat()
        
    async def _get_capability_performance(self, capability: str) -> Dict[str, float]:
        """Get performance metrics for a capability"""
        # Query knowledge graph for performance data
        perf_data = await self.knowledge_graph.query(
            f"MATCH (c:Capability {{name: '{capability}'}})-[:HAS_PERFORMANCE]->(p) RETURN p"
        )
        
        if perf_data:
            return perf_data[0].get('p', {})
        return {}
        
    def _adjust_learning_rate(self) -> float:
        """Adjust learning rate based on evolution success"""
        # Increase learning rate if evolution is successful
        if hasattr(self, '_evolution_success_rate'):
            if self._evolution_success_rate > 0.7:
                return min(0.3, self.state.learning_rate * 1.1)
            else:
                return max(0.01, self.state.learning_rate * 0.9)
        return self.state.learning_rate
        
    def _adjust_exploration_rate(self) -> float:
        """Adjust exploration rate based on time"""
        # Decrease exploration over time, but never to zero
        time_factor = 0.99 ** (self.state.generation - 4)
        return max(0.05, 0.3 * time_factor)
        
    async def _create_evolution_checkpoint(self, evolution_record: Dict[str, Any]) -> str:
        """Create a checkpoint after evolution"""
        checkpoint_id = f"gen_{self.state.generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_dir = self.checkpoint_path / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save evolution record
        with open(checkpoint_dir / "evolution_record.json", 'w') as f:
            json.dump(evolution_record, f, indent=2, default=str)
            
        # Save current state
        with open(checkpoint_dir / "orchestrator_state.json", 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
            
        # Save capability network
        network = self.capability_synthesizer.get_capability_network()
        with open(checkpoint_dir / "capability_network.json", 'w') as f:
            json.dump(network, f, indent=2)
            
        # Trigger event
        await self._trigger_event('checkpoint_created', {
            'checkpoint_id': checkpoint_id,
            'generation': self.state.generation
        })
        
        return checkpoint_id
        
    async def _save_evolution_record(self, record: Dict[str, Any]):
        """Save evolution record to history"""
        record_file = self.evolution_path / f"evolution_gen_{self.state.generation}.json"
        with open(record_file, 'w') as f:
            json.dump(record, f, indent=2, default=str)
            
    async def _health_monitor(self):
        """Monitor system health"""
        while True:
            try:
                # Check component health
                components_health = {
                    'knowledge_graph': self.knowledge_graph.is_healthy() if hasattr(self.knowledge_graph, 'is_healthy') else True,
                    'meta_learner': self.meta_learner.is_healthy() if hasattr(self.meta_learner, 'is_healthy') else True,
                    'self_healer': self.self_healer.is_healthy() if hasattr(self.self_healer, 'is_healthy') else True,
                    'data_validator': self.data_validator.is_healthy() if hasattr(self.data_validator, 'is_healthy') else True,
                    'capability_synthesizer': True
                }
                
                # Update state
                if all(components_health.values()):
                    self.state.health_status = "healthy"
                else:
                    self.state.health_status = "degraded"
                    unhealthy = [k for k, v in components_health.items() if not v]
                    print(f"⚠️ Unhealthy components: {unhealthy}")
                    
                # Save state
                self._save_state()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Health monitor error: {e}")
                await asyncio.sleep(60)
                
    async def _goal_processor(self):
        """Process and prioritize goals"""
        while True:
            try:
                if self.state.current_goals:
                    # Sort goals by priority
                    sorted_goals = sorted(
                        self.state.current_goals,
                        key=lambda g: g.get('priority', 0),
                        reverse=True
                    )
                    
                    # Process top goal
                    top_goal = sorted_goals[0]
                    
                    # Check if goal requires evolution
                    if top_goal.get('requires_evolution', False):
                        await self.evolution_queue.put({
                            'type': 'goal_driven_evolution',
                            'goal': top_goal
                        })
                        
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                await self._handle_error("goal_processor", e)
                await asyncio.sleep(30)
                
    async def submit_goal(self, goal: Dict[str, Any]):
        """Submit a new goal to the system"""
        goal['submitted'] = datetime.now().isoformat()
        goal['id'] = hashlib.md5(json.dumps(goal).encode()).hexdigest()[:8]
        
        self.state.current_goals.append(goal)
        self._save_state()
        
        print(f"📋 New goal submitted: {goal.get('description', 'Unknown')}")
        
    async def execute_capability(self, 
                                capability_name: str, 
                                input_data: Any,
                                context: Optional[Dict] = None) -> Any:
        """Execute a capability by name"""
        try:
            # Check cache first
            if capability_name in self.capability_cache:
                module = self.capability_cache[capability_name]
            else:
                # Load capability module
                cap_path = Path(f"shared_data/agi_evolution/capabilities/{capability_name}.py")
                if not cap_path.exists():
                    raise ValueError(f"Capability {capability_name} not found")
                    
                spec = importlib.util.spec_from_file_location(capability_name, cap_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.capability_cache[capability_name] = module
                
            # Execute
            result = await module.execute(input_data, context)
            
            # Update metrics
            await self._update_capability_metrics(capability_name, success=True)
            
            return result
            
        except Exception as e:
            await self._update_capability_metrics(capability_name, success=False)
            raise
            
    async def _update_capability_metrics(self, capability_name: str, success: bool):
        """Update capability performance metrics"""
        # Update in knowledge graph
        await self.knowledge_graph.update_performance(
            capability_name,
            {'success': success, 'timestamp': datetime.now().isoformat()}
        )
        
    async def _handle_error(self, context: str, error: Exception):
        """Handle errors in the system"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'error': str(error),
            'traceback': traceback.format_exc()
        }
        
        print(f"❌ Error in {context}: {error}")
        
        # Log error
        error_file = self.state_path / "errors.json"
        errors = []
        if error_file.exists():
            with open(error_file, 'r') as f:
                errors = json.load(f)
                
        errors.append(error_info)
        
        # Keep only last 100 errors
        if len(errors) > 100:
            errors = errors[-100:]
            
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
            
        # Trigger event
        await self._trigger_event('error_occurred', error_info)
        
        # Attempt self-healing
        if hasattr(self.self_healer, 'heal_component'):
            await self.self_healer.heal_component(context, error_info)
        
    def on(self, event: str, handler: Callable):
        """Register event handler"""
        if event in self.event_handlers:
            self.event_handlers[event].append(handler)
            
    async def _trigger_event(self, event: str, data: Any):
        """Trigger an event"""
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    print(f"Error in event handler for {event}: {e}")
                    
    def _save_state(self):
        """Save orchestrator state"""
        state_file = self.state_path / "current_state.json"
        with open(state_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
            
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'state': asdict(self.state),
            'active_capabilities': len(self.state.active_capabilities),
            'pending_goals': len(self.state.current_goals),
            'evolution_queue_size': self.evolution_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'components': {
                'knowledge_graph': 'healthy',
                'meta_learner': 'healthy',
                'self_healer': 'healthy',
                'data_validator': 'healthy',
                'capability_synthesizer': 'healthy'
            }
        }

from self_assessment import SelfAssessment

class AGIOrchestrator:
    def __init__(self, base_path: str = "shared_data/agi_evolution"):
        # ... existing code ...
        
        # Add self-assessment
        self.self_assessment = SelfAssessment()
        
        # ... rest of init ...

    async def _perform_evolution_step(self):
        """Perform a single evolution step with self-assessment"""
        print(f"\n🔄 Starting Evolution Step (Generation {self.state.generation})...")
        
        evolution_record = {
            'timestamp': datetime.now().isoformat(),
            'generation': self.state.generation,
            'actions': []
        }
        
        try:
            # ... existing evolution code ...
            
            # 4. Update meta-learner
            meta_updates = await self.meta_learner.learn_from_evolution(evolution_record)
            evolution_record['actions'].append({'type': 'meta_learning', 'result': meta_updates})
            
            # 5. Run self-assessment
            assessment = self.self_assessment.generate_report(
                knowledge_graph=self.knowledge_graph,
                evolution_data=evolution_record
            )
            evolution_record['assessment'] = assessment
            
            # Log assessment summary
            print(f"\n📊 Self-Assessment Summary:")
            print(f"  Learning Quality: {assessment['summary']['learning_quality']}")
            print(f"  Knowledge Maturity: {assessment['summary']['knowledge_maturity']}")
            print(f"  System Health: {assessment['summary']['system_health']}")
            
            if assessment['recommendations']:
                print(f"\n💡 Recommendations:")
                for rec in assessment['recommendations']:
                    print(f"  • {rec}")
            
            # 6. Update state based on assessment
            self.state.learning_rate = self._adjust_learning_rate(assessment)
            self.state.exploration_rate = self._adjust_exploration_rate(assessment)
            
            # ... rest of evolution step ...
            
        except Exception as e:
            await self._handle_error("evolution_step", e)
    
    def _adjust_learning_rate(self, assessment=None):
        """Adjust learning rate based on self-assessment"""
        if assessment and assessment.get('learning_metrics'):
            lm = assessment['learning_metrics']
            if lm.get('improvement_rate', 0) > 0.7:
                return min(0.3, self.state.learning_rate * 1.1)
            elif lm.get('improvement_rate', 0) < 0.3:
                return max(0.01, self.state.learning_rate * 0.9)
        return self.state.learning_rate
    
    def _adjust_exploration_rate(self, assessment=None):
        """Adjust exploration rate based on self-assessment"""
        if assessment and assessment.get('learning_metrics'):
            lm = assessment['learning_metrics']
            if lm.get('stability_score', 0) > 0.8:
                return max(0.05, self.state.exploration_rate * 0.95)
            elif lm.get('stability_score', 0) < 0.3:
                return min(0.5, self.state.exploration_rate * 1.1)
        return max(0.05, 0.3 * (0.99 ** (self.state.generation - 4)))
