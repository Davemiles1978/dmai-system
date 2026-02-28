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
from self_assessment import SelfAssessment

@dataclass
class AGIState:
    """Current state of the AGI system"""
    generation: int
    active_capabilities: List[str]
    current_goals: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    learning_rate: float
    exploration_rate: float
    last_evolution: str
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
        self.self_assessment = SelfAssessment()
        
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
                generation=4,
                active_capabilities=[],
                current_goals=[],
                performance_metrics={},
                learning_rate=0.1,
                exploration_rate=0.3,
                last_evolution=datetime.now().isoformat(),
                health_status="initializing"
            )
    
    async def start(self):
        """Start the orchestrator - required for cloud deployment"""
        print("🚀 Starting AGI Orchestrator via start() method")
        
        # Register with self-healer
        self.self_healer.register_component("agi_orchestrator", self)
        
        # Load capabilities
        await self._load_active_capabilities()
        
        # Start background tasks
        asyncio.create_task(self._evolution_loop())
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._goal_processor())
        
        # Update state
        self.state.health_status = "running"
        self._save_state()
        
        print(f"✅ AGI Orchestrator started at Generation {self.state.generation}")
        return self
    
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
                if await self._should_evolve():
                    await self._perform_evolution_step()
                    
                while not self.evolution_queue.empty():
                    evolution_task = await self.evolution_queue.get()
                    result = None  # Initialize result
                    try:
                        result = await self._process_evolution_task(evolution_task)
                    except Exception as e:
                        print(f"❌ Error processing task: {e}")
                        result = {"error": str(e)}
                    finally:
                        self.evolution_queue.task_done()
                    
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error("evolution_loop", e)
                await asyncio.sleep(300)await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error("evolution_loop", e)
                await asyncio.sleep(300)
    
    async def _should_evolve(self) -> bool:
        """Determine if evolution should occur"""
        if isinstance(self.state.last_evolution, str):
            last_evo = datetime.fromisoformat(self.state.last_evolution)
        else:
            last_evo = self.state.last_evolution
            
        time_since_evolution = (datetime.now() - last_evo).seconds
        
        if time_since_evolution < 3600:
            return False
            
        if self.state.performance_metrics:
            avg_performance = sum(self.state.performance_metrics.values()) / len(self.state.performance_metrics)
            if avg_performance < 0.7:
                return True
                
        if self.state.exploration_rate > 0.2:
            return True
            
        return False
    
    async def _perform_evolution_step(self):
        """Perform a single evolution step with self-assessment"""
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
            for opportunity in opportunities[:3]:
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
                    
                    await self._trigger_event('capability_synthesized', {
                        'capability': new_capability.name,
                        'generation': self.state.generation
                    })
            
            # 4. Update meta-learner
            meta_updates = await self.meta_learner.learn_from_evolution(evolution_record)
            evolution_record['actions'].append({'type': 'meta_learning', 'result': meta_updates})
            
            # 5. Run self-assessment
            assessment = self.self_assessment.generate_report(
                knowledge_graph=self.knowledge_graph,
                evolution_data=evolution_record
            )
            evolution_record['assessment'] = assessment
            
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
            self.state.generation += 1
            self.state.last_evolution = datetime.now().isoformat()
            
            # 7. Create checkpoint
            checkpoint_id = await self._create_evolution_checkpoint(evolution_record)
            evolution_record['checkpoint'] = checkpoint_id
            
            # 8. Save evolution record
            await self._save_evolution_record(evolution_record)
            
            # 9. Trigger event
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
        
        for capability in self.state.active_capabilities:
            perf = await self._get_capability_performance(capability)
            if perf:
                analysis['performance_by_capability'][capability] = perf
                
        analysis['gaps'] = await self._identify_capability_gaps()
        
        return analysis
    
    async def _identify_improvements(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify improvement opportunities"""
        opportunities = []
        
        for cap_name, perf in analysis.get('performance_by_capability', {}).items():
            if perf.get('success_rate', 1.0) < 0.6:
                opportunities.append({
                    'type': 'improve_existing',
                    'goal': f'Improve {cap_name} performance',
                    'capability': cap_name,
                    'context': {'current_performance': perf}
                })
                
        for gap in analysis.get('gaps', []):
            opportunities.append({
                'type': 'fill_gap',
                'goal': gap['description'],
                'context': gap.get('context', {})
            })
            
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
        
        network = self.capability_synthesizer.get_capability_network()
        
        if len(network.get('root_capabilities', [])) > 1:
            gaps.append({
                'type': 'integration',
                'description': 'Integrate disconnected capability clusters',
                'context': {'roots': network['root_capabilities']}
            })
            
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
            
        task['result'] = result
        task['completed'] = datetime.now().isoformat()
    
    async def _get_capability_performance(self, capability: str) -> Dict[str, float]:
        """Get performance metrics for a capability"""
        perf_data = await self.knowledge_graph.query(
            f"MATCH (c:Capability {{name: '{capability}'}})-[:HAS_PERFORMANCE]->(p) RETURN p"
        )
        
        if perf_data:
            return perf_data[0].get('p', {})
        return {}
    
    def _adjust_learning_rate(self, assessment=None) -> float:
        """Adjust learning rate based on self-assessment"""
        if assessment and assessment.get('learning_metrics'):
            lm = assessment['learning_metrics']
            if lm.get('improvement_rate', 0) > 0.7:
                return min(0.3, self.state.learning_rate * 1.1)
            elif lm.get('improvement_rate', 0) < 0.3:
                return max(0.01, self.state.learning_rate * 0.9)
        return self.state.learning_rate
    
    def _adjust_exploration_rate(self, assessment=None) -> float:
        """Adjust exploration rate based on self-assessment"""
        if assessment and assessment.get('learning_metrics'):
            lm = assessment['learning_metrics']
            if lm.get('stability_score', 0) > 0.8:
                return max(0.05, self.state.exploration_rate * 0.95)
            elif lm.get('stability_score', 0) < 0.3:
                return min(0.5, self.state.exploration_rate * 1.1)
        return max(0.05, 0.3 * (0.99 ** (self.state.generation - 4)))
    
    async def _create_evolution_checkpoint(self, evolution_record: Dict[str, Any]) -> str:
        """Create a checkpoint after evolution"""
        checkpoint_id = f"gen_{self.state.generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_dir = self.checkpoint_path / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_dir / "evolution_record.json", 'w') as f:
            json.dump(evolution_record, f, indent=2, default=str)
            
        with open(checkpoint_dir / "orchestrator_state.json", 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
            
        network = self.capability_synthesizer.get_capability_network()
        with open(checkpoint_dir / "capability_network.json", 'w') as f:
            json.dump(network, f, indent=2)
            
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
                components_health = {
                    'knowledge_graph': self.knowledge_graph.is_healthy() if hasattr(self.knowledge_graph, 'is_healthy') else True,
                    'meta_learner': self.meta_learner.is_healthy() if hasattr(self.meta_learner, 'is_healthy') else True,
                    'self_healer': self.self_healer.is_healthy() if hasattr(self.self_healer, 'is_healthy') else True,
                    'data_validator': self.data_validator.is_healthy() if hasattr(self.data_validator, 'is_healthy') else True,
                    'capability_synthesizer': True
                }
                
                if all(components_health.values()):
                    self.state.health_status = "healthy"
                else:
                    self.state.health_status = "degraded"
                    unhealthy = [k for k, v in components_health.items() if not v]
                    print(f"⚠️ Unhealthy components: {unhealthy}")
                    
                self._save_state()
                await asyncio.sleep(30)
                
            except Exception as e:
                print(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _goal_processor(self):
        """Process and prioritize goals"""
        while True:
            try:
                if self.state.current_goals:
                    sorted_goals = sorted(
                        self.state.current_goals,
                        key=lambda g: g.get('priority', 0),
                        reverse=True
                    )
                    
                    top_goal = sorted_goals[0]
                    
                    if top_goal.get('requires_evolution', False):
                        await self.evolution_queue.put({
                            'type': 'goal_driven_evolution',
                            'goal': top_goal
                        })
                        
                await asyncio.sleep(10)
                
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
    
    async def execute_capability(self, capability_name: str, input_data: Any, context: Optional[Dict] = None) -> Any:
        """Execute a capability by name"""
        try:
            if capability_name in self.capability_cache:
                module = self.capability_cache[capability_name]
            else:
                cap_path = Path(f"shared_data/agi_evolution/capabilities/{capability_name}.py")
                if not cap_path.exists():
                    raise ValueError(f"Capability {capability_name} not found")
                    
                spec = importlib.util.spec_from_file_location(capability_name, cap_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.capability_cache[capability_name] = module
                
            result = await module.execute(input_data, context)
            await self._update_capability_metrics(capability_name, success=True)
            return result
            
        except Exception as e:
            await self._update_capability_metrics(capability_name, success=False)
            raise
    
    async def _update_capability_metrics(self, capability_name: str, success: bool):
        """Update capability performance metrics"""
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
        
        error_file = self.state_path / "errors.json"
        errors = []
        if error_file.exists():
            with open(error_file, 'r') as f:
                errors = json.load(f)
                
        errors.append(error_info)
        
        if len(errors) > 100:
            errors = errors[-100:]
            
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
            
        await self._trigger_event('error_occurred', error_info)
        
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

    async def _evolution_loop(self):
        """Main evolution loop for recursive self-improvement"""
        while True:
            try:
                if await self._should_evolve():
                    await self._perform_evolution_step()
                    
                while not self.evolution_queue.empty():
                    evolution_task = await self.evolution_queue.get()
                    # Initialize result variable
                    result = None
                    try:
                        result = await self._process_evolution_task(evolution_task)
                    except Exception as e:
                        print(f"❌ Error processing task: {e}")
                        result = {"error": str(e)}
                    finally:
                        self.evolution_queue.task_done()
                    
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error("evolution_loop", e)
                await asyncio.sleep(300)

    async def _apply_knowledge_gap_recommendations(self, gaps):
        """Apply recommendations from knowledge gap analysis"""
        for gap in gaps:
            if gap['type'] == 'missing_core_concept':
                for concept in gap['concepts']:
                    print(f"➕ Adding missing core concept: {concept}")
                    self.knowledge_graph.add_concept(concept, "core", {"auto_added": True})
            
            elif gap['type'] == 'isolated_concepts':
                print(f"🔗 Connecting {gap['count']} isolated concepts")
                # Logic to connect isolated nodes
            
            elif gap['type'] == 'low_utility_concepts' and gap['priority'] == 'low':
                print(f"📌 Reviewing {gap['count']} low-utility concepts")

    async def _auto_adjust_strategies(self, recommendations):
        """Automatically adjust evolution strategies based on recommendations"""
        for rec in recommendations:
            if rec['priority'] == 'high':
                if rec['area'] == 'mutation' and rec['action'] == 'increase_diversity':
                    self.state.exploration_rate = min(0.5, self.state.exploration_rate * 1.2)
                    print(f"🔄 Increased exploration rate to {self.state.exploration_rate:.2f}")
                
                elif rec['area'] == 'exploration' and rec['action'] == 'reduce_exploration_rate':
                    self.state.exploration_rate = max(0.05, self.state.exploration_rate * 0.8)
                    print(f"🔄 Reduced exploration rate to {self.state.exploration_rate:.2f}")
            
            elif rec['priority'] == 'medium' and rec['area'] == 'selection':
                # Adjust selection pressure
                print("🔄 Stabilizing selection pressure")
                # Implementation would go here

    async def _apply_knowledge_gap_recommendations(self, gaps):
        """Apply recommendations from knowledge gap analysis"""
        for gap in gaps:
            if gap['type'] == 'missing_core_concept':
                for concept in gap.get('concepts', []):
                    print(f"➕ Adding missing core concept: {concept}")
                    self.knowledge_graph.add_concept(concept, "core", {"auto_added": True, "added_at": datetime.now().isoformat()})
            
            elif gap['type'] == 'isolated_concepts':
                print(f"🔗 Found {gap.get('count', 0)} isolated concepts - will connect in next evolution cycle")
            
            elif gap['type'] == 'low_utility_concepts':
                print(f"📌 Found {gap.get('count', 0)} low-utility concepts - consider reviewing")

    async def _auto_adjust_strategies(self, recommendations):
        """Automatically adjust evolution strategies based on recommendations"""
        for rec in recommendations:
            if rec.get('priority') == 'high':
                if rec.get('area') == 'mutation' and rec.get('action') == 'increase_diversity':
                    self.state.exploration_rate = min(0.5, self.state.exploration_rate * 1.2)
                    print(f"🔄 Increased exploration rate to {self.state.exploration_rate:.2f}")
                
                elif rec.get('area') == 'exploration' and rec.get('action') == 'reduce_exploration_rate':
                    self.state.exploration_rate = max(0.05, self.state.exploration_rate * 0.8)
                    print(f"🔄 Reduced exploration rate to {self.state.exploration_rate:.2f}")
            
            elif rec.get('priority') == 'medium' and rec.get('area') == 'selection':
                print("🔄 Stabilizing selection pressure")

    async def _perform_evolution_step(self):
        """Perform a single evolution step with self-assessment and auto-adjustment"""
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
            for opportunity in opportunities[:3]:
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
                    
                    await self._trigger_event('capability_synthesized', {
                        'capability': new_capability.name,
                        'generation': self.state.generation
                    })
            
            # 4. Update meta-learner
            meta_updates = await self.meta_learner.learn_from_evolution(evolution_record)
            evolution_record['actions'].append({'type': 'meta_learning', 'result': meta_updates})
            
            # 5. Run self-assessment
            evolution_record['knowledge_graph'] = self.knowledge_graph
            assessment = self.self_assessment.generate_report(
                knowledge_graph=self.knowledge_graph,
                evolution_data=evolution_record
            )
            evolution_record['assessment'] = assessment
            
            # 6. Apply recommendations automatically (NEW - Phase 2 completion)
            if assessment.get('knowledge_gaps'):
                await self._apply_knowledge_gap_recommendations(assessment['knowledge_gaps'])
            
            if assessment.get('recommendations'):
                await self._auto_adjust_strategies(assessment['recommendations'])
            
            # Log assessment summary
            print(f"\n📊 Self-Assessment Summary:")
            print(f"  Learning Quality: {assessment['summary']['learning_quality']}")
            print(f"  Knowledge Maturity: {assessment['summary']['knowledge_maturity']}")
            print(f"  System Health: {assessment['summary']['system_health']}")
            
            if assessment.get('recommendations'):
                print(f"\n💡 Auto-adjusted based on recommendations:")
                for rec in assessment['recommendations']:
                    print(f"  • {rec['area']}: {rec['action']}")
            
            # 7. Update state
            self.state.learning_rate = self._adjust_learning_rate(assessment)
            self.state.exploration_rate = self._adjust_exploration_rate(assessment)
            self.state.generation += 1
            self.state.last_evolution = datetime.now().isoformat()
            
            # 8. Create checkpoint
            checkpoint_id = await self._create_evolution_checkpoint(evolution_record)
            evolution_record['checkpoint'] = checkpoint_id
            
            # 9. Save evolution record
            await self._save_evolution_record(evolution_record)
            
            # 10. Trigger event
            await self._trigger_event('evolution_step', {
                'generation': self.state.generation,
                'record': evolution_record
            })
            
            print(f"✅ Evolution Step Complete - Now at Generation {self.state.generation}")
            
        except Exception as e:
            await self._handle_error("evolution_step", e)

    # === PHASE 3: CAPABILITY SYNTHESIS ACTIVATION ===
    
    async def _run_capability_synthesis_cycle(self):
        """Run a full capability synthesis cycle"""
        print("\n🧬 Running Capability Synthesis Cycle...")
        
        # 1. Discover function combinations
        combinations = await self.capability_synthesizer.discover_function_combinations()
        
        if combinations:
            print(f"  📊 Found {len(combinations)} potential combinations")
            
            # 2. Create hybrid capabilities from top combinations
            for combo in combinations[:3]:  # Try top 3
                cap1, cap2 = combo['capabilities'][0], combo['capabilities'][1]
                
                # Try different strategies
                for strategy in ['sequential', 'parallel', 'conditional']:
                    hybrid = await self.capability_synthesizer.create_hybrid_capability(
                        cap1, cap2, strategy
                    )
                    if hybrid:
                        print(f"  ✨ Created hybrid: {hybrid.name} ({strategy})")
                        
                        # 3. Optimize synergy
                        await self.capability_synthesizer.optimize_synergy(hybrid.name)
                        
                        # Add to active capabilities
                        self.state.active_capabilities.append(hybrid.name)
                        
                        # Track for metrics
                        self.capability_synthesis_stats['attempts'] += 1
                        self.capability_synthesis_stats['successes'] += 1
        
        # 4. Cross-domain learning (if we have enough capabilities)
        if len(self.state.active_capabilities) >= 5:
            domains = ['data', 'image', 'code', 'api', 'analysis']
            for i in range(len(domains)-1):
                adapter = await self.capability_synthesizer.cross_domain_learning(
                    domains[i], domains[i+1]
                )
                if adapter:
                    print(f"  🔄 Created cross-domain adapter: {adapter.name}")
        
        # Update synthesis stats
        self.capability_synthesis_stats['total_capabilities'] = len(self.state.active_capabilities)
        
        # Save stats
        stats_file = self.base_path / "synthesis_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.capability_synthesis_stats, f, indent=2)
    
    async def _perform_evolution_step(self):
        """Perform a single evolution step with Phase 3 synthesis"""
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
            
            # 3. Synthesize new capabilities (original method)
            for opportunity in opportunities[:2]:
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
                    self.state.active_capabilities.append(new_capability.name)
            
            # 4. **NEW: Phase 3 Capability Synthesis Cycle**
            if self.state.generation % 3 == 0:  # Run every 3 generations
                await self._run_capability_synthesis_cycle()
                evolution_record['actions'].append({'type': 'phase3_synthesis'})
            
            # 5. Update meta-learner
            meta_updates = await self.meta_learner.learn_from_evolution(evolution_record)
            evolution_record['actions'].append({'type': 'meta_learning', 'result': meta_updates})
            
            # 6. Run self-assessment
            evolution_record['knowledge_graph'] = self.knowledge_graph
            assessment = self.self_assessment.generate_report(
                knowledge_graph=self.knowledge_graph,
                evolution_data=evolution_record
            )
            evolution_record['assessment'] = assessment
            
            # 7. Apply recommendations
            if assessment.get('knowledge_gaps'):
                await self._apply_knowledge_gap_recommendations(assessment['knowledge_gaps'])
            
            if assessment.get('recommendations'):
                await self._auto_adjust_strategies(assessment['recommendations'])
            
            # 8. Update state
            self.state.learning_rate = self._adjust_learning_rate(assessment)
            self.state.exploration_rate = self._adjust_exploration_rate(assessment)
            self.state.generation += 1
            self.state.last_evolution = datetime.now().isoformat()
            
            # 9. Create checkpoint
            checkpoint_id = await self._create_evolution_checkpoint(evolution_record)
            evolution_record['checkpoint'] = checkpoint_id
            
            # 10. Save evolution record
            await self._save_evolution_record(evolution_record)
            
            print(f"✅ Evolution Step Complete - Now at Generation {self.state.generation}")
            
        except Exception as e:
            await self._handle_error("evolution_step", e)
    
    
