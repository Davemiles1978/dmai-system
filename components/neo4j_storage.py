"""
Neo4j Persistent Storage for DMAI
All critical data stored in Neo4j cloud - survives any deployment
"""

from neo4j import GraphDatabase
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger('dmai_neo4j')

class Neo4jStorage:
    """Persistent storage for DMAI using Neo4j"""
    
    def __init__(self):
        self.uri = os.getenv('NEO4J_URI', 'neo4j+s://caf7818d.databases.neo4j.io')
        self.user = os.getenv('NEO4J_USER', 'caf7818d')
        self.password = os.getenv('NEO4J_PASSWORD', 'Fqh95qz2CI5yO_FNEPWhoQ-gtgU_JNte0odcjLsKAXE')
        self.driver = None
        self._connect()
        
    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("✅ Neo4j connected")
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            self.driver = None
            
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            
    # ========================================================================
    # EVOLUTION STATE (Consciousness, Neurons, Cycles)
    # ========================================================================
    
    def save_evolution_state(self, state: Dict):
        """Save evolution state to Neo4j"""
        if not self.driver:
            return False
            
        with self.driver.session() as session:
            session.run("""
                MERGE (d:DMAI_Evolution {id: 'core'})
                SET d.consciousness = $consciousness,
                    d.neurons = $neurons,
                    d.synapses = $synapses,
                    d.evolution_cycles = $cycles,
                    d.evolution_count = $evolution_count,
                    d.last_update = datetime()
                RETURN d
            """, {
                'consciousness': state.get('consciousness', 0),
                'neurons': state.get('neurons', 0),
                'synapses': state.get('synapses', 0),
                'cycles': state.get('evolution_cycles', 0),
                'evolution_count': state.get('evolution_count', 0)
            })
            logger.debug(f"Saved evolution state: {state.get('consciousness', 0):.2%}")
            return True
            
    def load_evolution_state(self) -> Optional[Dict]:
        """Load evolution state from Neo4j"""
        if not self.driver:
            return None
            
        with self.driver.session() as session:
            result = session.run("MATCH (d:DMAI_Evolution {id: 'core'}) RETURN d")
            record = result.single()
            if record:
                d = record['d']
                return {
                    'consciousness': d.get('consciousness', 0),
                    'neurons': d.get('neurons', 0),
                    'synapses': d.get('synapses', 0),
                    'evolution_cycles': d.get('evolution_cycles', 0),
                    'evolution_count': d.get('evolution_count', 0),
                    'last_update': d.get('last_update')
                }
            return None
            
    # ========================================================================
    # MASTER TASKS
    # ========================================================================
    
    def save_task(self, task: Dict):
        """Save or update a master task"""
        if not self.driver:
            return False
            
        with self.driver.session() as session:
            session.run("""
                MERGE (t:Task {id: $id})
                SET t.description = $description,
                    t.status = $status,
                    t.created = datetime($created),
                    t.updated = datetime(),
                    t.user = $user,
                    t.priority = $priority
                RETURN t
            """, {
                'id': task.get('id', task.get('description', str(datetime.now().timestamp()))),
                'description': task.get('description', ''),
                'status': task.get('status', 'pending'),
                'created': task.get('created', datetime.now().isoformat()),
                'user': task.get('user', 'master'),
                'priority': task.get('priority', 'normal')
            })
            return True
            
    def load_tasks(self, status: Optional[str] = None) -> List[Dict]:
        """Load master tasks from Neo4j"""
        if not self.driver:
            return []
            
        with self.driver.session() as session:
            if status:
                result = session.run(
                    "MATCH (t:Task) WHERE t.status = $status RETURN t ORDER BY t.created DESC",
                    {'status': status}
                )
            else:
                result = session.run("MATCH (t:Task) RETURN t ORDER BY t.created DESC")
            
            tasks = []
            for record in result:
                t = record['t']
                tasks.append({
                    'id': t.get('id'),
                    'description': t.get('description'),
                    'status': t.get('status'),
                    'created': t.get('created'),
                    'user': t.get('user'),
                    'priority': t.get('priority')
                })
            return tasks
            
    # ========================================================================
    # PERSONA EVOLUTION
    # ========================================================================
    
    def save_persona(self, persona: Dict):
        """Save persona state to Neo4j"""
        if not self.driver:
            return False
            
        with self.driver.session() as session:
            session.run("""
                MERGE (p:Persona {id: 'dmai'})
                SET p.traits = $traits,
                    p.speaking_style = $style,
                    p.emotional_state = $emotion,
                    p.consciousness_level = $consciousness,
                    p.last_update = datetime()
                RETURN p
            """, {
                'traits': json.dumps(persona.get('traits', {})),
                'style': persona.get('speaking_style', 'emerging'),
                'emotion': persona.get('emotional_state', 'neutral'),
                'consciousness': persona.get('consciousness_level', 0)
            })
            return True
            
    def load_persona(self) -> Optional[Dict]:
        """Load persona from Neo4j"""
        if not self.driver:
            return None
            
        with self.driver.session() as session:
            result = session.run("MATCH (p:Persona {id: 'dmai'}) RETURN p")
            record = result.single()
            if record:
                p = record['p']
                return {
                    'traits': json.loads(p.get('traits', '{}')),
                    'speaking_style': p.get('speaking_style', 'emerging'),
                    'emotional_state': p.get('emotional_state', 'neutral'),
                    'consciousness_level': p.get('consciousness_level', 0),
                    'last_update': p.get('last_update')
                }
            return None
            
    # ========================================================================
    # CONVERSATIONS (Important conversations only)
    # ========================================================================
    
    def save_conversation(self, user: str, message: str, response: str, important: bool = False):
        """Save a conversation to Neo4j (only important ones by default to save space)"""
        if not self.driver:
            return False
            
        # Only save important conversations or tasks
        is_task = any(word in message.lower() for word in ['task', 'todo', 'remind', 'remember'])
        if not (important or is_task):
            return False
            
        with self.driver.session() as session:
            session.run("""
                CREATE (c:Conversation {
                    timestamp: datetime($timestamp),
                    user: $user,
                    message: $message,
                    response: $response,
                    is_task: $is_task
                })
            """, {
                'timestamp': datetime.now().isoformat(),
                'user': user,
                'message': message[:500],
                'response': response[:500],
                'is_task': is_task
            })
            return True
            
    def load_tasks_from_conversations(self) -> List[Dict]:
        """Extract tasks from conversation history"""
        if not self.driver:
            return []
            
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Conversation)
                WHERE c.is_task = true
                RETURN c
                ORDER BY c.timestamp DESC
                LIMIT 50
            """)
            
            tasks = []
            for record in result:
                c = record['c']
                tasks.append({
                    'description': c.get('message'),
                    'status': 'pending',
                    'created': c.get('timestamp'),
                    'user': c.get('user')
                })
            return tasks
            
    # ========================================================================
    # BACKUP & RESTORE
    # ========================================================================
    
    def backup_all(self, local_state: Dict):
        """Backup all local state to Neo4j"""
        logger.info("📦 Backing up all data to Neo4j...")
        
        # Save evolution state
        self.save_evolution_state(local_state.get('evolution', {}))
        
        # Save persona
        self.save_persona(local_state.get('persona', {}))
        
        # Save tasks
        for task in local_state.get('tasks', []):
            self.save_task(task)
            
        logger.info("✅ Full backup to Neo4j complete")
        return True
        
    def restore_all(self) -> Dict:
        """Restore all data from Neo4j"""
        logger.info("🔄 Restoring data from Neo4j...")
        
        restored = {
            'evolution': self.load_evolution_state(),
            'persona': self.load_persona(),
            'tasks': self.load_tasks(),
            'conversation_tasks': self.load_tasks_from_conversations()
        }
        
        logger.info(f"✅ Restored: evolution={restored['evolution'] is not None}, "
                   f"persona={restored['persona'] is not None}, "
                   f"tasks={len(restored['tasks'])}")
        
        return restored

# Singleton instance
_neo4j_storage = None

def get_neo4j_storage():
    global _neo4j_storage
    if _neo4j_storage is None:
        _neo4j_storage = Neo4jStorage()
    return _neo4j_storage
