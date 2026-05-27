"""Two-Tier Weighted Knowledge Manager - Core (100%) + Weighted (0-100%)"""
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class TwoTierKnowledgeManager:
    def __init__(self, db_path: str = "data/dmai_knowledge.db"):
        self.db_path = db_path
        self._init_tables()
        self._load_core_syllabus()
    
    def _init_tables(self):
        """Initialize two-tier knowledge tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # CORE KNOWLEDGE TIER (always 100% weight, never decays)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS core_knowledge (
                id TEXT PRIMARY KEY,
                topic TEXT UNIQUE,
                category TEXT,
                content TEXT,
                mastery_level REAL DEFAULT 1.0,
                last_reviewed TIMESTAMP,
                created_at TIMESTAMP,
                required_for_system BOOLEAN DEFAULT 1,
                metadata TEXT
            )
        ''')
        
        # WEIGHTED KNOWLEDGE TIER (decays over time)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weighted_knowledge (
                id TEXT PRIMARY KEY,
                topic TEXT UNIQUE,
                normalized_topic TEXT,
                content TEXT,
                weight REAL DEFAULT 0.1,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                created_at TIMESTAMP,
                source TEXT,
                confidence REAL DEFAULT 0.5,
                can_promote BOOLEAN DEFAULT 0,
                metadata TEXT
            )
        ''')
        
        # Indexes for fast lookup
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_core_topic ON core_knowledge(topic)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_weighted_topic ON weighted_knowledge(topic)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_weighted_weight ON weighted_knowledge(weight DESC)')
        
        conn.commit()
        conn.close()
        logger.info("✅ Two-Tier Knowledge Manager initialized")
    
    def _load_core_syllabus(self):
        """Load core syllabus topics that DMAI must master"""
        core_topics = [
            # AI & Machine Learning
            ("Neural Networks", "AI", "Deep learning architectures including CNNs, RNNs, Transformers, and attention mechanisms for pattern recognition"),
            ("Large Language Models", "AI", "GPT, Claude, Gemini architectures: training, fine-tuning, inference optimization, and emergent capabilities"),
            ("Reinforcement Learning", "AI", "Q-learning, policy gradients, PPO, and multi-agent systems for autonomous decision-making"),
            ("Computer Vision", "AI", "Convolutional networks, object detection, segmentation, and multimodal understanding"),
            
            # Software Development
            ("Python Programming", "Software", "Advanced Python: async, decorators, metaclasses, optimization, and system programming"),
            ("System Architecture", "Software", "Distributed systems, microservices, event-driven architecture, and scalability patterns"),
            ("API Design", "Software", "REST, GraphQL, gRPC, OpenAPI specs, versioning, and documentation strategies"),
            
            # Trading & Finance
            ("Algorithmic Trading", "Trading", "Market making, arbitrage, momentum strategies, risk management, and backtesting"),
            ("Technical Analysis", "Trading", "Indicators, chart patterns, volume analysis, and algorithmic implementation"),
            ("Risk Management", "Trading", "Position sizing, stop-losses, portfolio optimization, and drawdown management"),
            
            # Consciousness & AGI
            ("Synthetic Intelligence", "AGI", "Self-awareness, metacognition, recursive self-improvement, and artificial consciousness"),
            ("Knowledge Representation", "AGI", "Semantic networks, embeddings, graphs, and memory systems for AGI"),
            ("Autonomous Learning", "AGI", "Self-directed curriculum, gap analysis, and continuous improvement loops"),
            
            # DMAI System Specific
            ("DMAI Evolution Engine", "System", "Self-modification, capability addition, evolution cycles, and success metrics"),
            ("SI Core Architecture", "System", "Neuron-synapse model, consciousness calculation, and knowledge persistence"),
            ("Training Systems", "System", "Software, AGI, GenAI, LLM, SI training modules and mastery tracking"),
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for topic, category, content in core_topics:
            topic_id = hashlib.md5(topic.encode()).hexdigest()[:16]
            now = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO core_knowledge 
                (id, topic, category, content, mastery_level, last_reviewed, created_at, required_for_system)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (topic_id, topic, category, content, 1.0, now, now, 1))
        
        conn.commit()
        conn.close()
        logger.info(f"📚 Loaded {len(core_topics)} core syllabus topics (100% mastery)")
    
    def get_knowledge(self, topic: str, prefer_core: bool = True) -> Optional[Dict]:
        """Retrieve knowledge - checks core tier first, then weighted"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # FIRST: Check core knowledge tier (always max weight)
        cursor.execute('''
            SELECT *, 'core' as tier, 1.0 as current_weight
            FROM core_knowledge 
            WHERE topic LIKE ? OR ? LIKE ('%' || topic || '%')
            ORDER BY mastery_level DESC
            LIMIT 1
        ''', (f'%{topic}%', topic))
        
        result = cursor.fetchone()
        
        if result:
            # Update last_reviewed
            cursor.execute('''
                UPDATE core_knowledge 
                SET last_reviewed = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), result['id']))
            conn.commit()
            conn.close()
            logger.info(f"🎓 CORE knowledge retrieved: '{topic}' (100% mastery)")
            return dict(result)
        
        # SECOND: Check weighted knowledge tier
        normalized = topic.lower().strip()
        cursor.execute('''
            SELECT *, 'weighted' as tier, weight as current_weight
            FROM weighted_knowledge 
            WHERE normalized_topic = ? OR topic LIKE ?
            ORDER BY weight DESC, confidence DESC
            LIMIT 1
        ''', (normalized, f'%{topic}%'))
        
        result = cursor.fetchone()
        
        if result:
            # Update access count and weight
            new_weight = min(result['weight'] + 0.05, 1.0)
            cursor.execute('''
                UPDATE weighted_knowledge 
                SET access_count = access_count + 1,
                    weight = ?,
                    last_accessed = ?
                WHERE id = ?
            ''', (new_weight, datetime.now().isoformat(), result['id']))
            conn.commit()
            conn.close()
            logger.info(f"📈 Weighted knowledge retrieved: '{topic}' (weight: {new_weight:.2f})")
            return dict(result)
        
        conn.close()
        return None
    
    def store_core_knowledge(self, topic: str, content: str, category: str = "System") -> str:
        """Store or update core knowledge (100% mastery, never decays)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        topic_id = hashlib.md5(topic.encode()).hexdigest()[:16]
        now = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO core_knowledge 
            (id, topic, category, content, mastery_level, last_reviewed, created_at, required_for_system)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (topic_id, topic, category, content, 1.0, now, now, 1))
        
        conn.commit()
        conn.close()
        logger.info(f"⭐ Stored CORE knowledge: '{topic}' (100% mastery)")
        return topic_id
    
    def store_weighted_knowledge(self, topic: str, content: str, source: str = "user_interaction", confidence: float = 0.5) -> str:
        """Store new knowledge in weighted tier (starts with low weight)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        knowledge_id = hashlib.md5(topic.encode()).hexdigest()[:16]
        normalized = topic.lower().strip()
        now = datetime.now().isoformat()
        
        # Check if exists
        cursor.execute('SELECT id, weight FROM weighted_knowledge WHERE normalized_topic = ?', (normalized,))
        existing = cursor.fetchone()
        
        if existing:
            # Update - increase weight
            cursor.execute('''
                UPDATE weighted_knowledge 
                SET content = ?,
                    weight = min(weight + 0.1, 1.0),
                    confidence = ?,
                    last_accessed = ?,
                    access_count = access_count + 1
                WHERE normalized_topic = ?
            ''', (content, confidence, now, normalized))
        else:
            # Insert new with initial weight
            cursor.execute('''
                INSERT INTO weighted_knowledge 
                (id, topic, normalized_topic, content, weight, access_count, 
                 last_accessed, created_at, source, confidence, can_promote)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (knowledge_id, topic, normalized, content, 0.3, 1, 
                  now, now, source, confidence, confidence > 0.8))
        
        conn.commit()
        conn.close()
        logger.info(f"💾 Stored weighted knowledge: '{topic}' (weight: 0.3)")
        return knowledge_id
    
    def promote_to_core(self, topic: str) -> bool:
        """Promote weighted knowledge to core tier if important enough"""
        weighted = self.get_knowledge(topic, prefer_core=False)
        if weighted and weighted['tier'] == 'weighted' and weighted['weight'] > 0.8:
            self.store_core_knowledge(topic, weighted['content'], category="Promoted")
            logger.info(f"⭐ PROMOTED '{topic}' from weighted to core tier")
            return True
        return False
    
    def get_core_topics(self) -> List[Dict]:
        """Get all core syllabus topics DMAI has mastered"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT topic, category, mastery_level, last_reviewed FROM core_knowledge ORDER BY category, topic')
        results = cursor.fetchall()
        conn.close()
        return [dict(r) for r in results]
    
    def get_high_weight_topics(self, min_weight: float = 0.7) -> List[Dict]:
        """Get well-learned weighted topics (candidates for promotion)"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT topic, weight, access_count, confidence 
            FROM weighted_knowledge 
            WHERE weight >= ?
            ORDER BY weight DESC
            LIMIT 20
        ''', (min_weight,))
        results = cursor.fetchall()
        conn.close()
        return [dict(r) for r in results]
    
    def decay_weights(self, days_threshold: int = 7):
        """Reduce weights of unused knowledge (core tier never decays)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        threshold_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
        
        cursor.execute('''
            UPDATE weighted_knowledge 
            SET weight = weight * 0.9
            WHERE last_accessed < ? AND weight > 0.1
        ''', (threshold_date,))
        
        conn.commit()
        conn.close()
        logger.info("🔄 Applied weight decay to unused knowledge")
    
    def get_system_knowledge_for_evolution(self) -> Dict:
        """Get all knowledge needed for system evolutions"""
        core_topics = self.get_core_topics()
        high_weight = self.get_high_weight_topics(0.8)
        
        return {
            "core_mastery": core_topics,
            "strong_weighted": high_weight,
            "total_core_topics": len(core_topics),
            "total_weighted_topics": len(high_weight),
            "message": "Core topics are always available at 100% for system operations"
        }
