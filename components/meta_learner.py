"""Meta-Learning Evolution Engine - DMAI learns how to learn better"""
import sqlite3
import json
import time
import random
from datetime import datetime
import threading
import logging
from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

class MetaLearningEngine:
    """Continuously improves DMAI's learning strategies based on outcomes"""
    
    def __init__(self, db_path="data/dmai_knowledge.db"):
        self.db_path = db_path
        self.init_tables()
        self.learning_strategies = self.load_strategies()
        self.start_optimization_loop()
    
    def init_tables(self):
        conn = safe_open_kdb(self.db_path)
        cursor = conn.cursor()
        
        # Track learning outcomes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_outcomes (
                id SERIAL PRIMARY KEY,
                topic TEXT,
                strategy_used TEXT,
                weight_before INTEGER,
                weight_after INTEGER,
                response_quality REAL,
                time_spent REAL,
                timestamp TIMESTAMP
            )
        ''')
        
        # Track optimal strategies per topic type
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimal_strategies (
                topic_category TEXT,
                strategy TEXT,
                effectiveness REAL,
                sample_size INTEGER,
                last_updated TIMESTAMP,
                PRIMARY KEY (topic_category, strategy)
            )
        ''')
        
        # Store learned patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                pattern_data TEXT,
                success_rate REAL,
                times_used INTEGER,
                created_at TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("📚 Meta-learning tables initialized")
    
    def load_strategies(self):
        """Define learning strategies DMAI can use"""
        return {
            'direct_explanation': {
                'description': 'Provide direct, comprehensive explanation',
                'weight_increment': 1,
                'best_for': ['core', 'foundation', 'definition']
            },
            'example_driven': {
                'description': 'Teach through concrete examples and cases',
                'weight_increment': 1.2,
                'best_for': ['application', 'practical', 'implementation']
            },
            'comparative': {
                'description': 'Compare with known concepts',
                'weight_increment': 1.3,
                'best_for': ['relationship', 'connection', 'vs']
            },
            'question_based': {
                'description': 'Guide through Socratic questioning',
                'weight_increment': 1.4,
                'best_for': ['exploration', 'discovery', 'why']
            },
            'layered': {
                'description': 'Build from simple to complex',
                'weight_increment': 1.5,
                'best_for': ['complex', 'advanced', 'expert']
            }
        }
    
    def record_outcome(self, topic, strategy, weight_before, weight_after, response_quality, time_spent):
        """Record learning outcome for optimization"""
        conn = safe_open_kdb(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO learning_outcomes (topic, strategy_used, weight_before, weight_after, response_quality, time_spent, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (topic, strategy, weight_before, weight_after, response_quality, time_spent, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        # Update optimal strategies
        category = self.get_topic_category(topic)
        self.update_optimal_strategy(category, strategy, response_quality)
    
    def get_topic_category(self, topic):
        """Determine topic category for strategy optimization"""
        topic_lower = topic.lower()
        categories = {
            'technical': ['algorithm', 'code', 'programming', 'architecture', 'network'],
            'conceptual': ['theory', 'principle', 'fundamental', 'philosophy'],
            'practical': ['application', 'implementation', 'how to', 'tutorial'],
            'comparative': ['vs', 'compare', 'difference', 'versus']
        }
        
        for category, keywords in categories.items():
            if any(kw in topic_lower for kw in keywords):
                return category
        return 'general'
    
    def update_optimal_strategy(self, category, strategy, effectiveness):
        """Update which strategy works best for each category"""
        conn = safe_open_kdb(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO optimal_strategies (topic_category, strategy, effectiveness, sample_size, last_updated)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(topic_category, strategy) DO UPDATE SET
                effectiveness = (effectiveness * sample_size + ?) / (sample_size + 1),
                sample_size = sample_size + 1,
                last_updated = excluded.last_updated
        ''', (category, strategy, effectiveness, datetime.now().isoformat(), effectiveness))
        
        conn.commit()
        conn.close()
    
    def select_best_strategy(self, topic, weight):
        """Dynamically select the best learning strategy based on past effectiveness"""
        category = self.get_topic_category(topic)
        
        conn = safe_open_kdb(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT strategy, effectiveness FROM optimal_strategies 
            WHERE topic_category = ? 
            ORDER BY effectiveness DESC 
            LIMIT 1
        ''', (category,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            best_strategy = result[0]
            logger.info(f"Selected strategy '{best_strategy}' for {category} topic (effectiveness: {result[1]:.2f})")
            return best_strategy
        
        # Default strategies based on weight
        if weight < 3:
            return 'direct_explanation'
        elif weight < 6:
            return 'example_driven'
        elif weight < 10:
            return 'comparative'
        else:
            return 'layered'
    
    def analyze_learning_patterns(self):
        """Find patterns in what learning approaches work best"""
        conn = safe_open_kdb(self.db_path)
        cursor = conn.cursor()
        
        # Find which strategies lead to fastest weight gain
        cursor.execute('''
            SELECT strategy_used, AVG(weight_after - weight_before) as avg_gain, COUNT(*) as count
            FROM learning_outcomes
            GROUP BY strategy_used
            ORDER BY avg_gain DESC
        ''')
        patterns = cursor.fetchall()
        
        pattern_id = f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cursor.execute('''
            INSERT INTO learning_patterns (pattern_id, pattern_type, pattern_data, success_rate, times_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (pattern_id, 'strategy_effectiveness', json.dumps(patterns), patterns[0][1] if patterns else 0, len(patterns), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return patterns
    
    def continuous_optimization(self):
        """Run continuous optimization of learning strategies"""
        while True:
            time.sleep(3600)  # Run every hour
            
            try:
                # Analyze patterns
                patterns = self.analyze_learning_patterns()
                logger.info(f"📊 Learning pattern analysis complete: {len(patterns)} strategies evaluated")
                
                # Adjust strategies based on findings
                for strategy, avg_gain, count in patterns:
                    if avg_gain > 1.0:
                        logger.info(f"  ✅ {strategy}: {avg_gain:.2f}x weight gain ({count} samples)")
                    elif avg_gain < 0.5:
                        logger.info(f"  ⚠️ {strategy}: {avg_gain:.2f}x weight gain - needs improvement")
                
                # Log the best strategy
                if patterns:
                    best = patterns[0]
                    logger.info(f"🏆 Best strategy: {best[0]} with {best[1]:.2f}x weight gain")
                    
            except Exception as e:
                logger.error(f"Optimization error: {e}")
    
    def start_optimization_loop(self):
        """Start background optimization thread"""
        thread = threading.Thread(target=self.continuous_optimization, daemon=True)
        thread.start()
        logger.info("🔄 Meta-learning optimization loop started")
    
    def apply_learning_to_evolution(self, evolution_engine):
        """Apply meta-learning insights to the main evolution engine"""
        conn = safe_open_kdb(self.db_path)
        cursor = conn.cursor()
        
        # Get best strategies for each category
        cursor.execute('''
            SELECT topic_category, strategy, effectiveness 
            FROM optimal_strategies 
            WHERE effectiveness > 0.8
            ORDER BY effectiveness DESC
        ''')
        best_strategies = cursor.fetchall()
        conn.close()
        
        insights = []
        for category, strategy, effectiveness in best_strategies:
            insight = f"For {category} topics, {strategy} yields {effectiveness:.2f}x learning efficiency"
            insights.append(insight)
            
            # Log to evolution system if available
            if evolution_engine:
                try:
                    evolution_engine.add_learning_insight(insight)
                except:
                    pass
        
        return insights

# Initialize global instance
meta_learner = MetaLearningEngine()
logger.info("🧠 Meta-learning engine active - DMAI is learning how to learn")
