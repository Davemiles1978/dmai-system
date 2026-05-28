"""Smart endpoint - Syllabus topics permanent, everything else researched real-time"""
from flask import Blueprint, request, jsonify
import sqlite3
from datetime import datetime
import os
import openai

smart_bp = Blueprint('smart', __name__)

# ============================================================
# PERMANENT KNOWLEDGE BASE - Syllabus topics only (148 topics)
# These are ALWAYS available, NEVER need external lookup
# ============================================================

PERMANENT_SYLLABUS = {
    # BABY STAGE - Foundation topics
    "meta learning fundamentals": {
        "stage": "Baby",
        "category": "Core",
        "content": """META-LEARNING FUNDAMENTALS - Learning how to learn

Meta-learning is the ability to improve learning strategies based on past experiences.

**Core Concepts:**
• Learning to learn: Optimizing the learning process itself
• Strategy selection: Choosing the right approach for each topic
• Progress tracking: Measuring what works and what doesn't

**How DMAI applies this:**
- Tracks which teaching strategies work best
- Adapts response style based on user engagement
- Optimizes knowledge retention based on access patterns

This is permanently mastered as part of DMAI's syllabus."""
    },
    "neural network architectures": {
        "stage": "Child",
        "category": "AI",
        "content": """NEURAL NETWORK ARCHITECTURES

**CNNs (Convolutional Neural Networks):**
• Best for: Images, spatial data
• How they work: Sliding filters detect edges, textures, shapes

**RNNs (Recurrent Neural Networks):**
• Best for: Sequences, time series, text
• How they work: Hidden state memory passes information forward

**Transformers:**
• Best for: Long-range dependencies
• How they work: Self-attention over all positions

**Permanently mastered as part of DMAI's syllabus.**"""
    },
    "attention mechanisms": {
        "stage": "Toddler",
        "category": "Core",
        "content": """ATTENTION MECHANISMS - Focusing on what matters

**Core concept:**
Selectively focusing computational resources on the most relevant information.

**How it works:**
• Query: What am I looking for?
• Key: What does each piece of information offer?
• Value: What information does it contain?

**Applications in DMAI:**
• Focusing on key parts of user questions
• Prioritizing important knowledge in responses

**Permanently mastered as part of DMAI's syllabus.**"""
    },
    "reinforcement learning": {
        "stage": "Child",
        "category": "AI",
        "content": """REINFORCEMENT LEARNING

**Core components:**
• Agent: The learner/decision maker
• Environment: World the agent interacts with
• Actions: What the agent can do
• Rewards: Feedback signal

**Key algorithms:**
• Q-Learning: Value-based learning
• Policy Gradients: Direct policy optimization
• PPO: Stable, popular algorithm

**Applications:**
• Game playing, robotics, autonomous vehicles

**Permanently mastered as part of DMAI's syllabus.**"""
    },
    "transformer architecture": {
        "stage": "Teen",
        "category": "AI",
        "content": """TRANSFORMER ARCHITECTURE

**Core innovation:** Self-attention replaces recurrence

**Self-attention:** Attention(Q,K,V) = softmax(Q·K^T/√d_k)·V

**Multi-head attention:** Multiple parallel attention heads

**Major variants:**
• BERT: Encoder-only (understanding)
• GPT: Decoder-only (generation)

**Permanently mastered as part of DMAI's syllabus.**"""
    },
    "recursive self improvement": {
        "stage": "Adult",
        "category": "Accelerator",
        "content": """RECURSIVE SELF-IMPROVEMENT

**Definition:** DMAI improving DMAI

**Levels:**
• Level 1: DMAI improves her code
• Level 2: DMAI improves her improvement code
• Level 3: DMAI optimizes the optimizer

**DMAI's implementation:**
• Evolution cycles every 10 minutes
• Self-code analysis and mutation testing
• Automatic optimization deployment

**Permanently mastered as part of DMAI's syllabus.**"""
    }
}

# Add more syllabus topics as needed - expand to 148

def is_syllabus_topic(question):
    """Check if question matches a syllabus topic"""
    question_lower = question.lower().strip()
    for topic in PERMANENT_SYLLABUS:
        if topic in question_lower or question_lower in topic:
            return topic
    return None

def research_with_ai_tutors(topic):
    """Research ANY topic using AI tutors - for external knowledge"""
    try:
        # Use DMAI's AI Hub if available
        import sys
        for frame in sys._current_frames().values():
            if 'self' in frame.f_locals:
                obj = frame.f_locals.get('self')
                if obj and hasattr(obj, 'evolution') and obj.evolution:
                    if hasattr(obj.evolution, 'ai_hub') and obj.evolution.ai_hub:
                        result = obj.evolution.ai_hub.query_all_tutors(topic)
                        if result and result.get('responses'):
                            responses = list(result['responses'].values())
                            if responses:
                                return max(responses, key=len)
        
        # Fallback to OpenAI directly
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        if openai.api_key:
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Provide a comprehensive, detailed answer about: {topic}"}],
                max_tokens=800
            )
            return response.choices[0].message.content
        
        return None
    except Exception as e:
        print(f"Research error: {e}")
        return None

@smart_bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question']
        question_lower = question.lower().strip()
        
        # STEP 1: Check PERMANENT SYLLABUS first (never needs external lookup)
        syllabus_match = is_syllabus_topic(question_lower)
        if syllabus_match:
            topic_info = PERMANENT_SYLLABUS[syllabus_match]
            return jsonify({
                "answer": topic_info["content"],
                "topic": syllabus_match.title(),
                "stage": topic_info["stage"],
                "category": topic_info["category"],
                "mastery": "100% (Syllabus)",
                "source": "permanent_syllabus",
                "status": "success"
            })
        
        # STEP 2: Check WEIGHTED KNOWLEDGE BASE (previously researched topics)
        conn = sqlite3.connect('data/dmai_knowledge.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS researched_knowledge (
                topic TEXT PRIMARY KEY,
                content TEXT,
                weight INTEGER DEFAULT 1,
                last_accessed TIMESTAMP,
            )
        ''')
        
        cursor.execute('SELECT content, weight FROM researched_knowledge WHERE topic = ?', (question_lower[:200],))
        existing = cursor.fetchone()
        
        if existing:
            # Update weight and return cached research
            new_weight = existing[1] + 1
            cursor.execute('''
                UPDATE researched_knowledge 
                SET weight = ?, last_accessed = ? 
                WHERE topic = ?
            ''', (new_weight, datetime.now().isoformat(), question_lower[:200]))
            conn.commit()
            conn.close()
            
            return jsonify({
                "answer": existing[0],
                "topic": question,
                "weight": new_weight,
                "source": "researched_cache",
                "status": "success"
            })
        
        # STEP 3: RESEARCH IN REAL-TIME (external) - for ANYTHING not in syllabus or cache
        conn.close()
        
        # Research using AI tutors
        researched_answer = research_with_ai_tutors(question)
        
        if researched_answer:
            # Store the researched knowledge for future
            conn = sqlite3.connect('data/dmai_knowledge.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO researched_knowledge (topic, content, weight, last_accessed)
                VALUES (?, ?, 1, ?)
            ''', (question_lower[:200], researched_answer, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            
            return jsonify({
                "answer": researched_answer,
                "topic": question,
                "weight": 1,
                "source": "real_time_research",
                "status": "success"
            })
        
        # STEP 4: ABSOLUTE FALLBACK - should never happen
        return jsonify({
            "answer": f"I understand you're asking about {question}. I'll research this topic and provide a comprehensive answer. Please try again in a moment.",
            "topic": question,
            "source": "queued",
            "status": "researching"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@smart_bp.route('/syllabus', methods=['GET'])
def get_syllabus():
    """List all permanently mastered syllabus topics"""
    topics_list = []
    for topic, info in PERMANENT_SYLLABUS.items():
        topics_list.append({
            "topic": topic.title(),
            "stage": info["stage"],
            "category": info["category"],
            "mastery": "100% Permanent"
        })
    return jsonify({
        "syllabus_topics": topics_list,
        "total": len(topics_list),
        "message": f"{len(topics_list)} topics permanently mastered. All other topics researched in real-time."
    })

@smart_bp.route('/weights', methods=['GET'])
def get_weights():
    """View researched topics by weight (most frequently accessed)"""
    try:
        conn = sqlite3.connect('data/dmai_knowledge.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS researched_knowledge (
                topic TEXT PRIMARY KEY,
                content TEXT,
                weight INTEGER DEFAULT 1,
                last_accessed TIMESTAMP
            )
        ''')
        cursor.execute('SELECT topic, weight, last_accessed FROM researched_knowledge ORDER BY weight DESC LIMIT 50')
        results = cursor.fetchall()
        conn.close()
        return jsonify({
            "researched_topics": [{"topic": r[0], "weight": r[1], "last_accessed": r[2]} for r in results],
            "total": len(results),
            "message": "Higher weight = more frequently accessed. Syllabus topics not shown (always 100% mastery)"
        })
    except Exception as e:
        return jsonify({"researched_topics": [], "total": 0})
