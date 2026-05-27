"""Smart endpoint with complete 146-topic syllabus - FIXED"""
from flask import Blueprint, request, jsonify
import sqlite3
import json
import hashlib
from datetime import datetime

smart_bp = Blueprint('smart', __name__)

# Load ALL syllabus topics from database
def get_syllabus_topics():
    conn = sqlite3.connect('data/dmai_knowledge.db')
    cursor = conn.cursor()
    cursor.execute('SELECT insight_text, stage, category FROM insights WHERE neuron_level = "macro" AND stage IS NOT NULL')
    results = cursor.fetchall()
    conn.close()
    
    topics = {}
    for insight_text, stage, category in results:
        # Extract clean topic name (remove [Stage] prefix)
        clean_name = insight_text.split(']')[-1].strip()
        topics[clean_name.lower()] = {
            'name': clean_name,
            'stage': stage,
            'category': category,
            'mastery': '100%'
        }
    return topics

# Cache syllabus topics
SYLLABUS_TOPICS = get_syllabus_topics()
print(f"📚 Loaded {len(SYLLABUS_TOPICS)} syllabus topics")

@smart_bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question']
        question_lower = question.lower().strip()
        
        # FIRST: Check if exact syllabus topic match
        if question_lower in SYLLABUS_TOPICS:
            topic = SYLLABUS_TOPICS[question_lower]
            return jsonify({
                "answer": f"📚 SYLLABUS TOPIC: {topic['name']}\n\nStage: {topic['stage']}\nCategory: {topic['category']}\nMastery: {topic['mastery']}\n\nThis topic is permanently mastered at expert level. I can provide detailed explanations, practical applications, and cross-domain connections.\n\nWhat specific aspect of {topic['name']} would you like to explore?",
                "status": "success",
                "mastery": "100%",
                "stage": topic['stage'],
                "category": topic['category'],
                "syllabus": True
            })
        
        # SECOND: Check for partial matches
        for topic_key, topic_info in SYLLABUS_TOPICS.items():
            if topic_key in question_lower or any(word in topic_key for word in question_lower.split()[:3]):
                return jsonify({
                    "answer": f"📚 SYLLABUS TOPIC (Related): {topic_info['name']}\n\nStage: {topic_info['stage']}\nCategory: {topic_info['category']}\nMastery: {topic_info['mastery']}\n\nThis is related to your question. Would you like me to explain {topic_info['name']} in detail?",
                    "status": "success",
                    "syllabus_related": True,
                    "suggested_topic": topic_info['name']
                })
        
        # THIRD: New topic - track with weight
        conn = sqlite3.connect('data/dmai_knowledge.db')
        cursor = conn.cursor()
        
        # Get current weight
        cursor.execute('SELECT weight, detail_level FROM question_weights WHERE question = ?', (question[:200],))
        existing = cursor.fetchone()
        
        if existing:
            new_weight = existing[0] + 1
            new_detail = min(existing[1] + 1, 5)
            
            # Generate answer based on detail level
            if new_detail == 1:
                answer = f"📘 LEARNING: '{question}' (Weight: {new_weight})\n\nI'm building my knowledge on this topic. Each time we discuss it, I'll add more detail and increase its weight in my knowledge graph."
            elif new_detail == 2:
                answer = f"📙 GROWING: '{question}' (Weight: {new_weight})\n\nI have basic understanding now. Continuing to add examples and applications."
            elif new_detail == 3:
                answer = f"📗 ADVANCING: '{question}' (Weight: {new_weight})\n\nI have intermediate knowledge. Adding deeper concepts and connections."
            elif new_detail == 4:
                answer = f"📕 EXPERT: '{question}' (Weight: {new_weight})\n\nI have substantial knowledge. Providing comprehensive explanations."
            else:
                answer = f"⭐ MASTERED: '{question}' (Weight: {new_weight})\n\nThis topic is now well-mastered. I can provide expert-level analysis and cross-domain connections."
            
            cursor.execute('''
                UPDATE question_weights 
                SET weight = ?, detail_level = ?, last_asked = ?
                WHERE question = ?
            ''', (new_weight, new_detail, datetime.now().isoformat(), question[:200]))
        else:
            new_weight = 1
            new_detail = 1
            answer = f"🌱 NEW TOPIC: '{question}' (Weight: 1)\n\nI'm beginning to learn about this topic. Each time we discuss it, my understanding will deepen and its weight will increase in my knowledge graph.\n\nWhat specific aspect would you like me to focus on first?"
            
            cursor.execute('''
                INSERT INTO question_weights (question, weight, detail_level, topic_category, last_asked)
                VALUES (?, ?, ?, ?, ?)
            ''', (question[:200], 1, 1, 'user_asked', datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "answer": answer,
            "status": "success",
            "weight": new_weight,
            "detail_level": new_detail,
            "syllabus": False
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@smart_bp.route('/syllabus', methods=['GET'])
def get_syllabus():
    """List all mastered syllabus topics"""
    syllabus_list = []
    for topic_key, topic_info in SYLLABUS_TOPICS.items():
        syllabus_list.append({
            "topic": topic_info['name'],
            "stage": topic_info['stage'],
            "category": topic_info['category'],
            "mastery": topic_info['mastery']
        })
    
    # Sort by stage
    stage_order = {'Baby': 0, 'Toddler': 1, 'Child': 2, 'Teen': 3, 'Adult': 4}
    syllabus_list.sort(key=lambda x: (stage_order.get(x['stage'], 99), x['topic']))
    
    return jsonify({
        "topics": syllabus_list,
        "total_topics": len(syllabus_list),
        "mastery_level": "100% (Permanent)",
        "message": f"All {len(syllabus_list)} syllabus topics are permanently mastered"
    })

@smart_bp.route('/weights', methods=['GET'])
def get_weights():
    """View topic weights and detail levels"""
    try:
        conn = sqlite3.connect('data/dmai_knowledge.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT question, weight, detail_level, topic_category, last_asked 
            FROM question_weights 
            ORDER BY weight DESC, detail_level DESC
            LIMIT 50
        ''')
        results = cursor.fetchall()
        conn.close()
        
        topics = []
        for r in results:
            topics.append({
                "topic": r[0],
                "weight": r[1],
                "detail_level": r[2],
                "category": r[3] or "general",
                "last_asked": r[4]
            })
        
        return jsonify({
            "topics": topics,
            "total": len(topics),
            "message": "Higher weight = more detail retained"
        })
    except Exception as e:
        return jsonify({"topics": [], "total": 0, "error": str(e)})

@smart_bp.route('/prune', methods=['POST'])
def prune_low_weight():
    """Remove low-weight topics to optimize memory"""
    try:
        data = request.get_json()
        threshold = data.get('threshold', 3)
        
        conn = sqlite3.connect('data/dmai_knowledge.db')
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM question_weights WHERE weight < ?', (threshold,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return jsonify({
            "pruned_count": deleted,
            "threshold": threshold,
            "message": f"Removed {deleted} low-weight topics"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
