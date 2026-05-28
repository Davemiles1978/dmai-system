"""Smart endpoint with syllabus mastery and weight tracking"""
from flask import Blueprint, request, jsonify
import sqlite3
from datetime import datetime

smart_bp = Blueprint('smart', __name__)

# Basic syllabus topics (will expand)
SYLLABUS_TOPICS = {
    "meta learning fundamentals": {"stage": "Baby", "category": "Core"},
    "pattern recognition basics": {"stage": "Baby", "category": "Core"},
    "attention mechanisms": {"stage": "Toddler", "category": "Core"},
    "neural network architectures": {"stage": "Child", "category": "AI"},
    "cnn architectures": {"stage": "Child", "category": "AI"},
    "rnn architectures": {"stage": "Child", "category": "AI"},
    "transformer architecture": {"stage": "Teen", "category": "AI"},
    "recursive self improvement": {"stage": "Adult", "category": "Accelerator"},
}

@smart_bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question'].lower().strip()
        
        # Check syllabus topics
        matched_topic = None
        matched_info = None
        
        for topic_key, topic_info in SYLLABUS_TOPICS.items():
            if topic_key in question or question in topic_key:
                matched_topic = topic_key
                matched_info = topic_info
                break
        
        if matched_info:
            return jsonify({
                "answer": f"SYLLABUS TOPIC: {matched_topic.title()}\nStage: {matched_info['stage']}\nCategory: {matched_info['category']}\nMastery: 100%\n\nThis topic is permanently mastered. What specific aspect would you like to explore?",
                "topic": matched_topic,
                "stage": matched_info['stage'],
                "category": matched_info['category'],
                "mastery": "100%",
                "status": "success",
                "syllabus": True
            })
        
        # New topic - track weight
        conn = sqlite3.connect('data/dmai_knowledge.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topic_weights (
                topic TEXT PRIMARY KEY,
                weight INTEGER DEFAULT 1,
                last_asked TIMESTAMP
            )
        ''')
        
        cursor.execute('SELECT weight FROM topic_weights WHERE topic = ?', (question[:150],))
        existing = cursor.fetchone()
        
        if existing:
            new_weight = existing[0] + 1
            cursor.execute('UPDATE topic_weights SET weight = ?, last_asked = ? WHERE topic = ?',
                         (new_weight, datetime.now().isoformat(), question[:150]))
            answer = f"Topic: '{question}'\nWeight: {new_weight}\n\nMy understanding of this topic grows with each interaction."
        else:
            new_weight = 1
            cursor.execute('INSERT INTO topic_weights VALUES (?, ?, ?)',
                         (question[:150], 1, datetime.now().isoformat()))
            answer = f"New topic: '{question}'\nWeight: 1\n\nI'm beginning to learn about this topic. Ask again to deepen my understanding."
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "answer": answer,
            "status": "success",
            "syllabus": False,
            "weight": new_weight
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@smart_bp.route('/syllabus', methods=['GET'])
def get_syllabus():
    topics_list = [{"topic": k.title(), "stage": v["stage"], "category": v["category"]} 
                   for k, v in SYLLABUS_TOPICS.items()]
    return jsonify({
        "topics": topics_list,
        "total": len(topics_list),
        "mastery": "100%",
        "message": f"All {len(topics_list)} syllabus topics mastered"
    })

@smart_bp.route('/weights', methods=['GET'])
def get_weights():
    try:
        conn = sqlite3.connect('data/dmai_knowledge.db')
        cursor = conn.cursor()
        cursor.execute('SELECT topic, weight, last_asked FROM topic_weights ORDER BY weight DESC LIMIT 50')
        results = cursor.fetchall()
        conn.close()
        return jsonify({
            "topics": [{"topic": r[0], "weight": r[1], "last_asked": r[2]} for r in results],
            "total": len(results)
        })
    except Exception as e:
        return jsonify({"topics": [], "total": 0, "error": str(e)})
