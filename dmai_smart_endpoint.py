"""Smart endpoint - Uses local syllabus data, no external dependencies"""
from flask import Blueprint, request, jsonify
import sqlite3
import json
from datetime import datetime

smart_bp = Blueprint('smart', __name__)

# Embedded syllabus data (sample - expand with all 146)
SYLLABUS_TOPICS = {
    "meta-learning fundamentals": {"stage": "Baby", "category": "Core"},
    "self-code analysis": {"stage": "Baby", "category": "Accelerator"},
    "neural network architectures": {"stage": "Child", "category": "AI"},
    "reinforcement learning": {"stage": "Child", "category": "AI"},
    "transformer models": {"stage": "Teen", "category": "LLM"},
    "mixture of experts": {"stage": "Adult", "category": "Advanced"},
}

@smart_bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question'].lower()
        
        # Check syllabus
        for topic, info in SYLLABUS_TOPICS.items():
            if topic in question or question in topic:
                return jsonify({
                    "answer": f"SYLLABUS: {topic.title()} (Stage: {info['stage']})\n\nThis topic is mastered at 100%. What specific aspect would you like to explore?",
                    "status": "success",
                    "syllabus": True,
                    "mastery": "100%"
                })
        
        # Track new topics
        conn = sqlite3.connect('data/dmai_knowledge.db')
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS topic_weights (topic TEXT PRIMARY KEY, weight INTEGER, last_asked TIMESTAMP)')
        
        cursor.execute('SELECT weight FROM topic_weights WHERE topic = ?', (question[:100],))
        existing = cursor.fetchone()
        
        if existing:
            new_weight = existing[0] + 1
            cursor.execute('UPDATE topic_weights SET weight = ?, last_asked = ? WHERE topic = ?',
                         (new_weight, datetime.now().isoformat(), question[:100]))
            answer = f"Topic '{question}' - Weight: {new_weight}\n\nI'm building knowledge on this topic."
        else:
            new_weight = 1
            cursor.execute('INSERT INTO topic_weights VALUES (?, ?, ?)', (question[:100], 1, datetime.now().isoformat()))
            answer = f"New topic: '{question}'\nWeight: 1\n\nI'll learn more each time we discuss this."
        
        conn.commit()
        conn.close()
        
        return jsonify({"answer": answer, "status": "success", "weight": new_weight})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@smart_bp.route('/syllabus', methods=['GET'])
def get_syllabus():
    topics_list = [{"topic": k.title(), "stage": v["stage"]} for k, v in SYLLABUS_TOPICS.items()]
    return jsonify({"topics": topics_list, "total": len(topics_list)})

@smart_bp.route('/weights', methods=['GET'])
def get_weights():
    try:
        conn = sqlite3.connect('data/dmai_knowledge.db')
        cursor = conn.cursor()
        cursor.execute('SELECT topic, weight FROM topic_weights ORDER BY weight DESC LIMIT 50')
        results = cursor.fetchall()
        conn.close()
        return jsonify({"topics": [{"topic": r[0], "weight": r[1]} for r in results]})
    except:
        return jsonify({"topics": []})
