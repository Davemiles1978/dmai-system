"""Updated smart endpoint with syllabus mastery"""
from flask import Blueprint, request, jsonify
import sqlite3
import hashlib
from datetime import datetime
from dmai_syllabus_knowledge import get_syllabus_knowledge, SYLLABUS_KNOWLEDGE

smart_bp = Blueprint('smart', __name__)

def get_answer(question):
    """Get answer from syllabus first, then fallback"""
    question_lower = question.lower()
    
    # FIRST: Check syllabus (mastered at 100%)
    syllabus_answer = get_syllabus_knowledge(question)
    if syllabus_answer:
        return syllabus_answer
    
    # SECOND: Check for related syllabus topics
    for topic, content in SYLLABUS_KNOWLEDGE.items():
        if topic in question_lower or any(word in question_lower for word in topic.split()[:2]):
            return f"RELATED TOPIC - {topic.upper()}:\n\n{content}"
    
    # THIRD: Comprehensive fallback for unknown topics
    return f"""I understand you're asking about: {question}

This topic is not yet in my mastered syllabus. To master it:

1. I will research {question} using AI tutors (OpenAI, DeepSeek, Gemini, Claude)
2. Create detailed knowledge neurons
3. Build connections to related topics
4. Track weight based on how often we discuss it

The more we discuss this topic, the deeper my understanding becomes, and the higher its weight in my knowledge graph.

Would you like me to research this topic now?"""

@smart_bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question']
        answer = get_answer(question)
        
        # Store weight
        try:
            conn = sqlite3.connect('data/dmai_knowledge.db')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS question_weights (
                    question TEXT PRIMARY KEY,
                    weight INTEGER DEFAULT 1,
                    topic_category TEXT,
                    last_asked TIMESTAMP
                )
            ''')
            
            # Determine if syllabus topic
            is_syllabus = get_syllabus_knowledge(question) is not None
            
            cursor.execute('''
                INSERT INTO question_weights (question, weight, topic_category, last_asked)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(question) DO UPDATE SET
                    weight = weight + 1,
                    last_asked = excluded.last_asked
            ''', (question[:200], 'syllabus' if is_syllabus else 'general', datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Weight store error: {e}")
        
        return jsonify({
            "answer": answer,
            "status": "success",
            "mastery_level": "100%" if get_syllabus_knowledge(question) else "learning"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@smart_bp.route('/syllabus', methods=['GET'])
def get_syllabus():
    """List all mastered syllabus topics"""
    from dmai_syllabus_knowledge import get_all_topics
    return jsonify({
        "mastered_topics": get_all_topics(),
        "count": len(get_all_topics()),
        "mastery_level": "100%",
        "message": "These topics are permanently mastered at expert level"
    })

@smart_bp.route('/weights', methods=['GET'])
def get_weights():
    """View topic weights"""
    try:
        conn = sqlite3.connect('data/dmai_knowledge.db')
        cursor = conn.cursor()
        cursor.execute('SELECT question, weight, topic_category FROM question_weights ORDER BY weight DESC LIMIT 50')
        results = cursor.fetchall()
        conn.close()
        return jsonify({
            "topics": [{"topic": r[0], "weight": r[1], "category": r[2]} for r in results],
            "total": len(results)
        })
    except Exception as e:
        return jsonify({"topics": [], "total": 0, "error": str(e)})
