"""DMAI Core - Stable Production Version with Complete Syllabus"""
import os
import sys
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime

# Import the complete syllabus
from dmai_syllabus_data import SYLLABUS_TOPICS, TOTAL_TOPICS

# Disable all problematic features for Render
os.environ['DISABLE_NEO4J'] = 'true'
os.environ['DISABLE_AUTO_THREADS'] = 'true'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

def get_syllabus_topic(question):
    """Find matching syllabus topic"""
    question_lower = question.lower().strip()
    for topic in SYLLABUS_TOPICS:
        if topic in question_lower or question_lower in topic:
            return topic, SYLLABUS_TOPICS[topic]
    return None, None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "stable"
    })

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "running",
        "version": "stable",
        "syllabus_topics": TOTAL_TOPICS,
        "neo4j": "disabled",
        "threads": "controlled",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/v2/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question']
        
        # Check syllabus first
        topic, knowledge = get_syllabus_topic(question)
        
        if knowledge:
            return jsonify({
                "answer": knowledge["content"],
                "topic": topic.title(),
                "stage": knowledge["stage"],
                "category": knowledge["category"],
                "mastery": knowledge["mastery"],
                "source": "permanent_syllabus",
                "status": "success"
            })
        
        # For non-syllabus questions
        return jsonify({
            "answer": f"I understand you're asking about '{question}'. This topic isn't in my permanent syllabus yet. What specific aspect interests you?",
            "status": "success",
            "source": "general"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v2/syllabus', methods=['GET'])
def get_syllabus():
    topics = []
    for topic, info in SYLLABUS_TOPICS.items():
        topics.append({
            "topic": topic.title(),
            "stage": info["stage"],
            "category": info["category"],
            "mastery": info["mastery"]
        })
    return jsonify({
        "topics": topics,
        "total": len(topics),
        "message": f"{len(topics)} topics permanently mastered"
    })

@app.route('/v2/weights', methods=['GET'])
def get_weights():
    topics = []
    for topic, info in SYLLABUS_TOPICS.items():
        topics.append({
            "topic": topic.title(),
            "weight": 100,
            "mastery": info["mastery"]
        })
    return jsonify({
        "topics": topics,
        "total": len(topics),
        "message": "All syllabus topics at 100% mastery"
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting DMAI Stable on port {port}")
    logger.info(f"Syllabus loaded: {TOTAL_TOPICS} topics")
    app.run(host='0.0.0.0', port=port)
