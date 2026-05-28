"""Stable Smart Endpoint - No problematic dependencies"""
from flask import Blueprint, request, jsonify
import sqlite3
from datetime import datetime
import os

smart_bp = Blueprint('smart', __name__)

# Permanent syllabus knowledge (built-in, never needs external calls)
PERMANENT_KNOWLEDGE = {
    "neural network architectures": {
        "stage": "Child",
        "category": "AI",
        "content": "Neural networks come in several types. CNNs excel at images using sliding filters. RNNs handle sequences with memory. Transformers use self-attention for long-range dependencies."
    },
    "meta learning fundamentals": {
        "stage": "Baby",
        "category": "Core", 
        "content": "Meta-learning is learning how to learn. It optimizes learning strategies based on past experience."
    },
    "attention mechanisms": {
        "stage": "Toddler",
        "category": "Core",
        "content": "Attention focuses on relevant information. It uses Query, Key, Value to weight important inputs."
    },
    "reinforcement learning": {
        "stage": "Child",
        "category": "AI",
        "content": "Reinforcement learning trains agents through rewards. Key algorithms include Q-Learning and PPO."
    },
    "transformer architecture": {
        "stage": "Teen",
        "category": "AI",
        "content": "Transformers use self-attention instead of recurrence. They process sequences in parallel."
    },
    "recursive self improvement": {
        "stage": "Adult",
        "category": "Accelerator",
        "content": "Recursive self-improvement means DMAI improves her own improvement mechanisms."
    }
}

def is_syllabus_topic(question):
    question_lower = question.lower()
    for topic in PERMANENT_KNOWLEDGE:
        if topic in question_lower or question_lower in topic:
            return topic
    return None

@smart_bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question']
        
        # Check permanent knowledge first
        syllabus_match = is_syllabus_topic(question)
        if syllabus_match:
            info = PERMANENT_KNOWLEDGE[syllabus_match]
            return jsonify({
                "answer": info["content"],
                "topic": syllabus_match.title(),
                "stage": info["stage"],
                "category": info["category"],
                "mastery": "100%",
                "source": "permanent",
                "status": "success"
            })
        
        # For other questions, provide helpful response
        return jsonify({
            "answer": f"I understand you're asking about {question}. This topic isn't in my permanent syllabus, but I can help explore it. What specific aspect interests you?",
            "status": "success",
            "source": "general"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@smart_bp.route('/syllabus', methods=['GET'])
def get_syllabus():
    topics = [{"topic": k.title(), "stage": v["stage"], "category": v["category"]} 
              for k, v in PERMANENT_KNOWLEDGE.items()]
    return jsonify({
        "topics": topics,
        "total": len(topics),
        "message": "Permanent syllabus topics - always available"
    })

@smart_bp.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
