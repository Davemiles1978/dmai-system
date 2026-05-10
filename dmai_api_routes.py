"""Flask Blueprint – safe addition, zero main‑file edits."""
import logging, traceback
from flask import Blueprint, request, jsonify

import sqlite3
from pathlib import Path

def save_knowledge(topic: str, content: str, entity_type: str = 'core', source: str = 'syllabus'):
    """Write a knowledge snippet directly to SQLite with searchable source_title."""
    try:
        db_path = Path("data/dmai_knowledge.db")
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Ensure source_title is set so query_knowledge can find it
        cursor.execute('''
            INSERT INTO insights (insight_text, entity_type, source_title, source_type,
                                  confidence, created_at, neuron_level)
            VALUES (?, ?, ?, ?, 0.9, datetime('now'), 'micro')
        ''', (content[:5000], entity_type, topic, source))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save knowledge for '{topic}': {e}")
        return False

def query_knowledge(topic: str) -> str:
    """Query SQLite for knowledge on a topic. Returns text or None."""
    try:
        db_path = Path("data/dmai_knowledge.db")
        if not db_path.exists():
            return None
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT insight_text, source_title, LENGTH(insight_text) as len
            FROM insights
            WHERE source_title LIKE ? OR insight_text LIKE ?
            ORDER BY len DESC LIMIT 3
        ''', (f'%{topic}%', f'%{topic}%'))
        rows = cursor.fetchall()
        conn.close()
        if rows:
            return rows[0]['insight_text'][:2000]
        return None
    except Exception:
        return None

logger = logging.getLogger(__name__)
api_bp = Blueprint('custom_api', __name__)

# ---------- health ----------
@api_bp.route('/api/ping')
def ping():
    return jsonify({"status": "ok"})

# ---------- DIRECT KNOWLEDGE LOOKUP ----------
@api_bp.route('/api/knowledge/<topic>')
def get_knowledge(topic):
    """Return the stored knowledge for a given topic directly from SQLite."""
    try:
        import sqlite3
        from pathlib import Path
        
        db_path = Path("data/dmai_knowledge.db")
        if not db_path.exists():
            return jsonify({"error": "Knowledge database not found"}), 500
        
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Search source_title first, then insight_text, preferring longest content
        cursor.execute('''
            SELECT insight_text, entity_type, source_title, confidence,
                   LENGTH(insight_text) as len
            FROM insights
            WHERE source_title LIKE ? OR insight_text LIKE ?
            ORDER BY len DESC
            LIMIT 5
        ''', (f'%{topic}%', f'%{topic}%'))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return jsonify({"error": f"No knowledge found for '{topic}'"}), 404
        
        # Return the longest match (most detailed)
        best = rows[0]
        return jsonify({
            "topic": topic,
            "knowledge": best['insight_text'][:5000],
            "entity_type": best['entity_type'],
            "source_title": best['source_title'],
            "confidence": best['confidence'],
            "length": best['len']
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ---------- DEBUG: Raw SQL query ----------
@api_bp.route('/api/debug/db_query')
def db_query():
    """Run a simple diagnostic query against the knowledge database."""
    try:
        import sqlite3
        from pathlib import Path
        from flask import request
        
        db_path = Path("data/dmai_knowledge.db")
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Count insights with source_title set vs not
        total = cursor.execute("SELECT COUNT(*) as c FROM insights").fetchone()['c']
        with_title = cursor.execute(
            "SELECT COUNT(*) as c FROM insights WHERE source_title IS NOT NULL AND source_title != ''"
        ).fetchone()['c']
        without_title = total - with_title
        
        # Get sample topics with their source_title
        samples = cursor.execute(
            "SELECT insight_text, source_title, entity_type FROM insights "
            "WHERE LENGTH(insight_text) > 100 "
            "ORDER BY RANDOM() LIMIT 5"
        ).fetchall()
        
        # Look specifically for Input Processing
        input_rows = cursor.execute(
            "SELECT insight_text, source_title, entity_type FROM insights "
            "WHERE insight_text LIKE '%Input Processing%' OR source_title LIKE '%Input Processing%' "
            "LIMIT 5"
        ).fetchall()
        
        conn.close()
        
        return jsonify({
            "total_insights": total,
            "with_source_title": with_title,
            "without_source_title": without_title,
            "sample_insights": [dict(r) for r in samples],
            "input_processing_matches": [dict(r) for r in input_rows]
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ---------- FIX: Tag untagged insights ----------
@api_bp.route('/api/debug/tag_insights', methods=['POST'])
def tag_insights():
    """Tag insights that match known syllabus topics with source_title."""
    try:
        import sqlite3, re
        from pathlib import Path
        
        db_path = Path("data/dmai_knowledge.db")
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get all insights with no source_title
        cursor.execute(
            "SELECT id, insight_text FROM insights "
            "WHERE (source_title IS NULL OR source_title = '') "
            "AND LENGTH(insight_text) > 100"
        )
        rows = cursor.fetchall()
        
        tagged = 0
        for row in rows:
            text = row[1]
            # Try to extract topic name from the text
            match = re.match(r'^(.+?) is a (?:core|artistic|wealth|reverse|accelerator|external) concept', text)
            if match:
                topic = match.group(1).strip()
                cursor.execute(
                    "UPDATE insights SET source_title = ? WHERE id = ?",
                    (topic, row[0])
                )
                tagged += 1
            elif 'topic_name' in text.lower():
                # Try to find the topic in the content
                for keyword in ['Input Processing', 'Meta-Learning', 'Pattern Recognition',
                               'Feedback Loop', 'Memory Encoding', 'Correlation Detection',
                               'English Language', 'Python Programming', 'Vibe Coding']:
                    if keyword.lower() in text.lower():
                        cursor.execute(
                            "UPDATE insights SET source_title = ? WHERE id = ?",
                            (keyword, row[0])
                        )
                        tagged += 1
                        break
        
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "tagged": tagged, "total_untagged_was": len(rows)})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ---------- ingest knowledge ----------
@api_bp.route('/api/ingest_knowledge', methods=['POST'])
def ingest_knowledge():
    try:
        data = request.get_json()
        if not data or 'topic' not in data or 'category' not in data or 'content' not in data:
            return jsonify({"error": "Missing fields"}), 400

        # lazy import – safe at runtime
        from dmai_core_complete import _dmai_app_instance
        stage_learner = _dmai_app_instance.evolution.stage_learner
        result = stage_learner.ingest_external_knowledge(data['topic'], data['category'], data['content'])
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return str(e) + '\n' + traceback.format_exc(), 500

# ---------- add key ----------
@api_bp.route('/api/tutors/add_key', methods=['POST'])
def tutors_add_key():
    try:
        data = request.get_json()
        if not data or 'provider' not in data or 'key' not in data:
            return jsonify({"error": "Missing fields"}), 400

        from dmai_core_complete import _dmai_app_instance
        evolution = _dmai_app_instance.evolution

        # lazy key‑store init
        if not hasattr(_dmai_app_instance, 'api_key_store') or _dmai_app_instance.api_key_store is None:
            from components.api_key_store import APIKeyStore
            _dmai_app_instance.api_key_store = APIKeyStore()

        store = _dmai_app_instance.api_key_store
        is_new = store.add_key(data['provider'], data['key'], data.get('source', 'manual'))
        config = None
        if hasattr(evolution, 'tutor_configurator') and evolution.tutor_configurator:
            config = evolution.tutor_configurator.configure_tutor(data['provider'], data['key'])
        return jsonify({"success": True, "new": is_new, "configured": config is not None, "config_result": config})
    except Exception as e:
        return str(e) + '\n' + traceback.format_exc(), 500
