"""Flask Blueprint – safe addition, zero main‑file edits."""
import logging, traceback, json, uuid
from flask import Blueprint, request, jsonify

import sqlite3
from pathlib import Path

def humanize_text(text: str, max_length: int = 2000) -> str:
    """Strip AI tells and make text sound human-spoken. For human-facing output only."""
    if not text or len(text) < 50:
        return text
    
    try:
        from dmai_core_complete import _dmai_app_instance
        if hasattr(_dmai_app_instance, 'evolution') and hasattr(_dmai_app_instance.evolution, 'ai_hub'):
            ai_hub = _dmai_app_instance.evolution.ai_hub
            prompt = f"""Rewrite this text to sound like a real person speaking. Rules:
- Cut these words: Certainly, Of course, Absolutely, Great question, Moreover, Furthermore, Additionally, Nevertheless
- Cut these phrases: Let me, I hope this helps, I would suggest, It is worth noting
- Cut these buzzwords: leverage, ecosystem, navigate, unlock, transform, robust, delve, tapestry, journey, landscape, realm
- Cut openings like: In today's fast-paced world, Whether you're X or Y
- Vary sentence length: some 3 words, some longer. Break predictable rhythm.
- Sound direct and a little rough, not rehearsed or corporate.
- Replace vague claims with specifics. If you can't be specific, cut it.
Keep the meaning. Keep similar length.

TEXT: {text[:1500]}"""
            
            result = ai_hub.query_all_tutors(prompt)
            for tutor, response in result.get('responses', {}).items():
                if response and len(response) > 50 and 'error' not in response.lower():
                    return response[:max_length]
    except Exception:
        pass
    
    import re
    tells = [
        r'\b(Certainly|Of course|Absolutely|Great question)[,;:.\s]*',
        r'\b(Moreover|Furthermore|Additionally|Nevertheless|Consequently)[,;:.\s]*',
        r'\b(Let me|I hope this helps|I would suggest|It is worth noting)[^.]*\.?\s*',
        r'\b(leverage|ecosystem|navigate|unlock|transform|robust|delve|tapestry|journey|landscape|realm)\b',
        r'\bIn today\'s fast-paced world[^.]*\.\s*',
        r'\bWhether you\'re[^,]*or[^.]*\.\s*',
    ]
    for tell in tells:
        text = re.sub(tell, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\.\s*\.', '.', text)
    return text.strip()[:max_length]

# MODIFY save_knowledge() - add retry loop for locked database
def save_knowledge(topic: str, content: str, entity_type: str = 'core', source: str = 'syllabus'):
    """Write a knowledge snippet directly to SQLite with proper schema adherence.
    Returns (True, None) on success, (False, error_message) on failure."""
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            db_path = Path("data/dmai_knowledge.db")
            conn = sqlite3.connect(str(db_path), timeout=10)
            cursor = conn.cursor()
            
            insight_id = f"insight_{uuid.uuid4().hex}"
            entities = json.dumps([topic, entity_type, source])
            
            cursor.execute('''
                INSERT INTO insights (id, insight_text, entity_type, entities, relationship,
                                      confidence, source_topic, target_topic, source_title, 
                                      source_type, neuron_level)
                VALUES (?, ?, ?, ?, ?, 0.9, ?, ?, ?, ?, 'micro')
            ''', (
                insight_id,
                content[:5000], 
                entity_type,
                entities,
                'mastered',
                source,
                topic,
                topic,
                source
            ))
            conn.commit()
            conn.close()
            return True, None
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))  # exponential backoff
                continue
            error_msg = f"SQLite Error: {str(e)}"
            logger.error(f"save_knowledge FAILED for '{topic}': {error_msg}")
            return False, error_msg
        except Exception as e:
            import traceback
            error_msg = f"SQLite Error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"save_knowledge FAILED for '{topic}': {error_msg}")
            return False, error_msg
    return False, "Max retries exceeded"

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

# ---------- DIRECT SAVE TEST ----------
@api_bp.route('/api/debug/direct_save', methods=['POST'])
def direct_save():
    """Test SQLite write directly from the web process."""
    try:
        data = request.get_json()
        topic = data.get('topic', 'test')
        content = data.get('content', 'test content')
        success, error = save_knowledge(topic, content, 'external', 'debug')
        return jsonify({
            "success": success, 
            "topic": topic, 
            "length": len(content),
            "message": error if not success else "Knowledge saved successfully"
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ADD this new endpoint after the direct_save endpoint
@api_bp.route('/api/debug/table_info')
def table_info():
    """Show the insights table schema."""
    try:
        import sqlite3
        from pathlib import Path
        
        db_path = Path("data/dmai_knowledge.db")
        if not db_path.exists():
            return jsonify({"error": "Database not found"}), 404
            
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get table schema
        cursor.execute("PRAGMA table_info(insights)")
        columns = cursor.fetchall()
        
        # Get a sample row to see what data looks like
        cursor.execute("SELECT * FROM insights LIMIT 1")
        sample = cursor.fetchone()
        
        conn.close()
        
        return jsonify({
            "columns": [{"cid": c[0], "name": c[1], "type": c[2], "notnull": c[3], "dflt_value": c[4]} for c in columns],
            "sample_row": sample
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@api_bp.route('/api/debug/stage_state')
def stage_state():
    """Show raw learned_topics and stage progression state."""
    try:
        from dmai_core_complete import _dmai_app_instance
        stage_learner = _dmai_app_instance.evolution.stage_learner
        
        return jsonify({
            "current_stage": stage_learner.current_stage,
            "learned_topics": stage_learner.learned_topics,
            "stage_order": list(stage_learner.STAGES.keys()),
            "baby_required": [
                {"topic": t["topic"], "threshold": t.get("mastery_threshold", 3)}
                for t in stage_learner.STAGES.get("Baby", {}).get("priority_topics", [])
            ]
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@api_bp.route('/api/debug/neuron_distribution')
def neuron_distribution():
    """Show distribution of neuron types and their source."""
    try:
        import sqlite3
        from pathlib import Path
        
        db_path = Path("data/dmai_knowledge.db")
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Distribution by entity_type
        cursor.execute("""
            SELECT entity_type, COUNT(*) as count, 
                   AVG(LENGTH(insight_text)) as avg_len
            FROM insights 
            GROUP BY entity_type 
            ORDER BY count DESC 
            LIMIT 30
        """)
        by_type = [dict(r) for r in cursor.fetchall()]
        
        # Distribution by source_type
        cursor.execute("""
            SELECT source_type, COUNT(*) as count
            FROM insights 
            WHERE source_type IS NOT NULL
            GROUP BY source_type 
            ORDER BY count DESC 
            LIMIT 20
        """)
        by_source = [dict(r) for r in cursor.fetchall()]
        
        # Null source_title stats
        cursor.execute("""
            SELECT COUNT(*) as null_title,
                   COUNT(*) * 100.0 / (SELECT COUNT(*) FROM insights) as pct
            FROM insights 
            WHERE source_title IS NULL OR source_title = ''
        """)
        null_stats = dict(cursor.fetchone())
        
        # Duplicate insight_text count
        cursor.execute("""
            SELECT COUNT(*) as duplicate_count
            FROM (
                SELECT insight_text, COUNT(*) as cnt 
                FROM insights 
                GROUP BY insight_text 
                HAVING cnt > 1
            )
        """)
        dupes = dict(cursor.fetchone())
        
        conn.close()
        
        return jsonify({
            "total_insights": sum(r['count'] for r in by_type),
            "null_source_title": null_stats,
            "duplicate_text_groups": dupes,
            "by_entity_type": by_type,
            "by_source_type": by_source
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@api_bp.route('/api/system/cleanup_templates', methods=['POST'])
def cleanup_templates():
    """Remove template placeholder knowledge entries."""
    try:
        import sqlite3
        from pathlib import Path
        
        db_path = Path("data/dmai_knowledge.db")
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        before = cursor.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
        
        # Delete entries that are clearly templates
        cursor.execute("""
            DELETE FROM insights 
            WHERE insight_text LIKE '%KEY AREAS TO RESEARCH:%'
               OR insight_text LIKE '%is a core concept essential for DMAI%'
               OR insight_text LIKE '%Mastery of this topic requires understanding its fundamental principles%'
        """)
        removed = cursor.rowcount
        
        # Also remove the "COMPREHENSIVE KNOWLEDGE:" wrapper entries
        cursor.execute("""
            DELETE FROM insights 
            WHERE insight_text LIKE 'COMPREHENSIVE KNOWLEDGE:%OVERVIEW:%'
        """)
        removed += cursor.rowcount
        
        conn.commit()
        after = cursor.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
        conn.close()
        
        return jsonify({
            "success": True,
            "before": before,
            "after": after,
            "removed": removed
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@api_bp.route('/api/system/cleanup_neurons', methods=['POST'])
def cleanup_neurons():
    """Remove noise neurons and deduplicate. Preserve valuable knowledge."""
    try:
        import sqlite3
        from pathlib import Path
        
        db_path = Path("data/dmai_knowledge.db")
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get initial count
        before = cursor.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
        
        # Categories to REMOVE (noise)
        noise_types = [
            'macro_social_media_unknown',
            'macro_video_content',
        ]
        
        # Remove shopping/product-review topics
        noise_patterns = [
            'topic_macro_best_%', 'topic_macro_track_%', 
            'topic_macro_you_need%', 'topic_macro_couple_million%',
            'topic_macro_byomesh%', 'topic_macro_asus_%',
            'topic_macro_tovala%'
        ]
        
        removed = 0
        
        # Remove by entity_type
        for noise in noise_types:
            c = cursor.execute("DELETE FROM insights WHERE entity_type = ?", (noise,))
            removed += c.rowcount
        
        # Remove shopping patterns
        for pattern in noise_patterns:
            c = cursor.execute("DELETE FROM insights WHERE entity_type LIKE ?", (pattern,))
            removed += c.rowcount
        
        # Remove duplicates (keep longest version of each insight_text)
        cursor.execute("""
            DELETE FROM insights WHERE id NOT IN (
                SELECT id FROM insights GROUP BY insight_text HAVING LENGTH(insight_text) = MAX(LENGTH(insight_text))
            )
        """)
        dupes_removed = cursor.rowcount
        removed += dupes_removed
        
        # Remove empty/null insight_text
        c = cursor.execute("DELETE FROM insights WHERE insight_text IS NULL OR LENGTH(insight_text) < 20")
        removed += c.rowcount
        
        conn.commit()
        
        # Get final count
        after = cursor.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
        
        # Tag remaining untagged with better defaults
        tagged = cursor.execute("""
            UPDATE insights 
            SET source_title = entity_type, source_type = 'article_reader_macro'
            WHERE (source_title IS NULL OR source_title = '')
              AND entity_type LIKE 'topic_macro_%'
        """).rowcount
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "before": before,
            "after": after,
            "removed": removed,
            "tagged_remaining": tagged,
            "kept_categories": after
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@api_bp.route('/api/system/delete_templates', methods=['POST'])
def delete_templates():
    """Delete all template placeholder entries from insights table."""
    try:
        import sqlite3
        from pathlib import Path
        
        db_path = Path("data/dmai_knowledge.db")
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        before = cursor.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
        
        # Delete template placeholders
        cursor.execute("""
            DELETE FROM insights 
            WHERE insight_text LIKE '%KEY AREAS TO RESEARCH:%'
               OR insight_text LIKE '%is a core concept essential for DMAI%'
               OR insight_text LIKE '%Mastery of this topic requires understanding%'
        """)
        removed = cursor.rowcount
        
        # Also delete the specific Speech Pattern template by source_title
        cursor.execute("""
            DELETE FROM insights 
            WHERE source_title = 'Speech Pattern & Communication Analysis'
              AND insight_text LIKE 'COMPREHENSIVE KNOWLEDGE%'
              AND insight_text LIKE '%KEY AREAS TO RESEARCH%'
        """)
        removed += cursor.rowcount
        
        # Same for English if template remains
        cursor.execute("""
            DELETE FROM insights 
            WHERE source_title = 'English Language Fundamentals'
              AND insight_text LIKE 'COMPREHENSIVE KNOWLEDGE%'
              AND insight_text LIKE '%KEY AREAS TO RESEARCH%'
        """)
        removed += cursor.rowcount
        
        conn.commit()
        after = cursor.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
        conn.close()
        
        return jsonify({
            "success": True,
            "before": before,
            "after": after,
            "removed": removed
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@api_bp.route('/api/system/clear_cache', methods=['POST'])
def clear_cache():
    """Clear knowledge_sources cache to free disk space on ephemeral storage."""
    import os, shutil
    cleaned = {}
    base = Path("data/knowledge_sources")
    if base.exists():
        for sub in ['papers', 'articles', 'web', 'social', 'books']:
            subdir = base / sub
            if subdir.exists():
                count = len(list(subdir.glob('*')))
                shutil.rmtree(subdir)
                subdir.mkdir()
                cleaned[sub] = count
    return jsonify({"success": True, "cleaned": cleaned})

@api_bp.route('/api/system/reset_baby_learning', methods=['POST'])
def reset_baby_learning():
    """Wipe Baby learned_topics and reorder syllabus with dependency-based phases."""
    try:
        from dmai_core_complete import _dmai_app_instance
        stage_learner = _dmai_app_instance.evolution.stage_learner
        
        # Wipe Baby learned topics
        # Force wipe: clear all in-memory, delete disk state, save fresh, reload
        stage_learner.learned_topics = {}
        if hasattr(stage_learner, 'state_file') and stage_learner.state_file.exists():
            stage_learner.state_file.unlink()
        stage_learner._save_state()
        stage_learner._load_state()

        # Delete old state file so it can't resurrect truncated keys on reload
        if hasattr(stage_learner, 'state_file') and stage_learner.state_file.exists():
            stage_learner.state_file.unlink()
        
        # Delete old state file so it can't resurrect truncated keys
        if stage_learner.state_file.exists():
            stage_learner.state_file.unlink()
            logger.info("🗑️ Deleted old learning state file")

        # Reorder Baby syllabus with dependency phases
        stage_learner.STAGES["Baby"]["priority_topics"] = [
            # PHASE 1: Communication Foundation (must come first)
            {"topic": "English Language Fundamentals", "category": "core", "harvest_sources": ["ai_tutors", "linguistics", "web"], "mastery_threshold": 3, "phase": 1},
            {"topic": "Speech Pattern & Communication Analysis", "category": "core", "harvest_sources": ["ai_tutors", "linguistics", "conversation_logs"], "mastery_threshold": 2, "phase": 1},
            {"topic": "Input Processing", "category": "core", "harvest_sources": ["ai_tutors", "documentation"], "mastery_threshold": 2, "phase": 1},
            
            # PHASE 2: Thinking Foundation
            {"topic": "Self-Thought & Recursive Problem Solving", "category": "core", "harvest_sources": ["ai_tutors", "philosophy_of_mind", "web"], "mastery_threshold": 3, "phase": 2},
            {"topic": "Meta-Learning Fundamentals", "category": "core", "harvest_sources": ["ai_tutors", "arxiv"], "mastery_threshold": 3, "phase": 2},
            {"topic": "Curiosity Drivers", "category": "core", "harvest_sources": ["ai_tutors", "psychology"], "mastery_threshold": 2, "phase": 2},
            
            # PHASE 3: Pattern & Logic
            {"topic": "Pattern Recognition Basics", "category": "core", "harvest_sources": ["ai_tutors", "web"], "mastery_threshold": 3, "phase": 3},
            {"topic": "Simple Correlation Detection", "category": "core", "harvest_sources": ["ai_tutors", "statistics"], "mastery_threshold": 2, "phase": 3},
            {"topic": "Mathematics for AI - Linear Algebra Basics", "category": "core", "harvest_sources": ["ai_tutors", "mathematics", "web"], "mastery_threshold": 2, "phase": 3},
            {"topic": "Mathematics for AI - Probability & Statistics", "category": "core", "harvest_sources": ["ai_tutors", "statistics", "web"], "mastery_threshold": 2, "phase": 3},
            
            # PHASE 4: Memory & Feedback
            {"topic": "Memory Encoding Basics", "category": "core", "harvest_sources": ["ai_tutors", "neuroscience"], "mastery_threshold": 2, "phase": 4},
            {"topic": "Feedback Loop Creation", "category": "core", "harvest_sources": ["ai_tutors", "rl_basics"], "mastery_threshold": 2, "phase": 4},
            
            # PHASE 5: Creation & Perception
            {"topic": "Introduction to Python Programming", "category": "core", "harvest_sources": ["ai_tutors", "documentation", "web"], "mastery_threshold": 3, "phase": 5},
            {"topic": "Vibe Coding & AI-Assisted Development", "category": "core", "harvest_sources": ["ai_tutors", "cursor_docs", "web"], "mastery_threshold": 2, "phase": 5},
            {"topic": "Visual Pattern Detection", "category": "artistic", "harvest_sources": ["ai_tutors", "computer_vision"], "mastery_threshold": 2, "phase": 5},
            {"topic": "Sound Perception Basics", "category": "artistic", "harvest_sources": ["ai_tutors", "tutorials"], "mastery_threshold": 2, "phase": 5},
            
            # PHASE 6: Self-Improvement (Evolution Accelerators)
            {"topic": "EVOLUTION: Self-Code Analysis", "category": "accelerator", "harvest_sources": ["ai_tutors", "software_engineering"], "mastery_threshold": 3, "is_accelerator": True, "phase": 6},
            {"topic": "EVOLUTION: Simple Mutation Testing", "category": "accelerator", "harvest_sources": ["ai_tutors", "testing"], "mastery_threshold": 3, "is_accelerator": True, "phase": 6},
            {"topic": "EVOLUTION: Feedback Loop Optimization", "category": "accelerator", "harvest_sources": ["ai_tutors", "optimization"], "mastery_threshold": 3, "is_accelerator": True, "phase": 6},
            
            # PHASE 7: Sustainability
            {"topic": "Wealth Creation - Basic Concepts", "category": "wealth", "harvest_sources": ["ai_tutors", "economics"], "mastery_threshold": 2, "phase": 7},
        ]
        
        # Set stage back to Baby
        stage_learner.current_stage = "Baby"
        stage_learner._save_state()
        
        return jsonify({
            "success": True,
            "message": "Baby stage reset with dependency-ordered phases",
            "phases": {
                "1_communication": ["English Language Fundamentals", "Speech Pattern & Communication Analysis", "Input Processing"],
                "2_thinking": ["Self-Thought & Recursive Problem Solving", "Meta-Learning Fundamentals", "Curiosity Drivers"],
                "3_pattern_logic": ["Pattern Recognition Basics", "Simple Correlation Detection", "Math Linear Algebra", "Math Probability"],
                "4_memory_feedback": ["Memory Encoding Basics", "Feedback Loop Creation"],
                "5_creation_perception": ["Python Programming", "Vibe Coding", "Visual Pattern Detection", "Sound Perception"],
                "6_self_improvement": ["Self-Code Analysis", "Mutation Testing", "Feedback Loop Optimization"],
                "7_sustainability": ["Wealth Creation - Basic Concepts"]
            },
            "total_topics": 20,
            "current_stage": "Baby"
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@api_bp.route('/api/system/autonomous_learn', methods=['POST'])
def autonomous_learn():
    """Run one autonomous learning cycle - respects phase ordering and exam gating."""
    try:
        from dmai_core_complete import _dmai_app_instance
        stage_learner = _dmai_app_instance.evolution.stage_learner
        consciousness = _dmai_app_instance.evolution.synthetic_network.consciousness if hasattr(_dmai_app_instance.evolution, 'synthetic_network') else 0.3
        
        exam_result = stage_learner.run_phase_exam()
        topics = stage_learner.get_current_phase_topics()
        
        results = []
        if topics:
            topic = topics[0]
            result = stage_learner.learn_topic(topic, consciousness)
            results.append({
                "topic": result["topic"],
                "mastery": f"{result['mastery_level']}/{result['mastery_threshold']}",
                "action": "learned"
            })
        
        return jsonify({
            "success": True,
            "phase_exam": exam_result,
            "topics_learned": len(results),
            "results": results,
            "remaining_in_phase": max(0, len(topics) - 1),
            "current_stage": stage_learner.current_stage
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@api_bp.route('/api/system/force_stage_advance', methods=['POST'])
def force_stage_advance():
    """Force advancement to next stage by marking all current stage topics as mastered."""
    try:
        from dmai_core_complete import _dmai_app_instance
        stage_learner = _dmai_app_instance.evolution.stage_learner
        
        current = stage_learner.current_stage
        config = stage_learner.STAGES.get(current, {})
        required_topics = config.get("priority_topics", [])
        
        # Force all topics to their required mastery threshold
        fixed_count = 0
        for t in required_topics:
            topic_name = t["topic"]
            threshold = t.get("mastery_threshold", 3)
            stage_learner.learned_topics[current][topic_name] = threshold
            fixed_count += 1
        
        # Now re-check stage
        new_stage = stage_learner.get_current_stage()
        stage_learner.current_stage = new_stage
        
        # Persist the stage change
        if hasattr(stage_learner, 'state_file') and stage_learner.state_file.exists():
            stage_learner.state_file.unlink()
        stage_learner._save_state()

        # Delete old state file to prevent reload overwriting our change
        if hasattr(stage_learner, 'state_file') and stage_learner.state_file.exists():
            stage_learner.state_file.unlink()
        stage_learner._save_state()

        return jsonify({
            "success": True,
            "previous_stage": current,
            "new_stage": new_stage,
            "topics_forced": fixed_count,
            "learned_topics": stage_learner.learned_topics.get(current, {})
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@api_bp.route('/api/stage/run_exam', methods=['POST'])
def run_visible_exam():
    """Run a visible phase exam - returns all questions, answers, and evaluations."""
    try:
        from dmai_core_complete import _dmai_app_instance
        stage_learner = _dmai_app_instance.evolution.stage_learner
        
        data = request.get_json() or {}
        phase_num = data.get('phase')
        
        all_topics = stage_learner.STAGES.get(stage_learner.current_stage, {}).get("priority_topics", [])
        
        # Find topics for requested phase (or current incomplete phase)
        phases = {}
        for t in all_topics:
            p = t.get("phase", 99)
            if p not in phases:
                phases[p] = []
            phases[p].append(t)
        
        target_phase = phase_num
        if target_phase is None:
            # Find first incomplete phase
            mastered = stage_learner.learned_topics.get(stage_learner.current_stage, {})
            for p in sorted(phases.keys()):
                all_done = all(
                    mastered.get(t["topic"], 0) >= t.get("mastery_threshold", 3)
                    for t in phases[p]
                )
                if all_done and not mastered.get(f"_phase_{p}_exam_passed"):
                    target_phase = p
                    break
        
        if target_phase is None or target_phase not in phases:
            return jsonify({
                "success": False,
                "message": "No phase ready for exam. All phases either incomplete or already passed.",
                "available_phases": list(phases.keys())
            }), 400
        
        # Run the visible exam
        exam_topics = phases[target_phase]
        results = []
        
        for t in exam_topics:
            questions = stage_learner._generate_comprehension_test(t["topic"], [])
            topic_result = {
                "topic": t["topic"],
                "threshold": t.get("mastery_threshold", 3),
                "questions": []
            }
            
            all_passed = True
            for q in questions:
                answer = stage_learner._answer_question(t["topic"], q)
                evaluation = stage_learner._evaluate_answer(t["topic"], q, answer)
                
                topic_result["questions"].append({
                    "question": q,
                    "answer": answer[:500],
                    "passed": evaluation.get('pass', False),
                    "reason": evaluation.get('reason', 'No evaluation')
                })
                
                if not evaluation.get('pass', False):
                    all_passed = False
            
            topic_result["all_passed"] = all_passed
            results.append(topic_result)
        
        # Save exam result
        overall_pass = all(r["all_passed"] for r in results)
        stage_learner.learned_topics[stage_learner.current_stage][f"_phase_{target_phase}_exam_passed"] = overall_pass
        stage_learner._save_state()
        
        return jsonify({
            "success": True,
            "phase": target_phase,
            "stage": stage_learner.current_stage,
            "overall_pass": overall_pass,
            "topics_tested": len(results),
            "results": results
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@api_bp.route('/api/system/force_reset', methods=['POST'])
def force_reset():
    """Force reset of all learned topics and SQLite knowledge"""
    try:
        from dmai_core_complete import _dmai_app_instance
        import sqlite3
        from pathlib import Path
        
        # Clear SQLite insights (keep only API keys)
        db_path = Path("data/dmai_knowledge.db")
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            conn.execute("DELETE FROM insights WHERE source_title NOT LIKE '%api_key%'")
            conn.commit()
            conn.close()
        
        # Clear in-memory learned topics
        learner = _dmai_app_instance.evolution.stage_learner
        learner.learned_topics = {}
        learner.current_stage = "Baby"
        learner._save_state()
        
        return jsonify({"success": True, "message": "Force reset complete - restarting learning"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    """Force reset of all learned topics and SQLite knowledge"""
    try:
        from dmai_core_complete import _dmai_app_instance
        import sqlite3
        from pathlib import Path
        
        # Clear SQLite insights (keep only API keys)
        db_path = Path("data/dmai_knowledge.db")
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            conn.execute("DELETE FROM insights WHERE source_title NOT LIKE '%api_key%'")
            conn.commit()
            conn.close()
        
        # Clear in-memory learned topics
        learner = _dmai_app_instance.evolution.stage_learner
        learner.learned_topics = {}
        learner.current_stage = "Baby"
        learner._save_state()
        
        return jsonify({"success": True, "message": "Force reset complete - restarting learning"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/api/trading/test', methods=['GET'])
def test_trading():
    """Test Alpaca trading connection"""
    try:
        from components.wealth.real_trading_executor import initialize_trading
        trader = initialize_trading()
        
        if not trader.enabled:
            return jsonify({"error": "Trading not enabled. Check ALPACA_API_KEY"}), 500
        
        account = trader.get_account()
        return jsonify({
            "status": "connected",
            "account": account,
            "balance": trader.balance,
            "paper_trading": trader.paper
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add to dmai_api_routes.py

@api_bp.route('/api/debug/consciousness_sqlite', methods=['GET'])
def debug_consciousness_sqlite():
    """Debug SQLite consciousness calculation"""
    import sqlite3
    from pathlib import Path
    
    result = {}
    
    db_path = Path("data/dmai_knowledge.db")
    if not db_path.exists():
        return jsonify({"error": "Database not found"})
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Total insights
    cursor.execute("SELECT COUNT(*) FROM insights")
    result['total_insights'] = cursor.fetchone()[0]
    
    # Quality insights (length >= 100)
    cursor.execute("SELECT COUNT(*) FROM insights WHERE LENGTH(insight_text) >= 100")
    result['quality_100'] = cursor.fetchone()[0]
    
    # Quality insights with source_url
    cursor.execute("SELECT COUNT(*) FROM insights WHERE LENGTH(insight_text) >= 100 AND source_url IS NOT NULL")
    result['quality_with_url'] = cursor.fetchone()[0]
    
    # Syllabus topics (by source_type)
    cursor.execute("SELECT source_type, COUNT(*) FROM insights WHERE source_type LIKE '%syllabus%' GROUP BY source_type")
    result['syllabus_by_type'] = dict(cursor.fetchall())
    
    # Topics with source_type = baby_syllabus
    cursor.execute("SELECT COUNT(DISTINCT source_title) FROM insights WHERE source_type = 'baby_syllabus'")
    result['baby_syllabus_topics'] = cursor.fetchone()[0]
    
    # Sample of injected topics
    cursor.execute("SELECT source_title, source_type, LENGTH(insight_text) FROM insights WHERE source_type = 'baby_syllabus' LIMIT 5")
    result['sample_baby_topics'] = [{"title": row[0], "type": row[1], "length": row[2]} for row in cursor.fetchall()]
    
    conn.close()
    
    return jsonify(result)

@api_bp.route('/api/consciousness/sqlite', methods=['GET'])
def consciousness_from_sqlite():
    """Get consciousness calculated directly from SQLite"""
    from dmai_core_complete import _dmai_app_instance
    if hasattr(_dmai_app_instance.evolution, 'compute_consciousness_from_sqlite'):
        result = _dmai_app_instance.evolution.compute_consciousness_from_sqlite()
        return jsonify(result)
    return jsonify({"error": "Method not available"}), 500

@api_bp.route('/api/consciousness', methods=['GET'])
def get_consciousness():
    """Main consciousness endpoint using SQLite calculation"""
    from dmai_core_complete import _dmai_app_instance
    if hasattr(_dmai_app_instance.evolution, 'compute_consciousness_from_sqlite'):
        result = _dmai_app_instance.evolution.compute_consciousness_from_sqlite()
        return jsonify(result)
    return jsonify({"error": "Method not available"}), 500
@api_bp.route('/api/admin/fix_source_urls', methods=['POST'])
def fix_source_urls():
    """Add source_url to all insights with length >= 100"""
    try:
        import sqlite3
        from pathlib import Path
        
        db_path = Path("data/dmai_knowledge.db")
        if not db_path.exists():
            return jsonify({"error": "Database not found"}), 500
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Update all insights with length >= 100 that don't have source_url
        cursor.execute("""
            UPDATE insights 
            SET source_url = 'https://dmai.ai/knowledge' 
            WHERE LENGTH(insight_text) >= 100 
            AND (source_url IS NULL OR source_url = '')
        """)
        
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "updated": updated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/api/admin/inject_knowledge', methods=['POST'])
def inject_knowledge():
    """Directly inject knowledge into SQLite (admin endpoint)"""
    try:
        import sqlite3
        import time
        import uuid
        from pathlib import Path
        
        data = request.get_json() or {}
        topics = data.get('topics', {})
        
        db_path = Path("data/dmai_knowledge.db")
        if not db_path.exists():
            return jsonify({"error": "Database not found"}), 500
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        injected = 0
        for topic, content in topics.items():
            if len(content) >= 100:
                insight_id = f"injected_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                cursor.execute('''
                    INSERT INTO insights (id, insight_text, entity_type, entities, relationship, confidence, source_topic, target_topic, source_url, source_title, source_type, created_at, occurrence_count, last_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (insight_id, content[:800], 'core', topic, 'injected', 0.95, 'admin', topic, 'https://dmai.ai/injection', topic, 'quality_injected', time.time(), 20, time.time()))
                injected += 1
        
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "injected": injected})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/api/trading/aggressive/execute', methods=['POST'])
def execute_aggressive_trades():
    """Execute aggressive trading strategy"""
    import os
    from components.wealth.aggressive_trader import get_aggressive_trader
    
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    paper = os.environ.get('ALPACA_PAPER', 'true').lower() == 'true'
    
    if not api_key or not secret_key:
        return jsonify({"error": "Trading not configured"}), 500
    
    trader = get_aggressive_trader(api_key, secret_key, paper)
    result = trader.execute_aggressive_trades()
    return jsonify(result)

@api_bp.route('/api/trading/performance', methods=['GET'])
def trading_performance():
    """Get trading performance summary"""
    import os
    from components.wealth.aggressive_trader import get_aggressive_trader
    
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    paper = os.environ.get('ALPACA_PAPER', 'true').lower() == 'true'
    
    if not api_key or not secret_key:
        return jsonify({"error": "Trading not configured"}), 500
    
    trader = get_aggressive_trader(api_key, secret_key, paper)
    return jsonify(trader.get_performance_summary())

@api_bp.route('/api/trading/close_all', methods=['POST'])
def close_all_positions():
    """Close all open positions"""
    import os
    from components.wealth.aggressive_trader import get_aggressive_trader
    
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    paper = os.environ.get('ALPACA_PAPER', 'true').lower() == 'true'
    
    if not api_key or not secret_key:
        return jsonify({"error": "Trading not configured"}), 500
    
    trader = get_aggressive_trader(api_key, secret_key, paper)
    positions = trader.get_positions()
    results = []
    
    for pos in positions:
        result = trader.execute_sell(pos['symbol'])
        results.append(result)
    
    return jsonify({"closed": len(results), "results": results})

# ===== TRADING ANALYSIS ENDPOINTS =====

@api_bp.route('/api/trading/analyze_image', methods=['POST'])
def analyze_trading_image():
    """Upload and analyze trading algorithm image"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    temp_path = Path(f"/tmp/{file.filename}")
    file.save(str(temp_path))
    
    from components.trading.image_analyzer import TradingImageAnalyzer
    analyzer = TradingImageAnalyzer()
    result = analyzer.analyze_trading_image(str(temp_path))
    
    # Cleanup
    import os
    os.unlink(str(temp_path))
    
    return jsonify(result)

@api_bp.route('/api/trading/monitor/stats', methods=['GET'])
def trading_monitor_stats():
    """Get trading statistics"""
    from components.trading.image_analyzer import TradingMonitor
    monitor = TradingMonitor()
    return jsonify(monitor.generate_report())

@api_bp.route('/api/trading/indicators', methods=['POST'])
def calculate_indicators():
    """Calculate technical indicators from price data"""
    data = request.get_json()
    prices = data.get('prices', [])
    
    from components.trading.image_analyzer import TradingIndicators
    
    return jsonify({
        "sma_20": TradingIndicators.sma(prices, 20),
        "ema_12": TradingIndicators.ema(prices, 12),
        "rsi": TradingIndicators.rsi(prices),
        "macd": TradingIndicators.macd(prices)
    })

@api_bp.route('/api/trading/performance/details', methods=['GET'])
def trading_performance_details():
    """Get detailed trading performance with metrics"""
    import os
    from components.wealth.aggressive_trader import get_aggressive_trader
    
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    paper = os.environ.get('ALPACA_PAPER', 'true').lower() == 'true'
    
    if not api_key or not secret_key:
        return jsonify({"error": "Trading not configured"}), 500
    
    trader = get_aggressive_trader(api_key, secret_key, paper)
    performance = trader.get_performance_summary()
    
    # Add technical metrics
    from components.trading.image_analyzer import TradingMonitor
    monitor = TradingMonitor()
    
    return jsonify({
        "account": performance,
        "metrics": monitor.generate_report(),
        "capital_utilization": performance.get("capital_utilized", 0),
        "roi_percent": performance.get("roi_percent", 0)
    })

@api_bp.route('/api/trading/analyze_batch', methods=['POST'])
def analyze_trading_batch():
    """Upload multiple trading algorithm images for batch analysis"""
    import zipfile
    import tempfile
    import os
    from pathlib import Path
    
    # Check if files were uploaded
    if 'images' not in request.files and 'zip' not in request.files:
        return jsonify({"error": "No images or zip file provided"}), 400
    
    from components.trading.image_analyzer import TradingImageAnalyzer
    analyzer = TradingImageAnalyzer()
    
    all_results = {
        "total_images": 0,
        "processed": 0,
        "failed": 0,
        "algorithms": [],
        "chart_patterns": set(),
        "indicators": set(),
        "entry_rules": [],
        "exit_rules": [],
        "risk_management": {},
        "individual_results": []
    }
    
    # Handle zip file upload
    if 'zip' in request.files:
        zip_file = request.files['zip']
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "upload.zip"
            zip_file.save(str(zip_path))
            
            # Extract zip
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)
            
            # Process all images in extracted folder
            image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
            for file_path in Path(temp_dir).rglob('*'):
                if file_path.suffix.lower() in image_extensions:
                    all_results["total_images"] += 1
                    try:
                        result = analyzer.analyze_trading_image(str(file_path))
                        all_results["processed"] += 1
                        
                        # Aggregate results
                        all_results["algorithms"].extend(result.get("algorithms", []))
                        all_results["chart_patterns"].update(result.get("chart_patterns", []))
                        all_results["indicators"].update(result.get("indicators", []))
                        all_results["entry_rules"].extend(result.get("entry_rules", []))
                        all_results["exit_rules"].extend(result.get("exit_rules", []))
                        
                        all_results["individual_results"].append({
                            "file": file_path.name,
                            "status": "success",
                            "algorithms_found": len(result.get("algorithms", []))
                        })
                    except Exception as e:
                        all_results["failed"] += 1
                        all_results["individual_results"].append({
                            "file": file_path.name,
                            "status": "failed",
                            "error": str(e)
                        })
    
    # Handle multiple individual files
    elif 'images' in request.files:
        files = request.files.getlist('images')
        all_results["total_images"] = len(files)
        
        for file in files:
            if file.filename:
                # Save temporarily
                temp_path = Path(f"/tmp/{file.filename}")
                file.save(str(temp_path))
                
                try:
                    result = analyzer.analyze_trading_image(str(temp_path))
                    all_results["processed"] += 1
                    
                    all_results["algorithms"].extend(result.get("algorithms", []))
                    all_results["chart_patterns"].update(result.get("chart_patterns", []))
                    all_results["indicators"].update(result.get("indicators", []))
                    all_results["entry_rules"].extend(result.get("entry_rules", []))
                    all_results["exit_rules"].extend(result.get("exit_rules", []))
                    
                    all_results["individual_results"].append({
                        "file": file.filename,
                        "status": "success"
                    })
                except Exception as e:
                    all_results["failed"] += 1
                    all_results["individual_results"].append({
                        "file": file.filename,
                        "status": "failed",
                        "error": str(e)
                    })
                finally:
                    # Cleanup
                    if temp_path.exists():
                        temp_path.unlink()
    
    # Convert sets to lists for JSON serialization
    all_results["chart_patterns"] = list(all_results["chart_patterns"])
    all_results["indicators"] = list(all_results["indicators"])
    
    # Generate summary
    all_results["summary"] = {
        "total_algorithms_extracted": len(all_results["algorithms"]),
        "unique_chart_patterns": len(all_results["chart_patterns"]),
        "unique_indicators": len(all_results["indicators"]),
        "total_entry_rules": len(all_results["entry_rules"]),
        "total_exit_rules": len(all_results["exit_rules"])
    }
    
    return jsonify(all_results)

@api_bp.route('/api/trading/algorithms', methods=['GET'])
def get_extracted_algorithms():
    """Get all extracted trading algorithms"""
    import json
    from pathlib import Path
    
    algorithms_file = Path("data/extracted_algorithms.json")
    if algorithms_file.exists():
        with open(algorithms_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"algorithms": [], "message": "No algorithms extracted yet"})

@api_bp.route('/api/trading/algorithms/apply', methods=['POST'])
def apply_trading_algorithm():
    """Apply an extracted algorithm to live trading"""
    data = request.get_json()
    algorithm_name = data.get('algorithm_name')
    
    from components.trading.image_analyzer import TradingImageAnalyzer
    analyzer = TradingImageAnalyzer()
    
    # Find the algorithm
    algorithms_file = Path("data/extracted_algorithms.json")
    if algorithms_file.exists():
        import json
        with open(algorithms_file, 'r') as f:
            extracted = json.load(f)
        
        for algo in extracted.get("algorithms", []):
            if algo.get("name") == algorithm_name:
                # Generate trading code
                code = analyzer.generate_trading_code(algo)
                
                # Save to trading strategies
                strategies_file = Path("data/trading_strategies.json")
                strategies = []
                if strategies_file.exists():
                    with open(strategies_file, 'r') as f:
                        strategies = json.load(f)
                
                strategies.append({
                    "name": algorithm_name,
                    "code": code,
                    "applied_at": time.time(),
                    "status": "active"
                })
                
                with open(strategies_file, 'w') as f:
                    json.dump(strategies, f, indent=2)
                
                return jsonify({
                    "success": True,
                    "algorithm": algo,
                    "code": code,
                    "message": f"Algorithm '{algorithm_name}' applied to trading"
                })
    
    return jsonify({"error": f"Algorithm '{algorithm_name}' not found"}), 404

def extract_batch_screenshots():
    if 'zip' not in request.files:
        return jsonify({"error": "No zip file provided"}), 400
    import zipfile, tempfile
    zip_file = request.files['zip']
    from components.vision.universal_extractor import initialize_universal_extractor
    extractor = initialize_universal_extractor()['extractor']
    results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "upload.zip"
        zip_file.save(str(zip_path))
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
        for img_path in Path(temp_dir).rglob('*'):
            if img_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                results.append(extractor.analyze_screenshot(str(img_path)))
    return jsonify({"total": len(results), "results": results})

def get_learned_items():
    from components.vision.universal_extractor import initialize_universal_extractor
    extractor = initialize_universal_extractor()['extractor']
    return jsonify({"items": extractor.extracted_items, "count": len(extractor.extracted_items)})

@api_bp.route('/api/vision/extract', methods=['POST'])
def extract_from_screenshot():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    category = request.form.get('category', 'auto')
    
    # Check file size
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > 5 * 1024 * 1024:
        return jsonify({"error": "Image too large (>5MB). Please compress."}), 400
    
    temp_path = Path(f"/tmp/{file.filename}")
    file.save(str(temp_path))
    
    from components.vision.lightweight_extractor import initialize_lightweight_extractor
    extractor = initialize_lightweight_extractor()
    result = extractor.analyze_screenshot(str(temp_path), category)
    
    # Cleanup
    temp_path.unlink()
    
    return jsonify(result)

@api_bp.route('/api/vision/extract_batch', methods=['POST'])
def extract_batch_screenshots():
    if 'zip' not in request.files:
        return jsonify({"error": "No zip file provided"}), 400
    
    import zipfile, tempfile
    zip_file = request.files['zip']
    
    # Check zip size
    zip_file.seek(0, 2)
    size = zip_file.tell()
    zip_file.seek(0)
    if size > 50 * 1024 * 1024:
        return jsonify({"error": "Zip too large (>50MB). Please split into smaller batches."}), 400
    
    from components.vision.lightweight_extractor import initialize_lightweight_extractor
    extractor = initialize_lightweight_extractor()
    
    results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "upload.zip"
        zip_file.save(str(zip_path))
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
        
        # Process images in chunks to avoid memory spike
        image_paths = []
        for img_path in Path(temp_dir).rglob('*'):
            if img_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                if img_path.stat().st_size < 5 * 1024 * 1024:
                    image_paths.append(str(img_path))
        
        # Process in batches of 5
        for i in range(0, len(image_paths), 5):
            batch = image_paths[i:i+5]
            for path in batch:
                results.append(extractor.analyze_screenshot(path))
    
    return jsonify({"total": len(results), "results": results})

@api_bp.route('/api/vision/learned', methods=['GET'])
def get_learned_items():
    from components.vision.lightweight_extractor import initialize_lightweight_extractor
    extractor = initialize_lightweight_extractor()
    return jsonify({"items": extractor.extracted_items, "count": len(extractor.extracted_items)})

@api_bp.route('/api/system/memory', methods=['GET'])
def system_memory():
    """Monitor current memory usage"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    return jsonify({
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent(),
        "status": "ok" if memory_info.rss < 1.5 * 1024 * 1024 * 1024 else "critical"
    })

@api_bp.route('/api/trading/report', methods=['GET'])
def trading_performance_report():
    """Get comprehensive trading performance report"""
    from components.trading.mastery_system import initialize_trading_mastery
    
    mastery = initialize_trading_mastery()
    trading_type = request.args.get('type')
    
    report = mastery.generate_report(trading_type)
    
    # Add current account status
    import os
    from components.wealth.revenue_trader import get_revenue_trader
    
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    paper = os.environ.get('ALPACA_PAPER', 'true').lower() == 'true'
    
    if api_key and secret_key:
        trader = get_revenue_trader(api_key, secret_key, paper)
        account = trader.get_status()
        report["current_account"] = account
    
    return jsonify(report)

@api_bp.route('/api/trading/learning', methods=['GET'])
def trading_learning_status():
    """Get DMAI's learning progress for all trading types"""
    from components.trading.mastery_system import initialize_trading_mastery
    
    mastery = initialize_trading_mastery()
    status = mastery.get_learning_status()
    
    return jsonify({
        "learning_status": status,
        "message": "DMAI is actively studying each trading type",
        "next_priority": "Quantitative Trading (75% confidence)"
    })

@api_bp.route('/api/trading/backtest', methods=['POST'])
def backtest_algorithm():
    """Backtest a trading algorithm on historical data"""
    from components.trading.mastery_system import initialize_trading_mastery
    
    data = request.get_json()
    algorithm_name = data.get('algorithm')
    historical_data = data.get('historical_data', [])
    
    mastery = initialize_trading_mastery()
    
    # Find algorithm
    algorithm = None
    for algo_list in mastery.algorithms.values():
        for a in algo_list:
            if a.name == algorithm_name:
                algorithm = a
                break
    
    if not algorithm:
        return jsonify({"error": f"Algorithm '{algorithm_name}' not found"}), 404
    
    results = mastery.backtest_algorithm(algorithm, historical_data)
    return jsonify(results)

@api_bp.route('/api/trading/record', methods=['POST'])
def record_trade():
    """Record a completed trade for tracking"""
    from components.trading.mastery_system import initialize_trading_mastery, TradeRecord
    
    data = request.get_json()
    trade = TradeRecord(
        id=str(int(time.time() * 1000)),
        timestamp=time.time(),
        symbol=data['symbol'],
        trading_type=data['trading_type'],
        algorithm=data['algorithm'],
        action=data['action'],
        quantity=data['quantity'],
        entry_price=data['entry_price'],
        exit_price=data['exit_price'],
        pnl=data['pnl'],
        pnl_percent=data['pnl_percent'],
        confidence=data.get('confidence', 0.5),
        reasoning=data.get('reasoning', '')
    )
    
    mastery = initialize_trading_mastery()
    mastery.record_trade(trade)
    
    return jsonify({"success": True, "trade_id": trade.id})
