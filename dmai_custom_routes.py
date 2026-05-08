"""Custom DMAI endpoints – safely added without editing the core handler."""
import logging, traceback
from flask import request, jsonify

logger = logging.getLogger(__name__)

def add_custom_routes(app, dmai_instance):
    evolution = dmai_instance.evolution

    @app.route('/api/tutors/add_key', methods=['POST'])
    def tutors_add_key():
        try:
            data = request.get_json()
            if not data or 'provider' not in data or 'key' not in data:
                return jsonify({"error": "Missing 'provider' or 'key' in body"}), 400
            provider = data['provider']
            api_key = data['key']
            source = data.get('source', 'manual')
            if not hasattr(dmai_instance, 'api_key_store') or dmai_instance.api_key_store is None:
                return jsonify({"error": "API key store not initialized"}), 500
            is_new = dmai_instance.api_key_store.add_key(provider, api_key, source=source)
            configured = None
            if hasattr(evolution, 'tutor_configurator') and evolution.tutor_configurator:
                configured = evolution.tutor_configurator.configure_tutor(provider, api_key)
            return jsonify({"success": True, "new": is_new, "configured": configured is not None, "config_result": configured})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/ingest_knowledge', methods=['POST'])
    def ingest_knowledge():
        try:
            data = request.get_json()
            if not data or 'topic' not in data or 'category' not in data or 'content' not in data:
                return jsonify({"error": "Missing 'topic', 'category', or 'content'"}), 400
            topic = data['topic']
            category = data['category']
            content = data['content']
            stage_learner = evolution.stage_learner
            result = stage_learner.ingest_external_knowledge(topic, category, content)
            return jsonify({"success": True, "result": result})
        except Exception as e:
            logger.error(f"Ingest knowledge error: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

    logger.info("✅ Custom DMAI routes registered.")
