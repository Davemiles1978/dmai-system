"""Flask Blueprint – safe addition, zero main‑file edits."""
import logging, traceback
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)
api_bp = Blueprint('custom_api', __name__)

# ---------- health ----------
@api_bp.route('/api/ping')
def ping():
    return jsonify({"status": "ok"})

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
