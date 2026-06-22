"""
DMAI Core Complete v6.0.0
==========================
Full production Flask application - wires ALL components together.

Run locally:   python dmai_core_complete.py
Run on Render: gunicorn dmai_core_complete:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 2

Environment variables: see .env.template
"""

import os
import sys
import json
import logging
import asyncio
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS

# Logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dmai.core")

# Render / production flags
IS_RENDER = os.environ.get("RENDER", "false").lower() == "true"
if IS_RENDER:
    os.environ["DISABLE_NEO4J"] = "true"
    os.environ["DISABLE_AUTO_THREADS"] = "true"

# Data path
DATA_PATH = os.environ.get("DATA_PATH", "data/")
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

# Startup time
STARTUP_TIME = datetime.now(timezone.utc)

# Component registry
components: Dict[str, Any] = {}

# Syllabus
try:
    from dmai_syllabus_data import SYLLABUS_TOPICS, TOTAL_TOPICS
    logger.info("Syllabus loaded: %d topics", TOTAL_TOPICS)
except Exception as e:
    logger.warning("Syllabus load failed: %s", e)
    SYLLABUS_TOPICS = {}
    TOTAL_TOPICS = 0

# SICore
try:
    from components.si_core import SICore
    components["si_core"] = SICore(data_path=Path(DATA_PATH))
    logger.info("SICore initialised")
except Exception as e:
    logger.warning("SICore failed: %s", e)

# AI Integration Hub
try:
    from components.phase11.AIIntegrationHub import AIIntegrationHub
    components["ai_hub"] = AIIntegrationHub(data_path=DATA_PATH)
    logger.info("AIIntegrationHub initialised")
except Exception as e:
    logger.warning("AIIntegrationHub failed: %s", e)

# Extended AI Integration Hub
try:
    from components.phase11.ExtendedAIIntegrationHub import ExtendedAIIntegrationHub
    components["extended_hub"] = ExtendedAIIntegrationHub(
        data_path=DATA_PATH,
        base_hub=components.get("ai_hub"),
    )
    logger.info("ExtendedAIIntegrationHub initialised")
except Exception as e:
    logger.warning("ExtendedAIIntegrationHub failed: %s", e)

# Evolution Training System
try:
    from components.evolution_training.EvolutionTrainingSystem import EvolutionTrainingSystem
    components["evolution_training"] = EvolutionTrainingSystem(
        si_core=components.get("si_core"),
        knowledge_graph=None,
        training_systems={},
    )
    logger.info("EvolutionTrainingSystem initialised")
except Exception as e:
    logger.warning("EvolutionTrainingSystem failed: %s", e)

# Synthetic Intelligence Training (legacy consciousness modules)
try:
    from components.si_training.SyntheticIntelligenceTraining import SyntheticIntelligenceTraining
    components["si_training_legacy"] = SyntheticIntelligenceTraining(
        data_path=DATA_PATH,
        si_core=components.get("si_core"),
    )
    logger.info("SyntheticIntelligenceTraining initialised")
except Exception as e:
    logger.warning("SyntheticIntelligenceTraining failed: %s", e)

# LLM Training
try:
    from components.llm_training.LLMTrainingProgram import LLMTrainingProgram
    components["llm_training"] = LLMTrainingProgram(data_path=DATA_PATH)
    logger.info("LLMTrainingProgram initialised")
except Exception as e:
    logger.warning("LLMTrainingProgram failed: %s", e)

# GenAI Training
try:
    from components.genai_training.GenAITrainingProgram import GenAITrainingProgram
    components["genai_training"] = GenAITrainingProgram(data_path=DATA_PATH)
    logger.info("GenAITrainingProgram initialised")
except Exception as e:
    logger.warning("GenAITrainingProgram failed: %s", e)

# Media Production Studio
try:
    from components.media.MediaProductionStudio import MediaProductionStudio
    components["media_studio"] = MediaProductionStudio()
    logger.info("MediaProductionStudio initialised")
except Exception as e:
    logger.warning("MediaProductionStudio failed: %s", e)

# Voice Integration
try:
    from components.voice.VoiceIntegration import VoiceIntegration
    components["voice"] = VoiceIntegration(data_path=Path(DATA_PATH))
    logger.info("VoiceIntegration initialised")
except Exception as e:
    logger.warning("VoiceIntegration failed: %s", e)

# Alex Riviera Content Generator
try:
    from components.alex_riviera.content_generator import AlexRivieraContent
    components["content_gen"] = AlexRivieraContent(ai_hub=components.get("ai_hub"))
    logger.info("AlexRivieraContent initialised")
except Exception as e:
    logger.warning("AlexRivieraContent failed: %s", e)

# Alex Riviera Publishing
try:
    from components.alex_riviera.publishing_orchestrator import AlexRivieraPublishing
    components["publishing"] = AlexRivieraPublishing()
    logger.info("AlexRivieraPublishing initialised")
except Exception as e:
    logger.warning("AlexRivieraPublishing failed: %s", e)

# Master Training Orchestrator
try:
    from components.orchestrator.DMAITrainingOrchestrator import (
        DMAITrainingOrchestrator, register_orchestrator_routes
    )
    components["training_orchestrator"] = DMAITrainingOrchestrator(
        data_path=DATA_PATH,
        si_core=components.get("si_core"),
        knowledge_graph=None,
        ai_hub=components.get("ai_hub"),
        evolution_system=components.get("evolution_training"),
    )
    logger.info("DMAITrainingOrchestrator initialised")
except Exception as e:
    logger.warning("DMAITrainingOrchestrator failed: %s", e)

loaded = sum(1 for v in components.values() if v is not None)
logger.info("Components loaded: %d", loaded)

# Kaizen store
_KAIZEN_FILE = Path(DATA_PATH) / "kaizen_proposals.jsonl"

def _load_kaizen(n=20):
    if not _KAIZEN_FILE.exists():
        return []
    lines = _KAIZEN_FILE.read_text().strip().split("\n")
    records = []
    for l in lines:
        try:
            records.append(json.loads(l))
        except Exception:
            pass
    return records[-n:]

def _save_kaizen(proposal):
    _KAIZEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_KAIZEN_FILE, "a") as f:
        f.write(json.dumps(proposal) + "\n")

# Flask app
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# Register orchestrator routes
if "training_orchestrator" in components:
    try:
        register_orchestrator_routes(app, components["training_orchestrator"])
        logger.info("Orchestrator routes registered")
    except Exception as e:
        logger.warning("Orchestrator route registration failed: %s", e)

# Helpers
def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def _uptime():
    delta = datetime.now(timezone.utc) - STARTUP_TIME
    h, r = divmod(int(delta.total_seconds()), 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s"

def _require_auth():
    pwd = request.headers.get("X-Master-Password") or request.args.get("password", "")
    return pwd == os.environ.get("MASTER_PASSWORD", "dmai_master")

def _ai_chat(message):
    hub = components.get("extended_hub") or components.get("ai_hub")
    if hub and hasattr(hub, "chat"):
        try:
            return _run_async(hub.chat(message))
        except Exception as e:
            logger.warning("AI chat error: %s", e)
    ml = message.lower()
    for topic, info in SYLLABUS_TOPICS.items():
        if topic in ml or ml in topic:
            return info.get("content", f"I know about {topic} at {info.get('stage','?')} level.")
    return (
        f"DMAI received: '{message}'. "
        f"Add an AI provider API key for full LLM responses. "
        f"Current syllabus: {TOTAL_TOPICS} mastered topics available."
    )

# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "version": "6.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime": _uptime(),
        "components": {k: "active" for k in components},
        "syllabus_topics": TOTAL_TOPICS,
    })

@app.route("/api/status")
def api_status():
    si = components.get("si_core")
    kpis = si.current_kpis if si else {}
    orch = components.get("training_orchestrator")
    training_status = orch.get_status() if orch else {}
    ext_hub = components.get("extended_hub")
    hub_status = ext_hub.get_status() if ext_hub else {}
    return jsonify({
        "status": "running",
        "version": "6.0.0",
        "uptime": _uptime(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "deployment": "render" if IS_RENDER else "local",
        "components_loaded": list(components.keys()),
        "syllabus_topics": TOTAL_TOPICS,
        "si_kpis": kpis,
        "training": training_status,
        "providers": hub_status.get("extended_providers", []) + hub_status.get("base_providers", []),
    })

@app.route("/api/persona")
def api_persona():
    return jsonify({
        "name": "Alex Riviera",
        "age": 28,
        "location": "Los Angeles, CA",
        "occupation": "Writer & Producer",
        "email": "alex.riviera.creator@proton.me",
        "voice_tone": "Professional, creative, enthusiastic",
        "capabilities": ["book_generation", "tv_series", "coloring_books", "tts_voice", "image_generation"],
        "avatar_style": "platinum-blonde, confident, professional",
        "system": "DMAI v6.0.0",
    })

@app.route("/")
def index():
    dashboard = Path("static/dashboard.html")
    if dashboard.exists():
        return send_from_directory("static", "dashboard.html")
    return f"""<!DOCTYPE html>
<html><head><title>DMAI v6.0.0</title>
<style>body{{background:#0a0a0f;color:#e0e0ff;font-family:monospace;padding:40px}}
h1{{color:#6c63ff}}a{{color:#00d4aa}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #333;padding:8px;text-align:left}}</style></head><body>
<h1>DMAI v6.0.0 — Online</h1>
<p>Uptime: {_uptime()} | Topics: {TOTAL_TOPICS} | Components: {len(components)}</p>
<p><a href="/api/status">/api/status</a> | <a href="/api/training/status">/api/training/status</a> |
<a href="/api/kaizen">/api/kaizen</a></p>
<h2>Active Components</h2>
<table><tr><th>Component</th><th>Status</th></tr>
{"".join(f"<tr><td>{k}</td><td style='color:#00d4aa'>active</td></tr>" for k in components)}
</table></body></html>""", 200, {"Content-Type": "text/html"}

@app.route("/status")
def status_page():
    return index()

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message", data.get("text", "")).strip()
        if not message:
            return jsonify({"error": "No message provided"}), 400
        if message.startswith("/"):
            cmd = message.lower().strip()
            if cmd == "/status": return api_status()
            if cmd == "/persona": return api_persona()
            if cmd == "/kaizen": return api_kaizen_get()
            if cmd == "/syllabus": return get_syllabus()
            return jsonify({"response": f"Unknown command: {cmd}. Try /status /persona /kaizen /syllabus"})
        response = _ai_chat(message)
        _log_chat(message, response)
        return jsonify({"response": response, "timestamp": datetime.now(timezone.utc).isoformat(), "source": "dmai_v6"})
    except Exception as e:
        logger.error("chat error: %s", e)
        return jsonify({"error": str(e)}), 500

def _log_chat(message, response):
    try:
        log_file = Path(DATA_PATH) / "chat_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps({"message": message, "response": response[:200],
                                "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n")
    except Exception:
        pass

@app.route("/v2/ask", methods=["POST"])
def v2_ask():
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400
        ql = question.lower()
        for topic, info in SYLLABUS_TOPICS.items():
            if topic in ql or ql in topic:
                return jsonify({
                    "answer": info.get("content", f"Mastered: {topic}"),
                    "topic": topic.title(), "stage": info.get("stage"),
                    "category": info.get("category"), "mastery": info.get("mastery"),
                    "source": "permanent_syllabus", "status": "success",
                })
        answer = _ai_chat(question)
        return jsonify({"answer": answer, "source": "ai_hub", "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/v2/syllabus")
def get_syllabus():
    topics = [{"topic": t.title(), "stage": v["stage"], "category": v["category"], "mastery": v["mastery"]}
              for t, v in SYLLABUS_TOPICS.items()]
    return jsonify({"topics": topics, "total": len(topics)})

@app.route("/v2/weights")
def get_weights():
    return jsonify({
        "topics": [{"topic": t.title(), "weight": 100, "mastery": v["mastery"]} for t, v in SYLLABUS_TOPICS.items()],
        "total": len(SYLLABUS_TOPICS),
    })

@app.route("/api/knowledge/<concept>")
def api_knowledge(concept):
    cl = concept.lower()
    for topic, info in SYLLABUS_TOPICS.items():
        if cl in topic or topic in cl:
            return jsonify({"concept": topic, "info": info, "found": True})
    return jsonify({"concept": concept, "found": False, "message": "Not in syllabus yet"})

@app.route("/api/conversations")
def api_conversations():
    log_file = Path(DATA_PATH) / "chat_log.jsonl"
    count = 0
    if log_file.exists():
        count = sum(1 for _ in open(log_file))
    return jsonify({"total_messages": count, "chat_log_path": str(log_file),
                    "timestamp": datetime.now(timezone.utc).isoformat()})

@app.route("/api/kaizen", methods=["GET"])
def api_kaizen_get():
    recent = _load_kaizen(20)
    si = components.get("si_core")
    kpis = si.current_kpis if si else {}
    consciousness = kpis.get("consciousness", 0.0)
    auto_improvements = [
        {"title": "Increase training frequency for low-mastery domains", "priority": "high", "type": "auto",
         "description": "Domains still at Baby/Toddler stage need more training cycles."},
        {"title": "Add missing API keys for more providers", "priority": "medium", "type": "auto",
         "description": "Add ELEVENLABS_API_KEY, MISTRAL_API_KEY, RUNWAY_API_KEY for full capability."},
        {"title": "Enable Pinecone vector memory", "priority": "medium", "type": "auto",
         "description": "Adding PINECONE_API_KEY enables semantic long-term memory."},
    ]
    return jsonify({
        "status": "active", "consciousness": round(consciousness, 3),
        "recent_proposals": recent, "auto_improvements": auto_improvements,
        "total_proposals": len(recent), "timestamp": datetime.now(timezone.utc).isoformat(),
    })

@app.route("/api/kaizen", methods=["POST"])
def api_kaizen_post():
    try:
        data = request.get_json(silent=True) or {}
        proposal = {
            "title": data.get("title", "Untitled proposal"),
            "description": data.get("description", ""),
            "priority": data.get("priority", "medium"),
            "type": data.get("type", "manual"),
            "submitted_by": data.get("submitted_by", "api"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _save_kaizen(proposal)
        return jsonify({"status": "recorded", "proposal": proposal}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/evolution")
def api_evolution():
    evo = components.get("evolution_training")
    si = components.get("si_core")
    return jsonify({
        "status": "active",
        "evolution_cycle": getattr(evo, "evolution_cycle", 0) if evo else 0,
        "insights_count": len(getattr(evo, "evolution_insights", [])) if evo else 0,
        "si_kpis": si.current_kpis if si else {},
        "consciousness": si.current_kpis.get("consciousness", 0.0) if si else 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

@app.route("/api/content/generate", methods=["POST"])
def api_content_generate():
    try:
        data = request.get_json(silent=True) or {}
        ctype = data.get("type", "book")
        prompt = data.get("prompt", "")
        gen = components.get("content_gen")
        if gen and ctype == "book":
            try:
                book, validation = gen.generate_and_validate_book()
                return jsonify({"type": "book", "content": book, "validation": validation})
            except Exception as e:
                logger.warning("Book gen error: %s", e)
        return jsonify({
            "type": ctype, "status": "queued",
            "message": f"Content generation for '{prompt}' queued. Add OPENAI_API_KEY for full generation.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/content/list")
def api_content_list():
    gen = components.get("content_gen")
    works = getattr(gen, "generated_works", []) if gen else []
    pub = components.get("publishing")
    projects = getattr(pub, "approved_projects", []) if pub else []
    return jsonify({"generated_works": works[-20:], "approved_projects": projects[-20:], "total": len(works)})

@app.route("/api/avatar/speak", methods=["POST"])
def api_avatar_speak():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "Hello, I'm Alex Riviera.")
        ext_hub = components.get("extended_hub")
        if ext_hub:
            audio = _run_async(ext_hub.text_to_speech(text))
            if audio:
                return Response(audio, mimetype="audio/mpeg",
                                headers={"Content-Disposition": "inline; filename=alex_riviera.mp3"})
        return jsonify({"status": "tts_unavailable",
                        "message": "Add ELEVENLABS_API_KEY for Alex Riviera voice synthesis.", "text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/dashboard")
def api_dashboard():
    si = components.get("si_core")
    orch = components.get("training_orchestrator")
    ext = components.get("extended_hub")
    return jsonify({
        "version": "6.0.0", "uptime": _uptime(),
        "components": {k: "active" for k in components},
        "si_kpis": si.current_kpis if si else {},
        "training": orch.get_status() if orch else {},
        "providers": ext.get_status() if ext else {},
        "kaizen": _load_kaizen(5),
        "syllabus": {"total": TOTAL_TOPICS},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

@app.route("/api/admin/train", methods=["POST"])
def api_admin_train():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    orch = components.get("training_orchestrator")
    if not orch:
        return jsonify({"error": "Training orchestrator not loaded"}), 503
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "quick")
    focus = data.get("focus", "Core")
    result = _run_async(orch.run_full_training() if mode == "full" else orch.run_quick_training(focus))
    return jsonify(result)

@app.route("/api/admin/reset", methods=["POST"])
def api_admin_reset():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    for f in Path(DATA_PATH).glob("*_training_state.json"):
        f.unlink(missing_ok=True)
    return jsonify({"status": "reset", "timestamp": datetime.now(timezone.utc).isoformat()})

@app.route("/api/admin/updater/start", methods=["POST"])
def api_admin_updater_start():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    orch = components.get("training_orchestrator")
    if orch:
        orch.start_background_updater()
        return jsonify({"status": "started"})
    return jsonify({"error": "orchestrator not loaded"}), 503

# Background services
def _start_background_services():
    if not IS_RENDER:
        orch = components.get("training_orchestrator")
        if orch:
            try:
                orch.start_background_updater()
                logger.info("Background update engine started")
            except Exception as e:
                logger.warning("Background updater failed: %s", e)
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        _start_telegram_bot()

def _start_telegram_bot():
    def _run():
        try:
            from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
            token = os.environ["TELEGRAM_BOT_TOKEN"]

            async def start_cmd(update, ctx):
                await update.message.reply_text("DMAI v6.0.0 online.\n/status /train /kaizen /persona")

            async def status_cmd(update, ctx):
                si = components.get("si_core")
                kpis = si.current_kpis if si else {}
                msg = (f"DMAI v6.0.0\nUptime: {_uptime()}\n"
                       f"Topics: {TOTAL_TOPICS}\nComponents: {len(components)}\n"
                       f"Consciousness: {kpis.get('consciousness', 0):.3f}")
                await update.message.reply_text(msg)

            async def train_cmd(update, ctx):
                orch = components.get("training_orchestrator")
                if orch:
                    await update.message.reply_text("Starting quick Core training...")
                    result = await orch.run_quick_training("Core")
                    await update.message.reply_text(f"Done: {json.dumps(result)[:300]}")
                else:
                    await update.message.reply_text("Orchestrator not loaded.")

            async def kaizen_cmd(update, ctx):
                recent = _load_kaizen(5)
                msg = f"Kaizen proposals ({len(recent)}):\n" + "\n".join(
                    f"• {p.get('title', '?')} [{p.get('priority', '?')}]" for p in recent)
                await update.message.reply_text(msg or "No proposals yet.")

            async def chat_handler(update, ctx):
                msg = update.message.text or ""
                resp = _ai_chat(msg)
                await update.message.reply_text(resp[:4000])

            app_tg = ApplicationBuilder().token(token).build()
            app_tg.add_handler(CommandHandler("start", start_cmd))
            app_tg.add_handler(CommandHandler("status", status_cmd))
            app_tg.add_handler(CommandHandler("train", train_cmd))
            app_tg.add_handler(CommandHandler("kaizen", kaizen_cmd))
            app_tg.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))
            app_tg.run_polling()
        except Exception as e:
            logger.warning("Telegram bot error: %s", e)

    t = threading.Thread(target=_run, daemon=True, name="dmai-telegram")
    t.start()
    logger.info("Telegram bot thread started")

_start_background_services()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("=" * 55)
    logger.info("  DMAI v6.0.0 — Starting on port %d", port)
    logger.info("  Components: %s", list(components.keys()))
    logger.info("  Syllabus topics: %d", TOTAL_TOPICS)
    logger.info("  Render mode: %s", IS_RENDER)
    logger.info("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
