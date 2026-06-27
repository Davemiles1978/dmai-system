# DMAI v6.0.0 — Complete Install & Setup Guide

> **Goal:** Get DMAI running locally, trained, and live on Render — step by step.

---

## Table of Contents

1. [What DMAI Is](#1-what-dmai-is)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Clone & Install](#3-step-1--clone--install)
4. [Step 2 — Configure Environment Variables](#4-step-2--configure-environment-variables)
5. [Step 3 — Set Up the Database](#5-step-3--set-up-the-database)
6. [Step 4 — Run System Health Check](#6-step-4--run-system-health-check)
7. [Step 5 — Run Full Training Program](#7-step-5--run-full-training-program)
8. [Step 6 — Start the Server Locally](#8-step-6--start-the-server-locally)
9. [Step 7 — Deploy to Render (Go Live)](#9-step-7--deploy-to-render-go-live)
10. [Step 8 — Connect Telegram Bot](#10-step-8--connect-telegram-bot)
11. [Step 9 — Add API Keys for Full Capability](#11-step-9--add-api-keys-for-full-capability)
12. [Step 10 — Verify Live System](#12-step-10--verify-live-system)
13. [API Reference](#13-api-reference)
14. [Training System Reference](#14-training-system-reference)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. What DMAI Is

DMAI (Davemiles AI) is a self-evolving AGI system built on Flask. It:

- **Learns** through a 6-stage curriculum (Baby → Toddler → Child → Teen → Adult → Expert)
- **Evolves** via kaizen loops, SI consciousness tracking, and 8 intelligence KPIs
- **Creates** books, videos, images, and audio via Alex Riviera avatar
- **Integrates** 20+ AI providers (OpenAI, Anthropic, Gemini, Mistral, ElevenLabs, Runway, etc.)
- **Self-updates** by polling new models, retraining from feedback, and benchmarking itself
- **Deploys** on Render with a single command

---

## 2. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | `python --version` |
| pip | Latest | `pip install --upgrade pip` |
| Git | Any | For repo clone & push |
| PostgreSQL | 14+ (production) | Render provides this free |
| Render account | Free tier works | https://render.com |

**Minimum API keys to go live:**
- `MASTER_PASSWORD` — you choose this
- `OPENAI_API_KEY` — for chat/intelligence (or use any alternative)

**Everything else is optional** — DMAI degrades gracefully without them.

---

## 3. Step 1 — Clone & Install

```bash
# Clone your repo
git clone https://github.com/Davemiles1978/dmai-system.git
cd dmai-system

# Create a virtual environment
python -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt

# Install Playwright browsers (for identity/account creation features)
playwright install chromium
```

---

## 4. Step 2 — Configure Environment Variables

```bash
# Copy the template
cp .env.template .env

# Open and fill in your values
nano .env          # or: code .env / vim .env
```

### Minimum .env for local run:
```
PORT=5000
MASTER_PASSWORD=choose_a_strong_password_here
DATA_PATH=data/
RENDER=false
LOG_LEVEL=INFO
OPENAI_API_KEY=sk-...your-openai-key...
```

### For full capability, also add:
```
ANTHROPIC_API_KEY=sk-ant-...
ELEVENLABS_API_KEY=...          # Alex Riviera voice
MISTRAL_API_KEY=...             # Fast text generation
STABILITY_API_KEY=...           # Image generation
RUNWAY_API_KEY=...              # Video generation
REPLICATE_API_KEY=...           # Flux + open models
PINECONE_API_KEY=...            # Vector memory
TOGETHER_API_KEY=...            # Fast inference
COHERE_API_KEY=...              # Reranking
TELEGRAM_BOT_TOKEN=...          # Telegram control
TELEGRAM_CHAT_ID=...            # Your Telegram chat ID
DATABASE_URL=postgresql://...   # PostgreSQL connection string
```

---

## 5. Step 3 — Set Up the Database

### Option A — SQLite (local dev, no setup needed)
DMAI uses SQLite automatically when `DATABASE_URL` is not set. Just run.

### Option B — PostgreSQL (production, recommended)

**On Render:** Create a free PostgreSQL database:
1. Render Dashboard → New → PostgreSQL
2. Copy the "Internal Database URL"
3. Add it to your `.env` as `DATABASE_URL`

**Local PostgreSQL:**
```bash
createdb dmai_db
# Then set:
DATABASE_URL=postgresql://localhost/dmai_db
```

---

## 6. Step 4 — Run System Health Check

```bash
python scripts/check_system.py
```

This verifies:
- All required components import correctly
- All critical env vars are set
- Data directories exist
- Alex Riviera avatar files are present

**Expected output:**
```
✅  MASTER_PASSWORD — set
✅  components.si_core.SICore
✅  components.training.ComprehensiveAITraining.ComprehensiveAITraining
✅  components.orchestrator.DMAITrainingOrchestrator.DMAITrainingOrchestrator
...
🚀  All checks passed — DMAI is ready to run!
```

Fix any ❌ errors before proceeding.

---

## 7. Step 5 — Run Full Training Program

This is the most important step — it teaches DMAI everything it needs to know.

```bash
# Full training (AI + SI + Update engine) — recommended first run
python scripts/run_training.py --mode full

# Or run each program separately:
python scripts/run_training.py --mode ai        # AI curriculum (11 domains × 6 stages)
python scripts/run_training.py --mode si        # SI modules (8 × KPI-linked modules)
python scripts/run_training.py --mode update    # Update engine (model polling, benchmarks)

# Quick training (one category — good for testing):
python scripts/run_training.py --mode quick --focus Core
python scripts/run_training.py --mode quick --focus Accelerator
python scripts/run_training.py --mode quick --focus Artistic
python scripts/run_training.py --mode quick --focus Wealth

# Save results to file:
python scripts/run_training.py --mode full --output data/training_results.json
```

### Training Modes Explained

| Mode | What runs | Time |
|------|-----------|------|
| `full` | AI curriculum + SI modules + update engine | ~5-10 min |
| `ai` | 11 domains × all stages | ~3-5 min |
| `si` | 8 SI modules (tool mastery, metacognition, etc.) | ~2-3 min |
| `quick` | One category (Core/Accelerator/Artistic/Wealth) | ~1-2 min |
| `update` | Model polling + knowledge freshening + benchmark | ~1 min |
| `stage` | All domains currently at a specific stage | ~1-2 min |

### Training Progress

After training, check progress:
```bash
cat data/ai_training_state.json    # AI domain mastery by stage
cat data/si_training_state.json    # SI module scores + KPI values
```

---

## 8. Step 6 — Start the Server Locally

```bash
# Development server (auto-reload):
python start_dmai.py

# Or directly:
python dmai_core_complete.py

# Or with gunicorn (same as production):
gunicorn dmai_core_complete:app --bind 0.0.0.0:5000 --timeout 120 --workers 1 --threads 2
```

**Test it:**
```bash
curl http://localhost:5000/health
curl http://localhost:5000/api/status
curl -X POST http://localhost:5000/api/chat -H "Content-Type: application/json" -d '{"message": "Hello DMAI"}'
```

**Open the dashboard:**
→ http://localhost:5000

---

## 9. Step 7 — Deploy to Render (Go Live)

### Option A — Connect repo directly (recommended)

1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repo: `Davemiles1978/dmai-system`
3. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn dmai_core_complete:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 2 --max-requests 100 --max-requests-jitter 50`
   - **Python Version:** 3.11
4. Go to Environment → Add all variables from your `.env` file
   - At minimum: `MASTER_PASSWORD`, `RENDER=true`, `PORT=10000`
5. Click Deploy

### Option B — Use render.yaml (Infrastructure as Code)

```bash
# render.yaml is already in the repo — Render will auto-detect it
# Just connect the repo and Render reads the config automatically
git add render.yaml
git commit -m "add render.yaml"
git push
```

Then on Render: New → Blueprint → connect repo.

### After Deploy

1. Your live URL: `https://dmai-web.onrender.com`
2. Check health: `https://dmai-web.onrender.com/health`
3. Add PostgreSQL: Render Dashboard → New → PostgreSQL → copy Internal URL → add as `DATABASE_URL` env var

### Run Training on Live System

```bash
# Trigger training via API (from anywhere)
curl -X POST https://dmai-web.onrender.com/api/training/full \
  -H "Content-Type: application/json"

# Quick training
curl -X POST https://dmai-web.onrender.com/api/training/quick \
  -H "Content-Type: application/json" \
  -d '{"focus": "Core"}'

# Check training status
curl https://dmai-web.onrender.com/api/training/status
```

---

## 10. Step 8 — Connect Telegram Bot

1. Open Telegram → search `@BotFather`
2. Send `/newbot` → follow prompts → copy the bot token
3. Get your chat ID: message `@userinfobot`
4. Add to `.env` (or Render env vars):
   ```
   TELEGRAM_BOT_TOKEN=1234567890:ABCdef...
   TELEGRAM_CHAT_ID=987654321
   ```
5. Restart DMAI

**Bot commands:**
```
/start        — Welcome message
/status       — Full system status + KPIs
/train        — Trigger quick Core training
/kaizen       — Kaizen improvement report
/persona      — Alex Riviera info
```

---

## 11. Step 9 — Add API Keys for Full Capability

Add these in order of impact:

### Tier 1 — Core Intelligence (add first)
| Key | Where to get | What it enables |
|-----|-------------|-----------------|
| `OPENAI_API_KEY` | platform.openai.com | GPT-4o chat, DALL-E images, Whisper STT |
| `ANTHROPIC_API_KEY` | console.anthropic.com | Claude 3.5 Sonnet (best reasoning) |

### Tier 2 — Alex Riviera Voice & Video (add second)
| Key | Where to get | What it enables |
|-----|-------------|-----------------|
| `ELEVENLABS_API_KEY` | elevenlabs.io | Alex Riviera's actual voice (TTS + cloning) |
| `RUNWAY_API_KEY` | runwayml.com | Text-to-video generation |
| `STABILITY_API_KEY` | platform.stability.ai | SD3 image generation |

### Tier 3 — Extended Capabilities (add when ready)
| Key | Where to get | What it enables |
|-----|-------------|-----------------|
| `MISTRAL_API_KEY` | console.mistral.ai | Fast, cheap European LLM |
| `REPLICATE_API_KEY` | replicate.com | Flux, Llama, open-source models |
| `PINECONE_API_KEY` | app.pinecone.io | Vector memory for semantic search |
| `TOGETHER_API_KEY` | api.together.xyz | Fast open-source inference (Llama 3.1) |
| `COHERE_API_KEY` | dashboard.cohere.com | Reranking + embeddings |
| `GEMINI_API_KEY` | aistudio.google.com | Gemini 1.5 Pro (2M context) |
| `XAI_API_KEY` | console.x.ai | Grok (real-time web knowledge) |

### Tier 4 — Business & Publishing
| Key | Where to get | What it enables |
|-----|-------------|-----------------|
| `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` | alpaca.markets | Paper/live trading |
| `KDP_EMAIL` + `KDP_PASSWORD` | kdp.amazon.com | Auto-publish books to Amazon |

---

## 12. Step 10 — Verify Live System

```bash
# Run health checker against live URL
python scripts/check_system.py --url https://dmai-web.onrender.com

# Manual checks:
curl https://dmai-web.onrender.com/health
curl https://dmai-web.onrender.com/api/status
curl https://dmai-web.onrender.com/api/training/status
curl https://dmai-web.onrender.com/api/extended_hub/status
curl https://dmai-web.onrender.com/api/kaizen

# Test chat:
curl -X POST https://dmai-web.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are your capabilities?"}'

# Check SI KPIs:
curl https://dmai-web.onrender.com/api/evolution
```

---

## 13. API Reference

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/health` | Health check |
| GET | `/api/status` | Full system JSON status |
| GET | `/api/persona` | Alex Riviera persona |
| GET | `/api/dashboard` | Admin dashboard data |
| GET | `/api/evolution` | SI KPIs + evolution state |

### Chat & Intelligence

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| POST | `/api/chat` | `{"message": "..."}` | Chat with DMAI |
| POST | `/v2/ask` | `{"question": "..."}` | Syllabus-aware Q&A |
| GET | `/v2/syllabus` | — | All 500+ mastered topics |
| GET | `/api/knowledge/<concept>` | — | Knowledge lookup |

### Training

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| GET | `/api/training/status` | — | All training progress |
| POST | `/api/training/full` | — | Full AI+SI+update run |
| POST | `/api/training/quick` | `{"focus": "Core"}` | Quick category training |
| POST | `/api/training/update` | — | Update engine only |
| GET | `/api/training/ai/status` | — | AI training by domain |
| POST | `/api/training/ai/start` | `{"domains": [...]}` | AI training |
| GET | `/api/training/si/status` | — | SI module scores + KPIs |
| POST | `/api/training/si/start` | — | SI training |
| POST | `/api/training/si/module/<id>` | — | Single SI module |
| POST | `/api/training/si/kpi/<kpi>` | — | KPI-targeted training |

### Kaizen & Evolution

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| GET | `/api/kaizen` | — | Kaizen report + proposals |
| POST | `/api/kaizen` | `{"title":"...", "description":"...", "priority":"high"}` | Submit proposal |

### Content & Media

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| POST | `/api/content/generate` | `{"type":"book", "prompt":"..."}` | Generate content |
| GET | `/api/content/list` | — | List generated content |
| POST | `/api/avatar/speak` | `{"text":"..."}` | Alex Riviera TTS (returns MP3) |

### Extended Hub

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| GET | `/api/extended_hub/status` | — | All 20+ provider status |
| POST | `/api/extended_hub/chat` | `{"prompt":"...", "provider":"mistral"}` | Chat via specific provider |
| POST | `/api/extended_hub/image` | `{"prompt":"...", "provider":"stability"}` | Generate image (returns PNG) |
| POST | `/api/extended_hub/tts` | `{"text":"...", "voice_id":"..."}` | TTS audio (returns MP3) |
| POST | `/api/extended_hub/video` | `{"prompt":"...", "provider":"runway"}` | Generate video |

### Admin (requires X-Master-Password header)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/train` | Manual training trigger |
| POST | `/api/admin/reset` | Reset training state |
| POST | `/api/admin/updater/start` | Start background updater |

---

## 14. Training System Reference

### AI Training — 11 Domains

| Domain | Category | Stages |
|--------|----------|--------|
| Language Understanding | Core | Baby → Expert |
| Reasoning & Logic | Core | Baby → Expert |
| Memory & Context Management | Core | Baby → Expert |
| Knowledge Management & RAG | Core | Baby → Expert |
| Code Creation & Fixing | Accelerator | Baby → Expert |
| Agentic Task Execution | Accelerator | Baby → Expert |
| LLM Fine-Tuning & Adaptation | Accelerator | Baby → Expert |
| Image Generation & Editing | Artistic | Baby → Expert |
| Video & Avatar Creation | Artistic | Baby → Expert |
| Audio & Speech Synthesis | Artistic | Baby → Expert |
| Business & Revenue Generation | Wealth | Baby → Expert |

### SI Training — 8 New Modules

| Module | KPIs Targeted |
|--------|---------------|
| Tool Mastery | agentic_capability_score, skill_acquisition_rate |
| System Integration | agentic_capability_score, transfer_learning_rate |
| Autonomous Decision-Making | agentic_capability_score, recursive_self_improvement_rate, zero_shot_success_count |
| Metacognition | metacognition_accuracy, recursive_self_improvement_rate |
| Multi-Modal Fusion | multi_modal_integration_score, transfer_learning_rate |
| Recursive Self-Improvement | recursive_self_improvement_rate, sample_efficiency_trend |
| Social Intelligence | metacognition_accuracy, skill_acquisition_rate |
| Knowledge Synthesis | transfer_learning_rate, zero_shot_success_count, sample_efficiency_trend |

### SICore 8 KPIs (tracked in real-time)

```
skill_acquisition_rate        — How fast new skills are learned
transfer_learning_rate        — Applying knowledge across domains
zero_shot_success_count       — Tasks solved without prior examples
agentic_capability_score      — Autonomous task execution quality
recursive_self_improvement_rate — Self-improvement loop effectiveness
sample_efficiency_trend       — Learning more from fewer examples
metacognition_accuracy        — Accuracy of self-knowledge
multi_modal_integration_score — Cross-modality reasoning quality
consciousness                 — Weighted composite of all 8 KPIs
```

### Periodic Update Engine Jobs

| Job | Interval | What it does |
|-----|---------|--------------|
| Model Registry | Every 6h | Polls LiteLLM + OpenRouter for new AI models |
| Feedback Retraining | Every 12h | Processes user feedback → updates KPIs |
| Knowledge Freshening | Daily | Ingests HuggingFace + PapersWithCode papers |
| Benchmark | Every 48h | Tests DMAI against reference Q&A, flags regressions |
| Kaizen Integration | Every 6h | Submits improvement proposals to /api/kaizen |

---

## 15. Troubleshooting

### "Module not found" on startup
```bash
# Make sure you're in the repo root with venv active
cd dmai-system
source venv/bin/activate
python scripts/check_system.py
```

### Render deploy crashes immediately
- Check Render logs for the specific error
- Most common: missing `requirements.txt` dependency or import error
- Quick fix: add `DISABLE_NEO4J=true` and `DISABLE_AUTO_THREADS=true` env vars

### Chat returns "no provider available"
- Add at least one AI API key: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `MISTRAL_API_KEY`
- DMAI works offline with syllabus-only responses but needs a key for full LLM chat

### Training runs but shows 0% progress
- This is normal on first run — no real AI inference happens without API keys
- Training still works (uses internal mock responses for curriculum scoring)
- Add `OPENAI_API_KEY` for real scored training sessions

### Avatar/TTS returns "unavailable"
- Add `ELEVENLABS_API_KEY` for Alex Riviera voice
- The key is obtained from elevenlabs.io → Profile → API Key

### Render free tier sleeps after 15 min inactivity
- Add a free uptime monitor: uptimerobot.com → Monitor → HTTP → your Render URL
- Or upgrade to Render Starter plan ($7/month) for always-on

### PostgreSQL connection fails
- Check `DATABASE_URL` format: `postgresql://user:password@host:5432/dbname`
- On Render: use the **Internal** URL (not External) for lower latency

### Background updater not running
```bash
curl -X POST https://your-url.onrender.com/api/admin/updater/start \
  -H "X-Master-Password: your_password"
```

---

## Quick Reference Card

```bash
# LOCAL DEVELOPMENT
git clone https://github.com/Davemiles1978/dmai-system.git && cd dmai-system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.template .env && nano .env     # Add your API keys
python scripts/check_system.py         # Verify setup
python scripts/run_training.py         # Train DMAI
python dmai_core_complete.py           # Start server → http://localhost:5000

# PRODUCTION (after Render deploy)
curl https://dmai-web.onrender.com/health
curl -X POST https://dmai-web.onrender.com/api/training/full
curl https://dmai-web.onrender.com/api/training/status
curl https://dmai-web.onrender.com/api/evolution
```

---

*DMAI v6.0.0 — Built by David Miles | Repo: github.com/Davemiles1978/dmai-system*
