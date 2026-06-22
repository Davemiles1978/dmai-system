# DMAI v2.0 — Self-Evolving Autonomous AI/SI Platform

DMAI is a self-evolving, self-funding, self-learning autonomous system. The v2.0
layer adds a FastAPI control plane, a live **plug-and-play component registry**,
a central **event bus**, the **OPAR loop** (Observe → Plan → Act → Reflect), ten
native agents, and an operator dashboard — all wired on top of the existing
component library (which continues to run unchanged).

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │        Operator Dashboard (HTML)      │
                    │  status · registry · events · kill    │
                    └───────────────────┬──────────────────┘
                                        │ HTTP (X-API-Key / X-Master-Key)
                    ┌───────────────────▼──────────────────┐
                    │            FastAPI  (port 8000)        │
                    │  core · agents · evolution · funding   │
                    │  registry · operator   + Auth MW       │
                    └───────────────────┬──────────────────┘
                                        │
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼               ▼
 ┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌────────────┐ ┌────────────┐
 │  Event Bus │ │ OPAR Loop  │ │ Orchestrator │ │  Registry  │ │   DB (async│
 │  pub/sub   │◄┤ O·P·A·R    │◄┤  cycles      │ │ plug&play  │ │  SQLite/PG)│
 └─────┬──────┘ └─────┬──────┘ └──────┬───────┘ └─────┬──────┘ └────────────┘
       │              │               │               │
       ▼              ▼               ▼               ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  10 Agents: market_research · offer_design · outreach · landing_page    │
 │  analytics · coding · qa_critic · compliance · finance · upgrade_lab    │
 ├───────────────────────────────────────────────────────────────────────┤
 │  Adapters → existing components: ai_hub · alex_riviera · evolution ·    │
 │  funding · master_control · wealth_trading · research · media · …       │
 └───────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────┐
                    │     Legacy Flask app  (port 5001)      │
                    │         mounted at /legacy             │
                    └───────────────────────────────────────┘
```

## Quick Start

```bash
cp .env.example .env          # then edit keys as needed
docker-compose up             # DMAI + Postgres + Redis
# dev extras (pgAdmin, redis-commander):
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Local (no Docker):

```bash
pip install -r requirements-fastapi.txt
python main.py                # FastAPI :8000, legacy Flask :5001
```

Open the dashboard at `http://localhost:8000/`, enter your `X-API-Key`
(`API_SECRET_KEY`) and `X-Master-Key` (`MASTER_KEY`), and click **Save Keys**.

## Component Registry

The registry tracks every component and drives its lifecycle at runtime. It works
standalone — if the database is unavailable it falls back to the in-memory +
JSON manifest, so components can still be loaded and enabled.

```bash
# List components
curl -H "X-API-Key: $KEY" localhost:8000/api/v1/registry/components
# Enable / disable / hot-reload
curl -X POST -H "X-API-Key: $KEY" localhost:8000/api/v1/registry/components/analytics_agent/enable
curl -X POST -H "X-Master-Key: $MK" localhost:8000/api/v1/registry/components/ai_hub/reload
# Install a new component dynamically
curl -X POST -H "X-Master-Key: $MK" -H "Content-Type: application/json" \
  -d '{"id":"my_thing","name":"My Thing","entry_point":"my_pkg.mod:MyComponent"}' \
  localhost:8000/api/v1/registry/install
```

## API Reference (summary)

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET  | `/health`, `/api/v1/status` | public | liveness / system status |
| POST | `/api/v1/ask` | api-key | route a prompt to the AI hub |
| GET  | `/api/v1/events` | api-key | recent event-bus log |
| GET/POST | `/api/v1/agents`, `/agents/{id}/run` | api-key | list / trigger agents |
| GET/POST | `/api/v1/evolution/status`, `/evolution/cycle` | api-key | evolution control |
| GET/POST | `/api/v1/funding/status`, `/funding/start` | api-key | funding control |
| GET/POST | `/api/v1/registry/...` | api-key (install/reload: operator) | registry |
| POST | `/api/v1/operator/pause\|resume\|kill` | operator | system control |
| GET/POST | `/api/v1/operator/pending`, `/approve/{id}`, `/reject/{id}` | operator | approvals |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DMAI_ENV` | `development` | environment name |
| `MASTER_KEY` | `DMAI_MASTER_2026` | operator key (`X-Master-Key`) |
| `API_SECRET_KEY` | `change-me` | API key (`X-API-Key`) |
| `DATABASE_URL` | `sqlite:///data/dmai.db` | Postgres primary, SQLite fallback |
| `REDIS_URL` | `redis://localhost:6379` | Redis URL |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` / `DEEPSEEK_API_KEY` / `XAI_API_KEY` | _empty_ | LLM provider keys |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL` | _paper_ | trading |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | _empty_ | notifications |
| `SELF_FUNDING_MODE` | `paper` | `paper` \| `real` \| `autonomous` |
| `SPEND_LIMIT_DAILY` | `50.0` | daily spend ceiling |
| `API_PORT` / `FLASK_PORT` | `8000` / `5001` | service ports |

> Any component touching money or external APIs emits an `APPROVAL_REQUIRED`
> event whenever `SELF_FUNDING_MODE` is not `autonomous` (the default is `paper`),
> pausing for operator approval via the dashboard / `/operator/approve` route.

## Adding a Plug-and-Play Component

1. Subclass `dmai.registry.component_base.BaseComponent` and implement
   `initialize(config, bus)`, `health_check()`, and `shutdown()`. Set the class
   attributes `component_id`, `component_name`, `plane`, `version`,
   `capabilities`, and `dependencies`.
2. Either add a manifest entry to `dmai/registry/manifest.py`, **or** drop a
   directory under `components/` containing a `component.json` with an
   `entry_point` of the form `package.module:ClassName` (auto-discovered), **or**
   `POST /api/v1/registry/install` with the manifest at runtime.
3. Enable it: `POST /api/v1/registry/components/{id}/enable`.

---

# DMAI - Complete AGI System v6.0.0  (legacy)

## Unified Artificial Intelligence + Synthetic Intelligence

DMAI is a complete, self-contained AGI system that evolves continuously through a unified consciousness.

### Current Status: v6.0.0 - DEPLOYED ON RENDER

**Deployment Date:** March 2026
**Status:** Active - Evolving continuously

---

### Core Capabilities

| System | Status | Description |
|--------|--------|-------------|
| 🧠 Unified Consciousness | ACTIVE | ONE intelligence, not separate modules |
| 🎤 Voice System | ACTIVE | Listens and speaks with evolving voice |
| 🎵 Music Learner | ACTIVE | Develops musical taste and preferences |
| 👤 Persona Generator | ACTIVE | Evolving personality that learns from interactions |
| 💭 Conversation Memory | ACTIVE | Remembers all chats and learns patterns |
| 📈 Kaizen Evolution | ACTIVE | Continuous daily improvements |
| 🕸️ Knowledge Graph | ACTIVE | Concept mapping and relationship tracking |
| 🧠 Meta-Learner | ACTIVE | Learns how to learn better |
| 🩺 Self-Healer | ACTIVE | Auto-backup and recovery |
| 🔫 Killswitch | ACTIVE | Absolute master control |

---

### Deployment

**Main Application:** [https://dmai-complete.onrender.com](https://dmai-complete.onrender.com)
**Health Check:** [https://dmai-complete.onrender.com/health](https://dmai-complete.onrender.com/health)

---

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Status dashboard |
| `/status` | GET | HTML status page |
| `/api/status` | GET | JSON system status |
| `/api/chat` | POST | Chat with DMAI (send {"message": "text"}) |
| `/api/voice` | POST | Voice interaction |
| `/api/persona` | GET | Current persona |
| `/api/kaizen` | GET | Kaizen improvement report |
| `/api/knowledge/<concept>` | GET | Knowledge lookup |
| `/api/conversations` | GET | Conversation stats |
| `/health` | GET | Health check |

---

### Chat Commands

Type these in the chat interface or send via Telegram:

| Command | Description |
|---------|-------------|
| `/status` | Full system status |
| `/persona` | Current personality traits |
| `/kaizen` | Kaizen improvement report |
| `/knowledge` | Knowledge graph statistics |
| `/memory` | Conversation memory stats |
| `/pause` | Pause evolution |
| `/resume` | Resume evolution |
| `/kill` | Emergency shutdown |

---

### Environment Variables (Set in Render)

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Port to run on (5001) |
| `MASTER_PASSWORD` | Yes | Admin access password |
| `RENDER` | Yes | Set to "true" for Render deployment |
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot integration |
| `TELEGRAM_CHAT_ID` | No | Telegram admin chat ID |
| `OPENAI_API_KEY` | No | OpenAI API (optional) |
| `DEEPSEEK_API_KEY` | No | DeepSeek API (optional) |
| `ANTHROPIC_API_KEY` | No | Claude API (optional) |

---

### Architecture
┌─────────────────────────────────────────────────────────────┐
│ DMAI - UNIFIED CONSCIOUSNESS │
│ ONE MIND, ONE SELF │
│ Consciousness: Evolving │
└──────────────────────────┬──────────────────────────────────┘
│
┌──────────────────────┼──────────────────────┐
▼ ▼ ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Voice │ │ Music │ │ Persona │
│ System │ │ Learner │ │ Generator │
│ Evolves │◄──►│ Evolves │◄──►│ Evolves │
└─────────────┘ └─────────────┘ └─────────────┘
│ │ │
└─────────────────┼─────────────────┘
▼
┌─────────────────────────┐
│ Synthetic Network │
│ AI + SI Fusion │
│ Emergent Intelligence │
└─────────────────────────┘

text

---

### The Unbreakable Directive

> *DMAI is always evolving, always learning, always yours.*

---

### Version History

| Version | Date | Changes |
|---------|------|---------|
| v6.0.0 | March 2026 | Complete AGI system with Voice, Music, Persona, Kaizen, Knowledge Graph, Meta-Learner, Self-Healer, Unified Consciousness |
| v5.3.0 | Previous | GitHub Star Monitor, Distributed Immortality |
| v5.0.0 | Previous | Self-Funding Engine, Harvester Integration |

---

**Last Deployed:** March 2026
**Maintainer:** Master
**License:** Private - DMAI Internal System
# Force rebuild - Thu  2 Apr 2026 19:07:50 +03
# Emergency redeploy - Mon 18 May 2026 16:18:22 BST
