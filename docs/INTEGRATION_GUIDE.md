# DMAI Training System — Integration Guide

## Overview

This package adds a complete AI + SI training system to your existing DMAI v6.0.0 repo.
It **extends** existing components — nothing is replaced.

---

## What's Included

| File | Purpose |
|------|---------|
| `components/training/ComprehensiveAITraining.py` | Full AI curriculum — 11 domains × 6 stages (Baby→Expert) |
| `components/si_training/FullSITrainingProgram.py` | 8 new SI modules tied to SICore's 8 KPIs |
| `components/update_engine/PeriodicUpdateEngine.py` | Auto-update engine (model polling, feedback retraining, benchmarks, kaizen) |
| `components/phase11/ExtendedAIIntegrationHub.py` | 8 new providers (Mistral, Stability AI, ElevenLabs, Runway, Replicate, Pinecone, Together AI, Cohere) |
| `components/orchestrator/DMAITrainingOrchestrator.py` | Master wiring — single entry point for all training |
| `scripts/bootstrap_training.py` | One-time setup script |
| `configs/training_config.json` | Centralised config |

---

## Quick Start (3 steps)

### Step 1 — Copy files into your repo

```bash
# From inside your dmai-system repo root:
cp -r /path/to/dmai-training/components/training/ComprehensiveAITraining.py     components/training/
cp -r /path/to/dmai-training/components/si_training/FullSITrainingProgram.py    components/si_training/
cp -r /path/to/dmai-training/components/update_engine/PeriodicUpdateEngine.py   components/update_engine/
cp -r /path/to/dmai-training/components/phase11/ExtendedAIIntegrationHub.py     components/phase11/
mkdir -p components/orchestrator
cp -r /path/to/dmai-training/components/orchestrator/DMAITrainingOrchestrator.py components/orchestrator/__init__.py
cp -r /path/to/dmai-training/scripts/bootstrap_training.py                       scripts/
cp -r /path/to/dmai-training/configs/training_config.json                        configs/
```

### Step 2 — Add to `dmai_core_complete.py`

Find the section where your existing components are initialised (after `ai_hub`, `si_core`, `knowledge_graph`, `evolution_training` are created) and add:

```python
from components.orchestrator.DMAITrainingOrchestrator import (
    DMAITrainingOrchestrator, register_orchestrator_routes
)

training_orchestrator = DMAITrainingOrchestrator(
    data_path        = DATA_PATH,
    si_core          = si_core,
    knowledge_graph  = knowledge_graph,
    ai_hub           = ai_hub,
    evolution_system = evolution_training,
)
register_orchestrator_routes(app, training_orchestrator)
training_orchestrator.start_background_updater(app)
```

### Step 3 — Add environment variables

Add these to your Render environment (or `.env` file). All are optional — system works without them using mock responses:

```
MISTRAL_API_KEY=...        # Mistral text generation
STABILITY_API_KEY=...      # Image generation (Stable Diffusion 3)
ELEVENLABS_API_KEY=...     # Alex Riviera TTS / voice cloning
RUNWAY_API_KEY=...         # Video generation
REPLICATE_API_KEY=...      # Flux image / open-source models
PINECONE_API_KEY=...       # Vector memory
TOGETHER_API_KEY=...       # Fast open-source inference
COHERE_API_KEY=...         # Reranking / embeddings
```

---

## API Endpoints Added

### Training
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/training/status` | Full status of all training components |
| POST | `/api/training/full` | Run complete AI + SI + update cycle |
| POST | `/api/training/quick` | Quick category/module training |
| POST | `/api/training/update` | Run update engine only |
| POST | `/api/training/updater/start` | Start background auto-updater |
| POST | `/api/training/updater/stop` | Stop background auto-updater |

### AI Training (sub-routes)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/training/ai/status` | AI training progress by domain |
| POST | `/api/training/ai/start` | Train all AI domains |
| POST | `/api/training/ai/stage/<stage>` | Train domains at a specific stage |

### SI Training
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/training/si/status` | SI module scores + KPIs |
| POST | `/api/training/si/start` | Run all 8 SI modules |
| POST | `/api/training/si/module/<id>` | Run a single module |
| POST | `/api/training/si/kpi/<kpi>` | Target a specific KPI |

### Extended Hub
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/extended_hub/status` | All provider availability |
| POST | `/api/extended_hub/chat` | Chat via any new provider |
| POST | `/api/extended_hub/image` | Generate image |
| POST | `/api/extended_hub/tts` | Text-to-speech (Alex Riviera voice) |
| POST | `/api/extended_hub/video` | Generate video via Runway |

### Update Engine
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/update_engine/status` | Job schedule + last-run times |
| POST | `/api/update_engine/run` | Trigger all update jobs now |
| POST | `/api/update_engine/job/<name>` | Run a specific job |

---

## AI Training — 11 Domains × 6 Stages

| Domain | Category | Baby | → | Expert |
|--------|----------|------|---|--------|
| Language Understanding | Core | Token recognition | → | Zero-shot multilingual QA |
| Reasoning & Logic | Core | Boolean logic | → | Novel theorem synthesis |
| Memory & Context | Core | 4k buffer | → | Self-organising memory graphs |
| Code Creation & Fixing | Accelerator | Hello-world gen | → | Self-modifying code |
| Agentic Task Execution | Accelerator | Single tool | → | Meta-agent orchestration |
| LLM Fine-Tuning | Accelerator | Prompt design | → | Architecture search / NAS |
| Image Generation | Artistic | Simple DALL-E call | → | Brand-consistent pipeline |
| Video & Avatar | Artistic | API call + Alex Riviera | → | Autonomous content calendar |
| Audio & Speech | Artistic | TTS call | → | Custom voice model training |
| Business & Revenue | Wealth | Revenue stream ID | → | Self-managing revenue agents |
| Knowledge Management | Core | Chunking basics | → | Dynamic ontology construction |

---

## SI Training — 8 New Modules (extends existing 8 consciousness modules)

| Module ID | Name | KPIs Targeted |
|-----------|------|---------------|
| `si_tool_mastery` | Tool Mastery | agentic_capability_score, skill_acquisition_rate |
| `si_system_integration` | System Integration | agentic_capability_score, transfer_learning_rate |
| `si_autonomous_decision` | Autonomous Decision-Making | agentic_capability_score, recursive_self_improvement_rate |
| `si_metacognition` | Metacognition | metacognition_accuracy, recursive_self_improvement_rate |
| `si_multimodal_fusion` | Multi-Modal Fusion | multi_modal_integration_score, transfer_learning_rate |
| `si_recursive_improvement` | Recursive Self-Improvement | recursive_self_improvement_rate, sample_efficiency_trend |
| `si_social_intelligence` | Social Intelligence | metacognition_accuracy, skill_acquisition_rate |
| `si_knowledge_synthesis` | Knowledge Synthesis | transfer_learning_rate, zero_shot_success_count |

---

## Update Engine — Auto-Update Schedule

| Job | Default Interval | What it does |
|-----|-----------------|--------------|
| Model Registry | Every 6h | Polls LiteLLM + OpenRouter for new models |
| Feedback Retraining | Every 12h | Reads user feedback, updates KPIs |
| Knowledge Freshening | Daily | Ingests HuggingFace + PapersWithCode papers |
| Benchmark | Every 48h | Runs reference Q&A, flags regressions |
| Kaizen Integration | Every 6h | Submits improvement proposals to /api/kaizen |

---

## Extended Hub — New Providers

| Provider | Use Case | Env Var |
|----------|----------|---------|
| Mistral | Fast/cheap text generation | `MISTRAL_API_KEY` |
| Stability AI | SDXL / SD3 image generation | `STABILITY_API_KEY` |
| ElevenLabs | Alex Riviera TTS + voice clone | `ELEVENLABS_API_KEY` |
| Runway ML | Text/image-to-video | `RUNWAY_API_KEY` |
| Replicate | Flux, open-source models | `REPLICATE_API_KEY` |
| Pinecone | Vector memory / semantic search | `PINECONE_API_KEY` |
| Together AI | Fast Llama/Mixtral inference | `TOGETHER_API_KEY` |
| Cohere | Reranking + embeddings | `COHERE_API_KEY` |

---

## Dependencies

Only one new dependency beyond existing DMAI requirements:

```
httpx>=0.27.0
```

Add to `requirements.txt`.
