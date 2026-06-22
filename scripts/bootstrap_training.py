"""
DMAI Training System Bootstrap Script
=======================================
Run this once after cloning/deploying to wire the training system into DMAI.

Usage:
    cd /path/to/dmai-system
    python scripts/bootstrap_training.py [--data-path data/] [--run-now]
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dmai.bootstrap")


def parse_args():
    p = argparse.ArgumentParser(description="DMAI Training System Bootstrap")
    p.add_argument("--data-path", default="data/",   help="Data directory path")
    p.add_argument("--run-now",   action="store_true", help="Run full training after bootstrap")
    p.add_argument("--mode",      choices=["full", "quick", "update"], default="full",
                   help="Training mode if --run-now is set")
    p.add_argument("--focus",     default="Core",
                   help="Category/module for quick mode (Core/Accelerator/Artistic/Wealth/si_*)")
    return p.parse_args()


def check_environment():
    """Verify required environment variables are set."""
    required = []   # DMAI can run without these (uses mock responses)
    optional = {
        "OPENAI_API_KEY":      "OpenAI (existing DMAI integration)",
        "ANTHROPIC_API_KEY":   "Anthropic (existing DMAI integration)",
        "DEEPSEEK_API_KEY":    "DeepSeek (existing DMAI integration)",
        "MISTRAL_API_KEY":     "Mistral (new — text generation)",
        "STABILITY_API_KEY":   "Stability AI (new — image generation)",
        "ELEVENLABS_API_KEY":  "ElevenLabs (new — Alex Riviera TTS)",
        "RUNWAY_API_KEY":      "Runway ML (new — video generation)",
        "REPLICATE_API_KEY":   "Replicate (new — Flux/open-source models)",
        "PINECONE_API_KEY":    "Pinecone (new — vector memory)",
        "TOGETHER_API_KEY":    "Together AI (new — fast open-source inference)",
        "COHERE_API_KEY":      "Cohere (new — reranking/embeddings)",
    }

    print("\n📋  Environment Check")
    print("─" * 50)
    for key, desc in optional.items():
        val = __import__("os").environ.get(key, "")
        status = "✅" if val else "⬜"
        masked = f"{val[:4]}..." if val else "(not set)"
        print(f"  {status} {key:30s} {desc}")
        if not val:
            print(f"         → Add to .env or Render env vars to enable {desc.split(' (')[0]}")
    print()


async def run_bootstrap(args):
    from components.orchestrator.DMAITrainingOrchestrator import DMAITrainingOrchestrator

    print("🚀  Initialising DMAI Training Orchestrator...")
    orch = DMAITrainingOrchestrator(data_path=args.data_path)

    print("📊  Current Status:")
    status = orch.get_status()
    print(json.dumps(status, indent=2))

    if args.run_now:
        print(f"\n🎓  Running training (mode={args.mode}, focus={args.focus})...")
        if args.mode == "full":
            result = await orch.run_full_training()
        elif args.mode == "quick":
            result = await orch.run_quick_training(args.focus)
        elif args.mode == "update":
            result = await orch.run_update_only()
        else:
            result = {}

        print("\n✅  Training Result:")
        print(json.dumps(result, indent=2))

    print("\n📌  Bootstrap snippet for dmai_core_complete.py:")
    print("""
# Add this after your existing component initialisations (around line where ai_hub is created):

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
""")

    print("✅  Bootstrap complete!\n")


if __name__ == "__main__":
    args = parse_args()
    check_environment()
    asyncio.run(run_bootstrap(args))
