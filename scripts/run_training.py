#!/usr/bin/env python3
"""
DMAI Full Training Runner
==========================
Standalone script to kick off the complete AI + SI training program.
Run this directly from the repo root — no Flask server needed.

Usage:
    python scripts/run_training.py                    # Full training (all domains)
    python scripts/run_training.py --mode quick       # Quick Core category only
    python scripts/run_training.py --mode si          # SI modules only
    python scripts/run_training.py --mode update      # Update engine only
    python scripts/run_training.py --mode ai          # AI domains only
    python scripts/run_training.py --focus Artistic   # Specific category
    python scripts/run_training.py --stage Teen       # All domains at Teen stage
    python scripts/run_training.py --live             # Runs + starts background updater
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Add repo root to path ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Load .env if present ────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    print("✅  .env loaded")
except ImportError:
    print("⬜  python-dotenv not installed — using existing environment")

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dmai.runner")


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║          DMAI v6.0.0 — Training Runner               ║
║   Autonomous General Intelligence Training System    ║
╚══════════════════════════════════════════════════════╝
""")


def parse_args():
    p = argparse.ArgumentParser(description="DMAI Training Runner")
    p.add_argument("--mode",   default="full",
                   choices=["full", "quick", "si", "ai", "update", "stage"],
                   help="Training mode")
    p.add_argument("--focus",  default="Core",
                   help="Category for quick mode (Core/Accelerator/Artistic/Wealth) or SI module ID")
    p.add_argument("--stage",  default="Baby",
                   help="Stage for stage mode (Baby/Toddler/Child/Teen/Adult/Expert)")
    p.add_argument("--data-path", default="data/",
                   help="Data directory path")
    p.add_argument("--live",   action="store_true",
                   help="Start background updater after training")
    p.add_argument("--output", default=None,
                   help="Save results to JSON file")
    return p.parse_args()


async def run(args):
    data_path = args.data_path
    Path(data_path).mkdir(parents=True, exist_ok=True)

    # ── Init components ───────────────────────────────────────────────────
    print("🔧  Initialising components...")
    si_core = None
    ai_hub  = None
    evo     = None

    try:
        from components.si_core import SICore
        si_core = SICore(data_path=Path(data_path))
        print("  ✅  SICore")
    except Exception as e:
        print(f"  ⬜  SICore: {e}")

    try:
        from components.phase11.AIIntegrationHub import AIIntegrationHub
        ai_hub = AIIntegrationHub(data_path=data_path)
        print(f"  ✅  AIIntegrationHub")
    except Exception as e:
        print(f"  ⬜  AIIntegrationHub: {e}")

    try:
        from components.evolution_training.EvolutionTrainingSystem import EvolutionTrainingSystem
        evo = EvolutionTrainingSystem(si_core=si_core, knowledge_graph=None, training_systems={})
        print(f"  ✅  EvolutionTrainingSystem")
    except Exception as e:
        print(f"  ⬜  EvolutionTrainingSystem: {e}")

    # ── Init orchestrator ────────────────────────────────────────────────
    from components.orchestrator.DMAITrainingOrchestrator import DMAITrainingOrchestrator
    orch = DMAITrainingOrchestrator(
        data_path        = data_path,
        si_core          = si_core,
        knowledge_graph  = None,
        ai_hub           = ai_hub,
        evolution_system = evo,
    )
    print("  ✅  DMAITrainingOrchestrator\n")

    # ── Print current status ────────────────────────────────────────────
    status = orch.get_status()
    print("📊  Current Status:")
    print(f"    AI Training domains: {status.get('components', {}).get('ai_training', {}).get('domains', 'N/A')}")
    print(f"    SI Training modules: {status.get('components', {}).get('si_training', {}).get('modules', 'N/A')}")
    print(f"    Overall AI progress: {status.get('components', {}).get('ai_training', {}).get('progress', {}).get('avg_mastery', 0):.1%}")
    print(f"    SI overall score:    {status.get('components', {}).get('si_training', {}).get('overall_score', 0):.3f}\n")

    # ── Run ────────────────────────────────────────────────────────────
    start = datetime.now()
    print(f"🎓  Starting training — mode={args.mode}, focus={args.focus}")
    print(f"    Started at: {start.strftime('%H:%M:%S')}\n")

    result = {}

    if args.mode == "full":
        print("Running FULL training (AI + SI + Update engine)...")
        result = await orch.run_full_training()

    elif args.mode == "quick":
        print(f"Running QUICK training — focus: {args.focus}...")
        result = await orch.run_quick_training(args.focus)

    elif args.mode == "ai":
        print("Running AI-only training (all domains, all stages)...")
        if orch.ai_trainer:
            result = await orch.ai_trainer.run_full_program()
        else:
            result = {"error": "AI trainer not loaded"}

    elif args.mode == "si":
        print("Running SI-only training (8 new modules)...")
        if orch.si_trainer:
            result = await orch.si_trainer.run_full_si_program()
        else:
            result = {"error": "SI trainer not loaded"}

    elif args.mode == "update":
        print("Running update engine only...")
        result = await orch.run_update_only()

    elif args.mode == "stage":
        print(f"Running stage-targeted training — stage: {args.stage}...")
        if orch.ai_trainer:
            result = await orch.ai_trainer.train_stage(args.stage)
        else:
            result = {"error": "AI trainer not loaded"}

    # ── Results ────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n✅  Training complete in {elapsed:.1f}s\n")
    print("📈  Results:")

    if "progress" in result:
        p = result["progress"]
        print(f"    Avg mastery:   {p.get('avg_mastery', 0):.1%}")
        print(f"    Expert count:  {p.get('expert_count', 0)} / {p.get('domains_total', 0)}")
        print(f"    By stage:      {p.get('by_stage', {})}")

    if "overall_score" in result:
        print(f"    SI score:      {result['overall_score']:.3f}")

    if "kpis" in result:
        print("    KPIs:")
        for k, v in result["kpis"].items():
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"      {k:<40s} [{bar}] {v:.3f}")

    # ── Save output ────────────────────────────────────────────────────
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾  Results saved to: {args.output}")

    # ── Live mode: start background updater ───────────────────────────
    if args.live:
        print("\n🔄  Starting background update engine (live mode)...")
        orch.start_background_updater()
        print("    Update engine running. Press Ctrl+C to stop.")
        import time
        try:
            while True:
                time.sleep(60)
                status = orch.get_status()
                upd = status.get("components", {}).get("update_engine", {})
                print(f"    [{datetime.now().strftime('%H:%M:%S')}] Update engine alive, "
                      f"jobs: {list(upd.get('jobs', {}).keys())}")
        except KeyboardInterrupt:
            print("\n⛔  Stopped.")
            orch.stop_background_updater()

    return result


if __name__ == "__main__":
    print_banner()
    args = parse_args()
    asyncio.run(run(args))
