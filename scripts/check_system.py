#!/usr/bin/env python3
"""
DMAI System Health Checker
===========================
Run before first deploy to verify all components are importable and all
critical environment variables are set.

Usage:
    python scripts/check_system.py
    python scripts/check_system.py --url https://dmai-complete.onrender.com
"""

import sys
import os
import json
import importlib
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

RESET   = "\033[0m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
BOLD    = "\033[1m"

def ok(msg):   print(f"  {GREEN}✅  {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️   {msg}{RESET}")
def fail(msg): print(f"  {RED}❌  {msg}{RESET}")
def info(msg): print(f"  {CYAN}ℹ️   {msg}{RESET}")
def section(title): print(f"\n{BOLD}{CYAN}{'─'*55}\n  {title}\n{'─'*55}{RESET}")


def check_env_vars():
    section("Environment Variables")
    required = {
        "MASTER_PASSWORD": "Admin access password",
    }
    recommended = {
        "DATABASE_URL":       "PostgreSQL (required for production)",
        "OPENAI_API_KEY":     "Primary LLM provider",
        "ANTHROPIC_API_KEY":  "Claude (fallback LLM)",
        "ELEVENLABS_API_KEY": "Alex Riviera TTS voice",
        "TELEGRAM_BOT_TOKEN": "Telegram bot control",
        "TELEGRAM_CHAT_ID":   "Telegram notifications",
    }
    optional = {
        "DEEPSEEK_API_KEY":   "DeepSeek LLM",
        "GEMINI_API_KEY":     "Google Gemini",
        "PERPLEXITY_API_KEY": "Perplexity search",
        "XAI_API_KEY":        "xAI Grok",
        "MISTRAL_API_KEY":    "Mistral (extended hub)",
        "STABILITY_API_KEY":  "Stability AI images",
        "RUNWAY_API_KEY":     "Runway video generation",
        "REPLICATE_API_KEY":  "Replicate open models",
        "PINECONE_API_KEY":   "Pinecone vector DB",
        "TOGETHER_API_KEY":   "Together AI inference",
        "COHERE_API_KEY":     "Cohere reranking",
        "GITHUB_TOKEN_MAIN":  "GitHub integration",
        "ALPACA_API_KEY":     "Trading (Alpaca)",
        "KDP_EMAIL":          "Amazon KDP publishing",
    }

    issues = 0
    for var, desc in required.items():
        val = os.environ.get(var, "")
        if val:
            ok(f"{var} — set")
        else:
            fail(f"{var} — MISSING ({desc})")
            issues += 1

    print()
    set_count = 0
    for var, desc in recommended.items():
        val = os.environ.get(var, "")
        if val:
            ok(f"{var} — set")
            set_count += 1
        else:
            warn(f"{var} — not set ({desc})")
    info(f"{set_count}/{len(recommended)} recommended vars set")

    print()
    opt_count = 0
    for var, desc in optional.items():
        val = os.environ.get(var, "")
        if val:
            opt_count += 1
    info(f"{opt_count}/{len(optional)} optional vars set (more = more capabilities)")

    return issues


def check_imports():
    section("Component Imports")
    components = [
        ("dmai_syllabus_data",                                "SYLLABUS_TOPICS",         "required"),
        ("components.si_core",                                "SICore",                  "required"),
        ("components.phase11.AIIntegrationHub",               "AIIntegrationHub",        "required"),
        ("components.phase11.ExtendedAIIntegrationHub",       "ExtendedAIIntegrationHub","required"),
        ("components.training.ComprehensiveAITraining",       "ComprehensiveAITraining", "required"),
        ("components.si_training.FullSITrainingProgram",      "FullSITrainingProgram",   "required"),
        ("components.update_engine.PeriodicUpdateEngine",     "PeriodicUpdateEngine",    "required"),
        ("components.orchestrator.DMAITrainingOrchestrator",  "DMAITrainingOrchestrator","required"),
        ("components.evolution_training.EvolutionTrainingSystem","EvolutionTrainingSystem","required"),
        ("components.llm_training.LLMTrainingProgram",        "LLMTrainingProgram",      "optional"),
        ("components.genai_training.GenAITrainingProgram",    "GenAITrainingProgram",    "optional"),
        ("components.media.AvatarSystem",                     "AlexRivieraAvatar",       "optional"),
        ("components.media.MediaProductionStudio",            "MediaProductionStudio",   "optional"),
        ("components.alex_riviera.identity",                  "ALEX_RIVIERA",            "optional"),
        ("components.alex_riviera.content_generator",         "AlexRivieraContent",      "optional"),
        ("components.voice.VoiceIntegration",                 "VoiceIntegration",        "optional"),
    ]

    failed_required = 0
    for module_path, class_name, level in components:
        try:
            mod = importlib.import_module(module_path)
            if hasattr(mod, class_name):
                ok(f"{module_path}.{class_name}")
            else:
                warn(f"{module_path} imported but {class_name} not found")
        except ImportError as e:
            if level == "required":
                fail(f"{module_path} — {e}")
                failed_required += 1
            else:
                warn(f"{module_path} — {e} (optional)")
        except Exception as e:
            warn(f"{module_path} — runtime error: {str(e)[:60]}")

    return failed_required


def check_data_dirs():
    section("Data Directories")
    data_path = os.environ.get("DATA_PATH", "data/")
    dirs = [
        data_path,
        f"{data_path}avatars/canonical",
        f"{data_path}learning",
        f"{data_path}alex_projects",
        f"{data_path}art",
        f"{data_path}media/productions",
    ]
    for d in dirs:
        p = Path(d)
        if p.exists():
            ok(f"{d} exists")
        else:
            p.mkdir(parents=True, exist_ok=True)
            info(f"{d} — created")


def check_avatar_files():
    section("Alex Riviera Avatar Files")
    critical = [
        "data/avatars/canonical/alex_riviera_master_profile.json",
        "data/avatars/alex_riviera_avatar.json",
    ]
    for f in critical:
        p = Path(f)
        if p.exists():
            ok(f)
        else:
            warn(f"{f} — missing (AvatarSystem will fail)")


def check_live_endpoint(url: str):
    section(f"Live Endpoint: {url}")
    import urllib.request
    import urllib.error
    endpoints = [
        ("/health",               "Health check"),
        ("/api/status",           "API status"),
        ("/api/training/status",  "Training status"),
        ("/api/kaizen",           "Kaizen"),
    ]
    for path, desc in endpoints:
        full_url = url.rstrip("/") + path
        try:
            with urllib.request.urlopen(full_url, timeout=10) as resp:
                data = json.loads(resp.read())
                ok(f"{path} → {data.get('status', 'ok')}")
        except urllib.error.HTTPError as e:
            warn(f"{path} → HTTP {e.code}")
        except Exception as e:
            fail(f"{path} → {str(e)[:50]}")


def main():
    parser = argparse.ArgumentParser(description="DMAI System Health Checker")
    parser.add_argument("--url", default=None, help="Live deployment URL to test")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*55}")
    print("  DMAI v6.0.0 — System Health Check")
    print(f"{'═'*55}{RESET}")

    env_issues    = check_env_vars()
    import_issues = check_imports()
    check_data_dirs()
    check_avatar_files()

    if args.url:
        check_live_endpoint(args.url)

    section("Summary")
    if env_issues == 0 and import_issues == 0:
        print(f"  {GREEN}{BOLD}🚀  All checks passed — DMAI is ready to run!{RESET}")
    else:
        if env_issues > 0:
            warn(f"{env_issues} required env var(s) missing")
        if import_issues > 0:
            fail(f"{import_issues} required component(s) failed to import")
        print(f"\n  Fix the issues above, then run again.")

    print()


if __name__ == "__main__":
    main()
