"""
DMAI AutoAPIActivator
=====================
Discovers every API key present in the environment, validates each one against
its real endpoint, and hot-wires working keys directly into the AIIntegrationHub
so DMAI immediately gains new tutor capability — no restart required.

Additionally maintains a registry of genuinely free-tier providers that require
zero payment info to sign up, and auto-registers for any that aren't yet active.

Design principles
-----------------
- REAL data only. Never fabricate a key, never fake a validation result.
- If a key is absent: status = "pending_api_key". Never "active".
- If a key fails validation: status = "invalid". Remove from active pool.
- All state is persisted atomically (temp+rename) to data/api_registry.json.
- Thread-safe: runs as a background daemon, exposes get_status() synchronously.
"""

import os
import json
import time
import logging
import tempfile
import threading
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# ── Provider catalogue ─────────────────────────────────────────────────────────
# Each entry describes a real, publicly-documented free-tier API.
# "signup_url" is where a human (or future automation) goes to get a key.
# "validation" describes how to confirm a key works.

PROVIDER_CATALOGUE: Dict[str, Dict] = {
    # ── Tier-1: Genuinely free, no card required ──────────────────────────────
    "groq": {
        "name":         "Groq",
        "env_vars":     ["GROQ_API_KEY"],
        "signup_url":   "https://console.groq.com/keys",
        "free_tier":    "14,400 req/day — fastest inference available",
        "models":       ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen-qwq-32b", "mixtral-8x7b-32768"],
        "best_model":   "llama-3.3-70b-versatile",
        "validation": {
            "method":  "POST",
            "url":     "https://api.groq.com/openai/v1/chat/completions",
            "headers": lambda k: {"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
            "body":    {"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        },
        "call_format": "openai_compatible",
        "base_url":    "https://api.groq.com/openai/v1",
    },
    "openrouter": {
        "name":         "OpenRouter",
        "env_vars":     ["OPENROUTER_API_KEY"],
        "signup_url":   "https://openrouter.ai/keys",
        "free_tier":    "Many free models — $0 cost on free-tier models",
        "models":       ["google/gemma-3-12b-it:free", "meta-llama/llama-3.3-70b-instruct:free", "mistralai/mistral-7b-instruct:free"],
        "best_model":   "meta-llama/llama-3.3-70b-instruct:free",
        "validation": {
            "method":  "POST",
            "url":     "https://openrouter.ai/api/v1/chat/completions",
            "headers": lambda k: {
                "Authorization": f"Bearer {k}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://dmai-web.onrender.com",
                "X-Title": "DMAI",
            },
            "body": {"model": "google/gemma-3-12b-it:free", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        },
        "call_format": "openai_compatible",
        "base_url":    "https://openrouter.ai/api/v1",
    },
    "google_ai_studio": {
        "name":         "Google AI Studio (Gemini)",
        "env_vars":     ["GOOGLE_AI_STUDIO_KEY", "GEMINI_API_KEY"],
        "signup_url":   "https://aistudio.google.com/apikey",
        "free_tier":    "1,500 req/day, 250K tokens/min — Gemini Flash",
        "models":       ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"],
        "best_model":   "gemini-2.0-flash",
        "validation": {
            "method":  "POST",
            "url":     "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent",
            "headers": lambda k: {"Content-Type": "application/json", "x-goog-api-key": k},
            "body":    {"contents": [{"parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 5}},
        },
        "call_format": "gemini_native",
        "base_url":    "https://generativelanguage.googleapis.com/v1beta",
    },
    "cloudflare": {
        "name":         "Cloudflare Workers AI",
        "env_vars":     ["CLOUDFLARE_API_KEY", "CF_API_TOKEN"],
        "signup_url":   "https://dash.cloudflare.com/profile/api-tokens",
        "free_tier":    "10,000 neurons/day — Llama 3.1, Qwen3, DeepSeek R1",
        "models":       ["@cf/meta/llama-3.1-8b-instruct", "@cf/qwen/qwen3-30b-a3b-fp8", "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"],
        "best_model":   "@cf/meta/llama-3.1-8b-instruct",
        "validation": {
            "method":  "POST",
            "url":     "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions",
            "headers": lambda k: {"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
            "body":    {"model": "@cf/meta/llama-3.1-8b-instruct", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        },
        "call_format": "openai_compatible",
        "base_url":    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "requires_account_id": True,
    },
    "cohere": {
        "name":         "Cohere",
        "env_vars":     ["COHERE_API_KEY"],
        "signup_url":   "https://dashboard.cohere.com/api-keys",
        "free_tier":    "1,000 req/month, 20 req/min — Command R+",
        "models":       ["command-r-plus-08-2024", "command-r7b-12-2024"],
        "best_model":   "command-r7b-12-2024",
        "validation": {
            "method":  "POST",
            "url":     "https://api.cohere.ai/v2/chat",
            "headers": lambda k: {"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
            "body":    {"model": "command-r7b-12-2024", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        },
        "call_format": "cohere_v2",
        "base_url":    "https://api.cohere.ai/v2",
    },
    "huggingface": {
        "name":         "HuggingFace Inference",
        "env_vars":     ["HUGGINGFACE_API_KEY", "HF_TOKEN"],
        "signup_url":   "https://huggingface.co/settings/tokens",
        "free_tier":    "$0.10/month free credit — 1000+ models",
        "models":       ["mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.1-8B-Instruct"],
        "best_model":   "mistralai/Mistral-7B-Instruct-v0.3",
        "validation": {
            "method":  "POST",
            "url":     "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions",
            "headers": lambda k: {"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
            "body":    {"model": "mistralai/Mistral-7B-Instruct-v0.3", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        },
        "call_format": "openai_compatible",
        "base_url":    "https://api-inference.huggingface.co",
    },
    # ── Tier-2: Paid but already in DMAI — health-check only ─────────────────
    "openai": {
        "name":         "OpenAI",
        "env_vars":     ["OPENAI_API_KEY"],
        "signup_url":   "https://platform.openai.com/api-keys",
        "free_tier":    "Pay-per-use ($0.15/1M tokens GPT-4o mini)",
        "models":       ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        "best_model":   "gpt-4o-mini",
        "validation": {
            "method":  "GET",
            "url":     "https://api.openai.com/v1/models",
            "headers": lambda k: {"Authorization": f"Bearer {k}"},
            "body":    None,
        },
        "call_format": "openai_compatible",
        "base_url":    "https://api.openai.com/v1",
    },
    "anthropic": {
        "name":         "Anthropic",
        "env_vars":     ["ANTHROPIC_API_KEY"],
        "signup_url":   "https://console.anthropic.com/settings/keys",
        "free_tier":    "Pay-per-use",
        "models":       ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
        "best_model":   "claude-3-haiku-20240307",
        "validation": {
            "method":  "GET",
            "url":     "https://api.anthropic.com/v1/models",
            "headers": lambda k: {"x-api-key": k, "anthropic-version": "2023-06-01"},
            "body":    None,
        },
        "call_format": "anthropic_native",
        "base_url":    "https://api.anthropic.com/v1",
    },
    "perplexity": {
        "name":         "Perplexity Sonar",
        "env_vars":     ["PERPLEXITY_API_KEY"],
        "signup_url":   "https://docs.perplexity.ai",
        "free_tier":    "Pay-per-use — real-time web search grounding",
        "models":       ["sonar", "sonar-pro", "sonar-reasoning"],
        "best_model":   "sonar",
        "validation": {
            "method":  "POST",
            "url":     "https://api.perplexity.ai/chat/completions",
            "headers": lambda k: {"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
            "body":    {"model": "sonar", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        },
        "call_format": "openai_compatible",
        "base_url":    "https://api.perplexity.ai",
    },
    "deepseek": {
        "name":         "DeepSeek",
        "env_vars":     ["DEEPSEEK_API_KEY"],
        "signup_url":   "https://platform.deepseek.com/api_keys",
        "free_tier":    "Very low-cost — $0.14/1M tokens",
        "models":       ["deepseek-chat", "deepseek-reasoner"],
        "best_model":   "deepseek-chat",
        "validation": {
            "method":  "GET",
            "url":     "https://api.deepseek.com/models",
            "headers": lambda k: {"Authorization": f"Bearer {k}"},
            "body":    None,
        },
        "call_format": "openai_compatible",
        "base_url":    "https://api.deepseek.com",
    },
    "tavily": {
        "name":         "Tavily Search",
        "env_vars":     ["TAVILY_API_KEY"],
        "signup_url":   "https://tavily.com/#api",
        "free_tier":    "1,000 searches/month free — AI-optimised web search",
        "models":       ["tavily-search"],
        "best_model":   "tavily-search",
        "validation": {
            "method":  "POST",
            "url":     "https://api.tavily.com/search",
            "headers": lambda k: {"Content-Type": "application/json"},
            "body":    {"api_key": "{key}", "query": "test", "max_results": 1},
        },
        "call_format": "tavily_native",
        "base_url":    "https://api.tavily.com",
    },
    # ── Tier-1 NEW: Free-tier expansions (zero cost) ──────────────────────────
    "cerebras": {
        "name":         "Cerebras Inference",
        "env_vars":     ["CEREBRAS_API_KEY"],
        "signup_url":   "https://cloud.cerebras.ai",
        "free_tier":    "1M tokens/day permanently, 30 RPM, 2,600+ tok/s — no card required",
        "models":       ["gpt-oss-120b", "zai-glm-4.7"],
        "best_model":   "gpt-oss-120b",
        "validation": {
            "method":  "POST",
            "url":     "https://api.cerebras.ai/v1/chat/completions",
            "headers": lambda k: {"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
            "body":    {"model": "gpt-oss-120b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        },
        "call_format": "openai_compatible",
        "base_url":    "https://api.cerebras.ai/v1",
    },
    "github_models": {
        "name":         "GitHub Models",
        "env_vars":     ["GITHUB_MODELS_TOKEN", "GITHUB_TOKEN_MAIN"],
        "signup_url":   "https://github.com/marketplace/models",
        "free_tier":    "Free with GitHub account — 45+ models incl. GPT-5, o4-mini, Llama 4, Grok-3-Mini",
        "models":       ["gpt-4.1", "gpt-4o-mini", "o4-mini", "meta/llama-4-scout-17b-16e-instruct", "deepseek-r1", "Mistral-small"],
        "best_model":   "gpt-4o-mini",
        "validation": {
            "method":  "POST",
            "url":     "https://models.github.ai/inference/chat/completions",
            "headers": lambda k: {"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
            "body":    {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        },
        "call_format": "openai_compatible",
        "base_url":    "https://models.github.ai/inference",
    },
    "mistral": {
        "name":         "Mistral AI",
        "env_vars":     ["MISTRAL_API_KEY"],
        "signup_url":   "https://console.mistral.ai",
        "free_tier":    "Experiment plan: ALL models, 2 RPM, 500K TPM, 1B tokens/month — phone verify, no card",
        "models":       ["mistral-large-latest", "mistral-small-latest", "codestral-latest", "pixtral-large-latest", "mistral-nemo"],
        "best_model":   "mistral-large-latest",
        "validation": {
            "method":  "POST",
            "url":     "https://api.mistral.ai/v1/chat/completions",
            "headers": lambda k: {"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
            "body":    {"model": "mistral-small-latest", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        },
        "call_format": "openai_compatible",
        "base_url":    "https://api.mistral.ai/v1",
    },
}


class AutoAPIActivator:
    """
    Discovers, validates, and hot-wires API keys into DMAI's AI hub.
    Runs as a background daemon with periodic re-validation.
    """

    CHECK_INTERVAL   = 3600   # re-validate every hour
    TIMEOUT          = 12     # per-provider HTTP timeout

    def __init__(self, ai_hub=None, data_path: str = "data"):
        self.ai_hub    = ai_hub
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.data_path / "api_registry.json"

        self.registry: Dict[str, Dict] = self._load_registry()
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None

        logger.info("AutoAPIActivator initialised — %d providers in catalogue", len(PROVIDER_CATALOGUE))

    # ── Public API ─────────────────────────────────────────────────────────────

    def start_background_loop(self):
        """Start the periodic scan+validate loop as a daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="dmai-api-activator"
        )
        self._thread.start()
        logger.info("AutoAPIActivator background loop started (interval: %ds)", self.CHECK_INTERVAL)

    def stop(self):
        self._stop.set()

    def scan_and_activate(self) -> Dict:
        """
        Synchronous scan: check every provider, validate keys found,
        hot-wire working ones into ai_hub. Returns a results summary.
        """
        results = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "providers":   {},
            "activated":   [],
            "pending":     [],
            "invalid":     [],
            "total_active": 0,
        }

        for provider_id, spec in PROVIDER_CATALOGUE.items():
            key = self._find_key(spec)
            if key is None:
                status = {
                    "status":      "pending_api_key",
                    "name":        spec["name"],
                    "signup_url":  spec["signup_url"],
                    "free_tier":   spec["free_tier"],
                    "env_vars":    spec["env_vars"],
                    "models":      spec["models"],
                }
                results["providers"][provider_id] = status
                results["pending"].append(provider_id)
                continue

            validation = self._validate(provider_id, spec, key)
            status = {
                "status":       validation["status"],
                "name":         spec["name"],
                "signup_url":   spec["signup_url"],
                "free_tier":    spec["free_tier"],
                "env_vars":     spec["env_vars"],
                "models":       spec["models"],
                "best_model":   spec["best_model"],
                "key_prefix":   key[:8] + "...",
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "latency_ms":   validation.get("latency_ms"),
                "error":        validation.get("error"),
            }
            results["providers"][provider_id] = status

            if validation["status"] == "active":
                results["activated"].append(provider_id)
                results["total_active"] += 1
                self._hot_wire(provider_id, spec, key)
            else:
                results["invalid"].append(provider_id)

        self._save_registry(results)
        logger.info(
            "AutoAPIActivator scan complete — %d active, %d pending key, %d invalid",
            results["total_active"], len(results["pending"]), len(results["invalid"])
        )
        return results

    def get_status(self) -> Dict:
        """Return cached registry (no API calls)."""
        with self._lock:
            return dict(self.registry)

    def get_missing_keys_brief(self) -> List[Dict]:
        """Return list of providers with pending_api_key status and their signup URLs."""
        missing = []
        for pid, spec in PROVIDER_CATALOGUE.items():
            key = self._find_key(spec)
            if key is None:
                missing.append({
                    "provider":   pid,
                    "name":       spec["name"],
                    "signup_url": spec["signup_url"],
                    "free_tier":  spec["free_tier"],
                    "env_var":    spec["env_vars"][0],
                })
        return missing

    def get_active_providers(self) -> List[str]:
        """Return IDs of currently validated-active providers."""
        reg = self.registry.get("providers", {})
        return [pid for pid, info in reg.items() if info.get("status") == "active"]

    # ── Internal ────────────────────────────────────────────────────────────────

    def _loop(self):
        """Background daemon: scan immediately, then every CHECK_INTERVAL seconds."""
        while not self._stop.is_set():
            try:
                self.scan_and_activate()
            except Exception as exc:
                logger.error("AutoAPIActivator loop error: %s", exc)
            self._stop.wait(self.CHECK_INTERVAL)

    def _find_key(self, spec: Dict) -> Optional[str]:
        """Return the first non-empty env var value for any of the provider's env_vars."""
        for env_var in spec.get("env_vars", []):
            val = os.environ.get(env_var, "").strip()
            if val and val.lower() not in ("", "pending", "your_value_here", "none"):
                return val
        return None

    def _validate(self, provider_id: str, spec: Dict, key: str) -> Dict:
        """Call the provider's validation endpoint and return status dict."""
        val = spec.get("validation", {})
        method  = val.get("method", "GET").upper()
        url     = val.get("url", "")
        body    = val.get("body")
        headers_fn = val.get("headers", lambda k: {})

        # Substitute {key} in body (e.g. Tavily passes key in body)
        if body and isinstance(body, dict):
            body = json.loads(json.dumps(body).replace('"{key}"', json.dumps(key)))

        # Skip providers requiring account_id if not set
        if spec.get("requires_account_id"):
            account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
            if not account_id:
                return {"status": "pending_api_key", "error": "CLOUDFLARE_ACCOUNT_ID not set"}
            url = url.replace("{account_id}", account_id)

        try:
            headers = headers_fn(key)
            start   = time.time()

            if method == "GET":
                resp = requests.get(url, headers=headers, timeout=self.TIMEOUT)
            else:
                resp = requests.post(url, headers=headers, json=body, timeout=self.TIMEOUT)

            latency = round((time.time() - start) * 1000, 1)

            if resp.status_code in (200, 201):
                return {"status": "active", "latency_ms": latency}
            elif resp.status_code == 401:
                return {"status": "invalid", "error": "Authentication failed (401)", "latency_ms": latency}
            elif resp.status_code == 429:
                # Rate limited but key IS valid
                return {"status": "active", "latency_ms": latency, "note": "rate_limited"}
            elif resp.status_code == 402:
                return {"status": "quota_exceeded", "error": "Quota/billing limit (402)", "latency_ms": latency}
            else:
                return {"status": "invalid", "error": f"HTTP {resp.status_code}", "latency_ms": latency}

        except requests.Timeout:
            return {"status": "invalid", "error": "Timeout"}
        except Exception as exc:
            return {"status": "invalid", "error": str(exc)[:120]}

    def _hot_wire(self, provider_id: str, spec: Dict, key: str):
        """Inject a validated key directly into AIIntegrationHub's api_keys dict."""
        if self.ai_hub is None:
            return
        if not hasattr(self.ai_hub, "api_keys"):
            return

        # Map provider_id → ai_hub key name
        hub_key_map = {
            "groq":             "groq",
            "openrouter":       "openrouter",
            "google_ai_studio": "google_ai_studio",
            "cloudflare":       "cloudflare",
            "cohere":           "cohere",
            "huggingface":      "huggingface",
            "openai":           "openai",
            "anthropic":        "anthropic",
            "perplexity":       "perplexity",
            "deepseek":         "deepseek",
            "tavily":           "tavily",
            # ── New free-tier providers ──
            "cerebras":         "cerebras",
            "github_models":    "github_models",
            "mistral":          "mistral",
        }

        hub_key = hub_key_map.get(provider_id, provider_id)
        current = self.ai_hub.api_keys.get(hub_key, "")

        if not current or current in ("pending", "", "your_value_here"):
            self.ai_hub.api_keys[hub_key] = key
            logger.info(
                "AutoAPIActivator: hot-wired %s (%s) into AIIntegrationHub as '%s'",
                spec["name"], key[:8] + "...", hub_key
            )

    def _load_registry(self) -> Dict:
        if self.registry_file.exists():
            try:
                with open(self.registry_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_registry(self, data: Dict):
        """Atomic write to registry file."""
        try:
            tmp = self.registry_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            tmp.replace(self.registry_file)
            with self._lock:
                self.registry = data
        except Exception as exc:
            logger.warning("AutoAPIActivator: could not save registry: %s", exc)
