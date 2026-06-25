"""
DMAI New Provider Connectors
============================
Standalone health-check and query connectors for the three new free-tier
providers added to DMAI in the June 2026 expansion:

  1. Cerebras Inference   — 1M tokens/day free, 2,600+ tok/s
  2. GitHub Models        — 45+ frontier models free with a GitHub account
  3. Mistral AI           — Experiment plan: all models, 1B tokens/month free

These connectors are designed to be used by:
  - AutoAPIActivator (health checks + hot-wiring)
  - AIIntegrationHub  (query methods _query_cerebras, _query_github_models, _query_mistral)
  - Admin Harvester panel (status reporting)

Design principles
-----------------
  - REAL data only. Never fake a validation result or fabricate a response.
  - status = "pending_api_key" when the env var is missing — never "active".
  - 429 responses are treated as valid (key works, rate-limited).
  - All timeouts default to 15 seconds to stay within Render's 30 s limit.

Usage (standalone health check):
---------------------------------
    from components.integration.new_provider_connectors import check_all
    results = check_all()
    for provider, result in results.items():
        print(f"{provider}: {result['status']} — {result.get('message')}")

Usage (individual query):
--------------------------
    from components.integration.new_provider_connectors import (
        query_cerebras, query_github_models, query_mistral
    )
    resp = query_cerebras("Explain gradient descent in one sentence.")
    if resp["success"]:
        print(resp["response"])
"""

import os
import time
import logging
import requests
from typing import Dict, Optional

logger = logging.getLogger(__name__)

TIMEOUT = 15  # seconds — safe for Render's 30 s request limit


# ─────────────────────────────────────────────────────────────────────────────
# Provider specs (mirrors PROVIDER_CATALOGUE entries in auto_api_activator.py)
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER_SPECS: Dict[str, Dict] = {
    "cerebras": {
        "name":       "Cerebras Inference",
        "env_vars":   ["CEREBRAS_API_KEY"],
        "signup_url": "https://cloud.cerebras.ai",
        "free_tier":  "1M tokens/day permanently, 30 RPM, 14,400 RPD, 60K TPM — no card required",
        "base_url":   "https://api.cerebras.ai/v1",
        "default_model": "gpt-oss-120b",
        "models": [
            "gpt-oss-120b",
            "zai-glm-4.7",
        ],
        "rate_limits": {
            "rpm": 30,
            "rpd": 14_400,
            "tpm": 60_000,
            "tokens_per_day": 1_000_000,
        },
        "auth_header": lambda k: {"Authorization": f"Bearer {k}"},
        "health_endpoint": "https://api.cerebras.ai/v1/chat/completions",
        "health_body": {
            "model": "gpt-oss-120b",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 3,
        },
    },
    "github_models": {
        "name":       "GitHub Models",
        "env_vars":   ["GITHUB_MODELS_TOKEN", "GITHUB_TOKEN_MAIN"],
        "signup_url": "https://github.com/marketplace/models",
        "free_tier":  "Free with any GitHub account — 45+ models, no credit card",
        "base_url":   "https://models.github.ai/inference",
        "default_model": "gpt-4o-mini",
        "models": [
            "gpt-4.1",
            "gpt-4o-mini",
            "o4-mini",
            "meta/llama-4-scout-17b-16e-instruct",
            "meta/llama-4-maverick-17b-128e-instruct",
            "deepseek-r1",
            "Mistral-small",
            "xai/grok-3-mini",
        ],
        "rate_limits": {
            "low_tier_rpm": 15,
            "low_tier_rpd": 150,
            "high_tier_rpm": 10,
            "high_tier_rpd": 50,
            "max_input_tokens": 8_192,
            "max_output_tokens": 4_096,
        },
        "auth_header": lambda k: {"Authorization": f"Bearer {k}"},
        "health_endpoint": "https://models.github.ai/inference/chat/completions",
        "health_body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 3,
        },
    },
    "mistral": {
        "name":       "Mistral AI",
        "env_vars":   ["MISTRAL_API_KEY"],
        "signup_url": "https://console.mistral.ai",
        "free_tier":  "Experiment plan: ALL models, 2 RPM, 500K TPM, 1B tokens/month — phone verify, no card",
        "base_url":   "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
        "models": [
            "mistral-large-latest",
            "mistral-small-latest",
            "codestral-latest",
            "pixtral-large-latest",
            "mistral-nemo",
            "open-mistral-7b",
        ],
        "rate_limits": {
            "rpm": 2,          # Experiment plan — upgrade to ~2 RPM sustained
            "tpm": 500_000,
            "tokens_per_month": 1_000_000_000,  # 1B
        },
        "auth_header": lambda k: {"Authorization": f"Bearer {k}"},
        "health_endpoint": "https://api.mistral.ai/v1/chat/completions",
        "health_body": {
            "model": "mistral-small-latest",  # Use small for health checks to spare rate limit
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 3,
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_key(spec: Dict) -> Optional[str]:
    """Return the first non-empty env var for this provider, or None."""
    for var in spec["env_vars"]:
        val = os.environ.get(var, "").strip()
        if val and val.lower() not in ("", "pending", "your_value_here", "none"):
            return val
    return None


def _openai_compat_query(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.7,
    timeout: int = TIMEOUT,
) -> Dict:
    """
    Send a chat completion request to any OpenAI-compatible endpoint.
    Returns a normalised dict: {success, response, model, latency_ms} or {success, error}.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    start = time.time()
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=timeout,
        )
        latency = round((time.time() - start) * 1000, 1)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "response": data["choices"][0]["message"]["content"],
                "model": model,
                "latency_ms": latency,
            }
        elif resp.status_code == 429:
            return {
                "success": False,
                "error": f"Rate limited (429) — check limits in PROVIDER_SPECS",
                "latency_ms": latency,
            }
        elif resp.status_code == 401:
            return {"success": False, "error": "Invalid API key (401)", "latency_ms": latency}
        else:
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "latency_ms": latency,
            }
    except requests.Timeout:
        return {"success": False, "error": f"Request timed out after {timeout}s"}
    except Exception as exc:
        return {"success": False, "error": str(exc)[:200]}


def _health_check(provider_id: str) -> Dict:
    """
    Perform a lightweight health check for a single provider.
    Returns {status, latency_ms, error, key_prefix}.
    """
    spec = PROVIDER_SPECS[provider_id]
    key = _get_key(spec)

    if key is None:
        return {
            "provider": provider_id,
            "name": spec["name"],
            "status": "pending_api_key",
            "signup_url": spec["signup_url"],
            "free_tier": spec["free_tier"],
            "env_vars": spec["env_vars"],
            "message": f"Set {spec['env_vars'][0]} to activate — no cost required",
        }

    headers = spec["auth_header"](key)
    start = time.time()
    try:
        resp = requests.post(
            spec["health_endpoint"],
            headers={**headers, "Content-Type": "application/json"},
            json=spec["health_body"],
            timeout=TIMEOUT,
        )
        latency = round((time.time() - start) * 1000, 1)

        if resp.status_code in (200, 201):
            status = "active"
            message = f"Validated OK in {latency}ms"
        elif resp.status_code == 429:
            # Key is valid — just rate-limited
            status = "active"
            message = f"Rate-limited (429) — key valid, {latency}ms"
        elif resp.status_code == 401:
            status = "invalid"
            message = "Authentication failed (401) — regenerate key"
        elif resp.status_code == 402:
            status = "quota_exceeded"
            message = "Billing limit reached (402)"
        else:
            status = "error"
            message = f"HTTP {resp.status_code}: {resp.text[:100]}"

        return {
            "provider": provider_id,
            "name": spec["name"],
            "status": status,
            "latency_ms": latency,
            "key_prefix": key[:8] + "...",
            "message": message,
            "free_tier": spec["free_tier"],
            "models": spec["models"],
            "signup_url": spec["signup_url"],
        }

    except requests.Timeout:
        return {
            "provider": provider_id,
            "name": spec["name"],
            "status": "timeout",
            "message": f"Health check timed out after {TIMEOUT}s",
        }
    except Exception as exc:
        return {
            "provider": provider_id,
            "name": spec["name"],
            "status": "error",
            "message": str(exc)[:200],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public query functions
# ─────────────────────────────────────────────────────────────────────────────

def query_cerebras(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.7,
) -> Dict:
    """
    Query Cerebras Inference.

    Free tier: 1M tokens/day, 30 RPM, 2,600+ tok/s (world's fastest).
    Best model: llama-3.3-70b

    Returns {success, response, model, latency_ms} or {success, error}.
    """
    spec = PROVIDER_SPECS["cerebras"]
    key = _get_key(spec)
    if not key:
        return {
            "success": False,
            "tutor": "Cerebras",
            "error": f"No API key — sign up free at {spec['signup_url']}",
        }
    chosen_model = model or spec["default_model"]
    result = _openai_compat_query(
        base_url=spec["base_url"],
        api_key=key,
        model=chosen_model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result["tutor"] = "Cerebras Inference"
    return result


def query_github_models(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.7,
) -> Dict:
    """
    Query GitHub Models Marketplace.

    Free tier: free with any GitHub account, 15 RPM / 150 RPD (low-tier models).
    45+ models including GPT-4.1, GPT-5, o4-mini, Llama 4, DeepSeek R1, Grok-3-Mini.

    Returns {success, response, model, latency_ms} or {success, error}.
    """
    spec = PROVIDER_SPECS["github_models"]
    key = _get_key(spec)
    if not key:
        return {
            "success": False,
            "tutor": "GitHub Models",
            "error": f"No token — set GITHUB_MODELS_TOKEN or GITHUB_TOKEN_MAIN",
        }
    chosen_model = model or spec["default_model"]
    result = _openai_compat_query(
        base_url=spec["base_url"],
        api_key=key,
        model=chosen_model,
        prompt=prompt,
        max_tokens=min(max_tokens, 4096),  # GitHub hard cap
        temperature=temperature,
    )
    result["tutor"] = "GitHub Models"
    return result


def query_mistral(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.7,
) -> Dict:
    """
    Query Mistral AI Experiment plan.

    Free tier: all models (including Large), 2 RPM, 500K TPM, 1B tokens/month.
    Note: 2 RPM means parallel queries should be serialised or throttled.

    Returns {success, response, model, latency_ms} or {success, error}.
    """
    spec = PROVIDER_SPECS["mistral"]
    key = _get_key(spec)
    if not key:
        return {
            "success": False,
            "tutor": "Mistral AI",
            "error": f"No API key — sign up free at {spec['signup_url']}",
        }
    chosen_model = model or spec["default_model"]
    result = _openai_compat_query(
        base_url=spec["base_url"],
        api_key=key,
        model=chosen_model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result["tutor"] = "Mistral AI"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_all() -> Dict[str, Dict]:
    """
    Run health checks on all three new providers.
    Returns a dict keyed by provider_id.

    Example:
        {
            "cerebras":      {"status": "active", "latency_ms": 312, ...},
            "github_models": {"status": "pending_api_key", ...},
            "mistral":       {"status": "active", "latency_ms": 891, ...},
        }
    """
    results = {}
    for provider_id in PROVIDER_SPECS:
        logger.debug("Health-checking %s...", provider_id)
        results[provider_id] = _health_check(provider_id)
    return results


def get_signup_brief() -> str:
    """Return a human-readable signup brief for all three providers."""
    lines = [
        "New Free-Tier Provider Signup Guide",
        "=" * 40,
    ]
    for pid, spec in PROVIDER_SPECS.items():
        key = _get_key(spec)
        status = "ACTIVE" if key else "PENDING — no key set"
        lines += [
            f"\n{spec['name']}",
            f"  Status   : {status}",
            f"  Free tier: {spec['free_tier']}",
            f"  Env var  : {spec['env_vars'][0]}",
            f"  Sign up  : {spec['signup_url']}",
            f"  Models   : {', '.join(spec['models'][:3])} ...",
        ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (python -m components.integration.new_provider_connectors)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("\n── DMAI New Provider Health Check ──\n")
    results = check_all()
    for pid, result in results.items():
        icon = "✅" if result.get("status") == "active" else "⏳" if result.get("status") == "pending_api_key" else "❌"
        print(f"  {icon}  {result['name']}")
        print(f"       Status  : {result['status']}")
        if result.get("latency_ms"):
            print(f"       Latency : {result['latency_ms']}ms")
        print(f"       Message : {result.get('message', '')}")
        print()

    print("── Signup Brief ──\n")
    print(get_signup_brief())
