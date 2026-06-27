"""
DMAI Hourly Provider Health Check
===================================
Validates all 14 providers in the AutoAPIActivator PROVIDER_CATALOGUE against
their real endpoints, compares status codes against the 200/201/429 baseline,
logs every result to data/monitoring/health_history.jsonl (append-only), and
returns a structured report for use by the cron task notification layer.

Alert triggers (only one needs to be true):
  - ANY core provider returns 401 or 402
  - TWO or more secondary providers return 401 or 402

Core providers (system cannot function without at least one):
  groq, cerebras, google_ai_studio, tavily, deepseek

Secondary providers (degrade gracefully if down):
  openrouter, cloudflare, cohere, huggingface, openai, anthropic,
  perplexity, github_models, mistral

Run:
  python scripts/hourly_health_check.py
  python scripts/hourly_health_check.py --data-path /custom/data/path

Exit codes:
  0  — all providers healthy or only pending (no alert)
  1  — alert condition met (core provider down, or 2+ secondary down)
  2  — script error (connectivity, import failure, etc.)
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ── Allow running from repo root without installing the package ───────────────
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from components.integration.auto_api_activator import PROVIDER_CATALOGUE

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("dmai.health_check")

# ── Provider classification ───────────────────────────────────────────────────
# Core = providers the system relies on for primary chat/research/training.
# If ANY core provider returns 401 or 402, alert immediately.
CORE_PROVIDERS = {
    "groq",           # Primary free chat/reasoning — 14,400 req/day
    "cerebras",       # Fastest inference — 1M tokens/day
    "google_ai_studio",  # Long-context fallback — 1,500 req/day
    "tavily",         # DeepResearch grounding — 1,000 searches/month
    "deepseek",       # Best-value paid fallback — $0.14/1M tokens
}

# Secondary = useful but DMAI degrades gracefully without them
SECONDARY_PROVIDERS = {
    "openrouter",
    "cloudflare",
    "cohere",
    "huggingface",
    "openai",
    "anthropic",
    "perplexity",
    "github_models",
    "mistral",
}

# Alert-triggering HTTP codes
ALERT_CODES = {401, 402}

# Healthy response codes (key valid, provider reachable)
HEALTHY_CODES = {200, 201, 429}  # 429 = rate-limited but key is valid

# Timeout per provider (seconds)
VALIDATION_TIMEOUT = 12

# ── History file ──────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH = repo_root / "data"
HISTORY_FILE_REL  = Path("monitoring") / "health_history.jsonl"
MAX_HISTORY_LINES = 720  # 30 days × 24 hrs


# ─────────────────────────────────────────────────────────────────────────────
# Core validation logic (mirrors AutoAPIActivator._validate but standalone)
# ─────────────────────────────────────────────────────────────────────────────

def _get_key(spec: Dict) -> Optional[str]:
    for env_var in spec.get("env_vars", []):
        val = os.environ.get(env_var, "").strip()
        if val and val.lower() not in ("", "pending", "your_value_here", "none"):
            return val
    return None


def _validate_provider(provider_id: str, spec: Dict) -> Dict:
    """
    Validate a single provider. Returns a result dict:
      {provider_id, name, status, http_code, latency_ms, error, key_present}

    status values:
      active          — 200/201/429 response
      pending_api_key — env var not set
      auth_failure    — 401 (key invalid or revoked)
      quota_exceeded  — 402 (billing limit)
      unreachable     — timeout or network error
      unexpected      — any other HTTP code
    """
    key = _get_key(spec)
    if key is None:
        return {
            "provider_id": provider_id,
            "name":        spec["name"],
            "status":      "pending_api_key",
            "http_code":   None,
            "latency_ms":  None,
            "error":       f"Set {spec['env_vars'][0]} to activate",
            "key_present": False,
        }

    val     = spec.get("validation", {})
    method  = val.get("method", "GET").upper()
    url     = val.get("url", "")
    body    = val.get("body")
    hdr_fn  = val.get("headers", lambda k: {})

    # Inject real key into Tavily body (uses body key, not header)
    if body and isinstance(body, dict):
        body = json.loads(json.dumps(body).replace('"{key}"', json.dumps(key)))

    # Cloudflare needs account_id in URL
    if spec.get("requires_account_id"):
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
        if not account_id:
            return {
                "provider_id": provider_id,
                "name":        spec["name"],
                "status":      "pending_api_key",
                "http_code":   None,
                "latency_ms":  None,
                "error":       "CLOUDFLARE_ACCOUNT_ID not set",
                "key_present": True,
            }
        url = url.replace("{account_id}", account_id)

    headers = {**hdr_fn(key), "Content-Type": "application/json"}
    t0 = time.monotonic()
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=VALIDATION_TIMEOUT)
        else:
            resp = requests.post(url, headers=headers, json=body, timeout=VALIDATION_TIMEOUT)

        latency = round((time.monotonic() - t0) * 1000, 1)
        code    = resp.status_code

        if code in HEALTHY_CODES:
            status = "active"
            error  = "rate_limited" if code == 429 else None
        elif code == 401:
            status = "auth_failure"
            error  = "Invalid or revoked API key (401)"
        elif code == 402:
            status = "quota_exceeded"
            error  = "Billing limit or quota exceeded (402)"
        else:
            status = "unexpected"
            error  = f"HTTP {code}: {resp.text[:120]}"

        return {
            "provider_id": provider_id,
            "name":        spec["name"],
            "status":      status,
            "http_code":   code,
            "latency_ms":  latency,
            "error":       error,
            "key_present": True,
        }

    except requests.Timeout:
        return {
            "provider_id": provider_id,
            "name":        spec["name"],
            "status":      "unreachable",
            "http_code":   None,
            "latency_ms":  round((time.monotonic() - t0) * 1000, 1),
            "error":       f"Timed out after {VALIDATION_TIMEOUT}s",
            "key_present": True,
        }
    except Exception as exc:
        return {
            "provider_id": provider_id,
            "name":        spec["name"],
            "status":      "unreachable",
            "http_code":   None,
            "latency_ms":  None,
            "error":       str(exc)[:200],
            "key_present": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Alert logic
# ─────────────────────────────────────────────────────────────────────────────

def _assess_alert(results: List[Dict]) -> Tuple[bool, str]:
    """
    Returns (should_alert: bool, reason: str).

    Alert triggers:
      1. Any core provider returns auth_failure or quota_exceeded (401/402)
      2. Two or more secondary providers return auth_failure or quota_exceeded
    """
    core_failures    = []
    secondary_failures = []

    for r in results:
        pid    = r["provider_id"]
        status = r["status"]
        if status not in ("auth_failure", "quota_exceeded"):
            continue
        label  = f"{r['name']} ({status}, HTTP {r['http_code']})"
        if pid in CORE_PROVIDERS:
            core_failures.append(label)
        elif pid in SECONDARY_PROVIDERS:
            secondary_failures.append(label)

    if core_failures:
        return True, (
            f"CORE PROVIDER FAILURE — {len(core_failures)} core provider(s) down: "
            + "; ".join(core_failures)
        )

    if len(secondary_failures) >= 2:
        return True, (
            f"MULTIPLE SECONDARY FAILURES — {len(secondary_failures)} secondary providers down: "
            + "; ".join(secondary_failures)
        )

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# History logging
# ─────────────────────────────────────────────────────────────────────────────

def _append_history(data_path: Path, scan_record: Dict):
    """Append one JSONL line to the health history file. Rotate at MAX_HISTORY_LINES."""
    history_file = data_path / HISTORY_FILE_REL
    history_file.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(scan_record, default=str) + "\n"

    # Rotate: keep last (MAX_HISTORY_LINES - 1) lines, then append new one
    if history_file.exists():
        with open(history_file, "r") as f:
            lines = f.readlines()
        if len(lines) >= MAX_HISTORY_LINES:
            lines = lines[-(MAX_HISTORY_LINES - 1):]
        lines.append(line)
        tmp = history_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            f.writelines(lines)
        tmp.replace(history_file)
    else:
        with open(history_file, "w") as f:
            f.write(line)


# ─────────────────────────────────────────────────────────────────────────────
# Summary builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_summary(results: List[Dict], alert: bool, reason: str) -> str:
    """Build a human-readable summary for the notification body."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    active     = [r for r in results if r["status"] == "active"]
    pending    = [r for r in results if r["status"] == "pending_api_key"]
    auth_fail  = [r for r in results if r["status"] == "auth_failure"]
    quota      = [r for r in results if r["status"] == "quota_exceeded"]
    unreachable = [r for r in results if r["status"] == "unreachable"]

    lines = [
        f"DMAI Provider Health Check — {now_str}",
        f"Total: {len(results)} providers  |  Active: {len(active)}  |  Pending key: {len(pending)}",
    ]

    if auth_fail:
        lines.append(
            "Auth failures (401): " + ", ".join(r["name"] for r in auth_fail)
        )
    if quota:
        lines.append(
            "Quota exceeded (402): " + ", ".join(r["name"] for r in quota)
        )
    if unreachable:
        lines.append(
            "Unreachable: " + ", ".join(r["name"] for r in unreachable)
        )

    if alert:
        lines.append(f"\nALERT: {reason}")
    else:
        lines.append("\nAll systems nominal — no action required.")

    # Per-provider detail for active providers (latency)
    if active:
        lines.append("\nActive provider latencies:")
        for r in sorted(active, key=lambda x: (x["latency_ms"] or 9999)):
            tier = "CORE" if r["provider_id"] in CORE_PROVIDERS else "sec."
            lat  = f"{r['latency_ms']}ms" if r["latency_ms"] else "n/a"
            note = " [rate-limited]" if r.get("error") == "rate_limited" else ""
            lines.append(f"  [{tier}] {r['name']}: {lat}{note}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_health_check(data_path: Path) -> Dict:
    """
    Run a full health check across all 14 providers.
    Returns a structured report dict.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.info("Starting health check — %d providers", len(PROVIDER_CATALOGUE))

    results = []
    for provider_id, spec in PROVIDER_CATALOGUE.items():
        logger.info("  Checking %s...", spec["name"])
        result = _validate_provider(provider_id, spec)
        results.append(result)

        tier   = "CORE" if provider_id in CORE_PROVIDERS else "sec."
        status = result["status"]
        lat    = f"{result['latency_ms']}ms" if result["latency_ms"] else "n/a"
        logger.info("    [%s] %-28s  %-18s  %s", tier, spec["name"], status, lat)

    alert, reason = _assess_alert(results)

    # Counts
    active_count     = sum(1 for r in results if r["status"] == "active")
    pending_count    = sum(1 for r in results if r["status"] == "pending_api_key")
    failure_count    = sum(1 for r in results if r["status"] in ("auth_failure", "quota_exceeded", "unreachable"))

    scan_record = {
        "timestamp":       timestamp,
        "total_providers": len(results),
        "active_count":    active_count,
        "pending_count":   pending_count,
        "failure_count":   failure_count,
        "alert_triggered": alert,
        "alert_reason":    reason if alert else None,
        "providers":       results,
    }

    # Write to history
    _append_history(data_path, scan_record)
    history_file = data_path / HISTORY_FILE_REL
    logger.info("Health history updated: %s", history_file)

    summary = _build_summary(results, alert, reason)

    report = {
        **scan_record,
        "summary": summary,
    }

    if alert:
        logger.warning("ALERT CONDITION MET: %s", reason)
    else:
        logger.info("Check complete — %d active, %d pending, %d failures. No alert.", active_count, pending_count, failure_count)

    return report


def main():
    parser = argparse.ArgumentParser(description="DMAI hourly provider health check")
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Path to DMAI data directory (default: repo/data)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report to stdout",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)

    try:
        report = run_health_check(data_path)
    except Exception as exc:
        logger.exception("Health check failed: %s", exc)
        sys.exit(2)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("\n" + report["summary"] + "\n")

    # Exit 1 if alert condition met — allows cron to detect failure
    sys.exit(1 if report["alert_triggered"] else 0)


if __name__ == "__main__":
    main()
