"""Service toggle manager — enables/disables background services from admin UI."""
import os, json, logging
from pathlib import Path

logger = logging.getLogger("dmai.service_toggles")

DEFAULT_TOGGLES = {
    "autonomous_trader": {"enabled": True, "label": "Autonomous Trader"},
    "prolific_worker": {"enabled": True, "label": "Prolific Worker"},
    "fiverr_worker": {"enabled": True, "label": "Fiverr Worker"},
    "greyhound_runner": {"enabled": True, "label": "Greyhound Runner (Betting)"},
    "parallel_web_learner": {"enabled": True, "label": "Parallel Web Learner"},
    "alex_riviera_content": {"enabled": True, "label": "Alex Riviera Content"},
    "self_funding": {"enabled": True, "label": "Self-Funding System"},
    "muse_glimmer": {"enabled": True, "label": "Muse-Glimmer Ingestion"},
}

TOGGLE_FILE = "data/service_toggles.json"


def load_toggles() -> dict:
    """Load current toggle state from file. Falls back to defaults."""
    try:
        with open(TOGGLE_FILE, "r") as f:
            saved = json.load(f)
            # Merge with defaults (in case new services added)
            for key, val in DEFAULT_TOGGLES.items():
                if key not in saved:
                    saved[key] = val
            return saved
    except Exception:
        return dict(DEFAULT_TOGGLES)


def save_toggles(toggles: dict) -> bool:
    """Persist toggle state."""
    try:
        with open(TOGGLE_FILE, "w") as f:
            json.dump(toggles, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save toggles: {e}")
        return False


def is_enabled(service_key: str) -> bool:
    """Check if a service is enabled."""
    toggles = load_toggles()
    service = toggles.get(service_key, DEFAULT_TOGGLES.get(service_key))
    if service:
        return service.get("enabled", True)
    return True


def set_enabled(service_key: str, enabled: bool) -> bool:
    """Enable or disable a service."""
    toggles = load_toggles()
    if service_key in toggles:
        toggles[service_key]["enabled"] = enabled
        return save_toggles(toggles)
    return False
