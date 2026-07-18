"""PR CCC-1a: DMAI external-integration API.

Public HTTP surface for other systems (Aevora satellites, greyhound
tipster clients, trader bots, external partners) to read and write
against DMAI without borrowing the internal admin credentials.

Blueprint mount path: /api/external/*

Auth: per-integration API keys stored hashed in api_keys.key_hash,
with per-key scope (space-separated tokens) and per-minute rate limit.
See auth.py for the _require_external_key(scope) decorator.

v1 endpoints (this PR ships /api/external/status only; subsequent PRs
add /api/external/insight, /api/external/signal, /api/external/webhook/<source>).
"""
from .routes import external_api_bp  # noqa: F401

__all__ = ["external_api_bp"]
