"""
self_management_runner.py — orchestrates DMAI's three self-management components:

  1. SelfHealer     — monitors component health, auto-restarts, queues Kaizen proposals
  2. KaizenExecutor — reads kaizen queue, generates LLM patches, opens GitHub PRs
  3. RenderDeployHook — registered as Flask blueprint; triggers Render on merged auto-PRs

Call start_all(app) from _start_background_services() in dmai_core_complete.py.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("SelfManagementRunner")


def start_all(app=None, components: Optional[dict] = None):
    """
    Start all self-management background services.

    Args:
        app: Flask app instance (required for RenderDeployHook blueprint registration)
        components: the DMAI components dict (passed to SelfHealer so it can monitor them)
    """
    errors = []

    # ── 1. SelfHealer ──────────────────────────────────────────────────────────
    try:
        from components.self_management.self_healer import SelfHealer
        healer = SelfHealer(components=components or {})
        healer.start()
        logger.info("SelfHealer started")
    except Exception as e:
        logger.error("Failed to start SelfHealer: %s", e)
        errors.append(f"SelfHealer: {e}")

    # ── 2. KaizenExecutor ──────────────────────────────────────────────────────
    try:
        from components.self_management.kaizen_executor import start_background_loop
        start_background_loop()
        logger.info("KaizenExecutor started")
    except Exception as e:
        logger.error("Failed to start KaizenExecutor: %s", e)
        errors.append(f"KaizenExecutor: {e}")

    # ── 3. RenderDeployHook ────────────────────────────────────────────────────
    if app is not None:
        try:
            from components.self_management.render_deploy_hook import register
            register(app)
            logger.info("RenderDeployHook registered")
        except Exception as e:
            logger.error("Failed to register RenderDeployHook: %s", e)
            errors.append(f"RenderDeployHook: {e}")
    else:
        logger.warning("No Flask app passed to start_all — RenderDeployHook not registered")

    if errors:
        logger.warning("Self-management started with %d error(s): %s", len(errors), errors)
    else:
        logger.info("All 3 self-management components started successfully")

    return len(errors) == 0
