#!/usr/bin/env python3
"""DMAI unified entrypoint — FastAPI (port 8000) + legacy Flask (port 5001).

The FastAPI app runs as the main process. The existing Flask application is
started in a background daemon thread when it can be imported; if it cannot
(missing optional deps), DMAI still boots with the FastAPI layer only.
"""

from __future__ import annotations

import logging
import os
import threading

import uvicorn

from dmai.api.main import app as fastapi_app
from dmai.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dmai.main")


def run_legacy_flask() -> None:
    """Start the legacy Flask app on the configured Flask port."""
    os.environ.setdefault("DISABLE_NEO4J", "true")
    os.environ.setdefault("DISABLE_VOICE", "true")
    try:
        from dmai_core_complete_fixed import get_dmai_app

        flask_app = get_dmai_app().app
        port = int(os.environ.get("PORT", settings.flask_port))
        logger.info("Starting legacy Flask on port %s", port)
        flask_app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
    except Exception as exc:  # pragma: no cover - legacy is optional
        logger.warning("Legacy Flask not started: %s", exc)


if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_legacy_flask, daemon=True)
    flask_thread.start()

    uvicorn.run(fastapi_app, host="0.0.0.0", port=settings.api_port, reload=False)
