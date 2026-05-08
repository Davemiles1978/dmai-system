"""Minimal wrapper – adds blueprint routes without touching dmai_core_complete.py"""
import logging
from dmai_core_complete import get_dmai_app
from dmai_api_routes import api_bp

logger = logging.getLogger(__name__)

# Get the real DMAI app
dmai = get_dmai_app()
app = dmai.app

# Register our new routes
app.register_blueprint(api_bp)
logger.info("✅ Blueprint routes registered via wrapper")
