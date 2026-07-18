"""PR DDD-3: cron-driven email delivery with Slack fallback.

Blueprint mount path: /api/cron/*

Exposes /api/cron/promoter-drift/email so an external scheduler
(GitHub Actions in our setup, but any CRON_SECRET-holder) can push a
pre-composed drift report through Resend + Slack without needing
Perplexity credit to run.
"""
from .routes import cron_email_bp  # noqa: F401

__all__ = ["cron_email_bp"]
