"""Re-export of the registry router so it can mount under ``/api/v1``."""

from __future__ import annotations

from dmai.registry.api import router

__all__ = ["router"]
