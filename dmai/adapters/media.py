"""Adapter wrapping ``components/media/MediaProductionStudio``."""

from __future__ import annotations

import os
import sys
from typing import Any

from dmai.adapters._base import AdapterBase


class MediaAdapter(AdapterBase):
    """Exposes the media production studio."""

    component_id = "media_studio"
    component_name = "Media Production Studio"
    plane = "agent"
    version = "1.0.0"
    capabilities = ["media", "avatar", "video"]
    dependencies = []

    def _build_impl(self) -> Any:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        from components.media.MediaProductionStudio import MediaProductionStudio

        return MediaProductionStudio()

    async def produce(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Run a media production task if the studio exposes a method."""
        if self._impl is None:
            return {"error": self._init_error or "media unavailable"}
        for name in ("produce", "create", "generate", "run"):
            if hasattr(self._impl, name):
                return await self._call(getattr(self._impl, name), spec)
        return {"error": "no production method available"}
