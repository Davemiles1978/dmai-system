"""Adapter wrapping the Alex Riviera content/publishing modules."""

from __future__ import annotations

import os
import sys
from typing import Any

from dmai.adapters._base import AdapterBase


class AlexRivieraAdapter(AdapterBase):
    """Exposes content generation and publishing for the Alex Riviera persona."""

    component_id = "alex_riviera"
    component_name = "Alex Riviera Content Engine"
    plane = "agent"
    version = "1.0.0"
    capabilities = ["content_generation", "book_writing", "publishing"]
    dependencies = ["ai_hub"]

    def _build_impl(self) -> Any:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        from components.alex_riviera.publishing_orchestrator import AlexRivieraPublishing

        return AlexRivieraPublishing()

    async def generate_content(self, topic: str, content_type: str = "article") -> dict[str, Any]:
        """Generate content of *content_type* for *topic* via the AI hub."""
        from dmai.core.orchestrator import orchestrator

        hub = orchestrator.registry.get("ai_hub")
        if hub is None:
            return {"error": "ai_hub unavailable"}
        prompt = f"Write a {content_type} about '{topic}' in the voice of Alex Riviera."
        result = await hub.query(prompt)  # type: ignore[attr-defined]
        return {"topic": topic, "type": content_type, "content": result.get("text", "")}

    async def publish(self, content: dict[str, Any], platform: str = "pending") -> dict[str, Any]:
        """Submit content for approval before publishing (operator-gated)."""
        if self._impl is None:
            return {"error": self._init_error or "publishing unavailable"}
        return await self._call(self._impl.submit_for_approval, content, platform)
