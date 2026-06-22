"""Adapter wrapping ``components/phase11/AIIntegrationHub``."""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

from dmai.adapters._base import AdapterBase


class AIHubAdapter(AdapterBase):
    """Exposes multi-model LLM querying through the AI Integration Hub."""

    component_id = "ai_hub"
    component_name = "AI Integration Hub"
    plane = "tool"
    version = "1.0.0"
    capabilities = ["llm_query", "multi_model", "synthesis"]
    dependencies = []

    def _build_impl(self) -> Any:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        from components.phase11.AIIntegrationHub import AIIntegrationHub

        return AIIntegrationHub(data_path=self._data_path("ai_hub"))

    async def query(self, prompt: str, model_preference: Optional[str] = None) -> dict[str, Any]:
        """Query all tutors and return the best/synthesised response."""
        if self._impl is None:
            return {"text": "", "error": self._init_error or "ai_hub unavailable", "model": "none"}
        result = await self._call(self._impl.query_all_tutors, prompt)
        return self._normalise(result, model_preference)

    async def query_model(self, model: str, prompt: str) -> dict[str, Any]:
        """Query and prefer a specific model's response when present."""
        result = await self.query(prompt, model_preference=model)
        return result

    async def get_best_response(self, prompt: str) -> dict[str, Any]:
        """Convenience alias for :meth:`query`."""
        return await self.query(prompt)

    @staticmethod
    def _normalise(result: Any, model_preference: Optional[str]) -> dict[str, Any]:
        """Coerce the hub's response into a uniform ``{text, model}`` dict."""
        if isinstance(result, dict):
            if "error" in result and "text" not in result:
                return {"text": "", "error": result["error"], "model": "none"}
            responses = result.get("responses") or result.get("tutors") or {}
            if isinstance(responses, dict) and responses:
                if model_preference and model_preference in responses:
                    chosen = responses[model_preference]
                else:
                    chosen = next(iter(responses.values()))
                text = chosen.get("response") if isinstance(chosen, dict) else str(chosen)
                model = model_preference or "ensemble"
                return {"text": text or "", "model": model, "raw": result}
            for key in ("synthesis", "best", "response", "text", "answer"):
                if key in result:
                    return {"text": str(result[key]), "model": "ensemble", "raw": result}
            return {"text": str(result), "model": "ensemble", "raw": result}
        return {"text": str(result), "model": "ensemble"}
