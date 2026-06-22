"""
DMAI ExtendedAIIntegrationHub
==============================
Extends the existing AIIntegrationHub (phase11/AIIntegrationHub.py) with
the providers currently missing from DMAI's multi-provider layer:

  New providers added:
  ┌─────────────────┬──────────────────────────────────────────────────────┐
  │ Provider        │ Capabilities                                          │
  ├─────────────────┼──────────────────────────────────────────────────────┤
  │ Mistral         │ Text generation, chat, function calling               │
  │ Stability AI    │ Image generation (SDXL, SD3), image editing           │
  │ ElevenLabs      │ TTS, voice cloning, voice design                      │
  │ Runway ML       │ Text-to-video, image-to-video, frame interpolation    │
  │ Replicate       │ Open-source model hosting (Flux, Llama, etc.)         │
  │ Pinecone        │ Vector DB: upsert, query, delete                      │
  │ Together AI     │ Open-source model inference (fast & cheap)            │
  │ Cohere          │ Reranking, embeddings, command-r chat                 │
  └─────────────────┴──────────────────────────────────────────────────────┘

Usage:
    hub = ExtendedAIIntegrationHub(
        data_path = "data/",
        base_hub  = existing_ai_integration_hub_instance,  # pass existing hub
    )
    # Or use standalone:
    hub = ExtendedAIIntegrationHub(data_path="data/")
    response = asyncio.run(hub.chat("Hello", provider="mistral"))
    image    = asyncio.run(hub.generate_image("A sunset over the sea", provider="stability"))
    audio    = asyncio.run(hub.text_to_speech("Hello world", provider="elevenlabs"))
    video    = asyncio.run(hub.generate_video("A dog running", provider="runway"))
"""

import asyncio
import base64
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("dmai.extended_hub")

# ---------------------------------------------------------------------------
# Provider configs — read keys from environment
# ---------------------------------------------------------------------------
PROVIDER_CONFIGS = {
    "mistral": {
        "base_url":    "https://api.mistral.ai/v1",
        "key_env":     "MISTRAL_API_KEY",
        "models":      ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest",
                        "open-mistral-nemo", "codestral-latest"],
        "default_model": "mistral-large-latest",
        "capabilities": ["chat", "code", "function_calling"],
    },
    "stability": {
        "base_url":    "https://api.stability.ai",
        "key_env":     "STABILITY_API_KEY",
        "models":      ["sd3-large", "sd3-medium", "sdxl-1.0", "stable-image/generate/ultra"],
        "default_model": "sd3-large",
        "capabilities": ["image_generation", "image_editing", "image_upscale"],
    },
    "elevenlabs": {
        "base_url":    "https://api.elevenlabs.io/v1",
        "key_env":     "ELEVENLABS_API_KEY",
        "models":      ["eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5"],
        "default_model": "eleven_multilingual_v2",
        "capabilities": ["tts", "voice_clone", "voice_design"],
        "voices": {
            "alex_riviera": "21m00Tcm4TlvDq8ikWAM",  # default; override with actual voice ID
            "default":      "21m00Tcm4TlvDq8ikWAM",
        },
    },
    "runway": {
        "base_url":    "https://api.dev.runwayml.com/v1",
        "key_env":     "RUNWAY_API_KEY",
        "models":      ["gen3a_turbo", "gen4_turbo"],
        "default_model": "gen4_turbo",
        "capabilities": ["text_to_video", "image_to_video", "frame_interpolation"],
    },
    "replicate": {
        "base_url":    "https://api.replicate.com/v1",
        "key_env":     "REPLICATE_API_KEY",
        "models":      [
            "black-forest-labs/flux-schnell",
            "black-forest-labs/flux-dev",
            "meta/llama-3.1-405b-instruct",
            "stability-ai/sdxl",
        ],
        "default_model": "black-forest-labs/flux-schnell",
        "capabilities": ["image_generation", "chat", "video"],
    },
    "pinecone": {
        "base_url":    "https://api.pinecone.io",
        "key_env":     "PINECONE_API_KEY",
        "capabilities": ["vector_upsert", "vector_query", "vector_delete"],
        "index_name":  os.environ.get("PINECONE_INDEX", "dmai-knowledge"),
    },
    "together": {
        "base_url":    "https://api.together.xyz/v1",
        "key_env":     "TOGETHER_API_KEY",
        "models":      ["meta-llama/Llama-3.1-70B-Instruct-Turbo",
                        "mistralai/Mixtral-8x22B-Instruct-v0.1",
                        "Qwen/Qwen2-72B-Instruct"],
        "default_model": "meta-llama/Llama-3.1-70B-Instruct-Turbo",
        "capabilities": ["chat", "code", "fast_inference"],
    },
    "cohere": {
        "base_url":    "https://api.cohere.com/v2",
        "key_env":     "COHERE_API_KEY",
        "models":      ["command-r-plus-08-2024", "command-r-08-2024", "embed-english-v3.0"],
        "default_model": "command-r-plus-08-2024",
        "capabilities": ["chat", "rerank", "embed"],
    },
}


# ---------------------------------------------------------------------------
# Individual provider adapters
# ---------------------------------------------------------------------------

class MistralAdapter:
    def __init__(self, api_key: str):
        self.api_key  = api_key
        self.base_url = PROVIDER_CONFIGS["mistral"]["base_url"]

    async def chat(self, messages: List[Dict], model: str = None, **kwargs) -> str:
        model = model or PROVIDER_CONFIGS["mistral"]["default_model"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": model, "messages": messages, **kwargs},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


class StabilityAdapter:
    def __init__(self, api_key: str):
        self.api_key  = api_key
        self.base_url = PROVIDER_CONFIGS["stability"]["base_url"]

    async def generate_image(self, prompt: str, model: str = None, width: int = 1024,
                              height: int = 1024, output_format: str = "png") -> bytes:
        model = model or "sd3-large"
        endpoint = f"{self.base_url}/v2beta/stable-image/generate/{model.replace('stable-image/generate/', '')}"
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {self.api_key}", "Accept": "image/*"},
                data={"prompt": prompt, "output_format": output_format},
            )
            resp.raise_for_status()
            return resp.content

    async def edit_image(self, image_bytes: bytes, prompt: str, mode: str = "inpaint") -> bytes:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/v2beta/stable-image/edit/{mode}",
                headers={"Authorization": f"Bearer {self.api_key}", "Accept": "image/*"},
                files={"image": ("image.png", image_bytes, "image/png")},
                data={"prompt": prompt, "output_format": "png"},
            )
            resp.raise_for_status()
            return resp.content


class ElevenLabsAdapter:
    def __init__(self, api_key: str):
        self.api_key  = api_key
        self.base_url = PROVIDER_CONFIGS["elevenlabs"]["base_url"]

    async def text_to_speech(self, text: str, voice_id: str = None,
                              model: str = None, output_format: str = "mp3_44100_128") -> bytes:
        voice_id = voice_id or PROVIDER_CONFIGS["elevenlabs"]["voices"]["alex_riviera"]
        model    = model    or PROVIDER_CONFIGS["elevenlabs"]["default_model"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key":   self.api_key,
                    "Content-Type": "application/json",
                    "Accept":       "audio/mpeg",
                },
                json={
                    "text":            text,
                    "model_id":        model,
                    "output_format":   output_format,
                    "voice_settings":  {"stability": 0.5, "similarity_boost": 0.75},
                },
            )
            resp.raise_for_status()
            return resp.content

    async def list_voices(self) -> List[Dict]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{self.base_url}/voices",
                headers={"xi-api-key": self.api_key},
            )
            resp.raise_for_status()
            return resp.json().get("voices", [])


class RunwayAdapter:
    def __init__(self, api_key: str):
        self.api_key  = api_key
        self.base_url = PROVIDER_CONFIGS["runway"]["base_url"]

    async def generate_video(self, prompt: str, model: str = None,
                              duration: int = 5, ratio: str = "1280:720") -> Dict:
        model = model or PROVIDER_CONFIGS["runway"]["default_model"]
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/image_to_video",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                    "X-Runway-Version": "2024-11-06",
                },
                json={
                    "model":      model,
                    "promptText": prompt,
                    "duration":   duration,
                    "ratio":      ratio,
                },
            )
            resp.raise_for_status()
            task = resp.json()
            return {"task_id": task.get("id"), "status": "submitted", "model": model}

    async def poll_video(self, task_id: str) -> Dict:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{self.base_url}/tasks/{task_id}",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "X-Runway-Version": "2024-11-06"},
            )
            resp.raise_for_status()
            return resp.json()


class ReplicateAdapter:
    def __init__(self, api_key: str):
        self.api_key  = api_key
        self.base_url = PROVIDER_CONFIGS["replicate"]["base_url"]

    async def run(self, model: str, input_data: Dict, wait: bool = True) -> Any:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Create prediction
            resp = await client.post(
                f"{self.base_url}/predictions",
                headers={"Authorization": f"Token {self.api_key}"},
                json={"version": model, "input": input_data},
            )
            resp.raise_for_status()
            pred = resp.json()

            if not wait:
                return pred

            # Poll until complete
            pred_id = pred["id"]
            for _ in range(60):
                await asyncio.sleep(3)
                poll = await client.get(
                    f"{self.base_url}/predictions/{pred_id}",
                    headers={"Authorization": f"Token {self.api_key}"},
                )
                poll.raise_for_status()
                data = poll.json()
                if data["status"] in ("succeeded", "failed", "canceled"):
                    return data
            return pred


class PineconeAdapter:
    def __init__(self, api_key: str, index_name: str = None):
        self.api_key    = api_key
        self.index_name = index_name or PROVIDER_CONFIGS["pinecone"]["index_name"]
        self.host: Optional[str] = None   # set after describe_index()

    async def _ensure_host(self):
        if self.host:
            return
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"https://api.pinecone.io/indexes/{self.index_name}",
                headers={"Api-Key": self.api_key},
            )
            if resp.status_code == 200:
                self.host = resp.json().get("host", "")

    async def upsert(self, vectors: List[Dict]) -> Dict:
        """vectors: [{"id": "...", "values": [...], "metadata": {...}}]"""
        await self._ensure_host()
        base = f"https://{self.host}" if self.host else "https://api.pinecone.io"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base}/vectors/upsert",
                headers={"Api-Key": self.api_key, "Content-Type": "application/json"},
                json={"vectors": vectors},
            )
            resp.raise_for_status()
            return resp.json()

    async def query(self, vector: List[float], top_k: int = 10, filter: Dict = None) -> Dict:
        await self._ensure_host()
        base = f"https://{self.host}" if self.host else "https://api.pinecone.io"
        body: Dict = {"vector": vector, "topK": top_k, "includeMetadata": True}
        if filter:
            body["filter"] = filter
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{base}/query",
                headers={"Api-Key": self.api_key, "Content-Type": "application/json"},
                json=body,
            )
            resp.raise_for_status()
            return resp.json()


class TogetherAIAdapter:
    def __init__(self, api_key: str):
        self.api_key  = api_key
        self.base_url = PROVIDER_CONFIGS["together"]["base_url"]

    async def chat(self, messages: List[Dict], model: str = None, **kwargs) -> str:
        model = model or PROVIDER_CONFIGS["together"]["default_model"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": model, "messages": messages, **kwargs},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


class CohereAdapter:
    def __init__(self, api_key: str):
        self.api_key  = api_key
        self.base_url = PROVIDER_CONFIGS["cohere"]["base_url"]

    async def chat(self, messages: List[Dict], model: str = None, **kwargs) -> str:
        model = model or PROVIDER_CONFIGS["cohere"]["default_model"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                },
                json={"model": model, "messages": messages, **kwargs},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", [{}])[0].get("text", "")

    async def rerank(self, query: str, documents: List[str], model: str = "rerank-english-v3.0",
                     top_n: int = 5) -> List[Dict]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.cohere.com/v1/rerank",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": model, "query": query, "documents": documents, "top_n": top_n},
            )
            resp.raise_for_status()
            return resp.json().get("results", [])


# ---------------------------------------------------------------------------
# Main hub
# ---------------------------------------------------------------------------
class ExtendedAIIntegrationHub:
    """
    Drop-in extension layer on top of the existing AIIntegrationHub.
    Adds 8 new providers while preserving all existing functionality.
    """

    def __init__(
        self,
        data_path: str = "data/",
        base_hub=None,       # pass the existing AIIntegrationHub instance
        config: Optional[Dict] = None,
    ):
        self.data_path = Path(data_path)
        self.base_hub  = base_hub
        self.config    = config or {}
        self._adapters: Dict[str, Any] = {}
        self._init_adapters()
        logger.info("ExtendedAIIntegrationHub initialised — %d new providers", len(self._adapters))

    def _init_adapters(self):
        """Initialise adapters for all configured providers."""
        for provider, cfg in PROVIDER_CONFIGS.items():
            key = os.environ.get(cfg["key_env"], "")
            if not key:
                logger.debug("ExtendedHub: no key for %s — skipping", provider)
                continue
            try:
                if provider == "mistral":
                    self._adapters[provider] = MistralAdapter(key)
                elif provider == "stability":
                    self._adapters[provider] = StabilityAdapter(key)
                elif provider == "elevenlabs":
                    self._adapters[provider] = ElevenLabsAdapter(key)
                elif provider == "runway":
                    self._adapters[provider] = RunwayAdapter(key)
                elif provider == "replicate":
                    self._adapters[provider] = ReplicateAdapter(key)
                elif provider == "pinecone":
                    self._adapters[provider] = PineconeAdapter(key)
                elif provider == "together":
                    self._adapters[provider] = TogetherAIAdapter(key)
                elif provider == "cohere":
                    self._adapters[provider] = CohereAdapter(key)
                logger.info("ExtendedHub: %s initialised", provider)
            except Exception as e:
                logger.warning("ExtendedHub: failed to init %s — %s", provider, e)

    # ── Universal chat ────────────────────────────────────────────────────

    async def chat(self, prompt: str, provider: str = "auto", model: str = None,
                   system: str = None, **kwargs) -> str:
        """
        Unified chat interface.  provider="auto" picks the best available.
        Falls back to base_hub if provider not in extended set.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if provider == "auto":
            provider = self._pick_chat_provider()

        if provider in self._adapters:
            adapter = self._adapters[provider]
            if hasattr(adapter, "chat"):
                return await adapter.chat(messages, model=model, **kwargs)

        # Fall back to existing hub
        if self.base_hub and hasattr(self.base_hub, "chat"):
            return await self.base_hub.chat(prompt)

        return f"[ExtendedHub] No provider available for chat. Configured: {list(self._adapters.keys())}"

    # ── Image generation ──────────────────────────────────────────────────

    async def generate_image(self, prompt: str, provider: str = "auto",
                              save_path: str = None, **kwargs) -> Any:
        """Returns image bytes.  Optionally saves to save_path."""
        if provider == "auto":
            provider = self._pick_image_provider()

        image_bytes = None
        if provider == "stability" and "stability" in self._adapters:
            image_bytes = await self._adapters["stability"].generate_image(prompt, **kwargs)
        elif provider == "replicate" and "replicate" in self._adapters:
            model = kwargs.pop("model", PROVIDER_CONFIGS["replicate"]["default_model"])
            result = await self._adapters["replicate"].run(model, {"prompt": prompt, **kwargs})
            image_url = result.get("output", [None])[0] if isinstance(result.get("output"), list) else None
            if image_url:
                async with httpx.AsyncClient() as client:
                    image_bytes = (await client.get(image_url)).content

        if image_bytes and save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(image_bytes)

        return image_bytes

    # ── Text-to-speech ────────────────────────────────────────────────────

    async def text_to_speech(self, text: str, provider: str = "elevenlabs",
                              voice_id: str = None, save_path: str = None, **kwargs) -> Optional[bytes]:
        if provider == "elevenlabs" and "elevenlabs" in self._adapters:
            audio = await self._adapters["elevenlabs"].text_to_speech(text, voice_id=voice_id, **kwargs)
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(audio)
            return audio
        logger.warning("text_to_speech: provider %s not available", provider)
        return None

    # ── Video generation ──────────────────────────────────────────────────

    async def generate_video(self, prompt: str, provider: str = "runway", **kwargs) -> Dict:
        if provider == "runway" and "runway" in self._adapters:
            return await self._adapters["runway"].generate_video(prompt, **kwargs)
        if provider == "replicate" and "replicate" in self._adapters:
            model = kwargs.pop("model", "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351")
            result = await self._adapters["replicate"].run(model, {"prompt": prompt, **kwargs})
            return {"output": result.get("output"), "status": result.get("status")}
        return {"error": f"Video provider {provider} not available"}

    # ── Vector DB ─────────────────────────────────────────────────────────

    async def vector_upsert(self, vectors: List[Dict]) -> Dict:
        if "pinecone" in self._adapters:
            return await self._adapters["pinecone"].upsert(vectors)
        return {"error": "Pinecone not configured"}

    async def vector_query(self, vector: List[float], top_k: int = 10) -> Dict:
        if "pinecone" in self._adapters:
            return await self._adapters["pinecone"].query(vector, top_k)
        return {"error": "Pinecone not configured"}

    # ── Reranking ─────────────────────────────────────────────────────────

    async def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict]:
        if "cohere" in self._adapters:
            return await self._adapters["cohere"].rerank(query, documents, top_n=top_n)
        return []

    # ── Status ────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        base_providers = []
        if self.base_hub and hasattr(self.base_hub, "get_status"):
            try:
                base_status = self.base_hub.get_status()
                base_providers = base_status.get("providers", [])
            except Exception:
                pass

        return {
            "component":        "ExtendedAIIntegrationHub",
            "version":          "1.0.0",
            "extended_providers": list(self._adapters.keys()),
            "base_providers":   base_providers,
            "total_providers":  len(self._adapters) + len(base_providers),
            "capabilities": {
                "chat":              self._pick_chat_provider(),
                "image_generation":  self._pick_image_provider(),
                "tts":               "elevenlabs" if "elevenlabs" in self._adapters else None,
                "video":             "runway"     if "runway"     in self._adapters else None,
                "vector_db":         "pinecone"   if "pinecone"   in self._adapters else None,
                "reranking":         "cohere"     if "cohere"     in self._adapters else None,
            },
        }

    # ── Provider selection helpers ────────────────────────────────────────

    def _pick_chat_provider(self) -> str:
        preferred = ["mistral", "together", "cohere"]
        for p in preferred:
            if p in self._adapters:
                return p
        if self.base_hub:
            return "base_hub"
        return "none"

    def _pick_image_provider(self) -> str:
        preferred = ["stability", "replicate"]
        for p in preferred:
            if p in self._adapters:
                return p
        return "none"


# ---------------------------------------------------------------------------
# Flask integration helper
# ---------------------------------------------------------------------------
def register_extended_hub_routes(app, hub: ExtendedAIIntegrationHub):
    import asyncio
    from flask import jsonify, request

    @app.route("/api/extended_hub/status")
    def extended_hub_status():
        return jsonify(hub.get_status())

    @app.route("/api/extended_hub/chat", methods=["POST"])
    def extended_hub_chat():
        data     = request.get_json(silent=True) or {}
        prompt   = data.get("prompt", data.get("message", ""))
        provider = data.get("provider", "auto")
        model    = data.get("model")
        system   = data.get("system")
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(hub.chat(prompt, provider=provider, model=model, system=system))
        loop.close()
        return jsonify({"response": result, "provider": provider})

    @app.route("/api/extended_hub/tts", methods=["POST"])
    def extended_hub_tts():
        data     = request.get_json(silent=True) or {}
        text     = data.get("text", "")
        voice_id = data.get("voice_id")
        loop = asyncio.new_event_loop()
        audio = loop.run_until_complete(hub.text_to_speech(text, voice_id=voice_id))
        loop.close()
        if audio:
            from flask import Response
            return Response(audio, mimetype="audio/mpeg")
        return jsonify({"error": "TTS failed"}), 500

    @app.route("/api/extended_hub/image", methods=["POST"])
    def extended_hub_image():
        data     = request.get_json(silent=True) or {}
        prompt   = data.get("prompt", "")
        provider = data.get("provider", "auto")
        loop = asyncio.new_event_loop()
        image_bytes = loop.run_until_complete(hub.generate_image(prompt, provider=provider))
        loop.close()
        if image_bytes:
            from flask import Response
            return Response(image_bytes, mimetype="image/png")
        return jsonify({"error": "Image generation failed"}), 500

    @app.route("/api/extended_hub/video", methods=["POST"])
    def extended_hub_video():
        data     = request.get_json(silent=True) or {}
        prompt   = data.get("prompt", "")
        provider = data.get("provider", "runway")
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(hub.generate_video(prompt, provider=provider))
        loop.close()
        return jsonify(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hub = ExtendedAIIntegrationHub(data_path="/tmp/dmai_hub_test/")
    print(json.dumps(hub.get_status(), indent=2))
