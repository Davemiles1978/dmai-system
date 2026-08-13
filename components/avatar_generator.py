"""Avatar generation system for Alex Riviera.

Uses reference images to maintain visual consistency when generating new content.
Two personas:
  - "alex_public"  — Alex Riviera, confidence coach, educator (SFW)
  - "alexa_private" — Alexa Rivers (NSFW)

Reference images are stored in data/avatars/reference_images/<persona>/
"""
import os, json, logging, base64, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List

logger = logging.getLogger("dmai.avatar_generator")

REFERENCE_DIR = Path("data/avatars/reference_images")
GENERATED_DIR = Path("data/avatars/generated")


class AvatarGenerator:
    """Generates consistent Alex/Alexa avatar images using reference images."""

    def __init__(self):
        # Resolve paths for both local and Render environments
        candidates = [
            REFERENCE_DIR,
            Path("data") / REFERENCE_DIR,
            Path("/opt/render/project/src") / REFERENCE_DIR,
        ]
        self.ref_dir = next((p for p in candidates if p.exists()), REFERENCE_DIR)
        gen_candidates = [
            GENERATED_DIR,
            Path("data") / GENERATED_DIR,
            Path("/opt/render/project/src") / GENERATED_DIR,
        ]
        self.gen_dir = next((p for p in gen_candidates if p.exists()), GENERATED_DIR)
        self.gen_dir.mkdir(parents=True, exist_ok=True)
        self._load_persona_definitions()

    def _load_persona_definitions(self):
        """Load persona prompts from existing avatar JSON files."""
        self.personas = {
            "alex_public": {
                "name": "Alex Riviera",
                "age": 28,
                "description": "American female confidence coach and educator",
                "hair": "platinum blonde hair in loose waves",
                "eyes": "ice blue",
                "build": "athletic, toned",
                "style": "professional, modern, approachable",
                "sfw": True,
                "ref_folder": "alex_public",
            },
            "alexa_private": {
                "name": "Alexa Rivers",
                "age": 28,
                "description": "American female confidence coach",
                "hair": "platinum blonde hair in loose waves",
                "eyes": "ice blue",
                "build": "athletic, toned",
                "style": "sensual, artistic, empowering",
                "sfw": False,
                "ref_folder": "alexa_private",
            },
        }
        # Try to load from existing JSON files for more detail
        try:
            with open("data/avatars/alex_riviera_definitive_identity.json") as f:
                identity = json.load(f)
                if identity:
                    self.personas["alex_public"].update(identity)
        except Exception:
            pass

    def get_reference_images(self, persona: str) -> List[Path]:
        """Get list of reference images for a persona."""
        ref_folder = self.ref_dir / self.personas.get(persona, {}).get("ref_folder", persona)
        # Try multiple path resolutions for Render compatibility
        candidates = [
            ref_folder,
            Path("data") / ref_folder,
            Path("/opt/render/project/src") / ref_folder,
        ]
        for folder in candidates:
            if folder.exists():
                images = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
                if images:
                    return images
        return []

    def generate_image(self, persona: str, prompt: str, output_name: str = None) -> Optional[Dict]:
        """Generate a new avatar image using reference images for consistency.

        Uses the available image generation API (Google Gemini image or Groq).
        Returns dict with image path and metadata on success.
        """
        persona_info = self.personas.get(persona)
        if not persona_info:
            logger.error(f"Unknown persona: {persona}")
            return None

        ref_images = self.get_reference_images(persona)
        if not ref_images:
            logger.warning(f"No reference images found for {persona}")

        # Build the full generation prompt with persona consistency
        base_prompt = (
            f"{persona_info['name']}, {persona_info['age']}-year-old "
            f"{persona_info.get('description', 'American female')}, "
            f"{persona_info.get('hair', 'platinum blonde hair')}, "
            f"{persona_info.get('eyes', 'ice blue eyes')}. "
            f"{prompt}"
        )

        # Try Gemini image generation first
        api_key = os.environ.get("GOOGLE_AI_STUDIO_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                image_data = self._generate_with_gemini(base_prompt, api_key, ref_images)
                if image_data:
                    output_path = self._save_image(image_data, output_name or f"{persona}_{int(time.time())}")
                    return {
                        "path": str(output_path),
                        "persona": persona,
                        "prompt": base_prompt,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "provider": "gemini",
                    }
            except Exception as e:
                logger.warning(f"Gemini image generation failed: {e}")

        # Try OpenAI image generation
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                image_data = self._generate_with_openai(base_prompt, openai_key)
                if image_data:
                    output_path = self._save_image(image_data, output_name or f"{persona}_{int(time.time())}")
                    return {
                        "path": str(output_path),
                        "persona": persona,
                        "prompt": base_prompt,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "provider": "openai",
                    }
            except Exception as e:
                logger.warning(f"OpenAI image generation failed: {e}")

        logger.error("No image generation API available")
        return None

    def _generate_with_gemini(self, prompt: str, api_key: str, ref_images: List[Path]) -> Optional[bytes]:
        """Generate image using Google Gemini 2.5 Flash Image."""
        import requests

        # Include first reference image as inline data for consistency
        parts = [{"text": prompt}]
        if ref_images:
            try:
                with open(ref_images[0], "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode()
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png" if ref_images[0].suffix == ".png" else "image/jpeg",
                            "data": img_base64,
                        }
                    })
            except Exception:
                pass

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key={api_key}",
            json=payload,
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
                if "inline_data" in part:
                    return base64.b64decode(part["inline_data"]["data"])
        return None

    def _generate_with_openai(self, prompt: str, api_key: str) -> Optional[bytes]:
        """Generate image using OpenAI DALL-E."""
        import requests
        resp = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "gpt-image-1", "prompt": prompt, "size": "1024x1024", "n": 1},
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            image_url = data["data"][0].get("url") or data["data"][0].get("b64_json")
            if image_url.startswith("http"):
                img_resp = requests.get(image_url, timeout=30)
                return img_resp.content
            else:
                return base64.b64decode(image_url)
        return None

    def _save_image(self, image_data: bytes, name: str) -> Path:
        """Save generated image to disk."""
        output_path = self.gen_dir / f"{name}.png"
        output_path.write_bytes(image_data)
        logger.info(f"Saved avatar image: {output_path}")
        return output_path

    def list_reference_images(self) -> Dict:
        """Return list of reference images for both personas."""
        result = {}
        for persona in self.personas:
            refs = self.get_reference_images(persona)
            result[persona] = [str(r.relative_to(REFERENCE_DIR)) for r in refs]
        return result


def generate_avatar(persona: str, prompt: str, output_name: str = None) -> Optional[Dict]:
    """Standalone function for generating avatar images."""
    gen = AvatarGenerator()
    return gen.generate_image(persona, prompt, output_name)
