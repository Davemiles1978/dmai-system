"""
DMAI Self-Contained Content Generator
======================================
Generates images, video frames, and audio visualizations without external APIs.
Uses Python PIL/Pillow for image generation — no OpenAI, no Replicate, no cloud.

This is DMAI's own generator. She studied the techniques, now she builds.
Future: integrate learnings from Qwen2-Audio, Qwen-Image, MiniMax-M1 repos.

Capabilities:
  - Image generation (geometric, gradient, text-to-color, pattern)
  - Placeholder replacement with real generation
  - Video frame generation (sequence of PIL images)
  - Audio visualization (waveform images from audio data)
  - Inline-base64 output for chat display
"""

from __future__ import annotations

import io
import logging
import math
import random
from base64 import b64encode
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("dmai.content_generator")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUT_DIR = _REPO_ROOT / "data" / "generated_content"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Try to import PIL — available on Render and most Python environments
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL/Pillow not available — image generation disabled")


class SelfContainedGenerator:
    """
    DMAI's own content generator. No external APIs.
    Generates images, video frames, and audio visualizations.
    """

    # Color palettes for different styles
    PALETTES = {
        "default":    [(108, 99, 255), (0, 212, 170), (255, 107, 107), (255, 217, 61)],
        "cyberpunk":  [(255, 0, 128), (0, 255, 255), (128, 0, 255), (0, 0, 0)],
        "nature":     [(34, 139, 34), (135, 206, 235), (139, 90, 43), (255, 255, 255)],
        "sunset":     [(255, 94, 77), (255, 183, 77), (138, 43, 226), (255, 228, 181)],
        "ocean":      [(0, 105, 148), (0, 168, 204), (135, 206, 250), (240, 248, 255)],
        "fire":       [(255, 69, 0), (255, 140, 0), (255, 215, 0), (139, 0, 0)],
        "minimal":    [(245, 245, 245), (200, 200, 200), (100, 100, 100), (30, 30, 30)],
    }

    def __init__(self, data_path: Optional[Path] = None):
        self.root = data_path or _REPO_ROOT
        self.output_dir = self.root / "data" / "generated_content"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generation_count = 0
        logger.info("SelfContainedGenerator initialised (PIL=%s)", HAS_PIL)

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def generate_image(
        self,
        prompt: str = "",
        style: str = "default",
        width: int = 512,
        height: int = 512,
    ) -> Optional[Dict]:
        """
        Generate an image from a text prompt using procedural techniques.
        Returns {image_base64, view_url, prompt, style, dimensions}.
        """
        if not HAS_PIL:
            return self._placeholder(prompt, style, width, height)

        # Try ProceduralArtist first for scene-based generation
        try:
            from components.content.procedural_artist import ProceduralArtist
            seed = hash(prompt) if prompt else random.randint(0, 2**31)
            artist = ProceduralArtist(width=width, height=height, seed=seed)
            img = artist.compose_scene(prompt=prompt, style=style)
            logger.debug("SelfContainedGenerator: used ProceduralArtist for '%s'", prompt[:50])
        except ImportError:
            # Fallback to geometric primitives
            palette = self.PALETTES.get(style, self.PALETTES["default"])
            seed = hash(prompt) if prompt else random.randint(0, 2**31)
            random.seed(seed)

            img = Image.new("RGB", (width, height), palette[0])
            draw = ImageDraw.Draw(img)

            # Layer 1: Gradient background
            self._draw_gradient(draw, width, height, palette[0], palette[1])

            # Layer 2: Geometric shapes based on prompt
            num_shapes = 5 + (hash(prompt) % 15) if prompt else 8
            for i in range(num_shapes):
                self._draw_random_shape(draw, width, height, palette, i)

            # Layer 3: Texture overlay
            self._add_noise_texture(img, width, height, intensity=15)

            # Layer 4: Soft blur for polish
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

            random.seed()  # Reset seed

        # Save
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"dma_gen_{timestamp}_{self.generation_count}.png"
        filepath = self.output_dir / filename
        img.save(str(filepath), "PNG")

        # Base64 for inline display
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64_data = b64encode(buffer.getvalue()).decode("utf-8")

        self.generation_count += 1

        return {
            "ok": True,
            "image_base64": f"data:image/png;base64,{b64_data}",
            "view_url": f"/api/content/view/{filename}",
            "file": str(filepath),
            "filename": filename,
            "prompt": prompt,
            "style": style,
            "width": width,
            "height": height,
            "generator": "DMAI SelfContainedGenerator",
        }

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _draw_gradient(self, draw, w, h, c1, c2):
        """Vertical gradient from c1 to c2."""
        for y in range(h):
            ratio = y / h
            r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
            g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
            b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
            draw.line([(0, y), (w, y)], fill=(r, g, b))

    def _draw_random_shape(self, draw, w, h, palette, index):
        """Draw a random geometric shape."""
        color = palette[index % len(palette)]
        shape_type = index % 4

        if shape_type == 0:  # Circle
            cx = random.randint(w // 4, 3 * w // 4)
            cy = random.randint(h // 4, 3 * h // 4)
            r = random.randint(20, min(w, h) // 3)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)

        elif shape_type == 1:  # Rectangle
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = random.randint(x1 + 30, w)
            y2 = random.randint(y1 + 30, h)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        elif shape_type == 2:  # Line
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            x2 = random.randint(0, w)
            y2 = random.randint(0, h)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

        else:  # Polygon/triangle
            points = [
                (random.randint(0, w), random.randint(0, h)),
                (random.randint(0, w), random.randint(0, h)),
                (random.randint(0, w), random.randint(0, h)),
            ]
            draw.polygon(points, outline=color, fill=(*color, 40))

    def _add_noise_texture(self, img, w, h, intensity=10):
        """Add subtle noise texture."""
        pixels = img.load()
        for _ in range(w * h // 20):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            r, g, b = pixels[x, y]
            noise = random.randint(-intensity, intensity)
            pixels[x, y] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise)),
            )

    # ------------------------------------------------------------------
    # Placeholder fallback
    # ------------------------------------------------------------------

    def _placeholder(self, prompt, style, width, height):
        """Return a placeholder when PIL is not available."""
        return {
            "ok": True,
            "image_base64": None,
            "view_url": None,
            "placeholder": True,
            "message": f"Image generation for '{prompt}' queued. PIL not available — install Pillow for real generation.",
            "prompt": prompt,
            "style": style,
            "width": width,
            "height": height,
        }

    # ------------------------------------------------------------------
    # Video frame generation
    # ------------------------------------------------------------------

    def generate_video_frames(
        self,
        prompt: str = "",
        style: str = "default",
        num_frames: int = 24,
        width: int = 512,
        height: int = 512,
    ) -> Dict:
        """Generate a sequence of images as video frames."""
        frames = []
        for i in range(num_frames):
            frame_prompt = f"{prompt} frame {i}"
            result = self.generate_image(frame_prompt, style, width, height)
            if result and result.get("ok"):
                frames.append({
                    "frame": i,
                    "image_base64": result.get("image_base64"),
                    "filename": result.get("filename"),
                })

        return {
            "ok": True,
            "frames": frames,
            "frame_count": len(frames),
            "prompt": prompt,
            "style": style,
            "generator": "DMAI SelfContainedGenerator",
        }

    # ------------------------------------------------------------------
    # Audio visualization
    # ------------------------------------------------------------------

    def generate_audio_visualization(
        self,
        waveform_data: Optional[List[float]] = None,
        style: str = "default",
        width: int = 800,
        height: int = 200,
    ) -> Optional[Dict]:
        """Generate a waveform visualization image."""
        if not HAS_PIL:
            return None

        if waveform_data is None:
            # Generate sample waveform
            waveform_data = [
                math.sin(i * 0.1) * math.cos(i * 0.05) * 0.8 + random.uniform(-0.1, 0.1)
                for i in range(200)
            ]

        palette = self.PALETTES.get(style, self.PALETTES["default"])
        img = Image.new("RGB", (width, height), (20, 20, 30))
        draw = ImageDraw.Draw(img)

        # Draw waveform
        mid_y = height // 2
        step_x = width / len(waveform_data)
        points = []
        for i, val in enumerate(waveform_data):
            x = int(i * step_x)
            y = int(mid_y - val * (height // 2 - 10))
            points.append((x, y))

        # Draw filled waveform
        for i in range(len(points) - 1):
            color_idx = (i * len(palette) // len(points)) % len(palette)
            draw.line([points[i], points[i + 1]], fill=palette[color_idx], width=2)

        # Save
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"dma_waveform_{timestamp}.png"
        filepath = self.output_dir / filename
        img.save(str(filepath), "PNG")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64_data = b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "ok": True,
            "image_base64": f"data:image/png;base64,{b64_data}",
            "view_url": f"/api/content/view/{filename}",
            "file": str(filepath),
            "filename": filename,
            "type": "waveform",
            "generator": "DMAI SelfContainedGenerator",
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        return {
            "generation_count": self.generation_count,
            "pil_available": HAS_PIL,
            "output_dir": str(self.output_dir),
            "styles": list(self.PALETTES.keys()),
        }
