"""Create avatar image variants from reference images using PIL.

No external API needed — uses Pillow to create:
  - Cropped versions (headshot, bust, full)
  - Different backgrounds (solid colors, gradients)
  - Text overlays (quotes, captions)
  - Collages (multi-image)
"""
import os, logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from datetime import datetime, timezone

logger = logging.getLogger("dmai.avatar_variants")

class AvatarVariantGenerator:
    """Generate Instagram/TikTok-ready image variants from reference photos."""

    def __init__(self):
        self.ref_dir = Path("data/avatars/reference_images")
        self.out_dir = Path("data/avatars/generated")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def get_ref_images(self, persona: str = "alex_public") -> list:
        """Get reference images for a persona."""
        folder = self.ref_dir / persona
        if not folder.exists():
            return []
        return sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))

    def create_cropped(self, persona: str, output_name: str, crop: str = "headshot") -> Path:
        """Create a cropped version of a reference image."""
        refs = self.get_ref_images(persona)
        if not refs:
            logger.error(f"No reference images for {persona}")
            return None

        img = Image.open(refs[0])
        w, h = img.size

        if crop == "headshot":
            # Upper third of image
            box = (0, 0, w, int(h * 0.4))
        elif crop == "bust":
            box = (0, 0, w, int(h * 0.6))
        else:  # full
            box = (0, 0, w, h)

        cropped = img.crop(box)
        # Resize to Instagram-friendly 1080x1350 (4:5)
        cropped = ImageOps.fit(cropped, (1080, 1350), Image.LANCZOS)

        output = self.out_dir / f"{output_name}_{crop}.png"
        cropped.save(output)
        logger.info(f"Created crop variant: {output}")
        return output

    def create_text_overlay(self, persona: str, output_name: str, quote: str, bg_color: str = "#1a1a2e") -> Path:
        """Create an image with a quote overlay."""
        refs = self.get_ref_images(persona)
        if not refs:
            return None

        # Use a reference image as background, darkened
        img = Image.open(refs[0])
        img = ImageOps.fit(img, (1080, 1350), Image.LANCZOS)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 120))
        img = Image.alpha_composite(img.convert("RGBA"), overlay)

        draw = ImageDraw.Draw(img)
        # Use default font (system)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        except Exception:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Draw quote text
        margin = 80
        words = quote.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if draw.textlength(test_line, font=font) < img.size[0] - 2 * margin:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        y = img.size[1] - margin - (len(lines) * 60)
        for line in lines:
            draw.text((margin, y), line, fill="white", font=font)
            y += 60

        # Add handle
        draw.text((margin, img.size[1] - 50), "@alexriviera", fill="#ff6584", font=small_font)

        output = self.out_dir / f"{output_name}_quote.png"
        img.convert("RGB").save(output)
        logger.info(f"Created quote overlay: {output}")
        return output

    def create_collage(self, persona: str, output_name: str) -> Path:
        """Create a 2x2 collage from reference images."""
        refs = self.get_ref_images(persona)
        if len(refs) < 4:
            logger.error("Need at least 4 reference images for collage")
            return None

        images = []
        for ref in refs[:4]:
            img = Image.open(ref)
            img = ImageOps.fit(img, (540, 675), Image.LANCZOS)
            images.append(img)

        collage = Image.new("RGB", (1080, 1350))
        collage.paste(images[0], (0, 0))
        collage.paste(images[1], (540, 0))
        collage.paste(images[2], (0, 675))
        collage.paste(images[3], (540, 675))

        output = self.out_dir / f"{output_name}_collage.png"
        collage.save(output)
        logger.info(f"Created collage: {output}")
        return output


def create_variant(variant_type: str, persona: str = "alex_public", output_name: str = None,
                   quote: str = None, crop: str = "headshot") -> Path:
    """Generate a variant image from reference photos."""
    gen = AvatarVariantGenerator()
    if not output_name:
        output_name = f"{persona}_{variant_type}"

    if variant_type == "crop":
        return gen.create_cropped(persona, output_name, crop)
    elif variant_type == "quote":
        return gen.create_text_overlay(persona, output_name, quote or "Confidence is a skill.")
    elif variant_type == "collage":
        return gen.create_collage(persona, output_name)
    return None
