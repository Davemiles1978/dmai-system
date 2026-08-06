"""
DMAI ProceduralArtist — Pure Python procedural image generation.
No external dependencies beyond PIL (already available).
Teaches DMAI: noise functions, L-systems, color harmony, composition.

This replaces the geometric primitives in SelfContainedGenerator with
production-quality procedural techniques.
"""

import math
import random
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Tuple, Dict, Optional


class PerlinNoise:
    """Pure Python Perlin noise — no numpy needed."""
    
    def __init__(self, seed: int = 0):
        self.perm = list(range(256))
        random.seed(seed)
        random.shuffle(self.perm)
        self.perm += self.perm  # Double for wrapping
    
    def _fade(self, t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + t * (b - a)
    
    def _grad(self, hash: int, x: float, y: float) -> float:
        h = hash & 3
        u = x if h < 2 else y
        v = y if h < 2 else x
        return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)
    
    def noise2d(self, x: float, y: float) -> float:
        """Returns value between -1 and 1."""
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        x -= math.floor(x)
        y -= math.floor(y)
        u = self._fade(x)
        v = self._fade(y)
        aa = self.perm[self.perm[X] + Y]
        ab = self.perm[self.perm[X] + Y + 1]
        ba = self.perm[self.perm[X + 1] + Y]
        bb = self.perm[self.perm[X + 1] + Y + 1]
        return self._lerp(
            self._lerp(self._grad(aa, x, y), self._grad(ba, x - 1, y), u),
            self._lerp(self._grad(ab, x, y - 1), self._grad(bb, x - 1, y - 1), u),
            v
        )
    
    def octave_noise(self, x: float, y: float, octaves: int = 4, persistence: float = 0.5) -> float:
        """Fractal Brownian Motion — richer, more natural noise."""
        total = 0.0
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0
        for _ in range(octaves):
            total += self.noise2d(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2
        return total / max_value


class ColorPalette:
    """Color harmony engine — understands complementary, analogous, triadic."""
    
    @staticmethod
    def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
        """Hue 0-360, Saturation 0-1, Lightness 0-1 → RGB 0-255."""
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2
        if h < 60: r, g, b = c, x, 0
        elif h < 120: r, g, b = x, c, 0
        elif h < 180: r, g, b = 0, c, x
        elif h < 240: r, g, b = 0, x, c
        elif h < 300: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        return (
            int((r + m) * 255),
            int((g + m) * 255),
            int((b + m) * 255)
        )
    
    @classmethod
    def complementary(cls, base_hue: float) -> List[Tuple[int, int, int]]:
        """Two colors opposite on the wheel."""
        return [
            cls.hsl_to_rgb(base_hue, 0.7, 0.5),
            cls.hsl_to_rgb((base_hue + 180) % 360, 0.7, 0.5)
        ]
    
    @classmethod
    def analogous(cls, base_hue: float, count: int = 5) -> List[Tuple[int, int, int]]:
        """Adjacent colors on the wheel."""
        return [cls.hsl_to_rgb((base_hue + i * 30) % 360, 0.6 + i * 0.05, 0.4 + i * 0.1) for i in range(count)]
    
    @classmethod
    def triadic(cls, base_hue: float) -> List[Tuple[int, int, int]]:
        """Three evenly spaced colors."""
        return [cls.hsl_to_rgb((base_hue + i * 120) % 360, 0.7, 0.5) for i in range(3)]
    
    @classmethod
    def from_style(cls, style: str) -> List[Tuple[int, int, int]]:
        """Get a palette appropriate for a given style."""
        style_palettes = {
            "cyberpunk": [(255, 0, 128), (0, 255, 255), (128, 0, 255), (0, 0, 0)],
            "fantasy": [(50, 100, 180), (200, 150, 80), (80, 180, 80), (220, 200, 150)],
            "photorealistic": [(135, 180, 220), (80, 140, 80), (180, 160, 120), (220, 220, 220)],
            "anime": [(255, 180, 200), (180, 220, 255), (255, 255, 200), (150, 255, 150)],
            "cartoon": [(255, 80, 80), (80, 80, 255), (255, 255, 80), (80, 255, 80)],
            "horror": [(40, 10, 10), (80, 20, 20), (20, 20, 20), (120, 30, 30)],
        }
        return style_palettes.get(style, cls.analogous(random.randint(0, 360)))


class LSystem:
    """Lindenmayer System for generating plant-like structures."""
    
    def __init__(self, axiom: str, rules: Dict[str, str], angle: float = 25):
        self.axiom = axiom
        self.rules = rules
        self.angle = angle
    
    def generate(self, iterations: int = 4) -> str:
        result = self.axiom
        for _ in range(iterations):
            result = "".join(self.rules.get(c, c) for c in result)
        return result
    
    def draw(self, draw: ImageDraw.ImageDraw, start_x: float, start_y: float, 
             length: float = 10, angle: float = -90, width: int = 800, height: int = 600):
        """Draw the L-system on an ImageDraw context."""
        stack = []
        x, y = start_x, start_y
        instructions = self.generate()
        
        for cmd in instructions:
            if cmd == 'F':
                rad = math.radians(angle)
                new_x = x + length * math.cos(rad)
                new_y = y + length * math.sin(rad)
                draw.line([(x, y), (new_x, new_y)], fill=(60, 140, 60), width=2)
                x, y = new_x, new_y
            elif cmd == '+':
                angle += self.angle + random.uniform(-5, 5)
            elif cmd == '-':
                angle -= self.angle + random.uniform(-5, 5)
            elif cmd == '[':
                stack.append((x, y, angle, length))
            elif cmd == ']':
                if stack:
                    x, y, angle, length = stack.pop()
                    length *= 0.7
    
    @classmethod
    def tree(cls) -> "LSystem":
        """A simple fractal tree."""
        return cls(
            axiom="F",
            rules={"F": "F[+F]F[-F]F"},
            angle=25
        )
    
    @classmethod
    def plant(cls) -> "LSystem":
        """A bushier plant."""
        return cls(
            axiom="X",
            rules={"X": "F[+X]F[-X]+X", "F": "FF"},
            angle=20
        )


class ProceduralArtist:
    """
    Composes scenes using noise, shapes, L-systems, and color theory.
    Much better than random geometric primitives.
    """
    
    def __init__(self, width: int = 512, height: int = 512, seed: int = None):
        self.width = width
        self.height = height
        self.seed = seed or random.randint(0, 2**31)
        random.seed(self.seed)
        self.noise = PerlinNoise(seed=self.seed)
    
    def sky(self, draw: ImageDraw.ImageDraw, palette: List[Tuple[int, int, int]]):
        """Draw a natural-looking sky with clouds using Perlin noise."""
        sky_color = palette[0]
        cloud_color = palette[1] if len(palette) > 1 else (255, 255, 255)
        
        for y in range(self.height):
            ratio = y / self.height
            # Darken toward top, lighten toward horizon
            r = int(sky_color[0] * (1 - ratio * 0.4))
            g = int(sky_color[1] * (1 - ratio * 0.3))
            b = int(sky_color[2] * (1 - ratio * 0.2))
            draw.line([(0, y), (self.width, y)], fill=(r, g, b))
        
        # Add clouds using Perlin noise
        for y in range(0, self.height // 2, 3):
            for x in range(0, self.width, 3):
                n = self.noise.octave_noise(x / 100, y / 100, octaves=4)
                if n > 0.3:
                    alpha = min(1.0, (n - 0.3) * 3)
                    r = int(sky_color[0] + (cloud_color[0] - sky_color[0]) * alpha)
                    g = int(sky_color[1] + (cloud_color[1] - sky_color[1]) * alpha)
                    b = int(sky_color[2] + (cloud_color[2] - sky_color[2]) * alpha)
                    draw.point((x, y), fill=(r, g, b))
    
    def terrain(self, draw: ImageDraw.ImageDraw, palette: List[Tuple[int, int, int]], 
                horizon: float = 0.6):
        """Draw procedural terrain using fractal noise."""
        base_color = palette[0]
        highlight = palette[1] if len(palette) > 1 else (
            min(255, base_color[0] + 40),
            min(255, base_color[1] + 40),
            min(255, base_color[2] + 40)
        )
        horizon_y = int(self.height * horizon)
        
        for x in range(self.width):
            # Fractal noise for terrain height
            h = self.noise.octave_noise(x / 80, 0, octaves=6, persistence=0.55)
            ground_y = horizon_y + int(h * self.height * 0.3)
            
            for y in range(horizon_y, self.height):
                if y >= ground_y:
                    # Ground color with slight variation
                    n = self.noise.octave_noise(x / 40, y / 40, octaves=3)
                    shade = int(20 * n)
                    r = max(0, min(255, base_color[0] + shade))
                    g = max(0, min(255, base_color[1] + shade))
                    b = max(0, min(255, base_color[2] + shade))
                    draw.point((x, y), fill=(r, g, b))
                else:
                    # Sky gap fill
                    pass
    
    def mountains(self, draw: ImageDraw.ImageDraw, palette: List[Tuple[int, int, int]],
                  base_y: int = 400):
        """Draw procedural mountain silhouettes."""
        color = palette[0]
        
        for x in range(0, self.width, 2):
            h = abs(self.noise.octave_noise(x / 60, 0, octaves=5, persistence=0.5))
            peak_y = base_y - int(h * 250)
            
            # Mountain body
            for y in range(peak_y, self.height):
                alpha = (y - peak_y) / (self.height - peak_y)
                r = int(color[0] * (1 - alpha * 0.3))
                g = int(color[1] * (1 - alpha * 0.3))
                b = int(color[2] * (1 - alpha * 0.3))
                draw.point((x, y), fill=(r, g, b))
    
    def stars(self, draw: ImageDraw.ImageDraw, count: int = 100):
        """Add stars to a night sky."""
        for _ in range(count):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height // 2)
            brightness = random.randint(150, 255)
            size = random.randint(1, 2)
            color = (brightness, brightness, brightness)
            if size == 1:
                draw.point((x, y), fill=color)
            else:
                draw.ellipse([x-1, y-1, x+1, y+1], fill=color)
    
    def sun_moon(self, draw: ImageDraw.ImageDraw, x: int, y: int, 
                 radius: int, color: Tuple[int, int, int]):
        """Draw a glowing sun or moon."""
        # Outer glow
        for r in range(radius + 20, radius, -2):
            alpha = (r - radius) / 20
            glow_color = (
                int(color[0] * alpha),
                int(color[1] * alpha),
                int(color[2] * alpha)
            )
            draw.ellipse([x-r, y-r, x+r, y+r], fill=glow_color)
        # Core
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    
    def tree(self, draw: ImageDraw.ImageDraw, x: int, y: int, size: float = 1.0):
        """Draw an L-system tree."""
        lsystem = LSystem.tree()
        lsystem.draw(draw, x, y, length=15 * size, width=self.width, height=self.height)
    
    def compose_scene(self, prompt: str = "", style: str = "default") -> Image.Image:
        """Compose a complete scene based on prompt and style."""
        img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        palette = ColorPalette.from_style(style)
        
        # Parse prompt for scene type
        pl = prompt.lower()
        
        if "night" in pl or "space" in pl or "stars" in pl:
            # Night scene
            self.sky(draw, [(10, 10, 40), (30, 30, 80)])
            self.stars(draw, count=150)
            if "moon" in pl:
                self.sun_moon(draw, self.width - 100, 80, 40, (240, 240, 220))
        elif "sunset" in pl or "sunrise" in pl:
            # Sunset
            self.sky(draw, [(255, 120, 40), (255, 60, 20)])
            self.sun_moon(draw, self.width // 2, self.height // 3, 50, (255, 200, 50))
            self.mountains(draw, [(80, 30, 40)], base_y=int(self.height * 0.55))
        elif "mountain" in pl or "landscape" in pl:
            self.sky(draw, [(100, 160, 220), (180, 210, 240)])
            self.mountains(draw, [(60, 80, 100)], base_y=int(self.height * 0.5))
            self.terrain(draw, [(40, 100, 40), (80, 160, 60)], horizon=0.55)
        elif "forest" in pl or "tree" in pl or "nature" in pl:
            self.sky(draw, [(80, 140, 200), (160, 200, 230)])
            self.terrain(draw, [(34, 100, 34), (60, 150, 50)], horizon=0.5)
            for i in range(5):
                self.tree(draw, 100 + i * 90, int(self.height * 0.5) + random.randint(-20, 20), 
                         size=0.8 + random.random() * 0.8)
        elif "cyberpunk" in pl or "city" in pl:
            # Cyberpunk city silhouette
            self.sky(draw, [(20, 0, 40), (60, 0, 80)])
            self.stars(draw, count=50)
            # Buildings
            for i in range(8):
                bx = i * 70
                bh = random.randint(100, 300)
                by = self.height - bh
                color = palette[i % len(palette)]
                draw.rectangle([bx, by, bx + 50, self.height], fill=color)
                # Windows
                for wy in range(by + 10, self.height - 10, 15):
                    for wx in range(bx + 5, bx + 45, 10):
                        if random.random() > 0.3:
                            draw.point((wx, wy), fill=(255, 255, 100) if random.random() > 0.3 else (0, 255, 255))
        else:
            # Default: pleasant landscape
            self.sky(draw, palette[:2])
            self.terrain(draw, palette[2:4] if len(palette) > 3 else palette[:2], horizon=0.6)
            if random.random() > 0.5:
                self.tree(draw, self.width // 2, int(self.height * 0.55), size=1.2)
        
        # Polish
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        return img


# ── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import io, base64, os
    from datetime import datetime, timezone
    
    print("DMAI ProceduralArtist — Self Test")
    print("=" * 50)
    
    artist = ProceduralArtist(width=512, height=512, seed=42)
    
    test_scenes = [
        ("sunset over mountains", "default"),
        ("cyberpunk city at night", "cyberpunk"),
        ("forest with trees", "default"),
        ("starry night with moon", "fantasy"),
    ]
    
    os.makedirs("data/generated_content", exist_ok=True)
    
    for prompt, style in test_scenes:
        img = artist.compose_scene(prompt=prompt, style=style)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"dma_artist_{style}_{ts}.png"
        filepath = f"data/generated_content/{filename}"
        img.save(filepath, "PNG")
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        print(f"  {style:15s}: {prompt[:40]:40s} → {filename} ({len(b64)//1024} KB base64)")
    
    print("=" * 50)
    print("All scenes generated. DMAI can now compose scenes, not just draw shapes.")
