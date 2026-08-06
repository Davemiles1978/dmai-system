"""
DMAI ImageAnalyser — Self-contained image analysis.
Compares images, measures photorealism, detects features.
Pure Python + PIL. DMAI uses this to evaluate her own generated images.
"""

import math, io, base64
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageStat


class ImageAnalyser:
    """Analyzes images for quality, similarity, and photorealism."""
    
    def __init__(self):
        self.reference_features = {}
    
    def load_reference(self, name: str, image_path: str):
        """Load a reference image for comparison."""
        img = Image.open(image_path).convert("RGB")
        self.reference_features[name] = self._extract_features(img)
        return {"loaded": name, "size": img.size}
    
    def _extract_features(self, img: Image.Image) -> Dict:
        """Extract key features from an image."""
        w, h = img.size
        stat = ImageStat.Stat(img)
        
        # Color histogram (simplified - mean and stddev per channel)
        means = tuple(int(m) for m in stat.mean)
        stddevs = tuple(int(s) for s in stat.stddev)
        
        # Brightness
        brightness = sum(means) / 3
        
        # Edge detection (simple Sobel approximation)
        pixels = img.load()
        edge_count = 0
        total = 0
        step = max(1, min(w, h) // 100)
        for y in range(1, h - 1, step):
            for x in range(1, w - 1, step):
                total += 1
                r1 = pixels[x-1, y][0]
                r2 = pixels[x+1, y][0]
                r3 = pixels[x, y-1][0]
                r4 = pixels[x, y+1][0]
                gradient = abs(r1 - r2) + abs(r3 - r4)
                if gradient > 40:
                    edge_count += 1
        
        edge_density = edge_count / max(1, total)
        
        return {
            "size": (w, h),
            "mean_rgb": means,
            "stddev_rgb": stddevs,
            "brightness": round(brightness, 1),
            "edge_density": round(edge_density, 3),
        }
    
    def compare(self, image_path: str, reference_name: str) -> Dict:
        """Compare an image to a reference. Returns similarity score 0-100."""
        ref = self.reference_features.get(reference_name)
        if not ref:
            return {"error": f"Reference '{reference_name}' not loaded"}
        
        img = Image.open(image_path).convert("RGB")
        features = self._extract_features(img)
        
        # Compare multiple dimensions
        # 1. Color similarity (mean RGB distance)
        color_dist = math.sqrt(sum(
            (features["mean_rgb"][i] - ref["mean_rgb"][i])**2
            for i in range(3)
        ))
        color_score = max(0, 100 - color_dist / 2.55)
        
        # 2. Texture similarity (edge density)
        edge_diff = abs(features["edge_density"] - ref["edge_density"])
        texture_score = max(0, 100 - edge_diff * 500)
        
        # 3. Brightness similarity
        bright_diff = abs(features["brightness"] - ref["brightness"])
        bright_score = max(0, 100 - bright_diff * 2)
        
        # Weighted total
        similarity = round(color_score * 0.4 + texture_score * 0.35 + bright_score * 0.25, 1)
        
        return {
            "similarity_pct": similarity,
            "color_score": round(color_score, 1),
            "texture_score": round(texture_score, 1),
            "brightness_score": round(bright_score, 1),
            "reference": reference_name,
            "image_features": features,
            "reference_features": ref,
        }
    
    def measure_photorealism(self, image_path: str) -> Dict:
        """Estimate how photorealistic an image is (0-100)."""
        img = Image.open(image_path).convert("RGB")
        features = self._extract_features(img)
        
        # Photorealism heuristics:
        # 1. High edge density = likely real (not flat)
        # 2. High color variance = real textures
        # 3. Not too uniform brightness
        
        edge_score = min(100, features["edge_density"] * 300)
        variance_score = min(100, sum(features["stddev_rgb"]) / 3)
        brightness_penalty = abs(features["brightness"] - 128) * 0.3
        
        photo_score = round(max(0, min(100,
            edge_score * 0.4 + variance_score * 0.5 - brightness_penalty
        )))
        
        return {
            "photorealism_pct": photo_score,
            "edge_score": round(edge_score, 1),
            "variance_score": round(variance_score, 1),
            "brightness": features["brightness"],
            "verdict": "photorealistic" if photo_score > 70 else
                       "semi-realistic" if photo_score > 40 else
                       "artificial"
        }


if __name__ == "__main__":
    analyser = ImageAnalyser()
    print("DMAI ImageAnalyser ready.")
    print("  Methods: load_reference(), compare(), measure_photorealism()")
