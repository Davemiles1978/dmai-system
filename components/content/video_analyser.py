"""
DMAI VideoAnalyser — Self-contained video analysis.
Extracts frames, detects motion, analyzes scene changes.
Pure Python + PIL. DMAI uses this to evaluate her own video generation.
"""

import math
from pathlib import Path
from typing import Dict, List
from PIL import Image, ImageStat


class VideoAnalyser:
    """Analyzes video frames for quality and motion."""
    
    def analyse_frames(self, frame_paths: List[str]) -> Dict:
        """Analyze a sequence of image frames as video."""
        if len(frame_paths) < 2:
            return {"error": "Need at least 2 frames"}
        
        frames = []
        for fp in frame_paths:
            try:
                img = Image.open(fp).convert("RGB")
                frames.append({"path": fp, "image": img, "stat": ImageStat.Stat(img)})
            except Exception as e:
                frames.append({"path": fp, "error": str(e)})
        
        if len(frames) < 2:
            return {"error": "Could not load enough frames"}
        
        # ── Motion Detection ──
        motion_per_frame = []
        scene_changes = []
        avg_brightness = []
        
        for i in range(1, len(frames)):
            prev = frames[i-1]["image"]
            curr = frames[i]["image"]
            
            # Frame difference
            diff = self._frame_difference(prev, curr)
            motion_per_frame.append(diff)
            
            # Scene change detection (significant difference)
            if diff > 30:
                scene_changes.append(i)
            
            # Brightness tracking
            s = frames[i]["stat"]
            avg_brightness.append(sum(s.mean) / 3)
        
        # ── Quality Metrics ──
        avg_motion = sum(motion_per_frame) / len(motion_per_frame)
        motion_variance = sum((m - avg_motion)**2 for m in motion_per_frame) / len(motion_per_frame)
        
        # Flicker detection (rapid brightness changes)
        flicker_count = 0
        for i in range(1, len(avg_brightness)):
            if abs(avg_brightness[i] - avg_brightness[i-1]) > 20:
                flicker_count += 1
        
        return {
            "frame_count": len(frames),
            "scene_changes": scene_changes,
            "scene_change_count": len(scene_changes),
            "avg_motion": round(avg_motion, 2),
            "motion_variance": round(motion_variance, 2),
            "flicker_count": flicker_count,
            "avg_brightness": round(sum(avg_brightness)/max(1,len(avg_brightness)), 1),
            "verdict": "smooth" if motion_variance < 100 and flicker_count < 5 else "choppy",
        }
    
    def _frame_difference(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculate per-pixel difference between two frames."""
        w, h = img1.size
        if img2.size != (w, h):
            img2 = img2.resize((w, h))
        
        p1 = img1.load()
        p2 = img2.load()
        
        total_diff = 0
        step = max(1, min(w, h) // 50)
        count = 0
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                count += 1
                diff = sum(abs(p1[x, y][c] - p2[x, y][c]) for c in range(3))
                total_diff += diff
        
        return total_diff / max(1, count) / 3


if __name__ == "__main__":
    va = VideoAnalyser()
    print("DMAI VideoAnalyser ready.")
    print("  Methods: analyse_frames(frame_paths)")
