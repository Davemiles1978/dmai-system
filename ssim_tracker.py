"""
DMAI Avatar Identity Tracker
==============================
Tracks avatar identity drift across renders using structural similarity.
Uses SSIM (scikit-image) when available, perceptual hash (PIL) as fallback,
or MD5 hash as last resort. All history written atomically.
"""

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically using temp file + os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', dir=path.parent, suffix='.tmp',
        delete=False, encoding='utf-8'
    ) as tmp:
        json.dump(data, tmp, indent=2, default=str)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


class AvatarIdentityTracker:
    """
    Tracks identity drift across avatar renders.
    Compares each render against a registered canonical reference.
    """

    SSIM_THRESHOLD  = 0.85   # renders with SSIM below this are flagged
    HASH_THRESHOLD  = 15     # Hamming distance above this is flagged (0-64 scale)

    def __init__(
        self,
        data_path: Path,
        canonical_profile_path: Optional[Path] = None
    ):
        """Initialise tracker, detect available comparison methods."""
        self.data_path       = Path(data_path)
        self.identity_dir    = self.data_path / "avatar_identity"
        self.identity_dir.mkdir(parents=True, exist_ok=True)
        self.history_file    = self.identity_dir / "identity_history.json"
        self.canonical_path  = canonical_profile_path
        self._canonical_hash: Optional[str] = None
        self._ssim_available = self._check_ssim()
        self._pil_available  = self._check_pil()
        self._history: list  = self._load_history()

        method = "SSIM" if self._ssim_available else ("pHash" if self._pil_available else "MD5")
        logger.info("AvatarIdentityTracker initialised. Method: %s", method)

    def _check_ssim(self) -> bool:
        """Check if scikit-image structural_similarity is available."""
        try:
            from skimage.metrics import structural_similarity  # noqa: F401
            return True
        except ImportError:
            return False

    def _check_pil(self) -> bool:
        """Check if Pillow is available for perceptual hashing."""
        try:
            from PIL import Image  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_history(self) -> list:
        """Load existing identity history from disk."""
        if self.history_file.exists():
            try:
                return json.loads(self.history_file.read_text())
            except Exception:
                return []
        return []

    def _ssim_compare(self, img_path_a: Path, img_path_b: Path) -> float:
        """Compare two images using structural similarity. Returns 0.0-1.0."""
        try:
            import numpy as np
            from skimage.metrics import structural_similarity
            from skimage import io, color, transform

            img_a = io.imread(str(img_path_a))
            img_b = io.imread(str(img_path_b))

            # Convert to greyscale
            if img_a.ndim == 3:
                img_a = color.rgb2gray(img_a)
            if img_b.ndim == 3:
                img_b = color.rgb2gray(img_b)

            # Resize to match
            if img_a.shape != img_b.shape:
                img_b = transform.resize(img_b, img_a.shape, anti_aliasing=True)

            score, _ = structural_similarity(img_a, img_b, full=True, data_range=1.0)
            return float(score)
        except Exception as e:
            logger.warning("SSIM comparison failed: %s", e)
            return 0.0

    def _perceptual_hash(self, image_path: Path) -> str:
        """
        Compute 8x8 average perceptual hash using PIL.
        Returns 16-char hex string.
        """
        try:
            from PIL import Image
            img  = Image.open(str(image_path)).convert("L").resize((8, 8), Image.LANCZOS)
            pixels = list(img.getdata())
            avg    = sum(pixels) / len(pixels)
            bits   = "".join("1" if p > avg else "0" for p in pixels)
            return format(int(bits, 2), "016x")
        except Exception as e:
            logger.warning("pHash failed: %s. Using MD5 fallback.", e)
            return hashlib.md5(Path(image_path).read_bytes()).hexdigest()[:16]

    def _hamming_distance(self, h1: str, h2: str) -> int:
        """Hamming distance between two hex hash strings."""
        try:
            return bin(int(h1, 16) ^ int(h2, 16)).count("1")
        except Exception:
            return 64   # worst case

    def register_canonical(self, image_path: Path) -> None:
        """Register a canonical reference image for future comparisons."""
        self.canonical_path  = Path(image_path)
        if self._pil_available or not self._ssim_available:
            self._canonical_hash = self._perceptual_hash(image_path)
        logger.info(
            "Canonical reference registered: %s (hash=%s)",
            image_path.name,
            self._canonical_hash[:8] if self._canonical_hash else "N/A"
        )

    def compare_render(self, render_path: Path) -> dict:
        """
        Compare render against canonical reference.

        Returns:
            {
              "render": str,
              "score": float,
              "method": str,
              "drift_detected": bool,
              "threshold": float,
              "timestamp": str
            }
        """
        render_path = Path(render_path)
        ts = datetime.now(timezone.utc).isoformat()

        if not self.canonical_path or not Path(self.canonical_path).exists():
            return {
                "render": render_path.name,
                "score": None,
                "method": "none",
                "drift_detected": False,
                "threshold": self.SSIM_THRESHOLD,
                "timestamp": ts,
                "warning": "No canonical reference registered",
            }

        if self._ssim_available:
            score  = self._ssim_compare(self.canonical_path, render_path)
            method = "ssim"
            drift  = score < self.SSIM_THRESHOLD
            threshold = self.SSIM_THRESHOLD
        else:
            canonical_hash = self._canonical_hash or self._perceptual_hash(self.canonical_path)
            render_hash    = self._perceptual_hash(render_path)
            distance       = self._hamming_distance(canonical_hash, render_hash)
            score          = 1.0 - (distance / 64.0)   # normalise to 0-1
            method         = "phash"
            drift          = distance > self.HASH_THRESHOLD
            threshold      = 1.0 - (self.HASH_THRESHOLD / 64.0)

        if drift:
            logger.warning(
                "AVATAR DRIFT DETECTED: %s score=%.3f (threshold=%.3f) method=%s",
                render_path.name, score, threshold, method
            )

        return {
            "render":        render_path.name,
            "score":         round(score, 4),
            "method":        method,
            "drift_detected": drift,
            "threshold":     threshold,
            "timestamp":     ts,
        }

    def track_render(self, render_path: Path) -> dict:
        """Compare render, record in history, persist atomically. Returns result."""
        result = self.compare_render(render_path)
        self._history.append(result)
        _atomic_write_json(self.history_file, self._history)
        return result

    def get_drift_report(self) -> dict:
        """Return summary of identity drift across all tracked renders."""
        if not self._history:
            return {"total_renders": 0, "drifted": 0, "stable": 0, "drift_rate": 0.0}

        drifted = [r for r in self._history if r.get("drift_detected")]
        scores  = [r["score"] for r in self._history if r.get("score") is not None]

        return {
            "total_renders": len(self._history),
            "drifted":       len(drifted),
            "stable":        len(self._history) - len(drifted),
            "drift_rate":    round(len(drifted) / len(self._history), 3),
            "avg_score":     round(sum(scores) / len(scores), 4) if scores else None,
            "min_score":     round(min(scores), 4) if scores else None,
            "method":        self._history[-1].get("method", "unknown") if self._history else "unknown",
            "last_checked":  self._history[-1]["timestamp"] if self._history else None,
        }
