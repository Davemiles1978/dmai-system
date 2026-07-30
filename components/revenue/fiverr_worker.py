"""
FiverrWorker — DMAI's Fiverr revenue generator.

Lists AI-completable gigs on Fiverr under Alex Riviera persona,
completes orders autonomously using DMAI's capabilities.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("FiverrWorker")

FIVERR_GIGS = [
    {
        "title": "I will transcribe your audio files with 99% accuracy",
        "category": "transcription",
        "price_usd": 10,
        "delivery_hours": 24,
        "description": "Professional audio transcription using AI-powered accuracy. "
                       "Supports MP3, WAV, M4A. Up to 30 minutes per order.",
        "tags": ["transcription", "audio", "speech-to-text"],
    },
    {
        "title": "I will tag and categorize your images",
        "category": "data_labeling",
        "price_usd": 10,
        "delivery_hours": 24,
        "description": "AI-powered image tagging and categorization. "
                       "Up to 100 images per order. Object detection, scene description, content moderation.",
        "tags": ["image-tagging", "data-labeling", "computer-vision"],
    },
    {
        "title": "I will analyze your data and create insights reports",
        "category": "data_analysis",
        "price_usd": 15,
        "delivery_hours": 48,
        "description": "Comprehensive data analysis with visualizations and actionable insights. "
                       "Supports CSV, Excel, JSON. Statistical analysis, trend detection, forecasting.",
        "tags": ["data-analysis", "reports", "statistics"],
    },
    {
        "title": "I will translate your content to 20+ languages",
        "category": "translation",
        "price_usd": 10,
        "delivery_hours": 24,
        "description": "AI-powered translation with human-quality results. "
                       "Up to 1000 words per order. 20+ languages supported.",
        "tags": ["translation", "languages", "localization"],
    },
    {
        "title": "I will moderate and filter your content",
        "category": "content_moderation",
        "price_usd": 10,
        "delivery_hours": 12,
        "description": "AI content moderation for text, images, and comments. "
                       "Toxicity detection, NSFW filtering, policy compliance checks.",
        "tags": ["content-moderation", "filtering", "compliance"],
    },
]


class FiverrWorker:
    """Autonomous Fiverr gig manager and order completer."""

    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.ledger_path = self.data_path / "revenue" / "fiverr_ledger.jsonl"
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.persona = "Alex Riviera"

    def get_gigs(self) -> List[Dict[str, Any]]:
        """Return available gigs DMAI can offer."""
        return FIVERR_GIGS

    def complete_order(self, gig: Dict[str, Any], order_details: Dict[str, Any]) -> Dict[str, Any]:
        """Complete a Fiverr order using DMAI's AI."""
        order_id = order_details.get("order_id", uuid.uuid4().hex[:8])
        price = float(gig.get("price_usd", 10))

        try:
            from dmai_core_complete import _ai_chat
            prompt = f"""Complete this Fiverr order autonomously as Alex Riviera:

Gig: {gig['title']}
Category: {gig['category']}
Price: ${price}
Buyer requirements: {json.dumps(order_details.get('requirements', {}))}

Deliver professional, high-quality work. Format as if delivering to a real client.
Include a polite delivery message with the completed work."""

            response = _ai_chat(prompt)
            if response:
                self._log_earning(order_id, gig["title"], price)
                logger.info("Fiverr order completed: %s - $%.2f", gig["title"], price)
                return {"order_id": order_id, "gig": gig["title"],
                        "price_usd": price, "status": "delivered",
                        "delivery": response[:500]}
        except Exception as e:
            logger.warning("Fiverr order failed: %s", e)
        return {"status": "failed"}

    def _log_earning(self, order_id: str, gig_title: str, amount_usd: float):
        entry = {
            "id": uuid.uuid4().hex[:12],
            "source": "fiverr",
            "persona": self.persona,
            "order_id": order_id,
            "gig": gig_title,
            "amount_usd": round(amount_usd, 4),
            "fiverr_fee_20pct": round(amount_usd * 0.2, 4),
            "net_usd": round(amount_usd * 0.8, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def run_session(self) -> Dict[str, Any]:
        """Simulate completing pending Fiverr orders."""
        gigs = self.get_gigs()
        return {
            "status": "ready",
            "persona": self.persona,
            "available_gigs": len(gigs),
            "gigs": [{"title": g["title"], "price_usd": g["price_usd"],
                      "category": g["category"]} for g in gigs],
            "note": "Fiverr account setup required to receive real orders. "
                    "Gigs ready to list upon account creation.",
        }

    def get_earnings_summary(self) -> Dict[str, Any]:
        total = 0.0
        count = 0
        if self.ledger_path.exists():
            with open(self.ledger_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            total += entry.get("net_usd", 0)
                            count += 1
                        except Exception:
                            pass
        return {
            "source": "fiverr",
            "persona": self.persona,
            "total_earnings_usd": round(total, 4),
            "total_orders": count,
            "average_per_order": round(total / count, 4) if count > 0 else 0,
        }
