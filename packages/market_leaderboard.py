import datetime
from typing import Dict

class AIMarketLeaderboard:
    def __init__(self):
        self.competitors: Dict[str, Dict[str, float]] = {}
        self.self_scores: Dict[str, float] = {}

    def update_competitor(self, name: str, capabilities: Dict[str, float]):
        self.competitors[name] = capabilities

    def update_self(self, capability: str, score: float):
        self.self_scores[capability] = score

    def get_gaps(self) -> Dict:
        gaps = {}
        for cap, my_score in self.self_scores.items():
            best_comp = None
            best_score = -1
            for comp, caps in self.competitors.items():
                if cap in caps and caps[cap] > best_score:
                    best_score = caps[cap]
                    best_comp = comp
            gaps[cap] = {
                "dma_score": my_score,
                "competitor": best_comp,
                "competitor_score": best_score,
                "gap": best_score - my_score if best_score > my_score else 0,
                "leader": "DMAI" if my_score >= best_score else best_comp
            }
        return gaps

    def generate_leadership_report(self) -> Dict:
        gaps = self.get_gaps()
        overall_lead = all(v["leader"] == "DMAI" for v in gaps.values())
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_leader": "DMAI" if overall_lead else "Competitors ahead",
            "detailed_gaps": gaps,
            "action_required": [cap for cap, v in gaps.items() if v["leader"] != "DMAI"]
        }
