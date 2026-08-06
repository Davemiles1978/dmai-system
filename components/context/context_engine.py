"""
DMAI Context API — Prototype v0.1
A behavioral prediction engine for ambient intelligence.
Designed to work with ZERO budget on existing Render deployment.
When 6G arrives, the sensor layer upgrades — the AI logic stays the same.
"""

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

class ContextEngine:
    """
    Observes patterns → learns habits → predicts actions.
    Pure Python, no external dependencies beyond stdlib.
    """
    
    def __init__(self, data_dir: str = "data/context"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.habits_file = self.data_dir / "habits.json"
        self.habits = self._load()
    
    def _load(self) -> dict:
        if self.habits_file.exists():
            return json.loads(self.habits_file.read_text())
        return {"users": {}, "global_patterns": {}}
    
    def _save(self):
        self.habits_file.write_text(json.dumps(self.habits, indent=2))
    
    # ── Observation ──────────────────────────────────────────────────────
    def observe(self, user_id: str, event: dict) -> dict:
        """Record an event: {time, location, action, context}"""
        if user_id not in self.habits["users"]:
            self.habits["users"][user_id] = {"events": [], "patterns": {}}
        
        event["timestamp"] = event.get("timestamp", datetime.now(timezone.utc).isoformat())
        event["hour"] = datetime.fromisoformat(event["timestamp"]).hour
        event["day_of_week"] = datetime.fromisoformat(event["timestamp"]).weekday()
        
        self.habits["users"][user_id]["events"].append(event)
        
        # Keep last 1000 events per user
        if len(self.habits["users"][user_id]["events"]) > 1000:
            self.habits["users"][user_id]["events"] =                 self.habits["users"][user_id]["events"][-1000:]
        
        self._save()
        return {"status": "observed", "event_count": len(self.habits["users"][user_id]["events"])}
    
    # ── Pattern Learning ─────────────────────────────────────────────────
    def learn_patterns(self, user_id: str) -> dict:
        """Analyze event history and extract behavioral patterns."""
        events = self.habits.get("users", {}).get(user_id, {}).get("events", [])
        if len(events) < 10:
            return {"status": "insufficient_data", "events_needed": 10, "current": len(events)}
        
        patterns = defaultdict(lambda: defaultdict(int))
        
        for e in events:
            key = f"{e.get('day_of_week', 0)}:{e.get('hour', 0)}"
            action = e.get("action", "unknown")
            patterns[key][action] += 1
        
        # Normalize to probabilities
        normalized = {}
        for key, actions in patterns.items():
            total = sum(actions.values())
            normalized[key] = {a: round(c/total, 3) for a, c in actions.items()}
        
        self.habits["users"][user_id]["patterns"] = normalized
        self._save()
        
        return {
            "status": "learned",
            "patterns_found": len(normalized),
            "sample_patterns": dict(list(normalized.items())[:5])
        }
    
    # ── Prediction ───────────────────────────────────────────────────────
    def predict(self, user_id: str, current_context: dict = None) -> dict:
        """Predict the most likely next action given current context."""
        patterns = self.habits.get("users", {}).get(user_id, {}).get("patterns", {})
        if not patterns:
            return {"status": "no_patterns", "message": "Learn patterns first via /context/learn"}
        
        now = datetime.now(timezone.utc)
        key = f"{now.weekday()}:{now.hour}"
        
        predictions = patterns.get(key, {})
        if not predictions:
            # Fall back to global patterns for this hour
            global_key = f"*:{now.hour}"
            predictions = self.habits.get("global_patterns", {}).get(global_key, {})
        
        if not predictions:
            return {"status": "no_prediction", "message": "Insufficient data for this time window"}
        
        # Sort by probability
        ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "status": "predicted",
            "timestamp": now.isoformat(),
            "context_key": key,
            "top_prediction": ranked[0][0] if ranked else None,
            "confidence": ranked[0][1] if ranked else 0,
            "all_predictions": [{"action": a, "probability": p} for a, p in ranked[:5]]
        }
    
    # ── Automation ───────────────────────────────────────────────────────
    def act(self, user_id: str, prediction: dict) -> dict:
        """Execute an action based on prediction (if confidence is high enough)."""
        if prediction.get("status") != "predicted":
            return {"status": "no_action", "reason": "No valid prediction"}
        
        confidence = prediction.get("confidence", 0)
        action = prediction.get("top_prediction", "")
        
        if confidence < 0.7:
            return {"status": "low_confidence", "confidence": confidence, "action": action}
        
        # In a real deployment, this would trigger IFTTT, Home Assistant, etc.
        # For now, we log the action and return what we WOULD do.
        action_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "action": action,
            "confidence": confidence,
            "executed": False,  # Set to True when connected to real actuators
            "message": f"WOULD execute: {action} (confidence: {confidence:.1%})"
        }
        
        # Log to action history
        log_file = self.data_dir / "action_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(action_log) + "\n")
        
        return action_log
    
    # ── Status ───────────────────────────────────────────────────────────
    def status(self) -> dict:
        users = len(self.habits.get("users", {}))
        total_events = sum(
            len(u.get("events", [])) 
            for u in self.habits.get("users", {}).values()
        )
        return {
            "engine": "DMAI Context Engine v0.1",
            "users_tracked": users,
            "total_events": total_events,
            "data_dir": str(self.data_dir),
            "ready_for_6g": True,
            "message": "When 6G arrives, swap sensor layer — AI logic remains identical."
        }


# ── Flask Blueprint for Context API ──────────────────────────────────────
def register_context_routes(app):
    """Register Context API routes on a Flask app."""
    engine = ContextEngine()
    
    @app.route("/context/status", methods=["GET"])
    def context_status():
        return engine.status()
    
    @app.route("/context/observe", methods=["POST"])
    def context_observe():
        data = app.request.get_json(silent=True) or {}
        user_id = data.get("user_id", "anonymous")
        event = data.get("event", {})
        return engine.observe(user_id, event)
    
    @app.route("/context/learn", methods=["POST"])
    def context_learn():
        data = app.request.get_json(silent=True) or {}
        user_id = data.get("user_id", "anonymous")
        return engine.learn_patterns(user_id)
    
    @app.route("/context/predict", methods=["POST"])
    def context_predict():
        data = app.request.get_json(silent=True) or {}
        user_id = data.get("user_id", "anonymous")
        context = data.get("context", None)
        return engine.predict(user_id, context)
    
    @app.route("/context/act", methods=["POST"])
    def context_act():
        data = app.request.get_json(silent=True) or {}
        user_id = data.get("user_id", "anonymous")
        prediction = engine.predict(user_id)
        return engine.act(user_id, prediction)
    
    return engine


# ── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    engine = ContextEngine()
    
    print("=" * 60)
    print("DMAI Context API — Self Test")
    print("=" * 60)
    
    # Simulate a user's daily routine for 7 days
    print("\nSimulating 7 days of user behavior...")
    user = "test_user_1"
    routine = {
        7:  ("wake_up", "bedroom"),
        8:  ("make_coffee", "kitchen"),
        9:  ("start_work", "home_office"),
        12: ("lunch", "kitchen"),
        13: ("back_to_work", "home_office"),
        17: ("exercise", "living_room"),
        18: ("dinner", "kitchen"),
        20: ("watch_tv", "living_room"),
        22: ("go_to_bed", "bedroom"),
    }
    
    for day in range(7):
        for hour, (action, location) in routine.items():
            engine.observe(user, {
                "action": action,
                "location": location,
                "day_of_week": day,
                "hour": hour,
            })
    
    print(f"Recorded {7 * len(routine)} events")
    
    # Learn patterns
    result = engine.learn_patterns(user)
    print(f"\nLearned patterns: {result['patterns_found']}")
    
    # Predict current action
    prediction = engine.predict(user)
    print(f"\nPrediction: {json.dumps(prediction, indent=2)}")
    
    # Try to act
    action = engine.act(user, prediction)
    print(f"\nAction: {json.dumps(action, indent=2)}")
    
    print(f"\n{engine.status()['message']}")
    print("=" * 60)
