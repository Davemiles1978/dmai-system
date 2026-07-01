"""
PersonaRegistry — role-specific operating personas for DMAI components.

Loads `components/personas/seed/personas_v1.json` at boot, persists every
persona into SQLite, exposes:

  - resolve(component=..., task=...) -> persona dict
  - system_prompt(name, with_brain=True) -> string (optionally augmented
    with relevant ExpertBrain canonical entries)
  - all() / get(name) / list_for(domain)
  - usage_log: every resolve() call is recorded for telemetry

The registry is intentionally lean: it does not call any LLM itself.
Components fetch a persona, then pass `system_prompt` to whichever
provider they already use.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

SEED_PATH = Path(__file__).resolve().parent / "seed" / "personas_v1.json"


class PersonaRegistry:
    def __init__(
        self,
        data_path: str | Path = "data",
        expert_brain: Any = None,
        seed_path: Optional[Path] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.data_path / "dmai_knowledge.db")
        self.brain = expert_brain
        self.seed_path = seed_path or SEED_PATH
        self._lock = threading.RLock()
        self._seed: Dict[str, Any] = self._load_seed()
        self._init_db()
        self._persist_seed()

    # ── DB ────────────────────────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        c = safe_open_kdb(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS personas ("
                "name TEXT PRIMARY KEY, "
                "label TEXT, "
                "scope TEXT, "
                "used_by_json TEXT, "
                "brain_domains_json TEXT, "
                "model_pref_json TEXT, "
                "system_prompt TEXT, "
                "decision_rules_json TEXT, "
                "version TEXT, "
                "updated_at TEXT DEFAULT (datetime('now')))"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS persona_usage ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ts TEXT DEFAULT (datetime('now')), "
                "persona TEXT, "
                "component TEXT, "
                "task TEXT)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_persona "
                "ON persona_usage(persona, ts DESC)"
            )
            c.commit()

    def _load_seed(self) -> Dict[str, Any]:
        if not self.seed_path.exists():
            logger.error("PersonaRegistry: seed missing at %s", self.seed_path)
            return {"version": "0.0.0", "personas": {}, "routing": {}}
        try:
            with open(self.seed_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.exception("PersonaRegistry: seed read failed: %s", e)
            return {"version": "0.0.0", "personas": {}, "routing": {}}

    def _persist_seed(self) -> None:
        version = self._seed.get("version", "1.0.0")
        personas = self._seed.get("personas", {})
        with self._conn() as c:
            for name, p in personas.items():
                c.execute(
                    "INSERT OR REPLACE INTO personas("
                    "name, label, scope, used_by_json, brain_domains_json, "
                    "model_pref_json, system_prompt, decision_rules_json, "
                    "version, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                    (
                        name,
                        p.get("label", name),
                        p.get("scope", "internal"),
                        json.dumps(p.get("used_by", [])),
                        json.dumps(p.get("brain_domains", [])),
                        json.dumps(p.get("model_preference", [])),
                        p.get("system_prompt", ""),
                        json.dumps(p.get("decision_rules", [])),
                        version,
                    ),
                )
            c.commit()
        logger.info("PersonaRegistry: persisted %d personas (v%s)",
                    len(personas), version)

    # ── Public ────────────────────────────────────────────────────────────────
    def all(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT name, label, scope, used_by_json, brain_domains_json, "
                "model_pref_json, decision_rules_json, version "
                "FROM personas ORDER BY name"
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append({
                    "name": r["name"],
                    "label": r["label"],
                    "scope": r["scope"],
                    "used_by": json.loads(r["used_by_json"] or "[]"),
                    "brain_domains": json.loads(r["brain_domains_json"] or "[]"),
                    "model_preference": json.loads(r["model_pref_json"] or "[]"),
                    "decision_rules": json.loads(r["decision_rules_json"] or "[]"),
                    "version": r["version"],
                })
            return out

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute(
                "SELECT * FROM personas WHERE name = ?", (name,)
            ).fetchone()
            if not r:
                return None
            return {
                "name": r["name"],
                "label": r["label"],
                "scope": r["scope"],
                "used_by": json.loads(r["used_by_json"] or "[]"),
                "brain_domains": json.loads(r["brain_domains_json"] or "[]"),
                "model_preference": json.loads(r["model_pref_json"] or "[]"),
                "system_prompt": r["system_prompt"] or "",
                "decision_rules": json.loads(r["decision_rules_json"] or "[]"),
                "version": r["version"],
            }

    def resolve(
        self,
        component: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resolve the right persona by component first, task second, then default.
        Logs every resolution for telemetry.
        """
        routing = self._seed.get("routing", {})
        by_component = routing.get("by_component", {})
        by_task = routing.get("by_task", {})

        chosen: Optional[str] = None
        if component and component in by_component:
            chosen = by_component[component]
        elif task and task in by_task:
            chosen = by_task[task]
        else:
            chosen = by_task.get("default", "dmai_core")

        persona = self.get(chosen) or self.get("dmai_core")
        if not persona:
            persona = {"name": chosen or "dmai_core", "system_prompt": ""}

        # Log usage
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO persona_usage(persona, component, task) "
                    "VALUES (?, ?, ?)",
                    (persona["name"], component, task),
                )
                c.commit()
        except Exception:
            pass

        return persona

    def system_prompt(
        self,
        name: str,
        with_brain: bool = True,
        brain_char_budget: int = 2500,
        extra_context: Optional[str] = None,
    ) -> str:
        """
        Return the persona's system prompt, optionally appending grounding
        snippets from the brain for the persona's declared domains.
        """
        p = self.get(name)
        if not p:
            return ""
        prompt = p.get("system_prompt", "")
        rules = p.get("decision_rules", [])
        if rules:
            prompt += "\n\nDecision rules:\n- " + "\n- ".join(rules)

        if with_brain and self.brain and p.get("brain_domains"):
            ground = self._brain_grounding(
                p["brain_domains"], char_budget=brain_char_budget
            )
            if ground:
                prompt += (
                    "\n\nGrounding (canonical knowledge — cite the sources "
                    "when stating these facts):\n" + ground
                )

        if extra_context:
            prompt += "\n\nAdditional context:\n" + extra_context.strip()

        return prompt

    def _brain_grounding(self, domains: List[str], char_budget: int) -> str:
        if not self.brain:
            return ""
        chunks: List[str] = []
        used = 0
        try:
            for dom in domains:
                entries = self.brain.by_domain(dom)
                for e in entries:
                    block = (
                        f"### {e['topic']} ({dom})\n"
                        f"{e['content']}\nSource: {e['source_url']}\n"
                    )
                    if used + len(block) > char_budget:
                        return "\n".join(chunks)
                    chunks.append(block)
                    used += len(block)
        except Exception as e:
            logger.debug("brain grounding failed: %s", e)
        return "\n".join(chunks)

    def list_for_domain(self, domain: str) -> List[Dict[str, Any]]:
        return [p for p in self.all() if domain in p.get("brain_domains", [])]

    def usage_stats(self, days: int = 7) -> Dict[str, Any]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT persona, COUNT(*) AS n FROM persona_usage "
                "WHERE ts >= datetime('now', ?) GROUP BY persona ORDER BY n DESC",
                (f"-{days} days",),
            ).fetchall()
            # ``persona`` can come back as ``bytes`` when the column has BLOB
            # affinity, which makes Flask's ``jsonify`` reject the dict key with
            # "keys must be str, int, float, bool or None, not bytes". Coerce the
            # whole payload through the shared ``_jsonable`` helper (PR #152
            # family) so the route always serialises.
            from components.json_utils import _jsonable
            return _jsonable({
                "window_days": days,
                "by_persona": {r["persona"]: r["n"] for r in rows},
                "total": sum(r["n"] for r in rows),
            })

    def reload(self) -> Dict[str, Any]:
        with self._lock:
            self._seed = self._load_seed()
            self._persist_seed()
            return {
                "version": self._seed.get("version"),
                "personas": list(self._seed.get("personas", {}).keys()),
            }


def get_persona_registry(
    data_path: str = "data", expert_brain=None
) -> PersonaRegistry:
    return PersonaRegistry(data_path=data_path, expert_brain=expert_brain)
