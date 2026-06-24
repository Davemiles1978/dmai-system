"""
SelfScanner — audits DMAI's own routes, components, KPIs, DB tables, and capability gaps.
Runs on startup and every 30 minutes via SelfEvolutionOrchestrator.
"""
import os, json, sqlite3, threading, ast, logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

EXPECTED_THREADS = [
    "autonomous_researcher", "background_updater", "graph_evolution",
    "kaizen_repair", "kpi_seed", "parallel_learner", "stage_learner",
    "vocab_ingest", "self_evolution", "alex_riviera_content"
]

class SelfScanner:
    def __init__(self, app=None, data_path="data"):
        self.app = app
        self.data_path = data_path.rstrip("/")
        self.db_path = os.path.join(self.data_path, "dmai_knowledge.db")
        self.gap_report_path = os.path.join(self.data_path, "gap_report.json")
        self.target_caps_path = os.path.join(self.data_path, "target_capabilities.json")

    def run(self) -> dict:
        report = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "broken_routes": self._audit_routes(),
            "stub_components": self._audit_components(),
            "dead_threads": self._audit_threads(),
            "underperforming_kpis": self._audit_kpis(),
            "empty_tables": self._audit_db(),
            "capability_gaps": self._audit_capability_gaps()
        }
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.gap_report_path, "w") as f:
            json.dump(report, f, indent=2)
        total = sum(len(v) for v in report.values() if isinstance(v, list))
        logger.info(f"SelfScanner: {total} total gaps found")
        return report

    def _audit_routes(self) -> list:
        broken = []
        if not self.app:
            return broken
        try:
            with self.app.test_client() as client:
                for rule in self.app.url_map.iter_rules():
                    if "GET" not in rule.methods or "<" in str(rule):
                        continue
                    path = str(rule)
                    try:
                        import os as _os, base64 as _b64
                        _pw = _os.environ.get("MASTER_PASSWORD", "")
                        _tok = _b64.b64encode(f"admin:{_pw}".encode()).decode() if _pw else ""
                        _hdrs = {"Authorization": f"Basic {_tok}"} if _tok else {}
                        resp = client.get(path, headers=_hdrs)
                        if resp.status_code >= 500:
                            broken.append({"path": path, "error": str(resp.status_code)})
                        elif resp.status_code == 200:
                            try:
                                data = resp.get_json() or {}
                                if data.get("status") in ["not_implemented", "stub"]:
                                    broken.append({"path": path, "error": "stub"})
                            except Exception:
                                pass
                    except Exception as e:
                        broken.append({"path": path, "error": str(e)[:100]})
        except Exception as e:
            logger.warning(f"Route audit error: {e}")
        return broken

    @staticmethod
    def _has_real_not_implemented(source: str) -> bool:
        """True only if module body contains an actual `raise NotImplementedError(...)`
        statement — not a string literal, comment, or descriptor reference."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and node.exc is not None:
                target = node.exc
                if isinstance(target, ast.Call):
                    target = target.func
                if isinstance(target, ast.Name) and target.id == "NotImplementedError":
                    return True
                if isinstance(target, ast.Attribute) and target.attr == "NotImplementedError":
                    return True
        return False

    def _audit_components(self) -> list:
        stubs = []
        components_dir = Path("components")
        if not components_dir.exists():
            return stubs
        for py_file in components_dir.glob("*.py"):
            try:
                source = py_file.read_text(errors="ignore")
                if self._has_real_not_implemented(source):
                    stubs.append(str(py_file))
                    continue
                ast.parse(source)
            except SyntaxError as e:
                stubs.append(f"{py_file} (SyntaxError: {e})")
            except Exception as e:
                stubs.append(f"{py_file} ({e})")
        return stubs

    # Match each expected thread to ANY running thread by substring keyword.
    # Threads in DMAI are named with dmai- prefix, suffix variants, or
    # different casing (KpiSeedLoop vs kpi_seed). Strict equality undercounts.
    EXPECTED_THREAD_KEYWORDS = {
        "autonomous_researcher": ["research", "autonomous-researcher", "autonomous_researcher"],
        "background_updater":    ["updater", "background_updater", "background-updater"],
        "graph_evolution":       ["graph", "evolution"],
        "kaizen_repair":         ["kaizen", "repair"],
        "kpi_seed":              ["kpi"],
        "parallel_learner":      ["parallel", "web-learn", "web_learn", "web_learner", "web-learner", "parallel_learner"],
        "stage_learner":         ["stage", "stagelearner"],
        "vocab_ingest":          ["vocab", "ingest"],
        "self_evolution":        ["self_evo", "self-evo", "self_evolution"],
        "alex_riviera_content":  ["alex_riviera", "alex-riviera"],
    }

    def _audit_threads(self) -> list:
        running = [t.name.lower() for t in threading.enumerate()]
        dead = []
        for name in EXPECTED_THREADS:
            kws = self.EXPECTED_THREAD_KEYWORDS.get(name, [name])
            if not any(any(kw.lower() in n for kw in kws) for n in running):
                dead.append(name)
        return dead

    def _audit_kpis(self) -> list:
        underperforming = []
        try:
            state_path = os.path.join(self.data_path, "si_core_state.json")
            if not os.path.exists(state_path):
                return underperforming
            with open(state_path) as f:
                state = json.load(f)
            kpi_keys = [
                "skill_acquisition_rate", "transfer_learning_rate",
                "zero_shot_success_count", "agentic_capability_score",
                "recursive_self_improvement_rate", "sample_efficiency_trend",
                "metacognition_accuracy", "multi_modal_integration_score"
            ]
            for k in kpi_keys:
                val = state.get(k, 0)
                if isinstance(val, (int, float)) and val < 0.1:
                    underperforming.append({"kpi": k, "value": val, "target": 0.5})
        except Exception as e:
            logger.warning(f"KPI audit error: {e}")
        return underperforming

    def _audit_db(self) -> list:
        empty = []
        should_have_data = ["capabilities", "insights", "suggestions", "syllabus_content", "sources"]
        try:
            if not os.path.exists(self.db_path):
                return ["database_missing"]
            conn = sqlite3.connect(self.db_path)
            for table in should_have_data:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    if count == 0:
                        empty.append(table)
                except Exception:
                    empty.append(f"{table}(missing)")
            conn.close()
        except Exception as e:
            logger.warning(f"DB audit error: {e}")
        return empty

    def _audit_capability_gaps(self) -> list:
        gaps = []
        if not os.path.exists(self.target_caps_path):
            return [{"name": "capability_mapper_not_run", "priority": 0}]
        try:
            with open(self.target_caps_path) as f:
                caps = json.load(f)
            for name, info in caps.items():
                if name.startswith("_") or not isinstance(info, dict):
                    continue
                if not info.get("implemented", False):
                    gaps.append({"name": name, "description": info.get("description", ""), "priority": info.get("priority", 99), "component": info.get("component", "")})
            gaps.sort(key=lambda x: x.get("priority", 99))
        except Exception as e:
            logger.warning(f"Capability gap audit error: {e}")
        return gaps
