"""
SelfScanner — audits DMAI's own routes, components, KPIs, DB tables, and capability gaps.
Runs on startup and every 30 minutes via SelfEvolutionOrchestrator.
"""
import os, json, sqlite3, threading, ast, logging, inspect, textwrap
from datetime import datetime, timezone
from pathlib import Path
from components.db import safe_open_kdb

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

    def _iter_route_rules(self) -> list:
        """Offline enumeration of the app's URL rules. No request dispatch."""
        if not self.app:
            return []
        return list(self.app.url_map.iter_rules())

    def _audit_routes(self) -> list:
        """Audit registered routes without dispatching any requests.

        Previously this issued in-process ``test_client().get()`` probes
        against every GET route. Those synthetic requests ran the real view
        handlers, and handlers such as persona_registry.resolve() acquire the
        SQLite write mutex mid-request — starving the vocabulary_ingester flush
        and producing write_mutex_timeout / "database is locked" churn.

        This offline version walks Flask's ``url_map`` and introspects each
        view function's signature and source instead. No request path is
        exercised, no write mutex is touched, no DB connection is opened.
        """
        broken = []
        if not self.app:
            return broken
        try:
            view_functions = getattr(self.app, "view_functions", {}) or {}
            for rule in self._iter_route_rules():
                methods = rule.methods or set()
                if "GET" not in methods or "<" in str(rule):
                    continue
                path = str(rule)
                view_func = view_functions.get(rule.endpoint)
                if view_func is None:
                    broken.append({"path": path, "error": "no_view_function"})
                    continue
                # Introspect the signature offline; a view we cannot introspect
                # is not itself an error, so ignore signature failures.
                try:
                    inspect.signature(view_func)
                except (TypeError, ValueError):
                    pass
                if self._view_is_stub(view_func):
                    broken.append({"path": path, "error": "stub"})
        except Exception as e:
            logger.warning(f"Route audit error: {e}")
        return broken

    @classmethod
    def _view_is_stub(cls, view_func) -> bool:
        """Detect a stub view offline via its source: a real
        ``raise NotImplementedError`` or a returned not_implemented/stub status
        payload — mirroring the old live-probe stub detection without dispatch.
        """
        try:
            source = textwrap.dedent(inspect.getsource(view_func))
        except (OSError, TypeError):
            return False
        if cls._has_real_not_implemented(source):
            return True
        lowered = source.lower()
        if '"status"' in lowered or "'status'" in lowered:
            if "not_implemented" in lowered or "stub" in lowered:
                return True
        return False

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
                # Skip AST runtime quirks (recursion depth mismatches, constructor
                # warnings under newer Python versions) - these aren't stubs.
                msg = str(e)
                if "recursion depth" in msg or "constructor" in msg.lower():
                    continue
                stubs.append(f"{py_file} ({e})")
        return stubs

    # Match each expected thread to ANY running thread by substring keyword.
    # Threads in DMAI are named with dmai- prefix, suffix variants, or
    # different casing (KpiSeedLoop vs kpi_seed). Strict equality undercounts.
    EXPECTED_THREAD_KEYWORDS = {
        "autonomous_researcher": ["research", "autonomous-researcher", "autonomous_researcher"],
        "background_updater":    ["updater", "update", "background_updater", "background-updater", "dmai-update"],
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
            conn = safe_open_kdb(self.db_path)
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

    # ------------------------------------------------------------------
    # Layer 4 — typed capability-gap audit (chunk L4-1)
    # Additive: never touched by existing run() / _audit_capability_gaps().
    # ------------------------------------------------------------------
    _L4_KPI_GAP_MAP = {
        "skill_acquisition_rate":         ("skill_acquisition_engine",       1),
        "transfer_learning_rate":         ("transfer_learning_adapter",      1),
        "zero_shot_success_count":        ("zero_shot_capability_handler",   2),
        "agentic_capability_score":       ("agentic_task_executor",          1),
        "recursive_self_improvement_rate":("recursive_improvement_loop",     1),
        "sample_efficiency_trend":        ("sample_efficiency_optimizer",    2),
    }


    def _audit_pending_capabilities(self) -> list:
        """Check the capabilities table for pending implementations."""
        import sqlite3
        gaps = []
        try:
            db_path = "data/dmai_knowledge.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT name, category, description, status
                FROM capabilities
                WHERE status = 'pending'
            ''')
            rows = cursor.fetchall()
            for row in rows:
                gaps.append({
                    "name": row[0],
                    "description": row[2],
                    "priority": 5,
                    "component": row[1]
                })
            conn.close()
        except Exception as e:
            logger.warning(f"Pending capabilities audit error: {e}")
        return gaps
    def audit_capability_gaps_typed(self) -> list:
        """Return CapabilityGapEntry items for Layer 4 self-generation.

        Sources:
          (a) KPI-driven gaps  — si_core_state.json values < 0.5 → emit a
              capability shaped to move that KPI.
          (b) Registry-driven  — target_capabilities.json entries with no
              matching components/<slug>.py on disk.

        Never raises. Returns [] on any failure.
        """
        try:
            from components.capability_gap_entry import CapabilityGapEntry
        except Exception as e:
            logger.warning(f"L4 typed audit: CapabilityGapEntry import failed: {e}")
            return []

        entries = []

        # (a) KPI-driven gaps -------------------------------------------------
        kpis = {}
        try:
            state_path = os.path.join(self.data_path, "si_core_state.json")
            if os.path.exists(state_path):
                with open(state_path) as f:
                    kpis = json.load(f) or {}
        except Exception as e:
            logger.warning(f"L4 typed audit: KPI read failed: {e}")
        for kpi_key, (cap_name, priority) in self._L4_KPI_GAP_MAP.items():
            try:
                current = float(kpis.get(kpi_key, 0.0) or 0.0)
            except (TypeError, ValueError):
                current = 0.0
            if current < 0.5:
                entries.append(CapabilityGapEntry(
                    name=cap_name,
                    description=f"Implement {cap_name} to improve {kpi_key}",
                    priority=priority,
                    evidence_source=f"kpi:{kpi_key}",
                    target_kpi=kpi_key,
                    current_value=current,
                    target_value=0.5,
                ))

        # (b) Registry-driven gaps -------------------------------------------
        try:
            if os.path.exists(self.target_caps_path):
                with open(self.target_caps_path) as f:
                    caps = json.load(f) or {}
                existing = {p.stem for p in Path("components").glob("*.py")}
                # Names already covered by KPI map shouldn't double-emit.
                kpi_names = {cap for cap, _ in self._L4_KPI_GAP_MAP.values()}
                for slug, info in caps.items():
                    if slug.startswith("_") or not isinstance(info, dict):
                        continue
                    if info.get("implemented", False):
                        continue
                    if slug in existing or slug in kpi_names:
                        continue
                    entries.append(CapabilityGapEntry(
                        name=slug,
                        description=info.get("description", ""),
                        priority=int(info.get("priority", 3)),
                        evidence_source="registry:missing",
                        target_kpi=info.get("target_kpi", ""),
                    ))
        except Exception as e:
            logger.warning(f"L4 typed audit: registry scan failed: {e}")

        return entries
