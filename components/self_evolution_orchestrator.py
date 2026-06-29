"""
SelfEvolutionOrchestrator — master 24/7 self-generation loop.
Scan → prioritise (broken_routes, capability_gaps, empty_tables,
underperforming_kpis) → generate → commit → verify → sleep 30 min.

The orchestrator also REPORTS the health of well-known background
components (greyhound_runner, kaizen_auto_repair) in its status output,
but DOES NOT auto-restart them from inside its own thread — that path
is prone to thread/lock contention. Use the dedicated
/api/monetisation/tips/greyhound-runner/restart endpoint instead.
"""
import os, logging, time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Background components whose thread liveness we want to surface to operators.
_BG_COMPONENTS = (
    "greyhound_runner",
    "kaizen_auto_repair",
)


class SelfEvolutionOrchestrator:
    def __init__(self, app=None, data_path="data"):
        self.app = app
        self.data_path = data_path.rstrip("/")
        self._running = False
        self._cycle_count = 0
        self._last_cycle_ts = None
        self._items_fixed_this_week = 0
        self._last_bg_health = {}
        self.interval_seconds = int(os.environ.get("EVOLUTION_INTERVAL_SECONDS", "1800"))

        # Lazy-import components to avoid circular imports at module load
        self._scanner = None
        self._mapper = None
        self._generator = None
        self._committer = None

    def _init_components(self):
        if self._scanner is not None:
            return
        try:
            from components.self_scanner import SelfScanner
            from components.capability_mapper import CapabilityMapper
            from components.code_generator import CodeGenerator
            from components.self_committer import SelfCommitter
            self._scanner = SelfScanner(app=self.app, data_path=self.data_path)
            self._mapper = CapabilityMapper(data_path=self.data_path)
            self._generator = CodeGenerator(data_path=self.data_path)
            self._committer = SelfCommitter(data_path=self.data_path)
        except Exception as e:
            logger.error(f"SelfEvolutionOrchestrator: failed to init components: {e}")

    def run_forever(self):
        self._running = True
        logger.info("SelfEvolutionOrchestrator: starting 24/7 evolution loop")
        # Wait 2 minutes after startup for other services to initialise
        time.sleep(120)
        while self._running:
            try:
                self._run_cycle()
            except Exception as e:
                logger.error(f"SelfEvolutionOrchestrator: cycle error: {e}", exc_info=True)
            time.sleep(self.interval_seconds)

    def _run_cycle(self):
        self._init_components()
        if not self._scanner:
            logger.warning("SelfEvolutionOrchestrator: components not available, skipping cycle")
            return

        self._cycle_count += 1
        self._last_cycle_ts = datetime.now(timezone.utc).isoformat()
        logger.info(f"SelfEvolutionOrchestrator: starting cycle #{self._cycle_count}")

        # 1. Update capability map
        try:
            self._mapper.run()
        except Exception as e:
            logger.warning(f"SelfEvolutionOrchestrator: mapper error: {e}")

        # 2. Scan for gaps
        try:
            gap_report = self._scanner.run()
        except Exception as e:
            logger.warning(f"SelfEvolutionOrchestrator: scanner error: {e}")
            return

        # 3. Layer 3 self-repair hook (optional)
        # If enabled, attempt targeted micro-fixes for known gap patterns.
        # This runs BEFORE generation/commit so it can reduce noise in work_items.
        try:
            if os.environ.get("LAYER3_AUTO_REPAIR_ENABLED", "").strip().lower() in ("1", "true", "yes", "on"):
                from components.self_repair_orchestrator import SelfRepairOrchestrator
                sro = SelfRepairOrchestrator(repo_root=".")
                summary = sro.run_once(auto_approve=True)
                logger.info(
                    "Layer3 auto-repair tick: matched=%d enqueued=%d auto_approved=%d",
                    len(summary.matched_patterns or []),
                    len(summary.enqueued_edit_ids or []),
                    len(summary.auto_approved_edit_ids or []),
                )
        except Exception as e:
            logger.warning(f"Layer3 auto-repair tick failed: {e}")

        # 4. Prioritise — broken routes (1), capability gaps (2),
        #    empty tables (3), underperforming KPIs (4).
        work_items = []
        for item in gap_report.get("broken_routes", []):
            if item.get("error") != "stub":
                work_items.append({
                    "type": "broken_route",
                    "name": item["path"],
                    "description": f"Fix broken Flask route {item['path']} returning error {item.get('error')}",
                    "priority": 1
                })
        for item in gap_report.get("capability_gaps", []):
            if isinstance(item, dict) and item.get("name") != "capability_mapper_not_run":
                work_items.append({**item, "type": "capability_gap"})
        for item in gap_report.get("empty_tables", []) or []:
            if isinstance(item, dict):
                tname = item.get("name") or item.get("table") or "unknown_table"
                work_items.append({
                    "type": "empty_table",
                    "name": f"backfill_{tname}",
                    "description": (
                        f"Empty table {tname}: generate a backfill or seed-data "
                        f"job so downstream KPIs unblock."
                    ),
                    "component": item.get("component"),
                    "priority": 3,
                })
        for item in gap_report.get("underperforming_kpis", []) or []:
            if isinstance(item, dict) and item.get("component"):
                kpi = item.get("name") or item.get("kpi") or "kpi"
                work_items.append({
                    "type": "underperforming_kpi",
                    "name": f"improve_{kpi}",
                    "description": (
                        f"KPI '{kpi}' underperforming "
                        f"(value={item.get('value')}, target={item.get('target')}). "
                        f"Investigate component {item.get('component')} and propose fix."
                    ),
                    "component": item.get("component"),
                    "priority": 4,
                })

        # Report background-thread health (no auto-restart).
        try:
            self._last_bg_health = self._bg_health_snapshot()
        except Exception as e:
            logger.warning(f"SelfEvolutionOrchestrator: bg health snapshot failed: {e}")

        work_items.sort(key=lambda x: x.get("priority", 99))

        # 4. Generate and commit — max 3 items per cycle to manage rate limits
        fixed = 0
        for item in work_items[:3]:
            try:
                code = self._generator.generate(item)
                if code:
                    target_file = item.get("component") or f"components/{item['name'].replace('/', '_')}.py"
                    success = self._committer.commit(item["name"], code, target_file)
                    if success:
                        fixed += 1
                        self._items_fixed_this_week += 1
                        if item.get("type") == "capability_gap":
                            self._mapper.mark_implemented(item["name"])
            except Exception as e:
                logger.warning(f"SelfEvolutionOrchestrator: failed to fix {item.get('name')}: {e}")

        total_gaps = sum(len(v) for v in gap_report.values() if isinstance(v, list))
        logger.info(
            f"SelfEvolutionOrchestrator: cycle #{self._cycle_count} complete — "
            f"{fixed} fixed, {total_gaps} total gaps remaining"
        )

    def _bg_health_snapshot(self) -> dict:
        """Inspect known background components and return liveness summary.

        Read-only: never starts/stops/restarts threads. Safe in any thread.
        """
        out = {}
        try:
            from dmai_core_complete import components as registry  # type: ignore
        except Exception:
            return {}
        if not isinstance(registry, dict):
            return {}
        for key in _BG_COMPONENTS:
            comp = registry.get(key)
            if comp is None:
                out[key] = "missing"
                continue
            if not hasattr(comp, "_thread"):
                out[key] = "unknown"
                continue
            thread = getattr(comp, "_thread")
            is_alive_fn = getattr(thread, "is_alive", None) if thread else None
            out[key] = "alive" if (is_alive_fn and is_alive_fn()) else "dead"
        return out

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "last_cycle_ts": self._last_cycle_ts,
            "items_fixed_this_week": self._items_fixed_this_week,
            "interval_seconds": self.interval_seconds,
            "bg_health": self._last_bg_health,
        }

    def stop(self):
        self._running = False
