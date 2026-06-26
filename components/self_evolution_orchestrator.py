"""
SelfEvolutionOrchestrator — master 24/7 self-generation loop.
Scan → prioritise → generate → commit → verify → sleep 30 min.

Responsibilities:
  1. Run capability mapper + self-scanner each cycle.
  2. Watchdog: restart known background components whose worker thread has died.
  3. Prioritise gaps across types: broken_routes, capability_gaps,
     empty_tables, underperforming_kpis (with file hints). Generate + commit.
  4. Surface counts so /api/self-evolution/status reflects real activity.
"""
import os, logging, time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Components whose .start()/.start_repair_loop()/.run_forever() must be
# running 24/7. The watchdog tries to restart any that report a dead worker.
_BG_COMPONENTS = (
    # (registry_key, restart_method_name, description)
    ("greyhound_runner",      "start",             "Greyhound tipster runner"),
    ("kaizen_auto_repair",    "start_repair_loop", "Kaizen auto-repair loop"),
    ("autonomous_researcher", "start",             "Autonomous researcher"),
    ("autonomous_ingestor",   "start",             "Autonomous ingestor"),
)


class SelfEvolutionOrchestrator:
    def __init__(self, app=None, data_path="data"):
        self.app = app
        self.data_path = data_path.rstrip("/")
        self._running = False
        self._cycle_count = 0
        self._last_cycle_ts = None
        self._items_fixed_this_week = 0
        self._watchdog_restarts = 0
        self._last_watchdog_summary = {}
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

    def _watchdog(self) -> dict:
        """Restart any known background component whose worker thread is dead.

        Returns a per-component summary {key: state}: 'alive' | 'restarted' |
        'missing' | 'error:<msg>'.
        """
        summary: dict = {}
        try:
            from dmai_core_complete import components as registry  # type: ignore
        except Exception:
            registry = {}
        for key, restart_method, _desc in _BG_COMPONENTS:
            comp = registry.get(key) if isinstance(registry, dict) else None
            if comp is None:
                summary[key] = "missing"
                continue
            thread = getattr(comp, "_thread", None)
            alive = bool(thread and getattr(thread, "is_alive", lambda: False)())
            if alive:
                summary[key] = "alive"
                continue
            try:
                stop_evt = getattr(comp, "_stop", None)
                if stop_evt is not None and hasattr(stop_evt, "clear"):
                    stop_evt.clear()
                setattr(comp, "_thread", None)
                method = getattr(comp, restart_method, None)
                if method is None:
                    summary[key] = f"error: no method {restart_method}"
                    continue
                method()
                summary[key] = "restarted"
                self._watchdog_restarts += 1
                logger.info("Watchdog restarted %s (%s)", key, restart_method)
            except Exception as e:
                summary[key] = f"error: {e}"
                logger.warning("Watchdog failed to restart %s: %s", key, e)
        return summary

    def _run_cycle(self):
        self._init_components()
        if not self._scanner:
            logger.warning("SelfEvolutionOrchestrator: components not available, skipping cycle")
            return

        self._cycle_count += 1
        self._last_cycle_ts = datetime.now(timezone.utc).isoformat()
        logger.info(f"SelfEvolutionOrchestrator: starting cycle #{self._cycle_count}")

        # 0. Watchdog first: keep background loops alive before scanning.
        try:
            self._last_watchdog_summary = self._watchdog()
        except Exception as e:
            logger.warning(f"SelfEvolutionOrchestrator: watchdog error: {e}")

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

        # 3. Prioritise — broken routes first, stubs second, capability gaps third
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
                        f"Empty table {tname}: generate a backfill "
                        f"job or seed data so downstream KPIs unblock."
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

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "last_cycle_ts": self._last_cycle_ts,
            "items_fixed_this_week": self._items_fixed_this_week,
            "interval_seconds": self.interval_seconds,
            "watchdog_restarts_total": self._watchdog_restarts,
            "last_watchdog": self._last_watchdog_summary,
        }

    def stop(self):
        self._running = False
