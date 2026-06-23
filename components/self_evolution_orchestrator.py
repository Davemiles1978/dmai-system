"""
SelfEvolutionOrchestrator — master 24/7 self-generation loop.
Scan → prioritise → generate → commit → verify → sleep 30 min.
"""
import os, logging, time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SelfEvolutionOrchestrator:
    def __init__(self, app=None, data_path="data"):
        self.app = app
        self.data_path = data_path.rstrip("/")
        self._running = False
        self._cycle_count = 0
        self._last_cycle_ts = None
        self._items_fixed_this_week = 0
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
            "interval_seconds": self.interval_seconds
        }

    def stop(self):
        self._running = False
