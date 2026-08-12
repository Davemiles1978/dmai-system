"""DMAI System Monitor — tracks memory, CPU, disk, PG connections, threads, FDs.
Exposes state via /api/admin/system-monitor for live admin dashboard.
"""
import os, json, time, threading, logging, gc, sys
from datetime import datetime, timezone

logger = logging.getLogger("dmai.system_monitor")

class SystemMonitor:
    """Monitors system resources and exposes state as JSON."""
    def __init__(self, interval: int = 30):
        self.interval = interval
        self.peak_rss = 0
        self.peak_cpu = 0
        self.running = False
        self.thread = None
        self.state_file = "data/system_monitor_state.json"
        self.state = {}

    def _get_memory_mb(self) -> float:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 0

    def _get_cpu_pct(self) -> float:
        try:
            import psutil
            return psutil.Process().cpu_percent(interval=None)
        except Exception:
            return 0

    def _get_disk_usage(self) -> dict:
        try:
            import shutil
            total, used, free = shutil.disk_usage("/opt/render/project/src")
            return {"total_mb": round(total/1048576), "used_mb": round(used/1048576),
                    "free_mb": round(free/1048576), "pct_used": round(used/total*100, 1)}
        except Exception:
            return {}

    def _get_pg_connections(self) -> int:
        try:
            import psycopg2
            pg = psycopg2.connect(os.environ.get("DATABASE_URL", ""), connect_timeout=3)
            c = pg.cursor()
            c.execute("SELECT COUNT(*) FROM pg_stat_activity")
            count = c.fetchone()[0]
            c.close(); pg.close()
            return count
        except Exception:
            return -1

    def _get_fd_count(self) -> int:
        try:
            return len(os.listdir("/proc/self/fd"))
        except Exception:
            return -1

    def _get_thread_count(self) -> int:
        return threading.active_count()

    def _get_component_memory(self) -> dict:
        try:
            gc.collect()
            counts = {}
            for obj in gc.get_objects():
                module = type(obj).__module__.split('.')[0]
                try:
                    counts[module] = counts.get(module, 0) + sys.getsizeof(obj)
                except Exception:
                    pass
            sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
            return {k: round(v/1048576, 2) for k, v in sorted_items}
        except Exception:
            return {}

    def snapshot(self) -> dict:
        rss = self._get_memory_mb()
        cpu = self._get_cpu_pct()
        if rss > self.peak_rss:
            self.peak_rss = rss
        if cpu > self.peak_cpu:
            self.peak_cpu = cpu

        self.state = {
            "rss_mb": round(rss, 1),
            "cpu_pct": round(cpu, 1),
            "peak_rss_mb": round(self.peak_rss, 1),
            "peak_cpu_pct": round(self.peak_cpu, 1),
            "disk": self._get_disk_usage(),
            "pg_connections": self._get_pg_connections(),
            "fd_count": self._get_fd_count(),
            "thread_count": self._get_thread_count(),
            "top_components": self._get_component_memory(),
            "uptime_seconds": round(time.time() - getattr(self, "_start_time", time.time())),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Persist state file
        try:
            os.makedirs("data", exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass
        return self.state

    def start(self):
        if self.running:
            return
        self.running = True
        self._start_time = time.time()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="system_monitor")
        self.thread.start()
        logger.info(f"SystemMonitor started (interval={self.interval}s)")

    def _loop(self):
        while self.running:
            try:
                self.snapshot()
            except Exception as e:
                logger.debug(f"SystemMonitor error: {e}")
            time.sleep(self.interval)


def start_system_monitor(interval: int = 30):
    monitor = SystemMonitor(interval=interval)
    monitor.start()
    return monitor
