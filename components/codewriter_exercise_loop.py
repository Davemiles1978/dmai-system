"""
CodeWriter Exercise Loop
=========================
Background daemon that gives DMAI regular coding practice by generating
small utility components every 30 minutes.  This keeps the Code Writer
Activity counter active and builds DMAI's self-generation muscle.
"""
import logging
import random
import threading
import time

logger = logging.getLogger("dmai.codewriter_exercise")

_EXERCISES = [
    {"name": "utility_cleanup", "description": "A small Python utility that scans for and reports unused imports in the codebase", "requirements": ["ast", "pathlib"]},
    {"name": "health_checker", "description": "A lightweight health check endpoint that reports system status", "requirements": ["json", "datetime"]},
    {"name": "log_rotator", "description": "A log rotation utility that archives logs older than 7 days", "requirements": ["pathlib", "shutil", "datetime"]},
    {"name": "config_validator", "description": "Validates that all required environment variables are set", "requirements": ["os", "json"]},
    {"name": "string_utils", "description": "A collection of string manipulation utilities (slugify, truncate, sanitise)", "requirements": ["re", "unicodedata"]},
    {"name": "file_watcher", "description": "Watches a directory for new files and logs them", "requirements": ["pathlib", "time"]},
    {"name": "rate_limiter", "description": "A simple token-bucket rate limiter for API calls", "requirements": ["time", "threading"]},
    {"name": "cache_decorator", "description": "An in-memory LRU cache decorator for function results", "requirements": ["functools", "time"]},
]

_INTERVAL_SECONDS = 1800  # 30 minutes


def start_exercise_loop(components: dict):
    """Start the CodeWriter exercise daemon thread.

    Args:
        components: the global components dict from dmai_core_complete
    """

    def _exercise():
        time.sleep(120)  # Wait 2 min for system to stabilise after boot
        while True:
            try:
                cw = components.get("code_writer")
                if cw and hasattr(cw, "generate_component"):
                    ex = random.choice(_EXERCISES)
                    result = cw.generate_component(
                        component_name=ex["name"],
                        description=ex["description"],
                        requirements=ex["requirements"],
                        origin="codewriter_exercise_loop",
                    )
                    if result.get("ok"):
                        logger.info("CodeWriter exercise: generated %s", ex["name"])
                    else:
                        logger.debug("CodeWriter exercise skipped: %s", result.get("error", "unknown"))
            except Exception as e:
                logger.debug("CodeWriter exercise loop error: %s", e)
            time.sleep(_INTERVAL_SECONDS)

    t = threading.Thread(target=_exercise, daemon=True, name="CodeWriterExercise")
    t.start()
    logger.info("CodeWriter exercise loop started (every %d min)", _INTERVAL_SECONDS // 60)
