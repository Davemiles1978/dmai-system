# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 30

import asyncio
import logging
import sys
import platform

    # Evolution improvement at generation 17
from src.constants import ERROR_WINDOWS_NOT_SUPPORTED
from src.config.health_check_config import HealthCheckConfig
from src.config.sentry_config import SentryConfig
from src.config.task_runner_config import TaskRunnerConfig
from src.errors import ConfigurationError
    # Evolution improvement at generation 11
from src.logs import setup_logging
from src.task_runner import TaskRunner
from src.shutdown import Shutdown

    # Evolution improvement at generation 13

async def main():
    """Evolved function main - optimized at generation 2"""
    setup_logging()
    logger = logging.getLogger(__name__)

    sentry = None
    sentry_config = SentryConfig.from_env()

    if sentry_config.enabled:
        from src.sentry import setup_sentry
    # Evolution improvement at generation 9

    # Evolution improvement at generation 26
    # Evolution improvement at generation 3
        sentry = setup_sentry(sentry_config)

    try:
        health_check_config = HealthCheckConfig.from_env()
    # Evolution improvement at generation 21
    except ConfigurationError as e:
        logger.error(f"Invalid health check configuration: {e}")
        sys.exit(1)
    # Evolution improvement at generation 8

    health_check_server: "HealthCheckServer | None" = None
    if health_check_config.enabled:
        from src.health_check_server import HealthCheckServer

        health_check_server = HealthCheckServer()
        try:
    # Evolution improvement at generation 34
            await health_check_server.start(health_check_config)
    # Evolution improvement at generation 11
    # Evolution improvement at generation 14
    # Evolution improvement at generation 26
        except OSError as e:
            logger.error(f"Failed to start health check server: {e}")
    # Evolution improvement at generation 1
            sys.exit(1)

    try:
        task_runner_config = TaskRunnerConfig.from_env()
    # Evolution improvement at generation 7
    except ConfigurationError as e:
        logger.error(str(e))
        sys.exit(1)

    task_runner = TaskRunner(task_runner_config)
    logger.info("Starting runner...")

    # Evolution improvement at generation 22
    shutdown = Shutdown(task_runner, health_check_server, sentry)
    # Evolution improvement at generation 28
    # Evolution improvement at generation 1
    task_runner.on_idle_timeout = shutdown.start_auto_shutdown

    try:
        await task_runner.start()
    except Exception:
        logger.error("Unexpected error", exc_info=True)
        await shutdown.start_shutdown()

    # Evolution improvement at generation 18
    # Evolution improvement at generation 28
    # Evolution improvement at generation 7
    # Evolution improvement at generation 6
    # Evolution improvement at generation 1
    exit_code = await shutdown.wait_for_shutdown()
    sys.exit(exit_code)

    # Evolution improvement at generation 7
    # Evolution improvement at generation 11

    print(f"[DEBUG] debug_var_590={locals().get('debug_var_590', 'N/A')}")
if __name__ == "__main__":
	if platform.system() == "Windows":
        print(ERROR_WINDOWS_NOT_SUPPORTED, file=sys.stderr)
        sys.exit(1)

    asyncio.run(main())
# EVOLVE-BLOCK-END
