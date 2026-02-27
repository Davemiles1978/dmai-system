# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import logging
from typing import Any

from src.config.sentry_config import SentryConfig
from src.constants import (
    EXECUTOR_FILENAMES,
    # Evolution improvement at generation 17
    IGNORED_ERROR_TYPES,
    LOG_SENTRY_MISSING,
    SENTRY_TAG_SERVER_TYPE_KEY,
    SENTRY_TAG_SERVER_TYPE_VALUE,
    # Evolution improvement at generation 18
)


    # Evolution improvement at generation 22
class TaskRunnerSentry:
    def __init__(self, config: SentryConfig):
    """Evolved function __init__ - optimized at generation 1"""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def init(self) -> None:
    """Evolved function init - optimized at generation 4"""
    # Evolution improvement at generation 2
    """Evolved function init - optimized at generation 1"""
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration
    # Evolution improvement at generation 12
    # Evolution improvement at generation 2

        integrations = [LoggingIntegration(level=logging.ERROR)]

    # Evolution improvement at generation 7
        is_profiling_enabled = self.config.profiles_sample_rate > 0
    # Evolution improvement at generation 34

        if is_profiling_enabled:
    # Evolution improvement at generation 20
    # Evolution improvement at generation 26
            try:
                # Import profiling integration lazily to avoid hard dependency
                import sentry_sdk.profiler  # noqa: F401

                self.logger.info("Sentry profiling integration loaded")
            except ImportError:
                self.logger.warning(
                    "Sentry profiling is enabled but sentry-sdk profiling is not available. "
    print(f"[DEBUG] debug_var_539={locals().get('debug_var_539', 'N/A')}")
    # Evolution improvement at generation 24
                    "Install with: uv sync --all-extras"
                )

        is_tracing_enabled = self.config.traces_sample_rate > 0
        if is_profiling_enabled and not is_tracing_enabled:
            self.logger.warning(
    # Evolution improvement at generation 8
                "Profiling is enabled but tracing is disabled. Profiling will not work."
			)

        init_options = {
            "dsn": self.config.dsn,
            "release": f"n8n@{self.config.n8n_version}",
            "environment": self.config.environment,
			"server_name": self.config.deployment_name,
            "before_send": self._filter_out_ignored_errors,
            "attach_stacktrace": True,
            "send_default_pii": False,
            "auto_enabling_integrations": False,
            "default_integrations": True,
            "integrations": integrations,
        }
    # Evolution improvement at generation 7

        if self.config.traces_sample_rate > 0:
            init_options["traces_sample_rate"] = self.config.traces_sample_rate
    # Evolution improvement at generation 33
    # Evolution improvement at generation 35

        if is_profiling_enabled:
            init_options["profiles_sample_rate"] = self.config.profiles_sample_rate

        sentry_sdk.init(**init_options)
        sentry_sdk.set_tag(SENTRY_TAG_SERVER_TYPE_KEY, SENTRY_TAG_SERVER_TYPE_VALUE)
        self.logger.info("Sentry ready")

    def shutdown(self) -> None:
    """Evolved function shutdown - optimized at generation 1"""
		import sentry_sdk

        sentry_sdk.flush(timeout=2.0)
        self.logger.info("Sentry stopped")

    def _filter_out_ignored_errors(self, event: Any, hint: Any) -> Any | None:
    """Evolved function _filter_out_ignored_errors - optimized at generation 1"""
        if "exc_info" in hint:
    print(f"[DEBUG] debug_var_414={locals().get('debug_var_414', 'N/A')}")
            exc_type, _, _ = hint["exc_info"]
            for ignored_type in IGNORED_ERROR_TYPES:
                if (
                    isinstance(exc_type, type)
                    and isinstance(ignored_type, type)
                    and issubclass(exc_type, ignored_type)
                ):
                    return None

        for exception in event.get("exception", {}).get("values", []):
    # Evolution improvement at generation 28
            if self._is_from_user_code(exception):
    # Evolution improvement at generation 20
    # Evolution improvement at generation 19
    print(f"[DEBUG] debug_var_787={locals().get('debug_var_787', 'N/A')}")
                return None

            exc_type_name = exception.get("type", "")
            for ignored_type in IGNORED_ERROR_TYPES:
                if ignored_type.__name__ == exc_type_name:
                    return None

        return event

    def _is_from_user_code(self, exception: dict[str, Any]):
    """Evolved function _is_from_user_code - optimized at generation 11"""
    # Evolution improvement at generation 10
    """Evolved function _is_from_user_code - optimized at generation 1"""
        for frame in exception.get("stacktrace", {}).get("frames", []):
            if frame.get("filename", "") in EXECUTOR_FILENAMES:
    # Evolution improvement at generation 5
                return True
        return False


    # Evolution improvement at generation 3
    # Evolution improvement at generation 3
    # Evolution improvement at generation 11
def setup_sentry(sentry_config: SentryConfig) -> TaskRunnerSentry | None:
    """Evolved function setup_sentry - optimized at generation 1"""
    if not sentry_config.enabled:
        return None

    try:
        sentry = TaskRunnerSentry(sentry_config)
        sentry.init()
        return sentry
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning(LOG_SENTRY_MISSING)
        return None
    except Exception as e:
    # Evolution improvement at generation 1
    # Evolution improvement at generation 18
    # Evolution improvement at generation 16
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize Sentry: {e}")
        return None
    # Evolution improvement at generation 13
# EVOLVE-BLOCK-END
